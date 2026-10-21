"""Teil 0 ("den Rand schliessen") of
docs/residual_error_decomposition_hourly_spec.md.

Re-runs variant A and B (unchanged 7.2 configuration) over 2026-09-01 to the
last locally-complete delivery day -- derived from the freshly rebuilt
long-history data, not the clock -- then writes a daily bias/RMSE table for
October 2026 onward next to the live daily bias/RMSE read from Repo 2's
scratch_payload_vs_realized.csv (a read-only copy under data/raw/, see
data/README.md).

Sign convention: this repo's bias() is y_true - y_pred (positive =
underestimate). Repo 2's own comparison script computes pred - y_true
(positive = overestimate) -- its bias column is negated here before joining,
so the merged table uses one consistent convention throughout (spec section
2, point 3).

Partial-day rule (spec section 2, point 4): a delivery day only counts in
any table with a full hour count (23/24/25 for the backtest, 96 quarter-hours
for the live reference). Incomplete days are excluded and counted.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from energy_price_forecast.common.data.loaders import load_interim_hourly, load_processed_features
from energy_price_forecast.common.evaluation.metrics import bias, rmse
from energy_price_forecast.common.evaluation.walkforward import (
    last_complete_local_day,
    run_backtest,
    split_complete_delivery_days,
    walk_forward_splits,
)
from energy_price_forecast.common.features.merit_order import build_merit_order_frame
from energy_price_forecast.common.market_time import LOCAL_TZ
from energy_price_forecast.common.models.lgbm import LGBMForecaster
from energy_price_forecast.common.models.merit_order_forecaster import MeritOrderForecaster

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TRAIN_SPAN_DAYS = 90
BACKTEST_START = "2026-09-01"
DAILY_TABLE_START = pd.Timestamp("2026-10-01", tz=LOCAL_TZ)


def _daily_metrics(
    preds: pd.DataFrame, complete_days: pd.DatetimeIndex, label: str
) -> pd.DataFrame:
    sub = preds.loc[preds["delivery_day"].isin(complete_days)]
    rows = [
        {
            "delivery_day": day,
            f"bias_{label}": bias(g["y_true"], g["y_pred"]),
            f"rmse_{label}": rmse(g["y_true"], g["y_pred"]),
        }
        for day, g in sub.groupby("delivery_day")
    ]
    return pd.DataFrame(rows)


def _load_live_daily(repo2_csv: Path) -> tuple[pd.DataFrame, int]:
    """Daily live bias/RMSE, sign-flipped to this repo's convention, restricted
    to fully quarter-hourly-matched days. Returns (table, n_excluded_partial).
    """
    df = pd.read_csv(repo2_csv)
    complete = df.loc[df["n_matched"] == df["n"]].copy()
    n_excluded = len(df) - len(complete)
    complete["delivery_day"] = pd.to_datetime(complete["target_day"]).dt.tz_localize(LOCAL_TZ)
    complete["bias_live"] = -complete["bias"]  # Repo 2: pred - y_true -> flip to y_true - pred
    complete["rmse_live"] = complete["rmse"]
    return complete[["delivery_day", "bias_live", "rmse_live"]], n_excluded


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-path", type=Path, default=Path("data/interim/hourly_merit_order.parquet")
    )
    p.add_argument(
        "--features-path", type=Path, default=Path("data/processed/features_merit_order.parquet")
    )
    p.add_argument(
        "--live-csv", type=Path, default=Path("data/raw/scratch_payload_vs_realized.csv")
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/results/error_decomposition_edge_closure_daily.csv"),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    df = load_interim_hourly(args.data_path)
    price = df["day_ahead_price"]
    features = load_processed_features(args.features_path)
    y = price.reindex(features.index)
    x = features
    index = pd.DatetimeIndex(features.index)

    edge = last_complete_local_day(price)
    logger.info("last locally-complete delivery day (derived from data): %s", edge.date())

    long_history = build_merit_order_frame(df, pd.DatetimeIndex(df.index))

    folds = list(
        walk_forward_splits(
            index,
            test_start=BACKTEST_START,
            test_end=edge.strftime("%Y-%m-%d"),
            window="rolling",
            train_span_days=TRAIN_SPAN_DAYS,
        )
    )
    logger.info(
        "folds: %d, delivery days %s to %s", len(folds), folds[0].delivery_day.date(), edge.date()
    )

    models = {
        "A": LGBMForecaster(objective="quantile", alpha=0.5),
        "B": MeritOrderForecaster(long_history, srmc_definition="gas_only"),
    }

    daily_tables = []
    for name, model in models.items():
        preds = run_backtest(y, model, folds, refit_every=1, x=x)
        complete_days, incomplete_days = split_complete_delivery_days(preds)
        if len(incomplete_days):
            logger.info(
                "variant %s: %d incomplete delivery day(s) excluded: %s",
                name,
                len(incomplete_days),
                [d.date() for d in incomplete_days],
            )
        daily = _daily_metrics(preds, complete_days, name)
        daily_tables.append(daily.set_index("delivery_day"))

    model_daily = pd.concat(daily_tables, axis=1).reset_index()
    model_daily = model_daily.loc[model_daily["delivery_day"] >= DAILY_TABLE_START]

    live_daily, n_live_excluded = _load_live_daily(args.live_csv)
    logger.info(
        "live reference: %d complete days, %d excluded (partial or unmatched)",
        len(live_daily),
        n_live_excluded,
    )

    merged = model_daily.merge(live_daily, on="delivery_day", how="left").sort_values(
        "delivery_day"
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    logger.info("wrote %d rows to %s", len(merged), args.out_csv)


if __name__ == "__main__":
    main()
