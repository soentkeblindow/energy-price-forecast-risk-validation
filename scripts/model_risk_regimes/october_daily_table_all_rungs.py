"""Teil A, point 1 of docs/error_decomposition_followup_spec.md.

Extends Teil 0's October daily bias/RMSE table (A and B only) to all five
ladder rungs (L0, L1, L2, O_RE, O_ALL), using the already-cached 2026-window
predictions from 7.3 Teil 1 (data/processed/ladder_preds_2026/) -- no new
backtest runs. Answers the core question: was any rung clearly better than
A on L0 on the three outlier days (Oct 11/18/19)?
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from energy_price_forecast.common.evaluation.metrics import bias, rmse
from energy_price_forecast.common.evaluation.walkforward import split_complete_delivery_days
from energy_price_forecast.common.market_time import LOCAL_TZ

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

RUNGS: tuple[str, ...] = ("L0", "L1", "L2", "O_RE", "O_ALL")
VARIANTS: tuple[str, ...] = ("A", "B")
DAILY_TABLE_START = pd.Timestamp("2026-10-01", tz=LOCAL_TZ)
OUTLIER_DAYS = (
    pd.Timestamp("2026-10-11", tz=LOCAL_TZ),
    pd.Timestamp("2026-10-18", tz=LOCAL_TZ),
    pd.Timestamp("2026-10-19", tz=LOCAL_TZ),
)


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

    Same logic as close_data_edge.py's _load_live_daily -- duplicated rather
    than cross-script-imported, matching this project's existing pattern of
    each script owning its own small I/O glue.
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
        "--ladder-preds-dir-2026", type=Path, default=Path("data/processed/ladder_preds_2026")
    )
    p.add_argument(
        "--live-csv", type=Path, default=Path("data/raw/scratch_payload_vs_realized.csv")
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/results/error_decomposition_october_daily_all_rungs.csv"),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    daily_tables = []
    for rung in RUNGS:
        for variant in VARIANTS:
            label = f"{rung}_{variant}"
            preds = pd.read_parquet(args.ladder_preds_dir_2026 / f"ladder_preds_{label}.parquet")
            complete_days, incomplete_days = split_complete_delivery_days(preds)
            if len(incomplete_days):
                logger.info(
                    "%s: %d incomplete delivery day(s) excluded", label, len(incomplete_days)
                )
            daily = _daily_metrics(preds, complete_days, label)
            daily_tables.append(daily.set_index("delivery_day"))

    all_rungs_daily = pd.concat(daily_tables, axis=1).reset_index()
    all_rungs_daily = all_rungs_daily.loc[all_rungs_daily["delivery_day"] >= DAILY_TABLE_START]

    live_daily, n_live_excluded = _load_live_daily(args.live_csv)
    logger.info(
        "live reference: %d complete days, %d excluded (partial or unmatched)",
        len(live_daily),
        n_live_excluded,
    )

    merged = all_rungs_daily.merge(live_daily, on="delivery_day", how="left").sort_values(
        "delivery_day"
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    logger.info("wrote %d rows to %s", len(merged), args.out_csv)

    logger.info("--- outlier days: bias_L0_A as reference, other rungs for comparison ---")
    for day in OUTLIER_DAYS:
        row = merged.loc[merged["delivery_day"] == day]
        if row.empty:
            logger.info("%s: no data", day.date())
            continue
        row = row.iloc[0]
        l0_a = row["bias_L0_A"]
        others = {
            f"{rung}_{variant}": row[f"bias_{rung}_{variant}"]
            for rung in RUNGS
            for variant in VARIANTS
            if f"{rung}_{variant}" != "L0_A"
        }
        best_label, best_value = min(others.items(), key=lambda kv: abs(kv[1]))
        logger.info(
            "%s: bias_L0_A=%.2f, best alternative=%s (%.2f), live=%.2f",
            day.date(),
            l0_a,
            best_label,
            best_value,
            row["bias_live"] if pd.notna(row["bias_live"]) else float("nan"),
        )


if __name__ == "__main__":
    main()
