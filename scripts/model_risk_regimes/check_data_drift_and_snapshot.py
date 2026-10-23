"""Sprint 7 Step 7.6, Teil A (docs/sprint7_step7_6_spec.md, section 2).

A.1 -- Datendrift-Pruefauftrag: re-runs the L2_B configuration from 7.3's
error-decomposition ladder, but with ``beta_update_rule="rolling_30d"``
(the legacy rule 7.3's own cache was built under, before
``beta_update_rule`` existed at all) starting 2026-01-04, on today's data
stand. Daily bias on 2026-10-11/18/19 from this run (column (2)) is compared
against 7.3's own cached value (column (1), read live from
--reference-csv) and against the new calendar_month rule's value measured
in Sprint 7.5 (column (3), fixed below -- not recomputed here, see
docs/sprint7_step7_5_log.md's owner-review table).

A.2 -- Datenstand festhalten: fingerprints (name, time range, row count,
sha256) the two input files Teil B will read, via
common.data.snapshot.file_fingerprint, and writes
outputs/results/price_model_selection_data_snapshot.csv.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from energy_price_forecast.common.data.loaders import load_interim_hourly, load_processed_features
from energy_price_forecast.common.data.snapshot import file_fingerprint
from energy_price_forecast.common.evaluation.metrics import bias
from energy_price_forecast.common.evaluation.walkforward import (
    last_complete_local_day,
    run_backtest,
    walk_forward_splits,
)
from energy_price_forecast.common.features.core_gas_columns import L1_COLUMNS
from energy_price_forecast.common.features.merit_order import build_merit_order_frame
from energy_price_forecast.common.models.merit_order_forecaster import MeritOrderForecaster
from energy_price_forecast.model_risk_regimes.features.l2_matrix import build_l2_matrix

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TRAIN_SPAN_DAYS = 90
# Spec A.1's literal instruction: this run's cadence anchor must match 7.3's
# own cache (ladder_preds_L2_B.parquet, first fold 2026-01-04), so (1) vs (2)
# isolates pure data drift rather than a different beta cadence anchor.
RUN_START = "2026-01-04"
DRIFT_CHECK_DAYS = ("2026-10-11", "2026-10-18", "2026-10-19")

# Column (3): new rule (calendar_month), today's-at-the-time data stand,
# already measured in Sprint 7 Step 7.5's reproducibility run -- not
# recomputed here. Source: docs/sprint7_step7_5_log.md, owner-review table.
NEW_RULE_BIAS = {
    "2026-10-11": 31.3,
    "2026-10-18": 39.5,
    "2026-10-19": 29.7,
}

A2_INPUT_FILES: tuple[Path, ...] = (
    Path("data/interim/hourly_merit_order.parquet"),
    Path("data/processed/features_merit_order.parquet"),
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-path", type=Path, default=Path("data/interim/hourly_merit_order.parquet")
    )
    p.add_argument(
        "--features-path", type=Path, default=Path("data/processed/features_merit_order.parquet")
    )
    p.add_argument(
        "--renewables-path",
        type=Path,
        default=Path("data/processed/renewables_forecast_rolling365_l2.parquet"),
    )
    p.add_argument("--preds-dir", type=Path, default=Path("data/processed/data_drift_check_preds"))
    p.add_argument(
        "--reference-csv",
        type=Path,
        default=Path("outputs/results/error_decomposition_october_daily_all_rungs.csv"),
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/results/price_model_selection_data_snapshot.csv"),
    )
    return p.parse_args()


def _reference_bias(reference_csv: Path) -> dict[str, float]:
    """Column (1): 7.3's own L2_B daily bias, read live so the numbers never
    go stale if the CSV is regenerated.
    """
    df = pd.read_csv(reference_csv, index_col="delivery_day", parse_dates=True)
    df.index = df.index.normalize()
    out = {}
    for day in DRIFT_CHECK_DAYS:
        out[day] = float(df.loc[pd.Timestamp(day, tz=df.index.tz), "bias_L2_B"])
    return out


def main() -> None:
    args = _parse_args()
    df = load_interim_hourly(args.data_path)
    price = df["day_ahead_price"]
    features = load_processed_features(args.features_path)
    y = price.reindex(features.index)
    index = pd.DatetimeIndex(features.index)

    edge = last_complete_local_day(price)
    logger.info("data edge (last locally-complete delivery day): %s", edge.date())

    l1_matrix = features[list(L1_COLUMNS)]
    renewables_predictions = pd.read_parquet(args.renewables_path)
    l2_matrix = build_l2_matrix(l1_matrix, renewables_predictions, index)
    long_history = build_merit_order_frame(df, pd.DatetimeIndex(df.index))

    folds = list(
        walk_forward_splits(
            index,
            test_start=RUN_START,
            test_end=edge.strftime("%Y-%m-%d"),
            window="rolling",
            train_span_days=TRAIN_SPAN_DAYS,
        )
    )
    logger.info(
        "run window: %s to %s (%d folds) -- assumed fully inside L2's NWP "
        "reconstruction coverage (starts ~2024-03, per "
        "run_error_decomposition_ladder.py's own docstring)",
        folds[0].delivery_day.date(),
        folds[-1].delivery_day.date(),
        len(folds),
    )

    cache_path = args.preds_dir / "ladder_preds_L2_B_rolling30d_2026start.parquet"
    if cache_path.exists():
        logger.info("loaded cached predictions from %s", cache_path)
        preds = pd.read_parquet(cache_path)
    else:
        model = MeritOrderForecaster(
            long_history, srmc_definition="gas_only", beta_update_rule="rolling_30d"
        )
        t0 = time.monotonic()
        preds = run_backtest(y, model, folds, refit_every=1, x=l2_matrix)
        elapsed = time.monotonic() - t0
        logger.info(
            "backtest done: %d test hours in %.1fs (%.3fs/fold)",
            len(preds),
            elapsed,
            elapsed / len(folds) if folds else float("nan"),
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(cache_path)

    local_day = preds["delivery_day"].dt.normalize()
    old_rule_bias = {}
    for day in DRIFT_CHECK_DAYS:
        day_rows = preds[local_day == pd.Timestamp(day, tz=local_day.dt.tz)]
        old_rule_bias[day] = bias(day_rows["y_true"], day_rows["y_pred"])

    reference_bias = _reference_bias(args.reference_csv)

    logger.info("Drift/rule-effect table (Tag | (1) 7.3 | (2) alte Regel | (3) neue Regel):")
    for day in DRIFT_CHECK_DAYS:
        col1 = reference_bias[day]
        col2 = old_rule_bias[day]
        col3 = NEW_RULE_BIAS[day]
        logger.info(
            "%s | %.2f | %.2f | %.2f  (drift=(1)-(2)=%.2f, rule_effect=(2)-(3)=%.2f)",
            day,
            col1,
            col2,
            col3,
            col1 - col2,
            col2 - col3,
        )

    logger.info("A.2 data-stand snapshot:")
    rows = [file_fingerprint(p) for p in A2_INPUT_FILES]
    for row in rows:
        logger.info("%s", row)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    logger.info("wrote %s", args.out_csv)


if __name__ == "__main__":
    main()
