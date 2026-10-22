"""Hauptteil ("Franzoesische Preise als Feature") of docs/sprint7_step7_5_spec.md.

Runs 5 variants of B on L2 (R, R+F3, R+F1, R+F2, R+FR) over the same
common window 7.3 established for L2 (test_start 2024-06-15, 90-day
rolling train, daily refit), identical folds, to measure whether a lagged
French day-ahead price feature improves the hourly model (spec sections
3-4). F3 (spread_fr_de_lag_24h) is the pre-registered primary candidate
(spec section 2); F1/F2/R+FR are sensitivities, reported but never
selected as a winner after the fact.

Reuses (not copies) run_error_decomposition_ladder.py's _build_l2_matrix --
a sibling-script import, not a package dependency (see that module's own
docstring for the L1/L2 construction it implements).
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
from run_error_decomposition_ladder import _build_l2_matrix  # sibling script, see docstring

from energy_price_forecast.common.data.fr_prices import load_fr_day_ahead_hourly
from energy_price_forecast.common.data.loaders import load_interim_hourly, load_processed_features
from energy_price_forecast.common.evaluation.walkforward import (
    last_complete_local_day,
    run_backtest,
    walk_forward_splits,
)
from energy_price_forecast.common.features.availability import Feature, lag
from energy_price_forecast.common.features.core_gas_columns import L1_COLUMNS
from energy_price_forecast.common.features.merit_order import build_merit_order_frame
from energy_price_forecast.common.features.neighbor_prices import build_fr_price_features
from energy_price_forecast.common.models.merit_order_forecaster import MeritOrderForecaster

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TRAIN_SPAN_DAYS = 90
# 7.3's own L2 common-window start
# (docs/residual_error_decomposition_hourly_log.md, Teil 1) -- reused as-is
# per spec section 3 item 5, not re-derived.
TEST_START = "2024-06-15"
PROBE_FOLDS = 20
RUNTIME_BUDGET_HOURS = 3.0  # spec section 11, Rueckfrage trigger 3
VARIANTS: tuple[str, ...] = ("R", "R+F3", "R+F1", "R+F2", "R+FR")


def _variant_matrices(
    l2_matrix: pd.DataFrame, f1: Feature, f2: Feature, f3: Feature
) -> dict[str, pd.DataFrame]:
    return {
        "R": l2_matrix,
        "R+F3": l2_matrix.assign(**{f3.name: f3.values}),
        "R+F1": l2_matrix.assign(**{f1.name: f1.values}),
        "R+F2": l2_matrix.assign(**{f2.name: f2.values}),
        "R+FR": l2_matrix.assign(**{f1.name: f1.values, f2.name: f2.values, f3.name: f3.values}),
    }


def _get_or_run_backtest(
    name: str,
    model: MeritOrderForecaster,
    folds: list,
    x: pd.DataFrame,
    y: pd.Series,
    preds_dir: Path,
) -> pd.DataFrame:
    cache_path = preds_dir / f"fr_feature_preds_{name.replace('+', '')}.parquet"
    if cache_path.exists():
        logger.info("variant %s: loaded cached predictions from %s", name, cache_path)
        return pd.read_parquet(cache_path)
    t0 = time.monotonic()
    preds = run_backtest(y, model, folds, refit_every=1, x=x)
    elapsed = time.monotonic() - t0
    logger.info(
        "variant %s: %d test hours in %.1fs (%.3fs/fold)",
        name,
        len(preds),
        elapsed,
        elapsed / len(folds) if folds else float("nan"),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(cache_path)
    return preds


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
    p.add_argument("--fr-path", type=Path, default=Path("data/interim/fr_day_ahead_hourly.parquet"))
    p.add_argument("--preds-dir", type=Path, default=Path("data/processed/fr_feature_preds"))
    p.add_argument(
        "--runtime-budget-hours",
        type=float,
        default=RUNTIME_BUDGET_HOURS,
        help="Stop before the full run if the runtime probe projects more than this.",
    )
    p.add_argument(
        "--probe-only",
        action="store_true",
        help="Run only the runtime probe and report the projection, without the full backtests.",
    )
    return p.parse_args()


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
    l2_matrix = _build_l2_matrix(l1_matrix, renewables_predictions, index)

    fr_price = load_fr_day_ahead_hourly(args.fr_path)["fr_day_ahead_price"]
    de_price_lag_24h = lag(
        "price_lag_24h", df["day_ahead_price"], "day_ahead_price", hours=24, target_index=index
    )
    f1, f2, f3 = build_fr_price_features(fr_price, de_price_lag_24h, index)
    matrices = _variant_matrices(l2_matrix, f1, f2, f3)

    long_history = build_merit_order_frame(df, pd.DatetimeIndex(df.index))

    folds = list(
        walk_forward_splits(
            index,
            test_start=TEST_START,
            test_end=edge.strftime("%Y-%m-%d"),
            window="rolling",
            train_span_days=TRAIN_SPAN_DAYS,
        )
    )
    logger.info(
        "common window: %s to %s (%d folds)",
        folds[0].delivery_day.date(),
        folds[-1].delivery_day.date(),
        len(folds),
    )

    # --- Runtime probe (spec section 11, Rueckfrage trigger 3) --------------
    probe_folds = folds[:PROBE_FOLDS]
    t0 = time.monotonic()
    for name in VARIANTS:
        run_backtest(
            y,
            MeritOrderForecaster(long_history, srmc_definition="gas_only"),
            probe_folds,
            refit_every=1,
            x=matrices[name],
        )
    probe_elapsed = time.monotonic() - t0
    per_fold_per_variant_s = probe_elapsed / (len(VARIANTS) * len(probe_folds))
    projected_hours = per_fold_per_variant_s * len(folds) * len(VARIANTS) / 3600
    logger.info(
        "runtime probe: %.3fs/fold/variant, projected %.2fh for all %d variants over %d folds",
        per_fold_per_variant_s,
        projected_hours,
        len(VARIANTS),
        len(folds),
    )
    if projected_hours > args.runtime_budget_hours:
        logger.error(
            "RUECKFRAGE (spec section 11): projected runtime %.2fh exceeds the %.1fh budget -- "
            "stopping, not running the full 5-variant backtest.",
            projected_hours,
            args.runtime_budget_hours,
        )
        return
    if args.probe_only:
        logger.info("--probe-only given -- stopping after the runtime probe.")
        return

    # --- The 5 variant runs --------------------------------------------------
    args.preds_dir.mkdir(parents=True, exist_ok=True)
    for name in VARIANTS:
        _get_or_run_backtest(
            name,
            MeritOrderForecaster(long_history, srmc_definition="gas_only"),
            folds,
            matrices[name],
            y,
            args.preds_dir,
        )


if __name__ == "__main__":
    main()
