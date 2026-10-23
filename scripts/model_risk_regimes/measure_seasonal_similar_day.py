"""docs/seasonal_window_similar_day_spec.md -- Phase 1 (Ruckrechnung).

Runs the 2x2x2 factorial (train window: rolling90/seasonal x similar-day:
without/with x model: A/B) on L1, over the common window starting
2024-04-01 (chosen so the seasonal window's last-year part sits entirely
after the crisis period, FeatureConfig.post_crisis_start). B always uses
MeritOrderForecaster's default beta_update_rule="calendar_month" (Sprint
7.5 Teil 0); no extra wiring needed.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from energy_price_forecast.common.data.loaders import load_interim_hourly, load_processed_features
from energy_price_forecast.common.evaluation.walkforward import (
    Fold,
    last_complete_local_day,
    run_backtest,
    walk_forward_splits,
)
from energy_price_forecast.common.features.core_gas_columns import L1_COLUMNS
from energy_price_forecast.common.features.merit_order import build_merit_order_frame
from energy_price_forecast.common.features.similar_day import (
    build_similar_day_feature,
    is_complete_price_day,
    is_holiday_de,
)
from energy_price_forecast.common.models.lgbm import LGBMForecaster
from energy_price_forecast.common.models.merit_order_forecaster import MeritOrderForecaster

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TEST_START = "2024-04-01"
ROLLING_TRAIN_SPAN_DAYS = 90
SEASONAL_RECENT_DAYS = 60
SEASONAL_LAST_YEAR_DAYS = 30
PROBE_FOLDS = 20
RUNTIME_BUDGET_HOURS = 4.0  # spec section 9, Rueckfrage trigger 2

WINDOWS: tuple[str, ...] = ("rolling90", "seasonal")
SIMILAR_DAY_OPTIONS: tuple[str, ...] = ("no_similar", "similar")
MODELS: tuple[str, ...] = ("A", "B")


def _variant_name(window: str, similar_day: str, model: str) -> str:
    suffix = "" if similar_day == "no_similar" else "_similar"
    return f"{window}{suffix}_{model}"


def _build_folds(index: pd.DatetimeIndex, edge: pd.Timestamp, window: str) -> list[Fold]:
    if window == "rolling90":
        return list(
            walk_forward_splits(
                index,
                test_start=TEST_START,
                test_end=edge.strftime("%Y-%m-%d"),
                window="rolling",
                train_span_days=ROLLING_TRAIN_SPAN_DAYS,
            )
        )
    return list(
        walk_forward_splits(
            index,
            test_start=TEST_START,
            test_end=edge.strftime("%Y-%m-%d"),
            window="seasonal",
            recent_days=SEASONAL_RECENT_DAYS,
            last_year_days=SEASONAL_LAST_YEAR_DAYS,
        )
    )


def _build_model(model: str, long_history: pd.DataFrame) -> LGBMForecaster | MeritOrderForecaster:
    if model == "A":
        return LGBMForecaster(objective="quantile", alpha=0.5)
    return MeritOrderForecaster(long_history, srmc_definition="gas_only")


def _get_or_run_backtest(
    name: str,
    model: LGBMForecaster | MeritOrderForecaster,
    folds: list[Fold],
    x: pd.DataFrame,
    y: pd.Series,
    preds_dir: Path,
) -> pd.DataFrame:
    cache_path = preds_dir / f"seasonal_similar_day_preds_{name}.parquet"
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
        "--preds-dir", type=Path, default=Path("data/processed/seasonal_similar_day_preds")
    )
    p.add_argument("--runtime-budget-hours", type=float, default=RUNTIME_BUDGET_HOURS)
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
    similar_day_feature = build_similar_day_feature(
        price,
        index,
        is_holiday=is_holiday_de,
        is_complete=is_complete_price_day(price),
    )
    l1_matrix_similar = l1_matrix.assign(price_similar_day=similar_day_feature.values)

    matrices = {
        "no_similar": l1_matrix,
        "similar": l1_matrix_similar,
    }

    long_history = build_merit_order_frame(df, pd.DatetimeIndex(df.index))

    folds_by_window = {w: _build_folds(index, edge, w) for w in WINDOWS}
    for w, folds in folds_by_window.items():
        logger.info(
            "window %s: %d folds (%s to %s)",
            w,
            len(folds),
            folds[0].delivery_day.date(),
            folds[-1].delivery_day.date(),
        )
    rolling_days = {f.delivery_day for f in folds_by_window["rolling90"]}
    seasonal_days = {f.delivery_day for f in folds_by_window["seasonal"]}
    if rolling_days != seasonal_days:
        only_rolling = rolling_days - seasonal_days
        only_seasonal = seasonal_days - rolling_days
        logger.warning(
            "rolling90 and seasonal fold-day sets differ: %d only-rolling, %d only-seasonal "
            "-- metrics will not be computed on a strictly identical hour index",
            len(only_rolling),
            len(only_seasonal),
        )

    variant_configs = [
        (window, similar_day, model)
        for window in WINDOWS
        for similar_day in SIMILAR_DAY_OPTIONS
        for model in MODELS
    ]

    # --- Runtime probe (spec section 9, Rueckfrage trigger 2) ---------------
    # Never cached -- a stale cache would silently defeat a fresh timing probe.
    t0 = time.monotonic()
    for window, similar_day, model in variant_configs:
        probe_folds = folds_by_window[window][:PROBE_FOLDS]
        run_backtest(
            y,
            _build_model(model, long_history),
            probe_folds,
            refit_every=1,
            x=matrices[similar_day],
        )
    probe_elapsed = time.monotonic() - t0
    per_fold_per_variant_s = probe_elapsed / (len(variant_configs) * PROBE_FOLDS)
    total_variant_fold_count = sum(len(folds_by_window[window]) for window, _, _ in variant_configs)
    projected_hours = per_fold_per_variant_s * total_variant_fold_count / 3600
    logger.info(
        "runtime probe: %.3fs/fold/variant, projected %.2fh for all 8 variants",
        per_fold_per_variant_s,
        projected_hours,
    )
    if projected_hours > args.runtime_budget_hours:
        logger.error(
            "RUECKFRAGE (spec section 9): projected runtime %.2fh exceeds the %.1fh budget -- "
            "stopping, not running the full 8-variant backtest.",
            projected_hours,
            args.runtime_budget_hours,
        )
        return
    if args.probe_only:
        logger.info("--probe-only given -- stopping after the runtime probe.")
        return

    # --- The 8 variant runs ---------------------------------------------------
    args.preds_dir.mkdir(parents=True, exist_ok=True)
    for window, similar_day, model in variant_configs:
        name = _variant_name(window, similar_day, model)
        _get_or_run_backtest(
            name,
            _build_model(model, long_history),
            folds_by_window[window],
            matrices[similar_day],
            y,
            args.preds_dir,
        )


if __name__ == "__main__":
    main()
