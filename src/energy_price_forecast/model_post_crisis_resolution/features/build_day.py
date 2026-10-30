"""Per-day feature-set builders for the quarter-hour model.

Split out of common/features/build.py: these two functions and their
helpers depend on the quarter-hour model's own NWP-reconstruction
fundamentals (nwp_fundamentals.py), so common stays usable without the
quarter-hour model. build_feature_matrix and trim_warmup, which do not
depend on nwp_fundamentals, remain in common/features/build.py.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from energy_price_forecast.common.features.availability import build_matrix
from energy_price_forecast.common.features.calendar import build_calendar_features
from energy_price_forecast.common.features.config import FeatureConfig
from energy_price_forecast.common.features.fundamentals import (
    build_commodity_features,
    build_forecast_fundamentals,
)
from energy_price_forecast.common.features.lags import (
    build_actual_lags,
    build_cross_border_lags,
    build_forecast_error_lags,
    build_price_lags,
)
from energy_price_forecast.common.market_time import local_day_bounds
from energy_price_forecast.model_post_crisis_resolution.features.nwp_fundamentals import (
    build_nwp_forecast_fundamentals,
)


def _hourly_utc_index_for_local_day(target_day: dt.date) -> pd.DatetimeIndex:
    """UTC hourly timestamps of one local (Europe/Berlin) delivery day --
    23/24/25 rows across DST, derived purely from the calendar (never
    derived from the system clock)."""
    start, end = local_day_bounds(target_day)
    return pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")


def _build_day_matrix(
    target_index: pd.DatetimeIndex,
    fundamentals: list,
    df: pd.DataFrame,
    cfg: FeatureConfig,
) -> pd.DataFrame:
    """Shared Bausteinliste for one day's feature rows -- the same
    target_index, the same Bausteinliste, for both the live and the
    original feature-set builder below. Only the fundamentals (forecast
    columns) differ between the two callers; calendar, commodities and lags
    are identical."""
    features = [
        *build_calendar_features(target_index, cfg),
        *fundamentals,
        *build_commodity_features(df, target_index, cfg),
        *build_price_lags(df, target_index, cfg),
        *build_actual_lags(df, target_index, cfg),
        *build_forecast_error_lags(df, target_index, cfg),
        *build_cross_border_lags(df, target_index, cfg),
    ]
    return build_matrix(features)  # asserts no leakage, then concatenates


def build_feature_set_for_day(
    target_day: dt.date,
    df: pd.DataFrame,
    renewables_predictions: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """The complete, live-viable feature set for one local delivery day.

    Calendar/regime, NWP-reconstruction-based forecast fundamentals
    (features/nwp_fundamentals.py), commodities, and the unchanged
    price/actual/forecast-error/cross-border lags (fundamentals.py/lags.py,
    untouched) -- for exactly target_day's hourly rows. Contains no
    DA_FORECAST-class renewables column.

    Backtest and live call this function identically, with target_day as
    the only day-identifying argument -- never the system clock. What
    differs between the two contexts is only which rows `df` and
    `renewables_predictions` happen to contain at call time, never the
    code path.

    Raises IncompleteReconstructionError (features.nwp_fundamentals) if the
    renewables reconstruction does not fully cover target_day -- the whole
    day is unusable, not partially fillable.
    """
    cfg = config if config is not None else FeatureConfig()
    target_index = _hourly_utc_index_for_local_day(target_day)
    fundamentals = build_nwp_forecast_fundamentals(df, renewables_predictions, target_index)
    return _build_day_matrix(target_index, fundamentals, df, cfg)


def build_original_feature_set_for_day(
    target_day: dt.date,
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """The 'original' (upper-bound) feature set for one local delivery day
    -- the same day-by-day construction as build_feature_set_for_day above,
    sharing its Bausteinliste, but with the TSO-sourced fundamentals
    (build_forecast_fundamentals) instead of the NWP reconstruction.

    Never a live candidate: contains DA_FORECAST-class renewables columns
    that the availability audit found unavailable at gate closure. Exists
    so both feature sets can be measured through the identical per-day code
    path -- not by reading the precomputed features_hourly.parquet, a different
    code path.
    """
    cfg = config if config is not None else FeatureConfig()
    target_index = _hourly_utc_index_for_local_day(target_day)
    fundamentals = build_forecast_fundamentals(df, target_index)
    return _build_day_matrix(target_index, fundamentals, df, cfg)
