"""NWP-reconstruction-based forecast fundamentals.

Replaces exactly the renewables-forecast terms of fundamentals.py's
build_forecast_fundamentals (wind_onshore_forecast, wind_offshore_forecast,
solar_forecast, and the residual-load term derived from them) with this
project's own weather-run-based reconstruction (model_renewables) -- the
only terms the availability audit found unavailable at gate closure.
Everything else (load_forecast_day_ahead, forecast-error features,
price/actual/cross-border lags) stays on the original TSO series,
unchanged, in fundamentals.py/lags.py -- neither of those modules is touched
by this one.

New columns use an `_nwp` suffix and coexist with, rather than overwrite,
the originals: the two feature-building paths (build_forecast_fundamentals
vs. build_nwp_forecast_fundamentals) stay independently callable, so they
can be compared.
"""

from __future__ import annotations

import pandas as pd

from energy_price_forecast.common.features.availability import (
    Feature,
    combine,
    forecast_for_target,
    nwp_reconstruction_for_target,
)

# Artefact columns (evaluation/renewables_walkforward.run_renewables_backtest's
# output) that must fully cover a target day for it to be usable.
_NWP_SOURCE_MW_COLUMNS: tuple[str, ...] = (
    "wind_onshore_mw_pred",
    "wind_offshore_mw_pred",
    "solar_mw_pred",
)


class IncompleteReconstructionError(ValueError):
    """Raised when the NWP reconstruction artefact does not fully cover a
    target day's hourly rows.

    The whole target day is unusable, not just the missing hour(s) -- never
    caught and filled, interpolated, or backfilled from the TSO series (that
    would be exactly the leak this step exists to prevent). Callers building
    a backtest across many days must catch this per day, count it, and
    exclude the day.
    """


def _check_reconstruction_coverage(
    renewables_predictions: pd.DataFrame, target_index: pd.DatetimeIndex
) -> None:
    valid_time = pd.DatetimeIndex(renewables_predictions.index.get_level_values("valid_time_utc"))
    missing_hours = target_index.difference(valid_time)
    if len(missing_hours) > 0:
        raise IncompleteReconstructionError(
            f"{len(missing_hours)} hour(s) of {len(target_index)} missing entirely from the "
            f"NWP reconstruction artefact for this target day (first missing: "
            f"{missing_hours.min()})"
        )
    rows = renewables_predictions.loc[valid_time.isin(target_index)]
    for mw_column in _NWP_SOURCE_MW_COLUMNS:
        nan_hours = rows.index.get_level_values("valid_time_utc")[rows[mw_column].isna()]
        if len(nan_hours) > 0:
            raise IncompleteReconstructionError(
                f"{len(nan_hours)} hour(s) of {len(target_index)} are NaN in {mw_column!r} "
                f"for this target day (first: {min(nan_hours)})"
            )


def _residual_load_formula(
    load: pd.Series, won: pd.Series, woff: pd.Series, solar: pd.Series
) -> pd.Series:
    """Same formula as fundamentals.py's residual_load_forecast term -- read
    there first, reproduced here exactly, not shared code. Proven
    equivalent, not just visually identical, by the identity test: feeding
    this function TSO-sourced Features instead of the NWP reconstruction
    must reproduce the original residual_load_forecast column bit-exact.
    """
    return load - won - woff - solar


def build_residual_load_nwp(load: Feature, won: Feature, woff: Feature, solar: Feature) -> Feature:
    """residual_load_forecast_nwp = load - won - woff - solar.

    Exposed standalone, not inlined into build_nwp_forecast_fundamentals, so
    the identity test can call it directly with TSO-sourced won/woff/solar
    Features and compare the result against the original
    residual_load_forecast column.
    """
    return combine("residual_load_forecast_nwp", [load, won, woff, solar], _residual_load_formula)


def _renewable_share_formula(
    load: pd.Series, won: pd.Series, woff: pd.Series, solar: pd.Series
) -> pd.Series:
    """Same formula as fundamentals.py's renewable_share_forecast term --
    reproduced here exactly, not shared code, same identity-test discipline
    as _residual_load_formula above."""
    return (won + woff + solar) / load


def build_renewable_share_nwp(
    load: Feature, won: Feature, woff: Feature, solar: Feature
) -> Feature:
    """renewable_share_forecast_nwp = (won + woff + solar) / load.

    The 'original' feature set carries the TSO-sourced
    renewable_share_forecast, so 'live' needs an NWP-sourced counterpart or
    the two candidates would differ by more than reconstruction quality
    alone. Exposed standalone for the same reason as build_residual_load_nwp:
    the identity test calls it directly with TSO-sourced Features.
    """
    return combine(
        "renewable_share_forecast_nwp", [load, won, woff, solar], _renewable_share_formula
    )


def build_nwp_forecast_fundamentals(
    df: pd.DataFrame,
    renewables_predictions: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> list[Feature]:
    """The live-viable forecast fundamentals for target_index's rows.

    load_forecast_day_ahead is unchanged (still DA_FORECAST -- verified
    available at gate closure, no reason to rebuild it);
    wind_onshore_forecast_nwp / wind_offshore_forecast_nwp /
    solar_forecast_nwp / residual_load_forecast_nwp /
    renewable_share_forecast_nwp come from this project's own reconstruction
    (NWP_RECONSTRUCTION). Contains no DA_FORECAST-class renewables column.

    Raises IncompleteReconstructionError (whole target day) if
    renewables_predictions does not fully cover target_index for all three
    reconstructed series.
    """
    _check_reconstruction_coverage(renewables_predictions, target_index)

    load = forecast_for_target(
        "load_forecast_day_ahead",
        df["load_forecast_day_ahead"],
        "load_forecast_day_ahead",
        target_index=target_index,
    )
    won_nwp = nwp_reconstruction_for_target(
        "wind_onshore_forecast_nwp",
        renewables_predictions,
        "wind_onshore_mw_pred",
        "wind_onshore_forecast_nwp",
        target_index=target_index,
    )
    woff_nwp = nwp_reconstruction_for_target(
        "wind_offshore_forecast_nwp",
        renewables_predictions,
        "wind_offshore_mw_pred",
        "wind_offshore_forecast_nwp",
        target_index=target_index,
    )
    solar_nwp = nwp_reconstruction_for_target(
        "solar_forecast_nwp",
        renewables_predictions,
        "solar_mw_pred",
        "solar_forecast_nwp",
        target_index=target_index,
    )
    residual_nwp = build_residual_load_nwp(load, won_nwp, woff_nwp, solar_nwp)
    share_nwp = build_renewable_share_nwp(load, won_nwp, woff_nwp, solar_nwp)
    return [load, won_nwp, woff_nwp, solar_nwp, residual_nwp, share_nwp]
