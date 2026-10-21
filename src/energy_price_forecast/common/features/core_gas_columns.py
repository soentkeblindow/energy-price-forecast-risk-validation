"""Frozen column recipe for the live model's reduced feature set (``core_gas``),
used by the error-decomposition ladder (7.3, Step 3.2,
docs/residual_error_decomposition_hourly_spec.md). L1 is this set built from
ENTSO-E day-ahead forecasts; L2 is the same set with the five NWP-mapped
columns coming from the renewables reconstruction instead (7.3 section 3.2/3.3).

Source: ``sbl-energy-forecast`` commit 019e1e4abace5a2ed34cf2e983568ec9b2654aed
(2026-10-20), read-only (owner-approved for 7.3). Ranks 1 (``core_gas``) and
2 (``core_gas_loadpatch``) of ``src/energy_price_forecast/arena/candidates.py
::three_row_ladder`` share identical columns -- only the load-forecast
*source* differs between them (ENTSO-E vs. a same-day patch), not which
columns are used -- so there is no rank ambiguity to resolve here (unlike the
"mehrere Varianten je Fallback-Stufe" case the spec itself anticipated as a
possible Rueckfrage, section 11).

Column groups, traced to the real builders (not hand-guessed):
- calendar: ``src/energy_price_forecast/features/calendar.py
  ::build_calendar_features`` (11 columns, identical names to this repo's own
  ``build_calendar_features`` -- same shared heritage).
- price lags: ``features/lags.py::build_price_lags`` (same shared heritage,
  identical names to this repo's own price-lag columns).
- gas: ``scripts/ablation_core_minimal_feature_set.py::build_core_for_day``,
  ``commodities="gas_only"`` -> ``build_commodity_features(...)[:1]`` --
  TTF only, no EUA, no ``eua_missing``.
- NWP fundamentals: ``features/nwp_fundamentals.py
  ::build_nwp_forecast_fundamentals`` -- 6 columns: ``load_forecast_day_ahead``
  (unchanged DA_FORECAST) plus 5 reconstructed from ``renewables_predictions``
  (``wind_onshore_forecast_nwp``, ``wind_offshore_forecast_nwp``,
  ``solar_forecast_nwp``, ``residual_load_forecast_nwp``,
  ``renewable_share_forecast_nwp``).

Explicitly dropped relative to the full feature set (L0): actual-load lags,
forecast-error lags, cross-border-flow lags, EUA/eua_missing.
"""

from __future__ import annotations

CALENDAR_COLUMNS: tuple[str, ...] = (
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_holiday",
    "is_regional_holiday",
    "is_crisis",
    "is_post_crisis",
)

PRICE_LAG_COLUMNS: tuple[str, ...] = (
    "price_lag_24h",
    "price_lag_48h",
    "price_lag_168h",
    "price_roll_mean_24h",
    "price_roll_mean_168h",
)

# Gas-only: no EUA, no eua_missing (spec 7.2 section 2's eua_missing gap, and
# core_gas's own commodities="gas_only" choice in Repo 2).
GAS_COLUMNS: tuple[str, ...] = ("ttf_gas_lag_48h",)

# The 5 live NWP-reconstructed columns, mapped to their ENTSO-E-forecast
# counterparts for L1 (7.3 section 3.2: "Fuer L1 werden ... die NWP-Spalten
# durch ihre ENTSO-E-Gegenstuecke ersetzt"). load_forecast_day_ahead is not
# NWP-reconstructed live either way, so it is not in this mapping -- it is
# added to L1_COLUMNS directly below.
NWP_TO_ENTSOE_FORECAST_COLUMN: dict[str, str] = {
    "wind_onshore_forecast_nwp": "wind_onshore_forecast",
    "wind_offshore_forecast_nwp": "wind_offshore_forecast",
    "solar_forecast_nwp": "solar_forecast",
    "residual_load_forecast_nwp": "residual_load_forecast",
    "renewable_share_forecast_nwp": "renewable_share_forecast",
}

FUNDAMENTALS_COLUMNS: tuple[str, ...] = (
    "load_forecast_day_ahead",
    *NWP_TO_ENTSOE_FORECAST_COLUMN.values(),
)

# L1 (core_gas, built from ENTSO-E forecasts): a pure column subset of this
# repo's existing full feature matrix (features_merit_order.parquet) -- no
# new data build needed. L2 swaps FUNDAMENTALS_COLUMNS' 5 non-load columns
# for NWP-reconstructed values (7.3 section 3.3), same column names.
L1_COLUMNS: tuple[str, ...] = (
    *CALENDAR_COLUMNS,
    *PRICE_LAG_COLUMNS,
    *GAS_COLUMNS,
    *FUNDAMENTALS_COLUMNS,
)
