from energy_price_forecast.common.features.core_gas_columns import (
    CALENDAR_COLUMNS,
    FUNDAMENTALS_COLUMNS,
    GAS_COLUMNS,
    L1_COLUMNS,
    NWP_TO_ENTSOE_FORECAST_COLUMN,
    PRICE_LAG_COLUMNS,
)

# The known full feature set (L0), from a real run of build_features.py over
# hourly_merit_order.parquet -- kept here as a plain list rather than reading
# the actual (gitignored, environment-dependent) parquet file, so this test
# does not depend on local data state.
_L0_COLUMNS = [
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
    "load_forecast_day_ahead",
    "wind_onshore_forecast",
    "wind_offshore_forecast",
    "solar_forecast",
    "residual_load_forecast",
    "renewable_share_forecast",
    "ttf_gas_lag_48h",
    "eua_co2_lag_48h",
    "eua_missing",
    "price_lag_24h",
    "price_lag_48h",
    "price_lag_168h",
    "price_roll_mean_24h",
    "price_roll_mean_168h",
    "load_actual_lag_48h",
    "load_actual_lag_168h",
    "load_forecast_error_lag_48h",
    "wind_onshore_forecast_error_lag_48h",
    "wind_offshore_forecast_error_lag_48h",
    "solar_forecast_error_lag_48h",
    "scheduled_net_export_lag_24h",
    "physical_net_export_lag_48h",
    "cross_border_deviation_lag_48h",
]

# Columns deliberately excluded from core_gas relative to the full set (7.3
# section 3.2's "Dropped relative to live"): EUA/eua_missing, actual-load
# lags, forecast-error lags, cross-border lags.
_EXPECTED_DROPPED = {
    "eua_co2_lag_48h",
    "eua_missing",
    "load_actual_lag_48h",
    "load_actual_lag_168h",
    "load_forecast_error_lag_48h",
    "wind_onshore_forecast_error_lag_48h",
    "wind_offshore_forecast_error_lag_48h",
    "solar_forecast_error_lag_48h",
    "scheduled_net_export_lag_24h",
    "physical_net_export_lag_48h",
    "cross_border_deviation_lag_48h",
}


def test_l1_columns_is_subset_of_full_feature_set() -> None:
    assert set(L1_COLUMNS) <= set(_L0_COLUMNS)


def test_l1_columns_excludes_exactly_the_documented_drops() -> None:
    assert set(_L0_COLUMNS) - set(L1_COLUMNS) == _EXPECTED_DROPPED


def test_l1_columns_no_duplicates() -> None:
    assert len(L1_COLUMNS) == len(set(L1_COLUMNS))


def test_l1_columns_composed_of_the_four_groups() -> None:
    assert set(L1_COLUMNS) == (
        set(CALENDAR_COLUMNS)
        | set(PRICE_LAG_COLUMNS)
        | set(GAS_COLUMNS)
        | set(FUNDAMENTALS_COLUMNS)
    )


def test_fundamentals_columns_is_load_plus_nwp_mapping_targets() -> None:
    assert set(FUNDAMENTALS_COLUMNS) == {"load_forecast_day_ahead"} | set(
        NWP_TO_ENTSOE_FORECAST_COLUMN.values()
    )


def test_nwp_mapping_values_are_distinct_entsoe_columns() -> None:
    values = list(NWP_TO_ENTSOE_FORECAST_COLUMN.values())
    assert len(values) == len(set(values))
    assert all(isinstance(v, str) and v for v in values)
