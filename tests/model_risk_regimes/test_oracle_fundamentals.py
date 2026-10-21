import pandas as pd

from energy_price_forecast.common.features.merit_order import (
    renewable_share_from_components,
    residual_load_from_components,
)
from energy_price_forecast.model_risk_regimes.features.oracle_fundamentals import (
    ORACLE_SWAPPED_COLUMNS,
    build_oracle_feature_frame,
    build_oracle_long_history_frame,
)


def _synthetic_df_and_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2024-06-01", periods=5, freq="h", tz="UTC")
    # Forecast and actual deliberately differ in every component, so any
    # column NOT swapped would show up as unchanged (forecast values) and
    # any column that IS swapped shows up as the actual-derived values.
    df = pd.DataFrame(
        {
            "load_forecast_day_ahead": [100.0, 200.0, 300.0, 400.0, 500.0],
            "wind_onshore_forecast": [10.0, 20.0, 30.0, 40.0, 50.0],
            "wind_offshore_forecast": [1.0, 1.0, 1.0, 1.0, 1.0],
            "solar_forecast": [2.0, 2.0, 2.0, 2.0, 2.0],
            "load_actual": [110.0, 210.0, 310.0, 410.0, 510.0],
            "gen_wind_onshore": [11.0, 21.0, 31.0, 41.0, 51.0],
            "gen_wind_offshore": [1.5, 1.5, 1.5, 1.5, 1.5],
            "gen_solar": [2.5, 2.5, 2.5, 2.5, 2.5],
        },
        index=idx,
    )
    features = pd.DataFrame(
        {
            "hour_sin": [0.1, 0.2, 0.3, 0.4, 0.5],
            "price_lag_24h": [50.0, 51.0, 52.0, 53.0, 54.0],
            "ttf_gas_lag_48h": [30.0, 30.0, 30.0, 30.0, 30.0],
            "load_forecast_day_ahead": df["load_forecast_day_ahead"],
            "wind_onshore_forecast": df["wind_onshore_forecast"],
            "wind_offshore_forecast": df["wind_offshore_forecast"],
            "solar_forecast": df["solar_forecast"],
            "residual_load_forecast": residual_load_from_components(
                df["load_forecast_day_ahead"],
                df["wind_onshore_forecast"],
                df["wind_offshore_forecast"],
                df["solar_forecast"],
            ),
            "renewable_share_forecast": renewable_share_from_components(
                df["load_forecast_day_ahead"],
                df["wind_onshore_forecast"],
                df["wind_offshore_forecast"],
                df["solar_forecast"],
            ),
        },
        index=idx,
    )
    return df, features


def test_without_oracle_call_no_actual_column_in_the_matrix() -> None:
    # Abstinence (spec section 10): a plain L1-shaped matrix, never passed
    # through build_oracle_feature_frame, must not contain any *_actual
    # column and must keep the forecast-based residual_load_forecast.
    df, features = _synthetic_df_and_features()
    assert not any(col.endswith("_actual") for col in features.columns)
    expected_residual = residual_load_from_components(
        df["load_forecast_day_ahead"],
        df["wind_onshore_forecast"],
        df["wind_offshore_forecast"],
        df["solar_forecast"],
    )
    pd.testing.assert_series_equal(
        features["residual_load_forecast"], expected_residual, check_names=False
    )


def test_o_re_swaps_only_the_five_fundamentals_columns() -> None:
    df, features = _synthetic_df_and_features()
    oracle = build_oracle_feature_frame(features, df, "o_re")

    changed = [c for c in features.columns if not features[c].equals(oracle[c])]
    assert set(changed) == set(ORACLE_SWAPPED_COLUMNS)

    # load_forecast_day_ahead (O-RE keeps it) and every non-fundamentals
    # column stay bit-identical.
    unchanged = [c for c in features.columns if c not in ORACLE_SWAPPED_COLUMNS]
    for col in unchanged:
        pd.testing.assert_series_equal(features[col], oracle[col])

    pd.testing.assert_series_equal(
        oracle["wind_onshore_forecast"], df["gen_wind_onshore"], check_names=False
    )


def test_o_all_additionally_swaps_load() -> None:
    df, features = _synthetic_df_and_features()
    oracle = build_oracle_feature_frame(features, df, "o_all")

    changed = [c for c in features.columns if not features[c].equals(oracle[c])]
    assert set(changed) == set(ORACLE_SWAPPED_COLUMNS) | {"load_forecast_day_ahead"}
    pd.testing.assert_series_equal(
        oracle["load_forecast_day_ahead"], df["load_actual"], check_names=False
    )


def test_oracle_long_history_frame_replaces_residual_load_and_load() -> None:
    df, _ = _synthetic_df_and_features()
    long_history = pd.DataFrame(
        {
            "price": [1.0] * 5,
            "residual_load": [999.0] * 5,  # forecast-based, deliberately wrong
            "load": [999.0] * 5,
            "ttf_lag": [10.0] * 5,
            "eua_lag": [5.0] * 5,
            "is_crisis": False,
        },
        index=df.index,
    )

    oracle_re = build_oracle_long_history_frame(long_history, df, "o_re")
    expected_residual_re = residual_load_from_components(
        df["load_forecast_day_ahead"],
        df["gen_wind_onshore"],
        df["gen_wind_offshore"],
        df["gen_solar"],
    )
    pd.testing.assert_series_equal(
        oracle_re["residual_load"], expected_residual_re, check_names=False
    )
    pd.testing.assert_series_equal(
        oracle_re["load"], df["load_forecast_day_ahead"], check_names=False
    )
    # Untouched columns (price, ttf_lag, eua_lag, is_crisis) stay bit-identical.
    for col in ("price", "ttf_lag", "eua_lag", "is_crisis"):
        pd.testing.assert_series_equal(oracle_re[col], long_history[col])

    oracle_all = build_oracle_long_history_frame(long_history, df, "o_all")
    pd.testing.assert_series_equal(oracle_all["load"], df["load_actual"], check_names=False)
