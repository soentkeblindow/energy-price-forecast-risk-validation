import numpy as np
import pandas as pd
import pytest

from energy_price_forecast.common.features.fundamentals import build_forecast_fundamentals
from energy_price_forecast.common.features.merit_order import (
    BETA_CLAMP_MAX,
    CO2_INTENSITY_EL,
    GAS_EFFICIENCY,
    RL_BIN_EDGES_MW,
    RL_BIN_EDGES_RELATIVE,
    RL_BIN_LABELS,
    estimate_beta_per_bin,
    raw_beta_per_bin,
    renewable_share_from_components,
    residual_load_from_components,
    rl_bin,
    rl_bin_relative,
    rl_relative,
    smooth_beta_isotonic,
    srmc_gas_full,
    srmc_gas_only,
    ttf_out_of_range_flag,
)
from energy_price_forecast.common.market_time import LOCAL_TZ


def test_srmc_gas_only_scales_ttf_by_efficiency() -> None:
    ttf = pd.Series([0.0, 49.13, 98.26])
    result = srmc_gas_only(ttf)
    expected = ttf / GAS_EFFICIENCY
    pd.testing.assert_series_equal(result, expected)


def test_srmc_gas_full_adds_carbon_cost() -> None:
    ttf = pd.Series([49.13])
    eua = pd.Series([80.0])
    result = srmc_gas_full(ttf, eua)
    expected = ttf / GAS_EFFICIENCY + CO2_INTENSITY_EL * eua
    pd.testing.assert_series_equal(result, expected)


def test_rl_bin_negative_residual_load_falls_in_bin_zero() -> None:
    rl = pd.Series([-5000.0, -0.01])
    bins = rl_bin(rl)
    assert (bins == 0).all()


def test_rl_bin_boundary_values_go_to_upper_bin() -> None:
    # Half-open on the right: a value exactly on an edge belongs to the bin above.
    rl = pd.Series([0.0, RL_BIN_EDGES_MW[2]])
    bins = rl_bin(rl)
    assert bins.iloc[0] == 1
    assert bins.iloc[1] == 2


def test_rl_bin_extreme_values_land_in_first_and_last_bin() -> None:
    rl = pd.Series([-1e9, 1e9])
    bins = rl_bin(rl)
    assert bins.iloc[0] == RL_BIN_LABELS[0]
    assert bins.iloc[1] == RL_BIN_LABELS[-1]


def test_ttf_out_of_range_flag_detects_spike_but_not_its_own_leaked_window() -> None:
    # 120 local days, flat ttf=50 except a single spike to 200 on day 100.
    idx = pd.date_range("2019-01-01", periods=120 * 24, freq="h", tz=LOCAL_TZ).tz_convert("UTC")
    local_day = pd.DatetimeIndex(idx).tz_convert(LOCAL_TZ).normalize()
    day_number = (local_day - local_day[0]).days
    ttf = pd.Series(50.0, index=idx)
    ttf.loc[day_number == 100] = 200.0

    flag = ttf_out_of_range_flag(ttf, idx, train_span_days=90)

    # Day 100: its own 90-day train window (days 10-99) never saw the spike,
    # so the spike hours are correctly flagged.
    assert flag.loc[day_number == 100].all()
    # Day 101: the train window (days 11-100) now INCLUDES the spike day, so
    # its own (non-spiking) ttf is not out of range -- the flag must not leak
    # the test day's own value into its own training window one day early.
    assert not flag.loc[day_number == 101].any()


def test_rl_bin_label_count_matches_edge_count() -> None:
    assert len(RL_BIN_LABELS) == len(RL_BIN_EDGES_MW) - 1


def test_rl_bin_preserves_nan() -> None:
    rl = pd.Series([np.nan, 100.0])
    bins = rl_bin(rl)
    assert pd.isna(bins.iloc[0])


def test_smooth_beta_isotonic_enforces_monotonicity() -> None:
    raw = pd.Series([0.2, 0.1, 0.5, 0.4, 0.9], index=[0, 1, 2, 3, 4])
    smoothed = smooth_beta_isotonic(raw)
    assert (smoothed.diff().dropna() >= -1e-12).all()


def test_smooth_beta_isotonic_clamps_to_bounds() -> None:
    raw = pd.Series([-1.0, 0.0, 2.0, 5.0], index=[0, 1, 2, 3])
    smoothed = smooth_beta_isotonic(raw, lower=0.0, upper=1.5)
    assert smoothed.min() >= 0.0
    assert smoothed.max() <= 1.5


def test_smooth_beta_isotonic_fills_nan_bins() -> None:
    raw = pd.Series([0.1, np.nan, 0.5, np.nan, 0.9], index=[0, 1, 2, 3, 4])
    smoothed = smooth_beta_isotonic(raw)
    assert smoothed.notna().all()
    assert list(smoothed.index) == list(raw.index)


def test_smooth_beta_isotonic_requires_two_valid_points() -> None:
    raw = pd.Series([np.nan, 0.5, np.nan], index=[0, 1, 2])
    with pytest.raises(ValueError, match="at least two"):
        smooth_beta_isotonic(raw)


def test_smooth_beta_isotonic_upper_none_means_unbounded() -> None:
    raw = pd.Series([0.2, 0.5, 2.0, 5.0], index=[0, 1, 2, 3])
    smoothed = smooth_beta_isotonic(raw, lower=0.0, upper=None)
    assert smoothed.max() > 1.5
    assert (smoothed.diff().dropna() >= -1e-12).all()


def test_residual_load_from_components_matches_forecast_fundamentals() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "load_forecast_day_ahead": [100.0, 200.0, 300.0, 400.0, 500.0],
            "wind_onshore_forecast": [10.0, 20.0, 30.0, 40.0, 50.0],
            "wind_offshore_forecast": [5.0, 5.0, 5.0, 5.0, 5.0],
            "solar_forecast": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
        index=idx,
    )
    fundamentals = {f.name: f.values for f in build_forecast_fundamentals(df, idx)}

    result = residual_load_from_components(
        df["load_forecast_day_ahead"],
        df["wind_onshore_forecast"],
        df["wind_offshore_forecast"],
        df["solar_forecast"],
    )
    pd.testing.assert_series_equal(
        result, fundamentals["residual_load_forecast"], check_names=False
    )


def test_renewable_share_from_components_matches_forecast_fundamentals() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "load_forecast_day_ahead": [100.0, 200.0, 300.0, 400.0, 500.0],
            "wind_onshore_forecast": [10.0, 20.0, 30.0, 40.0, 50.0],
            "wind_offshore_forecast": [5.0, 5.0, 5.0, 5.0, 5.0],
            "solar_forecast": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
        index=idx,
    )
    fundamentals = {f.name: f.values for f in build_forecast_fundamentals(df, idx)}

    result = renewable_share_from_components(
        df["load_forecast_day_ahead"],
        df["wind_onshore_forecast"],
        df["wind_offshore_forecast"],
        df["solar_forecast"],
    )
    pd.testing.assert_series_equal(
        result, fundamentals["renewable_share_forecast"], check_names=False
    )


def test_rl_relative_divides_by_load() -> None:
    rl = pd.Series([1000.0, -500.0])
    load = pd.Series([2000.0, 50000.0])
    result = rl_relative(rl, load)
    expected = pd.Series([0.5, -0.01])
    pd.testing.assert_series_equal(result, expected)


def test_rl_bin_relative_negative_residual_load_falls_in_bin_zero() -> None:
    rel = pd.Series([-0.2, -0.0001])
    bins = rl_bin_relative(rel)
    assert (bins == 0).all()


def test_rl_bin_relative_boundary_values_go_to_upper_bin() -> None:
    rel = pd.Series([0.0, RL_BIN_EDGES_RELATIVE[2]])
    bins = rl_bin_relative(rel)
    assert bins.iloc[0] == 1
    assert bins.iloc[1] == 2


def test_rl_bin_relative_label_count_matches_edge_count() -> None:
    assert len(RL_BIN_LABELS) == len(RL_BIN_EDGES_RELATIVE) - 1


# ---------------------------------------------------------------------------
# raw_beta_per_bin / estimate_beta_per_bin (Sprint 7 Step 7.5, Teil 0)
# ---------------------------------------------------------------------------

_BIN5_RL = 34000.0  # inside RL_BIN_EDGES_MW -> bin 5
_BIN7_RL = 40000.0  # inside RL_BIN_EDGES_MW -> bin 7


def _long_history_frame(
    *, n_days: int, beta_bin5: float, beta_bin7: float, is_crisis: bool = False
) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n_days * 24, freq="h", tz=LOCAL_TZ).tz_convert("UTC")
    rng = np.random.default_rng(0)
    is_bin5 = np.arange(len(idx)) % 2 == 0
    residual_load = np.where(is_bin5, _BIN5_RL, _BIN7_RL)
    ttf = rng.uniform(10, 100, size=len(idx))
    eua = np.full(len(idx), 50.0)
    srmc = srmc_gas_only(pd.Series(ttf)).to_numpy()
    beta_true = np.where(is_bin5, beta_bin5, beta_bin7)
    price = 10.0 + beta_true * srmc
    return pd.DataFrame(
        {
            "price": price,
            "residual_load": residual_load,
            "ttf_lag": ttf,
            "eua_lag": eua,
            "is_crisis": is_crisis,
        },
        index=idx,
    )


def test_raw_beta_per_bin_requires_long_history_columns() -> None:
    frame = pd.DataFrame({"price": [1.0]}, index=pd.date_range("2024-01-01", periods=1, tz="UTC"))
    with pytest.raises(ValueError, match="residual_load"):
        raw_beta_per_bin(
            frame,
            pd.Timestamp("2024-02-01", tz=LOCAL_TZ),
            srmc_definition="gas_only",
            rl_measure="absolute",
            window_days=None,
        )


def test_raw_beta_per_bin_recovers_known_beta() -> None:
    frame = _long_history_frame(n_days=10, beta_bin5=0.6, beta_bin7=0.9)
    cutoff = pd.Timestamp("2024-01-11", tz=LOCAL_TZ)  # strictly after all 10 days
    raw = raw_beta_per_bin(
        frame, cutoff, srmc_definition="gas_only", rl_measure="absolute", window_days=None
    )
    assert abs(raw.loc[5, "beta_raw"] - 0.6) < 0.05
    assert abs(raw.loc[7, "beta_raw"] - 0.9) < 0.05
    assert raw.loc[5, "n"] == 120  # 10 days * 12 bin5-hours/day


def test_raw_beta_per_bin_below_min_obs_is_nan() -> None:
    frame = _long_history_frame(n_days=1, beta_bin5=0.6, beta_bin7=0.9)  # 12h/bin < 48
    cutoff = pd.Timestamp("2024-01-02", tz=LOCAL_TZ)
    raw = raw_beta_per_bin(
        frame, cutoff, srmc_definition="gas_only", rl_measure="absolute", window_days=None
    )
    assert pd.isna(raw.loc[5, "beta_raw"])
    assert raw.loc[5, "n"] == 12


def test_raw_beta_per_bin_excludes_crisis_rows() -> None:
    frame = _long_history_frame(n_days=10, beta_bin5=0.6, beta_bin7=0.9, is_crisis=True)
    cutoff = pd.Timestamp("2024-01-11", tz=LOCAL_TZ)
    raw = raw_beta_per_bin(
        frame, cutoff, srmc_definition="gas_only", rl_measure="absolute", window_days=None
    )
    assert raw["n"].eq(0).all()


def test_raw_beta_per_bin_window_days_excludes_data_before_window_start() -> None:
    # 10 contaminated days (beta jumps by +20) immediately followed by 10
    # clean days. An expanding pool mixes both; a 10-day window should see
    # only the clean tail.
    clean = _long_history_frame(n_days=10, beta_bin5=0.6, beta_bin7=0.9)
    contaminated = _long_history_frame(n_days=10, beta_bin5=20.6, beta_bin7=20.9)
    contaminated.index = contaminated.index - pd.Timedelta(days=10)
    frame = pd.concat([contaminated, clean]).sort_index()
    cutoff = pd.Timestamp("2024-01-11", tz=LOCAL_TZ)

    full_pool = raw_beta_per_bin(
        frame, cutoff, srmc_definition="gas_only", rl_measure="absolute", window_days=None
    )
    windowed = raw_beta_per_bin(
        frame, cutoff, srmc_definition="gas_only", rl_measure="absolute", window_days=10
    )
    assert abs(full_pool.loc[5, "beta_raw"] - 0.6) > 1.0  # pulled toward the contaminated regime
    assert abs(windowed.loc[5, "beta_raw"] - 0.6) < 0.05  # window excludes it


def test_estimate_beta_per_bin_applies_isotonic_smoothing_and_clamp() -> None:
    frame = _long_history_frame(n_days=10, beta_bin5=0.6, beta_bin7=3.0)
    cutoff = pd.Timestamp("2024-01-11", tz=LOCAL_TZ)
    smoothed = estimate_beta_per_bin(
        frame,
        cutoff,
        srmc_definition="gas_only",
        rl_measure="absolute",
        window_days=None,
        clamp_max=BETA_CLAMP_MAX,
    )
    assert smoothed.loc[7] <= BETA_CLAMP_MAX
    assert (smoothed.diff().dropna() >= -1e-12).all()
