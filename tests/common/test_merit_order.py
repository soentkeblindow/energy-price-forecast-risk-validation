import numpy as np
import pandas as pd
import pytest

from energy_price_forecast.common.features.merit_order import (
    CO2_INTENSITY_EL,
    GAS_EFFICIENCY,
    RL_BIN_EDGES_MW,
    RL_BIN_EDGES_RELATIVE,
    RL_BIN_LABELS,
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
