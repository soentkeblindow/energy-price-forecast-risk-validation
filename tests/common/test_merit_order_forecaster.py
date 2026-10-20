import numpy as np
import pandas as pd
import pytest

from energy_price_forecast.common.features.merit_order import srmc_gas_only
from energy_price_forecast.common.market_time import LOCAL_TZ
from energy_price_forecast.common.models.lgbm import LGBMForecaster
from energy_price_forecast.common.models.merit_order_forecaster import MeritOrderForecaster

# Two RL values landing in two different frozen bins (see
# RL_BIN_EDGES_MW), so the isotonic smoother has >= 2 points to fit on.
_RL_BIN_5 = 34000.0  # inside [32979.196, 36063.885) -> bin 5
_RL_BIN_7 = 40000.0  # inside [39024.152, 42378.031) -> bin 7


def _synthetic_long_history(
    *, n_days: int, beta_bin5: float, beta_bin7: float, switch_day: int | None = None
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Hourly long-history frame with a KNOWN, noiseless beta(RL) relationship.

    If ``switch_day`` is given, beta jumps to (beta_bin5 + 20, beta_bin7 + 20)
    from that local day onward -- a deliberately huge, easy-to-detect
    contamination signal for the leakage test.
    """
    idx = pd.date_range("2019-01-01", periods=n_days * 24, freq="h", tz=LOCAL_TZ).tz_convert("UTC")
    local_day = pd.DatetimeIndex(idx).tz_convert(LOCAL_TZ).normalize()
    day_number = (local_day - local_day[0]).days

    rng = np.random.default_rng(0)
    is_bin5 = np.arange(len(idx)) % 2 == 0
    residual_load = np.where(is_bin5, _RL_BIN_5, _RL_BIN_7)
    ttf = rng.uniform(10, 100, size=len(idx))
    eua = np.full(len(idx), 50.0)
    srmc = srmc_gas_only(pd.Series(ttf)).to_numpy()

    beta5 = np.full(len(idx), beta_bin5)
    beta7 = np.full(len(idx), beta_bin7)
    if switch_day is not None:
        switched = day_number >= switch_day
        beta5 = np.where(switched, beta_bin5 + 20.0, beta_bin5)
        beta7 = np.where(switched, beta_bin7 + 20.0, beta_bin7)
    beta_true = np.where(is_bin5, beta5, beta7)

    price = 10.0 + beta_true * srmc
    frame = pd.DataFrame(
        {
            "price": price,
            "residual_load": residual_load,
            "ttf_lag": ttf,
            "eua_lag": eua,
            "is_crisis": False,
        },
        index=idx,
    )
    return frame, idx


def _fold_x_y(
    frame: pd.DataFrame, idx: pd.DatetimeIndex, *, start_day: int, end_day: int
) -> tuple[pd.Series, pd.DataFrame]:
    local_day = pd.DatetimeIndex(idx).tz_convert(LOCAL_TZ).normalize()
    day_number = (local_day - local_day[0]).days
    mask = (day_number >= start_day) & (day_number < end_day)
    x_train = pd.DataFrame(
        {
            "residual_load_forecast": frame.loc[mask, "residual_load"],
            "ttf_gas_lag_48h": frame.loc[mask, "ttf_lag"],
            "eua_co2_lag_48h": frame.loc[mask, "eua_lag"],
        },
        index=idx[mask],
    )
    y_train = frame.loc[mask, "price"]
    return y_train, x_train


def test_beta_estimation_uses_only_data_before_fold_day() -> None:
    # Data before day 200 has beta(bin5)=0.6; from day 200 it jumps to 20.6.
    # The fold's own train window (days 190-199) sits entirely in the "before"
    # regime, so D = day 200. If post-D data leaked into the beta pool, the
    # recovered beta would be pulled sharply toward the +20 regime.
    frame, idx = _synthetic_long_history(n_days=300, beta_bin5=0.6, beta_bin7=0.9, switch_day=200)
    y_train, x_train = _fold_x_y(frame, idx, start_day=190, end_day=200)

    model = MeritOrderForecaster(frame, srmc_definition="gas_only", beta_update_cadence_days=30)
    model.fit(y_train, x_train)

    beta_bin5 = model._beta_per_bin.loc[5]  # type: ignore[union-attr]
    assert abs(beta_bin5 - 0.6) < 0.05
    assert abs(beta_bin5 - 20.6) > 1.0


def test_beta_is_cached_between_cadence_updates() -> None:
    frame, idx = _synthetic_long_history(n_days=300, beta_bin5=0.6, beta_bin7=0.9)
    model = MeritOrderForecaster(frame, srmc_definition="gas_only", beta_update_cadence_days=30)

    y1, x1 = _fold_x_y(frame, idx, start_day=190, end_day=200)
    model.fit(y1, x1)
    beta_after_first_fit = model._beta_per_bin

    # Only 5 days later: still within the 30-day cadence, so beta must be
    # reused verbatim (same object), not recomputed.
    y2, x2 = _fold_x_y(frame, idx, start_day=195, end_day=205)
    model.fit(y2, x2)
    assert model._beta_per_bin is beta_after_first_fit


_LOAD = 50000.0
_REL_BIN_5 = 0.65  # inside [0.6195223..., 0.6766913...) -> bin 5
_REL_BIN_7 = 0.75  # inside [0.7284595..., 0.7776308...) -> bin 7


def _synthetic_long_history_relative(
    *, n_days: int, beta_bin5: float, beta_bin7: float
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Like ``_synthetic_long_history``, but RL/load lands in two different
    RELATIVE-RL bins (see RL_BIN_EDGES_RELATIVE) at a fixed load level, for
    testing ``rl_measure="relative"``.
    """
    idx = pd.date_range("2019-01-01", periods=n_days * 24, freq="h", tz=LOCAL_TZ).tz_convert("UTC")
    rng = np.random.default_rng(0)
    is_bin5 = np.arange(len(idx)) % 2 == 0
    residual_load = np.where(is_bin5, _REL_BIN_5, _REL_BIN_7) * _LOAD
    ttf = rng.uniform(10, 100, size=len(idx))
    eua = np.full(len(idx), 50.0)
    srmc = srmc_gas_only(pd.Series(ttf)).to_numpy()

    beta_true = np.where(is_bin5, beta_bin5, beta_bin7)
    price = 10.0 + beta_true * srmc
    frame = pd.DataFrame(
        {
            "price": price,
            "residual_load": residual_load,
            "load": _LOAD,
            "ttf_lag": ttf,
            "eua_lag": eua,
            "is_crisis": False,
        },
        index=idx,
    )
    return frame, idx


def _fold_x_y_relative(
    frame: pd.DataFrame, idx: pd.DatetimeIndex, *, start_day: int, end_day: int
) -> tuple[pd.Series, pd.DataFrame]:
    local_day = pd.DatetimeIndex(idx).tz_convert(LOCAL_TZ).normalize()
    day_number = (local_day - local_day[0]).days
    mask = (day_number >= start_day) & (day_number < end_day)
    x_train = pd.DataFrame(
        {
            "residual_load_forecast": frame.loc[mask, "residual_load"],
            "load_forecast_day_ahead": frame.loc[mask, "load"],
            "ttf_gas_lag_48h": frame.loc[mask, "ttf_lag"],
            "eua_co2_lag_48h": frame.loc[mask, "eua_lag"],
        },
        index=idx[mask],
    )
    y_train = frame.loc[mask, "price"]
    return y_train, x_train


def test_rl_measure_relative_recovers_beta_per_relative_bin() -> None:
    frame, idx = _synthetic_long_history_relative(n_days=300, beta_bin5=0.6, beta_bin7=0.9)
    y_train, x_train = _fold_x_y_relative(frame, idx, start_day=190, end_day=200)

    model = MeritOrderForecaster(
        frame, srmc_definition="gas_only", rl_measure="relative", beta_update_cadence_days=30
    )
    model.fit(y_train, x_train)

    assert abs(model._beta_per_bin.loc[5] - 0.6) < 0.05  # type: ignore[union-attr]
    assert abs(model._beta_per_bin.loc[7] - 0.9) < 0.05  # type: ignore[union-attr]


def test_rl_measure_relative_requires_load_column() -> None:
    frame, idx = _synthetic_long_history(n_days=10, beta_bin5=0.6, beta_bin7=0.9)
    with pytest.raises(ValueError, match="load"):
        MeritOrderForecaster(frame, rl_measure="relative")


def test_b_extrapolates_past_training_ttf_max_while_a_clamps() -> None:
    """Spec section 9, "Transformationstest": for a synthetic dataset with a
    known beta (price = 1.0*srmc in the top RL bin, 0 in the bottom bin), B's
    reconstructed prediction continues the linear price/SRMC relationship
    past the training window's own TTF maximum, while plain LightGBM (A)
    clamps to its top training leaf -- this is the entire premise behind
    pulling the gas pass-through out of the tree as a target transform.
    """
    rl_bottom = -1000.0  # bin 0 (RL < 0), beta = 0.0
    rl_top = 60000.0  # bin len(RL_BIN_LABELS) - 1 (highest decile+), beta = 1.0
    idx = pd.date_range("2019-01-01", periods=300 * 24, freq="h", tz=LOCAL_TZ).tz_convert("UTC")
    rng = np.random.default_rng(0)
    is_top = np.arange(len(idx)) % 2 == 0
    residual_load = np.where(is_top, rl_top, rl_bottom)
    ttf_train = rng.uniform(10.0, 100.0, size=len(idx))
    eua = np.full(len(idx), 50.0)
    srmc_train = srmc_gas_only(pd.Series(ttf_train)).to_numpy()
    price = np.where(is_top, srmc_train, 0.0)

    frame = pd.DataFrame(
        {
            "price": price,
            "residual_load": residual_load,
            "load": 50000.0,
            "ttf_lag": ttf_train,
            "eua_lag": eua,
            "is_crisis": False,
        },
        index=idx,
    )
    x_train = pd.DataFrame(
        {
            "residual_load_forecast": residual_load,
            "ttf_gas_lag_48h": ttf_train,
            "eua_co2_lag_48h": eua,
        },
        index=idx,
    )
    y_train = frame["price"]

    model_a = LGBMForecaster(objective="quantile", alpha=0.5)
    model_a.fit(y_train, x_train)
    model_b = MeritOrderForecaster(frame, srmc_definition="gas_only", beta_update_cadence_days=30)
    model_b.fit(y_train, x_train)

    ttf_test = 300.0  # 3x the training TTF maximum (100)
    test_index = pd.DatetimeIndex([idx[-1] + pd.Timedelta(hours=1)])
    x_test = pd.DataFrame(
        {
            "residual_load_forecast": [rl_top],
            "ttf_gas_lag_48h": [ttf_test],
            "eua_co2_lag_48h": [50.0],
        },
        index=test_index,
    )
    expected_price = float(srmc_gas_only(pd.Series([ttf_test])).iloc[0])  # beta=1.0

    pred_a = float(model_a.predict(test_index, history=y_train, x_test=x_test).iloc[0])
    pred_b = float(model_b.predict(test_index, history=y_train, x_test=x_test).iloc[0])

    # A clamps: a tree's leaves are constants fit on the training range, so
    # it cannot predict anywhere near the true (far larger) extrapolated
    # price -- it stays bounded by roughly the training window's own price
    # range (max price there is srmc_gas_only(100) ~ 203.5).
    train_max_price = float(price[is_top].max())
    assert pred_a < train_max_price * 1.1
    assert pred_a < expected_price * 0.5

    # B continues the linear relationship via the beta*srmc side channel:
    # close to the true extrapolated price, not clamped near the training max.
    assert abs(pred_b - expected_price) < expected_price * 0.15


def test_crisis_rows_excluded_from_beta_estimation() -> None:
    frame, idx = _synthetic_long_history(n_days=300, beta_bin5=0.6, beta_bin7=0.9)
    # Mark everything before day 199 as crisis, leaving only one clean day
    # (199) in the beta pool for fold day D=200. One day is only 12 hours per
    # bin -- below _MIN_OBS_PER_BIN -- so if crisis rows are correctly
    # excluded, both bins stay NaN and smooth_beta_isotonic must raise. If
    # crisis exclusion were silently skipped, 199 days of crisis data would
    # still be available and the fit would succeed instead.
    local_day = pd.DatetimeIndex(idx).tz_convert(LOCAL_TZ).normalize()
    day_number = (local_day - local_day[0]).days
    frame["is_crisis"] = day_number < 199

    y_train, x_train = _fold_x_y(frame, idx, start_day=190, end_day=200)
    model = MeritOrderForecaster(frame, srmc_definition="gas_only", beta_update_cadence_days=30)

    with pytest.raises(ValueError, match="at least two"):
        model.fit(y_train, x_train)
