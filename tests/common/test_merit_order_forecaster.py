import numpy as np
import pandas as pd
import pytest

from energy_price_forecast.common.features.merit_order import (
    BETA_CLAMP_MAX,
    estimate_beta_per_bin,
    srmc_gas_full,
    srmc_gas_only,
)
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


def test_calendar_month_rule_excludes_data_between_anchor_and_fold_day() -> None:
    # Switch happens on day 185 (2019-07-05) -- strictly after anchor(D)=day
    # 181 (2019-07-01) but before D=day 200 (2019-07-20). The calendar_month
    # pool cutoff is anchor(D), so this switched data must not enter the
    # pool -- unlike a day-D cutoff (the pre-Teil-0 behaviour), which would
    # include it and visibly pull the estimate away from the clean value.
    frame, idx = _synthetic_long_history(n_days=250, beta_bin5=0.6, beta_bin7=0.9, switch_day=185)
    y_train, x_train = _fold_x_y(frame, idx, start_day=190, end_day=200)
    day_d_local = pd.DatetimeIndex(y_train.index).tz_convert(
        LOCAL_TZ
    ).normalize().max() + pd.DateOffset(days=1)

    model = MeritOrderForecaster(frame, srmc_definition="gas_only")
    model.fit(y_train, x_train)
    beta_with_anchor_cutoff = model._beta_per_bin.loc[5]  # type: ignore[union-attr]

    beta_with_day_d_cutoff = estimate_beta_per_bin(
        frame,
        day_d_local,
        srmc_definition="gas_only",
        rl_measure="absolute",
        window_days=None,
        clamp_max=BETA_CLAMP_MAX,
    ).loc[5]

    assert abs(beta_with_anchor_cutoff - 0.6) < 0.05
    assert abs(beta_with_day_d_cutoff - 0.6) > 0.5


def _run_sequential_folds(
    model: MeritOrderForecaster,
    frame: pd.DataFrame,
    idx: pd.DatetimeIndex,
    fold_days: list[int],
    *,
    train_span_days: int,
) -> dict[int, float]:
    """Drive ``model`` through a sequence of folds (fit on the preceding
    ``train_span_days`` days, predict for day ``d``), mimicking
    ``run_backtest``'s per-fold loop with ``refit_every=1``. Returns the
    predicted price at hour 0 of each fold day, keyed by day number.
    """
    preds = {}
    for d in fold_days:
        y_train, x_train = _fold_x_y(frame, idx, start_day=d - train_span_days, end_day=d)
        model.fit(y_train, x_train)
        _, x_test = _fold_x_y(frame, idx, start_day=d, end_day=d + 1)
        pred = model.predict(x_test.index, history=y_train, x_test=x_test)
        preds[d] = float(pred.iloc[0])
    return preds


def test_calendar_month_rule_is_reproducible_across_different_run_start_days() -> None:
    # Two independently driven fold sequences on the same data: one starting
    # on day 31 (2019-02-01, the month's own 1st), the other starting on day
    # 47 (2019-02-17, 16 days later in the same month) -- different
    # run-start days. They share fold days 100 (2019-04-11) and 130
    # (2019-05-11), each in a later month the "late" run never visited the
    # start of. Per Teil 0, beta for a shared day depends only on
    # anchor(day), not on either run's own history, so both must end up
    # bit-identical on every shared day (same seed, n_jobs=1 -- the
    # MeritOrderForecaster/LGBMForecaster defaults).
    frame, idx = _synthetic_long_history(n_days=150, beta_bin5=0.6, beta_bin7=0.9)

    model_early = MeritOrderForecaster(frame, srmc_definition="gas_only")
    model_late = MeritOrderForecaster(frame, srmc_definition="gas_only")

    preds_early = _run_sequential_folds(model_early, frame, idx, [31, 100, 130], train_span_days=20)
    preds_late = _run_sequential_folds(model_late, frame, idx, [47, 100, 130], train_span_days=20)

    for d in (100, 130):
        assert preds_early[d] == preds_late[d]
    pd.testing.assert_series_equal(model_early._beta_per_bin, model_late._beta_per_bin)  # type: ignore[arg-type]


def test_gas_only_works_on_a_feature_matrix_without_eua_column() -> None:
    # Regression test (7.3 Teil 1): L1/L2/Oracle feature matrices (core_gas-
    # derived) never carry eua_co2_lag_48h at all, by design -- gas_only
    # must not require it.
    frame, idx = _synthetic_long_history(n_days=300, beta_bin5=0.6, beta_bin7=0.9)
    y_train, x_train = _fold_x_y(frame, idx, start_day=190, end_day=200)
    x_train_no_eua = x_train.drop(columns=["eua_co2_lag_48h"])

    model = MeritOrderForecaster(frame, srmc_definition="gas_only", beta_update_cadence_days=30)
    model.fit(y_train, x_train_no_eua)
    preds = model.predict(x_train_no_eua.index, history=y_train, x_test=x_train_no_eua)
    assert preds.notna().all()


def test_beta_clamp_max_override_removes_ceiling() -> None:
    # bin7's true beta (3.0) is well above the default 1.5 ceiling.
    frame, idx = _synthetic_long_history(n_days=300, beta_bin5=0.6, beta_bin7=3.0)
    y_train, x_train = _fold_x_y(frame, idx, start_day=190, end_day=200)

    default_model = MeritOrderForecaster(frame, srmc_definition="gas_only")
    default_model.fit(y_train, x_train)
    assert default_model._beta_per_bin.loc[7] <= 1.5  # type: ignore[union-attr]

    unclamped_model = MeritOrderForecaster(frame, srmc_definition="gas_only", beta_clamp_max=None)
    unclamped_model.fit(y_train, x_train)
    assert abs(unclamped_model._beta_per_bin.loc[7] - 3.0) < 0.1  # type: ignore[union-attr]


def test_beta_is_cached_between_cadence_updates() -> None:
    # beta_update_cadence_days only applies to the legacy rolling_30d rule --
    # the default calendar_month rule's own "cached between updates" path is
    # covered by test_calendar_month_rule_is_reproducible_across_different_run_start_days.
    frame, idx = _synthetic_long_history(n_days=300, beta_bin5=0.6, beta_bin7=0.9)
    model = MeritOrderForecaster(
        frame,
        srmc_definition="gas_only",
        beta_update_rule="rolling_30d",
        beta_update_cadence_days=30,
    )

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


def test_gas_full_srmc_reads_eua_from_feature_matrix_and_fits() -> None:
    # Sprint 7 Step 7.6, Teil B (B-full): srmc_definition="gas_full" was only
    # ever exercised at the pure-function level (test_merit_order.py) before
    # this test -- never through MeritOrderForecaster.fit/.predict, which is
    # what actually reads eua_co2_lag_48h off the x matrix passed in at fold
    # time, not off long_history. eua varies independently of ttf (not a
    # constant, unlike the shared helper's fixture) so a correct fit must
    # reproduce the true beta via the full SRMC formula, not just the
    # gas-only one.
    n_days = 300
    idx = pd.date_range("2019-01-01", periods=n_days * 24, freq="h", tz=LOCAL_TZ).tz_convert("UTC")
    rng = np.random.default_rng(0)
    is_bin5 = np.arange(len(idx)) % 2 == 0
    residual_load = np.where(is_bin5, _RL_BIN_5, _RL_BIN_7)
    ttf = rng.uniform(10, 100, size=len(idx))
    eua = rng.uniform(20, 80, size=len(idx))
    beta_true = np.where(is_bin5, 0.6, 0.9)
    srmc = srmc_gas_full(pd.Series(ttf), pd.Series(eua)).to_numpy()
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
    local_day = pd.DatetimeIndex(idx).tz_convert(LOCAL_TZ).normalize()
    day_number = (local_day - local_day[0]).days
    mask = (day_number >= 190) & (day_number < 200)
    x_train = pd.DataFrame(
        {
            "residual_load_forecast": frame.loc[mask, "residual_load"],
            "ttf_gas_lag_48h": frame.loc[mask, "ttf_lag"],
            "eua_co2_lag_48h": frame.loc[mask, "eua_lag"],
        },
        index=idx[mask],
    )
    y_train = frame.loc[mask, "price"]

    model = MeritOrderForecaster(frame, srmc_definition="gas_full", beta_update_cadence_days=30)
    model.fit(y_train, x_train)
    preds = model.predict(x_train.index, history=y_train, x_test=x_train)

    assert preds.notna().all()
    assert abs(model._beta_per_bin.loc[5] - 0.6) < 0.05  # type: ignore[union-attr]
    assert abs(model._beta_per_bin.loc[7] - 0.9) < 0.05  # type: ignore[union-attr]


def test_gas_full_requires_eua_column_in_feature_matrix() -> None:
    # Inverse of test_gas_only_works_on_a_feature_matrix_without_eua_column:
    # gas_full's SRMC needs eua_co2_lag_48h from the matrix passed to
    # fit/predict, so dropping it must fail loudly, not silently fall back
    # to gas-only behaviour.
    frame, idx = _synthetic_long_history(n_days=300, beta_bin5=0.6, beta_bin7=0.9)
    y_train, x_train = _fold_x_y(frame, idx, start_day=190, end_day=200)
    x_train_no_eua = x_train.drop(columns=["eua_co2_lag_48h"])

    model = MeritOrderForecaster(frame, srmc_definition="gas_full", beta_update_cadence_days=30)
    with pytest.raises(KeyError, match="eua_co2_lag_48h"):
        model.fit(y_train, x_train_no_eua)


def test_beta_window_days_excludes_data_before_window_through_forecaster() -> None:
    # Sprint 7 Step 7.6, Teil B (B-roll): beta_window_days's exclusion is only
    # ever exercised directly on raw_beta_per_bin (test_merit_order.py), never
    # through the MeritOrderForecaster class B-roll actually instantiates.
    # Contaminated beta (+20) for days < 150, clean beta from day 150 onward;
    # a fold at day 300 with a 30-day window sits entirely in the clean
    # regime, while the default expanding pool mixes both regimes.
    n_days = 300
    idx = pd.date_range("2019-01-01", periods=n_days * 24, freq="h", tz=LOCAL_TZ).tz_convert("UTC")
    local_day = pd.DatetimeIndex(idx).tz_convert(LOCAL_TZ).normalize()
    day_number = (local_day - local_day[0]).days
    rng = np.random.default_rng(0)
    is_bin5 = np.arange(len(idx)) % 2 == 0
    residual_load = np.where(is_bin5, _RL_BIN_5, _RL_BIN_7)
    ttf = rng.uniform(10, 100, size=len(idx))
    eua = np.full(len(idx), 50.0)
    srmc = srmc_gas_only(pd.Series(ttf)).to_numpy()

    contaminated = day_number < 150
    beta5 = np.where(contaminated, 20.6, 0.6)
    beta7 = np.where(contaminated, 20.9, 0.9)
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
    y_train, x_train = _fold_x_y(frame, idx, start_day=280, end_day=300)

    windowed = MeritOrderForecaster(
        frame, srmc_definition="gas_only", beta_update_rule="rolling_30d", beta_window_days=30
    )
    windowed.fit(y_train, x_train)
    expanding = MeritOrderForecaster(
        frame, srmc_definition="gas_only", beta_update_rule="rolling_30d", beta_window_days=None
    )
    expanding.fit(y_train, x_train)

    # The expanding pool's raw (pre-clamp) estimate is pulled far above the
    # clean value by the contaminated half; the isotonic smoother clamps it
    # to BETA_CLAMP_MAX (1.5) rather than leaving it at the raw blend, so the
    # assertion compares against that clamped value, not an unbounded one.
    assert abs(windowed._beta_per_bin.loc[5] - 0.6) < 0.05  # type: ignore[union-attr]
    assert abs(expanding._beta_per_bin.loc[5] - 0.6) > 0.5  # type: ignore[union-attr]


def test_crisis_rows_excluded_from_beta_estimation() -> None:
    frame, idx = _synthetic_long_history(n_days=300, beta_bin5=0.6, beta_bin7=0.9)
    # Mark everything before day 199 as crisis. Fold day D=200 anchors to day
    # 181 (2019-07-01, calendar_month rule), so the pool (days < 181) is
    # entirely inside the crisis region and ends up empty after crisis rows
    # are excluded -- both bins stay NaN and smooth_beta_isotonic must raise.
    # If crisis exclusion were silently skipped, 181 days of crisis data
    # would still be available (well above _MIN_OBS_PER_BIN) and the fit
    # would succeed instead.
    local_day = pd.DatetimeIndex(idx).tz_convert(LOCAL_TZ).normalize()
    day_number = (local_day - local_day[0]).days
    frame["is_crisis"] = day_number < 199

    y_train, x_train = _fold_x_y(frame, idx, start_day=190, end_day=200)
    model = MeritOrderForecaster(frame, srmc_definition="gas_only", beta_update_cadence_days=30)

    with pytest.raises(ValueError, match="at least two"):
        model.fit(y_train, x_train)
