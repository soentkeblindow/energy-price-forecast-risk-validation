"""Variant B (Phase 2) of
docs/merit_order_extrapolation_structure_measurement_spec.md: a target-
transform wrapper around LGBMForecaster that pulls a residual-load-dependent
gas pass-through (beta(RL)) out of the tree before fitting, and adds it back
at predict time.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from ..features.merit_order import (
    BETA_CLAMP_MAX,
    RlMeasure,
    SrmcDefinition,
    estimate_beta_per_bin,
    rl_bin,
    rl_bin_relative,
    srmc_gas_full,
    srmc_gas_only,
)
from ..market_time import LOCAL_TZ, month_start_local
from .lgbm import LGBMForecaster


class MeritOrderForecaster:
    """Variant B: LGBMForecaster wrapped with a residual-load-dependent gas
    pass-through target transform (spec section 5.3).

    beta(RL) is estimated from a long-history side channel the caller
    supplies at construction time (columns ``price``/``residual_load``/
    ``ttf_lag``/``eua_lag``/``is_crisis``, plus ``load`` when
    ``rl_measure="relative"``, see
    ``common.features.merit_order.build_merit_order_frame``) -- NOT from the
    rolling 90-day ``x_train``/``y_train`` the walk-forward harness passes to
    ``fit``, which is too short and too recent to identify beta (spec section
    1 background: the gas-price swing needed to identify beta only shows up
    over long history).

    ``rl_measure`` selects which RL binning Phase 1's frozen bins use:
    ``"absolute"`` (default, fixed-MW ``RL_BIN_EDGES_MW``) or ``"relative"``
    (``RL_BIN_EDGES_RELATIVE``, RL as a fraction of load -- the B-rel
    sensitivity, spec section 5.3, meant to stay comparable across years as
    renewables build-out shifts the absolute RL distribution).

    ``beta_clamp_max`` overrides the isotonic smoother's upper bound (default
    1.5, same as Phase 2); pass ``None`` to remove the ceiling entirely (7.3
    section 3.4's control run -- monotonicity and the floor at 0 are
    untouched, only the ceiling is affected).

    ``beta_update_rule`` picks how beta is re-estimated (Sprint 7 Step 7.5,
    Teil 0):

    - ``"calendar_month"`` (default): the beta used for delivery day D is
      estimated from long-history rows strictly before ``anchor(D)``, the
      local midnight of the 1st of D's own calendar month -- *not* before D
      itself. Re-estimated whenever ``anchor(D)`` changes from the last
      anchor used. This makes beta a pure function of D's calendar month,
      independent of which day a given run happens to start on (two runs
      starting on different days of the same month produce bit-identical
      beta and predictions for every day they share -- the reproducibility
      this was introduced for; see
      ``tests/common/test_merit_order_forecaster.py``).
    - ``"rolling_30d"``: legacy mode, kept only so 7.2/7.3's already-reported
      numbers stay explainable. Re-estimated at most once every
      ``beta_update_cadence_days`` (default 30) counted from whichever fold
      happened to be first in a given run, with the pool cut off at D
      itself. **Not reproducible** across runs with different start dates
      (the exact problem Teil 0 fixes) -- do not use for anything new.

    Crisis-period rows (``is_crisis``) are always excluded from the beta
    estimation pool under either rule -- decided before Phase 2 results were
    seen (Phase 1 gave no reason to override this with a weighted-inclusion
    scheme), see docs/merit_order_extrapolation_structure_measurement_log.md.

    The actual per-bin regression + isotonic smoothing lives in
    ``common.features.merit_order.estimate_beta_per_bin``, a pure function
    this class calls with the cutoff each rule computes -- also used
    directly (bypassing this class) to build a monthly beta reference table.
    """

    def __init__(
        self,
        long_history: pd.DataFrame,
        *,
        srmc_definition: SrmcDefinition = "gas_only",
        rl_measure: RlMeasure = "absolute",
        beta_update_rule: Literal["calendar_month", "rolling_30d"] = "calendar_month",
        beta_update_cadence_days: int = 30,  # only used by beta_update_rule="rolling_30d"
        beta_window_days: int | None = None,  # None = expanding; else rolling N days
        beta_clamp_max: float | None = BETA_CLAMP_MAX,
        objective: Literal["quantile", "l2"] = "quantile",
        alpha: float | None = None,
        params: dict[str, Any] | None = None,
        random_state: int = 0,
        n_jobs: int = 1,
    ) -> None:
        required = {"price", "residual_load", "ttf_lag", "eua_lag", "is_crisis"}
        if rl_measure == "relative":
            required = required | {"load"}
        missing = required - set(long_history.columns)
        if missing:
            raise ValueError(f"long_history is missing columns: {sorted(missing)}")
        self._long_history = long_history
        self._srmc_definition = srmc_definition
        self._rl_measure = rl_measure
        self._beta_update_rule = beta_update_rule
        self._beta_update_cadence_days = beta_update_cadence_days
        self._beta_window_days = beta_window_days
        self._beta_clamp_max = beta_clamp_max
        self._inner = LGBMForecaster(
            objective=objective,
            alpha=alpha,
            params=params,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self._beta_per_bin: pd.Series | None = None
        self._last_beta_update: pd.Timestamp | None = (
            None  # rolling_30d: day_d_local of last update
        )
        self._last_anchor_used: pd.Timestamp | None = None  # calendar_month: anchor of last update

    def _srmc(self, ttf: pd.Series, eua: pd.Series | None) -> pd.Series:
        if self._srmc_definition == "gas_only":
            return srmc_gas_only(ttf)
        assert eua is not None  # gas_full always has eua available, see callers below
        return srmc_gas_full(ttf, eua)

    def _bin_and_srmc(self, x: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        if self._rl_measure == "relative":
            bin_ = rl_bin_relative(x["residual_load_forecast"] / x["load_forecast_day_ahead"])
        else:
            bin_ = rl_bin(x["residual_load_forecast"])
        # eua_co2_lag_48h is only read for gas_full: x may be a reduced
        # feature matrix (core_gas-derived rungs, 7.3) that never carries
        # EUA at all, by design.
        eua = x["eua_co2_lag_48h"] if self._srmc_definition == "gas_full" else None
        srmc = self._srmc(x["ttf_gas_lag_48h"], eua)
        return bin_, srmc

    def _beta_lookup(self, bin_: pd.Series) -> pd.Series:
        if self._beta_per_bin is None:
            raise RuntimeError("beta_per_bin not yet estimated")
        # .cat.codes maps NaN -> -1 and every real bin to its own label value
        # (labels are the contiguous ints 0..len(RL_BIN_LABELS)-1 in category
        # order), so a plain int-keyed .map needs no float/int coercion and
        # -1 safely misses every real key, yielding NaN for an unbinned row.
        codes = bin_.cat.codes
        return codes.map(self._beta_per_bin)

    def _maybe_update_beta(self, day_d_local: pd.Timestamp) -> None:
        if self._beta_update_rule == "calendar_month":
            anchor = month_start_local(day_d_local)
            if self._last_anchor_used is not None and anchor == self._last_anchor_used:
                return
            cutoff_local = anchor
            self._last_anchor_used = anchor
        else:
            due = (
                self._last_beta_update is None
                or (day_d_local - self._last_beta_update).days >= self._beta_update_cadence_days
            )
            if not due:
                return
            cutoff_local = day_d_local
            self._last_beta_update = day_d_local

        self._beta_per_bin = estimate_beta_per_bin(
            self._long_history,
            cutoff_local,
            srmc_definition=self._srmc_definition,
            rl_measure=self._rl_measure,
            window_days=self._beta_window_days,
            clamp_max=self._beta_clamp_max,
        )

    def fit(self, y_train: pd.Series, x_train: pd.DataFrame | None = None) -> None:
        if x_train is None:
            raise ValueError("MeritOrderForecaster requires a feature matrix (x_train)")
        # train_index covers delivery days < D (Fold contract), so D's local
        # midnight is the calendar day after the latest day in y_train.
        last_train_day_local = (
            pd.DatetimeIndex(y_train.index).tz_convert(LOCAL_TZ).normalize().max()
        )
        day_d_local = last_train_day_local + pd.DateOffset(days=1)
        self._maybe_update_beta(day_d_local)

        bin_train, srmc_train = self._bin_and_srmc(x_train)
        beta_train = self._beta_lookup(bin_train)
        y_prime = y_train - beta_train.to_numpy() * srmc_train.to_numpy()
        self._inner.fit(y_prime, x_train)

    def predict(
        self,
        test_index: pd.DatetimeIndex,
        *,
        history: pd.Series,
        x_test: pd.DataFrame | None = None,
    ) -> pd.Series:
        if x_test is None:
            raise ValueError("MeritOrderForecaster requires a feature matrix (x_test)")
        bin_test, srmc_test = self._bin_and_srmc(x_test)
        beta_test = self._beta_lookup(bin_test)
        tree_pred = self._inner.predict(test_index, history=history, x_test=x_test)
        y_hat = tree_pred.to_numpy() + beta_test.to_numpy() * srmc_test.to_numpy()
        return pd.Series(y_hat, index=test_index, name="y_pred")
