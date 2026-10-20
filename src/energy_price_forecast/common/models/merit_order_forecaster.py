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
    RL_BIN_LABELS,
    rl_bin,
    rl_bin_relative,
    smooth_beta_isotonic,
    srmc_gas_full,
    srmc_gas_only,
)
from ..market_time import LOCAL_TZ
from .lgbm import LGBMForecaster

SrmcDefinition = Literal["gas_only", "gas_full"]
RlMeasure = Literal["absolute", "relative"]

# Same pragmatic floor as the Phase 1 structural measurement (not HAC-derived
# here -- this is a point-estimate-only closed-form slope, not a
# standard-error report -- but kept at the same value so a bin only starts
# contributing a beta once it has a comparable amount of evidence).
_MIN_OBS_PER_BIN = 48


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

    Re-estimated at most once every ``beta_update_cadence_days`` (default 30,
    spec section 5.3 step 1); between updates the last smoothed
    beta-per-bin vector is reused. Crisis-period rows (``is_crisis``) are
    always excluded from the beta estimation pool -- decided before Phase 2
    results were seen (Phase 1 gave no reason to override this with a
    weighted-inclusion scheme), see
    docs/merit_order_extrapolation_structure_measurement_log.md.

    Leakage contract: only long-history rows strictly before the fold's
    delivery day D enter the beta estimation pool (``_maybe_update_beta``
    filters on ``< day D's local midnight, converted to UTC``) -- see
    ``tests/common/test_merit_order_forecaster.py``.
    """

    def __init__(
        self,
        long_history: pd.DataFrame,
        *,
        srmc_definition: SrmcDefinition = "gas_only",
        rl_measure: RlMeasure = "absolute",
        beta_update_cadence_days: int = 30,
        beta_window_days: int | None = None,  # None = expanding; else rolling N days
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
        self._beta_update_cadence_days = beta_update_cadence_days
        self._beta_window_days = beta_window_days
        self._inner = LGBMForecaster(
            objective=objective,
            alpha=alpha,
            params=params,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self._beta_per_bin: pd.Series | None = None
        self._last_beta_update: pd.Timestamp | None = None

    def _srmc(self, ttf: pd.Series, eua: pd.Series) -> pd.Series:
        if self._srmc_definition == "gas_only":
            return srmc_gas_only(ttf)
        return srmc_gas_full(ttf, eua)

    def _bin_and_srmc(self, x: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        if self._rl_measure == "relative":
            bin_ = rl_bin_relative(x["residual_load_forecast"] / x["load_forecast_day_ahead"])
        else:
            bin_ = rl_bin(x["residual_load_forecast"])
        srmc = self._srmc(x["ttf_gas_lag_48h"], x["eua_co2_lag_48h"])
        return bin_, srmc

    def _bin_pool(self, pool: pd.DataFrame) -> pd.Series:
        if self._rl_measure == "relative":
            return rl_bin_relative(pool["residual_load"] / pool["load"])
        return rl_bin(pool["residual_load"])

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
        due = (
            self._last_beta_update is None
            or (day_d_local - self._last_beta_update).days >= self._beta_update_cadence_days
        )
        if not due:
            return

        cutoff_utc = day_d_local.tz_convert("UTC")
        pool = self._long_history.loc[self._long_history.index < cutoff_utc]
        if self._beta_window_days is not None:
            window_start_local = day_d_local - pd.DateOffset(days=self._beta_window_days)
            pool = pool.loc[pool.index >= window_start_local.tz_convert("UTC")]
        pool = pool.loc[~pool["is_crisis"]]

        bin_pool = self._bin_pool(pool)
        srmc_pool = self._srmc(pool["ttf_lag"], pool["eua_lag"])
        valid = pool["price"].notna() & srmc_pool.notna() & bin_pool.notna()

        raw_beta_values: dict[int, float] = {}
        for label in RL_BIN_LABELS:
            cell_mask = valid & (bin_pool == label)
            n = int(cell_mask.sum())
            if n < _MIN_OBS_PER_BIN:
                continue
            y = pool.loc[cell_mask, "price"].to_numpy()
            x = srmc_pool.loc[cell_mask].to_numpy()
            x_centered = x - x.mean()
            denom = float((x_centered**2).sum())
            if denom <= 0:
                continue
            raw_beta_values[label] = float((x_centered * (y - y.mean())).sum() / denom)

        raw_beta = pd.Series(raw_beta_values, dtype=float).reindex(list(RL_BIN_LABELS))
        self._beta_per_bin = smooth_beta_isotonic(raw_beta)
        self._last_beta_update = day_d_local

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
