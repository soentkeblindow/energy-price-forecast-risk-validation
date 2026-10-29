"""`hourly_model_v2` (Sprint 7 Step 7.6): a target-transform wrapper around
``LGBMForecaster`` that pulls a residual-load-dependent gas pass-through
(``beta(RL)``) out of the tree before fitting, and adds it back at predict
time. SRMC = gas + CO2; see ``common.models.price_model.HOURLY_MODEL_V2`` for the
full, named configuration.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from ..features.merit_order import BETA_CLAMP_MAX, estimate_beta_per_bin, rl_bin, srmc_gas_full
from ..features.subset import drop_features
from ..market_time import LOCAL_TZ, month_start_local
from .lgbm import LGBMForecaster

# The two columns _bin_and_srmc reads SRMC from -- also the ones a
# `hide_from_tree` ablation study would want to hide (see its own docstring).
SRMC_COLUMNS: tuple[str, str] = ("ttf_gas_lag_48h", "eua_co2_lag_48h")


class MeritOrderForecaster:
    """``hourly_model_v2``: ``LGBMForecaster`` wrapped with a residual-load-dependent
    gas pass-through target transform (``price - beta(RL)*SRMC``, tree on
    the residual). SRMC is gas + CO2 (``srmc_gas_full``): the feature
    matrix passed to ``fit``/``predict`` and the ``long_history`` side
    channel must both carry ``eua_co2_lag_48h``.

    ``beta(RL)`` is estimated from a long-history side channel supplied at
    construction time (columns ``price``/``residual_load``/``ttf_lag``/
    ``eua_lag``/``is_crisis``) -- NOT from the rolling 90-day ``x_train``/
    ``y_train`` the walk-forward harness passes to ``fit``, which is too
    short and too recent to identify beta.

    Re-estimated once per calendar month: the beta used for delivery day D
    is estimated from long-history rows strictly before ``anchor(D)``, the
    local midnight of the 1st of D's own calendar month -- not before D
    itself, so beta is a pure function of D's calendar month, independent
    of which day a given run happens to start on.

    ``beta_window_days`` selects the pool: ``None`` is an expanding pool
    from all history before the anchor; an integer is a rolling N-day pool
    (``hourly_model_v2`` uses a 3-year rolling window). Crisis-period rows
    (``is_crisis``) are always excluded from the pool. The resulting
    per-bin estimates are isotonically smoothed and clamped to
    ``[0, BETA_CLAMP_MAX]`` (fixed MW bins, ``common.features.merit_order
    .RL_BIN_EDGES_MW``).

    The actual per-bin regression + isotonic smoothing lives in
    ``common.features.merit_order.estimate_beta_per_bin``, a pure function
    this class calls with the cutoff each month computes -- also used
    directly (bypassing this class) to build the monthly beta reference
    table (``scripts/model_risk_regimes/build_price_beta_table.py``).

    ``hide_from_tree``: ablation-only, defaults to ``()`` (hourly_model_v2's
    canonical, bit-identical behaviour -- the inner tree sees every column
    of ``x_train``/``x_test``, including ``SRMC_COLUMNS``). SRMC is always
    computed from ``x`` itself (``_bin_and_srmc``), never from
    ``long_history`` (that side channel only ever feeds the beta(RL)
    regression) -- so simply dropping ``SRMC_COLUMNS`` from the feature
    matrix before it reaches this class breaks SRMC entirely, it does not
    "ablate a redundant tree feature". ``hide_from_tree`` answers the
    actual question (does the tree still need its own copy of a column
    SRMC already uses?) correctly: SRMC/beta are computed from the full
    ``x`` as always, and only afterwards are these column(s) dropped from
    what reaches the inner ``LGBMForecaster``.
    """

    def __init__(
        self,
        long_history: pd.DataFrame,
        *,
        beta_window_days: int | None,
        objective: Literal["quantile", "l2"] = "quantile",
        alpha: float | None = None,
        params: dict[str, Any] | None = None,
        random_state: int = 0,
        n_jobs: int = 1,
        hide_from_tree: tuple[str, ...] = (),
    ) -> None:
        required = {"price", "residual_load", "ttf_lag", "eua_lag", "is_crisis"}
        missing = required - set(long_history.columns)
        if missing:
            raise ValueError(f"long_history is missing columns: {sorted(missing)}")
        self._long_history = long_history
        self._beta_window_days = beta_window_days
        self._hide_from_tree = hide_from_tree
        self._inner = LGBMForecaster(
            objective=objective,
            alpha=alpha,
            params=params,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self._beta_per_bin: pd.Series | None = None
        self._last_anchor_used: pd.Timestamp | None = None

    def _for_tree(self, x: pd.DataFrame) -> pd.DataFrame:
        if not self._hide_from_tree:
            return x
        return drop_features(x, list(self._hide_from_tree), strict=False)

    def _bin_and_srmc(self, x: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        bin_ = rl_bin(x["residual_load_forecast"])
        srmc = srmc_gas_full(x[SRMC_COLUMNS[0]], x[SRMC_COLUMNS[1]])
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
        anchor = month_start_local(day_d_local)
        if self._last_anchor_used is not None and anchor == self._last_anchor_used:
            return
        self._last_anchor_used = anchor
        self._beta_per_bin = estimate_beta_per_bin(
            self._long_history,
            anchor,
            srmc_definition="gas_full",
            rl_measure="absolute",
            window_days=self._beta_window_days,
            clamp_max=BETA_CLAMP_MAX,
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
        self._inner.fit(y_prime, self._for_tree(x_train))

    def predict_components(
        self,
        test_index: pd.DatetimeIndex,
        *,
        history: pd.Series,
        x_test: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Like ``predict``, but returns every intermediate quantity
        (``tree_pred``, ``beta``, ``srmc``, ``rl_bin``) alongside ``y_pred``
        -- for the per-hour parity/decomposition columns in the reference
        prediction export.
        """
        if x_test is None:
            raise ValueError("MeritOrderForecaster requires a feature matrix (x_test)")
        bin_test, srmc_test = self._bin_and_srmc(x_test)
        beta_test = self._beta_lookup(bin_test)
        tree_pred = self._inner.predict(test_index, history=history, x_test=self._for_tree(x_test))
        y_hat = tree_pred.to_numpy() + beta_test.to_numpy() * srmc_test.to_numpy()
        return pd.DataFrame(
            {
                "y_pred": y_hat,
                "tree_pred": tree_pred.to_numpy(),
                "beta": beta_test.to_numpy(),
                "srmc": srmc_test.to_numpy(),
                "rl_bin": bin_test.to_numpy(),
            },
            index=test_index,
        )

    def predict(
        self,
        test_index: pd.DatetimeIndex,
        *,
        history: pd.Series,
        x_test: pd.DataFrame | None = None,
    ) -> pd.Series:
        components = self.predict_components(test_index, history=history, x_test=x_test)
        return components["y_pred"].rename("y_pred")
