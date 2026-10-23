"""Named price-model configurations (Sprint 7 Step 7.6, section 5.0): the
single place every setting of ``price_v1`` and ``price_v2`` is defined. No
other script sets model parameters directly -- it calls the factory
functions below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from ..evaluation.walkforward import Forecaster
from ..features.core_gas_columns import L1_COLUMNS
from .lgbm import LGBMForecaster
from .merit_order_forecaster import MeritOrderForecaster

ModelVersion = Literal["price_v1", "price_v2"]


@dataclass(frozen=True)
class PriceModelConfig:
    """Every setting of one named price-model version.

    ``use_transform=False`` (``price_v1``): plain LightGBM, no merit-order
    target transform -- ``beta_window_days`` is unused.

    ``use_transform=True`` (``price_v2``): the merit-order target transform
    (``MeritOrderForecaster``). SRMC = gas + CO2, absolute RL bins frozen on
    ``RL_BIN_EDGES_MW``, calendar-month beta re-estimation, crisis excluded,
    beta clamped to ``[0, 1.5]`` and isotonically smoothed, minimum 48
    observations per bin -- all fixed inside ``MeritOrderForecaster``
    itself (that class now supports only this one path), not configurable
    here.
    """

    model_version: ModelVersion
    use_transform: bool
    beta_window_days: int | None  # only meaningful when use_transform
    feature_columns: tuple[str, ...]
    objective: Literal["quantile", "l2"] = "quantile"
    alpha: float | None = 0.5
    train_span_days: int = 90
    refit_every: int = 1
    lgbm_params: dict[str, Any] | None = None
    random_state: int = 0
    n_jobs: int = 1


PRICE_V1 = PriceModelConfig(
    model_version="price_v1",
    use_transform=False,
    beta_window_days=None,
    feature_columns=L1_COLUMNS,
)

PRICE_V2 = PriceModelConfig(
    model_version="price_v2",
    use_transform=True,
    beta_window_days=3 * 365,
    feature_columns=(*L1_COLUMNS, "eua_co2_lag_48h"),
)


def feature_columns(config: PriceModelConfig) -> list[str]:
    """The feature-matrix columns ``config`` needs, in order."""
    return list(config.feature_columns)


def build_price_model(config: PriceModelConfig, long_history: pd.DataFrame) -> Forecaster:
    """Build the ``Forecaster`` for ``config``.

    ``long_history`` is the beta long-history side channel
    (``common.features.merit_order.build_merit_order_frame``'s output) --
    unused for ``price_v1``.
    """
    if not config.use_transform:
        return LGBMForecaster(
            objective=config.objective,
            alpha=config.alpha,
            params=config.lgbm_params,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
        )
    return MeritOrderForecaster(
        long_history,
        beta_window_days=config.beta_window_days,
        objective=config.objective,
        alpha=config.alpha,
        params=config.lgbm_params,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )
