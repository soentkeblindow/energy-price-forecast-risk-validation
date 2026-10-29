from __future__ import annotations

import numpy as np
import pandas as pd

from energy_price_forecast.common.features.core_gas_columns import L1_COLUMNS
from energy_price_forecast.common.market_time import LOCAL_TZ
from energy_price_forecast.common.models.lgbm import LGBMForecaster
from energy_price_forecast.common.models.merit_order_forecaster import MeritOrderForecaster
from energy_price_forecast.common.models.price_model import (
    HOURLY_MODEL_V1,
    HOURLY_MODEL_V2,
    build_price_model,
    feature_columns,
)


def test_hourly_model_v1_matches_sprint7_step7_6_section_5_0() -> None:
    # Written out, not derived from the same constants HOURLY_MODEL_V1 itself
    # uses -- an independent comparison catches an accidental edit to
    # HOURLY_MODEL_V1 that a shared-constant comparison would miss.
    assert HOURLY_MODEL_V1.model_version == "hourly_model_v1"
    assert HOURLY_MODEL_V1.use_transform is False
    assert HOURLY_MODEL_V1.beta_window_days is None
    assert HOURLY_MODEL_V1.objective == "quantile"
    assert HOURLY_MODEL_V1.alpha == 0.5
    assert HOURLY_MODEL_V1.train_span_days == 90
    assert HOURLY_MODEL_V1.refit_every == 1
    assert HOURLY_MODEL_V1.random_state == 0
    assert HOURLY_MODEL_V1.n_jobs == 1
    assert list(HOURLY_MODEL_V1.feature_columns) == list(L1_COLUMNS)


def test_hourly_model_v2_matches_sprint7_step7_6_section_5_0() -> None:
    assert HOURLY_MODEL_V2.model_version == "hourly_model_v2"
    assert HOURLY_MODEL_V2.use_transform is True
    assert HOURLY_MODEL_V2.beta_window_days == 3 * 365
    assert HOURLY_MODEL_V2.objective == "quantile"
    assert HOURLY_MODEL_V2.alpha == 0.5
    assert HOURLY_MODEL_V2.train_span_days == 90
    assert HOURLY_MODEL_V2.refit_every == 1
    assert HOURLY_MODEL_V2.random_state == 0
    assert HOURLY_MODEL_V2.n_jobs == 1
    assert list(HOURLY_MODEL_V2.feature_columns) == [*L1_COLUMNS, "eua_co2_lag_48h"]


def test_feature_columns_hourly_model_v1_is_core_gas() -> None:
    assert feature_columns(HOURLY_MODEL_V1) == list(L1_COLUMNS)


def test_feature_columns_hourly_model_v2_is_core_gas_plus_eua() -> None:
    assert feature_columns(HOURLY_MODEL_V2) == [*L1_COLUMNS, "eua_co2_lag_48h"]


def _long_history() -> pd.DataFrame:
    idx = pd.date_range("2019-01-01", periods=10 * 24, freq="h", tz=LOCAL_TZ).tz_convert("UTC")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "price": rng.uniform(10, 100, size=len(idx)),
            "residual_load": rng.uniform(0, 50000, size=len(idx)),
            "ttf_lag": rng.uniform(10, 100, size=len(idx)),
            "eua_lag": rng.uniform(20, 80, size=len(idx)),
            "is_crisis": False,
        },
        index=idx,
    )


def test_build_price_model_hourly_model_v1_is_plain_lgbm() -> None:
    model = build_price_model(HOURLY_MODEL_V1, _long_history())
    assert isinstance(model, LGBMForecaster)


def test_build_price_model_hourly_model_v2_is_merit_order_forecaster_with_rolling_window() -> None:
    model = build_price_model(HOURLY_MODEL_V2, _long_history())
    assert isinstance(model, MeritOrderForecaster)
    assert model._beta_window_days == 3 * 365  # type: ignore[attr-defined]
