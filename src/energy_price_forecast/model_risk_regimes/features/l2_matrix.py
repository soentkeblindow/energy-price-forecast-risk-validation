"""L2 feature-matrix construction for the error-decomposition ladder
(docs/residual_error_decomposition_hourly_spec.md, section 3.3) -- extracted
from scripts/model_risk_regimes/run_error_decomposition_ladder.py so a second
driver script (Sprint 7 Step 7.6, Teil A) can reuse the exact same L2
construction without duplicating it.
"""

from __future__ import annotations

import pandas as pd

from energy_price_forecast.common.features.merit_order import (
    renewable_share_from_components,
    residual_load_from_components,
)


def build_l2_matrix(
    l1_matrix: pd.DataFrame, renewables_predictions: pd.DataFrame, target_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """L1 with its 5 NWP-mapped fundamentals columns (not load) replaced by
    the renewables reconstruction (spec section 3.3). ``renewables_predictions``
    is run_renewables_backtest's (run_init_utc, valid_time_utc)-indexed
    output; each valid_time appears exactly once (one walk-forward fold per
    day), so flattening to valid_time_utc is unambiguous.
    """
    flat = renewables_predictions.set_index(
        renewables_predictions.index.get_level_values("valid_time_utc")
    )
    won = flat["wind_onshore_mw_pred"].reindex(target_index)
    woff = flat["wind_offshore_mw_pred"].reindex(target_index)
    solar = flat["solar_mw_pred"].reindex(target_index)
    load = l1_matrix["load_forecast_day_ahead"]
    return l1_matrix.assign(
        wind_onshore_forecast=won,
        wind_offshore_forecast=woff,
        solar_forecast=solar,
        residual_load_forecast=residual_load_from_components(load, won, woff, solar),
        renewable_share_forecast=renewable_share_from_components(load, won, woff, solar),
    )
