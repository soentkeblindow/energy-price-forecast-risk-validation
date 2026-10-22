"""French (FR) day-ahead price features for the hourly model (Sprint 7 Step
7.5 Hauptteil, docs/sprint7_step7_5_spec.md section 2).

F1/F2 are built with the exact same availability.lag/rolling_mean calls as
the existing German price_lag_24h/price_roll_mean_168h
(common/features/lags.py::build_price_lags), just on the FR raw column
instead. F3 is the plain spread between F1 and the existing
price_lag_24h -- the German lag's own Feature is passed in rather than
rebuilt, so F3 always matches whatever lag convention that feature already
uses.
"""

from __future__ import annotations

import pandas as pd

from .availability import Feature, combine, lag, rolling_mean

FR_PRICE_COLUMN = "fr_day_ahead_price"


def build_fr_price_features(
    fr_price: pd.Series,
    de_price_lag_24h: Feature,
    target_index: pd.DatetimeIndex,
) -> list[Feature]:
    """F1 (fr_price_lag_24h), F2 (fr_price_roll_mean_168h), F3
    (spread_fr_de_lag_24h = F1 - de_price_lag_24h).
    """
    f1 = lag(
        "fr_price_lag_24h",
        fr_price,
        FR_PRICE_COLUMN,
        hours=24,
        target_index=target_index,
    )
    f2 = rolling_mean(
        "fr_price_roll_mean_168h",
        fr_price,
        FR_PRICE_COLUMN,
        window_hours=168,
        lag_hours=24,
        target_index=target_index,
    )
    f3 = combine("spread_fr_de_lag_24h", [f1, de_price_lag_24h], lambda fr, de: fr - de)
    return [f1, f2, f3]
