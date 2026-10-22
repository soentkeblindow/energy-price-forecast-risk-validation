import numpy as np
import pandas as pd

from energy_price_forecast.common.features.availability import assert_no_leakage, lag
from energy_price_forecast.common.features.neighbor_prices import build_fr_price_features


def _hourly_utc(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="h", tz="UTC")


# Berlin day 2024-01-15 (CET = UTC+1, 24 hours), same fixture as test_availability.py.
_TARGET_JAN15 = pd.date_range("2024-01-14 23:00", periods=24, freq="h", tz="UTC")


def _de_price_lag_24h() -> object:
    de_idx = _hourly_utc("2024-01-13 23:00", 24)
    de_price = pd.Series(np.arange(24, dtype=float) * 2.0, index=de_idx, name="day_ahead_price")
    return lag("price_lag_24h", de_price, "day_ahead_price", hours=24, target_index=_TARGET_JAN15)


def _fr_price() -> pd.Series:
    # 60 days of history, comfortably covering F2's 168h window + 24h lag
    # leading up to the Jan-15 target day.
    idx = _hourly_utc("2023-12-01 00:00", 24 * 60)
    return pd.Series(np.arange(len(idx), dtype=float), index=idx, name="fr_day_ahead_price")


def test_build_fr_price_features_names_and_leakage() -> None:
    de_lag = _de_price_lag_24h()
    feats = build_fr_price_features(_fr_price(), de_lag, _TARGET_JAN15)

    assert [f.name for f in feats] == [
        "fr_price_lag_24h",
        "fr_price_roll_mean_168h",
        "spread_fr_de_lag_24h",
    ]
    assert_no_leakage(feats)


def test_f1_is_a_plain_24h_lag_of_fr_price() -> None:
    fr_price = _fr_price()
    de_lag = _de_price_lag_24h()
    f1, _, _ = build_fr_price_features(fr_price, de_lag, _TARGET_JAN15)

    expected = fr_price.reindex(_TARGET_JAN15 - pd.Timedelta(hours=24)).to_numpy()
    np.testing.assert_array_equal(f1.values.to_numpy(), expected)


def test_f3_is_the_spread_between_f1_and_de_lag() -> None:
    de_lag = _de_price_lag_24h()
    f1, _, f3 = build_fr_price_features(_fr_price(), de_lag, _TARGET_JAN15)

    expected = (f1.values - de_lag.values).rename("spread_fr_de_lag_24h")
    pd.testing.assert_series_equal(f3.values, expected)


def test_f2_is_na_free_once_the_168h_window_is_covered() -> None:
    de_lag = _de_price_lag_24h()
    _, f2, _ = build_fr_price_features(_fr_price(), de_lag, _TARGET_JAN15)
    assert f2.values.notna().all()
