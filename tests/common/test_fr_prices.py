"""Wiring tests for common/data/fr_prices.py (Sprint 7 Step 7.5 Hauptteil,
Phase 0). The VWAP math itself is already covered generically in
test_normalize.py -- this only checks that build_fr_day_ahead_hourly wires
FR's own price/load/load-forecast series into to_hourly_vwap correctly, via
mocked fetches (no live ENTSO-E call).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from energy_price_forecast.common.data import fr_prices


def _hourly_utc(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="h", tz="UTC")


def test_build_fr_day_ahead_hourly_wires_load_actual_as_weight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    idx = _hourly_utc("2024-01-01 00:00", 4)
    price = pd.DataFrame({"day_ahead_price": [100.0, 200.0, 300.0, 400.0]}, index=idx)
    load = pd.DataFrame(
        {"load_actual": [1.0, 3.0, 0.0, float("nan")], "load_forecast_day_ahead": [1.0] * 4},
        index=idx,
    )
    monkeypatch.setattr(fr_prices, "fetch_day_ahead_prices", lambda *a, **k: price)
    monkeypatch.setattr(fr_prices, "fetch_load", lambda *a, **k: load)

    out_path = tmp_path / "fr_day_ahead_hourly.parquet"
    result = fr_prices.build_fr_day_ahead_hourly(idx[0], idx[-1], path=out_path)

    assert list(result.columns) == ["fr_day_ahead_price"]
    assert out_path.exists()
    # Row 0: weight 1 -> 100. Row 2 (weight 0) and row 3 (weight NaN, fallback
    # 1.0) both end up fallback-driven -- the exact VWAP/fallback arithmetic
    # is to_hourly_vwap's own concern (test_normalize.py); only the wiring
    # (price in, load_actual as weight, load_forecast_day_ahead as fallback)
    # is asserted here.
    assert result["fr_day_ahead_price"].iloc[0] == 100.0
    assert not result["fr_day_ahead_price"].isna().any()


def test_load_fr_day_ahead_hourly_roundtrips(tmp_path: Path) -> None:
    idx = _hourly_utc("2024-01-01 00:00", 3)
    frame = pd.DataFrame({"fr_day_ahead_price": [1.0, 2.0, 3.0]}, index=idx)
    path = tmp_path / "fr_day_ahead_hourly.parquet"
    frame.to_parquet(path)

    loaded = fr_prices.load_fr_day_ahead_hourly(path)
    pd.testing.assert_frame_equal(loaded, frame, check_freq=False)


def test_load_fr_day_ahead_hourly_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="fr_prices.build_fr_day_ahead_hourly"):
        fr_prices.load_fr_day_ahead_hourly(tmp_path / "does_not_exist.parquet")
