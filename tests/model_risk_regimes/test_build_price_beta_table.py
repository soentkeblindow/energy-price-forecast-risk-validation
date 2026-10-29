"""Unit tests for scripts/model_risk_regimes/build_price_beta_table.py."""

import numpy as np
import pandas as pd

from energy_price_forecast.common.market_time import LOCAL_TZ
from scripts.model_risk_regimes.build_price_beta_table import (
    build_hourly_model_v2_beta_table,
    merge_beta_table,
)


def _long_history(n_days: int) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n_days * 24, freq="h", tz=LOCAL_TZ).tz_convert("UTC")
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


def test_build_hourly_model_v2_beta_table_one_row_per_anchor_and_bin() -> None:
    long_history = _long_history(n_days=400)
    anchors = [pd.Timestamp("2023-06-01", tz=LOCAL_TZ), pd.Timestamp("2023-07-01", tz=LOCAL_TZ)]
    table = build_hourly_model_v2_beta_table(long_history, anchors)

    assert len(table) == 2 * 11
    assert set(table["anchor_month"]) == {"2023-06", "2023-07"}
    assert set(table["model_version"]) == {"hourly_model_v2"}
    assert list(table.columns) == [
        "anchor_month",
        "bin",
        "bin_lower_mw",
        "bin_upper_mw",
        "n",
        "beta_raw",
        "beta_smoothed",
        "clamp_flag",
        "model_version",
    ]


def test_merge_beta_table_appends_new_anchor_without_touching_existing() -> None:
    existing = pd.DataFrame(
        {"anchor_month": ["2023-05", "2023-05"], "bin": [0, 1], "beta_smoothed": [0.1, 0.2]}
    )
    new_rows = pd.DataFrame(
        {"anchor_month": ["2023-06", "2023-06"], "bin": [0, 1], "beta_smoothed": [0.3, 0.4]}
    )
    merged = merge_beta_table(existing, new_rows)

    assert len(merged) == 4
    assert set(merged["anchor_month"]) == {"2023-05", "2023-06"}
    existing_rows = merged.loc[merged["anchor_month"] == "2023-05"].sort_values("bin")
    pd.testing.assert_series_equal(
        existing_rows["beta_smoothed"].reset_index(drop=True),
        pd.Series([0.1, 0.2], name="beta_smoothed"),
    )


def test_merge_beta_table_reappending_same_month_does_not_duplicate() -> None:
    existing = pd.DataFrame(
        {"anchor_month": ["2023-05", "2023-05"], "bin": [0, 1], "beta_smoothed": [0.1, 0.2]}
    )
    updated_rows = pd.DataFrame(
        {"anchor_month": ["2023-05", "2023-05"], "bin": [0, 1], "beta_smoothed": [0.9, 0.9]}
    )
    merged = merge_beta_table(existing, updated_rows)

    assert len(merged) == 2
    assert (merged["beta_smoothed"] == 0.9).all()
