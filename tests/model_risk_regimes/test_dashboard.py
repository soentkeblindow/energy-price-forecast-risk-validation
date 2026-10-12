from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import plotly.graph_objects as go
import pytest
from matplotlib.figure import Figure

from energy_price_forecast.model_risk_regimes.dashboard.loading import (
    BACKTEST_COVERAGE_FILENAME,
    COVERAGE_SUMMARY_FILENAME,
    MODEL_COMPARISON_BY_REGIME_FILENAME,
    MODEL_COMPARISON_FILENAME,
    RISK_HEADLINE_FILENAME,
    load_snapshot,
    load_summary_tables,
)
from energy_price_forecast.model_risk_regimes.dashboard.metrics import (
    LOW_SUPPORT_MIN_HOURS,
    filter_by_regime,
    interval_breach_rate,
    regime_metrics,
    traffic_light,
)
from energy_price_forecast.model_risk_regimes.dashboard.plots import (
    forecast_actual_figure,
    regime_mae_bar,
    reliability_curves,
)
from energy_price_forecast.model_risk_regimes.evaluation.regimes import (
    MACRO_REGIME_COLUMN,
    REGIME_FLAG_COLUMNS,
)
from energy_price_forecast.model_risk_regimes.reporting.snapshot import SNAPSHOT_COLUMNS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snapshot_fixture(n: int = 10) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    data: dict[str, object] = {
        "price_actual": [50.0 + i for i in range(n)],
        "forecast_median": [50.0 + i for i in range(n)],
        "lo_raw": [40.0 + i for i in range(n)],
        "hi_raw": [60.0 + i for i in range(n)],
        "lo_calibrated": [45.0 + i for i in range(n)],
        "hi_calibrated": [55.0 + i for i in range(n)],
    }
    for col in REGIME_FLAG_COLUMNS:
        data[col] = [False] * n
    data["normal"] = [True] * n
    data[MACRO_REGIME_COLUMN] = ["calm"] * (n // 2) + ["crisis"] * (n - n // 2)
    frame = pd.DataFrame(data, index=idx)
    frame[MACRO_REGIME_COLUMN] = frame[MACRO_REGIME_COLUMN].astype("category")
    for col in REGIME_FLAG_COLUMNS:
        frame[col] = frame[col].astype(bool)
    return frame


def _write_summary_csvs(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "model": ["Naive", "LightGBM"],
            "mae": [34.77, 15.4],
            "rmse": [56.2, 26.59],
            "wape": [0.29, 0.13],
        }
    ).to_csv(tmp_path / MODEL_COMPARISON_FILENAME, index=False)
    pd.DataFrame(
        {
            "model": ["Naive", "LightGBM"],
            "regime": ["overall", "overall"],
            "axis": ["overall", "overall"],
            "n": [100, 100],
            "mae": [34.77, 15.4],
            "rmse": [56.2, 26.59],
            "wape": [0.29, 0.13],
        }
    ).to_csv(tmp_path / MODEL_COMPARISON_BY_REGIME_FILENAME, index=False)
    pd.DataFrame(
        {
            "nominal_level": [0.05, 0.5],
            "lightgbm_raw_coverage": [0.17, 0.49],
            "lightgbm_calibrated_coverage": [0.05, 0.50],
            "arimax_raw_coverage": [0.12, None],
        }
    ).to_csv(tmp_path / COVERAGE_SUMMARY_FILENAME, index=False)
    pd.DataFrame(
        {
            "variant": ["calibrated", "fhs"],
            "side": ["long", "long"],
            "subset": ["overall", "overall"],
            "breach_rate_ci_excludes_alpha": [False, True],
        }
    ).to_csv(tmp_path / BACKTEST_COVERAGE_FILENAME, index=False)
    pd.DataFrame(
        {
            "variant": ["raw", "calibrated"],
            "side": ["long", "long"],
            "breach_rate_ci_low": [None, 0.04],
            "breach_rate_ci_high": [None, 0.06],
        }
    ).to_csv(tmp_path / RISK_HEADLINE_FILENAME, index=False)


# ---------------------------------------------------------------------------
# loading.load_snapshot
# ---------------------------------------------------------------------------


def test_load_snapshot_returns_expected_columns_and_dtypes(tmp_path: Path) -> None:
    fixture = _snapshot_fixture()
    fixture.to_parquet(tmp_path / "predictions_snapshot.parquet")

    frame = load_snapshot(tmp_path)

    assert list(frame.columns) == list(SNAPSHOT_COLUMNS)
    assert frame["price_actual"].dtype == "float64"
    assert frame["normal"].dtype == bool
    assert str(frame[MACRO_REGIME_COLUMN].dtype) == "category"


def test_load_snapshot_missing_file_raises_with_export_hint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="export_report_assets.py"):
        load_snapshot(tmp_path)


def test_load_snapshot_missing_column_raises_value_error(tmp_path: Path) -> None:
    fixture = _snapshot_fixture().drop(columns=["price_spike"])
    fixture.to_parquet(tmp_path / "predictions_snapshot.parquet")

    with pytest.raises(ValueError, match="price_spike"):
        load_snapshot(tmp_path)


# ---------------------------------------------------------------------------
# loading.load_summary_tables
# ---------------------------------------------------------------------------


def test_load_summary_tables_reads_all_five(tmp_path: Path) -> None:
    _write_summary_csvs(tmp_path)

    tables = load_summary_tables(tmp_path)

    assert len(tables.model_comparison) == 2
    assert len(tables.model_comparison_by_regime) == 2
    assert len(tables.coverage_summary) == 2
    assert len(tables.backtest_coverage) == 2
    assert len(tables.risk_headline) == 2


def test_load_summary_tables_collects_all_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as exc_info:
        load_summary_tables(tmp_path)

    message = str(exc_info.value)
    for filename in (
        MODEL_COMPARISON_FILENAME,
        MODEL_COMPARISON_BY_REGIME_FILENAME,
        COVERAGE_SUMMARY_FILENAME,
        BACKTEST_COVERAGE_FILENAME,
        RISK_HEADLINE_FILENAME,
    ):
        assert filename in message


# ---------------------------------------------------------------------------
# metrics.interval_breach_rate
# ---------------------------------------------------------------------------


def test_interval_breach_rate_exact() -> None:
    actual = pd.Series([10.0, 20.0, 30.0, 40.0])
    lo = pd.Series([0.0, 0.0, 0.0, 0.0])
    hi = pd.Series([15.0, 15.0, 15.0, 15.0])

    # 3 of 4 (20, 30, 40) breach the upper bound.
    assert interval_breach_rate(actual, lo, hi) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# metrics.regime_metrics
# ---------------------------------------------------------------------------


def _by_regime_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["LightGBM"] * 5,
            "regime": ["overall", "normal", "calm", "crisis", "price_spike"],
            "axis": ["overall", "flag", "macro", "macro", "flag"],
            "n": [4, 4, 2, 2, 0],
            "mae": [15.4, 15.4, 10.0, 20.0, float("nan")],
            "rmse": [26.59, 26.59, 18.0, 30.0, float("nan")],
            "wape": [0.1285, 0.1285, 0.09, 0.15, float("nan")],
        }
    )


def test_regime_metrics_uses_csv_point_metrics_and_live_breach_rate() -> None:
    snapshot = _snapshot_fixture(n=4)
    # price_actual stays at the fixture default (50..53), well below
    # lo_calibrated (45..48)/hi_calibrated (55..58) -- so every non-empty
    # regime's calibrated breach rate is 0.0 here, computed live from the
    # snapshot regardless of the (unrelated, hand-picked) CSV point metrics.
    by_regime = _by_regime_fixture()

    table = regime_metrics(snapshot, by_regime)

    overall = table.set_index("regime").loc["overall"]
    assert overall["n"] == 4
    assert overall["mae"] == 15.4  # from the CSV fixture, not computed from the snapshot
    assert overall["calibrated_breach_rate"] == pytest.approx(0.0)

    crisis = table.set_index("regime").loc["crisis"]
    assert crisis["n"] == 2
    assert crisis["mae"] == 20.0

    # "price_spike" has n=0 in the CSV fixture -> empty regime, no crash.
    spike = table.set_index("regime").loc["price_spike"]
    assert spike["n"] == 0
    assert spike[["mae", "calibrated_breach_rate"]].isna().all()
    assert bool(spike["low_support"])

    assert set(table["regime"]) == {"overall", "normal", "calm", "crisis", "price_spike"}


def test_regime_metrics_low_support_threshold() -> None:
    snapshot = _snapshot_fixture(n=4)
    by_regime = pd.DataFrame(
        {
            "model": ["LightGBM", "LightGBM"],
            "regime": ["renewable_scarcity", "high_wind"],
            "axis": ["flag", "flag"],
            "n": [LOW_SUPPORT_MIN_HOURS - 1, LOW_SUPPORT_MIN_HOURS + 1],
            "mae": [10.0, 10.0],
            "rmse": [15.0, 15.0],
            "wape": [0.1, 0.1],
        }
    )

    table = regime_metrics(snapshot, by_regime).set_index("regime")

    assert bool(table.loc["renewable_scarcity", "low_support"])
    assert not bool(table.loc["high_wind", "low_support"])


# ---------------------------------------------------------------------------
# metrics.filter_by_regime
# ---------------------------------------------------------------------------


def test_filter_by_regime_returns_one_row_per_model() -> None:
    by_regime = pd.DataFrame(
        {
            "model": ["Naive", "LightGBM", "Naive", "LightGBM"],
            "regime": ["overall", "overall", "calm", "calm"],
            "axis": ["overall", "overall", "macro", "macro"],
            "n": [100, 100, 40, 40],
            "mae": [30.0, 15.0, 20.0, 10.0],
            "rmse": [40.0, 20.0, 25.0, 12.0],
            "wape": [0.3, 0.15, 0.2, 0.1],
        }
    )

    result = filter_by_regime(by_regime, "calm")

    assert list(result["model"]) == ["Naive", "LightGBM"]
    assert list(result["mae"]) == [20.0, 10.0]
    assert "regime" not in result.columns
    assert "axis" not in result.columns


# ---------------------------------------------------------------------------
# metrics.traffic_light
# ---------------------------------------------------------------------------


def test_traffic_light_status_by_ci_containment() -> None:
    risk_headline = pd.DataFrame(
        {
            "variant": ["calibrated", "calibrated", "raw"],
            "breach_rate_ci_low": [0.04, 0.10, None],
            "breach_rate_ci_high": [0.06, 0.20, None],
        }
    )

    result = traffic_light(risk_headline)

    assert list(result["status"]) == ["green", "red", "n/a"]


def test_traffic_light_formats_buffer_factor_for_display() -> None:
    risk_headline = pd.DataFrame(
        {
            "variant": ["calibrated", "fhs"],
            "breach_rate_ci_low": [0.04, 0.04],
            "breach_rate_ci_high": [0.06, 0.06],
            "buffer_factor": [1.529135531359624, None],
        }
    )

    result = traffic_light(risk_headline)

    assert list(result["buffer_factor"]) == ["1.53x", "n/a (calibrated only)"]


# ---------------------------------------------------------------------------
# plots.forecast_actual_figure (smoke)
# ---------------------------------------------------------------------------


def test_forecast_actual_figure_traces_and_english_text() -> None:
    window = _snapshot_fixture(n=5)

    fig_without_raw = forecast_actual_figure(window, show_raw=False)
    fig_with_raw = forecast_actual_figure(window, show_raw=True)

    assert isinstance(fig_without_raw, go.Figure)
    names_without_raw = {trace.name for trace in fig_without_raw.data}
    assert "Actual" in names_without_raw
    assert "Forecast (median)" in names_without_raw
    assert "Calibrated 90% interval" in names_without_raw
    assert "Raw 90% interval" not in names_without_raw

    names_with_raw = {trace.name for trace in fig_with_raw.data}
    assert "Raw 90% interval" in names_with_raw

    assert fig_without_raw.layout.xaxis.title.text == "Time (UTC)"
    assert fig_without_raw.layout.yaxis.title.text == "Day-ahead price (EUR/MWh)"


def test_forecast_actual_figure_plots_full_range_with_initial_zoom() -> None:
    full = _snapshot_fixture(n=100)
    zoom_start, zoom_end = full.index[10], full.index[20]

    fig = forecast_actual_figure(full, show_raw=False, initial_range=(zoom_start, zoom_end))

    actual_trace = next(trace for trace in fig.data if trace.name == "Actual")
    # The figure carries every row of `full`, not just the zoomed slice --
    # panning past the initial view must reveal real data, not empty space.
    assert len(actual_trace.x) == len(full)
    assert list(fig.layout.xaxis.range) == [zoom_start, zoom_end]


# ---------------------------------------------------------------------------
# plots.reliability_curves
# ---------------------------------------------------------------------------


def test_reliability_curves_reshapes_and_drops_nan() -> None:
    coverage_summary = pd.DataFrame(
        {
            "nominal_level": [0.05, 0.5],
            "lightgbm_raw_coverage": [0.17, 0.49],
            "lightgbm_calibrated_coverage": [0.05, 0.50],
            "arimax_raw_coverage": [0.12, None],
        }
    )

    curves = reliability_curves(coverage_summary)

    assert set(curves) == {"raw", "calibrated_sorted", "arimax"}
    assert list(curves["arimax"].index) == [0.05]
    assert curves["calibrated_sorted"]["empirical"].tolist() == [0.05, 0.50]
    for frame in curves.values():
        assert "empirical" in frame.columns


# ---------------------------------------------------------------------------
# plots.regime_mae_bar (smoke)
# ---------------------------------------------------------------------------


def test_regime_mae_bar_returns_figure() -> None:
    table = pd.DataFrame({"regime": ["overall", "high_wind"], "mae": [15.4, 12.1]})

    fig = regime_mae_bar(table)

    assert isinstance(fig, Figure)
