"""Dashboard figure construction (Sprint 5.3).

`forecast_actual_figure` is the one genuinely new chart (interactive plotly,
View 1). `reliability_curves` and `regime_mae_bar` are thin adapters that
reshape the exported `outputs/results/` CSVs into the shapes the Sprint 5.1
matplotlib functions (`reporting/assets.py`) or a small local helper expect --
`reporting/` itself is not touched. No Streamlit import anywhere in this
module; `app.py` is the only caller of `st.*`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from matplotlib.figure import Figure

from ..reporting.assets import VARIANT_COLORS

# `plot_reliability` (reporting/assets.py) keys its internal color/marker styling
# off "calibrated_sorted", not "calibrated" -- an unmatched key silently falls
# back to matplotlib's default color cycle instead of raising, so this mapping
# must stay exactly in sync with that dict.
_RELIABILITY_COLUMN_LABELS = {
    "lightgbm_raw_coverage": "raw",
    "lightgbm_calibrated_coverage": "calibrated_sorted",
    "arimax_raw_coverage": "arimax",
}


def forecast_actual_figure(
    window: pd.DataFrame,
    *,
    show_raw: bool,
    initial_range: tuple[object, object] | None = None,
) -> go.Figure:
    """Interactive forecast-vs-actual chart for View 1.

    `window` must carry `price_actual`, `forecast_median`, `lo_raw`, `hi_raw`,
    `lo_calibrated`, `hi_calibrated` on a `DatetimeIndex`. Quantile crossing
    (`lo > hi`) is drawn as-is, not repaired (documented Sprint 3/4 finding).

    `window` should normally span the FULL test period, not just the
    currently selected date range -- plotly's zoom/pan only ever reveals data
    that is actually in the figure, so passing a pre-filtered window makes
    panning past its edges show nothing. `initial_range` sets the chart's
    starting x-axis view (e.g. the default Dunkelflaute window) without
    limiting what is plotted; the user can then zoom/pan out to the rest of
    `window`.
    """
    x = window.index
    x_band = list(x) + list(x[::-1])

    fig = go.Figure()
    if show_raw:
        fig.add_trace(
            go.Scatter(
                x=x_band,
                y=list(window["hi_raw"]) + list(window["lo_raw"][::-1]),
                fill="toself",
                fillcolor="rgba(158, 158, 158, 0.25)",
                line=dict(color="rgba(0, 0, 0, 0)"),
                name="Raw 90% interval",
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=x_band,
            y=list(window["hi_calibrated"]) + list(window["lo_calibrated"][::-1]),
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.25)",
            line=dict(color="rgba(0, 0, 0, 0)"),
            name="Calibrated 90% interval",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=window["price_actual"],
            mode="lines",
            name="Actual",
            line=dict(color="black", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=window["forecast_median"],
            mode="lines",
            name="Forecast (median)",
            line=dict(color=VARIANT_COLORS["calibrated"], width=1.5),
        )
    )
    fig.update_layout(
        title="Forecast vs. actual",
        xaxis_title="Time (UTC)",
        yaxis_title="Day-ahead price (EUR/MWh)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    if initial_range is not None:
        fig.update_xaxes(range=list(initial_range))
    return fig


def reliability_curves(coverage_summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Reshape the wide `coverage_summary.csv` into the per-curve, level-indexed
    long format `reporting.assets.plot_reliability` expects (one frame per
    label, indexed by `level`, with an `empirical` column). Drops NaN levels
    per curve (e.g. ARIMAX is only populated at a subset of nominal levels) and
    skips a curve entirely if its column is absent or empty.
    """
    curves: dict[str, pd.DataFrame] = {}
    for column, label in _RELIABILITY_COLUMN_LABELS.items():
        if column not in coverage_summary.columns:
            continue
        series = coverage_summary.set_index("nominal_level")[column].dropna()
        if series.empty:
            continue
        frame = series.rename("empirical").to_frame()
        frame.index.name = "level"
        curves[label] = frame
    return curves


def regime_mae_bar(regime_table: pd.DataFrame) -> Figure:
    """Small matplotlib bar chart, one bar per regime, from `metrics.regime_metrics`'s
    output. Optional per spec -- View 2's tables carry the finding on their own.
    """
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(regime_table["regime"], regime_table["mae"], color=VARIANT_COLORS["calibrated"])
    ax.set_ylabel("MAE (EUR/MWh)")
    ax.set_title("LightGBM MAE by regime")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig
