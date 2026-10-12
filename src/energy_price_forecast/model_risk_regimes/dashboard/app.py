"""Backtest Explorer -- thin Streamlit presentation layer (Sprint 5.3).

Reads ONLY the Sprint 5.1 snapshot contract under `outputs/results/` (default,
overridable via the `DASHBOARD_RESULTS_DIR` env var or a CLI argument) -- no
`data/`, no ENTSO-E/MLflow calls, no model runs at runtime. All data logic
lives in `loading.py`/`metrics.py`/`plots.py` as pure, tested functions; this
module only wires them to Streamlit widgets (load -> compute -> `st.*`
render) and is intentionally not covered by tests (spec decision 2).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from energy_price_forecast.model_risk_regimes.dashboard.loading import (
    EXPORT_SCRIPT,
    SummaryTables,
    load_snapshot,
    load_summary_tables,
)
from energy_price_forecast.model_risk_regimes.dashboard.metrics import (
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
from energy_price_forecast.model_risk_regimes.reporting.assets import (
    plot_coverage_forest,
    plot_reliability,
)

DEFAULT_RESULTS_DIR = Path("outputs/results")

# Same Dunkelflaute window as the README fan-chart -- contains the single
# highest-spread day of the test period, so the demo opens on its most
# legible phase (owner-confirmed, spec sec. 12).
DEFAULT_WINDOW_START = "2024-12-05"
DEFAULT_WINDOW_END = "2024-12-19"

REGIME_LABELS: dict[str, str] = {
    "renewable_scarcity": "Dunkelflaute (renewable scarcity)",
    "high_wind": "High wind",
    "negative_price": "Negative price",
    "price_spike": "Price spike",
    "normal": "Normal",
}


def _resolve_results_dir() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    env = os.environ.get("DASHBOARD_RESULTS_DIR")
    return Path(env) if env else DEFAULT_RESULTS_DIR


@st.cache_data
def _load_snapshot_cached(results_dir: Path) -> pd.DataFrame:
    return load_snapshot(results_dir)


@st.cache_data
def _load_summary_tables_cached(results_dir: Path) -> SummaryTables:
    return load_summary_tables(results_dir)


def _style_regime_row(row: pd.Series, *, selected: str) -> list[str]:
    style = "color: #9e9e9e; font-style: italic;" if bool(row["low_support"]) else ""
    if row["regime"] == selected:
        style += " background-color: #e8f0fe; font-weight: bold;"
    return [style] * len(row)


def _render_forecast_tab(snapshot: pd.DataFrame) -> None:
    idx_min, idx_max = snapshot.index.min(), snapshot.index.max()
    default_start = max(pd.Timestamp(DEFAULT_WINDOW_START, tz="UTC"), idx_min).date()
    default_end = min(pd.Timestamp(DEFAULT_WINDOW_END, tz="UTC"), idx_max).date()

    # Two single-value pickers instead of one range-tuple `st.date_input`: the
    # range-tuple variant ships a "quick select" dropdown (Past week/month/...)
    # that ignores min_value/max_value and can raise a spurious end-date
    # validation error (streamlit/streamlit#12293, #11939 -- unresolved upstream).
    date_col1, date_col2 = st.columns(2)
    start = date_col1.date_input(
        "Start date", value=default_start, min_value=idx_min.date(), max_value=idx_max.date()
    )
    end = date_col2.date_input(
        "End date", value=default_end, min_value=idx_min.date(), max_value=idx_max.date()
    )
    if start > end:
        start, end = end, start
    show_raw = st.toggle("Show raw interval", value=False)

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    window = snapshot.loc[(snapshot.index >= start_ts) & (snapshot.index < end_ts)]

    # The figure always carries the FULL test period so plotly's zoom/pan can
    # reveal data beyond the selected window; only the initial view is set to
    # that window (a pre-filtered `window` would show nothing past its edges).
    fig = forecast_actual_figure(snapshot, show_raw=show_raw, initial_range=(start_ts, end_ts))
    st.plotly_chart(fig, width="stretch")

    calibrated_rate = interval_breach_rate(
        window["price_actual"], window["lo_calibrated"], window["hi_calibrated"]
    )
    raw_rate = interval_breach_rate(window["price_actual"], window["lo_raw"], window["hi_raw"])
    col1, col2 = st.columns(2)
    col1.metric("Calibrated breach rate (selected date range)", f"{calibrated_rate:.2%}")
    col2.metric("Raw breach rate (selected date range)", f"{raw_rate:.2%}")


def _render_regime_tab(snapshot: pd.DataFrame, model_comparison_by_regime: pd.DataFrame) -> None:
    table = regime_metrics(snapshot, model_comparison_by_regime)
    macro_categories = list(snapshot[MACRO_REGIME_COLUMN].cat.categories)
    options = ["overall", *REGIME_FLAG_COLUMNS, *macro_categories]
    labels = {opt: REGIME_LABELS.get(opt, str(opt).replace("_", " ").title()) for opt in options}
    selected = st.selectbox("Regime", options, format_func=lambda opt: labels[opt])
    selected_row = table.loc[table["regime"] == selected]

    st.subheader("LightGBM regime breakdown")
    st.caption(
        "n/MAE/RMSE/WAPE from model_comparison_by_regime.csv; calibrated breach rate "
        "computed live from the snapshot. The selected regime's row is highlighted below."
    )
    st.dataframe(table.style.apply(_style_regime_row, selected=selected, axis=1))
    if not selected_row.empty:
        n_hours = int(selected_row.iloc[0]["n"])
        st.caption(f"Selected: {labels[selected]} -- n = {n_hours} hours")

    st.subheader(f"Point-accuracy comparison -- regime: {labels[selected]}")
    st.dataframe(filter_by_regime(model_comparison_by_regime, selected))

    st.pyplot(regime_mae_bar(table))


def _render_risk_tab(tables: SummaryTables) -> None:
    st.subheader("Reliability diagram")
    curves = reliability_curves(tables.coverage_summary)
    # `st.pyplot` defaults to `width="stretch"`, blowing up this deliberately
    # compact 5.5x5.5in figure to the full (wide-layout) column width; "content"
    # renders it at its native, actually-readable size instead.
    st.pyplot(plot_reliability(curves), width="content")

    st.subheader("ES/VaR traffic light")
    lit = traffic_light(tables.risk_headline)
    # "raw" has no bootstrap CI (status "n/a") and is only a counterfactual --
    # dropped from this table so it doesn't crowd the two variants that
    # actually go into a status decision (calibrated, fhs).
    lit = lit.loc[lit["variant"] != "raw"]
    st.dataframe(lit)
    st.caption(
        "Status is green when the nominal 0.05 level lies inside the bootstrap CI "
        "[breach_rate_ci_low, breach_rate_ci_high]; the Kupiec p-value is shown for "
        "reference only -- it over-rejects on the hourly series and does not decide status."
    )

    st.subheader("Conditional coverage")
    coverage = tables.backtest_coverage
    validated = coverage[coverage["variant"].isin(["calibrated", "fhs"])]
    n_total = len(validated)
    n_breaking = int(validated["breach_rate_ci_excludes_alpha"].astype(bool).sum())
    st.metric("Conditional cells breaking coverage", f"{n_breaking} of {n_total}")
    st.pyplot(plot_coverage_forest(coverage, alpha=0.05))
    st.caption(
        "Unconditional coverage holds; conditional (regime-sliced) coverage breaks -- "
        "that gap is the project's central finding."
    )


def main() -> None:
    st.set_page_config(page_title="Day-Ahead Price Forecast -- Backtest Explorer", layout="wide")
    st.title("Day-Ahead Price Forecast -- Backtest Explorer")

    results_dir = _resolve_results_dir()
    try:
        snapshot = _load_snapshot_cached(results_dir)
        tables = _load_summary_tables_cached(results_dir)
    except (FileNotFoundError, ValueError) as exc:
        st.error(
            f"{exc}\n\nResults snapshot not found or invalid -- "
            f"run `uv run python {EXPORT_SCRIPT}` first."
        )
        return

    tab1, tab2, tab3 = st.tabs(["Forecast vs. Actual", "Regime", "Calibration & Risk"])
    with tab1:
        _render_forecast_tab(snapshot)
    with tab2:
        _render_regime_tab(snapshot, tables.model_comparison_by_regime)
    with tab3:
        _render_risk_tab(tables)


if __name__ == "__main__":
    main()
