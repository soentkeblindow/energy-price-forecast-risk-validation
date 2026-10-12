"""Dashboard data-loading layer (Sprint 5.3).

Pure I/O + schema validation for the Streamlit Backtest Explorer: reads the
Sprint 5.1 snapshot contract from `outputs/results/` and nothing else -- no
`data/`, no ENTSO-E/MLflow calls, no model runs. No Streamlit import here;
caching is the app layer's job (`app.py`), not this module's, so these
functions stay testable without a Streamlit runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..evaluation.regimes import MACRO_REGIME_COLUMN, REGIME_FLAG_COLUMNS
from ..reporting.snapshot import SNAPSHOT_COLUMNS, SNAPSHOT_FILENAME
from ..reporting.tables import (
    BACKTEST_COVERAGE_FILENAME,
    COVERAGE_SUMMARY_FILENAME,
    MODEL_COMPARISON_BY_REGIME_FILENAME,
    MODEL_COMPARISON_FILENAME,
    RISK_HEADLINE_FILENAME,
)

EXPORT_SCRIPT = "scripts/export_report_assets.py"

_FLOAT_COLUMNS = (
    "price_actual",
    "forecast_median",
    "lo_raw",
    "hi_raw",
    "lo_calibrated",
    "hi_calibrated",
)


@dataclass(frozen=True)
class SummaryTables:
    """The summary CSV artefacts from the Sprint 5.1/5.3.1 report-assets export."""

    model_comparison: pd.DataFrame
    model_comparison_by_regime: pd.DataFrame
    coverage_summary: pd.DataFrame
    backtest_coverage: pd.DataFrame
    risk_headline: pd.DataFrame


def load_snapshot(results_dir: Path) -> pd.DataFrame:
    """Load and validate `predictions_snapshot.parquet`.

    Casts the on-disk float16 price/quantile columns to float64 before
    returning -- the storage-only float16 rounding (Sprint 5.1 gotcha) is
    fine for display but should not degrade metric computations.

    Raises `FileNotFoundError` if the file is missing, `ValueError` if it is
    missing one of `SNAPSHOT_COLUMNS`.
    """
    path = results_dir / SNAPSHOT_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `uv run python {EXPORT_SCRIPT}` first.")
    frame = pd.read_parquet(path)

    missing = [col for col in SNAPSHOT_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(
            f"{path} is missing expected column(s) {missing}; "
            f"expected {list(SNAPSHOT_COLUMNS)}, found {list(frame.columns)}. "
            f"Re-run `uv run python {EXPORT_SCRIPT}` if the snapshot contract changed."
        )

    frame = frame[list(SNAPSHOT_COLUMNS)].copy()
    for col in _FLOAT_COLUMNS:
        frame[col] = frame[col].astype("float64")
    for col in REGIME_FLAG_COLUMNS:
        frame[col] = frame[col].astype(bool)
    frame[MACRO_REGIME_COLUMN] = frame[MACRO_REGIME_COLUMN].astype("category")
    return frame


def load_summary_tables(results_dir: Path) -> SummaryTables:
    """Load the summary CSVs, collecting *all* missing-file errors at once."""
    filenames = {
        "model_comparison": MODEL_COMPARISON_FILENAME,
        "model_comparison_by_regime": MODEL_COMPARISON_BY_REGIME_FILENAME,
        "coverage_summary": COVERAGE_SUMMARY_FILENAME,
        "backtest_coverage": BACKTEST_COVERAGE_FILENAME,
        "risk_headline": RISK_HEADLINE_FILENAME,
    }
    paths = {name: results_dir / filename for name, filename in filenames.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing result file(s): {missing} -- run `uv run python {EXPORT_SCRIPT}` first."
        )
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    return SummaryTables(**frames)
