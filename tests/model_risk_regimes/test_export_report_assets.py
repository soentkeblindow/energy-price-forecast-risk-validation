from pathlib import Path

import pandas as pd
import pytest

from scripts.model_risk_regimes.export_report_assets import (
    _INPUTS,
    _load_csv,
    _load_parquet,
    _reliability_curve_for_plot,
    _require_inputs,
)


def test_require_inputs_collects_all_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as exc_info:
        _require_inputs(tmp_path)

    message = str(exc_info.value)
    # Every declared input is reported, not just the first.
    for filename, producer in _INPUTS.values():
        assert filename in message
        assert producer in message


def test_require_inputs_resolves_present_files(tmp_path: Path) -> None:
    for filename, _producer in _INPUTS.values():
        (tmp_path / filename).write_text("")

    resolved = _require_inputs(tmp_path)

    assert set(resolved) == set(_INPUTS)
    for name, path in resolved.items():
        assert path == tmp_path / _INPUTS[name][0]


def test_load_parquet_raises_on_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "frame.parquet"
    pd.DataFrame({"a": [1, 2]}).to_parquet(path)

    with pytest.raises(ValueError, match="missing expected columns"):
        _load_parquet(path, required_columns=("a", "b"))


def test_load_csv_raises_on_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "frame.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing expected columns"):
        _load_csv(path, required_columns=("a", "b"))


def test_reliability_curve_for_plot_filters_overall_and_renames() -> None:
    frame = pd.DataFrame(
        {
            "bucket": ["overall", "negative"],
            "level": [0.05, 0.05],
            "coverage": [0.17, 0.20],
        }
    )

    out = _reliability_curve_for_plot(frame)

    assert list(out.columns) == ["empirical"]
    assert out.index.name == "level"
    assert out.loc[0.05]["empirical"] == 0.17
