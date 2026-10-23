"""Unit tests for the data-file fingerprint helper."""

from pathlib import Path

import pandas as pd
import pytest

from energy_price_forecast.common.data.snapshot import file_fingerprint

MODULE = "energy_price_forecast.common.data.snapshot"

_INDEX = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")


def _write_parquet(path: Path, values: list[float], index: pd.DatetimeIndex = _INDEX) -> None:
    pd.DataFrame({"value": values}, index=index).to_parquet(path)


def test_fingerprint_reports_filename_range_and_row_count(tmp_path: Path) -> None:
    path = tmp_path / "data.parquet"
    _write_parquet(path, [1.0, 2.0, 3.0, 4.0])

    result = file_fingerprint(path)

    assert result["filename"] == "data.parquet"
    assert result["row_count"] == 4
    assert result["first_timestamp"] == _INDEX.min().isoformat()
    assert result["last_timestamp"] == _INDEX.max().isoformat()


def test_fingerprint_hash_is_stable_across_calls(tmp_path: Path) -> None:
    path = tmp_path / "data.parquet"
    _write_parquet(path, [1.0, 2.0, 3.0, 4.0])

    first = file_fingerprint(path)
    second = file_fingerprint(path)

    assert first["sha256"] == second["sha256"]


def test_fingerprint_hash_changes_with_content(tmp_path: Path) -> None:
    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    _write_parquet(path_a, [1.0, 2.0, 3.0, 4.0])
    _write_parquet(path_b, [1.0, 2.0, 3.0, 5.0])

    assert file_fingerprint(path_a)["sha256"] != file_fingerprint(path_b)["sha256"]


def test_fingerprint_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "missing.parquet"

    with pytest.raises(FileNotFoundError, match="missing.parquet"):
        file_fingerprint(missing)


def test_fingerprint_non_datetime_index_raises(tmp_path: Path) -> None:
    path = tmp_path / "plain.parquet"
    pd.DataFrame({"value": [1.0, 2.0, 3.0]}).to_parquet(path)

    with pytest.raises(ValueError, match="DatetimeIndex"):
        file_fingerprint(path)
