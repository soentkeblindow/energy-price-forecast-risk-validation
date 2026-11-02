import sys

import pytest

from scripts.model_risk_regimes.run_price_model import _parse_args


def _run_parse_args(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["run_price_model.py", *argv])
    _parse_args()


def test_include_extra_features_unknown_group_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        _run_parse_args(monkeypatch, ["--include-extra-features", "bogus", "--tag", "t"])
    assert "unknown group" in capsys.readouterr().err


def test_include_extra_features_without_tag_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        _run_parse_args(monkeypatch, ["--include-extra-features", "temperature"])
    assert "--tag is required" in capsys.readouterr().err


def test_include_extra_features_conflicts_with_drop_features_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        _run_parse_args(
            monkeypatch,
            [
                "--include-extra-features",
                "temperature",
                "--drop-features",
                "temperature_north",
                "--tag",
                "t",
            ],
        )
    err = capsys.readouterr().err
    assert "temperature_north" in err


def test_drop_features_unknown_column_still_errors_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: --include-extra-features's new validation must not change
    # --drop-features' own (unaffected) behaviour. The unknown-column check
    # itself only runs in main() against the real feature matrices, not in
    # _parse_args -- here we only confirm --tag gating still fires for a
    # plain --drop-features override, unaffected by the new flag's presence.
    with pytest.raises(SystemExit):
        _run_parse_args(monkeypatch, ["--drop-features", "ttf_gas_lag_48h"])


def test_include_extra_features_and_drop_features_together_is_fine_when_disjoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_parse_args(
        monkeypatch,
        [
            "--include-extra-features",
            "temperature",
            "--drop-features",
            "ttf_gas_lag_48h",
            "--tag",
            "t",
        ],
    )
