"""Unit tests for scripts/model_risk_regimes/backtest.py's own helper functions."""

from scripts.model_risk_regimes.backtest import _resolve_target_transform


def test_lasso_defaults_to_identity() -> None:
    assert _resolve_target_transform(None, "lasso") == "identity"


def test_ridge_defaults_to_asinh() -> None:
    assert _resolve_target_transform(None, "ridge") == "asinh"


def test_ols_defaults_to_asinh() -> None:
    assert _resolve_target_transform(None, "ols") == "asinh"


def test_explicit_value_always_wins() -> None:
    assert _resolve_target_transform("asinh", "lasso") == "asinh"
    assert _resolve_target_transform("identity", "ridge") == "identity"
