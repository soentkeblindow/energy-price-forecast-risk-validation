"""Smoke tests to confirm the project skeleton is importable."""

import energy_price_forecast


def test_package_imports() -> None:
    """The main package should be importable without errors."""
    assert energy_price_forecast is not None


def test_subpackages_importable() -> None:
    """The common and model_risk_regimes top-level packages should be importable."""
    from energy_price_forecast import common, model_risk_regimes

    assert common is not None
    assert model_risk_regimes is not None
