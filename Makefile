.PHONY: help data backtest test lint dashboard report-assets dm-test requirements price-model price-beta renewables-update

.DEFAULT_GOAL := help

help:
	@echo "Available targets:"
	@echo "  data            Build interim + feature parquet from raw data (needs data/raw/ and an ENTSO-E API key in .env -- see README 'Getting started')"
	@echo "  backtest        Run the flagship LightGBM walk-forward backtest (needs data/processed/features.parquet from 'make data'; no API key needed directly)"
	@echo "  test            Run the test suite (pytest)"
	@echo "  lint            Run ruff check, ruff format --check, and mypy (mirrors CI)"
	@echo "  dashboard       Launch the Streamlit Backtest Explorer locally"
	@echo "  report-assets   Regenerate README figures/tables from outputs/results/ (no API key needed)"
	@echo "  dm-test         Run the Diebold-Mariano significance test (no API key needed)"
	@echo "  requirements    Regenerate requirements.txt for Streamlit Community Cloud deployment"
	@echo "  price-model     Recompute the price_v1/price_v2 reference run (outputs/results/price_v2/)"
	@echo "  price-beta      Extend price_v2's monthly beta(RL) table and figure by one more anchor month"
	@echo "  renewables-update  Extend the renewables L2 feature store with fresh data (merge, not overwrite)"

data:
	uv run python scripts/model_risk_regimes/build_interim.py
	uv run python scripts/model_risk_regimes/build_features.py

backtest:
	uv run python scripts/model_risk_regimes/backtest.py --model lgbm

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src

dashboard:
	uv run --extra dashboard streamlit run src/energy_price_forecast/model_risk_regimes/dashboard/app.py

report-assets:
	uv run python scripts/model_risk_regimes/export_report_assets.py

dm-test:
	uv run python scripts/model_risk_regimes/run_dm_test.py

requirements:
	uv export --format requirements.txt --extra dashboard --no-dev --no-hashes -o src/energy_price_forecast/model_risk_regimes/dashboard/requirements.txt

price-model:
	uv run python scripts/model_risk_regimes/run_price_model.py --versions price_v1,price_v2 --rungs L1,L2 --out-dir outputs/results/price_v2

price-beta:
	uv run python scripts/model_risk_regimes/build_price_beta_table.py

renewables-update:
	uv run python scripts/model_renewables/run_renewables_model.py --store-new
