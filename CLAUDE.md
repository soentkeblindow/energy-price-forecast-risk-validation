# energy-price-forecast-risk-validation — project notes for Claude Code

Public research repo for probabilistic day-ahead electricity price forecasting (DE-LU), with conformal calibration, VaR/ES backtesting, regime analysis, and a Streamlit dashboard.

## Relationship to the live sibling repo

This repo was copied (Sprint 6) into `sbl-energy-forecast`, which became the live production pipeline for the KIT Energy Arena (daily automated submission, Cloudflare-triggered GitHub Actions, its own data store and fallback ladder). Two new models were built there against real operating constraints; Sprint 7 Step 7.0 brought both back here as `model_renewables` and `model_post_crisis_resolution`, restructured so research work (backtests, model comparisons, diagnostics) happens in this repo going forward, while the sibling repo stays pure live operation.

**Never write to the sibling repo from here**, not even by accident through a path. Read-only, if ever needed.

## Project structure and dependency rule

```
src/energy_price_forecast/
  common/                       # shared: config, market_time, data clients, generic evaluation/features, LightGBM wrapper
  model_risk_regimes/           # hourly day-ahead price model: risk validation, regimes, dashboard (the original project)
  model_renewables/             # wind/solar capacity-factor reconstruction from ECMWF IFS weather runs (Open-Meteo)
  model_post_crisis_resolution/ # quarter-hourly price model: hourly LightGBM + shape-profile bridge
```

Enforced by `tests/test_dependency_rules.py` (AST-based, parametrized over every file in `src/energy_price_forecast/{common,model_risk_regimes,model_renewables,model_post_crisis_resolution}` and `scripts/{...}`):

| Folder | May import |
|---|---|
| `common` | `common` only |
| `model_risk_regimes` | `common` |
| `model_renewables` | `common` |
| `model_post_crisis_resolution` | `common`, `model_renewables` |

`tests/`, `scripts/` mirror the same four-way split. Importing `arena`/`ops` (the sibling repo's live-operation-only packages) anywhere in this repo is rejected by the same test — those never belong here.

## Working conventions

- **Language:** all code, docstrings, inline comments, log/test messages — English only. Spec documents under `docs/` (German, for the owner) are the one exception; this file follows the same rule as the sibling repo's `CLAUDE.md`.
- **Lint:** `uv run ruff check .` / `uv run ruff format --check .`, both CI-gated (`.github/workflows/ci.yml`).
- **Typecheck:** `uv run mypy src` is CI-gated (scoped to `src` only — `scripts/` can't join that invocation: `scripts/model_risk_regimes/run_dm_test.py` collides as "found twice" against `tests/common/test_dm_test.py`'s import of it, a structural, pre-existing issue, not something to chase). Locally, mypy is sometimes blocked by Windows Smart App Control (`DLL load failed`); `uv run pyright` is the fallback (`pyrightconfig.json` at repo root), with a manual diff-check of new imports / `# type: ignore` comments against what CI's mypy would actually say before pushing. `pyright` itself isn't CI-gated and carries its own small baseline of pre-existing errors — the question after a change is whether the error *set* grew, not the raw count.
- **Tests:** `uv run pytest` (`-m "not integration"` by default — deselects tests that hit the real ENTSO-E API). Run a single model's tests with `pytest tests/<folder>`. `tests/common/test_provenance.py` is the golden-fixture test guarding byte-identical outputs on known inputs — don't touch its fixtures casually.
- **Leakage / knowledge-time discipline:** every feature is classified by *when it was actually known* (`common/features/availability.py`'s `Availability` enum — `DETERMINISTIC`/`DA_FIXED`/`DA_FORECAST`/`RT_ACTUAL`/`COMMODITY`/`NWP_RECONSTRUCTION`), not by when it's dated. Gate closure and calendar-day arithmetic live in `common/market_time.py` and use `pd.DateOffset`/plain `date` arithmetic, never a fixed-duration offset on a tz-aware `Timestamp` — the latter silently produces the wrong wall-clock time across a DST transition. Any new backtest, probe, or walk-forward fold must derive its reference time from the data/fold itself, never read `pd.Timestamp.now()`/`dt.date.today()`.

## Where things live

- **Specs and implementation logs:** `docs/*.md` (gitignored, not on GitHub — read them locally for the "why" behind a given sprint; `docs/sprint7_step7_0_spec.md` + `docs/sprint7_step7_0_log.md` for this restructure).
- **Reports and results meant to be public:** `outputs/` (checked in) — `outputs/results/` (CSV snapshots the README/dashboard read from), `outputs/assets/` (figures), `outputs/model_validation_report.md`.
- **Data:** `data/README.md` documents provenance (which file came from where, which script regenerates it) for the artifacts copied from the sibling repo in Sprint 7.0 (quarter-hourly prices, weather archive, renewables forecasts, weather/ENTSO-E caches).
