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

## Hourly price model (`hourly_model_v2`)

`hourly_model_v2` is the current hourly day-ahead price model (Sprint 7 Step
7.6/7.7); its configuration lives in exactly one place,
`src/energy_price_forecast/common/models/price_model.py`
(`PriceModelConfig`, `HOURLY_MODEL_V1`/`HOURLY_MODEL_V2`, `build_price_model`) — no
other script sets model parameters directly. `scripts/model_risk_regimes/
run_price_model.py` is the main entry point to run it: by default
`--versions hourly_model_v2 --rungs L2` only (the live case -- `hourly_model_v1` and L1,
ENTSO-E's own forecasts never available live, are opt-in). Every run
writes to `data/processed/price_model_runs/` (gitignored) by default, no
exceptions -- the script has no notion of one "canonical" invocation.
Its reference numbers in `outputs/results/hourly_model_v2/` (`build_price_beta_table.py`
also contributes to that directory) are published by deliberately passing
`--out-dir outputs/results/hourly_model_v2` -- what `make price-model` runs,
alongside `--versions hourly_model_v1,hourly_model_v2 --rungs L1,L2`. `run_price_model.py`
also takes harness-parameter flags -- `--train-span-days`/`--test-start`/
`--test-end`/`--window`/`--refit-every`/`--drop-features`, gated behind a
required `--tag` so two different sweeps never overwrite each other's
output files (never MeritOrderForecaster's own structure). New measurements against
the hourly model run against `hourly_model_v2`, not against an ad hoc
reconfiguration of `MeritOrderForecaster` (it now only supports
`hourly_model_v2`'s own path, with one deliberate ablation exception:
`hide_from_tree`/`--drop-features` can hide a column from the *inner*
LightGBM tree while SRMC/beta are still computed from it as always).
Completed Sprint 7 studies (merit-order
structure, variant comparisons, error decomposition, the rejected
seasonal-window/similar-day/French-price features, the Step 7.6
selection measurement) were deleted from `main` in Step 7.7's cleanup
and live on in the `sprint7-studies` git tag — `main` shows the current
state, the git history (and that tag) is the archive. Every file on
`main` has a reader (README, the model card, a test, or one of the three
models' own code); what's public is the result with its evidence, what's
private is the path that got there; discarded ideas are still named,
with their result, in `outputs/price_model_development.md`.

## Where things live

- **Specs and implementation logs:** `docs/*.md` (gitignored, not on GitHub — read them locally for the "why" behind a given sprint; `docs/sprint7_step7_0_spec.md` + `docs/sprint7_step7_0_log.md` for this restructure).
- **Reports and results meant to be public:** `outputs/` (checked in) — `outputs/results/` (CSV snapshots the README/dashboard read from, incl. `outputs/results/hourly_model_v2/` and the public `outputs/results/model_card_hourly_price.md`), `outputs/assets/` (figures), `outputs/model_validation_report.md`, `outputs/price_model_development.md`.
- **Data:** `data/README.md` documents provenance (which file came from where, which script regenerates it) for the artifacts copied from the sibling repo in Sprint 7.0 (quarter-hourly prices, weather archive, renewables forecasts, weather/ENTSO-E caches). Since 2026-10-30, `data/interim/hourly_data.parquet`/`data/processed/features_hourly.parquet` is the single, live-growing source of truth for hourly data (pre-2020 history archived separately, see `data/README.md`); `model_risk_regimes`'s published legacy numbers are pinned to a 2020-01-01–2025-12-31 data window and archived unchanged under `outputs/archive_251231/` rather than frozen in code.
