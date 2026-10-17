# Data

`data/raw`, `data/interim`, `data/processed`, and `data/cache` are gitignored
(see `.gitignore`) -- too large and partly non-public for a public repo.
`config.py` resolves all four relative to the repo root, so any checkout only
needs the files below in place to run the test suite and the scripts.

Most of `data/raw/entsoe` and the processed backtest artefacts are built by
the `model_risk_regimes` pipeline itself (`scripts/model_risk_regimes/build_interim.py`,
`build_features.py`, etc.) and regenerate from a fresh ENTSO-E pull. The four
files below are the exception: they were copied once from `sbl-energy-forecast`
(Sprint 7 Step 7.0, 4.5) because regenerating them either hits a third-party
daily rate limit or would require re-running an expensive walk-forward training
job just to reproduce numbers that already exist.

| File | Used by | Regenerate with |
|---|---|---|
| `interim/quarterhourly_prices.parquet` | `model_post_crisis_resolution` (native 15-minute price series, shape-profile bridge) | `scripts/model_post_crisis_resolution/build_quarterhourly_prices.py` -- re-fetches from the ENTSO-E cache, no rate limit of its own |
| `interim/weather_ifs_run00.parquet` | `model_renewables` (ECMWF IFS weather artefact feeding the wind/solar reconstruction) | `scripts/model_renewables/build_weather_artefact.py` -- rebuilds from `cache/weather_single_runs/`, no new Open-Meteo calls needed as long as that cache is present |
| `processed/renewables_forecast_rolling365_l2.parquet` | `model_post_crisis_resolution` features (`loaders.load_renewables_predictions`) | `scripts/model_renewables/train_renewables.py --variant rolling365_l2` -- a full walk-forward training run, not cheap; copied rather than re-run to match the numbers already validated in `sbl-energy-forecast` |
| `cache/weather_single_runs/ecmwf_ifs/` (926 per-run files) | input to `build_weather_artefact.py` above | Open-Meteo ECMWF IFS archive, one HTTP call per run -- **do not re-fetch**, the provider enforces a daily call limit |

`data/raw/entsoe` also received 48 additional cached months (Aug-Oct 2026,
across all six ENTSO-E series) copied from `sbl-energy-forecast`'s cache,
additively -- Repo 1's own older history (back to 2019) was left untouched.
Without those months, `quarterhourly_prices.parquet` and the renewables
pipeline would need to hit the ENTSO-E API for a range this repo had not
previously fetched.
