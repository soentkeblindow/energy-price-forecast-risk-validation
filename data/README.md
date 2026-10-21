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

`data/raw/scratch_payload_vs_realized.csv` -- copied 2026-10-20 (Sprint 7 Step
7.3, Teil 0) from `sbl-energy-forecast`'s repo root (read-only, owner-approved
for 7.3). Daily live-submission-vs-realized-price bias/RMSE/MAE at
quarter-hourly resolution, 2026-09-15 to 2026-10-19, used by
`scripts/model_risk_regimes/close_data_edge.py` as the live reference row in
the edge-closure daily comparison table. Not regenerable from Repo 1 -- it is
a snapshot of Repo 2's own live submission logs
(`scratch_compare_payloads_realized.py` there). Its `bias` column is `pred -
realized` (positive = overestimate); this repo's own `bias()` convention is
the opposite (`y_true - y_pred`, positive = underestimate) -- the sign gets
flipped on load, not in this copy.

**NWP backfill to the 7.3 edge (2026-10-20), done 2026-10-20 (Sprint 7 Step
7.3, Teil 1):**
- `cache/weather_single_runs/ecmwf_ifs/grid_v1_e820fc3a/2026/10/{02..20}T00Z.parquet`
  (19 files) copied from `sbl-energy-forecast`'s identical cache (read-only,
  owner-approved for 7.3) -- plain file copies of already-fetched ECMWF IFS
  runs, no new Open-Meteo calls. September was already complete in Repo 1
  (30/30 days).
- `interim/weather_ifs_run00.parquet` rebuilt with
  `scripts/model_renewables/build_weather_artefact.py --start 2024-03-14 --end
  2026-10-20` -- 951 candidate days, all cache hits (0 new API calls), same 4
  pre-existing known-defect days (2025-08-05/06/08/09, provider outage,
  already in `KNOWN_WEATHER_DEFECTS`).
- `processed/renewables_forecast_rolling365_l2.parquet` extended, **not
  regenerated from scratch**: a full `rolling365_l2` re-run over the entire
  2024-2026 history is exactly the "not cheap" full walk-forward job this file
  was originally copied (not re-run) to avoid. Instead, `run_renewables_backtest`
  was called directly for a short window (`target_hourly` restricted to
  2025-07-01..2026-10-20, built from `interim/hourly_merit_order.parquet`
  -- not the shared `interim/hourly.parquet`, which only extends to
  2025-12-31 and was deliberately left untouched, same reasoning as 7.2
  Phase 0) with the variant's exact parameters (`window="rolling"`,
  `train_span_days=365`, `objective="l2"`, default `refit_every=7`,
  matching the plain `train_renewables.py --variant rolling365_l2` invocation
  this table's "Regenerate with" column already documented). Only the rows
  with `valid_time_utc` after the existing file's previous max
  (2026-09-07 09:00 UTC) were appended (1,044 new rows, zero duplicate
  `valid_time_utc` after the merge) -- the overlapping ~2 months the
  restricted run also produced were discarded, used only to give the
  365-day rolling window enough runway.
