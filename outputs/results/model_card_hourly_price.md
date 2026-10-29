# Model Card: Hourly Day-Ahead Price Model (DE-LU)

This card describes two versions of the hourly day-ahead price model
evaluated and selected in Sprint 7 Step 7.6 of this repository
(`energy-price-forecast-risk-validation`). It makes no statement about
live operation or about the sibling repository (`day-ahead-forecast-live-ops`)
— whatever reaches production there is that repository's own decision.

## 1. Identity

| | `hourly_model_v1` | `hourly_model_v2` |
|---|---|---|
| Name | hourly_model_v1 | hourly_model_v2 (B-full-roll) |
| Status | superseded | current |
| Date of this determination | — | 2026-10-23 |
| Reference | Step 7.2 (merit-order structure measurement) | Step 7.6, Part B (comparison measurement) + owner decision |

`hourly_model_v2` was selected in Sprint 7 Step 7.6: Part B measured three
candidates (`B`, `B-full`, `B-roll`) against the historical reference
`A` (= `hourly_model_v1`) and found them statistically equivalent (no
Diebold-Mariano test between the candidates significant). The owner then
chose `B-full-roll` — a combination not itself run as a separate
candidate in Part B — on economic and physical grounds, not on measured
performance. Part C.1 subsequently ran `B-full-roll` as a confirmation:
its predictions and beta estimates match `B-full`'s exactly through
anchor 2026-04, diverging only slightly afterward as the rolling window
narrows. See Section 3's `rationale` column and `price_model_development.md`
(section 2, "Selection: B/B-full/B-roll") for the full reasoning, and
Section 5's known limitations below for what it acknowledges in exchange.

## 2. Purpose and scope

Both versions forecast the hourly German-Luxembourg (DE-LU) day-ahead
electricity price for a given delivery day D. The information state is
restricted to what is known at gate closure for D (D−1, roughly noon
CET): no feature used by either version is dated later than that, per
this repository's leakage-time discipline
(`common/features/availability.py`).

Resolution is hourly throughout. Neither version directly covers
quarter-hourly prices — the project's separate quarter-hourly model
(`model_post_crisis_resolution`) bridges from this hourly model's output
to quarter-hour shape, and is out of scope for this card.

This card describes only the models as measured in this repository's
own research; it makes no further claim about live deployment beyond
the note above.

## 3. Configuration

"Best performance" is not cited as a rationale for any `hourly_model_v2` row
below: Step 7.6 Part B found `B`, `B-full` and `B-roll` statistically
indistinguishable (no significant DM test among them). Where a
`hourly_model_v2` setting differs from `hourly_model_v1`, the rationale is economic or
physical reasoning, stated explicitly by the owner after Part B.

| Setting | `hourly_model_v1` | `hourly_model_v2` | Rationale | Evidence |
|---|---|---|---|---|
| Base model | Plain LightGBM, no target transform | Target-transform model: `price − β(RL)·SRMC`, LightGBM on the residual | The transform passed all of Step 7.2's criteria; `hourly_model_v2` builds on it | `price_model_development.md`, section 2 ("Variants A/B/C"); `src/energy_price_forecast/common/models/merit_order_forecaster.py` |
| SRMC definition | n/a (no transform) | `gas_full` (TTF gas + EUA CO2) | A gas plant's short-run marginal cost includes fuel and carbon; gas-only folds the CO2 share into β, where the 1.5 ceiling binds persistently in the top residual-load bin | owner decision; `price_model_development.md`, section 2 ("Selection: B/B-full/B-roll"); `src/energy_price_forecast/common/features/merit_order.py::srmc_gas_full` |
| RL segmentation | n/a | Absolute, fixed MW bins, frozen on 2019-2020 deciles | Not re-measured in Step 7.6; owner decision based on Step 7.2's evidence: the relative-bin sensitivity (`B-rel`) was significantly worse than the plain reference in the post-crisis period (DM test, daily p=0.0162, hourly p=0.0046, higher loss for `B-rel`) — no direct `B-rel`-vs-`B` test exists, this is the only available evidence | [`merit_order_variants_dm_test.csv`](https://github.com/soentkeblindow/energy-price-forecast-risk-validation/blob/sprint7-studies/outputs/results/merit_order_variants_dm_test.csv) (`B_rel_vs_A`, `post_crisis`); `src/energy_price_forecast/common/features/merit_order.py::RL_BIN_EDGES_MW` |
| Beta window | n/a | Rolling 3 years (`beta_window_days=3*365`), re-anchored monthly | The merit-order fleet is not stationary (2023 nuclear phase-out, coal held in reserve, storage build-out); a rolling window tracks the current fleet rather than averaging over the whole history | owner decision; `price_model_development.md`, section 2 ("Selection: B/B-full/B-roll") |
| Crisis exclusion | n/a | 2021-09-01 to 2023-04-01 excluded from the beta pool | Decided before Step 7.2's Phase 2; the crisis-period price/gas relationship is not representative | `src/energy_price_forecast/common/features/config.py::FeatureConfig.crisis_start/post_crisis_start` |
| Beta bounds | n/a | `[0, 1.5]`, isotonically smoothed | A pass-through below 0 or above 1.5x the commodity price has no physical reading in a thin bin | `src/energy_price_forecast/common/features/merit_order.py::BETA_CLAMP_MIN/BETA_CLAMP_MAX` |
| Isotonic smoothing | n/a | Yes, non-decreasing over bin order | Smooths thin-bin noise while enforcing the expected monotonic merit-order relationship | `src/energy_price_forecast/common/features/merit_order.py::smooth_beta_isotonic` |
| Beta re-estimation | n/a | Calendar month (`beta_update_rule="calendar_month"`) | Reproducible across runs with different start dates | `price_model_development.md`, section 2 ("Calendar rule for β"); `src/energy_price_forecast/common/models/merit_order_forecaster.py` |
| Min. observations per bin | n/a | 48 | Same floor as Step 7.2's own structural measurement | `src/energy_price_forecast/common/features/merit_order.py::_MIN_OBS_PER_BIN` |
| Objective function | Median (`objective="quantile"`, α=0.5) | Median (`objective="quantile"`, α=0.5) | Step 7.3 found `l2` not significantly different | `price_model_development.md`, section 2 ("Error decomposition") |
| Training window | 90 days, rolling | 90 days, rolling | A seasonal (60+30-day) window was tested in a separate investigation and found not to carry | `price_model_development.md`, section 2 ("Seasonal training window") |
| Refit cadence | Daily | Daily | Unchanged | — |
| Feature set | `core_gas` (`common/features/core_gas_columns.py::L1_COLUMNS`) | `core_gas` + `eua_co2_lag_48h` | The L0→L1 feature reduction was free of cost (Step 7.3); EUA is added for `hourly_model_v2`'s SRMC and, separately, as its own tree feature | `src/energy_price_forecast/common/features/core_gas_columns.py` |
| LightGBM parameters | `n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=1.0, colsample_bytree=1.0` | same | Unchanged default | `src/energy_price_forecast/common/models/lgbm.py::_DEFAULT_PARAMS` |
| Seed / parallelism | `random_state=0, n_jobs=1` | same | Unchanged default | `src/energy_price_forecast/common/models/lgbm.py`, `merit_order_forecaster.py` constructor defaults |

## 4. Data

| Input | Availability class | Notes |
|---|---|---|
| DE-LU day-ahead price (`day_ahead_price`) | `DA_FIXED` | Target variable; also the `price` column in the beta long-history side channel |
| Load forecast (`load_forecast_day_ahead`) | `DA_FORECAST` | ENTSO-E day-ahead forecast |
| Wind onshore/offshore, solar forecast | `DA_FORECAST` | ENTSO-E day-ahead forecasts; residual load and renewable share are derived from these plus load |
| TTF gas price (`ttf_gas_eur_per_mwh`, 48h lag) | `COMMODITY` | Daily settlement, known from the next day; underlies the SRMC in both `hourly_model_v2`'s gas-only and gas-full terms |
| EUA CO2 price (`eua_co2_eur_per_t`, 48h lag) | `COMMODITY` | `hourly_model_v2` only (SRMC's CO2 term and a separate tree feature); real data starts 2021-10-19 |

All inputs are hourly. The underlying price source is quarter-hourly from
the market's own switch to 15-minute delivery; quarter-hour values are
volume-weighted-averaged into hourly values
(`common/data/normalize.py::to_hourly_vwap`) before either model sees
them — this model operates purely at hourly resolution, see Section 2.

**Data stand** (Step 7.7, Teil C — [`data_snapshot.csv`](hourly_model_v2/data_snapshot.csv)):

| File | Period | Rows | SHA-256 |
|---|---|---|---|
| `hourly_merit_order.parquet` | 2019-01-01T00:00:00+00:00 – 2026-10-20T21:00:00+00:00 | 68,398 | `68fe91ae1004d7f58634be8f54ec5850224fe02eff9bc972f6b671610210e3fe` |
| `features_merit_order.parquet` | 2019-01-08T23:00:00+00:00 – 2026-10-20T21:00:00+00:00 | 68,207 | `8cf507346cb199dd4ffa81f2762cf5c582b7fe6dcbf48415dfd7d93716fe1ce0` |
| `renewables_forecast_rolling365_l2.parquet` (L2's own weather reconstruction) | 2024-03-15T23:00:00+00:00 – 2026-10-20T21:00:00+00:00 | 22,463 | `65662bdea761509458de6540e66fe1069cf785118f799848f86ce72bf8e0b0c6` |

## 5. Reference results and known limitations

Walk-forward, identical fold list, 2024-06-15 to 2026-10-20 (819 delivery
days). **L2** (own weather-reconstructed wind/solar) is the headline
number — it matches the live information state. **L1** (TSO day-ahead
forecasts) is reported alongside as an explicit **upper bound**: those
forecasts are not available at gate closure.

| Period | Metric | `hourly_model_v1` L2 | `hourly_model_v2` L2 | DM p-value (`hourly_model_v2` vs. `hourly_model_v1`) |
|---|---|---|---|---|
| Overall | RMSE / Bias | 25.49 / 1.27 | 25.44 / 0.61 | 0.82 (daily) |
| Autumn/winter (Sep-Dec, all years) | RMSE / Bias | 32.01 / 1.09 | 31.95 / 0.19 | 0.90 (daily) |
| Autumn 2026 | RMSE / Bias | 30.81 / 1.99 | 29.46 / −1.23 | 0.33 (daily) |

**L1, upper bound:**

| Period | Metric | `hourly_model_v1` L1 | `hourly_model_v2` L1 | DM p-value (`hourly_model_v2` vs. `hourly_model_v1`) |
|---|---|---|---|---|
| Overall | RMSE / Bias | 24.30 / 1.28 | 23.99 / 0.69 | 0.14 (daily) |
| Autumn/winter | RMSE / Bias | 30.75 / 1.14 | 30.46 / 0.40 | 0.44 (daily) |
| Autumn 2026 | RMSE / Bias | 29.01 / 1.91 | 28.30 / −1.09 | 0.68 (daily) |

`hourly_model_v2` is not significantly different from `hourly_model_v1` in any period
shown (all p-values above 0.1). The L1→L2 cost is highly significant for
both versions (p<1e-6) and matches the error-decomposition study's own
1.1-1.6 EUR/MWh range. Full tables: `outputs/results/hourly_model_v2/`.

**Known limitations:**

- In the anchor month 2023-12, β is 0 in all 11 bins: the raw per-bin
  estimates are not monotonic in residual load that month (despite
  enough observations per bin), and isotonic smoothing pulls the whole
  curve to its floor. `hourly_model_v2` was effectively untransformed that one
  month.
- In autumn 2026 specifically, `B-full-roll`'s rolling 3-year β window is
  significantly *worse* than the expanding-window alternative (`B-full`,
  daily DM p=0.028, hourly p=0.007, Step 7.6 Teil C.1) — the rolling
  window was chosen for its economic rationale, not because it measured
  better; this is the one period where that trade-off is visible.
- β learns a gas-price rise with roughly 1-2 months' delay (it only sees
  a price move once it enters the long-history pool); in long calm gas
  periods, β drifts back down.
- The 3-year β-window length is a set choice, not a measured one.
