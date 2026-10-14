# Day-Ahead Power Price Forecasting & Risk Backtesting (DE-LU)

Probabilistic forecasts with conformal calibration, ES/VaR backtesting, and regime-conditional validation.

[![CI](https://github.com/soentkeblindow/energy-price-forecast-risk-validation/actions/workflows/ci.yml/badge.svg)](https://github.com/soentkeblindow/energy-price-forecast-risk-validation/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230?logo=ruff)](https://github.com/astral-sh/ruff)
[![Typed with mypy](https://img.shields.io/badge/typed-mypy-1f5082?logo=python)](https://mypy-lang.org/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Probabilistic day-ahead price forecasting for the German-Luxembourg bidding zone (ENTSO-E data), comparing regularised linear baselines, LightGBM quantile regression, and ARIMAX. All backtests run under a strictly leakage-free walk-forward evaluation at gate-closure information (~12:00 D-1). The forecast intervals are then conformal-calibrated, and the resulting trading-book Value-at-Risk / Expected Shortfall of a hypothetical 10 MW book is **backtested in its own right**, using the same lens an independent model-validation unit would apply, not just a point-accuracy comparison.

![Forecast vs. actual, daily means](outputs/assets/forecast_vs_actual.png)

Daily-mean actual price vs. LightGBM median forecast over the full test period. The 2021/22 energy-crisis peak and the subsequent normalisation are clearly visible; the model tracks both regimes closely.


## Headline results

All metrics on the common 2021-2025 walk-forward test set (rolling-90-day window, `refit_every=1` for LightGBM, `refit_every=7` for ARIMAX).

| Model | MAE (EUR/MWh) | RMSE (EUR/MWh) | WAPE |
|---|---|---|---|
| Seasonal Naive | 34.77 | 56.20 | 0.290 |
| Lasso | 58.17 | 139.08 | 0.485 |
| ARIMAX | 22.68 | 35.55 | 0.189 |
| **LightGBM** | **14.87** | **25.64** | **0.124** |

- **LightGBM leads by a wide margin:** MAE 14.9 EUR/MWh vs. 22.7 for ARIMAX (the best alternative) and 34.8 for the naive benchmark — roughly 34% lower error than the best alternative model. Lasso's 58.2 is not representative of baseline linear performance; see the Lasso finding under Honest limitations below.
- **That edge is not noise:** a Diebold-Mariano test on daily mean absolute-error loss differentials (Newey-West HAC, Harvey correction) confirms the LightGBM advantage over both benchmarks; an hourly HAC variant yields the same conclusion (see `outputs/results/dm_test.csv`).

| Comparison | Mean loss diff. (EUR/MWh) | DM statistic | p-value | n (days) |
|---|---|---|---|---|
| LightGBM vs. Lasso | -43.27 | -8.24 | < 0.001 | 1827 |
| LightGBM vs. ARIMAX | -7.80 | -14.55 | < 0.001 | 1827 |

**Calibration and risk:**

- **Raw quantile intervals massively under-cover:** reporting the raw, uncalibrated q05 as a 5% VaR limit breaches in 18.0% (long) / 17.3% (short) of hours, not 5%. Scaled conformal recalibration fixes this: breach rates drop to 4.9% (long) / 5.2% (short), and the implied capital buffer is 1.53x (long) / 1.72x (short) larger than the uncalibrated one would have reported.
- **The calibrated risk measure passes the unconditional backtest but fails the conditional one:** Kupiec does not reject at `overall`, yet 33 of 60 conditioning cells (hour-of-day phase, forecast-height bucket, forecast-based regime) break coverage under a rigorous bootstrap CI (see the forest plot figure below).

## What makes this honest

- **Gate-closure leakage discipline.** Every feature is restricted to information known by ~12:00 Europe/Berlin on D-1, the real day-ahead gate-closure time, not information available only in hindsight.
- **Walk-forward, not a random split.** All backtests use a rolling or expanding time-ordered evaluation; hyperparameter tuning happens in an inner loop nested inside that walk-forward, never touching test-period data.
- **Calibration before risk, never the other way round.** The trading-book VaR/ES is computed on top of already-calibrated quantile forecasts, not on raw model output dressed up as a risk number.
- **The risk measure is backtested, not just computed.** Three independent statistical test families: Kupiec (frequency), Christoffersen (clustering), and Acerbi-Szekely (severity); plus a month-stratified, day-block bootstrap for reliable confidence intervals, applied to all 60 (variant x side x subset) cells.
- **Findings are reported even when they complicate the story.** The conditional coverage breaks, the day-level breach clustering, and the ARIMAX under-coverage are stated as findings, not smoothed over (see Honest limitations below).

## Figures

![Fan chart: raw vs. calibrated quantile bands](outputs/assets/fan_chart_calibration.png)

Fan chart over a two-week window (2024-12-05 to 2024-12-19) containing the single highest-spread day in the test period (2024-12-12, ~829 EUR/MWh spread, a documented winter Dunkelflaute event). The raw 90% band (left) is visibly too narrow during the spike; the conformal-calibrated band (right) widens specifically where the raw band would have missed the realised price.

<img src="outputs/assets/reliability_diagram.png" alt="Reliability diagram" width="420">


Nominal vs. empirical one-sided coverage per quantile level. Raw LightGBM sits below the diagonal (under-coverage); the calibrated curve tracks the diagonal closely; ARIMAX (pale reference, 3-level grid only) under-covers more severely: at the 0.95 level, empirical coverage is 62.4% against a 95% nominal target.

<img src="outputs/assets/coverage_forest.png" alt="Coverage forest plot" width="600">


Unconditionally the calibrated risk measure holds: Kupiec does not reject the breach-rate null anywhere at the `overall` subset. Conditionally it does not (see Honest limitations below for the full cell-by-cell breakdown). That gap between unconditional and conditional calibration is a central **model limitation and an interesting finding:** stress events known ahead of forecast time still seem hard for a general modelling approach to incorporate, even for advanced tree-based predictors.

## Project structure

```text
├── data/
│   ├── raw/
│   ├── interim/            # hourly price + fundamental data
│   └── processed/          # feature matrix + backtest/calibration/risk results
├── outputs/
│   ├── assets/              # exported PNG figures (this README's images)
│   └── results/             # exported CSV tables + prediction snapshot (checked in)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_model_diagnostics.ipynb   # point comparison, SHAP, fan charts, ablation
│   └── 04_regime_risk.ipynb         # calibration + trading-book VaR/ES, backtest validation
├── src/energy_price_forecast/
│   ├── data/                # loaders, ENTSO-E + commodities clients
│   ├── features/            # calendar, fundamentals, lags, availability, subset
│   ├── models/               # SimilarDayNaive, Lasso/Ridge/OLS, LGBMForecaster, ARIMAXForecaster, tuning
│   ├── evaluation/           # walk-forward harness, metrics, regimes, calibration, risk, backtest, bootstrap
│   ├── reporting/            # pure functions behind this README's figures/tables/snapshot
│   └── dashboard/            # Streamlit app (Sprint 5.3)
├── scripts/                  # backtest.py, calibrate_conformal.py, compute_risk.py, backtest_risk.py, ...
└── tests/                    # 469 tests, CI green
```

Data flows raw (ENTSO-E/commodities APIs) -> interim (hourly, normalised) -> features (leakage-safe) -> backtest predictions -> calibration/risk evaluation -> `outputs/` (the small, checked-in snapshot this README and the future dashboard read from).

## Getting started

```bash
uv sync
cp .env.example .env   # fill in your ENTSO-E API key (Transparency Platform)

# Rebuild the feature pipeline from raw data (optional, not needed to read this README)
uv run python scripts/build_interim.py
uv run python scripts/build_features.py

# Regenerate the figures/tables this README embeds, from already-persisted results
make report-assets
```

`outputs/results/` contains checked-in, versioned snapshots of the backtest, calibration, and risk results. The figures above and the numbers in this README are reproducible directly from those files, without needing an ENTSO-E API key or a full pipeline re-run. Raw data sources: ENTSO-E Transparency Platform (day-ahead prices, load, generation, cross-border flows) and Yahoo Finance (TTF gas, EUA CO2 futures); raw data itself is not redistributed in this repository.

## Honest limitations

- **Conditional coverage breaks in 33 of 60 cells**, concentrated in the evening ramp and the tail forecast-height buckets. The calibration corrects the unconditional scale but not this conditional structure. In the evening ramp, `calibrated`/short alone breaches at 9.3% against a nominal 5% (CI [8.2%, 10.4%]), close to double the target rate in exactly the hours the book is most exposed.
- **Day-level breach clustering survives the conformal correction.** Christoffersen's day-level independence test rejects independence for every validated (variant, side) combination at `overall` (`LR_ind` 49-79): breaches bunch into multi-day runs (cold snaps, Dunkelflauten), not just isolated hours.
- **Kupiec on the raw hourly series is anti-conservative.** It rejects 45 of the same 60 cells, more than the bootstrap's 33, because intra-day breaches are correlated and the hourly chi-square test does not know that. The day-block-bootstrapped test is the one this project reports as the headline; Kupiec is shown alongside it, not in place of it.
- **ARIMAX's quantile intervals under-cover more severely than LightGBM's, even before calibration.** At the 0.95 level, empirical coverage is 62.4% against a 95% nominal target, vs. LightGBM raw's 81.7% at the same level. This is the documented cost of a classical linear-Gaussian interval assumption against a heavy-tailed, regime-switching price series.
- **The energy-crisis period (2021/22) is a structural stress test for every model**, not a regime any of them was specifically tuned for; performance there is reported, not excluded.
- **The Lasso baseline is fragile to an out-of-distribution commodity price.** A data-pipeline fix (forward-filling the daily TTF gas price by real elapsed hours rather than by row count) now correctly carries forward the actual TTF spot price from the August 2022 energy-crisis peak into hours where it was previously `NaN`. That single feature value sits far outside Lasso's training range; its `asinh`/`sinh` target transform extrapolates explosively on it (one affected hour: a 4,478 EUR/MWh prediction against a realised 861 EUR/MWh). LightGBM (tree splits don't extrapolate linearly) and ARIMAX (no commodity exogenous regressor) are unaffected — this is a reported finding about Lasso's extrapolation fragility, not a bug fix applied to Lasso itself; Lasso remains a baseline comparison here, not a candidate model.

## Future work

- **LSTM/Transformer comparison** — deliberately out of scope: the marginal accuracy gain over a well-tuned LightGBM quantile model is unlikely to justify the added complexity and interpretability cost for this use case.
- **Regime-adaptive calibration** — the conditional coverage breaks documented above point directly at this as the next methodological step, not attempted here.
- **A ramp term in the local-scale (sigma) estimate** — a narrower fix targeting the specific intraday heteroskedasticity pattern behind the evening-ramp coverage breaks.
- **Further risk measures** (drawdown statistics, extreme quantiles beyond q05/q95) — out of scope for a backtesting-focused deliverable.
- **A standalone 15-minute-resolution model**, rather than resampling the known post-2025-09-30 resolution change to an hourly grid.

## Links & docs

- [`01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb): ENTSO-E data exploration, ACF analysis.
- [`02_baselines.ipynb`](notebooks/02_baselines.ipynb): linear baselines (Lasso/Ridge/OLS) vs. naive benchmark.
- [`03_model_diagnostics.ipynb`](notebooks/03_model_diagnostics.ipynb): LightGBM/ARIMAX point comparison, SHAP attribution, feature ablation.
- [`04_regime_risk.ipynb`](notebooks/04_regime_risk.ipynb): conformal calibration, trading-book VaR/ES, full backtest validation (source of this README's headline figures/numbers).
- [`model_validation_report.md`](outputs/model_validation_report.md): independent-validation-style review with a numbered findings catalogue (Limitation / Accepted / Remediated / Future Work).
- **[Live demo](https://energy-price-forecast-azetvtdc9hgc7yylhqp84q.streamlit.app/)**: interactive Streamlit Backtest Explorer (Forecast vs. Actual, Regime breakdown, Calibration & Risk views), read-only against the checked-in results snapshot.

  ![Backtest Explorer demo](outputs/assets/dashboard_demo.gif)

## License

[MIT](LICENSE). Data sources: ENTSO-E Transparency Platform and Yahoo Finance; see their respective terms for redistribution of the underlying raw data.
