"""Dashboard metric computations (Sprint 5.3).

Pure functions over already-loaded DataFrames -- no file I/O, no Streamlit.
Shared by View 1 (live breach rate) and View 2 (regime breakdown table) and
View 3 (ES/VaR traffic light) of the Backtest Explorer.
"""

from __future__ import annotations

import pandas as pd

from ..evaluation.regimes import MACRO_REGIME_COLUMN, REGIME_FLAG_COLUMNS

# Below this hour count, a regime row is marked low-support in the dashboard
# table (illustrative only, not dropped) -- roughly one month of hours, the
# same order of magnitude as the backtest bootstrap's day-level low_support
# cutoff (`RiskConfig.min_cell_days = 30`), though this is a separate,
# hour-based number since this table aggregates snapshot rows directly, not
# bootstrap day-cells.
LOW_SUPPORT_MIN_HOURS = 720

# Nominal one-sided VaR/ES level used throughout Sprint 4's risk backtesting
# (alpha = 1 - RiskConfig.level = 1 - 0.95); hardcoded here to mirror the
# spec's own literal "0.05" rather than importing the risk-bootstrap config
# into the dashboard package.
NOMINAL_LEVEL = 0.05


def interval_breach_rate(actual: pd.Series, lo: pd.Series, hi: pd.Series) -> float:
    """Fraction of hours where `actual` falls outside `[lo, hi]`.

    Rows where any of the three series is NaN are dropped before computing
    the rate. Returns NaN if nothing is left to evaluate.
    """
    frame = pd.concat([actual.rename("actual"), lo.rename("lo"), hi.rename("hi")], axis=1).dropna()
    if frame.empty:
        return float("nan")
    breached = (frame["actual"] < frame["lo"]) | (frame["actual"] > frame["hi"])
    return float(breached.mean())


def regime_metrics(
    snapshot: pd.DataFrame, model_comparison_by_regime: pd.DataFrame
) -> pd.DataFrame:
    """LightGBM's regime breakdown for View 2(a): `regime`, `n`, `mae`, `rmse`,
    `wape`, `calibrated_breach_rate`, `low_support` -- one row per regime label
    present in `model_comparison_by_regime` (`overall`, macro categories, flags).

    Point metrics (`n`, `mae`, `rmse`, `wape`) are the `model == "LightGBM"`
    slice of `model_comparison_by_regime.csv` (Sprint 5.3.1 Decision 3 -- one
    source of truth for point metrics, so LightGBM's MAE never shows up twice
    with a possible float16-vs-CSV rounding mismatch). The calibrated interval
    breach rate is NOT in that CSV (only LightGBM carries bands, and only in
    the snapshot) and stays live-computed here, joined onto the CSV's rows by
    regime label. `low_support` uses the CSV's `n` -- the same threshold as
    before, now applied to the authoritative count.
    """
    lightgbm = model_comparison_by_regime.loc[
        model_comparison_by_regime["model"] == "LightGBM"
    ].set_index("regime")

    def mask_for(regime: str) -> pd.Series:
        if regime == "overall":
            return pd.Series(True, index=snapshot.index)
        if regime in REGIME_FLAG_COLUMNS:
            return snapshot[regime]
        return snapshot[MACRO_REGIME_COLUMN] == regime

    rows: list[dict[str, object]] = []
    for regime, row in lightgbm.iterrows():
        subset = snapshot.loc[mask_for(str(regime))]
        breach_rate = (
            interval_breach_rate(
                subset["price_actual"], subset["lo_calibrated"], subset["hi_calibrated"]
            )
            if len(subset) > 0
            else float("nan")
        )
        n = int(row["n"])
        rows.append(
            {
                "regime": regime,
                "n": n,
                "mae": row["mae"],
                "rmse": row["rmse"],
                "wape": row["wape"],
                "calibrated_breach_rate": breach_rate,
                "low_support": n < LOW_SUPPORT_MIN_HOURS,
            }
        )

    return pd.DataFrame(rows)


def filter_by_regime(model_comparison_by_regime: pd.DataFrame, regime: str) -> pd.DataFrame:
    """`model_comparison_by_regime` rows for one `regime` label, one row per
    model (`regime`/`axis` columns dropped -- redundant once filtered). For
    `regime="overall"` this reconciles exactly with `model_comparison.csv`.
    """
    return (
        model_comparison_by_regime.loc[model_comparison_by_regime["regime"] == regime]
        .drop(columns=["regime", "axis"])
        .reset_index(drop=True)
    )


def traffic_light(risk_headline: pd.DataFrame) -> pd.DataFrame:
    """Append a `status` column ("green"/"red"/"n/a") to `risk_headline`.

    Green iff `NOMINAL_LEVEL` lies inside `[breach_rate_ci_low, breach_rate_ci_high]`;
    red otherwise. Rows without a bootstrap CI (e.g. the `raw` variant, shown only as
    a counterfactual) get "n/a" -- there is no CI to test containment against.
    """
    frame = risk_headline.copy()
    lo = frame["breach_rate_ci_low"]
    hi = frame["breach_rate_ci_high"]
    has_ci = lo.notna() & hi.notna()
    contains_nominal = has_ci & lo.le(NOMINAL_LEVEL) & hi.ge(NOMINAL_LEVEL)
    frame["status"] = "red"
    frame.loc[contains_nominal, "status"] = "green"
    frame.loc[~has_ci, "status"] = "n/a"

    if "buffer_factor" in frame.columns:
        # buffer_factor (mean_var(calibrated) / mean_var(raw)) is structurally
        # only defined for the "calibrated" variant (reporting/tables.py) --
        # a blank cell for "raw"/"fhs" is correct but reads as broken, so it
        # gets an explicit label instead of a NaN/empty cell.
        frame["buffer_factor"] = frame["buffer_factor"].apply(
            lambda v: "n/a (calibrated only)" if pd.isna(v) else f"{v:.2f}x"
        )
    return frame
