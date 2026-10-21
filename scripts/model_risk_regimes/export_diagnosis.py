"""Teil B of docs/error_decomposition_followup_spec.md: ex-post export diagnosis.

Tests whether the model's error on low/mid residual-load hours correlates
with Germany's scheduled net export -- the hypothesis being that coupled
neighbor demand pulls the German price along without showing up in the
German residual load. Uses ONLY already-cached A/B-on-L0 predictions (7.2's
long-history cache, extended to the Teil-0 edge, same as Teil 2) and raw
cross-border-flow columns already in hourly_merit_order.parquet -- no new
backtests, no new features, no model changes.

EX POST, NOT A FEATURE: scheduled/physical flows of the target day are
settled in the same market-coupling auction as the price itself and are
unknown at gate closure. They are used here only to explain the error after
the fact, never as model input -- every output file/figure carries "ex
post - not a feature" in its name or title (spec section B.2/B.6). Because
flows and price co-determine each other, a correlation here is evidence
consistent with the hypothesis, not proof of causation (spec B.2): an
unexpectedly high German price would, if anything, suppress exports, so a
positive "more export -> more underestimation" relationship argues for
foreign demand pulling the price rather than a price echo running the
other way.

Known, deliberately unclosed gap: CZ/BE/DK2/SE4/NO2 are not in this
dataset. "Total export" below is the export over the corridors that ARE
present (FR/NL/AT/PL/CH/DK_1), not Germany's full foreign trade.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from energy_price_forecast.common.data.loaders import load_interim_hourly, load_processed_features
from energy_price_forecast.common.evaluation.walkforward import (
    extend_predictions_to_edge,
    last_complete_local_day,
    split_complete_delivery_days,
)
from energy_price_forecast.common.features.config import FeatureConfig
from energy_price_forecast.common.features.merit_order import build_merit_order_frame
from energy_price_forecast.common.market_time import LOCAL_TZ
from energy_price_forecast.common.models.lgbm import LGBMForecaster
from energy_price_forecast.common.models.merit_order_forecaster import MeritOrderForecaster

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TRAIN_SPAN_DAYS = 90
HAC_LAG = 24
N_RL_TERCILES = 3
N_EXPORT_QUINTILES = 5
RL_TERCILE_LABELS = ("low", "mid", "high")
OUTLIER_DAYS = (
    pd.Timestamp("2026-10-11", tz=LOCAL_TZ),
    pd.Timestamp("2026-10-18", tz=LOCAL_TZ),
    pd.Timestamp("2026-10-19", tz=LOCAL_TZ),
)
# Spec B.5 thresholds.
MONOTONIC_SPREAD_THRESHOLD_EUR_MWH = 5.0
SIGNIFICANCE_LEVEL = 0.05

_SCHEDULED_PREFIX = "scheduled_net_de_to_"
_PHYSICAL_PREFIX = "physical_net_de_to_"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-path", type=Path, default=Path("data/interim/hourly_merit_order.parquet")
    )
    p.add_argument(
        "--features-path", type=Path, default=Path("data/processed/features_merit_order.parquet")
    )
    p.add_argument(
        "--merit-order-preds-dir", type=Path, default=Path("data/processed"), help="7.2's cache."
    )
    p.add_argument("--out-dir", type=Path, default=Path("outputs/results"))
    p.add_argument(
        "--fig-path",
        type=Path,
        default=Path("outputs/assets/export_diagnosis_expost_heatmap.png"),
    )
    return p.parse_args()


def _load_extended_predictions(
    df: pd.DataFrame, features: pd.DataFrame, preds_dir: Path, edge: pd.Timestamp
) -> dict[str, pd.DataFrame]:
    """A/B on L0, 7.2's cache extended to the Teil-0 edge (same pattern as
    Teil 2's run_residual_error_decomposition.py, now shared via
    common.evaluation.walkforward.extend_predictions_to_edge)."""
    price = df["day_ahead_price"]
    y = price.reindex(features.index)
    x = features
    long_history = build_merit_order_frame(df, pd.DatetimeIndex(df.index))

    models = {
        "A": LGBMForecaster(objective="quantile", alpha=0.5),
        "B": MeritOrderForecaster(long_history, srmc_definition="gas_only"),
    }
    preds = {}
    for name, model in models.items():
        cached = pd.read_parquet(preds_dir / f"merit_order_preds_{name}.parquet")
        preds[name] = extend_predictions_to_edge(
            cached, model, x, y, edge, train_span_days=TRAIN_SPAN_DAYS, label=f"variant {name}"
        )
    return preds


def _total_flow(df: pd.DataFrame, prefix: str) -> pd.Series:
    cols = sorted(c for c in df.columns if c.startswith(prefix))
    return df[cols].sum(axis=1)


def _build_analysis_frame(
    df: pd.DataFrame,
    features: pd.DataFrame,
    preds: dict[str, pd.DataFrame],
    period_start: pd.Timestamp,
    edge: pd.Timestamp,
) -> pd.DataFrame:
    """One row per complete hour in [period_start, edge], both variants'
    errors plus the four export definitions and the RL tercile source."""
    complete_days_by_variant = {}
    for name, p in preds.items():
        complete_days, incomplete_days = split_complete_delivery_days(p)
        if len(incomplete_days):
            logger.info(
                "variant %s: %d incomplete delivery day(s) excluded", name, len(incomplete_days)
            )
        complete_days_by_variant[name] = set(complete_days)
    common_complete_days = set.intersection(*complete_days_by_variant.values())
    common_complete_days = {d for d in common_complete_days if period_start <= d <= edge}

    frame = pd.DataFrame(index=pd.DatetimeIndex(features.index))
    for name, p in preds.items():
        sub = p.loc[p["delivery_day"].isin(common_complete_days)]
        frame[f"error_{name}"] = (sub["y_true"] - sub["y_pred"]).reindex(frame.index)

    frame["export_total"] = _total_flow(df, _SCHEDULED_PREFIX).reindex(frame.index)
    frame["export_fr"] = df["scheduled_net_de_to_fr"].reindex(frame.index)
    frame["export_total_lag24"] = features["scheduled_net_export_lag_24h"].reindex(frame.index)
    frame["export_total_physical"] = _total_flow(df, _PHYSICAL_PREFIX).reindex(frame.index)
    frame["residual_load"] = features["residual_load_forecast"].reindex(frame.index)
    frame["delivery_day"] = pd.DatetimeIndex(frame.index).tz_convert(LOCAL_TZ).normalize()

    return frame.dropna(subset=["error_A", "error_B"])


def _assign_terciles(series: pd.Series) -> tuple[pd.Series, list[float]]:
    labels, edges = pd.qcut(series, N_RL_TERCILES, labels=RL_TERCILE_LABELS, retbins=True)
    return labels, list(edges)


def _assign_quintiles(series: pd.Series) -> tuple[pd.Series, list[float]]:
    labels, edges = pd.qcut(
        series, N_EXPORT_QUINTILES, labels=False, retbins=True, duplicates="drop"
    )
    return labels, list(edges)


def _cross_tab(
    frame: pd.DataFrame, export_col: str, export_label: str, rl_tercile: pd.Series
) -> pd.DataFrame:
    quintile, edges = _assign_quintiles(frame[export_col])
    logger.info("%s: quintile edges %s", export_label, [round(e, 0) for e in edges])
    rows = []
    for variant in ("A", "B"):
        cell = pd.DataFrame(
            {"error": frame[f"error_{variant}"], "quintile": quintile, "tercile": rl_tercile}
        )
        agg = cell.groupby(["tercile", "quintile"], observed=True)["error"].agg(["mean", "count"])
        for (tercile, q), row in agg.iterrows():
            rows.append(
                {
                    "export_definition": export_label,
                    "variant": variant,
                    "rl_tercile": tercile,
                    "export_quintile": int(q),
                    "mean_bias": row["mean"],
                    "n": int(row["count"]),
                }
            )
    return pd.DataFrame(rows)


def _hac_regression(frame: pd.DataFrame, export_col: str, variant: str) -> pd.DataFrame:
    """error ~ C(RL-tercile):export (own slope and intercept per tercile), HAC
    (Newey-West, 24h lag) standard errors. Design matrix built manually with
    pandas dummies rather than a patsy formula string, for direct control."""
    tercile, _ = _assign_terciles(frame["residual_load"])
    dummies = pd.get_dummies(tercile, prefix="tercile").astype(float)
    export = frame[export_col]
    interactions = dummies.multiply(export, axis=0)
    interactions.columns = [f"{c}_x_export" for c in dummies.columns]
    x = pd.concat([dummies, interactions], axis=1)
    y = frame[f"error_{variant}"]

    model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAG})
    rows = []
    for label in RL_TERCILE_LABELS:
        coef_name = f"tercile_{label}_x_export"
        rows.append(
            {
                "rl_tercile": label,
                "export_slope_eur_mwh_per_gw": model.params[coef_name] * 1000,
                "p_value": model.pvalues[coef_name],
                "n_obs": int(model.nobs),
            }
        )
    return pd.DataFrame(rows)


def _outlier_day_table(df: pd.DataFrame, daily_bias: pd.DataFrame) -> pd.DataFrame:
    """Daily mean export (total, FR) for the three outlier days, expressed as
    a percentile within that day's own trailing 90-day training window."""
    total = _total_flow(df, _SCHEDULED_PREFIX)
    fr = df["scheduled_net_de_to_fr"]
    local_day = pd.DatetimeIndex(df.index).tz_convert(LOCAL_TZ).normalize()
    daily_total = total.groupby(local_day).mean()
    daily_fr = fr.groupby(local_day).mean()

    rows = []
    for day in OUTLIER_DAYS:
        window_start = day - pd.DateOffset(days=TRAIN_SPAN_DAYS)
        window_total = daily_total.loc[window_start : day - pd.DateOffset(days=1)]
        window_fr = daily_fr.loc[window_start : day - pd.DateOffset(days=1)]
        day_total = daily_total.get(day, float("nan"))
        day_fr = daily_fr.get(day, float("nan"))
        pct_total = (
            float((window_total < day_total).mean() * 100) if len(window_total) else float("nan")
        )
        pct_fr = float((window_fr < day_fr).mean() * 100) if len(window_fr) else float("nan")
        bias_row = daily_bias.loc[daily_bias["delivery_day"] == day]
        rows.append(
            {
                "delivery_day": day,
                "export_total_mw": day_total,
                "export_total_percentile_in_train_window": pct_total,
                "export_fr_mw": day_fr,
                "export_fr_percentile_in_train_window": pct_fr,
                "bias_A": bias_row["bias_A"].iloc[0] if not bias_row.empty else float("nan"),
                "bias_B": bias_row["bias_B"].iloc[0] if not bias_row.empty else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _monthly_regression(frame: pd.DataFrame, export_col: str, variant: str) -> pd.DataFrame:
    month = pd.DatetimeIndex(frame.index).tz_convert(LOCAL_TZ).month
    autumn_mask = month.isin([9, 10, 11, 12])
    rows = []
    for season, mask in (("autumn_sep_dec", autumn_mask), ("rest_of_year", ~autumn_mask)):
        sub = frame.loc[mask]
        if sub.empty:
            continue
        reg = _hac_regression(sub, export_col, variant)
        reg["season"] = season
        rows.append(reg)
    return pd.concat(rows, ignore_index=True)


def _assess_hypothesis(
    crosstab_total: pd.DataFrame, regression_total_a: pd.DataFrame, outlier_table: pd.DataFrame
) -> dict[str, object]:
    """Spec B.5: pre-registered, programmatic check of all three criteria for
    variant A. Hypothesis is "gestuetzt" (supported) only if ALL three hold."""
    crit1_ok = True
    crit1_detail = {}
    for tercile in ("low", "mid"):
        sub = crosstab_total.loc[
            (crosstab_total["variant"] == "A") & (crosstab_total["rl_tercile"] == tercile)
        ].sort_values("export_quintile")
        biases = sub["mean_bias"].to_numpy()
        spread = biases[-1] - biases[0]
        diffs = np.diff(biases)
        n_violations = int((diffs < 0).sum())
        monotonic_enough = n_violations <= 1
        tercile_ok = monotonic_enough and spread >= MONOTONIC_SPREAD_THRESHOLD_EUR_MWH
        crit1_ok = crit1_ok and tercile_ok
        crit1_detail[tercile] = {
            "spread_eur_mwh": float(spread),
            "n_monotonicity_violations": n_violations,
            "ok": bool(tercile_ok),
        }

    crit2_ok = True
    crit2_detail = {}
    for tercile in ("low", "mid"):
        row = regression_total_a.loc[regression_total_a["rl_tercile"] == tercile].iloc[0]
        tercile_ok = bool(
            row["export_slope_eur_mwh_per_gw"] > 0 and row["p_value"] < SIGNIFICANCE_LEVEL
        )
        crit2_ok = crit2_ok and tercile_ok
        crit2_detail[tercile] = {
            "slope": float(row["export_slope_eur_mwh_per_gw"]),
            "p_value": float(row["p_value"]),
            "ok": tercile_ok,
        }

    top_quintile_count = int((outlier_table["export_total_percentile_in_train_window"] >= 80).sum())
    crit3_ok = top_quintile_count >= 2

    supported = crit1_ok and crit2_ok and crit3_ok
    return {
        "supported": supported,
        "criterion_1_monotonic_spread": {"ok": crit1_ok, "detail": crit1_detail},
        "criterion_2_significant_slope": {"ok": crit2_ok, "detail": crit2_detail},
        "criterion_3_outlier_days_top_quintile": {
            "ok": crit3_ok,
            "n_in_top_quintile": top_quintile_count,
            "n_outlier_days": len(outlier_table),
        },
    }


def _plot_heatmap(crosstab_total: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, variant in zip(axes, ("A", "B"), strict=True):
        sub = crosstab_total.loc[crosstab_total["variant"] == variant]
        pivot = sub.pivot(index="rl_tercile", columns="export_quintile", values="mean_bias")
        pivot = pivot.reindex(RL_TERCILE_LABELS)
        vmax = crosstab_total["mean_bias"].abs().max()
        im = ax.imshow(pivot, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels([f"Q{i}" for i in pivot.columns])
        ax.set_yticks(range(len(RL_TERCILE_LABELS)))
        ax.set_yticklabels(RL_TERCILE_LABELS)
        ax.set_xlabel("export quintile (Q0=import, Q4=export)")
        ax.set_title(f"variant {variant}")
        for (row, col), value in np.ndenumerate(pivot.to_numpy()):
            if not np.isnan(value):
                ax.text(col, row, f"{value:.0f}", ha="center", va="center", fontsize=8)
    axes[0].set_ylabel("residual-load tercile")
    fig.colorbar(im, ax=axes, shrink=0.8, label="mean bias (EUR/MWh)")
    fig.suptitle("Bias by export quintile x RL tercile -- ex post, not a feature")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    df = load_interim_hourly(args.data_path)
    features = load_processed_features(args.features_path)
    edge = last_complete_local_day(df["day_ahead_price"])
    logger.info("Teil-0 edge: %s", edge.date())

    cfg = FeatureConfig()
    preds = _load_extended_predictions(df, features, args.merit_order_preds_dir, edge)
    frame = _build_analysis_frame(df, features, preds, cfg.post_crisis_start, edge)
    logger.info(
        "analysis frame: %d hours, %s to %s", len(frame), frame.index.min(), frame.index.max()
    )

    rl_tercile, rl_edges = _assign_terciles(frame["residual_load"])
    logger.info("RL tercile edges (MW): %s", [round(e, 0) for e in rl_edges])

    args.out_dir.mkdir(parents=True, exist_ok=True)

    export_defs = [
        ("export_total", "total"),
        ("export_fr", "france"),
        ("export_total_lag24", "total_lag24h"),
        ("export_total_physical", "total_physical"),
    ]
    crosstabs = [_cross_tab(frame, col, label, rl_tercile) for col, label in export_defs]
    crosstab_table = pd.concat(crosstabs, ignore_index=True)
    crosstab_path = args.out_dir / "export_diagnosis_expost_crosstab.csv"
    crosstab_table.to_csv(crosstab_path, index=False)
    logger.info("wrote %s (%d rows)", crosstab_path, len(crosstab_table))

    regression_rows = []
    for col, label in (("export_total", "total"), ("export_fr", "france")):
        for variant in ("A", "B"):
            reg = _hac_regression(frame, col, variant)
            reg["export_definition"] = label
            reg["variant"] = variant
            regression_rows.append(reg)
    regression_table = pd.concat(regression_rows, ignore_index=True)
    regression_path = args.out_dir / "export_diagnosis_expost_regression.csv"
    regression_table.to_csv(regression_path, index=False)
    logger.info("wrote %s (%d rows)", regression_path, len(regression_table))

    daily_bias = _daily_bias_table(preds, frame)
    outlier_table = _outlier_day_table(df, daily_bias)
    outlier_path = args.out_dir / "export_diagnosis_expost_outlier_days.csv"
    outlier_table.to_csv(outlier_path, index=False)
    logger.info("wrote %s\n%s", outlier_path, outlier_table.to_string(index=False))

    monthly_rows = []
    for col, label in (("export_total", "total"), ("export_fr", "france")):
        for variant in ("A", "B"):
            reg = _monthly_regression(frame, col, variant)
            reg["export_definition"] = label
            reg["variant"] = variant
            monthly_rows.append(reg)
    monthly_table = pd.concat(monthly_rows, ignore_index=True)
    monthly_path = args.out_dir / "export_diagnosis_expost_monthly.csv"
    monthly_table.to_csv(monthly_path, index=False)
    logger.info("wrote %s (%d rows)", monthly_path, len(monthly_table))

    regression_total_a = regression_table.loc[
        (regression_table["export_definition"] == "total") & (regression_table["variant"] == "A")
    ]
    crosstab_total = crosstab_table.loc[crosstab_table["export_definition"] == "total"]
    assessment = _assess_hypothesis(crosstab_total, regression_total_a, outlier_table)
    logger.info("--- B.5 assessment: supported=%s ---", assessment["supported"])
    logger.info(
        "criterion 1 (monotonic spread >= %.0f): %s",
        MONOTONIC_SPREAD_THRESHOLD_EUR_MWH,
        assessment["criterion_1_monotonic_spread"],
    )
    logger.info(
        "criterion 2 (significant positive slope, p<%.2f): %s",
        SIGNIFICANCE_LEVEL,
        assessment["criterion_2_significant_slope"],
    )
    logger.info(
        "criterion 3 (>=2/3 outlier days in top export quintile): %s",
        assessment["criterion_3_outlier_days_top_quintile"],
    )
    assessment_path = args.out_dir / "export_diagnosis_expost_assessment.csv"
    pd.DataFrame(
        [
            {
                "supported": assessment["supported"],
                **{
                    f"criterion_{i}_ok": assessment[k]["ok"]
                    for i, k in enumerate(
                        (
                            "criterion_1_monotonic_spread",
                            "criterion_2_significant_slope",
                            "criterion_3_outlier_days_top_quintile",
                        ),
                        start=1,
                    )
                },
            }
        ]
    ).to_csv(assessment_path, index=False)
    logger.info("wrote %s", assessment_path)

    _plot_heatmap(crosstab_total, args.fig_path)
    logger.info("wrote figure to %s", args.fig_path)


def _daily_bias_table(preds: dict[str, pd.DataFrame], frame: pd.DataFrame) -> pd.DataFrame:
    """Daily bias A/B over the same complete-day-filtered hours as `frame`,
    for the outlier-day table (reuses Teil A.1's delivery_day convention)."""
    from energy_price_forecast.common.evaluation.metrics import bias

    valid_days = set(frame["delivery_day"].unique())
    rows = []
    for day in valid_days:
        row = {"delivery_day": day}
        for variant, p in preds.items():
            sub = p.loc[p["delivery_day"] == day]
            if not sub.empty:
                row[f"bias_{variant}"] = bias(sub["y_true"], sub["y_pred"])
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
