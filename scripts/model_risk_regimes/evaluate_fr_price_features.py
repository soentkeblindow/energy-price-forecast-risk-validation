"""Phase 2 ("Auswertung") of docs/sprint7_step7_5_spec.md's Hauptteil.

Reads the 5 cached variant predictions from measure_fr_price_features.py
(data/processed/fr_feature_preds/) and produces metrics, DM-tests, the
export cross-tab/slope cut (ex post -- not a feature), outlier-day and
ex-post diagnostics, the K1/K2 assessment, and the rolling-RMSE figure.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from energy_price_forecast.common.data.fr_prices import load_fr_day_ahead_hourly
from energy_price_forecast.common.data.loaders import load_interim_hourly, load_processed_features
from energy_price_forecast.common.evaluation.dm_test import daily_mean_loss, dm_test
from energy_price_forecast.common.evaluation.metrics import bias, mae, rmse
from energy_price_forecast.common.evaluation.walkforward import split_complete_delivery_days
from energy_price_forecast.common.features.availability import lag
from energy_price_forecast.common.features.config import FeatureConfig
from energy_price_forecast.common.features.neighbor_prices import build_fr_price_features
from energy_price_forecast.common.market_time import LOCAL_TZ

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

VARIANTS: tuple[str, ...] = ("R", "R+F3", "R+F1", "R+F2", "R+FR")
PRIMARY_CANDIDATE = "R+F3"

# Frozen reference window for the export-cut's quintile/tercile edges (spec
# section 5): the export diagnosis's own fixed period, NOT the current data
# edge -- re-used so the cells match 7.3's, not re-derived per run.
FROZEN_EDGE_START = FeatureConfig().post_crisis_start  # 2023-04-01
FROZEN_EDGE_END = pd.Timestamp("2026-10-20", tz=LOCAL_TZ)

N_RL_TERCILES = 3
N_EXPORT_QUINTILES = 5
RL_TERCILE_LABELS = ("low", "mid", "high")
HAC_LAG_EXPORT = 24

_DAILY_HAC_LAG = 7
_DAILY_HORIZON = 1
_HOURLY_HAC_LAG = 48
_HOURLY_HORIZON = 24

AUTUMN_WINTER_MONTHS = (9, 10, 11, 12)
AUTUMN_2026_START = pd.Timestamp("2026-09-01", tz=LOCAL_TZ)

OUTLIER_DAYS = (
    pd.Timestamp("2026-10-11", tz=LOCAL_TZ),
    pd.Timestamp("2026-10-18", tz=LOCAL_TZ),
    pd.Timestamp("2026-10-19", tz=LOCAL_TZ),
)

_SCHEDULED_PREFIX = "scheduled_net_de_to_"

# K1/K2 (spec section 6).
K2_RELATIVE_SHRINK_THRESHOLD = 1 / 3
SIGNIFICANCE_LEVEL = 0.05


def _load_predictions(preds_dir: Path) -> dict[str, pd.DataFrame]:
    preds = {}
    for name in VARIANTS:
        path = preds_dir / f"fr_feature_preds_{name.replace('+', '')}.parquet"
        preds[name] = pd.read_parquet(path)
    return preds


def _complete_predictions(preds: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Drop incomplete delivery days per variant (spec section 3 item 5 --
    the known 24./25.06.2026 defect days fall out here)."""
    out = {}
    for name, p in preds.items():
        complete_days, incomplete_days = split_complete_delivery_days(p)
        if len(incomplete_days):
            logger.info(
                "variant %s: %d incomplete delivery day(s) excluded: %s",
                name,
                len(incomplete_days),
                [d.date() for d in incomplete_days],
            )
        out[name] = p.loc[p["delivery_day"].isin(complete_days)]
    return out


def _period_masks(index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    local = pd.DatetimeIndex(index).tz_convert(LOCAL_TZ)
    return {
        "overall": pd.Series(True, index=index),
        "autumn_winter": pd.Series(local.month.isin(AUTUMN_WINTER_MONTHS), index=index),
        "autumn_2026": pd.Series(index >= AUTUMN_2026_START, index=index),
    }


def _metrics_table(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, p in preds.items():
        masks = _period_masks(p.index)
        for period, mask in masks.items():
            sub = p.loc[mask]
            rows.append(
                {
                    "variant": name,
                    "period": period,
                    "n": len(sub),
                    "rmse": rmse(sub["y_true"], sub["y_pred"]),
                    "mae": mae(sub["y_true"], sub["y_pred"]),
                    "bias": bias(sub["y_true"], sub["y_pred"]),
                }
            )
    return pd.DataFrame(rows)


def _dm_rows(comparison: str, period: str, loss_a: pd.Series, loss_b: pd.Series) -> list[dict]:
    aligned = pd.concat([loss_a.rename("a"), loss_b.rename("b")], axis=1).dropna()
    if len(aligned) < 2 * _HOURLY_HAC_LAG:
        return []
    daily_a, _ = daily_mean_loss(aligned["a"])
    daily_b, _ = daily_mean_loss(aligned["b"])
    daily_result = dm_test(daily_a, daily_b, hac_lag=_DAILY_HAC_LAG, horizon=_DAILY_HORIZON)
    hourly_result = dm_test(
        aligned["a"], aligned["b"], hac_lag=_HOURLY_HAC_LAG, horizon=_HOURLY_HORIZON
    )
    rows = []
    for block, result in (("daily", daily_result), ("hourly", hourly_result)):
        rows.append(
            {
                "comparison": comparison,
                "period": period,
                "block": block,
                "n_obs": result.n_obs,
                "hac_lag": result.hac_lag,
                "mean_loss_diff": result.mean_loss_diff,
                "dm_stat": result.dm_stat,
                "p_value": result.p_value,
            }
        )
    return rows


def _dm_table(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Each variant vs. R, absolute-error loss, all 3 periods (spec section 5)."""
    r = preds["R"]
    r_loss = (r["y_true"] - r["y_pred"]).abs()
    rows = []
    for name in VARIANTS:
        if name == "R":
            continue
        p = preds[name]
        p_loss = (p["y_true"] - p["y_pred"]).abs()
        masks = _period_masks(p_loss.index)
        for period, mask in masks.items():
            rows += _dm_rows(f"{name}_vs_R", period, p_loss.loc[mask], r_loss.loc[mask])
    return pd.DataFrame(rows)


def _frozen_rl_tercile_edges(features: pd.DataFrame) -> np.ndarray:
    rl = features["residual_load_forecast"]
    rl = rl.loc[(rl.index >= FROZEN_EDGE_START) & (rl.index <= FROZEN_EDGE_END)].dropna()
    _, edges = pd.qcut(rl, N_RL_TERCILES, labels=RL_TERCILE_LABELS, retbins=True)
    return np.asarray(edges)


def _frozen_export_quintile_edges(series: pd.Series) -> np.ndarray:
    sub = series.loc[(series.index >= FROZEN_EDGE_START) & (series.index <= FROZEN_EDGE_END)]
    sub = sub.dropna()
    _, edges = pd.qcut(sub, N_EXPORT_QUINTILES, labels=False, retbins=True, duplicates="drop")
    return np.asarray(edges)


def _total_export(df: pd.DataFrame) -> pd.Series:
    cols = sorted(c for c in df.columns if c.startswith(_SCHEDULED_PREFIX))
    return df[cols].sum(axis=1)


def _analysis_frame(
    preds: dict[str, pd.DataFrame],
    features: pd.DataFrame,
    df: pd.DataFrame,
    rl_edges: np.ndarray,
    export_total_edges: np.ndarray,
    export_fr_edges: np.ndarray,
) -> pd.DataFrame:
    """One row per hour common to all variants' complete days; error per
    variant, RL tercile + export quintiles via the FROZEN edges."""
    common_days = set.intersection(*[set(p["delivery_day"].unique()) for p in preds.values()])
    index = preds["R"].index
    frame = pd.DataFrame(index=index)
    for name, p in preds.items():
        sub = p.loc[p["delivery_day"].isin(common_days)]
        frame[f"error_{name}"] = (sub["y_true"] - sub["y_pred"]).reindex(frame.index)

    frame["residual_load"] = features["residual_load_forecast"].reindex(frame.index)
    frame["export_total"] = _total_export(df).reindex(frame.index)
    frame["export_fr"] = df["scheduled_net_de_to_fr"].reindex(frame.index)
    frame["rl_tercile"] = pd.cut(frame["residual_load"], bins=rl_edges, labels=RL_TERCILE_LABELS)
    frame["export_total_quintile"] = pd.cut(
        frame["export_total"], bins=export_total_edges, labels=False
    )
    frame["export_fr_quintile"] = pd.cut(frame["export_fr"], bins=export_fr_edges, labels=False)
    frame["delivery_day"] = pd.DatetimeIndex(frame.index).tz_convert(LOCAL_TZ).normalize()
    error_cols = [f"error_{name}" for name in preds]
    return frame.dropna(subset=error_cols)


def _crosstab(frame: pd.DataFrame, export_col: str, export_label: str) -> pd.DataFrame:
    rows = []
    for name in VARIANTS:
        cell = frame[[f"error_{name}", "rl_tercile", export_col]].rename(
            columns={f"error_{name}": "error", export_col: "quintile"}
        )
        agg = cell.groupby(["rl_tercile", "quintile"], observed=True)["error"].agg(
            ["mean", "count"]
        )
        for (tercile, q), row in agg.iterrows():
            rows.append(
                {
                    "export_definition": export_label,
                    "variant": name,
                    "rl_tercile": tercile,
                    "export_quintile": int(q),
                    "mean_bias": row["mean"],
                    "n": int(row["count"]),
                }
            )
    return pd.DataFrame(rows)


def _hac_slope(frame: pd.DataFrame, export_col: str, variant: str) -> pd.DataFrame:
    dummies = pd.get_dummies(frame["rl_tercile"], prefix="tercile").astype(float)
    export = frame[export_col]
    interactions = dummies.multiply(export, axis=0)
    interactions.columns = [f"{c}_x_export" for c in dummies.columns]
    x = pd.concat([dummies, interactions], axis=1)
    y = frame[f"error_{variant}"]
    model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAG_EXPORT})
    rows = []
    for label in RL_TERCILE_LABELS:
        coef_name = f"tercile_{label}_x_export"
        rows.append(
            {
                "variant": variant,
                "rl_tercile": label,
                "export_slope_eur_mwh_per_gw": model.params[coef_name] * 1000,
                "p_value": model.pvalues[coef_name],
                "n_obs": int(model.nobs),
            }
        )
    return pd.DataFrame(rows)


def _slopes_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for export_col, label in (("export_total", "total"), ("export_fr", "france")):
        for name in VARIANTS:
            reg = _hac_slope(frame, export_col, name)
            reg["export_definition"] = label
            rows.append(reg)
    return pd.concat(rows, ignore_index=True)


def _outlier_day_table(preds: dict[str, pd.DataFrame], f3: pd.Series) -> pd.DataFrame:
    rows = []
    for day in OUTLIER_DAYS:
        row: dict[str, object] = {"delivery_day": day}
        window_start = day - pd.DateOffset(days=90)
        window_f3 = f3.loc[window_start : day - pd.DateOffset(hours=1)]
        day_f3 = f3.reindex(pd.date_range(day, periods=24, freq="h", tz=LOCAL_TZ)).mean()
        row["f3_spread_mean"] = float(day_f3)
        row["f3_percentile_in_train_window"] = (
            float((window_f3 < day_f3).mean() * 100) if len(window_f3) else float("nan")
        )
        for name, p in preds.items():
            sub = p.loc[p["delivery_day"] == day]
            if not sub.empty:
                row[f"bias_{name}"] = bias(sub["y_true"], sub["y_pred"])
                row[f"rmse_{name}"] = rmse(sub["y_true"], sub["y_pred"])
        rows.append(row)
    return pd.DataFrame(rows)


def _expost_fr_diagnosis(
    r_preds: pd.DataFrame, fr_price: pd.Series, features: pd.DataFrame
) -> pd.DataFrame:
    """Upper bound (spec section 5, 'Diagnose ohne Feature'): regress R's
    error on FR-Preis(D) - Prognose R(D), same-hour (ex post, not lagged),
    per RL tercile, HAC-24."""
    frame = pd.DataFrame(index=r_preds.index)
    frame["error"] = r_preds["y_true"] - r_preds["y_pred"]
    frame["fr_minus_pred"] = fr_price.reindex(frame.index) - r_preds["y_pred"]
    frame["residual_load"] = features["residual_load_forecast"].reindex(frame.index)
    frame = frame.dropna()
    rl_edges = _frozen_rl_tercile_edges(features)
    frame["rl_tercile"] = pd.cut(frame["residual_load"], bins=rl_edges, labels=RL_TERCILE_LABELS)

    rows = []
    for label in RL_TERCILE_LABELS:
        sub = frame.loc[frame["rl_tercile"] == label]
        if len(sub) < 2 * HAC_LAG_EXPORT:
            continue
        x = sm.add_constant(sub["fr_minus_pred"])
        model = sm.OLS(sub["error"], x).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAG_EXPORT})
        rows.append(
            {
                "rl_tercile": label,
                "slope": model.params["fr_minus_pred"],
                "p_value": model.pvalues["fr_minus_pred"],
                "n_obs": int(model.nobs),
            }
        )
    return pd.DataFrame(rows)


def _assess_k1_k2(metrics: pd.DataFrame, dm: pd.DataFrame, slopes: pd.DataFrame) -> dict:
    r_rmse = metrics.loc[
        (metrics["variant"] == "R") & (metrics["period"] == "overall"), "rmse"
    ].iloc[0]
    cand_rmse = metrics.loc[
        (metrics["variant"] == PRIMARY_CANDIDATE) & (metrics["period"] == "overall"), "rmse"
    ].iloc[0]
    dm_row = dm.loc[
        (dm["comparison"] == f"{PRIMARY_CANDIDATE}_vs_R")
        & (dm["period"] == "overall")
        & (dm["block"] == "daily")
    ]
    dm_p = float(dm_row["p_value"].iloc[0]) if len(dm_row) else float("nan")
    k1_ok = bool(cand_rmse < r_rmse and dm_p < SIGNIFICANCE_LEVEL)

    k2_detail = {}
    k2_ok = True
    for tercile in ("low", "mid"):
        r_slope = slopes.loc[
            (slopes["variant"] == "R")
            & (slopes["export_definition"] == "total")
            & (slopes["rl_tercile"] == tercile),
            "export_slope_eur_mwh_per_gw",
        ].iloc[0]
        cand_slope = slopes.loc[
            (slopes["variant"] == PRIMARY_CANDIDATE)
            & (slopes["export_definition"] == "total")
            & (slopes["rl_tercile"] == tercile),
            "export_slope_eur_mwh_per_gw",
        ].iloc[0]
        shrink = 1 - abs(cand_slope) / abs(r_slope) if r_slope != 0 else float("nan")
        tercile_ok = bool(shrink >= K2_RELATIVE_SHRINK_THRESHOLD)
        k2_ok = k2_ok and tercile_ok
        k2_detail[tercile] = {
            "r_slope": float(r_slope),
            "candidate_slope": float(cand_slope),
            "relative_shrink": float(shrink),
            "ok": tercile_ok,
        }

    significantly_worse = bool(cand_rmse > r_rmse and dm_p < SIGNIFICANCE_LEVEL)
    if significantly_worse:
        verdict = "traegt nicht (signifikant schlechter als R)"
    elif k1_ok and k2_ok:
        verdict = "traegt"
    elif k1_ok or k2_ok:
        verdict = "traegt teilweise"
    else:
        verdict = "traegt nicht"

    return {
        "k1_ok": k1_ok,
        "k1_rmse_r": float(r_rmse),
        "k1_rmse_candidate": float(cand_rmse),
        "k1_dm_p_value": dm_p,
        "k2_ok": k2_ok,
        "k2_detail": k2_detail,
        "significantly_worse_than_r": significantly_worse,
        "verdict": verdict,
    }


def _rolling_rmse_figure(preds: dict[str, pd.DataFrame], f3: pd.Series, out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(11, 5))
    for name, color in (("R", "tab:blue"), (PRIMARY_CANDIDATE, "tab:orange")):
        p = preds[name]
        daily_sq_err = ((p["y_true"] - p["y_pred"]) ** 2).groupby(p["delivery_day"]).mean()
        rolling_rmse = np.sqrt(daily_sq_err.rolling(30, min_periods=30).mean())
        ax1.plot(rolling_rmse.index, rolling_rmse.to_numpy(), label=name, color=color)
    ax1.set_ylabel("rolling 30-day RMSE (EUR/MWh)")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    daily_f3 = f3.groupby(pd.DatetimeIndex(f3.index).tz_convert(LOCAL_TZ).normalize()).mean()
    ax2.plot(daily_f3.index, daily_f3.to_numpy(), color="grey", alpha=0.5, label="FR-DE spread")
    ax2.set_ylabel("FR-DE price spread, 24h-lagged (EUR/MWh)")
    fig.suptitle(f"Rolling RMSE: R vs. {PRIMARY_CANDIDATE}, with FR-DE spread")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-path", type=Path, default=Path("data/interim/hourly_merit_order.parquet")
    )
    p.add_argument(
        "--features-path", type=Path, default=Path("data/processed/features_merit_order.parquet")
    )
    p.add_argument("--fr-path", type=Path, default=Path("data/interim/fr_day_ahead_hourly.parquet"))
    p.add_argument("--preds-dir", type=Path, default=Path("data/processed/fr_feature_preds"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/results"))
    p.add_argument(
        "--fig-path",
        type=Path,
        default=Path("outputs/assets/fr_price_features_rolling_rmse.png"),
    )
    return p.parse_args()


def _f3_spread(df: pd.DataFrame, fr_price: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """spread_fr_de_lag_24h over target_index, built the exact same way the
    R+F3/R+FR feature matrices were (common/features/neighbor_prices.py)."""
    de_price_lag_24h = lag(
        "price_lag_24h",
        df["day_ahead_price"],
        "day_ahead_price",
        hours=24,
        target_index=target_index,
    )
    _, _, f3 = build_fr_price_features(fr_price, de_price_lag_24h, target_index)
    return f3.values


def main() -> None:
    args = _parse_args()
    df = load_interim_hourly(args.data_path)
    features = load_processed_features(args.features_path)
    fr_price = load_fr_day_ahead_hourly(args.fr_path)["fr_day_ahead_price"]

    raw_preds = _load_predictions(args.preds_dir)
    preds = _complete_predictions(raw_preds)
    f3_spread = _f3_spread(df, fr_price, preds["R"].index)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    metrics = _metrics_table(preds)
    metrics_path = args.out_dir / "fr_price_features_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    logger.info("wrote %s (%d rows)", metrics_path, len(metrics))

    dm = _dm_table(preds)
    dm_path = args.out_dir / "fr_price_features_dm_test.csv"
    dm.to_csv(dm_path, index=False)
    logger.info("wrote %s (%d rows)", dm_path, len(dm))

    rl_edges = _frozen_rl_tercile_edges(features)
    export_total_edges = _frozen_export_quintile_edges(_total_export(df))
    export_fr_edges = _frozen_export_quintile_edges(df["scheduled_net_de_to_fr"])
    logger.info("frozen RL tercile edges (MW): %s", [round(float(e), 0) for e in rl_edges])

    frame = _analysis_frame(preds, features, df, rl_edges, export_total_edges, export_fr_edges)
    crosstab = pd.concat(
        [
            _crosstab(frame, "export_total_quintile", "total"),
            _crosstab(frame, "export_fr_quintile", "france"),
        ],
        ignore_index=True,
    )
    crosstab_path = args.out_dir / "fr_price_features_expost_crosstab.csv"
    crosstab.to_csv(crosstab_path, index=False)
    logger.info("wrote %s (%d rows) -- ex post, not a feature", crosstab_path, len(crosstab))

    slopes = _slopes_table(frame)
    slopes_path = args.out_dir / "fr_price_features_expost_slopes.csv"
    slopes.to_csv(slopes_path, index=False)
    logger.info("wrote %s (%d rows) -- ex post, not a feature", slopes_path, len(slopes))

    outlier_table = _outlier_day_table(preds, f3_spread)
    outlier_path = args.out_dir / "fr_price_features_outlier_days.csv"
    outlier_table.to_csv(outlier_path, index=False)
    logger.info("wrote %s\n%s", outlier_path, outlier_table.to_string(index=False))

    expost_diag = _expost_fr_diagnosis(preds["R"], fr_price, features)
    expost_diag_path = args.out_dir / "fr_price_features_expost_fr_diagnosis.csv"
    expost_diag.to_csv(expost_diag_path, index=False)
    logger.info(
        "wrote %s -- ex post, not a feature (upper bound)\n%s",
        expost_diag_path,
        expost_diag.to_string(index=False),
    )

    assessment = _assess_k1_k2(metrics, dm, slopes)
    logger.info(
        "--- K1/K2 assessment for %s vs. R: %s ---", PRIMARY_CANDIDATE, assessment["verdict"]
    )
    logger.info("K1 (RMSE lower + daily DM p<0.05): %s", assessment["k1_ok"])
    logger.info("K2 (export slope shrink >=1/3, low+mid tercile): %s", assessment["k2_ok"])
    assessment_path = args.out_dir / "fr_price_features_assessment.csv"
    pd.json_normalize(assessment, sep="_").to_csv(assessment_path, index=False)
    logger.info("wrote %s", assessment_path)

    _rolling_rmse_figure(preds, f3_spread, args.fig_path)
    logger.info("wrote figure to %s", args.fig_path)


if __name__ == "__main__":
    main()
