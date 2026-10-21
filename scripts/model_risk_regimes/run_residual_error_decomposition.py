"""Teil 2 ("Restfehler im Modell") of
docs/residual_error_decomposition_hourly_spec.md.

Three analyses, all on top of already-cached predictions from 7.2 (Phase 2)
and 7.3 Teil 1 (the ladder) -- no new backtest campaigns, matching the
spec's "Dafuer sind keine neuen Laeufe noetig" (beyond a 2-day extension of
7.2's long-history cache to the Teil-0 edge, which is cheap enough to redo
on every run instead of persisting as its own cache file):

1. Seasonal bias (year x month) for A/B over the full 2019-to-edge history.
2. Same seasonal bias for the O_ALL oracle rung, over the ladder's
   --window-start-floored 2026 window (2026-01-04 to 2026-10-19) -- the
   primary (2024-06-15 to 2026-05-31) ladder cache was deleted during the
   oracle-fix rerun and not regenerated (a full 15-run/611-fold rerun was
   judged not worth the ~2-2.5h cost), so this covers 2026 only, not the
   full ladder history.
3. Objective function (quantile vs l2) for L0/L2, A/B: reuses
   scripts/model_post_crisis_resolution/compare_objectives.py UNCHANGED via
   subprocess (not an import -- importing a model_post_crisis_resolution
   script from a model_risk_regimes one would violate
   tests/test_dependency_rules.py), plus a small supplementary bias-only
   table since that script's own metrics (mae/rmse/wape) do not include bias.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from energy_price_forecast.common.data.loaders import load_interim_hourly, load_processed_features
from energy_price_forecast.common.evaluation.metrics import bias, mae, rmse
from energy_price_forecast.common.evaluation.walkforward import (
    extend_predictions_to_edge,
    last_complete_local_day,
)
from energy_price_forecast.common.features.merit_order import build_merit_order_frame
from energy_price_forecast.common.market_time import LOCAL_TZ
from energy_price_forecast.common.models.lgbm import LGBMForecaster
from energy_price_forecast.common.models.merit_order_forecaster import MeritOrderForecaster

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TRAIN_SPAN_DAYS = 90

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPARE_OBJECTIVES_SCRIPT = (
    _REPO_ROOT / "scripts" / "model_post_crisis_resolution" / "compare_objectives.py"
)
_OBJECTIVE_RUNGS_VARIANTS = (("L0", "A"), ("L0", "B"), ("L2", "A"), ("L2", "B"))


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
    p.add_argument(
        "--ladder-preds-dir-2026",
        type=Path,
        default=Path("data/processed/ladder_preds_2026"),
        help="Teil 1's 2026-window (2026-01-04 to 2026-10-19) ladder cache -- the "
        "primary 611-day window's cache was deleted during the oracle-fix rerun "
        "and not regenerated (owner's call: not worth the ~2-2.5h), so points 2 "
        "and 3 use this window only.",
    )
    p.add_argument("--out-dir", type=Path, default=Path("outputs/results"))
    p.add_argument(
        "--fig-path",
        type=Path,
        default=Path("outputs/assets/error_decomposition_seasonal_bias_heatmap.png"),
    )
    return p.parse_args()


# --- Point 1: seasonal bias, A/B, full long history --------------------------


def _monthly_metrics(preds: pd.DataFrame, variant_name: str) -> pd.DataFrame:
    """Bias/RMSE/MAE per (local calendar year, month), one row per cell."""
    local_day = pd.DatetimeIndex(preds["delivery_day"]).tz_convert(LOCAL_TZ)
    frame = preds.assign(year=local_day.year, month=local_day.month)
    rows = []
    for (year, month), group in frame.groupby(["year", "month"]):
        rows.append(
            {
                "variant": variant_name,
                "year": int(year),
                "month": int(month),
                "n": len(group),
                "bias": bias(group["y_true"], group["y_pred"]),
                "rmse": rmse(group["y_true"], group["y_pred"]),
                "mae": mae(group["y_true"], group["y_pred"]),
            }
        )
    return pd.DataFrame(rows)


def _seasonal_bias_full_history(
    df: pd.DataFrame, features: pd.DataFrame, preds_dir: Path, edge: pd.Timestamp
) -> pd.DataFrame:
    price = df["day_ahead_price"]
    y = price.reindex(features.index)
    x = features
    long_history = build_merit_order_frame(df, pd.DatetimeIndex(df.index))

    models = {
        "A": LGBMForecaster(objective="quantile", alpha=0.5),
        "B": MeritOrderForecaster(long_history, srmc_definition="gas_only"),
    }
    tables = []
    for name, model in models.items():
        cache_path = preds_dir / f"merit_order_preds_{name}.parquet"
        preds = pd.read_parquet(cache_path)
        preds = extend_predictions_to_edge(
            preds, model, x, y, edge, train_span_days=TRAIN_SPAN_DAYS, label=f"variant {name}"
        )
        tables.append(_monthly_metrics(preds, name))
    return pd.concat(tables, ignore_index=True)


# --- Point 2: seasonal bias, O_ALL oracle rung, 2026 ladder window ----------
#
# The primary (611-day, 2024-06-15 to 2026-05-31) ladder cache was deleted
# while re-running Teil 1 after the oracle-actuals column fix and was not
# regenerated (a full 15-run/611-fold rerun was judged not worth the ~2-2.5h
# cost -- owner's call, 2026-10-21). Point 2 therefore covers only the
# --window-start-floored 2026 run's own window (2026-01-04 to 2026-10-19),
# not the full 2024-06-to-edge range the original spec/plan intended.


def _seasonal_bias_oracle(ladder_dir_2026: Path) -> pd.DataFrame:
    tables = []
    for variant in ("A", "B"):
        preds = pd.read_parquet(ladder_dir_2026 / f"ladder_preds_O_ALL_{variant}.parquet")
        tables.append(_monthly_metrics(preds, variant))
    return pd.concat(tables, ignore_index=True)


# --- Combined figure -----------------------------------------------------------


def _plot_seasonal_heatmaps(
    full_history_table: pd.DataFrame, oracle_table: pd.DataFrame, out_path: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=False)
    panels = [
        ("A", full_history_table, "A (full history)", axes[0, 0]),
        ("B", full_history_table, "B (full history)", axes[0, 1]),
        ("A", oracle_table, "A (O_ALL, ORACLE - no forecast)", axes[1, 0]),
        ("B", oracle_table, "B (O_ALL, ORACLE - no forecast)", axes[1, 1]),
    ]
    vmax = max(
        table.loc[table["variant"] == variant, "bias"].abs().max()
        for variant, table, _, _ in panels
    )
    im = None
    for variant, table, title, ax in panels:
        subset = table.loc[table["variant"] == variant]
        pivot = subset.pivot(index="year", columns="month", values="bias")
        pivot = pivot.reindex(columns=range(1, 13))
        im = ax.imshow(pivot, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(12))
        ax.set_xticklabels(range(1, 13))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(title)
        ax.set_xlabel("month")
        ax.set_ylabel("year")
        for (row, col), value in np.ndenumerate(pivot.to_numpy()):
            if not np.isnan(value):
                ax.text(col, row, f"{value:.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.8, label="bias (EUR/MWh)")
    fig.suptitle("Seasonal bias (year x month): full history vs. oracle fundamentals")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --- Point 3: objective function (quantile vs l2) -----------------------------


def _run_compare_objectives(
    rung: str, variant: str, ladder_dir: Path, interim_path: Path, out_dir: Path
) -> None:
    median_path = ladder_dir / f"ladder_preds_{rung}_{variant}.parquet"
    mean_path = ladder_dir / f"ladder_preds_{rung}_{variant}_l2obj.parquet"
    out_path = out_dir / f"error_decomposition_objective_{rung.lower()}_{variant.lower()}.csv"
    dm_out_path = out_dir / f"error_decomposition_objective_dm_{rung.lower()}_{variant.lower()}.csv"
    cmd = [
        sys.executable,
        str(_COMPARE_OBJECTIVES_SCRIPT),
        "--median-path",
        str(median_path),
        "--mean-path",
        str(mean_path),
        "--interim-path",
        str(interim_path),
        "--out",
        str(out_path),
        "--dm-out",
        str(dm_out_path),
    ]
    logger.info("running compare_objectives.py for %s_%s", rung, variant)
    subprocess.run(cmd, check=True, cwd=_REPO_ROOT)


def _objective_bias_summary(ladder_dir: Path, out_dir: Path) -> pd.DataFrame:
    """Bias-only comparison (median/quantile vs l2), since compare_objectives.py's
    own metrics (breakdown_point) do not include bias -- kept as a separate,
    small supplementary table rather than modifying that frozen script.
    """
    rows = []
    for rung, variant in _OBJECTIVE_RUNGS_VARIANTS:
        median_preds = pd.read_parquet(ladder_dir / f"ladder_preds_{rung}_{variant}.parquet")
        l2_preds = pd.read_parquet(ladder_dir / f"ladder_preds_{rung}_{variant}_l2obj.parquet")
        periods = {
            "overall": (median_preds["delivery_day"].min(), median_preds["delivery_day"].max()),
            "autumn_2026": (
                pd.Timestamp("2026-09-01", tz=LOCAL_TZ),
                median_preds["delivery_day"].max(),
            ),
        }
        for period_name, (start, end) in periods.items():
            median_mask = (median_preds["delivery_day"] >= start) & (
                median_preds["delivery_day"] <= end
            )
            l2_mask = (l2_preds["delivery_day"] >= start) & (l2_preds["delivery_day"] <= end)
            median_sub = median_preds.loc[median_mask]
            l2_sub = l2_preds.loc[l2_mask]
            if median_sub.empty or l2_sub.empty:
                continue
            bias_median = bias(median_sub["y_true"], median_sub["y_pred"])
            bias_l2 = bias(l2_sub["y_true"], l2_sub["y_pred"])
            rows.append(
                {
                    "rung": rung,
                    "variant": variant,
                    "period": period_name,
                    "bias_median": bias_median,
                    "bias_l2": bias_l2,
                    "bias_delta": bias_l2 - bias_median,
                }
            )
    table = pd.DataFrame(rows)
    out_path = out_dir / "error_decomposition_objective_bias_summary.csv"
    table.to_csv(out_path, index=False)
    logger.info("wrote %s (%d rows)", out_path, len(table))
    return table


def main() -> None:
    args = _parse_args()
    t0 = time.monotonic()

    df = load_interim_hourly(args.data_path)
    features = load_processed_features(args.features_path)
    edge = last_complete_local_day(df["day_ahead_price"])
    logger.info("Teil-0 edge (last locally-complete delivery day): %s", edge.date())

    args.out_dir.mkdir(parents=True, exist_ok=True)

    full_history_table = _seasonal_bias_full_history(df, features, args.merit_order_preds_dir, edge)
    full_history_out = args.out_dir / "error_decomposition_seasonal_bias_by_month.csv"
    full_history_table.to_csv(full_history_out, index=False)
    logger.info("wrote %s (%d rows)", full_history_out, len(full_history_table))

    oracle_table = _seasonal_bias_oracle(args.ladder_preds_dir_2026)
    oracle_out = args.out_dir / "error_decomposition_oracle_seasonal_bias_by_month.csv"
    oracle_table.to_csv(oracle_out, index=False)
    logger.info("wrote %s (%d rows)", oracle_out, len(oracle_table))

    _plot_seasonal_heatmaps(full_history_table, oracle_table, args.fig_path)
    logger.info("wrote figure to %s", args.fig_path)

    for rung, variant in _OBJECTIVE_RUNGS_VARIANTS:
        _run_compare_objectives(
            rung, variant, args.ladder_preds_dir_2026, args.data_path, args.out_dir
        )
    _objective_bias_summary(args.ladder_preds_dir_2026, args.out_dir)

    logger.info("done in %.1fs", time.monotonic() - t0)


if __name__ == "__main__":
    main()
