"""Sprint 7 Step 7.6, Teil B (docs/sprint7_step7_6_spec.md, sections 3-4).

Compares three price-model-selection candidates -- B (srmc_gas_only), B-full
(srmc_gas_full, EUA also a tree feature), B-roll (3-year rolling beta
window) -- against the historical reference A (plain LightGBM, no
target-transform), on an identical fold list, L1 feature matrix (core_gas
with ENTSO-E forecasts), 2023-05-01 through the data edge. Purely
descriptive: no winner is picked here. The owner decides the SRMC
definition, beta window, beta bounds and segmentation afterwards, based on
this evidence (docs/sprint7_step7_6_spec.md section 1).

Three comparisons:
  V1: B-full vs B      -- CO2 and the younger (post-crisis) beta pool together
  V2: B-roll vs B      -- the beta window alone
  V3: B-full vs B-roll -- CO2 alone, restricted to the anchor window where
                          both variants' beta pools contain the exact same
                          historical hours (see _v3_pool_window)
Plus each candidate vs A over the whole period, for context only.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from build_beta_monthly_table import build_beta_monthly_table

from energy_price_forecast.common.data.loaders import load_interim_hourly, load_processed_features
from energy_price_forecast.common.evaluation.dm_test import daily_mean_loss, dm_test
from energy_price_forecast.common.evaluation.metrics import bias, mae, rmse
from energy_price_forecast.common.evaluation.walkforward import (
    Fold,
    Forecaster,
    last_complete_local_day,
    run_backtest,
    split_complete_delivery_days,
    walk_forward_splits,
)
from energy_price_forecast.common.features.config import FeatureConfig
from energy_price_forecast.common.features.core_gas_columns import L1_COLUMNS
from energy_price_forecast.common.features.merit_order import (
    GAS_EFFICIENCY,
    build_merit_order_frame,
    raw_beta_per_bin,
    rl_bin,
    ttf_out_of_range_flag,
)
from energy_price_forecast.common.market_time import LOCAL_TZ
from energy_price_forecast.common.models.lgbm import LGBMForecaster
from energy_price_forecast.common.models.merit_order_forecaster import MeritOrderForecaster

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TRAIN_SPAN_DAYS = 90
TEST_START = "2023-05-01"  # spec B.2
BETA_ROLL_WINDOW_DAYS = 3 * 365

RUNTIME_BUDGET_HOURS = 2.0  # spec B.5
PROBE_FOLDS = 20

UPPER_THIRD_BINS = (8, 9, 10)  # same definition as 7.2/7.3

_DAILY_HAC_LAG = 7
_DAILY_HORIZON = 1
_HOURLY_HAC_LAG = 48
_HOURLY_HORIZON = 24

# Trial range to bracket the true V3 anchor window empirically (spec's own
# estimate is "about 2024-09 to 2026-04") -- wide enough to find both edges
# even if the real boundary differs from that estimate.
V3_TRIAL_ANCHORS_FIRST = pd.Timestamp("2024-01-01", tz=LOCAL_TZ)


def _month_anchors(first: pd.Timestamp, last_local_day: pd.Timestamp) -> list[pd.Timestamp]:
    last = pd.Timestamp(year=last_local_day.year, month=last_local_day.month, day=1, tz=LOCAL_TZ)
    anchors = []
    current = first
    while current <= last:
        anchors.append(current)
        current = (current + pd.DateOffset(months=1)).normalize()
    return anchors


def _v3_pool_window(
    long_history: pd.DataFrame, edge: pd.Timestamp
) -> tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]:
    """Empirically find the anchor range where B-full's and B-roll's beta
    pools contain the exact same historical hours per bin (spec B.4):
    compares ``raw_beta_per_bin``'s ``n`` column for both configs at every
    trial anchor, rather than trusting the spec's own closed-form estimate.

    Returns (first_matching_anchor, last_matching_anchor, per-anchor
    n-comparison table) for the log.
    """
    anchors = _month_anchors(V3_TRIAL_ANCHORS_FIRST, edge)
    rows = []
    for anchor in anchors:
        n_full = raw_beta_per_bin(
            long_history,
            anchor,
            srmc_definition="gas_full",
            rl_measure="absolute",
            window_days=None,
        )["n"]
        n_roll = raw_beta_per_bin(
            long_history,
            anchor,
            srmc_definition="gas_only",
            rl_measure="absolute",
            window_days=BETA_ROLL_WINDOW_DAYS,
        )["n"]
        matches = bool((n_full == n_roll).all())
        rows.append(
            {
                "anchor_month": anchor.strftime("%Y-%m"),
                "n_full_total": int(n_full.sum()),
                "n_roll_total": int(n_roll.sum()),
                "pools_match": matches,
                "max_bin_diff": int((n_full - n_roll).abs().max()),
            }
        )
    table = pd.DataFrame(rows)
    matching = table.loc[table["pools_match"]]
    if matching.empty:
        raise ValueError("no anchor in the trial range has matching B-full/B-roll beta pools")
    first_match = pd.Timestamp(matching["anchor_month"].iloc[0] + "-01", tz=LOCAL_TZ)
    last_match = pd.Timestamp(matching["anchor_month"].iloc[-1] + "-01", tz=LOCAL_TZ)
    return first_match, last_match, table


def _in_period(
    delivery_day: pd.Series,
    period: str,
    *,
    test_start: pd.Timestamp,
    edge: pd.Timestamp,
    v3_start: pd.Timestamp,
    v3_end: pd.Timestamp,
) -> pd.Series:
    """Boolean mask over ``delivery_day`` for a named period. Unlike a simple
    ``{name: (start, end)}`` dict, ``autumn_winter`` is a recurring
    month-filter (every September-December in range), not one contiguous
    span -- so this is a function, not a static table.
    """
    autumn_2026_start = pd.Timestamp("2026-09-01", tz=LOCAL_TZ)
    v3_mask = (delivery_day >= v3_start) & (delivery_day <= v3_end)
    if period == "overall":
        return (delivery_day >= test_start) & (delivery_day <= edge)
    if period == "v3_window":
        return v3_mask
    if period == "autumn_winter":
        return delivery_day.dt.month.isin([9, 10, 11, 12])
    if period == "autumn_2026":
        return delivery_day >= autumn_2026_start
    if period == "v3_autumn_winter":
        return v3_mask & delivery_day.dt.month.isin([9, 10, 11, 12])
    if period == "v3_autumn_2026":
        return v3_mask & (delivery_day >= autumn_2026_start)
    raise ValueError(f"unknown period {period!r}")


def _get_or_run_backtest(
    name: str, model: Forecaster, folds: list[Fold], x: pd.DataFrame, y: pd.Series, preds_dir: Path
) -> pd.DataFrame:
    cache_path = preds_dir / f"price_model_selection_preds_{name}.parquet"
    if cache_path.exists():
        logger.info("variant %s: loaded cached predictions from %s", name, cache_path)
        return pd.read_parquet(cache_path)
    t0 = time.monotonic()
    preds = run_backtest(y, model, folds, refit_every=1, x=x)
    elapsed = time.monotonic() - t0
    logger.info(
        "variant %s: %d test hours in %.1fs (%.3fs/fold)",
        name,
        len(preds),
        elapsed,
        elapsed / len(folds) if folds else float("nan"),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(cache_path)
    return preds


def _metrics_row(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    aligned = pd.concat([y_true.rename("t"), y_pred.rename("p")], axis=1).dropna()
    return {
        "n": len(aligned),
        "rmse": rmse(aligned["t"], aligned["p"]),
        "mae": mae(aligned["t"], aligned["p"]),
        "bias": bias(aligned["t"], aligned["p"]),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-path", type=Path, default=Path("data/interim/hourly_merit_order.parquet")
    )
    p.add_argument(
        "--features-path", type=Path, default=Path("data/processed/features_merit_order.parquet")
    )
    p.add_argument("--out-dir", type=Path, default=Path("outputs/results"))
    p.add_argument(
        "--fig-path", type=Path, default=Path("outputs/assets/price_model_selection_beta.png")
    )
    p.add_argument(
        "--preds-dir", type=Path, default=Path("data/processed/price_model_selection_preds")
    )
    p.add_argument("--runtime-budget-hours", type=float, default=RUNTIME_BUDGET_HOURS)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    df = load_interim_hourly(args.data_path)
    price = df["day_ahead_price"]
    features = load_processed_features(args.features_path)
    y = price.reindex(features.index)
    index = pd.DatetimeIndex(features.index)

    l1_matrix = features[list(L1_COLUMNS)]
    l1_matrix_full = l1_matrix.assign(eua_co2_lag_48h=features["eua_co2_lag_48h"])
    long_history = build_merit_order_frame(df, pd.DatetimeIndex(df.index))

    edge = last_complete_local_day(price)
    logger.info("data edge (last locally-complete delivery day): %s", edge.date())

    folds = list(
        walk_forward_splits(
            index,
            test_start=TEST_START,
            test_end=edge.strftime("%Y-%m-%d"),
            window="rolling",
            train_span_days=TRAIN_SPAN_DAYS,
        )
    )
    if not folds:
        raise ValueError("no folds produced -- check the data range against TEST_START")
    test_start = folds[0].delivery_day
    logger.info(
        "identical fold list for all 4 variants: %d folds, %s to %s",
        len(folds),
        test_start.date(),
        folds[-1].delivery_day.date(),
    )

    # --- Runtime probe (spec B.5) -------------------------------------------
    probe_folds = folds[:PROBE_FOLDS]
    t0 = time.monotonic()
    run_backtest(
        y, MeritOrderForecaster(long_history, srmc_definition="gas_only"), probe_folds, x=l1_matrix
    )
    probe_elapsed = time.monotonic() - t0
    projected_hours = probe_elapsed / len(probe_folds) * len(folds) * 4 / 3600
    logger.info(
        "runtime probe: %d folds in %.1fs -> projected %.2fh for all 4 variants",
        len(probe_folds),
        probe_elapsed,
        projected_hours,
    )
    if projected_hours > args.runtime_budget_hours:
        logger.error(
            "RUECKFRAGE (spec B.5): projected runtime %.2fh exceeds the %.1fh budget -- stopping.",
            projected_hours,
            args.runtime_budget_hours,
        )
        return

    # --- Backtests, identical fold list for all 4 (spec B.2) ---------------
    models: dict[str, Forecaster] = {
        "A": LGBMForecaster(objective="quantile", alpha=0.5),
        "B": MeritOrderForecaster(long_history, srmc_definition="gas_only"),
        "B_full": MeritOrderForecaster(long_history, srmc_definition="gas_full"),
        "B_roll": MeritOrderForecaster(long_history, beta_window_days=BETA_ROLL_WINDOW_DAYS),
    }
    x_for = {"A": l1_matrix, "B": l1_matrix, "B_full": l1_matrix_full, "B_roll": l1_matrix}

    args.preds_dir.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, pd.DataFrame] = {}
    for name, model in models.items():
        preds = _get_or_run_backtest(name, model, folds, x_for[name], y, args.preds_dir)
        complete_days, incomplete_days = split_complete_delivery_days(preds)
        if len(incomplete_days) > 0:
            logger.info(
                "variant %s: dropping %d incomplete delivery day(s)", name, len(incomplete_days)
            )
        predictions[name] = preds.loc[preds["delivery_day"].isin(complete_days)]

    fold_counts = {name: preds["delivery_day"].nunique() for name, preds in predictions.items()}
    logger.info("delivery-day counts per variant (must match): %s", fold_counts)

    # --- V3 pool-equality proof (spec B.4) ----------------------------------
    v3_start, v3_end, v3_table = _v3_pool_window(long_history, edge)
    logger.info(
        "V3 anchor window with identical B-full/B-roll beta pools: %s to %s",
        v3_start.strftime("%Y-%m"),
        v3_end.strftime("%Y-%m"),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    v3_table.to_csv(args.out_dir / "price_model_selection_v3_pool_check.csv", index=False)

    # --- Metrics / by-RL-bin / OOR tables (spec 4.1) ------------------------
    bin_assignment = rl_bin(l1_matrix["residual_load_forecast"]).reindex(index)
    oor_flag = ttf_out_of_range_flag(
        long_history["ttf_lag"], index, train_span_days=TRAIN_SPAN_DAYS
    )
    upper_third_mask = bin_assignment.isin(UPPER_THIRD_BINS)
    oor_upper_third_mask = oor_flag.reindex(index).fillna(False) & upper_third_mask.fillna(False)

    period_names = ("overall", "v3_window", "autumn_winter", "autumn_2026")
    bin_period_names = ("overall", "autumn_2026")

    metrics_rows = []
    bin_rows = []
    oor_rows = []
    for name, preds in predictions.items():
        for period_name in period_names:
            in_period = _in_period(
                preds["delivery_day"],
                period_name,
                test_start=test_start,
                edge=edge,
                v3_start=v3_start,
                v3_end=v3_end,
            )
            sub = preds.loc[in_period]
            if sub.empty:
                continue
            metrics_rows.append(
                {
                    "variant": name,
                    "period": period_name,
                    **_metrics_row(sub["y_true"], sub["y_pred"]),
                }
            )

            if period_name in bin_period_names:
                bins_here = bin_assignment.reindex(sub.index)
                for bin_label in sorted(bins_here.dropna().unique()):
                    cell = sub.loc[bins_here == bin_label]
                    if cell.empty:
                        continue
                    bin_rows.append(
                        {
                            "variant": name,
                            "period": period_name,
                            "bin": int(bin_label),
                            **_metrics_row(cell["y_true"], cell["y_pred"]),
                        }
                    )

            oor_here = oor_upper_third_mask.reindex(sub.index).fillna(False)
            oor_cell = sub.loc[oor_here]
            if not oor_cell.empty:
                oor_rows.append(
                    {
                        "variant": name,
                        "period": period_name,
                        **_metrics_row(oor_cell["y_true"], oor_cell["y_pred"]),
                    }
                )

    pd.DataFrame(metrics_rows).to_csv(
        args.out_dir / "price_model_selection_metrics.csv", index=False
    )
    pd.DataFrame(bin_rows).to_csv(args.out_dir / "price_model_selection_by_rlbin.csv", index=False)
    pd.DataFrame(oor_rows).to_csv(
        args.out_dir / "price_model_selection_oor_upper_third.csv", index=False
    )

    # --- DM-tests (spec 4.2) ------------------------------------------------
    comparisons: list[tuple[str, str, str, tuple[str, ...]]] = [
        ("V1_Bfull_vs_B", "B_full", "B", ("overall", "autumn_winter", "autumn_2026")),
        ("V2_Broll_vs_B", "B_roll", "B", ("overall", "autumn_winter", "autumn_2026")),
        (
            "V3_Bfull_vs_Broll",
            "B_full",
            "B_roll",
            ("v3_window", "v3_autumn_winter", "v3_autumn_2026"),
        ),
        ("B_vs_A", "B", "A", ("overall",)),
        ("Bfull_vs_A", "B_full", "A", ("overall",)),
        ("Broll_vs_A", "B_roll", "A", ("overall",)),
    ]
    dm_rows = []
    for comparison_name, challenger_name, baseline_name, periods_to_test in comparisons:
        challenger = predictions[challenger_name]
        baseline = predictions[baseline_name]
        loss_challenger = (challenger["y_true"] - challenger["y_pred"]) ** 2
        loss_baseline = (baseline["y_true"] - baseline["y_pred"]) ** 2
        for period_name in periods_to_test:
            in_period = _in_period(
                challenger["delivery_day"],
                period_name,
                test_start=test_start,
                edge=edge,
                v3_start=v3_start,
                v3_end=v3_end,
            )
            lc = loss_challenger.loc[in_period]
            lb = loss_baseline.reindex(lc.index)
            aligned = pd.concat([lc.rename("c"), lb.rename("b")], axis=1).dropna()
            if len(aligned) < 2 * _HOURLY_HAC_LAG:
                continue
            daily_c, _ = daily_mean_loss(aligned["c"], group_by="local_day")
            daily_b, _ = daily_mean_loss(aligned["b"], group_by="local_day")
            daily_result = dm_test(daily_c, daily_b, hac_lag=_DAILY_HAC_LAG, horizon=_DAILY_HORIZON)
            hourly_result = dm_test(
                aligned["c"], aligned["b"], hac_lag=_HOURLY_HAC_LAG, horizon=_HOURLY_HORIZON
            )
            for block, result in (("daily", daily_result), ("hourly", hourly_result)):
                dm_rows.append(
                    {
                        "comparison": comparison_name,
                        "period": period_name,
                        "block": block,
                        "n_obs": result.n_obs,
                        "hac_lag": result.hac_lag,
                        "horizon": result.horizon,
                        "mean_loss_diff": result.mean_loss_diff,
                        "dm_stat": result.dm_stat,
                        "p_value": result.p_value,
                    }
                )
    pd.DataFrame(dm_rows).to_csv(args.out_dir / "price_model_selection_dm_test.csv", index=False)

    # --- Beta-monthly tables (spec 4.3) -------------------------------------
    beta_anchors = _month_anchors(pd.Timestamp(TEST_START, tz=LOCAL_TZ), edge)
    beta_configs: dict[str, dict[str, object]] = {
        "B": {"srmc_definition": "gas_only", "rl_measure": "absolute", "window_days": None},
        "B_full": {"srmc_definition": "gas_full", "rl_measure": "absolute", "window_days": None},
        "B_roll": {
            "srmc_definition": "gas_only",
            "rl_measure": "absolute",
            "window_days": BETA_ROLL_WINDOW_DAYS,
        },
    }
    beta_tables = []
    for variant, cfg in beta_configs.items():
        table = build_beta_monthly_table(long_history, beta_anchors, **cfg)  # type: ignore[arg-type]
        table["variant"] = variant
        table["implicit_efficiency"] = GAS_EFFICIENCY / table["beta_smoothed"]
        table["clamp_flag"] = table["beta_raw"] > 1.5
        beta_tables.append(table)
    beta_monthly = pd.concat(beta_tables, ignore_index=True)
    beta_monthly.to_csv(args.out_dir / "price_model_selection_beta_monthly.csv", index=False)

    # --- Figure: beta(RL) per variant, last anchor month --------------------
    last_anchor = beta_anchors[-1].strftime("%Y-%m")
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in beta_configs:
        row = beta_monthly.loc[
            (beta_monthly["variant"] == variant) & (beta_monthly["anchor_month"] == last_anchor)
        ].sort_values("bin")
        ax.plot(row["bin"], row["beta_smoothed"], marker="o", label=variant)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
    ax.axhline(1.5, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("RL bin")
    ax.set_ylabel("beta_smoothed")
    ax.set_title(f"beta(RL) by variant, anchor {last_anchor}")
    ax.legend()
    fig.tight_layout()
    args.fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig_path, dpi=150)
    plt.close(fig)

    logger.info(
        "done; wrote metrics/DM-test/beta CSVs to %s, figure to %s", args.out_dir, args.fig_path
    )

    _cfg = FeatureConfig()
    logger.info(
        "crisis_start=%s post_crisis_start=%s (for the log's closed-form V3 derivation)",
        _cfg.crisis_start.date(),
        _cfg.post_crisis_start.date(),
    )


if __name__ == "__main__":
    main()
