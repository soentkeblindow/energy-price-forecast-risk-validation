"""Sprint 7 Step 7.6, Teil C.1 (docs/sprint7_step7_6_spec.md, section 5.1).

Confirmation run for the owner-selected `price_v2` candidate B-full-roll
(srmc_gas_full + eua_co2_lag_48h tree feature, like B-full in Teil B, PLUS
a 3-year rolling beta window, like B-roll) -- not run as its own variant in
Teil B. Reuses Teil B's exact data/matrix/fold construction
(compare_price_model_selection_variants.py) so the equality check below is
comparing like for like.

Before anchor 2026-04 (inclusive), B-full-roll's beta pool is provably
identical to B-full's (same reasoning as Teil B's V3 proof: B-full's
effective pool is already bounded below by post_crisis_start, and a
3-year rolling window anchored before ~2026-04 still reaches back past
that bound) -- predictions and beta estimates must match exactly. From
2026-05 onward the rolling window starts cutting into 2023, so the two
diverge. Purely descriptive; no criterion, no winner.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
from build_beta_monthly_table import build_beta_monthly_table

from energy_price_forecast.common.data.loaders import load_interim_hourly, load_processed_features
from energy_price_forecast.common.data.snapshot import file_fingerprint
from energy_price_forecast.common.evaluation.dm_test import daily_mean_loss, dm_test
from energy_price_forecast.common.evaluation.metrics import bias, mae, rmse
from energy_price_forecast.common.evaluation.walkforward import (
    last_complete_local_day,
    run_backtest,
    split_complete_delivery_days,
    walk_forward_splits,
)
from energy_price_forecast.common.features.core_gas_columns import L1_COLUMNS
from energy_price_forecast.common.features.merit_order import (
    GAS_EFFICIENCY,
    build_merit_order_frame,
    raw_beta_per_bin,
    rl_bin,
    ttf_out_of_range_flag,
)
from energy_price_forecast.common.market_time import LOCAL_TZ
from energy_price_forecast.common.models.merit_order_forecaster import MeritOrderForecaster

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TRAIN_SPAN_DAYS = 90
TEST_START = "2023-05-01"  # identical to Teil B, spec B.2
BETA_ROLL_WINDOW_DAYS = 3 * 365
EQUALITY_ANCHOR_CUTOFF = pd.Timestamp("2026-04-01", tz=LOCAL_TZ)  # spec 5.1, inclusive
EQUALITY_TOLERANCE = 1e-9

UPPER_THIRD_BINS = (8, 9, 10)

_DAILY_HAC_LAG = 7
_DAILY_HORIZON = 1
_HOURLY_HAC_LAG = 48
_HOURLY_HORIZON = 24

A2_INPUT_FILES: tuple[Path, ...] = (
    Path("data/interim/hourly_merit_order.parquet"),
    Path("data/processed/features_merit_order.parquet"),
)


def _month_anchors(first: pd.Timestamp, last_local_day: pd.Timestamp) -> list[pd.Timestamp]:
    last = pd.Timestamp(year=last_local_day.year, month=last_local_day.month, day=1, tz=LOCAL_TZ)
    anchors = []
    current = first
    while current <= last:
        anchors.append(current)
        current = (current + pd.DateOffset(months=1)).normalize()
    return anchors


def _in_period(
    delivery_day: pd.Series, period: str, *, test_start: pd.Timestamp, edge: pd.Timestamp
) -> pd.Series:
    autumn_2026_start = pd.Timestamp("2026-09-01", tz=LOCAL_TZ)
    if period == "overall":
        return (delivery_day >= test_start) & (delivery_day <= edge)
    if period == "autumn_winter":
        return delivery_day.dt.month.isin([9, 10, 11, 12])
    if period == "autumn_2026":
        return delivery_day >= autumn_2026_start
    raise ValueError(f"unknown period {period!r}")


def _metrics_row(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    aligned = pd.concat([y_true.rename("t"), y_pred.rename("p")], axis=1).dropna()
    return {
        "n": len(aligned),
        "rmse": rmse(aligned["t"], aligned["p"]),
        "mae": mae(aligned["t"], aligned["p"]),
        "bias": bias(aligned["t"], aligned["p"]),
    }


def _append_rows(path: Path, new_rows: pd.DataFrame, *, variant_or_comparison_col: str) -> None:
    """Append ``new_rows`` to the CSV at ``path``, dropping any pre-existing
    rows for the same variant/comparison first so a re-run is idempotent
    (spec 5.1: "ergaenzt, nicht ersetzt").
    """
    existing = pd.read_csv(path)
    new_keys = set(new_rows[variant_or_comparison_col].unique())
    existing = existing.loc[~existing[variant_or_comparison_col].isin(new_keys)]
    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined.to_csv(path, index=False)
    logger.info("appended %d row(s) to %s (now %d total)", len(new_rows), path, len(combined))


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
        "--preds-dir", type=Path, default=Path("data/processed/price_model_selection_preds")
    )
    p.add_argument(
        "--snapshot-csv",
        type=Path,
        default=Path("outputs/results/price_model_selection_data_snapshot.csv"),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # --- Hash check against Teil A's recorded data stand (spec 5.1) --------
    snapshot = pd.read_csv(args.snapshot_csv, index_col="filename")
    hash_mismatches = []
    for path in A2_INPUT_FILES:
        current = file_fingerprint(path)
        recorded_sha256 = snapshot.loc[path.name, "sha256"]
        if current["sha256"] != recorded_sha256:
            hash_mismatches.append(path.name)
    if hash_mismatches:
        logger.error(
            "RUECKFRAGE (spec 5.1): data stand changed since Teil A for %s -- stopping.",
            hash_mismatches,
        )
        return
    logger.info("hash check passed: both A.2 input files match the recorded Teil-A data stand")

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
    test_start = folds[0].delivery_day
    logger.info(
        "fold list (identical to Teil B): %d folds, %s to %s",
        len(folds),
        test_start.date(),
        folds[-1].delivery_day.date(),
    )

    # --- Backtest B-full-roll -----------------------------------------------
    cache_path = args.preds_dir / "price_model_selection_preds_B_full_roll.parquet"
    if cache_path.exists():
        logger.info("loaded cached predictions from %s", cache_path)
        preds_b_full_roll = pd.read_parquet(cache_path)
    else:
        model = MeritOrderForecaster(
            long_history, srmc_definition="gas_full", beta_window_days=BETA_ROLL_WINDOW_DAYS
        )
        t0 = time.monotonic()
        preds_b_full_roll = run_backtest(y, model, folds, refit_every=1, x=l1_matrix_full)
        elapsed = time.monotonic() - t0
        logger.info(
            "B_full_roll: %d test hours in %.1fs (%.3fs/fold)",
            len(preds_b_full_roll),
            elapsed,
            elapsed / len(folds) if folds else float("nan"),
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        preds_b_full_roll.to_parquet(cache_path)

    complete_days, incomplete_days = split_complete_delivery_days(preds_b_full_roll)
    if len(incomplete_days) > 0:
        logger.info("B_full_roll: dropping %d incomplete delivery day(s)", len(incomplete_days))
    preds_b_full_roll = preds_b_full_roll.loc[preds_b_full_roll["delivery_day"].isin(complete_days)]

    preds_b_full = pd.read_parquet(args.preds_dir / "price_model_selection_preds_B_full.parquet")

    # --- Equality check through anchor 2026-04 inclusive (spec 5.1) --------
    equal_mask = preds_b_full_roll["delivery_day"] <= EQUALITY_ANCHOR_CUTOFF + pd.DateOffset(
        months=1, days=-1
    )
    equal_window = preds_b_full_roll.loc[equal_mask]
    compare = equal_window.join(preds_b_full[["y_pred"]], how="inner", rsuffix="_b_full")
    compare["abs_diff"] = (compare["y_pred"] - compare["y_pred_b_full"]).abs()
    max_pred_diff = float(compare["abs_diff"].max())
    passes = max_pred_diff <= EQUALITY_TOLERANCE
    logger.info(
        "prediction equality check (anchors through %s): max |diff| = %.3e over %d hours -- %s",
        EQUALITY_ANCHOR_CUTOFF.strftime("%Y-%m"),
        max_pred_diff,
        len(compare),
        "PASS" if passes else "deviation found (see per-month breakdown below)",
    )
    by_month = compare["abs_diff"].groupby(compare["delivery_day"].dt.strftime("%Y-%m")).max()
    logger.info("prediction equality check, max |diff| per anchor month:\n%s", by_month.to_string())

    beta_anchors = _month_anchors(pd.Timestamp(TEST_START, tz=LOCAL_TZ), edge)
    equal_anchors = [a for a in beta_anchors if a <= EQUALITY_ANCHOR_CUTOFF]
    beta_full = build_beta_monthly_table(
        long_history,
        equal_anchors,
        srmc_definition="gas_full",
        rl_measure="absolute",
        window_days=None,
    )
    beta_full_roll_equal = build_beta_monthly_table(
        long_history,
        equal_anchors,
        srmc_definition="gas_full",
        rl_measure="absolute",
        window_days=BETA_ROLL_WINDOW_DAYS,
    )
    beta_compare = beta_full.merge(
        beta_full_roll_equal, on=["anchor_month", "bin"], suffixes=("_full", "_full_roll")
    )
    max_beta_diff_by_anchor = (
        (beta_compare["beta_smoothed_full"] - beta_compare["beta_smoothed_full_roll"])
        .abs()
        .groupby(beta_compare["anchor_month"])
        .max()
    )
    logger.info(
        "beta_smoothed equality check per anchor (through %s): %s",
        EQUALITY_ANCHOR_CUTOFF.strftime("%Y-%m"),
        max_beta_diff_by_anchor.to_dict(),
    )

    # --- Deviation after 2026-04 (spec 5.1) ---------------------------------
    later_anchors = [a for a in beta_anchors if a > EQUALITY_ANCHOR_CUTOFF]
    deviation_rows = []
    for anchor in later_anchors:
        n_full = raw_beta_per_bin(
            long_history,
            anchor,
            srmc_definition="gas_full",
            rl_measure="absolute",
            window_days=None,
        )["n"]
        beta_full_anchor = build_beta_monthly_table(
            long_history,
            [anchor],
            srmc_definition="gas_full",
            rl_measure="absolute",
            window_days=None,
        )
        beta_roll_anchor = build_beta_monthly_table(
            long_history,
            [anchor],
            srmc_definition="gas_full",
            rl_measure="absolute",
            window_days=BETA_ROLL_WINDOW_DAYS,
        )
        merged = beta_full_anchor.merge(beta_roll_anchor, on="bin", suffixes=("_full", "_roll"))
        max_diff = float((merged["beta_smoothed_full"] - merged["beta_smoothed_roll"]).abs().max())
        deviation_rows.append(
            {
                "anchor_month": anchor.strftime("%Y-%m"),
                "n_pool_full": int(n_full.sum()),
                "max_beta_smoothed_diff": max_diff,
            }
        )
    deviation_table = pd.DataFrame(deviation_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    deviation_table.to_csv(
        args.out_dir / "price_model_selection_b_full_roll_deviation_after.csv", index=False
    )
    logger.info(
        "deviation-after table (anchors > %s):\n%s",
        EQUALITY_ANCHOR_CUTOFF.strftime("%Y-%m"),
        deviation_table,
    )

    # --- Metrics / by-RL-bin / OOR, appended to Teil B's CSVs (spec 5.1) ---
    bin_assignment = rl_bin(l1_matrix["residual_load_forecast"]).reindex(index)
    oor_flag = ttf_out_of_range_flag(
        long_history["ttf_lag"], index, train_span_days=TRAIN_SPAN_DAYS
    )
    upper_third_mask = bin_assignment.isin(UPPER_THIRD_BINS)
    oor_upper_third_mask = oor_flag.reindex(index).fillna(False) & upper_third_mask.fillna(False)

    period_names = ("overall", "autumn_winter", "autumn_2026")
    bin_period_names = ("overall", "autumn_2026")

    metrics_rows = []
    bin_rows = []
    oor_rows = []
    for period_name in period_names:
        in_period = _in_period(
            preds_b_full_roll["delivery_day"], period_name, test_start=test_start, edge=edge
        )
        sub = preds_b_full_roll.loc[in_period]
        if sub.empty:
            continue
        metrics_rows.append(
            {
                "variant": "B_full_roll",
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
                        "variant": "B_full_roll",
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
                    "variant": "B_full_roll",
                    "period": period_name,
                    **_metrics_row(oor_cell["y_true"], oor_cell["y_pred"]),
                }
            )

    _append_rows(
        args.out_dir / "price_model_selection_metrics.csv",
        pd.DataFrame(metrics_rows),
        variant_or_comparison_col="variant",
    )
    _append_rows(
        args.out_dir / "price_model_selection_by_rlbin.csv",
        pd.DataFrame(bin_rows),
        variant_or_comparison_col="variant",
    )
    _append_rows(
        args.out_dir / "price_model_selection_oor_upper_third.csv",
        pd.DataFrame(oor_rows),
        variant_or_comparison_col="variant",
    )

    # --- DM-tests: B-full-roll vs B-full, vs A (spec 5.1) -------------------
    preds_a = pd.read_parquet(args.preds_dir / "price_model_selection_preds_A.parquet")
    loss_b_full_roll = (preds_b_full_roll["y_true"] - preds_b_full_roll["y_pred"]) ** 2
    loss_b_full = (preds_b_full["y_true"] - preds_b_full["y_pred"]) ** 2
    loss_a = (preds_a["y_true"] - preds_a["y_pred"]) ** 2

    dm_rows = []
    for comparison_name, baseline_loss in (
        ("Bfullroll_vs_Bfull", loss_b_full),
        ("Bfullroll_vs_A", loss_a),
    ):
        for period_name in period_names:
            in_period = _in_period(
                preds_b_full_roll["delivery_day"], period_name, test_start=test_start, edge=edge
            )
            lc = loss_b_full_roll.loc[in_period]
            lb = baseline_loss.reindex(lc.index)
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
    _append_rows(
        args.out_dir / "price_model_selection_dm_test.csv",
        pd.DataFrame(dm_rows),
        variant_or_comparison_col="comparison",
    )

    # --- Beta-monthly table: append B_full_roll (spec 5.1) -----------------
    beta_table = build_beta_monthly_table(
        long_history,
        beta_anchors,
        srmc_definition="gas_full",
        rl_measure="absolute",
        window_days=BETA_ROLL_WINDOW_DAYS,
    )
    beta_table["variant"] = "B_full_roll"
    beta_table["implicit_efficiency"] = GAS_EFFICIENCY / beta_table["beta_smoothed"]
    beta_table["clamp_flag"] = beta_table["beta_raw"] > 1.5
    _append_rows(
        args.out_dir / "price_model_selection_beta_monthly.csv",
        beta_table,
        variant_or_comparison_col="variant",
    )

    logger.info("done; appended B_full_roll rows to Teil B's CSVs in %s", args.out_dir)


if __name__ == "__main__":
    main()
