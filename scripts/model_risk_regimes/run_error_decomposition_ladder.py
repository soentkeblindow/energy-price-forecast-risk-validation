"""Teil 1 ("die Leiter") of
docs/residual_error_decomposition_hourly_spec.md.

Runs variant A (plain LightGBM) and variant B (merit-order target transform,
7.2 primary configuration) on five feature "rungs" (spec section 3.1):

  L0    full feature set (today's A/B from 7.2)
  L1    core_gas's reduced feature set, ENTSO-E forecasts (section 3.2)
  L2    same reduced set, NWP-reconstructed wind/solar/residual-load/share
        (section 3.3)
  O_RE  Oracle: wind/solar/residual-load/share from ACTUALS, load stays a
        forecast (section 3.1/4)
  O_ALL Oracle: load is also replaced by its actual

-- plus 4 extra runs (L0/L2 x A/B with objective="l2" instead of "quantile",
section 5.3) and 1 control run (L0, B, no beta ceiling, section 3.4).

Variant B's long-history beta side channel is the SAME ENTSO-E-forecast-based
long_history for L0/L1/L2 (table 3.4: estimating on forecasts, while L2's
*application* of beta already sees NWP-RL because L2's own feature matrix
is what gets binned at fit/predict time -- no extra parameter needed). O_RE/
O_ALL use their own oracle long-history frame (section 3.4's clarification:
beta is estimated on the SAME data the tree trains on).

Common evaluation window (section 3.5) is derived from the data: the
intersection of all five rungs' complete local days (no NaN in any rung's
own matrix for that day), starting no earlier than L2's own first valid day
(the binding constraint -- NWP reconstruction starts ~2024-03), through the
Teil-0 edge. Stops instead of running (Rueckfrage trigger, spec section 11)
if that window is under 12 months.

Before the full 15-run suite, a short runtime probe projects the total
runtime; stops instead of running if the projection exceeds ~4h (section 11).

Per-variant-per-rung predictions are cached under --preds-dir (same
_get_or_run_backtest pattern as compare_merit_order_variants.py), in a
directory separate from 7.2's caches.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from energy_price_forecast.common.data.loaders import load_interim_hourly, load_processed_features
from energy_price_forecast.common.evaluation.dm_test import daily_mean_loss, dm_test
from energy_price_forecast.common.evaluation.metrics import bias, mae, rmse
from energy_price_forecast.common.evaluation.walkforward import (
    Forecaster,
    last_complete_local_day,
    run_backtest,
    walk_forward_splits,
)
from energy_price_forecast.common.features.core_gas_columns import L1_COLUMNS
from energy_price_forecast.common.features.merit_order import (
    build_merit_order_frame,
    rl_bin,
    ttf_out_of_range_flag,
)
from energy_price_forecast.common.market_time import LOCAL_TZ
from energy_price_forecast.common.models.lgbm import LGBMForecaster
from energy_price_forecast.common.models.merit_order_forecaster import MeritOrderForecaster
from energy_price_forecast.model_risk_regimes.features.l2_matrix import build_l2_matrix
from energy_price_forecast.model_risk_regimes.features.oracle_fundamentals import (
    build_oracle_feature_frame,
    build_oracle_long_history_frame,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TRAIN_SPAN_DAYS = 90
RUNGS: tuple[str, ...] = ("L0", "L1", "L2", "O_RE", "O_ALL")
MIN_COMMON_WINDOW_DAYS = 365  # spec section 11, Rueckfrage trigger 2
RUNTIME_BUDGET_HOURS = 4.0  # spec section 11, Rueckfrage trigger 5
PROBE_FOLDS = 20

UPPER_THIRD_BINS = (8, 9, 10)  # same definition as 7.2 (compare_merit_order_variants.py)

_DAILY_HAC_LAG = 7
_DAILY_HORIZON = 1
_HOURLY_HAC_LAG = 48
_HOURLY_HORIZON = 24


def _incomplete_local_days(frame: pd.DataFrame) -> set[pd.Timestamp]:
    """Local (Europe/Berlin) days where at least one hour has a NaN in at
    least one column -- spec section 3.5's "Tage ohne vollstaendige
    Abdeckung fallen aus dem gemeinsamen Auswertungszeitraum ... heraus."
    """
    local_day = pd.DatetimeIndex(frame.index).tz_convert(LOCAL_TZ).normalize()
    any_nan = frame.isna().any(axis=1)
    bad = any_nan.groupby(local_day).any()
    return set(bad.index[bad])


def _first_valid_local_day(frame: pd.DataFrame, columns: list[str]) -> pd.Timestamp:
    """First local day where ``columns`` are all non-NaN for every hour --
    L2's own reconstruction-coverage start, the binding constraint on the
    common evaluation window (spec section 3.5).
    """
    local_day = pd.DatetimeIndex(frame.index).tz_convert(LOCAL_TZ).normalize()
    all_present = frame[columns].notna().all(axis=1)
    good_days = all_present.groupby(local_day).all()
    return good_days.index[good_days][0]


def _get_or_run_backtest(
    name: str,
    model: Forecaster,
    folds: list,
    x: pd.DataFrame,
    y: pd.Series,
    preds_dir: Path,
) -> pd.DataFrame:
    cache_path = preds_dir / f"ladder_preds_{name}.parquet"
    if cache_path.exists():
        logger.info("run %s: loaded cached predictions from %s", name, cache_path)
        return pd.read_parquet(cache_path)
    t0 = time.monotonic()
    preds = run_backtest(y, model, folds, refit_every=1, x=x)
    elapsed = time.monotonic() - t0
    logger.info(
        "run %s: %d test hours in %.1fs (%.3fs/fold)",
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


def _dm_row(
    comparison: str,
    period: str,
    loss_a: pd.Series,
    loss_b: pd.Series,
) -> list[dict[str, float | str | int]]:
    aligned = pd.concat([loss_a.rename("a"), loss_b.rename("b")], axis=1).dropna()
    if len(aligned) < 2 * _HOURLY_HAC_LAG:
        return []
    daily_a, _ = daily_mean_loss(aligned["a"])
    daily_b, _ = daily_mean_loss(aligned["b"])
    daily_result = dm_test(daily_b, daily_a, hac_lag=_DAILY_HAC_LAG, horizon=_DAILY_HORIZON)
    hourly_result = dm_test(
        aligned["b"], aligned["a"], hac_lag=_HOURLY_HAC_LAG, horizon=_HOURLY_HORIZON
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
                "horizon": result.horizon,
                "mean_loss_diff": result.mean_loss_diff,
                "dm_stat": result.dm_stat,
                "p_value": result.p_value,
            }
        )
    return rows


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-path", type=Path, default=Path("data/interim/hourly_merit_order.parquet")
    )
    p.add_argument(
        "--features-path", type=Path, default=Path("data/processed/features_merit_order.parquet")
    )
    p.add_argument(
        "--renewables-path",
        type=Path,
        default=Path("data/processed/renewables_forecast_rolling365_l2.parquet"),
    )
    p.add_argument("--out-dir", type=Path, default=Path("outputs/results"))
    p.add_argument(
        "--fig-path", type=Path, default=Path("outputs/assets/error_decomposition_ladder_rmse.png")
    )
    p.add_argument("--preds-dir", type=Path, default=Path("data/processed/ladder_preds"))
    p.add_argument(
        "--runtime-budget-hours",
        type=float,
        default=RUNTIME_BUDGET_HOURS,
        help="Stop before the full run if the runtime probe projects more than this.",
    )
    p.add_argument(
        "--window-start",
        type=str,
        default=None,
        help="Floor the common window's start at this date, for a fast partial "
        "run (e.g. a quick 2026-only check after a data/feature fix) instead of "
        "the full L2-bounded history. Also lifts the 12-month minimum-window "
        "Rueckfrage check (spec section 11), since a deliberately short window "
        "is the point of passing this.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    df = load_interim_hourly(args.data_path)
    price = df["day_ahead_price"]
    features = load_processed_features(args.features_path)
    y = price.reindex(features.index)
    index = pd.DatetimeIndex(features.index)

    edge = last_complete_local_day(price)
    logger.info("Teil-0 edge (last locally-complete delivery day): %s", edge.date())

    # --- Build the five rung feature matrices -----------------------------
    l0_matrix = features
    l1_matrix = features[list(L1_COLUMNS)]
    renewables_predictions = pd.read_parquet(args.renewables_path)
    l2_matrix = build_l2_matrix(l1_matrix, renewables_predictions, index)
    o_re_matrix = build_oracle_feature_frame(l1_matrix, df, "o_re")
    o_all_matrix = build_oracle_feature_frame(l1_matrix, df, "o_all")

    rung_matrices: dict[str, pd.DataFrame] = {
        "L0": l0_matrix,
        "L1": l1_matrix,
        "L2": l2_matrix,
        "O_RE": o_re_matrix,
        "O_ALL": o_all_matrix,
    }

    # --- Common evaluation window (spec section 3.5) -----------------------
    l2_first_valid_day = _first_valid_local_day(
        l2_matrix,
        ["wind_onshore_forecast", "wind_offshore_forecast", "solar_forecast"],
    )
    logger.info(
        "L2 (NWP reconstruction) first fully-covered local day: %s", l2_first_valid_day.date()
    )

    incomplete_days_by_rung = {name: _incomplete_local_days(m) for name, m in rung_matrices.items()}
    for name, days in incomplete_days_by_rung.items():
        logger.info("rung %s: %d incomplete local day(s)", name, len(days))
    excluded_days = set().union(*incomplete_days_by_rung.values())

    # The first SAFE test day needs its own full TRAIN_SPAN_DAYS trailing
    # window to already be inside L2's valid range too -- walk_forward_splits
    # only guarantees a trailing window exists in the raw calendar-day index
    # (which spans back to 2019 for L0/L1), not that L2's own NaN-free region
    # covers it. Without this, the earliest folds' entire 90-day training
    # window predates L2's existence, every row's RL bin is NaN, and the B
    # variant's target-transform step ends up with an all-NaN y' (a real bug
    # hit once, see docs/residual_error_decomposition_hourly_log.md Teil 1).
    candidate_start = l2_first_valid_day + pd.DateOffset(days=TRAIN_SPAN_DAYS + 1)
    if args.window_start is not None:
        candidate_start = max(candidate_start, pd.Timestamp(args.window_start, tz=LOCAL_TZ))
    candidate_folds = list(
        walk_forward_splits(
            index,
            test_start=candidate_start.strftime("%Y-%m-%d"),
            test_end=edge.strftime("%Y-%m-%d"),
            window="rolling",
            train_span_days=TRAIN_SPAN_DAYS,
        )
    )
    folds = [f for f in candidate_folds if f.delivery_day not in excluded_days]
    if not folds:
        raise ValueError("no evaluable folds in the common window -- check rung matrices")

    window_days = (folds[-1].delivery_day - folds[0].delivery_day).days + 1
    logger.info(
        "common evaluation window: %s to %s (%d folds, %d calendar days, %d excluded as incomplete)",
        folds[0].delivery_day.date(),
        folds[-1].delivery_day.date(),
        len(folds),
        window_days,
        len(candidate_folds) - len(folds),
    )
    if window_days < MIN_COMMON_WINDOW_DAYS and args.window_start is None:
        logger.error(
            "RUECKFRAGE (spec section 11): common evaluation window is only %d days (< %d) -- "
            "stopping, not running the ladder.",
            window_days,
            MIN_COMMON_WINDOW_DAYS,
        )
        return
    if window_days < MIN_COMMON_WINDOW_DAYS:
        logger.warning(
            "window is only %d days (< %d) -- proceeding anyway because --window-start was "
            "given (a deliberate partial run, not the official common-window determination).",
            window_days,
            MIN_COMMON_WINDOW_DAYS,
        )

    # --- Long-history beta side channels per rung --------------------------
    long_history = build_merit_order_frame(df, pd.DatetimeIndex(df.index))
    long_history_o_re = build_oracle_long_history_frame(long_history, df, "o_re")
    long_history_o_all = build_oracle_long_history_frame(long_history, df, "o_all")
    long_history_by_rung: dict[str, pd.DataFrame] = {
        "L0": long_history,
        "L1": long_history,
        "L2": long_history,  # table 3.4: estimate on ENTSO-E-forecast RL
        "O_RE": long_history_o_re,
        "O_ALL": long_history_o_all,
    }

    # --- Runtime probe (spec section 11, Rueckfrage trigger 5) --------------
    probe_folds = folds[:PROBE_FOLDS]
    t0 = time.monotonic()
    run_backtest(
        y, LGBMForecaster(objective="quantile", alpha=0.5), probe_folds, refit_every=1, x=l0_matrix
    )
    run_backtest(
        y,
        MeritOrderForecaster(long_history_o_all, srmc_definition="gas_only"),
        probe_folds,
        refit_every=1,
        x=o_all_matrix,
    )
    probe_elapsed = time.monotonic() - t0
    per_fold_s = probe_elapsed / (2 * len(probe_folds))
    projected_hours = per_fold_s * len(folds) * 15 / 3600
    logger.info(
        "runtime probe: %.3fs/fold (avg of L0-A and O_ALL-B), projected %.2fh for all 15 runs "
        "over %d folds",
        per_fold_s,
        projected_hours,
        len(folds),
    )
    if projected_hours > args.runtime_budget_hours:
        logger.error(
            "RUECKFRAGE (spec section 11): projected runtime %.2fh exceeds the %.1fh budget -- "
            "stopping, not running the full ladder.",
            projected_hours,
            args.runtime_budget_hours,
        )
        return

    # --- The 10 ladder runs (5 rungs x A/B) ---------------------------------
    args.preds_dir.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, pd.DataFrame] = {}
    for rung in RUNGS:
        x = rung_matrices[rung]
        lh = long_history_by_rung[rung]
        predictions[f"{rung}_A"] = _get_or_run_backtest(
            f"{rung}_A",
            LGBMForecaster(objective="quantile", alpha=0.5),
            folds,
            x,
            y,
            args.preds_dir,
        )
        predictions[f"{rung}_B"] = _get_or_run_backtest(
            f"{rung}_B",
            MeritOrderForecaster(lh, srmc_definition="gas_only"),
            folds,
            x,
            y,
            args.preds_dir,
        )

    # --- 4 l2-objective runs (L0/L2 x A/B, spec section 5.3) ----------------
    for rung in ("L0", "L2"):
        x = rung_matrices[rung]
        lh = long_history_by_rung[rung]
        predictions[f"{rung}_A_l2obj"] = _get_or_run_backtest(
            f"{rung}_A_l2obj", LGBMForecaster(objective="l2"), folds, x, y, args.preds_dir
        )
        predictions[f"{rung}_B_l2obj"] = _get_or_run_backtest(
            f"{rung}_B_l2obj",
            MeritOrderForecaster(lh, srmc_definition="gas_only", objective="l2"),
            folds,
            x,
            y,
            args.preds_dir,
        )

    # --- 1 beta-no-clamp control run (L0, B, spec section 3.4) -------------
    predictions["L0_B_noclamp"] = _get_or_run_backtest(
        "L0_B_noclamp",
        MeritOrderForecaster(long_history, srmc_definition="gas_only", beta_clamp_max=None),
        folds,
        l0_matrix,
        y,
        args.preds_dir,
    )

    # --- Metrics per run x period --------------------------------------------
    common_start, common_end = folds[0].delivery_day, folds[-1].delivery_day
    periods = {
        "overall": (common_start, common_end),
        "autumn_2026": (max(common_start, pd.Timestamp("2026-09-01", tz=LOCAL_TZ)), common_end),
    }

    bin_assignment = rl_bin(l0_matrix["residual_load_forecast"]).reindex(index)
    oor_flag = ttf_out_of_range_flag(
        long_history["ttf_lag"], index, train_span_days=TRAIN_SPAN_DAYS
    )
    upper_third_mask = bin_assignment.isin(UPPER_THIRD_BINS)
    oor_upper_third_mask = oor_flag.reindex(index).fillna(False) & upper_third_mask.fillna(False)

    metrics_rows = []
    bin_rows = []
    oor_rows = []
    for name, preds in predictions.items():
        for period_name, (start, end) in periods.items():
            in_period = (preds["delivery_day"] >= start) & (preds["delivery_day"] <= end)
            sub = preds.loc[in_period]
            if sub.empty:
                continue
            metrics_rows.append(
                {"run": name, "period": period_name, **_metrics_row(sub["y_true"], sub["y_pred"])}
            )

            bins_here = bin_assignment.reindex(sub.index)
            for bin_label in sorted(bins_here.dropna().unique()):
                cell = sub.loc[bins_here == bin_label]
                if cell.empty:
                    continue
                bin_rows.append(
                    {
                        "run": name,
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
                        "run": name,
                        "period": period_name,
                        **_metrics_row(oor_cell["y_true"], oor_cell["y_pred"]),
                    }
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_rows).to_csv(
        args.out_dir / "error_decomposition_ladder_metrics.csv", index=False
    )
    pd.DataFrame(bin_rows).to_csv(
        args.out_dir / "error_decomposition_ladder_by_rlbin.csv", index=False
    )
    pd.DataFrame(oor_rows).to_csv(
        args.out_dir / "error_decomposition_ladder_oor_upper_third.csv", index=False
    )

    # --- Decomposition DM-tests: rung-to-rung and B-vs-A per rung -----------
    dm_rows: list[dict[str, float | str | int]] = []
    rung_pairs = [("L0", "L1"), ("L1", "L2"), ("L1", "O_RE"), ("O_RE", "O_ALL")]
    for variant in ("A", "B"):
        for left, right in rung_pairs:
            left_name, right_name = f"{left}_{variant}", f"{right}_{variant}"
            if left_name not in predictions or right_name not in predictions:
                continue
            loss_left = (predictions[left_name]["y_true"] - predictions[left_name]["y_pred"]) ** 2
            loss_right = (
                predictions[right_name]["y_true"] - predictions[right_name]["y_pred"]
            ) ** 2
            for period_name, (start, end) in periods.items():
                mask_left = (predictions[left_name]["delivery_day"] >= start) & (
                    predictions[left_name]["delivery_day"] <= end
                )
                dm_rows.extend(
                    _dm_row(
                        f"{right}_vs_{left}_{variant}",
                        period_name,
                        loss_left.loc[mask_left],
                        loss_right.reindex(loss_left.index).loc[mask_left],
                    )
                )

    for rung in RUNGS:
        loss_a = (predictions[f"{rung}_A"]["y_true"] - predictions[f"{rung}_A"]["y_pred"]) ** 2
        loss_b = (predictions[f"{rung}_B"]["y_true"] - predictions[f"{rung}_B"]["y_pred"]) ** 2
        for period_name, (start, end) in periods.items():
            mask = (predictions[f"{rung}_A"]["delivery_day"] >= start) & (
                predictions[f"{rung}_A"]["delivery_day"] <= end
            )
            dm_rows.extend(
                _dm_row(f"B_vs_A_{rung}", period_name, loss_a.loc[mask], loss_b.loc[mask])
            )

    pd.DataFrame(dm_rows).to_csv(
        args.out_dir / "error_decomposition_ladder_dm_test.csv", index=False
    )

    # --- Staircase figure: RMSE per rung, A vs B ----------------------------
    overall_metrics = pd.DataFrame(metrics_rows)
    overall_metrics = overall_metrics[overall_metrics["period"] == "overall"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant, marker in (("A", "o"), ("B", "s")):
        rmse_by_rung = [
            overall_metrics.loc[overall_metrics["run"] == f"{rung}_{variant}", "rmse"].iloc[0]
            for rung in RUNGS
        ]
        ax.plot(RUNGS, rmse_by_rung, marker=marker, label=variant)
    ax.set_xlabel("rung")
    ax.set_ylabel("RMSE (EUR/MWh), overall common window")
    ax.set_title("Error decomposition ladder: RMSE by rung")
    ax.legend(title="variant")
    fig.tight_layout()
    args.fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig_path, dpi=150)
    plt.close(fig)

    logger.info(
        "done; wrote metrics/DM-test CSVs to %s and figure to %s", args.out_dir, args.fig_path
    )


if __name__ == "__main__":
    main()
