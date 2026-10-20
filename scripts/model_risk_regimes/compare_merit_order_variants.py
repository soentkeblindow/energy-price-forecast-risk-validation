"""Phase 2 (variant comparison) of
docs/merit_order_extrapolation_structure_measurement_spec.md.

Runs the walk-forward backtest (rolling 90-day window, daily refit) for
variant A (today's model), variant B (merit-order target transform), and
variant C (linear_tree, explorative) against a persistence baseline, over the
same test period, and writes RMSE/MAE/bias overall, by period, by RL bin, and
on the "TTF-out-of-range AND RL in the upper third" cut, plus
Diebold-Mariano tests (B vs A, C vs A) and a rolling-30-day-RMSE figure.

Variant C (spec 5.4) is a two-stage procedure: all three linear_lambda
candidates are run (and reported) only on the pre-2024 period; the winner
(lowest RMSE there) is then extended, unchanged, over the rest of the test
period -- no further tuning once the winner is picked.

Also runs the three B sensitivities (spec 5.3: all three reported, none
picked as a winner by test result): B-full (srmc_gas_full instead of srmc_gas_only),
B-rel (relative-RL bins instead of fixed-MW bins), B-roll (3-year rolling
beta window instead of expanding). They go through the same metrics/DM-test
pipeline as A/B/C, with no separate success-criterion check.

Per-variant predictions are cached under --preds-dir; a variant already on
disk there is loaded instead of re-run, so re-running this script after
adding a new variant does not repeat the earlier ones.
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
    run_backtest,
    walk_forward_splits,
)
from energy_price_forecast.common.features.config import FeatureConfig
from energy_price_forecast.common.features.merit_order import (
    build_merit_order_frame,
    rl_bin,
    ttf_out_of_range_flag,
)
from energy_price_forecast.common.market_time import LOCAL_TZ
from energy_price_forecast.common.models.lgbm import _DEFAULT_PARAMS, LGBMForecaster
from energy_price_forecast.common.models.merit_order_forecaster import MeritOrderForecaster
from energy_price_forecast.model_risk_regimes.models.baseline import SimilarDayNaive

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TRAIN_SPAN_DAYS = 90

# Daily-block primary, hourly for robustness -- same constants as
# scripts/model_post_crisis_resolution/compare_objectives.py.
_DAILY_HAC_LAG = 7
_DAILY_HORIZON = 1
_HOURLY_HAC_LAG = 48
_HOURLY_HORIZON = 24

# "RL in the upper third" (spec 5.6): the three highest of the eleven frozen
# RL bins (bin 8's lower edge sits at the 70th percentile of 2019-2020
# residual load, close enough to the upper tercile that a bin-boundary
# alignment is preferred over splitting a bin).
UPPER_THIRD_BINS = (8, 9, 10)

# Variant C (spec 5.4): all three candidates, winner picked only on the
# pre-cutoff period and then applied unchanged -- no further search.
LINEAR_LAMBDAS = (0, 1, 10)
C_WINNER_CUTOFF = pd.Timestamp("2024-01-01", tz=LOCAL_TZ)

# B-roll sensitivity (spec 5.3): 3-year rolling beta window instead of expanding.
BETA_ROLL_WINDOW_DAYS = 3 * 365

# All three B sensitivities, reported without a success-criterion check (spec 5.3).
SENSITIVITY_NAMES = ("B_full", "B_rel", "B_roll")


def _period_bounds(
    test_start: pd.Timestamp, test_end: pd.Timestamp
) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    def d(s: str) -> pd.Timestamp:
        return pd.Timestamp(s, tz=LOCAL_TZ)

    cfg = FeatureConfig()
    return {
        "overall": (test_start, test_end),
        "autumn_2021": (d("2021-08-01"), d("2021-12-31")),
        "crisis": (cfg.crisis_start, cfg.post_crisis_start - pd.DateOffset(days=1)),
        "post_crisis": (cfg.post_crisis_start, test_end),
        "autumn_2026": (d("2026-09-01"), test_end),
    }


def _get_or_run_backtest(
    name: str,
    model: Forecaster,
    folds: list,
    x_arg: pd.DataFrame | None,
    y: pd.Series,
    preds_dir: Path,
) -> pd.DataFrame:
    """Load cached predictions for ``name`` if present, else run and cache them.

    Lets this script be re-run after adding a new variant without repeating
    the (expensive) earlier ones.
    """
    cache_path = preds_dir / f"merit_order_preds_{name}.parquet"
    if cache_path.exists():
        logger.info("variant %s: loaded cached predictions from %s", name, cache_path)
        return pd.read_parquet(cache_path)
    if not folds:
        # e.g. B_full's test period starts before its EUA/post-crisis data
        # requirement is met for the --test-start/--test-end window given --
        # an empty, but well-formed, result rather than run_backtest's
        # "No objects to concatenate" on an empty fold list.
        logger.info("variant %s: 0 folds in range, skipping (empty result)", name)
        return pd.DataFrame(columns=["y_true", "y_pred", "delivery_day"]).set_index(
            pd.DatetimeIndex([], tz="UTC")
        )
    t0 = time.monotonic()
    preds = run_backtest(y, model, folds, refit_every=1, x=x_arg)
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


def _run_variant_c(
    y: pd.Series, x: pd.DataFrame, folds: list, preds_dir: Path
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Two-stage linear_tree variant (spec 5.4).

    All three linear_lambda candidates run (and are reported) on the
    pre-cutoff period only; the winner (lowest RMSE there) is then extended,
    unchanged, over the rest of the test period -- no further hyperparameter
    search once the winner is picked.
    """
    pre_folds = [f for f in folds if f.delivery_day < C_WINNER_CUTOFF]
    post_folds = [f for f in folds if f.delivery_day >= C_WINNER_CUTOFF]
    if not pre_folds:
        raise ValueError("no folds before the C-winner cutoff (2024-01-01)")

    lambda_preds: dict[int, pd.DataFrame] = {}
    lambda_rmse: dict[int, float] = {}
    for lam in LINEAR_LAMBDAS:
        params = {**_DEFAULT_PARAMS, "linear_tree": True, "linear_lambda": lam}
        model = LGBMForecaster(objective="quantile", alpha=0.5, params=params)
        preds = _get_or_run_backtest(f"C_lambda{lam}_pre2024", model, pre_folds, x, y, preds_dir)
        lambda_preds[lam] = preds
        lambda_rmse[lam] = rmse(preds["y_true"], preds["y_pred"])
        logger.info("variant C lambda=%d pre-2024 RMSE=%.3f", lam, lambda_rmse[lam])

    winner = min(lambda_rmse, key=lambda x_: lambda_rmse[x_])
    logger.info(
        "variant C winner: linear_lambda=%d (pre-2024 RMSE=%.3f)", winner, lambda_rmse[winner]
    )

    if post_folds:
        winner_params = {**_DEFAULT_PARAMS, "linear_tree": True, "linear_lambda": winner}
        winner_model = LGBMForecaster(objective="quantile", alpha=0.5, params=winner_params)
        post_preds = _get_or_run_backtest(
            "C_winner_post2024", winner_model, post_folds, x, y, preds_dir
        )
        c_preds = pd.concat([lambda_preds[winner], post_preds]).sort_index()
    else:
        c_preds = lambda_preds[winner]

    selection_info: dict[str, float | int] = {"winner_lambda": winner}
    for lam, r in lambda_rmse.items():
        selection_info[f"pre2024_rmse_lambda{lam}"] = r
    return c_preds, selection_info


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
    p.add_argument("--test-start", default="2019-01-01")
    p.add_argument("--test-end", default=None)
    p.add_argument(
        "--out-dir", type=Path, default=Path("outputs/results"), help="Where the CSV tables go."
    )
    p.add_argument(
        "--fig-path",
        type=Path,
        default=Path("outputs/assets/merit_order_variants_rolling_rmse.png"),
    )
    p.add_argument(
        "--preds-dir",
        type=Path,
        default=Path("data/processed"),
        help="Where raw per-variant prediction parquets go (gitignored working artefacts).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    df = load_interim_hourly(args.data_path)
    price = df["day_ahead_price"]
    features = load_processed_features(args.features_path)
    y = price.reindex(features.index)
    x = features
    index = pd.DatetimeIndex(features.index)

    long_history = build_merit_order_frame(df, pd.DatetimeIndex(df.index))

    folds = list(
        walk_forward_splits(
            index,
            test_start=args.test_start,
            test_end=args.test_end,
            window="rolling",
            train_span_days=TRAIN_SPAN_DAYS,
        )
    )
    if not folds:
        raise ValueError(
            "no folds produced -- check --test-start/--test-end against the data range"
        )
    test_start = folds[0].delivery_day
    test_end = folds[-1].delivery_day
    logger.info("folds: %d, delivery days %s to %s", len(folds), test_start.date(), test_end.date())

    models: dict[str, Forecaster] = {
        "baseline": SimilarDayNaive(),
        "A": LGBMForecaster(objective="quantile", alpha=0.5),
        "B": MeritOrderForecaster(long_history, srmc_definition="gas_only"),
    }

    predictions: dict[str, pd.DataFrame] = {}
    args.preds_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        model_x = None if name == "baseline" else x
        predictions[name] = _get_or_run_backtest(name, model, folds, model_x, y, args.preds_dir)

    # B-full needs a fully EUA-covered 90-day training window (srmc_gas_full
    # is NaN before EUA starts, 2021-10-19 here) -- otherwise LGBMForecaster
    # drops every training row and errors on an empty fit. Derived from the
    # data (first valid eua_co2_lag_48h), not a literal date, so a future
    # data refresh can't silently desync this from RL_BIN_EDGES_MW-style
    # hardcoding.
    eua_valid_from_utc = x["eua_co2_lag_48h"].first_valid_index()
    eua_valid_from_local_day = pd.Timestamp(eua_valid_from_utc).tz_convert(LOCAL_TZ).normalize()
    eua_window_cutoff = eua_valid_from_local_day + pd.DateOffset(days=TRAIN_SPAN_DAYS + 1)
    # The beta POOL (long-history side channel) also excludes the crisis
    # period -- and EUA availability (from 2021-10-19) starts inside the
    # crisis window (2021-09-01..2023-04-01), so no non-crisis EUA data
    # exists at all until post_crisis_start. +30 days gives the pool enough
    # accumulated post-crisis hours to clear _MIN_OBS_PER_BIN in >= 2 bins.
    cfg = FeatureConfig()
    crisis_cleared_cutoff = cfg.post_crisis_start + pd.DateOffset(days=30)
    b_full_cutoff = max(eua_window_cutoff, crisis_cleared_cutoff)
    b_full_folds = [f for f in folds if f.delivery_day >= b_full_cutoff]
    logger.info(
        "B_full: restricted to folds from %s onward (%d of %d folds, EUA coverage)",
        b_full_cutoff.date(),
        len(b_full_folds),
        len(folds),
    )

    sensitivity_models: dict[str, tuple[Forecaster, list]] = {
        "B_full": (MeritOrderForecaster(long_history, srmc_definition="gas_full"), b_full_folds),
        "B_rel": (MeritOrderForecaster(long_history, rl_measure="relative"), folds),
        "B_roll": (
            MeritOrderForecaster(long_history, beta_window_days=BETA_ROLL_WINDOW_DAYS),
            folds,
        ),
    }
    for name, (model, model_folds) in sensitivity_models.items():
        predictions[name] = _get_or_run_backtest(name, model, model_folds, x, y, args.preds_dir)

    c_cache = args.preds_dir / "merit_order_preds_C.parquet"
    if c_cache.exists():
        logger.info("variant C: loaded cached predictions from %s", c_cache)
        predictions["C"] = pd.read_parquet(c_cache)
    else:
        c_preds, selection_info = _run_variant_c(y, x, folds, args.preds_dir)
        c_preds.to_parquet(c_cache)
        predictions["C"] = c_preds
        selection_rows = [
            {"linear_lambda": lam, "pre2024_rmse": selection_info[f"pre2024_rmse_lambda{lam}"]}
            for lam in LINEAR_LAMBDAS
        ]
        selection_df = pd.DataFrame(selection_rows)
        selection_df["is_winner"] = selection_df["linear_lambda"] == selection_info["winner_lambda"]
        args.out_dir.mkdir(parents=True, exist_ok=True)
        selection_df.to_csv(
            args.out_dir / "merit_order_variant_c_lambda_selection.csv", index=False
        )

    bin_assignment = rl_bin(x["residual_load_forecast"]).reindex(index)
    oor_flag = ttf_out_of_range_flag(
        long_history["ttf_lag"], index, train_span_days=TRAIN_SPAN_DAYS
    )
    upper_third_mask = bin_assignment.isin(UPPER_THIRD_BINS)
    oor_upper_third_mask = oor_flag.reindex(index).fillna(False) & upper_third_mask.fillna(False)

    periods = _period_bounds(test_start, test_end)

    metrics_rows = []
    bin_rows = []
    oor_rows = []
    for name, preds in predictions.items():
        for period_name, (start, end) in periods.items():
            in_period = (preds["delivery_day"] >= start) & (preds["delivery_day"] <= end)
            sub = preds.loc[in_period]
            if sub.empty:
                continue
            row = {
                "variant": name,
                "period": period_name,
                **_metrics_row(sub["y_true"], sub["y_pred"]),
            }
            metrics_rows.append(row)

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

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_rows).to_csv(
        args.out_dir / "merit_order_variants_metrics.csv", index=False
    )
    pd.DataFrame(bin_rows).to_csv(args.out_dir / "merit_order_variants_by_rlbin.csv", index=False)
    pd.DataFrame(oor_rows).to_csv(
        args.out_dir / "merit_order_variants_oor_upper_third.csv", index=False
    )

    dm_rows = []
    # Squared error, not absolute error: the success criterion (spec 5.7)
    # judges B (and C) against A on RMSE, so the DM-test must target the same
    # loss (squared error) that RMSE itself is an average of.
    loss_a = (predictions["A"]["y_true"] - predictions["A"]["y_pred"]) ** 2
    for challenger in ("B", "C", *SENSITIVITY_NAMES):
        if challenger not in predictions:
            continue
        chal_preds = predictions[challenger]
        loss_challenger = (chal_preds["y_true"] - chal_preds["y_pred"]) ** 2
        # Reindex A's loss (and use the challenger's OWN delivery_day for the
        # period mask) rather than A's index/mask directly -- B_full's
        # predictions only cover a sub-range of A's, and a boolean mask built
        # on a differently-sized index raises on misaligned NaNs instead of
        # silently restricting to the overlap.
        loss_a_aligned = loss_a.reindex(chal_preds.index)
        for period_name, (start, end) in periods.items():
            day_mask = (chal_preds["delivery_day"] >= start) & (chal_preds["delivery_day"] <= end)
            la = loss_a_aligned.loc[day_mask]
            lb = loss_challenger.loc[day_mask]
            aligned = pd.concat([la.rename("a"), lb.rename("b")], axis=1).dropna()
            if len(aligned) < 2 * _HOURLY_HAC_LAG:
                continue
            daily_a, _ = daily_mean_loss(aligned["a"])
            daily_b, _ = daily_mean_loss(aligned["b"])
            daily_result = dm_test(daily_b, daily_a, hac_lag=_DAILY_HAC_LAG, horizon=_DAILY_HORIZON)
            hourly_result = dm_test(
                aligned["b"], aligned["a"], hac_lag=_HOURLY_HAC_LAG, horizon=_HOURLY_HORIZON
            )
            for block, result in (("daily", daily_result), ("hourly", hourly_result)):
                dm_rows.append(
                    {
                        "comparison": f"{challenger}_vs_A",
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
    pd.DataFrame(dm_rows).to_csv(args.out_dir / "merit_order_variants_dm_test.csv", index=False)

    # Rolling 30-day RMSE figure with TTF on a second axis.
    daily_rmse: dict[str, pd.Series] = {}
    for name, preds in predictions.items():
        per_day_sq_err = (
            preds.assign(sq_err=lambda df_: (df_["y_true"] - df_["y_pred"]) ** 2)
            .groupby("delivery_day")["sq_err"]
            .mean()
        )
        daily_rmse[name] = per_day_sq_err.rolling(30, min_periods=15).apply(
            lambda s: s.mean() ** 0.5
        )

    fig, ax1 = plt.subplots(figsize=(11, 5))
    for name, series in daily_rmse.items():
        ax1.plot(series.index, series.to_numpy(), label=name)
    ax1.set_ylabel("rolling 30-day RMSE (EUR/MWh)")
    ax1.set_xlabel("delivery day")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    local_day_full = pd.DatetimeIndex(long_history.index).tz_convert(LOCAL_TZ).normalize()
    daily_ttf = long_history["ttf_lag"].groupby(local_day_full).first()
    ax2.plot(daily_ttf.index, daily_ttf.to_numpy(), color="grey", alpha=0.4, linewidth=0.8)
    ax2.set_ylabel("TTF gas (EUR/MWh, grey)")

    fig.tight_layout()
    args.fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig_path, dpi=150)
    plt.close(fig)

    logger.info(
        "done; wrote metrics/cuts/DM-test CSVs to %s and figure to %s", args.out_dir, args.fig_path
    )


if __name__ == "__main__":
    main()
