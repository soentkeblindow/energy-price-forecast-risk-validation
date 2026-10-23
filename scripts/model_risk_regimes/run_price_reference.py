"""Sprint 7 Step 7.7, Teil C (docs/sprint7_step7_7_spec.md, section 4).

Reference recomputation for `price_v1`/`price_v2` (`common.models.price_model`):
four runs on an identical fold list, two rungs each --

  L1  core_gas with ENTSO-E day-ahead forecasts (upper bound: not available
      live)
  L2  same columns, wind/solar/residual-load/renewable-share from the own
      NWP reconstruction (matches the live information state)

No model change, no new measurement, no tuning -- this only proves the
Step 7.7 code simplification (Teil B) did not change price_v2's L1
numbers, which must match Step 7.6's own cached predictions exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from energy_price_forecast.common.data.loaders import load_interim_hourly, load_processed_features
from energy_price_forecast.common.data.snapshot import file_fingerprint
from energy_price_forecast.common.evaluation.dm_test import daily_mean_loss, dm_test
from energy_price_forecast.common.evaluation.metrics import bias, mae, rmse
from energy_price_forecast.common.evaluation.walkforward import (
    Fold,
    last_complete_local_day,
    split_complete_delivery_days,
    walk_forward_splits,
)
from energy_price_forecast.common.features.core_gas_columns import L1_COLUMNS
from energy_price_forecast.common.features.merit_order import (
    build_merit_order_frame,
    rl_bin,
    ttf_out_of_range_flag,
)
from energy_price_forecast.common.market_time import LOCAL_TZ, month_start_local
from energy_price_forecast.common.models.price_model import PRICE_V1, PRICE_V2, build_price_model
from energy_price_forecast.model_risk_regimes.features.l2_matrix import build_l2_matrix

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TRAIN_SPAN_DAYS = 90
MAX_PREDICTIONS_MB = 5.0

PERIODS = ("overall", "2024", "2025", "2026", "autumn_winter", "autumn_2026", "october_2026")
BIN_STRESS_PERIODS = ("overall", "autumn_2026")
OUTLIER_DAYS = ("2026-10-11", "2026-10-18", "2026-10-19")
UPPER_THIRD_BINS = (8, 9, 10)

_DAILY_HAC_LAG = 7
_DAILY_HORIZON = 1
_HOURLY_HAC_LAG = 48
_HOURLY_HORIZON = 24

A2_INPUT_FILES: tuple[Path, ...] = (
    Path("data/interim/hourly_merit_order.parquet"),
    Path("data/processed/features_merit_order.parquet"),
)
RENEWABLES_FILE = Path("data/processed/renewables_forecast_rolling365_l2.parquet")


def _renewables_fingerprint(path: Path) -> dict[str, object]:
    """Like ``file_fingerprint``, but for this one file's
    ``(run_init_utc, valid_time_utc)`` MultiIndex -- the time range is taken
    from the ``valid_time_utc`` level, since that is the index
    ``file_fingerprint``'s generic (plain-DatetimeIndex) contract assumes.
    """
    df = pd.read_parquet(path)
    valid_time = df.index.get_level_values("valid_time_utc")
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "filename": path.name,
        "first_timestamp": valid_time.min().isoformat(),
        "last_timestamp": valid_time.max().isoformat(),
        "row_count": len(df),
        "sha256": digest.hexdigest(),
    }


def _incomplete_local_days(frame: pd.DataFrame) -> set[pd.Timestamp]:
    local_day = pd.DatetimeIndex(frame.index).tz_convert(LOCAL_TZ).normalize()
    any_nan = frame.isna().any(axis=1)
    bad = any_nan.groupby(local_day).any()
    return set(bad.index[bad])


def _first_valid_local_day(frame: pd.DataFrame, columns: list[str]) -> pd.Timestamp:
    local_day = pd.DatetimeIndex(frame.index).tz_convert(LOCAL_TZ).normalize()
    all_present = frame[columns].notna().all(axis=1)
    good_days = all_present.groupby(local_day).all()
    return good_days.index[good_days][0]


def _run_reference_backtest(
    y: pd.Series, model: object, folds: list[Fold], x: pd.DataFrame
) -> pd.DataFrame:
    """Drive ``model`` through ``folds`` (daily refit, matching the fixed
    refit cadence both price_v1/price_v2 use). If ``model`` exposes
    ``predict_components`` (price_v2's MeritOrderForecaster), captures
    tree_pred/beta/srmc/rl_bin/anchor_month alongside y_pred directly from
    the same pass -- no second, post-hoc computation.
    """
    has_components = hasattr(model, "predict_components")
    records: list[pd.DataFrame] = []
    for fold in folds:
        history = y.loc[fold.train_index]
        x_train = x.loc[fold.train_index]
        model.fit(history, x_train)  # type: ignore[attr-defined]
        x_test = x.loc[fold.test_index]
        if has_components:
            comp = model.predict_components(  # type: ignore[attr-defined]
                fold.test_index, history=history, x_test=x_test
            )
            row = comp.assign(
                y_true=y.loc[fold.test_index].to_numpy(),
                delivery_day=fold.delivery_day,
                anchor_month=month_start_local(fold.delivery_day).strftime("%Y-%m"),
            )
        else:
            y_pred = model.predict(  # type: ignore[attr-defined]
                fold.test_index, history=history, x_test=x_test
            )
            row = pd.DataFrame(
                {
                    "y_pred": np.asarray(y_pred),
                    "y_true": y.loc[fold.test_index].to_numpy(),
                    "delivery_day": fold.delivery_day,
                    "tree_pred": np.nan,
                    "beta": np.nan,
                    "srmc": np.nan,
                    "rl_bin": np.nan,
                    "anchor_month": np.nan,
                },
                index=fold.test_index,
            )
        records.append(row)
    return pd.concat(records).sort_index()


def _get_or_run(
    name: str, model: object, folds: list[Fold], x: pd.DataFrame, y: pd.Series, preds_dir: Path
) -> pd.DataFrame:
    cache_path = preds_dir / f"price_reference_preds_{name}.parquet"
    if cache_path.exists():
        logger.info("%s: loaded cached predictions from %s", name, cache_path)
        return pd.read_parquet(cache_path)
    t0 = time.monotonic()
    preds = _run_reference_backtest(y, model, folds, x)
    elapsed = time.monotonic() - t0
    logger.info(
        "%s: %d test hours in %.1fs (%.3fs/fold)", name, len(preds), elapsed, elapsed / len(folds)
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(cache_path)
    return preds


def _period_mask(delivery_day: pd.Series, period: str) -> pd.Series:
    if period == "overall":
        return pd.Series(True, index=delivery_day.index)
    if period in ("2024", "2025", "2026"):
        return delivery_day.dt.year == int(period)
    if period == "autumn_winter":
        return delivery_day.dt.month.isin([9, 10, 11, 12])
    if period == "autumn_2026":
        return delivery_day >= pd.Timestamp("2026-09-01", tz=LOCAL_TZ)
    if period == "october_2026":
        return (delivery_day >= pd.Timestamp("2026-10-01", tz=LOCAL_TZ)) & (
            delivery_day <= pd.Timestamp("2026-10-31", tz=LOCAL_TZ)
        )
    raise ValueError(f"unknown period {period!r}")


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
    p.add_argument("--renewables-path", type=Path, default=RENEWABLES_FILE)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/results/price_v2"))
    p.add_argument("--preds-dir", type=Path, default=Path("data/processed/price_reference_preds"))
    p.add_argument(
        "--snapshot-csv",
        type=Path,
        default=Path("outputs/results/price_v2/data_snapshot.csv"),
        help="Compared against before the run if present (this script's own "
        "previous data_snapshot.csv output); skipped on a first run.",
    )
    p.add_argument(
        "--b-full-roll-cache",
        type=Path,
        default=Path(
            "data/processed/price_model_selection_preds/price_model_selection_preds_B_full_roll.parquet"
        ),
    )
    p.add_argument(
        "--a-cache",
        type=Path,
        default=Path(
            "data/processed/price_model_selection_preds/price_model_selection_preds_A.parquet"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # --- Hash check against this script's own last recorded data stand -----
    if args.snapshot_csv.exists():
        snapshot = pd.read_csv(args.snapshot_csv, index_col="filename")
        for path in A2_INPUT_FILES:
            current = file_fingerprint(path)
            recorded = snapshot.loc[path.name, "sha256"]
            if current["sha256"] != recorded:
                logger.error(
                    "RUECKFRAGE: data stand changed since the last recorded run for %s -- stopping.",
                    path,
                )
                return
        logger.info("hash check passed against %s", args.snapshot_csv)
    else:
        logger.info("no prior %s -- first run, skipping the hash check", args.snapshot_csv)
    renewables_fp = _renewables_fingerprint(args.renewables_path)
    logger.info("renewables file fingerprint captured: %s", renewables_fp["sha256"])

    df = load_interim_hourly(args.data_path)
    price = df["day_ahead_price"]
    features = load_processed_features(args.features_path)
    y = price.reindex(features.index)
    index = pd.DatetimeIndex(features.index)

    l1_matrix = features[list(L1_COLUMNS)]
    renewables_predictions = pd.read_parquet(args.renewables_path)
    l2_matrix = build_l2_matrix(l1_matrix, renewables_predictions, index)
    l1_matrix_v2 = l1_matrix.assign(eua_co2_lag_48h=features["eua_co2_lag_48h"])
    l2_matrix_v2 = l2_matrix.assign(eua_co2_lag_48h=features["eua_co2_lag_48h"])
    long_history = build_merit_order_frame(df, pd.DatetimeIndex(df.index))

    edge = last_complete_local_day(price)
    logger.info("data edge: %s", edge.date())

    # --- Common L1/L2 window (same derivation as 7.3) -----------------------
    l2_first_valid_day = _first_valid_local_day(
        l2_matrix, ["wind_onshore_forecast", "wind_offshore_forecast", "solar_forecast"]
    )
    logger.info("L2 first fully-covered local day: %s", l2_first_valid_day.date())
    excluded_days = _incomplete_local_days(l1_matrix) | _incomplete_local_days(l2_matrix)
    candidate_start = l2_first_valid_day + pd.DateOffset(days=TRAIN_SPAN_DAYS + 1)
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
        raise ValueError("no evaluable folds in the common L1/L2 window")
    logger.info(
        "common window: %s to %s (%d folds, identical for all 4 runs)",
        folds[0].delivery_day.date(),
        folds[-1].delivery_day.date(),
        len(folds),
    )

    # --- Four runs -----------------------------------------------------------
    args.preds_dir.mkdir(parents=True, exist_ok=True)
    runs = {
        ("price_v1", "L1"): (build_price_model(PRICE_V1, long_history), l1_matrix),
        ("price_v1", "L2"): (build_price_model(PRICE_V1, long_history), l2_matrix),
        ("price_v2", "L1"): (build_price_model(PRICE_V2, long_history), l1_matrix_v2),
        ("price_v2", "L2"): (build_price_model(PRICE_V2, long_history), l2_matrix_v2),
    }
    predictions: dict[tuple[str, str], pd.DataFrame] = {}
    for (version, rung), (model, x) in runs.items():
        name = f"{version}_{rung}"
        preds = _get_or_run(name, model, folds, x, y, args.preds_dir)
        complete_days, incomplete_days = split_complete_delivery_days(preds)
        if len(incomplete_days) > 0:
            logger.info("%s: dropping %d incomplete delivery day(s)", name, len(incomplete_days))
        predictions[(version, rung)] = preds.loc[preds["delivery_day"].isin(complete_days)]

    fold_counts = {k: v["delivery_day"].nunique() for k, v in predictions.items()}
    logger.info("delivery-day counts per run (must match): %s", fold_counts)

    # --- C.2 Equality checks against Step 7.6's caches -----------------------
    # One-time migration proof (Teil B's code simplification changed nothing)
    # -- the gitignored 7.6 prediction caches this compares against are local
    # working artefacts, not guaranteed to exist on a later run or a fresh
    # checkout, so this step is skipped gracefully rather than failing.
    if args.b_full_roll_cache.exists() and args.a_cache.exists():
        b_full_roll_cache = pd.read_parquet(args.b_full_roll_cache)
        a_cache = pd.read_parquet(args.a_cache)
        for label, new_preds, old_cache in (
            (
                "price_v2 L1 vs B_full_roll (7.6)",
                predictions[("price_v2", "L1")],
                b_full_roll_cache,
            ),
            ("price_v1 L1 vs A (7.6)", predictions[("price_v1", "L1")], a_cache),
        ):
            common_index = new_preds.index.intersection(old_cache.index)
            diff = (
                new_preds.loc[common_index, "y_pred"] - old_cache.loc[common_index, "y_pred"]
            ).abs()
            max_diff = float(diff.max())
            logger.info(
                "equality check [%s]: max |diff| = %.3e over %d common hours -- %s",
                label,
                max_diff,
                len(common_index),
                "PASS" if max_diff <= 1e-9 else "RUECKFRAGE: FAILED",
            )
    else:
        logger.info(
            "no Step 7.6 prediction caches found (%s / %s) -- skipping the one-time "
            "migration equality check",
            args.b_full_roll_cache,
            args.a_cache,
        )

    # --- Determinism check: price_v2 L2, first 20 folds, run twice ----------
    probe_folds = folds[:20]
    model_a = build_price_model(PRICE_V2, long_history)
    model_b = build_price_model(PRICE_V2, long_history)
    probe_a = _run_reference_backtest(y, model_a, probe_folds, l2_matrix_v2)
    probe_b = _run_reference_backtest(y, model_b, probe_folds, l2_matrix_v2)
    determinism_diff = float((probe_a["y_pred"] - probe_b["y_pred"]).abs().max())
    logger.info(
        "determinism check [price_v2 L2, first 20 folds]: max |diff| = %.3e -- %s",
        determinism_diff,
        "PASS" if determinism_diff == 0.0 else "RUECKFRAGE: FAILED",
    )

    # --- C.4 outputs ----------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_rows = [file_fingerprint(p) for p in A2_INPUT_FILES] + [renewables_fp]
    pd.DataFrame(snapshot_rows).to_csv(args.out_dir / "data_snapshot.csv", index=False)

    bin_assignment_l1 = rl_bin(l1_matrix["residual_load_forecast"]).reindex(index)
    bin_assignment_l2 = rl_bin(l2_matrix["residual_load_forecast"]).reindex(index)
    bin_assignment_by_rung = {"L1": bin_assignment_l1, "L2": bin_assignment_l2}
    oor_flag_l1 = ttf_out_of_range_flag(
        long_history["ttf_lag"], index, train_span_days=TRAIN_SPAN_DAYS
    )

    metrics_rows = []
    for (version, rung), preds in predictions.items():
        bin_assignment = bin_assignment_by_rung[rung]
        for period in PERIODS:
            in_period = _period_mask(preds["delivery_day"], period)
            sub = preds.loc[in_period]
            if sub.empty:
                continue
            metrics_rows.append(
                {
                    "model_version": version,
                    "rung": rung,
                    "breakdown": "period",
                    "group": period,
                    **_metrics_row(sub["y_true"], sub["y_pred"]),
                }
            )
            if period in BIN_STRESS_PERIODS:
                bins_here = bin_assignment.reindex(sub.index)
                for bin_label in sorted(bins_here.dropna().unique()):
                    cell = sub.loc[bins_here == bin_label]
                    if cell.empty:
                        continue
                    metrics_rows.append(
                        {
                            "model_version": version,
                            "rung": rung,
                            "breakdown": "rl_bin",
                            "group": f"{period}/{int(bin_label)}",
                            **_metrics_row(cell["y_true"], cell["y_pred"]),
                        }
                    )
                upper_third_mask = bin_assignment.isin(UPPER_THIRD_BINS)
                oor_upper_third_mask = oor_flag_l1.reindex(index).fillna(
                    False
                ) & upper_third_mask.fillna(False)
                stress_here = oor_upper_third_mask.reindex(sub.index).fillna(False)
                stress_cell = sub.loc[stress_here]
                if not stress_cell.empty:
                    metrics_rows.append(
                        {
                            "model_version": version,
                            "rung": rung,
                            "breakdown": "stress",
                            "group": period,
                            **_metrics_row(stress_cell["y_true"], stress_cell["y_pred"]),
                        }
                    )

        month_group = preds["delivery_day"].dt.strftime("%Y-%m")
        for month, cell in preds.groupby(month_group):
            metrics_rows.append(
                {
                    "model_version": version,
                    "rung": rung,
                    "breakdown": "month",
                    "group": month,
                    **_metrics_row(cell["y_true"], cell["y_pred"]),
                }
            )

        weekday_group = preds["delivery_day"].dt.strftime("%a")
        for weekday, cell in preds.groupby(weekday_group):
            metrics_rows.append(
                {
                    "model_version": version,
                    "rung": rung,
                    "breakdown": "weekday",
                    "group": weekday,
                    **_metrics_row(cell["y_true"], cell["y_pred"]),
                }
            )

        for day in OUTLIER_DAYS:
            day_ts = pd.Timestamp(day, tz=LOCAL_TZ)
            cell = preds.loc[preds["delivery_day"] == day_ts]
            if cell.empty:
                continue
            metrics_rows.append(
                {
                    "model_version": version,
                    "rung": rung,
                    "breakdown": "day",
                    "group": day,
                    **_metrics_row(cell["y_true"], cell["y_pred"]),
                }
            )

    pd.DataFrame(metrics_rows).to_csv(args.out_dir / "reference_metrics.csv", index=False)

    # --- C.5 DM-tests: price_v2 vs price_v1 (L1, L2); L2 vs L1 (each version) ---
    dm_rows = []
    dm_periods = ("overall", "autumn_winter", "autumn_2026")
    comparisons: list[tuple[str, tuple[str, str], tuple[str, str]]] = [
        ("price_v2_vs_price_v1_L1", ("price_v2", "L1"), ("price_v1", "L1")),
        ("price_v2_vs_price_v1_L2", ("price_v2", "L2"), ("price_v1", "L2")),
        ("L2_vs_L1_price_v1", ("price_v1", "L2"), ("price_v1", "L1")),
        ("L2_vs_L1_price_v2", ("price_v2", "L2"), ("price_v2", "L1")),
    ]
    for comparison_name, challenger_key, baseline_key in comparisons:
        challenger = predictions[challenger_key]
        baseline = predictions[baseline_key]
        loss_c = (challenger["y_true"] - challenger["y_pred"]) ** 2
        loss_b = (baseline["y_true"] - baseline["y_pred"]) ** 2
        periods_to_test = ("overall",) if comparison_name.startswith("L2_vs_L1") else dm_periods
        for period in periods_to_test:
            in_period = _period_mask(challenger["delivery_day"], period)
            lc = loss_c.loc[in_period]
            lb = loss_b.reindex(lc.index)
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
    pd.DataFrame(dm_rows).to_csv(args.out_dir / "reference_dm_test.csv", index=False)

    # --- reference_predictions.parquet ---------------------------------------
    export_rows = []
    for (version, rung), preds in predictions.items():
        row = pd.DataFrame(
            {
                "timestamp_utc": preds.index,
                "delivery_day": preds["delivery_day"].to_numpy(),
                "model_version": version,
                "rung": rung,
                "y_true": preds["y_true"].to_numpy(),
                "y_pred": preds["y_pred"].to_numpy(),
                "anchor_month": preds.get(
                    "anchor_month", pd.Series(np.nan, index=preds.index)
                ).to_numpy(),
                "rl_bin": preds.get("rl_bin", pd.Series(np.nan, index=preds.index)).to_numpy(),
                "beta": preds.get("beta", pd.Series(np.nan, index=preds.index)).to_numpy(),
                "srmc": preds.get("srmc", pd.Series(np.nan, index=preds.index)).to_numpy(),
                "tree_pred": preds.get(
                    "tree_pred", pd.Series(np.nan, index=preds.index)
                ).to_numpy(),
            }
        )
        export_rows.append(row)
    export = pd.concat(export_rows, ignore_index=True)
    export_path = args.out_dir / "reference_predictions.parquet"
    export.to_parquet(export_path)
    size_mb = export_path.stat().st_size / (1024 * 1024)
    logger.info("reference_predictions.parquet: %.2f MB (%d rows)", size_mb, len(export))
    if size_mb > MAX_PREDICTIONS_MB:
        logger.error(
            "RUECKFRAGE: reference_predictions.parquet is %.2f MB, over the %.1f MB budget.",
            size_mb,
            MAX_PREDICTIONS_MB,
        )

    # --- Parity check: tree_pred + beta*srmc == y_pred for price_v2 ---------
    for rung in ("L1", "L2"):
        preds = predictions[("price_v2", rung)]
        reconstructed = preds["tree_pred"] + preds["beta"] * preds["srmc"]
        max_parity_diff = float((reconstructed - preds["y_pred"]).abs().max())
        logger.info(
            "parity check [price_v2 %s]: max |tree_pred + beta*srmc - y_pred| = %.3e",
            rung,
            max_parity_diff,
        )

    logger.info(
        "done; wrote 4 files to %s (beta_monthly.csv comes from build_price_beta_table.py)",
        args.out_dir,
    )


if __name__ == "__main__":
    main()
