"""Main entry point for running the renewables capacity-factor model.

Thin CLI/MLflow wrapper around
model_renewables.evaluation.renewables_walkforward.run_renewables_backtest.
By default, runs the one live variant (`rolling365_l2`) and just evaluates
it -- nothing is written to the canonical feature store
(`data/processed/renewables_forecast_rolling365_l2.parquet`, consumed by
`hourly_model_v2`'s L2 feature matrix) or to the published
`outputs/results/renewables_backtest.csv`; output goes to the gitignored
`data/processed/renewables_runs/` instead. The two comparison variants
(`expanding_l2`, `rolling365_quantile`) stay available via `--variant` for
methodology checks -- "the fourth combination (expanding + quantile) is
never run" is enforced by the tool, not just by convention.

Pass `--store-new` to fold this run's new predictions into the canonical
feature store instead: only valid for `--variant rolling365_l2`, restricts
the run to a thin recent slice by default (just enough trailing history
for the model's own 365-day rolling window), merges in only the rows after
the store's current last `valid_time_utc` (never overwrites the existing
history), and appends to the published metrics CSV. Workflow: refresh the
raw data first (`build_interim.py`/`build_weather_artefact.py`), then run
this with `--store-new`.

Metrics: MAE and RMSE per target, in capacity factor and MW, against the
persistence baseline, plus a coarse daylight/dark and calendar-season
breakdown computed inline here (no new module -- evaluation/metrics.py
stays unmodified). RMSE is the primary metric; MAE is reported alongside,
never decisive.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import time
from pathlib import Path
from typing import cast

import mlflow
import pandas as pd

from energy_price_forecast.common.config import PROJECT_ROOT
from energy_price_forecast.common.data.loaders import (
    load_interim_hourly,
    load_interim_weather,
    load_renewables_predictions,
)
from energy_price_forecast.common.evaluation.metrics import mae, rmse
from energy_price_forecast.common.market_time import LOCAL_TZ
from energy_price_forecast.model_renewables.data.capacity import (
    CapacityExtrapolation,
    CapacitySource,
    ProductionType,
)
from energy_price_forecast.model_renewables.evaluation.renewables_walkforward import (
    TARGET_COLUMNS,
    persistence_baseline_cf,
    run_renewables_backtest,
)
from energy_price_forecast.model_renewables.models.renewables import (
    DEFAULT_LGBM_PARAMS,
    DEFAULT_SEED,
    capacity_factor_label,
)

logger = logging.getLogger(__name__)

_DATA_PATH = Path("data/interim/hourly_merit_order.parquet")
_WEATHER_PATH = Path("data/interim/weather_ifs_run00.parquet")
_RENEWABLES_PATH = Path("data/processed/renewables_forecast_rolling365_l2.parquet")
_SCRATCH_OUT_DIR = PROJECT_ROOT / "data" / "processed" / "renewables_runs"
_RESULTS_PATH = PROJECT_ROOT / "outputs" / "results" / "renewables_backtest.csv"
_STORE_NEW_BUFFER_DAYS = 5  # safety margin on top of train_span_days, DST/off-by-one

# Exactly these three, the fourth combination (expanding + quantile) is
# deliberately never run.
_VARIANTS: dict[str, dict[str, object]] = {
    "rolling365_l2": {
        "window": "rolling",
        "train_span_days": 365,
        "min_history_days": None,
        "objective": "l2",
    },
    "expanding_l2": {
        "window": "expanding",
        "train_span_days": None,
        "min_history_days": 365,
        "objective": "l2",
    },
    "rolling365_quantile": {
        "window": "rolling",
        "train_span_days": 365,
        "min_history_days": None,
        "objective": "quantile",
    },
}


def _baseline_safe_mask(valid_time: pd.DatetimeIndex, label_flat: pd.Series) -> pd.Series:
    """True for hours whose D-1 local wall-clock lookup (``persistence_baseline_cf``'s
    contract) is well-defined: not ambiguous/nonexistent across a DST
    changeover, *and* actually present in ``label_flat`` (the raw ENTSO-E
    series has occasional gaps). Same reindex ``persistence_baseline_cf``
    itself does, just returning the mask instead of raising. The normal
    walk-forward (``require_baseline=True``) already excludes unsafe hours
    per-fold inside ``run_renewables_backtest``; only reachable here via
    ``--store-new``'s deliberate ``require_baseline=False`` live path, which
    keeps a fold's prediction even when it isn't baseline-scorable -- so
    the metrics step (which the live path still runs, for the published
    CSV) needs its own, separate exclusion.
    """
    local_naive = valid_time.tz_convert(LOCAL_TZ).tz_localize(None)
    prev_naive = local_naive - pd.DateOffset(days=1)
    prev_local = prev_naive.tz_localize(LOCAL_TZ, ambiguous="NaT", nonexistent="NaT")
    prev_utc = pd.DatetimeIndex(prev_local).tz_convert("UTC")
    return pd.Series(label_flat.reindex(prev_utc).notna().to_numpy(), index=valid_time)


def _last_valid_local_day(series: pd.Series) -> pd.Timestamp:
    """Last local (Europe/Berlin) calendar day with no missing hours in
    ``series`` -- the renewables-equivalent of
    ``run_price_model.py``'s ``_first_valid_local_day``, just from the other
    end: used here to find the data edge, not the start.
    """
    local_day = pd.DatetimeIndex(series.index).tz_convert(LOCAL_TZ).normalize()
    present = series.notna()
    good_days = present.groupby(local_day).all()
    return good_days.index[good_days][-1]


_SEASON_BY_MONTH = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}  # fmt: skip


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--variant",
        default="rolling365_l2",
        choices=sorted(_VARIANTS),
        help="Default: rolling365_l2, the one live variant. The other two are "
        "comparison-only and invalid together with --store-new.",
    )
    p.add_argument("--refit-every", type=int, default=7)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--source",
        default=CapacitySource.PUBLIC_REGISTRY.value,
        choices=[s.value for s in CapacitySource],
    )
    p.add_argument(
        "--method",
        default=CapacityExtrapolation.LAST_INCREMENT.value,
        choices=[m.value for m in CapacityExtrapolation],
    )
    p.add_argument("--experiment", default="arena_models")
    p.add_argument("--note", default="")
    p.add_argument(
        "--data-path",
        type=Path,
        default=_DATA_PATH,
        help="Interim hourly Parquet to take the targets (wind/solar generation forecast "
        "columns) from. Default is the live-extending file, not the frozen "
        "data/interim/hourly.parquet.",
    )
    p.add_argument("--weather-path", type=Path, default=_WEATHER_PATH)
    p.add_argument(
        "--renewables-path",
        type=Path,
        default=_RENEWABLES_PATH,
        help="The canonical feature-store file --store-new reads from and merges into.",
    )
    p.add_argument(
        "--results-path",
        type=Path,
        default=_RESULTS_PATH,
        help="Published metrics CSV --store-new appends to.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where this run's own predictions+metrics are written when --store-new is "
        "not set. Default: data/processed/renewables_runs/ (gitignored).",
    )
    p.add_argument(
        "--store-new",
        action="store_true",
        help="Fold this run's new predictions into --renewables-path (merge, never "
        "overwrite -- only rows after the store's current last valid_time_utc are "
        "added) and append to the published outputs/results/renewables_backtest.csv. "
        "Only valid with --variant rolling365_l2. Without this flag (the default), "
        "nothing outside --out-dir is touched.",
    )
    p.add_argument(
        "--start",
        default=None,
        help="ISO date, restricts target_hourly. Default: full available history, "
        "unless --store-new and the feature store already exists, in which case it "
        "defaults to just enough trailing history for the rolling window "
        "(train_span_days + a few days' buffer) before the store's current edge.",
    )
    p.add_argument(
        "--end",
        default=None,
        help="ISO date, restricts target_hourly. Default: the data edge (the last "
        "local day with complete target coverage).",
    )
    args = p.parse_args()
    if args.store_new and args.variant != "rolling365_l2":
        p.error("--store-new is only valid with --variant rolling365_l2")
    return args


def _metrics_for_target(
    predictions: pd.DataFrame, target: ProductionType, full_label_flat: pd.Series
) -> list[dict[str, object]]:
    """``full_label_flat`` is the capacity-factor label over the *entire*
    original target_hourly range (not just the walk-forward output), so a
    D-1 lookup for the output's very first evaluated day -- whose D-1 was
    only ever training data, never itself a row in ``predictions`` -- still
    resolves.
    """
    rows: list[dict[str, object]] = []
    cf_actual = predictions[f"{target.value}_cf_actual"]
    cf_pred = predictions[f"{target.value}_cf_pred"]
    mw_actual = predictions[f"{target.value}_mw_actual"]
    mw_pred = predictions[f"{target.value}_mw_pred"]

    valid_time = pd.DatetimeIndex(predictions.index.get_level_values("valid_time_utc"))
    baseline_cf = persistence_baseline_cf(full_label_flat, valid_time)
    baseline_mw = baseline_cf.to_numpy() * predictions[f"{target.value}_capacity_mw"].to_numpy()

    def _add(label: str, mask: pd.Series | None) -> None:
        idx = slice(None) if mask is None else mask.to_numpy()
        rows.append(
            {
                "target": target.value,
                "breakdown": label,
                "n": int(len(cf_actual[idx])),
                "rmse_cf_model": rmse(cf_actual[idx], cf_pred[idx]),
                "mae_cf_model": mae(cf_actual[idx], cf_pred[idx]),
                "rmse_mw_model": rmse(mw_actual[idx], mw_pred[idx]),
                "mae_mw_model": mae(mw_actual[idx], mw_pred[idx]),
                "rmse_cf_baseline": rmse(
                    cf_actual[idx],
                    pd.Series(baseline_cf.to_numpy()[idx], index=cf_actual[idx].index),
                ),
                "mae_cf_baseline": mae(
                    cf_actual[idx],
                    pd.Series(baseline_cf.to_numpy()[idx], index=cf_actual[idx].index),
                ),
                "rmse_mw_baseline": rmse(
                    mw_actual[idx], pd.Series(baseline_mw[idx], index=mw_actual[idx].index)
                ),
                "mae_mw_baseline": mae(
                    mw_actual[idx], pd.Series(baseline_mw[idx], index=mw_actual[idx].index)
                ),
            }
        )

    _add("overall", None)

    daylight = predictions["is_daylight_hour"]
    _add("daylight", daylight)
    _add("dark", ~daylight)

    season = pd.Series(
        valid_time.tz_convert("Europe/Berlin").month.map(_SEASON_BY_MONTH), index=predictions.index
    )
    for season_name in ("winter", "spring", "summer", "autumn"):
        _add(f"season_{season_name}", season == season_name)

    return rows


def _store_new_update(predictions: pd.DataFrame, path: Path) -> int:
    """Merge ``predictions`` into the canonical feature-store ``path``,
    atomically, never overwriting existing rows -- only rows with
    ``valid_time_utc`` after the store's current max are added (the overlap
    a thin restricted run also produces is discarded, same as the Step 7.3
    precedent, ``data/README.md``). Returns the number of rows actually
    added.
    """
    new_rows = predictions.astype("float32", errors="ignore")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        new_rows.to_parquet(tmp_path)
        os.replace(tmp_path, path)
        logger.info(
            "store-new: no existing store at %s -- wrote %d row(s) (first run)", path, len(new_rows)
        )
        return len(new_rows)

    existing = load_renewables_predictions(path=path)
    existing_max = existing.index.get_level_values("valid_time_utc").max()
    new_only = new_rows.loc[new_rows.index.get_level_values("valid_time_utc") > existing_max]
    if new_only.empty:
        logger.info(
            "store-new: no rows after the store's current max (%s) -- already up to date",
            existing_max,
        )
        return 0

    merged = pd.concat([existing, new_only]).sort_index()
    dup = merged.index.get_level_values("valid_time_utc").duplicated()
    if dup.any():
        raise ValueError(
            f"store-new merge produced {int(dup.sum())} duplicate valid_time_utc rows -- "
            "aborting, canonical file NOT written"
        )
    backup_path = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup_path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    merged.to_parquet(tmp_path)
    os.replace(tmp_path, path)
    logger.info(
        "store-new: added %d new row(s) (previous max %s, new max %s); backed up previous "
        "store to %s",
        len(new_only),
        existing_max,
        merged.index.get_level_values("valid_time_utc").max(),
        backup_path,
    )
    return len(new_only)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.monotonic()
    args = _parse_args()
    variant = _VARIANTS[args.variant]

    hourly = load_interim_hourly(args.data_path)
    target_hourly = hourly[list(TARGET_COLUMNS.values())]
    weather = load_interim_weather(args.weather_path)

    edge = min(_last_valid_local_day(target_hourly[col]) for col in TARGET_COLUMNS.values())
    logger.info("data edge (last local day with complete target coverage): %s", edge.date())

    start = args.start
    if start is None and args.store_new and args.renewables_path.exists():
        existing_for_start = load_renewables_predictions(path=args.renewables_path)
        existing_max_vt = existing_for_start.index.get_level_values("valid_time_utc").max()
        # always 365: --store-new is guarded to --variant rolling365_l2 above
        train_span = cast(int, variant["train_span_days"])
        start = (
            existing_max_vt.tz_convert(LOCAL_TZ).normalize()
            - pd.DateOffset(days=train_span + _STORE_NEW_BUFFER_DAYS)
        ).strftime("%Y-%m-%d")
        logger.info(
            "store-new: auto-restricting --start to %s (store's current max %s, "
            "train_span_days=%d + %d day buffer)",
            start,
            existing_max_vt.date(),
            train_span,
            _STORE_NEW_BUFFER_DAYS,
        )
    end = args.end or edge.strftime("%Y-%m-%d")

    if start:
        target_hourly = target_hourly.loc[start:]
    target_hourly = target_hourly.loc[:end]  # type: ignore[misc]

    require_baseline = not args.store_new
    keep_rows_for = None
    if args.store_new:
        keep_rows_for = (
            pd.DatetimeIndex(target_hourly.index).tz_convert(LOCAL_TZ).normalize().max().date()
        )

    logger.info(
        "running variant=%s refit_every=%d seed=%d store_new=%s window=%s..%s",
        args.variant,
        args.refit_every,
        args.seed,
        args.store_new,
        target_hourly.index.min().date(),
        target_hourly.index.max().date(),
    )

    predictions = run_renewables_backtest(
        target_hourly,
        weather,
        window=variant["window"],  # type: ignore[arg-type]
        train_span_days=variant["train_span_days"],  # type: ignore[arg-type]
        min_history_days=variant["min_history_days"],  # type: ignore[arg-type]
        refit_every=args.refit_every,
        objective=variant["objective"],  # type: ignore[arg-type]
        seed=args.seed,
        source=CapacitySource(args.source),
        method=CapacityExtrapolation(args.method),
        keep_rows_for=keep_rows_for,
        require_baseline=require_baseline,
    )

    elapsed = time.monotonic() - t0
    logger.info("walk-forward finished in %.0fs, %d rows", elapsed, len(predictions))

    # --store-new's require_baseline=False can keep a fold whose D-1
    # persistence baseline is undefined (DST changeover, or one of the raw
    # ENTSO-E series' occasional gaps) -- fine for the stored prediction
    # itself, but persistence_baseline_cf below needs a defined D-1 for
    # every row, so those hours are excluded from the metrics computation
    # only (per target: each has its own gaps), never from the stored
    # `predictions`.
    metric_rows: list[dict[str, object]] = []
    for target in ProductionType:
        raw = target_hourly[TARGET_COLUMNS[target]]
        available = raw.notna()
        full_label = capacity_factor_label(
            raw[available],
            pd.DatetimeIndex(raw.index[available]),
            target,
            source=CapacitySource(args.source),
            method=CapacityExtrapolation(args.method),
        )
        full_label_flat = pd.Series(
            full_label.to_numpy(), index=pd.DatetimeIndex(raw.index[available])
        )
        valid_time = pd.DatetimeIndex(predictions.index.get_level_values("valid_time_utc"))
        safe = _baseline_safe_mask(valid_time, full_label_flat)
        predictions_for_target = predictions
        if not safe.all():
            logger.info(
                "%s: excluding %d hour(s) from metrics (D-1 baseline undefined); still "
                "included in the stored predictions",
                target.value,
                int((~safe).sum()),
            )
            predictions_for_target = predictions.loc[safe.to_numpy()]
        metric_rows.extend(_metrics_for_target(predictions_for_target, target, full_label_flat))
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.insert(0, "variant", args.variant)

    logger.info("PRIMARY METRIC: RMSE. MAE is a secondary metric, reported but not decisive.")
    overall = metrics_df[metrics_df["breakdown"] == "overall"]
    for row in overall.itertuples():
        logger.info(
            "%-14s RMSE(cf) model=%.4f baseline=%.4f | RMSE(MW) model=%.1f baseline=%.1f | MAE(cf) model=%.4f baseline=%.4f",
            row.target,
            row.rmse_cf_model,
            row.rmse_cf_baseline,
            row.rmse_mw_model,
            row.rmse_mw_baseline,
            row.mae_cf_model,
            row.mae_cf_baseline,
        )

    # --- Outputs: canonical store (--store-new) vs. gitignored scratch -----
    out_dir = args.out_dir or _SCRATCH_OUT_DIR
    if args.store_new:
        n_new = _store_new_update(predictions, args.renewables_path)
        predictions_out_path = args.renewables_path
        args.results_path.parent.mkdir(parents=True, exist_ok=True)
        header = not args.results_path.exists()
        metrics_df.to_csv(
            args.results_path, mode="a" if not header else "w", header=header, index=False
        )
        metrics_out_path = args.results_path
        logger.info("appended %d metric row(s) to %s", len(metrics_df), args.results_path)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        n_new = len(predictions)
        predictions_out_path = out_dir / f"renewables_forecast_{args.variant}.parquet"
        predictions.astype("float32", errors="ignore").to_parquet(predictions_out_path)
        metrics_out_path = out_dir / f"renewables_metrics_{args.variant}.csv"
        metrics_df.to_csv(metrics_out_path, index=False)
        logger.info("wrote %d row(s) to %s", len(predictions), predictions_out_path)

    log_params: dict[str, object] = {
        "variant": args.variant,
        "window": variant["window"],
        "train_span_days": variant["train_span_days"],
        "min_history_days": variant["min_history_days"],
        "objective": variant["objective"],
        "refit_every": args.refit_every,
        "seed": args.seed,
        "source": args.source,
        "method": args.method,
        "store_new": args.store_new,
        "start": start,
        "end": end,
        **DEFAULT_LGBM_PARAMS,
    }
    log_tags: dict[str, str] = {"study": f"renewables_{args.variant}", "note": args.note}
    log_metrics: dict[str, float] = {
        f"{row.target}_{metric}": getattr(row, metric)
        for row in overall.itertuples()
        for metric in (
            "rmse_cf_model",
            "rmse_mw_model",
            "mae_cf_model",
            "mae_mw_model",
            "rmse_cf_baseline",
            "rmse_mw_baseline",
        )  # fmt: skip
    }
    log_metrics["elapsed_seconds"] = elapsed

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(args.experiment)
    run_name = (
        f"renewables_{args.variant}_store_new" if args.store_new else f"renewables_{args.variant}"
    )
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(log_params)
        mlflow.set_tags(log_tags)
        mlflow.log_metrics(log_metrics)
        mlflow.log_artifact(str(predictions_out_path))
        mlflow.log_artifact(str(metrics_out_path))

    logger.info("done in %.0fs", elapsed)

    # --- Printed summary: the numbers and file paths, front and center -----
    print()
    print("=" * 78)
    print(
        f"RESULTS  variant={args.variant}  store_new={args.store_new}  "
        f"window={target_hourly.index.min().date()}..{target_hourly.index.max().date()}"
    )
    print("-" * 78)
    for row in overall.itertuples():
        print(
            f"  {row.target}:  RMSE(cf)={row.rmse_cf_model:.4f} "
            f"(baseline {row.rmse_cf_baseline:.4f})  MAE(cf)={row.mae_cf_model:.4f}  (n={row.n})"
        )
    print("-" * 78)
    print("Files written:")
    print(f"  {predictions_out_path}")
    print(f"  {metrics_out_path}")
    if args.store_new:
        print(f"  (canonical feature store: +{n_new} new row(s))")
    print(f"MLflow: experiment={args.experiment!r}  run={run_name!r}  run_id={run.info.run_id}")
    print("  browse with: uv run mlflow ui --backend-store-uri file:./mlruns")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
