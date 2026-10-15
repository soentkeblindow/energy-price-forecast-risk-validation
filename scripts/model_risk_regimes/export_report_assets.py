"""Sprint 5.1: reproducible export of report/dashboard figures and tables.

Thin I/O layer only -- all preparation logic lives in
``energy_price_forecast.model_risk_regimes.reporting`` (pure functions, tested against
fixtures in ``tests/model_risk_regimes/test_reporting.py``). This script loads already
PERSISTED Sprint 1-4 outputs, calls those functions, and writes PNGs/CSVs/
the prediction snapshot to ``--output-dir`` (default ``outputs/``).

Reads ONLY persisted files -- no model runs, no backtests, no MLflow
queries at runtime (Decision 1). If an input is missing, fails fast and
loud, naming the script that produces it.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

from energy_price_forecast.common.data.loaders import load_interim_hourly  # noqa: E402
from energy_price_forecast.common.evaluation.config import BacktestConfig  # noqa: E402
from energy_price_forecast.common.evaluation.regimes import tag_regimes  # noqa: E402
from energy_price_forecast.model_risk_regimes.reporting import (  # noqa: E402
    assets,
    snapshot,
    tables,
)

log = logging.getLogger(__name__)

_DPI = 150
_MAX_RESULTS_MB = 15.0

# Every input this script reads (filename under --data-dir, tagged with the
# script/command that produces it) -- used to build one collected, actionable
# error message if something is missing (spec 5.1 §7).
_INPUTS: dict[str, tuple[str, str]] = {
    "backtest_similarday": (
        "backtest_similarday.parquet",
        "uv run python scripts/model_risk_regimes/backtest.py --model naive",
    ),
    "backtest_lasso": (
        "backtest_lasso.parquet",
        "uv run python scripts/model_risk_regimes/backtest.py --model lasso",
    ),
    # Pinned deliberately: preds_lgbm_q50.parquet, NOT backtest_lgbm.parquet
    # -- a different, older LightGBM run (spec 5.1 §3.2).
    "preds_lgbm_q50": (
        "preds_lgbm_q50.parquet",
        "uv run python scripts/model_risk_regimes/backtest.py --model lgbm --alpha 0.5",
    ),
    # Pinned: the v2 ARIMAX run, not the original preds_arimax_q50.parquet.
    "preds_arimax_v2_q50": (
        "preds_arimax_v2_q50.parquet",
        "uv run python scripts/model_risk_regimes/backtest.py --model arimax --study arimax_benchmark_v2",
    ),
    "preds_lgbm_q05": (
        "preds_lgbm_q05.parquet",
        "uv run python scripts/model_risk_regimes/backtest.py --model lgbm --alpha 0.05",
    ),
    "preds_lgbm_q95": (
        "preds_lgbm_q95.parquet",
        "uv run python scripts/model_risk_regimes/backtest.py --model lgbm --alpha 0.95",
    ),
    "preds_lgbm_q05_calibrated_sorted": (
        "preds_lgbm_q05_calibrated_sorted.parquet",
        "uv run python scripts/model_risk_regimes/calibrate_conformal.py",
    ),
    "preds_lgbm_q95_calibrated_sorted": (
        "preds_lgbm_q95_calibrated_sorted.parquet",
        "uv run python scripts/model_risk_regimes/calibrate_conformal.py",
    ),
    "conformal_reliability_raw": (
        "conformal_reliability_raw.csv",
        "uv run python scripts/model_risk_regimes/calibrate_conformal.py",
    ),
    "conformal_reliability_sorted": (
        "conformal_reliability_sorted.csv",
        "uv run python scripts/model_risk_regimes/calibrate_conformal.py",
    ),
    "reliability_curve": (
        "reliability_curve.csv",
        "uv run python scripts/model_risk_regimes/evaluate_reliability.py",
    ),
    "backtest_coverage": (
        "backtest_coverage.csv",
        "uv run python scripts/model_risk_regimes/backtest_risk.py",
    ),
    "risk_summary": (
        "risk_summary.csv",
        "uv run python scripts/model_risk_regimes/compute_risk.py",
    ),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export report/dashboard figures, tables, and the prediction snapshot."
    )
    p.add_argument("--data-dir", default=Path("data/processed"), type=Path)
    p.add_argument("--interim-path", default=Path("data/interim/hourly.parquet"), type=Path)
    p.add_argument("--output-dir", default=Path("outputs"), type=Path)
    # Fan-chart example window -- proposed default, confirm/override with the owner
    # (spec 5.1 §11 point 5): the highest-spread day in the test period
    # (2024-12-12, ~829 EUR/MWh spread, a documented Dunkelflaute event),
    # embedded in a two-week window that also shows calmer days around it.
    p.add_argument(
        "--fan-chart-start", default="2024-12-05", help="Fan-chart window start (local date)."
    )
    p.add_argument(
        "--fan-chart-end",
        default="2024-12-19",
        help="Fan-chart window end (local date, exclusive).",
    )
    return p.parse_args()


def _require_inputs(data_dir: Path) -> dict[str, Path]:
    """Resolve every `_INPUTS` entry under `data_dir`, collecting ALL missing
    files into one error (not just the first) -- spec 5.1 §7.
    """
    resolved: dict[str, Path] = {}
    missing: list[str] = []
    for name, (filename, producer) in _INPUTS.items():
        path = data_dir / filename
        if not path.exists():
            missing.append(f"  {name}: {path} (run: {producer})")
        else:
            resolved[name] = path
    if missing:
        raise FileNotFoundError("Missing required input file(s):\n" + "\n".join(missing))
    return resolved


def _load_parquet(path: Path, required_columns: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = [c for c in required_columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing expected columns {missing} (got {list(frame.columns)})")
    return frame


def _load_csv(path: Path, required_columns: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [c for c in required_columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing expected columns {missing} (got {list(frame.columns)})")
    return frame


def _reliability_curve_for_plot(frame: pd.DataFrame, *, bucket_col: str = "bucket") -> pd.DataFrame:
    """`bucket == "overall"` rows, indexed by level, renamed to the `empirical`
    column `plot_reliability` expects.
    """
    overall = frame[frame[bucket_col] == "overall"].set_index("level")
    return overall[["coverage"]].rename(columns={"coverage": "empirical"})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()

    inputs = _require_inputs(args.data_dir)

    assets_dir = args.output_dir / "assets"
    results_dir = args.output_dir / "results"
    assets_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Load ---------------------------------------------------------------
    naive = _load_parquet(inputs["backtest_similarday"], ("y_true", "y_pred", "delivery_day"))
    lasso = _load_parquet(inputs["backtest_lasso"], ("y_true", "y_pred", "delivery_day"))
    lgbm_q50 = _load_parquet(inputs["preds_lgbm_q50"], ("y_true", "y_pred", "delivery_day"))
    arimax_q50 = _load_parquet(inputs["preds_arimax_v2_q50"], ("y_true", "y_pred", "delivery_day"))

    lgbm_q05 = _load_parquet(inputs["preds_lgbm_q05"], ("y_pred",))
    lgbm_q95 = _load_parquet(inputs["preds_lgbm_q95"], ("y_pred",))
    lgbm_q05_cal = _load_parquet(inputs["preds_lgbm_q05_calibrated_sorted"], ("y_pred",))
    lgbm_q95_cal = _load_parquet(inputs["preds_lgbm_q95_calibrated_sorted"], ("y_pred",))

    conformal_raw = _load_csv(
        inputs["conformal_reliability_raw"], ("bucket", "level", "coverage", "n")
    )
    conformal_sorted = _load_csv(
        inputs["conformal_reliability_sorted"], ("bucket", "level", "coverage", "n")
    )
    reliability_curve_df = _load_csv(
        inputs["reliability_curve"], ("bucket", "level", "coverage", "n", "model")
    )
    backtest_coverage = _load_csv(
        inputs["backtest_coverage"], ("variant", "side", "subset", "breach_rate", "low_support")
    )
    risk_summary = _load_csv(inputs["risk_summary"], ("variant", "side", "mean_var"))

    # tag_regimes needs the FULL interim history (the price-spike threshold is a
    # per-macro-regime quantile over the whole frame). `full_regime_flags` (unreindexed)
    # feeds `breakdown_point` below (Sprint 5.3.1); the snapshot join further down needs
    # it restricted to the prediction period instead -- interim spans the pre-test
    # training history too, which would otherwise look like a large, spurious index
    # mismatch to build_snapshot (known gotcha from Sprint 4.5).
    interim = load_interim_hourly(args.interim_path)
    full_regime_flags = tag_regimes(interim)

    # --- Tables ---------------------------------------------------------------
    predictions_by_model = {
        "Naive": naive,
        "Lasso": lasso,
        "LightGBM": lgbm_q50,
        "ARIMAX": arimax_q50,
    }
    model_comparison_table = tables.model_comparison(predictions_by_model)
    model_comparison_by_regime_table = tables.model_comparison_by_regime(
        predictions_by_model, full_regime_flags
    )
    tables.assert_model_comparison_reconciles(
        model_comparison_table, model_comparison_by_regime_table
    )
    coverage_summary_table = tables.coverage_summary(
        conformal_raw, conformal_sorted, reliability_curve_df
    )
    backtest_coverage_table = tables.backtest_coverage_export(backtest_coverage)
    risk_headline_table = tables.risk_headline(backtest_coverage, risk_summary)

    # lineterminator="\n": Windows' pandas default writes "\r\n", which the repo's
    # mixed-line-ending pre-commit hook (--fix=lf) would otherwise rewrite every run.
    model_comparison_table.to_csv(
        results_dir / tables.MODEL_COMPARISON_FILENAME, index=False, lineterminator="\n"
    )
    model_comparison_by_regime_table.to_csv(
        results_dir / tables.MODEL_COMPARISON_BY_REGIME_FILENAME, index=False, lineterminator="\n"
    )
    coverage_summary_table.to_csv(
        results_dir / tables.COVERAGE_SUMMARY_FILENAME, index=False, lineterminator="\n"
    )
    backtest_coverage_table.to_csv(
        results_dir / tables.BACKTEST_COVERAGE_FILENAME, index=False, lineterminator="\n"
    )
    risk_headline_table.to_csv(
        results_dir / tables.RISK_HEADLINE_FILENAME, index=False, lineterminator="\n"
    )
    log.info("written 5 CSVs to %s", results_dir)

    # --- Snapshot ---------------------------------------------------------------
    regime_flags = full_regime_flags.reindex(lgbm_q50.index)

    snapshot_frame, stats = snapshot.build_snapshot(
        price_actual=lgbm_q50["y_true"],
        forecast_median=lgbm_q50["y_pred"],
        lo_raw=lgbm_q05["y_pred"],
        hi_raw=lgbm_q95["y_pred"],
        lo_calibrated=lgbm_q05_cal["y_pred"],
        hi_calibrated=lgbm_q95_cal["y_pred"],
        regime_flags=regime_flags,
    )
    snapshot_path = results_dir / snapshot.SNAPSHOT_FILENAME
    # Storage-only precision reduction (display/dashboard artifact, not a
    # statistics source -- those live in the full-precision CSVs above):
    # float16 keeps max round-trip error ~0.25 EUR/MWh, imperceptible on a
    # 0-900 EUR/MWh chart, and keeps the checked-in file under the repo's
    # pre-commit 1000 KB single-file limit.
    float_columns = snapshot_frame.select_dtypes(include="float64").columns
    compact = snapshot_frame.astype({col: "float16" for col in float_columns})
    compact.to_parquet(snapshot_path, compression="brotli")
    log.info(
        "written %s (%d rows; dropped %d index-mismatch, %d missing-required; "
        "quantile-crossing rate raw=%.3f calibrated=%.3f -- NOT repaired, only measured)",
        snapshot_path,
        stats.n_rows,
        stats.n_dropped_index_mismatch,
        stats.n_dropped_missing_required,
        stats.crossing_rate_raw,
        stats.crossing_rate_calibrated,
    )

    # --- Figures ---------------------------------------------------------------
    alpha = 1.0 - BacktestConfig().level  # same derivation as the notebook's Figure 6
    fig_coverage_forest = assets.plot_coverage_forest(backtest_coverage, alpha=alpha)
    fig_coverage_forest.savefig(assets_dir / "coverage_forest.png", dpi=_DPI)

    reliability_curves = {
        "raw": _reliability_curve_for_plot(conformal_raw),
        "calibrated_sorted": _reliability_curve_for_plot(conformal_sorted),
        "arimax": _reliability_curve_for_plot(
            reliability_curve_df[reliability_curve_df["model"] == "arimax"]
        ),
    }
    fig_reliability = assets.plot_reliability(reliability_curves)
    fig_reliability.savefig(assets_dir / "reliability_diagram.png", dpi=_DPI)

    fig_forecast_vs_actual = assets.plot_forecast_vs_actual(
        lgbm_q50["y_true"], lgbm_q50["y_pred"], model_label="LightGBM"
    )
    fig_forecast_vs_actual.savefig(assets_dir / "forecast_vs_actual.png", dpi=_DPI)

    window_start = pd.Timestamp(args.fan_chart_start, tz="Europe/Berlin").tz_convert("UTC")
    window_end = pd.Timestamp(args.fan_chart_end, tz="Europe/Berlin").tz_convert("UTC")
    window_mask = (lgbm_q50.index >= window_start) & (lgbm_q50.index < window_end)
    y_true_window = lgbm_q50.loc[window_mask, "y_true"]

    import matplotlib.pyplot as plt  # local import: keep the Agg backend set first

    fig_fan, fan_axes = plt.subplots(1, 2, figsize=(15, 3.5))
    raw_quantiles = {
        0.05: lgbm_q05.loc[window_mask, "y_pred"],
        0.5: lgbm_q50.loc[window_mask, "y_pred"],
        0.95: lgbm_q95.loc[window_mask, "y_pred"],
    }
    calibrated_quantiles = {
        0.05: lgbm_q05_cal.loc[window_mask, "y_pred"],
        0.5: lgbm_q50.loc[window_mask, "y_pred"],
        0.95: lgbm_q95_cal.loc[window_mask, "y_pred"],
    }
    assets.plot_fan_chart(raw_quantiles, y_true_window, title="Raw quantiles", ax=fan_axes[0])
    assets.plot_fan_chart(
        calibrated_quantiles, y_true_window, title="Calibrated (conformal)", ax=fan_axes[1]
    )
    fig_fan.suptitle(
        f"Fan chart, before vs. after calibration ({args.fan_chart_start} .. {args.fan_chart_end})"
    )
    fig_fan.tight_layout()
    fig_fan.savefig(assets_dir / "fan_chart_calibration.png", dpi=_DPI)
    log.info("written 4 PNGs to %s", assets_dir)

    # --- Size budget (spec 5.1 §7: warn, do not fail) --------------------------
    total_bytes = sum(f.stat().st_size for f in results_dir.rglob("*") if f.is_file())
    total_mb = total_bytes / (1024 * 1024)
    if total_mb > _MAX_RESULTS_MB:
        log.warning(
            "%s is %.1f MB, above the %.0f MB budget (spec 5.1 §2.5).",
            results_dir,
            total_mb,
            _MAX_RESULTS_MB,
        )
    else:
        log.info("%s is %.2f MB (budget: %.0f MB).", results_dir, total_mb, _MAX_RESULTS_MB)


if __name__ == "__main__":
    main()
