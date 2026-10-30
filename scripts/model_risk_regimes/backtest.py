"""Walk-forward backtest for baseline or LightGBM forecasters; logs to MLflow."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Literal

import mlflow
import pandas as pd

from energy_price_forecast.common.data.loaders import load_interim_hourly, load_processed_features
from energy_price_forecast.common.evaluation.config import EXPERIMENT_NAME, SPRINT3_EXPERIMENT_NAME
from energy_price_forecast.common.evaluation.metrics import pinball, summarise
from energy_price_forecast.common.evaluation.walkforward import run_backtest, walk_forward_splits
from energy_price_forecast.common.features.subset import drop_features
from energy_price_forecast.common.models.lgbm import _DEFAULT_PARAMS, LGBMForecaster
from energy_price_forecast.model_risk_regimes.models.arimax import (
    ARIMAX_EXOG_COLUMNS,
    ARIMAXForecaster,
    select_arimax_exog,
)
from energy_price_forecast.model_risk_regimes.models.baseline import (
    LassoForecaster,
    OLSForecaster,
    RidgeForecaster,
    SimilarDayNaive,
)


def _resolve_window(window_arg: str | None, model: str) -> Literal["expanding", "rolling"]:
    """Return the effective window type, applying per-model defaults."""
    if window_arg == "rolling":
        return "rolling"
    if window_arg == "expanding":
        return "expanding"
    return "rolling" if model == "arimax" else "expanding"


def _resolve_target_transform(
    transform_arg: str | None, model: str
) -> Literal["asinh", "identity"]:
    """Return the effective target transform, applying per-model defaults.

    Lasso's asinh/sinh round-trip extrapolates explosively on a feature value
    far outside its training range (an out-of-distribution commodity price
    reaching the model via a now-corrected forward-fill, sprint 7.0 step 4.2):
    one affected hour predicted 4,478 EUR/MWh against a realised 861. Measured
    fix: identity halves both MAE and RMSE relative to the broken asinh run and
    removes the outlier entirely, with no corresponding fragility in ridge/ols
    observed or tested -- their default stays asinh unless overridden.
    """
    if transform_arg == "asinh":
        return "asinh"
    if transform_arg == "identity":
        return "identity"
    return "identity" if model == "lasso" else "asinh"


def _parse_order(s: str) -> tuple[int, int, int]:
    """Parse '2,0,1' into (2, 0, 1) for ARIMA order."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"--arima-order must be p,d,q (got '{s}')")
    p, d, q = (int(x) for x in parts)
    return p, d, q


def _fingerprint(df: pd.DataFrame) -> str:
    return f"{df.index.min()}_{df.index.max()}_{df.shape}"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward backtest (naive or lasso).")
    p.add_argument(
        "--model", default="naive", choices=["naive", "lasso", "ridge", "ols", "lgbm", "arimax"]
    )
    p.add_argument(
        "--objective",
        default="quantile",
        choices=["quantile", "l2"],
        help="LightGBM loss function (--model lgbm only): 'quantile' (default) predicts "
        "the conditional median at --alpha; 'l2' predicts the conditional mean for "
        "RMSE-scored use cases and takes no --alpha.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Quantile level for LightGBM/ARIMAX (ignored for other models); "
        "defaults to 0.5 when not set. Must not be combined with --objective l2.",
    )
    p.add_argument("--random-state", type=int, default=0, help="Random seed for LightGBM.")
    p.add_argument(
        "--target-transform",
        default=None,
        choices=["asinh", "identity"],
        help="Target transform for lasso/ridge/ols. Default: 'identity' for lasso "
        "(asinh's sinh back-transform extrapolates explosively on out-of-distribution "
        "feature values, measured on the commodity-price case, sprint 7.0 step 4.2), "
        "'asinh' for ridge/ols.",
    )
    p.add_argument("--study", default="adhoc", help="MLflow tag: logical study grouping.")
    p.add_argument(
        "--experiment",
        default=None,
        help="Override the MLflow experiment name (default: model-dependent, e.g. "
        "'sprint3_models' for lgbm/arimax).",
    )
    p.add_argument("--note", default="", help="MLflow tag: free-text run note.")
    p.add_argument("--test-start", default="2021-01-01")
    p.add_argument("--test-end", default=None)
    p.add_argument(
        "--window",
        default=None,
        choices=["expanding", "rolling"],
        help="Walk-forward window type. Default: 'rolling' for arimax, 'expanding' for others.",
    )
    p.add_argument(
        "--arima-order",
        default="2,0,0",
        help="ARIMA order as p,d,q for --model arimax (default: 2,0,0).",
    )
    p.add_argument("--train-span-days", type=int, default=None)
    p.add_argument(
        "--refit-every",
        type=int,
        default=None,
        help="Refit cadence in days. Defaults to 1 for naive, 7 for lasso.",
    )
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument("--features-path", default=None, type=Path)
    p.add_argument("--data-path", default=None, type=Path)
    p.add_argument("--out", default=None, type=Path)
    p.add_argument(
        "--drop-features",
        default="",
        help="Comma-separated feature columns to drop in-memory before the lgbm backtest "
        "(ablation, step 3.5). Fails fast if any name is not found in the feature matrix.",
    )
    p.add_argument(
        "--params-path",
        default=None,
        type=Path,
        help="Path to a frozen params JSON produced by scripts/tune.py. "
        "When set, injects the tuned params into LGBMForecaster and uses the "
        "'tuned' run-name variant.",
    )
    p.add_argument(
        "--tuned-from",
        default="",
        help="MLflow run ID of the tune.py run that produced --params-path "
        "(logged as tag tuned_from for lineage tracking).",
    )
    args = p.parse_args()
    if args.objective == "l2" and args.model != "lgbm":
        p.error("--objective l2 is only valid together with --model lgbm")
    if args.objective == "l2" and args.alpha is not None:
        p.error(
            "--objective l2 predicts the conditional mean and takes no --alpha "
            f"(got --alpha {args.alpha})"
        )
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)
    t0 = time.monotonic()
    args = _parse_args()

    # Resolve per-model window defaults (arimax defaults to rolling/90; others to expanding).
    _window = _resolve_window(args.window, args.model)
    _train_span_days: int | None = (
        args.train_span_days
        if args.train_span_days is not None
        else (90 if args.model == "arimax" else None)
    )

    df = load_interim_hourly(args.data_path) if args.data_path else load_interim_hourly()
    price: pd.Series = df["day_ahead_price"].rename("day_ahead_price")

    experiment_name = EXPERIMENT_NAME
    extra_metrics: dict[str, float] = {}
    # --alpha defaults to None on the CLI so that "not passed" is
    # distinguishable from "explicitly 0.5" for the --objective l2 guard in
    # _parse_args; resolve it to the historical 0.5 default for everything
    # below that needs a concrete number (arimax, quantile run naming/logging).
    # Unused for --model naive, bound here regardless so the pinball-metric
    # block below (which checks args.model, not this branch) stays well-typed.
    resolved_alpha = args.alpha if args.alpha is not None else 0.5

    if args.model == "naive":
        refit_every = args.refit_every if args.refit_every is not None else 1
        model: (
            SimilarDayNaive
            | LassoForecaster
            | RidgeForecaster
            | OLSForecaster
            | LGBMForecaster
            | ARIMAXForecaster
        ) = SimilarDayNaive()
        index = pd.DatetimeIndex(price.index)
        y = price
        x = None
        run_name = "similarday_naive"
        out = args.out or Path("data/processed/backtest_similarday.parquet")
        log_params: dict[str, object] = {
            "model": "similarday_naive",
            "target_transform": "none",
            "window": _window,
            "refit_every": refit_every,
            "test_start": args.test_start,
            "test_end": str(args.test_end),
        }
        log_tags: dict[str, str] = {
            "study": args.study,
            "note": args.note,
        }
    else:
        refit_every = args.refit_every if args.refit_every is not None else 7
        features_path = args.features_path or Path("data/processed/features_hourly.parquet")
        features = load_processed_features(features_path)
        y = price.reindex(features.index)
        x = features
        index = pd.DatetimeIndex(features.index)
        # Resolved unconditionally (cheap, side-effect-free): only the lasso/ridge/ols
        # branches below consume it, but computing it here avoids relying on static
        # analysis to prove it's bound only where args.model matches again.
        target_transform = _resolve_target_transform(args.target_transform, args.model)
        if args.model == "lasso":
            model = LassoForecaster(
                target_transform=target_transform,
                cv_splits=args.cv_splits,
                n_jobs=args.n_jobs,
            )
            extra_params: dict[str, object] = {"cv_splits": args.cv_splits}
            run_name = f"{args.model}_{target_transform}"
        elif args.model == "ridge":
            model = RidgeForecaster(target_transform=target_transform)
            extra_params = {}
            run_name = f"{args.model}_{target_transform}"
        elif args.model == "ols":
            model = OLSForecaster(target_transform=target_transform)
            extra_params = {}
            run_name = f"{args.model}_{target_transform}"
        elif args.model == "arimax":
            x = select_arimax_exog(features)
            order = _parse_order(args.arima_order)
            model = ARIMAXForecaster(alpha=resolved_alpha, order=order)
            run_name = f"arimax_q{int(resolved_alpha * 100):02d}"
            experiment_name = SPRINT3_EXPERIMENT_NAME
            extra_params = {
                "alpha": resolved_alpha,
                "order": str(model.order),
                "fourier_daily_k": model.fourier_daily_k,
                "fourier_weekly_k": model.fourier_weekly_k,
                "exog_set": "|".join(ARIMAX_EXOG_COLUMNS),
                "standardize": True,
                "train_span_days": _train_span_days,
                "random_state": 0,
            }
        else:  # lgbm
            drop_cols = [c.strip() for c in args.drop_features.split(",") if c.strip()]
            if drop_cols:
                features = drop_features(features, drop_cols, strict=True)
                x = features
                log.info("dropped %d features for ablation: %s", len(drop_cols), drop_cols)
            n_jobs = args.n_jobs if args.n_jobs is not None else 1
            frozen_params: dict[str, object] | None = None
            if args.params_path is not None:
                raw = json.loads(Path(args.params_path).read_text())
                frozen_params = raw["params"]
                log.info(
                    "loaded frozen params from %s (n_estimators=%s)",
                    args.params_path,
                    frozen_params.get("n_estimators"),
                )
                if args.objective == "l2":
                    log.warning(
                        "--params-path was tuned under objective='quantile' (pinball@0.5); "
                        "using it under --objective l2 is untested by that tuning run."
                    )
            tuned = frozen_params is not None
            log.info(
                "initialising LGBMForecaster (objective=%s, alpha=%s, tuned=%s)",
                args.objective,
                resolved_alpha if args.objective == "quantile" else "none",
                tuned,
            )
            model = LGBMForecaster(
                objective=args.objective,
                alpha=args.alpha,  # None unless objective=quantile and explicitly set
                params=frozen_params,  # None → uses _DEFAULT_PARAMS inside LGBMForecaster
                random_state=args.random_state,
                n_jobs=n_jobs,
            )
            extra_params = {
                "alpha": resolved_alpha if args.objective == "quantile" else "none",
                "objective": args.objective,
                "random_state": args.random_state,
                "n_jobs": n_jobs,
                "dropped_features": ",".join(drop_cols) if drop_cols else "none",
                **(frozen_params if frozen_params is not None else _DEFAULT_PARAMS),
            }
            if args.objective == "l2":
                run_name = f"lgbm_mean{'_tuned' if tuned else ''}"
            else:
                run_name = f"lgbm_q{int(resolved_alpha * 100):02d}{'_tuned' if tuned else ''}"
            experiment_name = SPRINT3_EXPERIMENT_NAME
        if args.model == "arimax":
            out = args.out or Path(
                f"data/processed/preds_arimax_q{int(resolved_alpha * 100):02d}.parquet"
            )
        else:
            out = args.out or Path(f"data/processed/backtest_{args.model}.parquet")
        log_params = {
            "model": args.model,
            "window": _window,
            "refit_every": refit_every,
            "test_start": args.test_start,
            "test_end": str(args.test_end),
            **extra_params,
        }
        log_tags = {
            "study": args.study,
            "note": args.note,
            "features_fingerprint": _fingerprint(features),
            "tuned_from": args.tuned_from,
        }

    folds = list(
        walk_forward_splits(
            index,
            test_start=args.test_start,
            test_end=args.test_end,
            window=_window,
            train_span_days=_train_span_days,
        )
    )

    predictions = run_backtest(y, model, folds, refit_every=refit_every, x=x)
    summary = summarise(predictions)

    if args.model in ("lgbm", "arimax"):
        # For objective="l2" resolved_alpha is always 0.5 (the CLI guard forbids
        # --alpha with --objective l2): pinball_0.50 = 0.5 * MAE by identity, so
        # this is a metric-comparability conversion, not a calibration claim.
        extra_metrics[f"pinball_{resolved_alpha:.2f}"] = pinball(
            predictions["y_true"], predictions["y_pred"], resolved_alpha
        )

    if args.experiment is not None:
        experiment_name = args.experiment

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(log_params)
        mlflow.set_tags(log_tags)
        mlflow.log_metrics({**summary, **extra_metrics})
        out.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(out)
        mlflow.log_artifact(str(out))

    elapsed = time.monotonic() - t0
    mae_val = summary.get("mae", float("nan"))
    log.info("backtest finished in %.0fs (MAE=%.4f)", elapsed, mae_val)
    print("Summary:")
    for k, v in {**summary, **extra_metrics}.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
