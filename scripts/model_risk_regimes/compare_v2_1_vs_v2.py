"""Sprint 7 Step 7.4a: compare hourly_model_v2 (published reference) against
two hourly_model_v2.1 variants -- 365-day training window, with and without
the temperature tree feature -- on persisted predictions only.

Three pairwise comparisons isolate what's driving any difference:
  - v2.1_temp    vs v2            (365-day window + temperature, combined)
  - v2.1_notemp  vs v2            (365-day window alone)
  - v2.1_temp    vs v2.1_notemp   (temperature alone, window held fixed)

Reads only already-written reference_predictions.parquet files (the
published v2 one, the owner's own reproduction-gate reference run, and the
owner's own no-temperature control run from this step's Rueckfrage) -- no
model runs here. The published v2 file still carries the pre-rename
"price_v2"/"price_v1" model_version strings baked in at generation time
(Sprint 7 Step 7.6, before the rename in 3864c9a) -- renaming source code
does not retroactively rewrite already-committed parquet column values.

Writes docs/sprint7_step7_4a_log.md (gitignored, no images, no outputs/
writes -- spec section 5).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from energy_price_forecast.common.evaluation.dm_test import daily_mean_loss, dm_test
from energy_price_forecast.common.evaluation.metrics import bias, mae, rmse

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_DAILY_HAC_LAG = 7
_DAILY_HORIZON = 1
_HOURLY_HAC_LAG = 48
_HOURLY_HORIZON = 24

# name -> (path, model_version string actually stored in that file, rung)
_RUNS: dict[str, tuple[Path, str, str]] = {
    "v2": (
        Path("outputs/results/hourly_model_v2/reference_predictions.parquet"),
        "price_v2",
        "L2",
    ),
    "v2.1_notemp": (
        Path(
            "data/processed/price_model_runs/"
            "reference_predictions_v2_1_rolling365_baseline_check.parquet"
        ),
        "hourly_model_v2",
        "L2",
    ),
    "v2.1_temp": (
        Path(
            "data/processed/price_model_runs/"
            "reference_predictions_v2_1_rolling365_temperature.parquet"
        ),
        "hourly_model_v2",
        "L2",
    ),
}

_PAIRS: tuple[tuple[str, str], ...] = (
    ("v2.1_temp", "v2"),
    ("v2.1_notemp", "v2"),
    ("v2.1_temp", "v2.1_notemp"),
)


def _load(path: Path, model_version: str, rung: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[(df["model_version"] == model_version) & (df["rung"] == rung)]
    return df.set_index("timestamp_utc").sort_index()


def _period_mask(delivery_day: pd.Series, period: str) -> pd.Series:
    if period == "overall":
        return pd.Series(True, index=delivery_day.index)
    if period == "sep_dec":
        return delivery_day.dt.month.isin([9, 10, 11, 12])
    if period == "autumn_2026":
        tz = delivery_day.dt.tz
        return (delivery_day >= pd.Timestamp("2026-09-01", tz=tz)) & (
            delivery_day <= pd.Timestamp("2026-10-20", tz=tz)
        )
    raise ValueError(f"unknown period {period!r}")


def _dm_row(pair: str, loss_type: str, blocking: str, result: object) -> str:
    return (
        f"| {pair} | {loss_type} | {blocking} | {result.n_obs} | "  # type: ignore[attr-defined]
        f"{result.mean_loss_diff:.4f} | {result.dm_stat:.3f} | {result.p_value:.4f} |"  # type: ignore[attr-defined]
    )


def main() -> None:
    frames = {name: _load(path, mv, rung) for name, (path, mv, rung) in _RUNS.items()}
    for name, df in frames.items():
        logger.info("%s: %d rows, %s..%s", name, len(df), df.index.min(), df.index.max())

    common_index = frames["v2"].index
    for df in frames.values():
        common_index = common_index.intersection(df.index)
    logger.info(
        "common window: %s..%s (%d hours)",
        common_index.min(),
        common_index.max(),
        len(common_index),
    )
    frames = {name: df.loc[common_index].sort_index() for name, df in frames.items()}

    lines: list[str] = [
        "# Sprint 7 Step 7.4a -- v2 vs v2.1 (with/without temperature)",
        "",
        f"Common window: {common_index.min()} to {common_index.max()} ({len(common_index)} hours).",
        "",
        "## 1. Metrics (MAE / RMSE / Bias)",
        "",
    ]

    for period, label in (
        ("overall", "Overall (common window)"),
        ("sep_dec", "Sep-Dec (all years in window)"),
        ("autumn_2026", "Autumn 2026 (2026-09-01..2026-10-20)"),
    ):
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| Model | n | MAE | RMSE | Bias |")
        lines.append("|---|---|---|---|---|")
        for name, df in frames.items():
            mask = _period_mask(df["delivery_day"], period)
            sub = df.loc[mask]
            lines.append(
                f"| {name} | {len(sub)} | {mae(sub['y_true'], sub['y_pred']):.4f} | "
                f"{rmse(sub['y_true'], sub['y_pred']):.4f} | "
                f"{bias(sub['y_true'], sub['y_pred']):+.4f} |"
            )
        lines.append("")

    lines.append("## 2. Diebold-Mariano tests")
    lines.append("")
    lines.append(
        "Sign convention: mean_loss_diff = mean(loss_a - loss_b); negative means the "
        "first-named model has the lower expected loss."
    )
    lines.append("")
    lines.append("| Pair (a vs b) | Loss | Blocking | n | mean_loss_diff | DM stat | p-value |")
    lines.append("|---|---|---|---|---|---|---|")
    for name_a, name_b in _PAIRS:
        pair = f"{name_a} vs {name_b}"
        y_true_a = frames[name_a]["y_true"]
        y_true_b = frames[name_b]["y_true"]
        err_a_sq = (frames[name_a]["y_pred"] - y_true_a) ** 2
        err_b_sq = (frames[name_b]["y_pred"] - y_true_b) ** 2
        err_a_abs = (frames[name_a]["y_pred"] - y_true_a).abs()
        err_b_abs = (frames[name_b]["y_pred"] - y_true_b).abs()

        daily_sq_a, _ = daily_mean_loss(err_a_sq)
        daily_sq_b, _ = daily_mean_loss(err_b_sq)
        r = dm_test(daily_sq_a, daily_sq_b, hac_lag=_DAILY_HAC_LAG, horizon=_DAILY_HORIZON)
        lines.append(_dm_row(pair, "squared", "daily", r))

        r = dm_test(err_a_sq, err_b_sq, hac_lag=_HOURLY_HAC_LAG, horizon=_HOURLY_HORIZON)
        lines.append(_dm_row(pair, "squared", "hourly", r))

        daily_abs_a, _ = daily_mean_loss(err_a_abs)
        daily_abs_b, _ = daily_mean_loss(err_b_abs)
        r = dm_test(daily_abs_a, daily_abs_b, hac_lag=_DAILY_HAC_LAG, horizon=_DAILY_HORIZON)
        lines.append(_dm_row(pair, "absolute", "daily", r))
    lines.append("")

    lines.append("## 3. Data-stand control (beta/SRMC max |diff| on common hours)")
    lines.append("")
    lines.append(
        "Expectation near 0: the beta logic is unchanged for v2.1 (same calendar-month "
        "rule, same 3-year rolling pool, same beta_monthly.csv); SRMC depends only on the "
        "gas/CO2 feature columns, not on train_span_days or the temperature flag. A "
        "nonzero value means the underlying data moved since the v2 reference was built, "
        "not a code difference."
    )
    lines.append("")
    lines.append("| Pair | max\\|beta diff\\| | max\\|srmc diff\\| |")
    lines.append("|---|---|---|")
    for name_a, name_b in _PAIRS:
        beta_diff = (frames[name_a]["beta"] - frames[name_b]["beta"]).abs().max()
        srmc_diff = (frames[name_a]["srmc"] - frames[name_b]["srmc"]).abs().max()
        lines.append(f"| {name_a} vs {name_b} | {beta_diff:.3e} | {srmc_diff:.3e} |")
    lines.append("")

    out_path = Path("docs/sprint7_step7_4a_log.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
