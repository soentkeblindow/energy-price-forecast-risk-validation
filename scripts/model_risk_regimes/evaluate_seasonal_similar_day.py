"""docs/seasonal_window_similar_day_spec.md -- Phase 2 (Auswertung).

Reads the 8 cached variant predictions from measure_seasonal_similar_day.py
(data/processed/seasonal_similar_day_preds/) and produces metrics (overall,
by month, by weekday, holiday subset, outlier days), DM-tests against each
variant's same-model reference (local delivery days), the criteria 1/2
assessment per factor, and the by-month RMSE figure.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from energy_price_forecast.common.data.loaders import load_interim_hourly
from energy_price_forecast.common.evaluation.dm_test import daily_mean_loss, dm_test
from energy_price_forecast.common.evaluation.metrics import bias, mae, rmse
from energy_price_forecast.common.features.similar_day import (
    choose_reference_day,
    is_complete_price_day,
    is_holiday_de,
)
from energy_price_forecast.common.market_time import LOCAL_TZ

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

WINDOWS: tuple[str, ...] = ("rolling90", "seasonal")
SIMILAR_DAY_OPTIONS: tuple[str, ...] = ("no_similar", "similar")
MODELS: tuple[str, ...] = ("A", "B")

AUTUMN_2026_START = pd.Timestamp("2026-09-01", tz=LOCAL_TZ)
SEP_NOV_MONTHS = (9, 10, 11)
MAR_APR_MONTHS = (3, 4)
MONDAY = 0
SUNDAY = 6

OUTLIER_DAYS = (
    pd.Timestamp("2026-10-11", tz=LOCAL_TZ),
    pd.Timestamp("2026-10-18", tz=LOCAL_TZ),
    pd.Timestamp("2026-10-19", tz=LOCAL_TZ),
)

_DAILY_HAC_LAG = 7
_DAILY_HORIZON = 1
_HOURLY_HAC_LAG = 48
_HOURLY_HORIZON = 24
SIGNIFICANCE_LEVEL = 0.05
SEASONAL_SHIFT_THRESHOLD = 0.03  # spec section 5, criterion 2 (seasonal)
MONDAY_BIAS_SHRINK_THRESHOLD = 1 / 3  # spec section 5, criterion 2 (similar-day)


def _variant_name(window: str, similar_day: str, model: str) -> str:
    suffix = "" if similar_day == "no_similar" else "_similar"
    return f"{window}{suffix}_{model}"


def _reference_name(model: str) -> str:
    return _variant_name("rolling90", "no_similar", model)


def _load_predictions(preds_dir: Path) -> dict[str, pd.DataFrame]:
    preds = {}
    for window in WINDOWS:
        for similar_day in SIMILAR_DAY_OPTIONS:
            for model in MODELS:
                name = _variant_name(window, similar_day, model)
                preds[name] = pd.read_parquet(
                    preds_dir / f"seasonal_similar_day_preds_{name}.parquet"
                )
    return preds


def _local_weekday(index: pd.DatetimeIndex) -> np.ndarray:
    return pd.DatetimeIndex(index).tz_convert(LOCAL_TZ).weekday.to_numpy()


def _local_month(index: pd.DatetimeIndex) -> np.ndarray:
    return pd.DatetimeIndex(index).tz_convert(LOCAL_TZ).month.to_numpy()


def _local_year_month(index: pd.DatetimeIndex) -> pd.PeriodIndex:
    return pd.DatetimeIndex(index).tz_convert(LOCAL_TZ).to_period("M")


def _metrics_row(sub: pd.DataFrame) -> dict[str, float]:
    return {
        "n": len(sub),
        "rmse": rmse(sub["y_true"], sub["y_pred"]),
        "mae": mae(sub["y_true"], sub["y_pred"]),
        "bias": bias(sub["y_true"], sub["y_pred"]),
    }


def _overall_metrics_table(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, p in preds.items():
        for period, mask in (
            ("overall", pd.Series(True, index=p.index)),
            ("autumn_2026", p.index >= AUTUMN_2026_START),
            ("sep_nov", np.isin(_local_month(p.index), SEP_NOV_MONTHS)),
            ("mar_apr", np.isin(_local_month(p.index), MAR_APR_MONTHS)),
        ):
            sub = p.loc[mask]
            rows.append({"variant": name, "period": period, **_metrics_row(sub)})
    return pd.DataFrame(rows)


def _by_month_table(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, p in preds.items():
        year_month = _local_year_month(p.index)
        for ym in sorted(year_month.unique()):
            sub = p.loc[year_month == ym]
            rows.append({"variant": name, "year_month": str(ym), **_metrics_row(sub)})
    return pd.DataFrame(rows)


def _by_weekday_table(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, p in preds.items():
        weekday = _local_weekday(p.index)
        for wd in range(7):
            sub = p.loc[weekday == wd]
            rows.append({"variant": name, "weekday": wd, **_metrics_row(sub)})
    return pd.DataFrame(rows)


def _holiday_subset_days(
    local_days: pd.DatetimeIndex, price: pd.Series
) -> tuple[set[dt.date], set[dt.date]]:
    """(own-holiday days, reference-shifted-for-holiday days) -- spec section 4."""
    own_holiday = {d.date() for d in local_days if is_holiday_de(d.date())}
    is_complete = is_complete_price_day(price)
    shifted = set()
    for d in local_days:
        day = d.date()
        ref = choose_reference_day(day, is_holiday=is_holiday_de, is_complete=is_complete)
        if ref is not None and any(reason == "holiday" for _, reason in ref.skipped):
            shifted.add(day)
    return own_holiday, shifted


def _holiday_subset_table(preds: dict[str, pd.DataFrame], price: pd.Series) -> pd.DataFrame:
    any_pred = next(iter(preds.values()))
    local_days = pd.DatetimeIndex(
        pd.DatetimeIndex(any_pred.index).tz_convert(LOCAL_TZ).normalize().unique()
    )
    own_holiday, shifted = _holiday_subset_days(local_days, price)
    subset_days = own_holiday | shifted
    logger.info(
        "holiday subset: %d days (%d own-holiday, %d reference-shifted-for-holiday)",
        len(subset_days),
        len(own_holiday),
        len(shifted),
    )

    rows = []
    for name, p in preds.items():
        local_day = pd.DatetimeIndex(p.index).tz_convert(LOCAL_TZ).normalize()
        mask = pd.Series(local_day.date, index=p.index).isin(subset_days)
        sub = p.loc[mask.to_numpy()]
        rows.append({"variant": name, "n_days": len(subset_days), **_metrics_row(sub)})
    return pd.DataFrame(rows), len(subset_days)


def _outlier_day_table(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for day in OUTLIER_DAYS:
        row: dict[str, object] = {"delivery_day": day}
        for name, p in preds.items():
            local_day = pd.DatetimeIndex(p.index).tz_convert(LOCAL_TZ).normalize()
            sub = p.loc[local_day == day]
            if not sub.empty:
                row[f"bias_{name}"] = bias(sub["y_true"], sub["y_pred"])
                row[f"rmse_{name}"] = rmse(sub["y_true"], sub["y_pred"])
        rows.append(row)
    return pd.DataFrame(rows)


def _dm_rows(comparison: str, loss_a: pd.Series, loss_b: pd.Series) -> list[dict]:
    aligned = pd.concat([loss_a.rename("a"), loss_b.rename("b")], axis=1).dropna()
    if len(aligned) < 2 * _HOURLY_HAC_LAG:
        return []
    daily_a, _ = daily_mean_loss(aligned["a"], group_by="local_day")
    daily_b, _ = daily_mean_loss(aligned["b"], group_by="local_day")
    daily_result = dm_test(daily_a, daily_b, hac_lag=_DAILY_HAC_LAG, horizon=_DAILY_HORIZON)
    hourly_result = dm_test(
        aligned["a"], aligned["b"], hac_lag=_HOURLY_HAC_LAG, horizon=_HOURLY_HORIZON
    )
    rows = []
    for block, result in (("daily_local", daily_result), ("hourly", hourly_result)):
        rows.append(
            {
                "comparison": comparison,
                "block": block,
                "n_obs": result.n_obs,
                "hac_lag": result.hac_lag,
                "mean_loss_diff": result.mean_loss_diff,
                "dm_stat": result.dm_stat,
                "p_value": result.p_value,
            }
        )
    return rows


def _dm_table(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for window in WINDOWS:
        for similar_day in SIMILAR_DAY_OPTIONS:
            for model in MODELS:
                name = _variant_name(window, similar_day, model)
                ref_name = _reference_name(model)
                if name == ref_name:
                    continue
                a = preds[name]
                b = preds[ref_name]
                loss_a = (a["y_true"] - a["y_pred"]) ** 2
                loss_b = (b["y_true"] - b["y_pred"]) ** 2
                rows += _dm_rows(f"{name}_vs_{ref_name}", loss_a, loss_b)
    return pd.DataFrame(rows)


def _criterion_1_ok(dm: pd.DataFrame, name: str, ref_name: str) -> bool:
    row = dm.loc[(dm["comparison"] == f"{name}_vs_{ref_name}") & (dm["block"] == "daily_local")]
    if row.empty:
        return False
    p_value = float(row["p_value"].iloc[0])
    mean_loss_diff = float(row["mean_loss_diff"].iloc[0])  # a - b; positive = a (variant) worse
    significantly_worse = mean_loss_diff > 0 and p_value < SIGNIFICANCE_LEVEL
    return not significantly_worse


def _assess_seasonal(
    overall: pd.DataFrame, dm: pd.DataFrame, window: str, similar_day: str, model: str
) -> dict:
    name = _variant_name(window, similar_day, model)
    ref_name = _reference_name(model)
    crit1 = _criterion_1_ok(dm, name, ref_name)

    sep_nov_var = overall.loc[
        (overall["variant"] == name) & (overall["period"] == "sep_nov"), "rmse"
    ].iloc[0]
    sep_nov_ref = overall.loc[
        (overall["variant"] == ref_name) & (overall["period"] == "sep_nov"), "rmse"
    ].iloc[0]
    sep_nov_change = (sep_nov_var - sep_nov_ref) / sep_nov_ref

    mar_apr_var = overall.loc[
        (overall["variant"] == name) & (overall["period"] == "mar_apr"), "rmse"
    ].iloc[0]
    mar_apr_ref = overall.loc[
        (overall["variant"] == ref_name) & (overall["period"] == "mar_apr"), "rmse"
    ].iloc[0]
    mar_apr_change = (mar_apr_var - mar_apr_ref) / mar_apr_ref

    crit2 = bool(
        sep_nov_change <= -SEASONAL_SHIFT_THRESHOLD and mar_apr_change <= SEASONAL_SHIFT_THRESHOLD
    )
    return {
        "variant": name,
        "reference": ref_name,
        "factor": "seasonal",
        "criterion_1_not_significantly_worse": crit1,
        "sep_nov_rmse_change_pct": sep_nov_change * 100,
        "mar_apr_rmse_change_pct": mar_apr_change * 100,
        "criterion_2_ok": crit2,
        "improvement": bool(crit1 and crit2),
    }


def _assess_similar_day(
    weekday: pd.DataFrame,
    holiday: pd.DataFrame,
    dm: pd.DataFrame,
    window: str,
    similar_day: str,
    model: str,
) -> dict:
    name = _variant_name(window, similar_day, model)
    ref_name = _reference_name(model)
    crit1 = _criterion_1_ok(dm, name, ref_name)

    mon_bias_var = weekday.loc[
        (weekday["variant"] == name) & (weekday["weekday"] == MONDAY), "bias"
    ].iloc[0]
    mon_bias_ref = weekday.loc[
        (weekday["variant"] == ref_name) & (weekday["weekday"] == MONDAY), "bias"
    ].iloc[0]
    monday_shrink = 1 - abs(mon_bias_var) / abs(mon_bias_ref) if mon_bias_ref != 0 else float("nan")

    holiday_rmse_var = holiday.loc[holiday["variant"] == name, "rmse"].iloc[0]
    holiday_rmse_ref = holiday.loc[holiday["variant"] == ref_name, "rmse"].iloc[0]
    holiday_not_worse = bool(holiday_rmse_var <= holiday_rmse_ref)

    crit2 = bool(monday_shrink >= MONDAY_BIAS_SHRINK_THRESHOLD and holiday_not_worse)
    return {
        "variant": name,
        "reference": ref_name,
        "factor": "similar_day",
        "criterion_1_not_significantly_worse": crit1,
        "monday_bias_var": mon_bias_var,
        "monday_bias_ref": mon_bias_ref,
        "monday_bias_relative_shrink": monday_shrink,
        "holiday_rmse_var": holiday_rmse_var,
        "holiday_rmse_ref": holiday_rmse_ref,
        "holiday_rmse_not_worse": holiday_not_worse,
        "criterion_2_ok": crit2,
        "improvement": bool(crit1 and crit2),
    }


def _assess_combination(
    overall: pd.DataFrame,
    weekday: pd.DataFrame,
    holiday: pd.DataFrame,
    dm: pd.DataFrame,
    model: str,
) -> dict:
    seasonal = _assess_seasonal(overall, dm, "seasonal", "similar", model)
    similar = _assess_similar_day(weekday, holiday, dm, "seasonal", "similar", model)
    crit1 = seasonal["criterion_1_not_significantly_worse"]
    crit2 = bool(seasonal["criterion_2_ok"] and similar["criterion_2_ok"])
    return {
        "variant": seasonal["variant"],
        "reference": seasonal["reference"],
        "factor": "combination",
        "criterion_1_not_significantly_worse": crit1,
        "seasonal_criterion_2_ok": seasonal["criterion_2_ok"],
        "similar_day_criterion_2_ok": similar["criterion_2_ok"],
        "criterion_2_ok": crit2,
        "improvement": bool(crit1 and crit2),
    }


def _rolling_rmse_by_month_figure(by_month: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for window in WINDOWS:
        for similar_day in SIMILAR_DAY_OPTIONS:
            for model in MODELS:
                name = _variant_name(window, similar_day, model)
                sub = by_month.loc[by_month["variant"] == name].sort_values("year_month")
                style = "--" if similar_day == "similar" else "-"
                color = "tab:blue" if window == "rolling90" else "tab:orange"
                marker = "o" if model == "A" else "s"
                ax.plot(
                    sub["year_month"], sub["rmse"], style, color=color, marker=marker, label=name
                )
    ax.set_ylabel("RMSE (EUR/MWh)")
    ax.set_xlabel("month")
    ax.tick_params(axis="x", rotation=90)
    ax.legend(ncol=2, fontsize=8)
    fig.suptitle("RMSE by month -- 8 variants")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-path", type=Path, default=Path("data/interim/hourly_merit_order.parquet")
    )
    p.add_argument(
        "--preds-dir", type=Path, default=Path("data/processed/seasonal_similar_day_preds")
    )
    p.add_argument("--out-dir", type=Path, default=Path("outputs/results"))
    p.add_argument(
        "--fig-path",
        type=Path,
        default=Path("outputs/assets/seasonal_similar_day_rmse_by_month.png"),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    df = load_interim_hourly(args.data_path)
    price = df["day_ahead_price"]
    preds = _load_predictions(args.preds_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    overall = _overall_metrics_table(preds)
    overall.to_csv(args.out_dir / "seasonal_similar_day_metrics.csv", index=False)

    by_month = _by_month_table(preds)
    by_month.to_csv(args.out_dir / "seasonal_similar_day_metrics_by_month.csv", index=False)

    weekday = _by_weekday_table(preds)
    weekday.to_csv(args.out_dir / "seasonal_similar_day_metrics_by_weekday.csv", index=False)

    holiday, n_holiday_days = _holiday_subset_table(preds, price)
    holiday.to_csv(args.out_dir / "seasonal_similar_day_holiday_subset.csv", index=False)

    outliers = _outlier_day_table(preds)
    outliers.to_csv(args.out_dir / "seasonal_similar_day_outlier_days.csv", index=False)

    dm = _dm_table(preds)
    dm.to_csv(args.out_dir / "seasonal_similar_day_dm_test.csv", index=False)

    assessments = []
    for model in MODELS:
        # Pure seasonal effect: seasonal_X vs rolling90_X, similar-day held off.
        assessments.append(_assess_seasonal(overall, dm, "seasonal", "no_similar", model))
        # Pure similar-day effect: rolling90_similar_X vs rolling90_X, window held at rolling90.
        assessments.append(_assess_similar_day(weekday, holiday, dm, "rolling90", "similar", model))
        # Combination: seasonal_similar_X vs rolling90_X, both factors together.
        assessments.append(_assess_combination(overall, weekday, holiday, dm, model))
    assessment_table = pd.DataFrame(assessments)
    assessment_table.to_csv(args.out_dir / "seasonal_similar_day_assessment.csv", index=False)

    for row in assessments:
        logger.info(
            "%s (factor=%s) vs %s: improvement=%s (crit1=%s, crit2=%s)",
            row["variant"],
            row["factor"],
            row["reference"],
            row["improvement"],
            row["criterion_1_not_significantly_worse"],
            row["criterion_2_ok"],
        )

    _rolling_rmse_by_month_figure(by_month, args.fig_path)
    logger.info("wrote figure to %s", args.fig_path)
    logger.info("n holiday-subset days: %d", n_holiday_days)


if __name__ == "__main__":
    main()
