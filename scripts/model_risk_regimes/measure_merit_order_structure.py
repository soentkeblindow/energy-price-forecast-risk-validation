"""Phase 1 (structural measurement) of
docs/merit_order_extrapolation_structure_measurement_spec.md.

For each calendar year and residual-load (RL) bin, fits
``price = a + beta * srmc + eps`` with Newey-West (HAC) standard errors
(lag 24h). Purely descriptive -- no walk-forward, no forecast, full calendar
years used throughout, as the spec explicitly allows for this phase. Run
twice: once with srmc_gas_only (all years), once with srmc_gas_full (from
2021-10, where EUA data starts). Writes the per-year-bin table plus a pooled
non-crisis-year estimate per bin, and a figure of beta over RL bin with one
line per year.

Also runs a seasonality check (follow-up, not in the original spec): the
same regression pooled across all non-crisis years but split by calendar
month instead of year, asking whether the pass-through itself varies by
month rather than by year -- writes its own CSV/figure.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

from energy_price_forecast.common.features.merit_order import (
    RL_BIN_EDGES_MW,
    RL_BIN_LABELS,
    build_merit_order_frame,
    rl_bin,
    srmc_gas_full,
    srmc_gas_only,
)
from energy_price_forecast.common.market_time import LOCAL_TZ

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

HAC_MAXLAGS = 24
# Below this many hourly observations, a lag-24 HAC covariance is not
# reliably estimable (too few effective degrees of freedom) -- the cell is
# reported with its actual n but NaN a/beta/se/r2 rather than a fit that
# looks precise but isn't.
MIN_OBS_FOR_REGRESSION = 2 * HAC_MAXLAGS


def _fit_one(y: pd.Series, x: pd.Series) -> dict[str, float]:
    n = len(y)
    if n < MIN_OBS_FOR_REGRESSION:
        return {
            "n": n,
            "a": float("nan"),
            "beta": float("nan"),
            "beta_se": float("nan"),
            "r2": float("nan"),
        }
    exog = sm.add_constant(x.to_numpy())
    fit = sm.OLS(y.to_numpy(), exog).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_MAXLAGS})
    return {
        "n": n,
        "a": float(fit.params[0]),
        "beta": float(fit.params[1]),
        "beta_se": float(fit.bse[1]),
        "r2": float(fit.rsquared),
    }


def _bin_bounds(bin_label: int) -> tuple[float, float]:
    return RL_BIN_EDGES_MW[bin_label], RL_BIN_EDGES_MW[bin_label + 1]


def _load_frame(data_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(data_path)
    target_index = pd.DatetimeIndex(df.index)

    frame = build_merit_order_frame(df, target_index)
    frame["bin"] = rl_bin(frame["residual_load"])
    local_index = pd.DatetimeIndex(frame.index).tz_convert(LOCAL_TZ)
    frame["year"] = local_index.year
    frame["month"] = local_index.month
    frame["srmc_gas_only"] = srmc_gas_only(frame["ttf_lag"])
    frame["srmc_gas_full"] = srmc_gas_full(frame["ttf_lag"], frame["eua_lag"])
    return frame


def _rows_for_definition(frame: pd.DataFrame, srmc_col: str, srmc_label: str) -> list[dict]:
    rows: list[dict] = []
    valid = frame.dropna(subset=["price", srmc_col, "bin"])

    for year, year_group in valid.groupby("year"):
        for bin_label in RL_BIN_LABELS:
            cell = year_group.loc[year_group["bin"] == bin_label]
            bin_low, bin_high = _bin_bounds(bin_label)
            fit = _fit_one(cell["price"], cell[srmc_col])
            rows.append(
                {
                    "srmc_definition": srmc_label,
                    "year": year,
                    "bin": bin_label,
                    "bin_low_mw": bin_low,
                    "bin_high_mw": bin_high,
                    **fit,
                }
            )

    non_crisis = valid.loc[~valid["is_crisis"]]
    for bin_label in RL_BIN_LABELS:
        cell = non_crisis.loc[non_crisis["bin"] == bin_label]
        bin_low, bin_high = _bin_bounds(bin_label)
        fit = _fit_one(cell["price"], cell[srmc_col])
        rows.append(
            {
                "srmc_definition": srmc_label,
                "year": "pooled_non_crisis",
                "bin": bin_label,
                "bin_low_mw": bin_low,
                "bin_high_mw": bin_high,
                **fit,
            }
        )
    return rows


def _rows_by_month(frame: pd.DataFrame, srmc_col: str, srmc_label: str) -> list[dict]:
    """Beta per calendar month and RL bin, pooled across all non-crisis
    years -- a seasonality check on top of the per-year table, asking
    whether the pass-through itself (not just the residual bias) varies by
    month rather than by year.
    """
    valid = frame.dropna(subset=["price", srmc_col, "bin"])
    non_crisis = valid.loc[~valid["is_crisis"]]

    rows: list[dict] = []
    for month in range(1, 13):
        month_group = non_crisis.loc[non_crisis["month"] == month]
        for bin_label in RL_BIN_LABELS:
            cell = month_group.loc[month_group["bin"] == bin_label]
            bin_low, bin_high = _bin_bounds(bin_label)
            fit = _fit_one(cell["price"], cell[srmc_col])
            rows.append(
                {
                    "srmc_definition": srmc_label,
                    "month": month,
                    "bin": bin_label,
                    "bin_low_mw": bin_low,
                    "bin_high_mw": bin_high,
                    **fit,
                }
            )
    return rows


def _plot_beta_by_month(table: pd.DataFrame, srmc_label: str, out_path: Path) -> None:
    subset = table[table["srmc_definition"] == srmc_label].sort_values(["month", "bin"])

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("twilight")
    for month, group in subset.groupby("month"):
        ax.plot(
            group["bin"],
            group["beta"],
            marker="o",
            color=cmap((month - 1) / 12),
            alpha=0.85,
            label=str(month),
        )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("RL bin (0 = RL < 0, highest = top decile and up)")
    ax.set_ylabel("beta (gas pass-through)")
    ax.set_title(
        f"Merit-order pass-through beta by calendar month and RL bin "
        f"({srmc_label}, pooled non-crisis years)"
    )
    ax.set_xticks(list(RL_BIN_LABELS))
    ax.legend(title="month", ncol=2, fontsize="small")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_beta_by_year(table: pd.DataFrame, srmc_label: str, out_path: Path) -> None:
    subset = table[
        (table["srmc_definition"] == srmc_label) & (table["year"] != "pooled_non_crisis")
    ].copy()
    subset["year"] = subset["year"].astype(int)
    crisis_years = {2021, 2022, 2023}
    subset["is_crisis_year"] = subset["year"].isin(crisis_years)
    subset = subset.sort_values(["year", "bin"])

    fig, ax = plt.subplots(figsize=(9, 6))
    for year, group in subset.groupby("year"):
        is_crisis_year = bool(group["is_crisis_year"].iloc[0])
        ax.plot(
            group["bin"],
            group["beta"],
            marker="o",
            linestyle="--" if is_crisis_year else "-",
            color="#d62728" if is_crisis_year else None,
            alpha=0.6 if is_crisis_year else 0.9,
            label=str(year),
        )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("RL bin (0 = RL < 0, highest = top decile and up)")
    ax.set_ylabel("beta (gas pass-through)")
    ax.set_title(f"Merit-order pass-through beta by year and RL bin ({srmc_label})")
    ax.set_xticks(list(RL_BIN_LABELS))
    ax.legend(title="year", ncol=2, fontsize="small")
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
        "--out-csv", type=Path, default=Path("outputs/results/merit_order_beta_by_year_bin.csv")
    )
    p.add_argument(
        "--out-fig", type=Path, default=Path("outputs/assets/merit_order_beta_by_year.png")
    )
    p.add_argument(
        "--out-csv-month",
        type=Path,
        default=Path("outputs/results/merit_order_beta_by_month_bin.csv"),
    )
    p.add_argument(
        "--out-fig-month", type=Path, default=Path("outputs/assets/merit_order_beta_by_month.png")
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    frame = _load_frame(args.data_path)
    logger.info("loaded %d rows, %s to %s", len(frame), frame.index.min(), frame.index.max())

    rows = _rows_for_definition(frame, "srmc_gas_only", "gas_only")
    rows += _rows_for_definition(frame, "srmc_gas_full", "gas_full")
    table = pd.DataFrame(rows)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_csv, index=False)
    logger.info("wrote %d rows to %s", len(table), args.out_csv)

    _plot_beta_by_year(table, "gas_only", args.out_fig)
    logger.info("wrote figure to %s", args.out_fig)

    month_rows = _rows_by_month(frame, "srmc_gas_only", "gas_only")
    month_rows += _rows_by_month(frame, "srmc_gas_full", "gas_full")
    month_table = pd.DataFrame(month_rows)

    args.out_csv_month.parent.mkdir(parents=True, exist_ok=True)
    month_table.to_csv(args.out_csv_month, index=False)
    logger.info("wrote %d rows to %s", len(month_table), args.out_csv_month)

    _plot_beta_by_month(month_table, "gas_only", args.out_fig_month)
    logger.info("wrote figure to %s", args.out_fig_month)


if __name__ == "__main__":
    main()
