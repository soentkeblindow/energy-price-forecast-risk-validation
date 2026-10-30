"""Sprint 7 Step 7.7, Teil C.3 (docs/sprint7_step7_7_spec.md, section 4, C.3).

Monthly beta(RL) reference table for `hourly_model_v2` only (replaces
`build_beta_monthly_table.py`, which covered several candidate
configurations during Step 7.6's selection measurement). Settings are read
exclusively from `common.models.price_model.HOURLY_MODEL_V2`.

A single-anchor-month call (``--anchor-month``) appends just that month to
the existing table without touching the other rows, so the table can be
extended monthly going forward (the basis for Step 7.4's own parity
check) without recomputing history.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from energy_price_forecast.common.data.loaders import load_interim_hourly
from energy_price_forecast.common.evaluation.walkforward import last_complete_local_day
from energy_price_forecast.common.features.merit_order import (
    RL_BIN_EDGES_MW,
    RL_BIN_LABELS,
    build_merit_order_frame,
    raw_beta_per_bin,
    smooth_beta_isotonic,
)
from energy_price_forecast.common.market_time import LOCAL_TZ
from energy_price_forecast.common.models.price_model import HOURLY_MODEL_V2

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

FIRST_ANCHOR_MONTH = pd.Timestamp("2023-05-01", tz=LOCAL_TZ)
CLAMP_THRESHOLD = 1.5


def _month_anchors(first: pd.Timestamp, last_local_day: pd.Timestamp) -> list[pd.Timestamp]:
    last = pd.Timestamp(year=last_local_day.year, month=last_local_day.month, day=1, tz=LOCAL_TZ)
    anchors = []
    current = first
    while current <= last:
        anchors.append(current)
        current = (current + pd.DateOffset(months=1)).normalize()
    return anchors


def build_hourly_model_v2_beta_table(
    long_history: pd.DataFrame, anchors: list[pd.Timestamp]
) -> pd.DataFrame:
    """One row per (anchor month, RL bin), using exactly HOURLY_MODEL_V2's own
    settings (``srmc_definition="gas_full"``, absolute RL bins,
    ``beta_window_days`` from the config).
    """
    rows = []
    for anchor in anchors:
        raw = raw_beta_per_bin(
            long_history,
            anchor,
            srmc_definition="gas_full",
            rl_measure="absolute",
            window_days=HOURLY_MODEL_V2.beta_window_days,
        )
        smoothed = smooth_beta_isotonic(raw["beta_raw"], upper=CLAMP_THRESHOLD)
        for label in RL_BIN_LABELS:
            rows.append(
                {
                    "anchor_month": anchor.strftime("%Y-%m"),
                    "bin": label,
                    "bin_lower_mw": RL_BIN_EDGES_MW[label],
                    "bin_upper_mw": RL_BIN_EDGES_MW[label + 1],
                    "n": int(raw.loc[label, "n"]),
                    "beta_raw": raw.loc[label, "beta_raw"],
                    "beta_smoothed": smoothed.loc[label],
                    "clamp_flag": bool(raw.loc[label, "beta_raw"] > CLAMP_THRESHOLD),
                    "model_version": "hourly_model_v2",
                }
            )
    return pd.DataFrame(rows)


def merge_beta_table(existing: pd.DataFrame | None, new_rows: pd.DataFrame) -> pd.DataFrame:
    """Append ``new_rows`` to ``existing``, dropping any pre-existing rows
    for the same anchor months first so re-running (or re-appending the
    same month twice) is idempotent -- never duplicates a row.
    """
    if existing is None:
        table = new_rows
    else:
        new_anchor_months = set(new_rows["anchor_month"].unique())
        existing = existing.loc[~existing["anchor_month"].isin(new_anchor_months)]
        table = pd.concat([existing, new_rows], ignore_index=True)
    return table.sort_values(["anchor_month", "bin"]).reset_index(drop=True)


def _draw_figure(table: pd.DataFrame, fig_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    ax.set_facecolor("white")
    anchor_months = sorted(table["anchor_month"].unique())
    last_anchor = anchor_months[-1]
    for anchor in anchor_months:
        row = table.loc[table["anchor_month"] == anchor].sort_values("bin")
        is_last = anchor == last_anchor
        ax.plot(
            row["bin"],
            row["beta_smoothed"],
            color="#1f77b4" if is_last else "#bbbbbb",
            linewidth=2.2 if is_last else 0.7,
            zorder=2 if is_last else 1,
            label=f"last anchor ({last_anchor})" if is_last else None,
        )
    ax.axhline(1.5, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("RL bin")
    ax.set_ylabel("beta_smoothed")
    ax.set_title("hourly_model_v2: beta(RL) by anchor month")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, facecolor="white")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", type=Path, default=Path("data/interim/hourly_data.parquet"))
    p.add_argument(
        "--out-csv", type=Path, default=Path("outputs/results/hourly_model_v2/beta_monthly.csv")
    )
    p.add_argument("--fig-path", type=Path, default=Path("outputs/assets/hourly_model_v2_beta.png"))
    p.add_argument(
        "--compare-csv",
        type=Path,
        default=Path("outputs/results/price_model_selection_beta_monthly.csv"),
    )
    p.add_argument(
        "--anchor-month",
        type=str,
        default=None,
        help="Build/append only this one YYYY-MM anchor instead of the full 2023-05..edge range.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    df = load_interim_hourly(args.data_path)
    price = df["day_ahead_price"]
    edge = last_complete_local_day(price)
    logger.info("data edge: %s", edge.date())

    long_history = build_merit_order_frame(df, pd.DatetimeIndex(df.index))

    if args.anchor_month:
        anchors = [pd.Timestamp(f"{args.anchor_month}-01", tz=LOCAL_TZ)]
    else:
        anchors = _month_anchors(FIRST_ANCHOR_MONTH, edge)
    logger.info(
        "anchors: %s .. %s (%d months)", anchors[0].date(), anchors[-1].date(), len(anchors)
    )

    new_rows = build_hourly_model_v2_beta_table(long_history, anchors)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(args.out_csv) if args.out_csv.exists() else None
    table = merge_beta_table(existing, new_rows)
    table.to_csv(args.out_csv, index=False)
    logger.info(
        "wrote %d rows (%d months x %d bins) to %s",
        len(table),
        len(anchors),
        len(RL_BIN_LABELS),
        args.out_csv,
    )

    # --- Equality check against Step 7.6's B_full_roll rows -----------------
    if args.compare_csv.exists():
        compare = pd.read_csv(args.compare_csv)
        compare_b_full_roll = compare.loc[compare["variant"] == "B_full_roll"]
        merged = new_rows.merge(
            compare_b_full_roll,
            on=["anchor_month", "bin"],
            suffixes=("_new", "_7_6"),
            how="inner",
        )
        if merged.empty:
            logger.info(
                "equality check: no overlapping anchors with price_model_selection_beta_monthly.csv"
            )
        else:
            max_diff_beta_raw = float((merged["beta_raw_new"] - merged["beta_raw_7_6"]).abs().max())
            max_diff_beta_smoothed = float(
                (merged["beta_smoothed_new"] - merged["beta_smoothed_7_6"]).abs().max()
            )
            max_diff_n = int((merged["n_new"] - merged["n_7_6"]).abs().max())
            logger.info(
                "equality check vs 7.6 B_full_roll (%d overlapping anchor-bin rows): "
                "max|beta_raw diff|=%.3e, max|beta_smoothed diff|=%.3e, max|n diff|=%d -- %s",
                len(merged),
                max_diff_beta_raw,
                max_diff_beta_smoothed,
                max_diff_n,
                "PASS"
                if max_diff_beta_raw <= 1e-9 and max_diff_beta_smoothed <= 1e-9 and max_diff_n == 0
                else "RUECKFRAGE: FAILED",
            )

    _draw_figure(table, args.fig_path)
    logger.info("wrote figure to %s", args.fig_path)


if __name__ == "__main__":
    main()
