"""Teil 0 ("Monatliche beta-Tabelle als Ausgabe") of
docs/sprint7_step7_5_spec.md.

Builds the calendar-anchored beta(RL) reference table that 7.4's parity
check (live output vs. backtest-recomputed output) will compare against:
one row per (anchor month, RL bin) from 2024-01 through the last month with
a locally-complete delivery day in the data -- derived from the data edge,
never from the system clock -- using the primary configuration
(``srmc_gas_only``, fixed MW bins, expanding pool from the long-history
frame's own start, crisis excluded).

Calls ``raw_beta_per_bin``/``smooth_beta_isotonic`` directly, bypassing
``MeritOrderForecaster`` entirely -- this table has no fold/backtest
structure, just one estimate per calendar month.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from energy_price_forecast.common.data.loaders import load_interim_hourly
from energy_price_forecast.common.evaluation.walkforward import last_complete_local_day
from energy_price_forecast.common.features.merit_order import (
    BETA_CLAMP_MAX,
    RL_BIN_LABELS,
    RlMeasure,
    SrmcDefinition,
    build_merit_order_frame,
    raw_beta_per_bin,
    smooth_beta_isotonic,
)
from energy_price_forecast.common.market_time import LOCAL_TZ

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

FIRST_ANCHOR_MONTH = pd.Timestamp("2024-01-01", tz=LOCAL_TZ)


def _month_anchors(first: pd.Timestamp, last_local_day: pd.Timestamp) -> list[pd.Timestamp]:
    """Month-start anchors from ``first`` through the anchor of
    ``last_local_day``'s own month, inclusive.
    """
    last = pd.Timestamp(year=last_local_day.year, month=last_local_day.month, day=1, tz=LOCAL_TZ)
    anchors = []
    current = first
    while current <= last:
        anchors.append(current)
        current = (current + pd.DateOffset(months=1)).normalize()
    return anchors


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-path", type=Path, default=Path("data/interim/hourly_merit_order.parquet")
    )
    p.add_argument(
        "--out-csv", type=Path, default=Path("outputs/results/merit_order_beta_monthly.csv")
    )
    return p.parse_args()


def build_beta_monthly_table(
    long_history: pd.DataFrame,
    anchors: list[pd.Timestamp],
    *,
    srmc_definition: SrmcDefinition,
    rl_measure: RlMeasure,
    window_days: int | None,
) -> pd.DataFrame:
    """One row per (anchor month, RL bin): ``anchor_month``, ``bin``,
    ``beta_raw``, ``beta_smoothed``, ``n``. Extracted so a second driver
    (Sprint 7 Step 7.6, Teil B) can build this table for several
    ``MeritOrderForecaster`` configurations without duplicating the loop.
    """
    rows = []
    for anchor in anchors:
        raw = raw_beta_per_bin(
            long_history,
            anchor,
            srmc_definition=srmc_definition,
            rl_measure=rl_measure,
            window_days=window_days,
        )
        smoothed = smooth_beta_isotonic(raw["beta_raw"], upper=BETA_CLAMP_MAX)
        for label in RL_BIN_LABELS:
            rows.append(
                {
                    "anchor_month": anchor.strftime("%Y-%m"),
                    "bin": label,
                    "beta_raw": raw.loc[label, "beta_raw"],
                    "beta_smoothed": smoothed.loc[label],
                    "n": int(raw.loc[label, "n"]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    df = load_interim_hourly(args.data_path)
    price = df["day_ahead_price"]
    edge = last_complete_local_day(price)
    logger.info("last locally-complete delivery day (derived from data): %s", edge.date())

    long_history = build_merit_order_frame(df, pd.DatetimeIndex(df.index))
    anchors = _month_anchors(FIRST_ANCHOR_MONTH, edge)
    logger.info(
        "anchor months: %s .. %s (%d months)", anchors[0].date(), anchors[-1].date(), len(anchors)
    )

    table = build_beta_monthly_table(
        long_history,
        anchors,
        srmc_definition="gas_only",
        rl_measure="absolute",
        window_days=None,
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_csv, index=False)
    logger.info(
        "wrote %d rows (%d months x %d bins) to %s",
        len(table),
        len(anchors),
        len(RL_BIN_LABELS),
        args.out_csv,
    )


if __name__ == "__main__":
    main()
