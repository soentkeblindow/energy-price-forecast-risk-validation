"""Phase 0 ("FR-Preise laden") of docs/sprint7_step7_5_spec.md's Hauptteil.

Thin CLI wrapper around common.data.fr_prices.build_fr_day_ahead_hourly --
fetches FR day-ahead price and load (native resolution) from ENTSO-E,
collapses to hourly, and persists to data/interim/fr_day_ahead_hourly.parquet.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from energy_price_forecast.common.data.fr_prices import build_fr_day_ahead_hourly
from energy_price_forecast.common.data.loaders import load_interim_hourly
from energy_price_forecast.common.evaluation.walkforward import last_complete_local_day

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=str, default="2023-01-01")
    p.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/interim/hourly_merit_order.parquet"),
        help="Used only to derive the data edge (--end default) from the DE data.",
    )
    p.add_argument("--end", type=str, default=None, help="Defaults to the DE data edge.")
    p.add_argument(
        "--out-path", type=Path, default=Path("data/interim/fr_day_ahead_hourly.parquet")
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    start = pd.Timestamp(args.start, tz="UTC")
    if args.end is not None:
        end = pd.Timestamp(args.end, tz="UTC")
    else:
        de = load_interim_hourly(args.data_path)
        edge = last_complete_local_day(de["day_ahead_price"])
        end = edge.tz_convert("UTC") + pd.Timedelta(days=1)
        logger.info("end defaulted to DE data edge: %s", edge.date())

    build_fr_day_ahead_hourly(start, end, path=args.out_path)


if __name__ == "__main__":
    main()
