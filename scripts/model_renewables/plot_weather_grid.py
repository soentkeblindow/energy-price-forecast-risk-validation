"""Map of the ECMWF IFS grid points that feed the renewables reconstruction.

Draws the 18 points of ``common/data/weather_grid.py::GRID_POINTS`` on a
map of Germany and writes ``outputs/assets/weather_grid_map.png`` (the
README figure). The coordinates are read from that module, never typed in
here, so the figure cannot drift away from the point set the model actually
uses.

These are model grid points, not weather stations: each one is the location
for which the ECMWF IFS forecast is requested from the Open-Meteo Single Runs
API (Open-Meteo snaps the request to the nearest ~9 km grid cell).

Short labels follow the point set's own ordering and the naming used in the
planning documents: O1-O3 offshore, L1-L13 onshore (used by the wind
onshore AND the solar model), S1-S2 solar-only.

Boundaries come from ``data/geo/germany_boundaries.geojson`` -- a small,
committed, pre-simplified extract of Natural Earth (public domain); see
``data/README.md`` for how it was produced. No network access, no
dependency beyond matplotlib.

Usage::

    uv run python scripts/model_renewables/plot_weather_grid.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from energy_price_forecast.common.config import PROJECT_ROOT
from energy_price_forecast.common.data.weather_grid import (
    GRID_POINTS,
    GRID_VERSION,
    GridPoint,
)

logger = logging.getLogger(__name__)

_GEO_PATH = PROJECT_ROOT / "data" / "geo" / "germany_boundaries.geojson"
_OUT_PATH = PROJECT_ROOT / "outputs" / "assets" / "weather_grid_map.png"

# Map extent (lon_min, lon_max, lat_min, lat_max): Germany plus the offshore
# clusters in the North Sea and the Baltic.
_EXTENT = (5.3, 15.6, 47.1, 55.4)

_SEA = "#eaf2f8"
_LAND_OTHER = "#f4f4f2"
_LAND_DE = "#ffffff"
_BORDER_OTHER = "#c9c9c4"
_BORDER_STATE = "#bdbdb8"
_BORDER_DE = "#6f6f6a"
_TEXT = "#2b2b28"

# Colour encodes the aggregate region (the north/central/south means the wind
# and solar features are built from); marker shape encodes which models use
# the point. Shape alone is enough to read the figure without colour.
_STYLE: dict[str, dict[str, Any]] = {
    "offshore": {"color": "#1f5f99", "marker": "D", "size": 70},
    "north": {"color": "#2a9d8f", "marker": "o", "size": 80},
    "central": {"color": "#e09f3e", "marker": "o", "size": 80},
    "south": {"color": "#9c4f96", "marker": "o", "size": 80},
    "solar_only": {"color": "#9c4f96", "marker": "*", "size": 230},
}

# Label offsets in points (dx, dy) where the default would collide with a
# neighbour or a coastline.
_LABEL_OFFSET: dict[str, tuple[float, float]] = {
    "O1": (-9, 8),
    "O2": (8, 6),
    "L1": (8, 4),
    "L2": (-10, -13),
    "L3": (8, -10),
    "S1": (10, -4),
    "S2": (-24, -4),
}


def short_labels(points: Sequence[GridPoint]) -> dict[str, str]:
    """O1.., L1.., S1.. in the point set's own order (matches the planning docs)."""
    prefix = {"offshore": "O", "onshore": "L", "solar_only": "S"}
    counters = {"O": 0, "L": 0, "S": 0}
    labels: dict[str, str] = {}
    for point in points:
        key = prefix[point.kind]
        counters[key] += 1
        labels[point.point_id] = f"{key}{counters[key]}"
    return labels


def _style_key(point: GridPoint) -> str:
    if point.kind == "solar_only":
        return "solar_only"
    return point.region


def _rings(geometry: dict[str, Any]) -> Iterator[list[list[float]]]:
    """Yield every polygon ring (exterior and holes) of a (Multi)Polygon."""
    if geometry["type"] == "Polygon":
        yield from geometry["coordinates"]
    elif geometry["type"] == "MultiPolygon":
        for polygon in geometry["coordinates"]:
            yield from polygon


def _draw_layer(
    ax: Any,
    features: list[dict[str, Any]],
    *,
    facecolor: str,
    edgecolor: str,
    linewidth: float,
    zorder: float,
) -> None:
    for feature in features:
        for ring in _rings(feature["geometry"]):
            ax.add_patch(
                Polygon(
                    ring,
                    closed=True,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    zorder=zorder,
                )
            )


def plot_weather_grid(geo_path: Path, out_path: Path) -> None:
    geo = json.loads(geo_path.read_text(encoding="utf-8"))
    features = geo["features"]
    countries = [f for f in features if f["properties"]["layer"] == "country"]
    others = [f for f in countries if f["properties"]["iso_a2"] != "DE"]
    germany = [f for f in countries if f["properties"]["iso_a2"] == "DE"]
    states = [f for f in features if f["properties"]["layer"] == "de_state"]
    if len(germany) != 1 or len(states) != 16:
        raise ValueError(
            f"{geo_path}: expected 1 Germany outline and 16 states, "
            f"found {len(germany)} and {len(states)}"
        )

    fig, ax = plt.subplots(figsize=(8.0, 10.2), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(_SEA)

    _draw_layer(ax, others, facecolor=_LAND_OTHER, edgecolor=_BORDER_OTHER, linewidth=0.6, zorder=1)
    _draw_layer(ax, states, facecolor=_LAND_DE, edgecolor=_BORDER_STATE, linewidth=0.5, zorder=2)
    _draw_layer(ax, germany, facecolor="none", edgecolor=_BORDER_DE, linewidth=1.1, zorder=3)

    labels = short_labels(GRID_POINTS)
    for point in GRID_POINTS:
        style = _STYLE[_style_key(point)]
        ax.scatter(
            point.longitude,
            point.latitude,
            s=style["size"],
            marker=style["marker"],
            color=style["color"],
            edgecolors="white",
            linewidths=1.0,
            zorder=5,
        )
        label = labels[point.point_id]
        dx, dy = _LABEL_OFFSET.get(label, (8, 4))
        ax.annotate(
            label,
            (point.longitude, point.latitude),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=_TEXT,
            zorder=6,
        )

    lon_min, lon_max, lat_min, lat_max = _EXTENT
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    # Equirectangular with the usual cos(latitude) correction: good enough at
    # this scale, and avoids a projection library dependency.
    ax.set_aspect(1.0 / math.cos(math.radians((lat_min + lat_max) / 2)))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(_BORDER_OTHER)

    legend_items = [
        ("Offshore wind (O1-O3)", "offshore"),
        ("Onshore wind + solar, north (L1-L7)", "north"),
        ("Onshore wind + solar, central (L8-L10)", "central"),
        ("Onshore wind + solar, south (L11-L13)", "south"),
        ("Solar only, south (S1-S2)", "solar_only"),
    ]
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=_STYLE[key]["marker"],
            markersize=13 if key == "solar_only" else 8,
            markerfacecolor=_STYLE[key]["color"],
            markeredgecolor="white",
            label=text,
        )
        for text, key in legend_items
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
        fontsize=8.5,
        frameon=True,
        facecolor="white",
        edgecolor=_BORDER_OTHER,
        framealpha=0.95,
    )

    ax.set_title(
        f"ECMWF IFS grid points used for the wind/solar reconstruction ({len(GRID_POINTS)})",
        fontsize=11.5,
        color=_TEXT,
        loc="left",
        pad=10,
    )
    fig.text(
        0.02,
        0.008,
        f"Point set {GRID_VERSION}, from common/data/weather_grid.py. "
        "Boundaries: Natural Earth (public domain).",
        fontsize=7.5,
        color="#6f6f6a",
    )

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    logger.info("Wrote %s (%d grid points)", out_path, len(GRID_POINTS))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--geo-path", type=Path, default=_GEO_PATH)
    parser.add_argument("--out", type=Path, default=_OUT_PATH)
    args = parser.parse_args()
    plot_weather_grid(args.geo_path, args.out)


if __name__ == "__main__":
    main()
