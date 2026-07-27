"""Figure 11: CONUS predicted suction map.

Renders a single-day CONUS map of log10 suction (cm H2O) from the
gap-filled RF predictions. State boundaries overlaid. Colorbar with
both log10 cm and approximate kPa labels.

Optional --four-season flag produces a 2x2 panel (Jan/Apr/Jul/Oct).

Usage:
    uv run python viz/presentation/fig11_conus_suction.py
    uv run python viz/presentation/fig11_conus_suction.py --four-season
"""

import argparse
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap
from rasterio.transform import array_bounds

PRED_DIR = Path("/nas/soils/swapstress/releases/global_pruned_refresh_20260520/gapfill")
STATES_SHP = Path("/tmp/us_states/cb_2022_us_state_20m.shp")

NODATA = -9999.0
VMIN, VMAX = 1.2, 4.8

# Exclude non-CONUS
EXCLUDE_STUSPS = {"AK", "HI", "AS", "GU", "MP", "PR", "VI"}

# Custom colormap: wet blue → green → tan → orange → dry red
_CMAP_COLORS = [
    (0.00, "#1B4F72"),  # deep blue  – saturated
    (0.15, "#2E86C1"),  # mid blue   – wet
    (0.30, "#27AE60"),  # green      – field capacity
    (0.50, "#D4AC0D"),  # gold       – moderate stress
    (0.70, "#E67E22"),  # orange     – dry
    (0.85, "#C0392B"),  # red        – wilting
    (1.00, "#641E16"),  # dark red   – extreme
]
CMAP = LinearSegmentedColormap.from_list(
    "suction",
    [(pos, c) for pos, c in _CMAP_COLORS],
)

# Tick labels: log10 cm → approximate kPa
_LOG10_TICKS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
_KPA_LABELS = {
    1.5: "3",
    2.0: "10",
    2.5: "31",
    3.0: "98",
    3.5: "310",
    4.0: "980",
    4.5: "3100",
}


def _load_states():
    """Load CONUS state boundaries, reprojected to raster CRS."""
    states = gpd.read_file(STATES_SHP)
    conus = states[~states.STUSPS.isin(EXCLUDE_STUSPS)].copy()
    return conus


def _load_raster(date_str):
    """Load a single day's suction raster, return (array, transform, crs)."""
    path = PRED_DIR / f"suction_{date_str}.tif"
    if not path.exists():
        raise FileNotFoundError(path)
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
    data[(data == NODATA) | ~np.isfinite(data)] = np.nan
    return data, transform, crs


def _render_panel(ax, data, transform, crs, states, date_str, show_cbar=True):
    """Render suction raster + state boundaries on an axes."""
    h, w = data.shape
    left, bottom, right, top = array_bounds(h, w, transform)
    extent = [left, right, bottom, top]

    im = ax.imshow(
        data,
        cmap=CMAP,
        vmin=VMIN,
        vmax=VMAX,
        extent=extent,
        origin="upper",
        interpolation="nearest",
        zorder=1,
    )

    states_proj = states.to_crs(crs)
    states_proj.boundary.plot(
        ax=ax,
        edgecolor="#2C2C2A",
        linewidth=0.4,
        zorder=2,
    )

    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal")
    ax.set_axis_off()
    dt = datetime.strptime(date_str, "%Y%m%d")
    ax.set_title(dt.strftime("%B %-d, %Y"), fontsize=11, pad=4)

    if show_cbar:
        cb = plt.colorbar(
            im,
            ax=ax,
            fraction=0.025,
            pad=0.02,
            shrink=0.85,
        )
        cb.set_label(r"log$_{10}$ suction (cm H$_2$O)", fontsize=9)
        cb.set_ticks(_LOG10_TICKS)
        cb.ax.tick_params(labelsize=8)

        # Secondary kPa labels on the right
        cb_ax2 = cb.ax.twinx()
        cb_ax2.set_ylim(VMIN, VMAX)
        cb_ax2.set_yticks(_LOG10_TICKS)
        cb_ax2.set_yticklabels(
            [_KPA_LABELS[t] for t in _LOG10_TICKS],
            fontsize=7,
        )
        cb_ax2.set_ylabel("kPa", fontsize=8, rotation=270, labelpad=10)
        cb_ax2.tick_params(length=0)

    return im


def single_day(date_str, output_dir):
    """Render a single-day CONUS suction map."""
    data, transform, crs = _load_raster(date_str)
    states = _load_states()

    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=200)
    fig.subplots_adjust(left=0.02, right=0.88, top=0.92, bottom=0.02)

    _render_panel(ax, data, transform, crs, states, date_str)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(
            out / f"fig11_conus_suction_{date_str}.{ext}",
            dpi=200,
            bbox_inches="tight",
        )
    plt.close(fig)
    path = out / f"fig11_conus_suction_{date_str}.png"
    print(f"Saved to {path}")


def four_season(year, output_dir):
    """Render a 2x2 panel for Jan 15 / Apr 15 / Jul 15 / Oct 15."""
    dates = [f"{year}0115", f"{year}0415", f"{year}0715", f"{year}1015"]
    labels = ["January 15", "April 15", "July 15", "October 15"]
    states = _load_states()

    fig, axes = plt.subplots(2, 2, figsize=(14, 7), dpi=200)
    fig.subplots_adjust(wspace=0.05, hspace=0.12)

    for ax, date_str, label in zip(axes.ravel(), dates, labels):
        data, transform, crs = _load_raster(date_str)
        _render_panel(
            ax,
            data,
            transform,
            crs,
            states,
            f"{label}, {year}",
            show_cbar=False,
        )

    # Shared colorbar
    sm = plt.cm.ScalarMappable(
        cmap=CMAP,
        norm=plt.Normalize(vmin=VMIN, vmax=VMAX),
    )
    cb = fig.colorbar(
        sm,
        ax=axes,
        fraction=0.02,
        pad=0.03,
        shrink=0.85,
    )
    cb.set_label(r"log$_{10}$ suction (cm H$_2$O)", fontsize=10)
    cb.set_ticks(_LOG10_TICKS)
    cb.ax.tick_params(labelsize=8)

    cb_ax2 = cb.ax.twinx()
    cb_ax2.set_ylim(VMIN, VMAX)
    cb_ax2.set_yticks(_LOG10_TICKS)
    cb_ax2.set_yticklabels(
        [_KPA_LABELS[t] for t in _LOG10_TICKS],
        fontsize=7,
    )
    cb_ax2.set_ylabel("kPa", fontsize=8, rotation=270, labelpad=10)
    cb_ax2.tick_params(length=0)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(
            out / f"fig11_conus_suction_4season_{year}.{ext}",
            dpi=200,
            bbox_inches="tight",
        )
    plt.close(fig)
    path = out / f"fig11_conus_suction_4season_{year}.png"
    print(f"Saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figure 11: CONUS suction map")
    parser.add_argument(
        "--date", default="20230715", help="Date YYYYMMDD (default: 20230715)"
    )
    parser.add_argument(
        "--four-season",
        action="store_true",
        help="Render 2x2 seasonal panel instead of single day",
    )
    parser.add_argument(
        "--year", default="2023", help="Year for four-season panel (default: 2023)"
    )
    parser.add_argument(
        "--output-dir", default="figs/presentation", help="Output directory"
    )
    args = parser.parse_args()

    if args.four_season:
        four_season(args.year, args.output_dir)
    else:
        single_day(args.date, args.output_dir)
