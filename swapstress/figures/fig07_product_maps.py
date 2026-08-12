"""Descriptor Fig 7: the released product -- what it says, and how sure it is.

Two maps of the same month, side by side, from the Level 1 record:

a  Matric potential, July 2023: the per-pixel mean over the month's SMAP
   retrieval days of the released daily QRF median, presented as log10|psi| in
   MPa. This is the product's central statement: dry (yellow) in the interior
   West, wet (dark) in the East -- the psi field a reuser would actually take.
   The title carries no statistic word because "median" up front reads as a
   temporal median; the caption defines the aggregation.
b  95% prediction-interval width for the same days. The map is the monthly mean
   of ``q975 - q025`` in log10 units. The point is that the uncertainty is
   spatially structured rather than flat: the interval is a per-pixel statement
   tracking soil and climate gradients, not a global error bar quoted once.
   Drawn in ``style.SEQUENTIAL_ALT`` (single-hue purples) rather than the
   median's viridis, so yellow does not mean "dry" on one map and "uncertain"
   on the other.

Land the month never retrieved is filled ``style.NO_DATA_GRAY`` and keyed once
in panel a; white stays reserved for water and land outside the domain.

A monthly composite rather than a single day, because Level 1 follows the SMAP
swath: any one day is stripes, and the stripes are about orbit geometry, not
the product. Averaging the month's available days (24 of 31 in July 2023)
fills the swath pattern without touching the gap-filled level, whose pixels
carry no interval of their own.

Both aggregations happen in log space. For the median that is a deliberate
presentation choice (the map is drawn in log10|MPa|, so its monthly summary is
the mean of what is drawn); for the width it is the only option that means
anything, since the pair is released in signed MPa but its width is defined in
log units. No unit conversion touches the width arithmetic: the source bands
are ``log10(cm)`` and the presentation is ``log10|MPa|``, and because that
change is an additive shift the difference of two bands is the same number in
either. The median *is* shifted, by that same exact constant
(``swapstress.units.log10_suction_cm_to_log10_abs_mpa``).

Inputs are the 0.3 release run's CONUS inference rasters, which carry the
quantile pair (``swapstress-predict --release-quantiles``). Rasters from a
model trained without quantiles carry the median alone, and the figure says so
rather than drawing something else.

The calibration story (PICP sweep, coverage by theta decile) moved to the
supplementary candidates; the released 95% level's empirical coverage is
quoted in the text.

Usage:
    uv run swapstress-figures --figure uncertainty
    uv run swapstress-figures --figure uncertainty -- --month 2023-07
"""

from __future__ import annotations

import argparse
import calendar
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Patch
from pyproj import Transformer

from swapstress.figures import style
from swapstress.figures.basemap import load_conus_states, pixel_corner_lonlat
from swapstress.inference.product import Q025_BAND_NAME, Q975_BAND_NAME
from swapstress.units import log10_suction_cm_to_log10_abs_mpa

matplotlib.use("Agg")

RELEASE_DIR = Path("/nas/soils/swapstress/releases/v03_20260729")
# Level 1 rather than gap-filled: the interval is the model's own statement
# about a same-day retrieval, and a gap-filled pixel has no quantiles of its
# own -- so the two panels can only agree on the raw record.
PRED_DIR = RELEASE_DIR / "inference_conus"

MEDIAN_BAND_NAME = "log10_suction_cm"

# NAD83 / Conus Albers. The rasters are on an equal-area grid already, so
# drawing them equal-area keeps CONUS the shape readers know; plotting straight
# lon/lat would stretch the north of the country sideways.
MAP_CRS = "EPSG:5070"
MAP_PAD = 0.015

BOUNDARY_COLOR = "#4d4d4d"
BOUNDARY_WIDTH = 0.3

FIG_HEIGHT_MM = 66.0

NOTE_PT = 6.0

# Robust limits, since neither field has a fixed range: the far tails are a
# handful of pixels and would spend most of the ramp on them.
LIMIT_PCTL = (2.0, 98.0)


def month_paths(pred_dir: Path, month: str) -> list[Path]:
    """The month's daily rasters, in date order.

    *month* is ``YYYY-MM``. Level 1 follows the SMAP swath, so a month never
    has all its calendar days; the composite is over the days that exist.
    """
    stamp = month.replace("-", "")
    if len(stamp) != 6 or not stamp.isdigit():
        raise ValueError(f"--month wants YYYY-MM, got {month!r}")
    paths = sorted(pred_dir.glob(f"suction_{stamp}*.tif"))
    if not paths:
        raise FileNotFoundError(
            f"No prediction rasters matching suction_{stamp}*.tif in "
            f"{pred_dir}. Fig 7 composites a month of the 0.3 release run "
            "(`swapstress-predict --release-quantiles`)."
        )
    return paths


def month_title(month: str) -> str:
    """``2023-07`` as ``July 2023``, for the panel title."""
    year, mm = month.split("-")
    return f"{calendar.month_name[int(mm)]} {year}"


def _missing_quantiles(path: Path, names: list[str]) -> str:
    missing = [n for n in (Q025_BAND_NAME, Q975_BAND_NAME) if n not in names]
    return (
        f"{path} carries bands {names} and is missing {missing}. Panel b maps "
        "the released q025/q975 pair, so this figure needs a run made with "
        "`swapstress-predict --release-quantiles` against a model trained "
        "with `swapstress-train --quantile`. Rasters from the 0.2 release "
        "carry the median alone."
    )


def read_day(path: Path):
    """One day's ``(median, width)`` in source log10 units, NaN where invalid.

    Valid means all three bands are valid: the released day writes the trio
    together, so a pixel with a median but no pair would be a malformed raster
    rather than a state to average around.
    """
    with rasterio.open(path) as src:
        names = list(src.descriptions)
        if not {Q025_BAND_NAME, Q975_BAND_NAME} <= set(names):
            raise ValueError(_missing_quantiles(path, names))
        median = src.read(names.index(MEDIAN_BAND_NAME) + 1)
        q025 = src.read(names.index(Q025_BAND_NAME) + 1)
        q975 = src.read(names.index(Q975_BAND_NAME) + 1)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs

    valid = np.isfinite(median) & np.isfinite(q025) & np.isfinite(q975)
    if nodata is not None and not np.isnan(nodata):
        valid &= (median != nodata) & (q025 != nodata) & (q975 != nodata)

    median = np.where(valid, median, np.nan).astype(np.float32)
    width = np.where(valid, q975 - q025, np.nan).astype(np.float32)
    return median, width, transform, crs


def monthly_composite(paths: list[Path]):
    """Per-pixel monthly means of the median and the interval width.

    Returns ``(median_log10_abs_mpa, width, n_days, transform, crs)`` where the
    means are over each pixel's valid days and NaN where the month never saw a
    retrieval. The median comes back already shifted to log10|MPa|; the width
    needs no shift.
    """
    median_sum = width_sum = count = None
    transform = crs = None

    for path in paths:
        median, width, transform, crs = read_day(path)
        valid = np.isfinite(median)
        if median_sum is None:
            median_sum = np.zeros(median.shape, dtype=np.float64)
            width_sum = np.zeros(median.shape, dtype=np.float64)
            count = np.zeros(median.shape, dtype=np.int32)
        median_sum[valid] += median[valid]
        width_sum[valid] += width[valid]
        count[valid] += 1

    seen = count > 0
    median_mean = np.full(count.shape, np.nan, dtype=np.float32)
    width_mean = np.full(count.shape, np.nan, dtype=np.float32)
    median_mean[seen] = log10_suction_cm_to_log10_abs_mpa(
        median_sum[seen] / count[seen]
    ).astype(np.float32)
    width_mean[seen] = (width_sum[seen] / count[seen]).astype(np.float32)

    if not seen.any():
        raise ValueError(
            "The monthly composite is empty: no pixel has a valid retrieval "
            "on any of the month's days, so there is nothing to map."
        )
    return median_mean, width_mean, len(paths), transform, crs


def albers_corner_mesh(transform, shape, crs):
    """Pixel *corner* mesh in Conus Albers, for a flat-shaded ``pcolormesh``.

    Corners rather than centres: flat shading wants one more node than cells in
    each direction, and centres would shift the image half a 9 km pixel.
    """
    lon, lat = pixel_corner_lonlat(transform, shape, crs)
    to_albers = Transformer.from_crs("EPSG:4326", MAP_CRS, always_xy=True)
    return to_albers.transform(lon, lat)


def padded_extent(bounds, pad=MAP_PAD):
    x0, y0, x1, y1 = bounds
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    return x0 - dx, y0 - dy, x1 + dx, y1 + dy


def robust_limits(field, step=0.05):
    """Robust colour limits, rounded outward to something a tick can say."""
    finite = field[np.isfinite(field)]
    lo, hi = np.percentile(finite, LIMIT_PCTL)
    if hi <= lo:
        # A constant field is degenerate for a colour ramp but not an error;
        # open a small window around it so the bar has an axis.
        lo, hi = lo - step, hi + step
    return float(np.floor(lo / step) * step), float(np.ceil(hi / step) * step)


def draw_map(ax, mesh_x, mesh_y, field, states, extent, vmin, vmax, cmap):
    """A CONUS field under recessive state outlines.

    The states are filled ``style.NO_DATA_GRAY`` under the mesh, so land the
    month never retrieved reads as gray rather than borrowing white from the
    water and outside-domain background -- the set-wide missing-data
    convention, keyed once in panel a.
    """
    states.plot(ax=ax, facecolor=style.NO_DATA_GRAY, edgecolor="none", zorder=0.5)
    mesh = ax.pcolormesh(
        mesh_x,
        mesh_y,
        field,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="flat",
        zorder=1,
        # Rasterises the 9 km cells only. Outlines, ticks and every label stay
        # vector, which is what the artwork guide actually asks for.
        rasterized=True,
    )
    states.boundary.plot(
        ax=ax, edgecolor=BOUNDARY_COLOR, linewidth=BOUNDARY_WIDTH, zorder=2
    )
    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])
    ax.set_aspect("equal")
    ax.set_axis_off()
    return mesh


def _map_colorbar(fig, mesh, ax, label):
    # ``ax=`` rather than a dedicated cell: the bar steals its space from the
    # map it belongs to, and a nested gridspec thin enough to hold a colour
    # bar collapses the constrained layout.
    cbar = fig.colorbar(
        mesh,
        ax=ax,
        orientation="horizontal",
        extend="both",
        shrink=0.72,
        aspect=34,
        pad=0.015,
    )
    cbar.set_label(label, fontsize=NOTE_PT, labelpad=1.5)
    cbar.ax.tick_params(labelsize=NOTE_PT, length=1.8, width=0.4, pad=1.5)
    cbar.outline.set_linewidth(0.4)
    cbar.outline.set_edgecolor(style.AXIS_COLOR)
    return cbar


def render(median, width, n_days, mesh_x, mesh_y, states, month, output_dir):
    """Draw and save the two panels from data already in memory."""
    style.apply()

    fig = plt.figure(
        figsize=style.figsize(style.DOUBLE_COLUMN_MM, FIG_HEIGHT_MM),
        layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03)
    gs = fig.add_gridspec(1, 2)
    ax_median = fig.add_subplot(gs[0, 0])
    ax_width = fig.add_subplot(gs[0, 1])

    extent = padded_extent(states.total_bounds)

    mesh_m = draw_map(
        ax_median,
        mesh_x,
        mesh_y,
        median,
        states,
        extent,
        *robust_limits(median),
        cmap=style.SEQUENTIAL,
    )
    # The month lives in the title rather than a corner note: land fills the
    # frame on this window, so any in-axes annotation sits on data pixels.
    # "Matric potential" without a statistic -- "median" up front reads as a
    # temporal median, when the field is the monthly mean of the daily QRF
    # medians; the caption defines the aggregation.
    ax_median.set_title(
        f"Matric potential, {month_title(month)}",
        fontsize=style.MAX_TEXT_PT,
        pad=2.5,
    )
    style.panel_label(ax_median, "a", dx=0.0, dy=1.0)
    _map_colorbar(fig, mesh_m, ax_median, style.LOG10_ABS_MPA_AXIS)
    # The gray no-data convention is keyed once, on the first panel.
    ax_median.legend(
        handles=[Patch(facecolor=style.NO_DATA_GRAY, edgecolor="none")],
        labels=["land, no retrieval this month"],
        loc="lower left",
        fontsize=style.MIN_TEXT_PT,
        frameon=True,
        framealpha=0.8,
        edgecolor="none",
        facecolor="white",
        handlelength=1.1,
        handleheight=1.1,
        handletextpad=0.4,
        borderaxespad=0.1,
    )

    # A second sequential ramp for the width: with viridis on both panels,
    # yellow would mean "dry" on one map and "uncertain" on the other.
    mesh_w = draw_map(
        ax_width,
        mesh_x,
        mesh_y,
        width,
        states,
        extent,
        *robust_limits(width),
        cmap=style.SEQUENTIAL_ALT,
    )
    ax_width.set_title(
        "95% prediction-interval width", fontsize=style.MAX_TEXT_PT, pad=2.5
    )
    style.panel_label(ax_width, "b", dx=0.0, dy=1.0)
    _map_colorbar(fig, mesh_w, ax_width, f"q975 − q025 ({style.LOG10_ABS_MPA_UNIT})")

    print(f"Composite over {n_days} retrieval days in {month}")

    out = Path(output_dir)
    path = style.save(fig, out / "fig07_product_maps")
    print(f"Saved: {path}")
    print(f"Saved: {path.with_suffix('.pdf')}")
    return path


def build_figure(month, output_dir, pred_dir=PRED_DIR):
    paths = month_paths(Path(pred_dir), month)
    median, width, n_days, transform, crs = monthly_composite(paths)
    states = load_conus_states(crs=MAP_CRS)
    mesh_x, mesh_y = albers_corner_mesh(transform, median.shape, crs)
    return render(median, width, n_days, mesh_x, mesh_y, states, month, output_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Descriptor Fig 7: monthly median matric potential and "
        "prediction-interval width"
    )
    parser.add_argument("--month", default="2023-07")
    parser.add_argument("--output-dir", default="figs/descriptor")
    parser.add_argument("--pred-dir", default=str(PRED_DIR))
    args = parser.parse_args(argv)
    build_figure(args.month, args.output_dir, args.pred_dir)


if __name__ == "__main__":
    main()
