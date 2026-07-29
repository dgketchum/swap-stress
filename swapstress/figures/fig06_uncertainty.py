"""Figure 6: the released prediction interval -- how wide, and how honest.

The product ships its QRF uncertainty as a quantile pair, ``q025``/``q975``, so
the figure asks the two questions a reuser has about it.

a  How wide is the interval, and where? The map is ``q975 - q025`` in log10
   units of matric potential for one example day. The point is that it is
   spatially structured rather than flat: the interval is a per-pixel
   statement, not a global error bar quoted once in a table.
b  Is the interval honest? Empirical coverage against nominal, swept from 50%
   to 99% on the spatial holdout. A single number at 95% cannot tell an
   interval that is well calibrated everywhere from one that is too wide in the
   middle and too narrow in the tails; the sweep can. The released 95% level is
   marked, and its PICP is the number the descriptor quotes.
c  Where is it honest? Coverage of that same 95% interval by theta decile. The
   product's known weak spot is the dry end, where SMAP floors out and where
   water stress actually matters, so coverage conditional on theta is the panel
   that speaks to fitness-for-use.

Width is drawn rather than shipped: it is exactly derivable from the pair, and
the pair is not derivable from a width, so the release carries only the pair and
this figure does the subtraction. No conversion is applied to the width and none
is needed: the bands are stored as ``log10(cm)`` and the descriptor presents
``log10|MPa|``, and because that unit change is an additive shift in log space
the difference of the two bands is already the same number in either. The bar
is therefore labelled in the presentation unit while the arithmetic is untouched.

The valid region is swath-shaped because the underlying prediction is Level 1:
same-day retrievals only, no gap filling. That is intended, and it is *not*
fixed by switching to ``inference_l4`` -- SMAP L4 assimilates PTF-derived
hydraulic parameters, so using it here would be circular.

Inputs come from the 0.3 release run: daily rasters carrying the quantile pair
(``swapstress-predict --release-quantiles``) and the holdout coverage tables
(``swapstress-validate --analysis quantile-coverage``). Neither exists for a
model trained without quantiles, and the figure says so rather than drawing
something else.

Usage:
    uv run swapstress-figures --figure uncertainty
    uv run swapstress-figures --figure uncertainty -- --date 2024-04-15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from swapstress.figures import style
from swapstress.figures.basemap import load_conus_states, pixel_corner_lonlat
from swapstress.inference.product import (
    Q025_BAND_NAME,
    Q975_BAND_NAME,
    RELEASE_NOMINAL,
)

matplotlib.use("Agg")

RELEASE_DIR = Path("/nas/soils/swapstress/releases/v03_20260729")
# Level 1 rather than gap-filled: the interval is the model's own statement
# about a same-day retrieval, and a gap-filled pixel has no quantiles of its own.
# The CONUS window, not the global grid: the released product is CONUS
# (global training, CONUS application) and the map should show the deposit.
PRED_DIR = RELEASE_DIR / "inference_conus"
# Stage 04 writes its tables straight into ``--output-dir``, with no
# subdirectory of its own, so the release run's coverage CSVs sit here.
ERROR_DIR = RELEASE_DIR / "evaluation"
COVERAGE_CSV = "quantile_coverage.csv"
COVERAGE_BY_THETA_CSV = "quantile_coverage_by_theta.csv"

# NAD83 / Conus Albers. The rasters are already on an equal-area grid, so
# drawing them equal-area keeps CONUS the shape readers know; plotting straight
# lon/lat would stretch the north of the country sideways.
MAP_CRS = "EPSG:5070"
MAP_PAD = 0.015
CONUS_LONLAT = (-125.0, 24.5, -66.5, 49.5)

BOUNDARY_COLOR = "#4d4d4d"
BOUNDARY_WIDTH = 0.3
REFERENCE_COLOR = "#8c8c8c"

FIG_HEIGHT_MM = 86.0
# The map is the wider column: it carries the spatial claim, and the two
# calibration panels are small-multiples beside it.
WIDTH_RATIOS = (1.42, 1.0)

NOTE_PT = 6.0

# Written with a solidus rather than a negative exponent on purpose: the
# Helvetica/Arial clones the guide's typeface resolves to have no U+207B, so a
# literal negative exponent would drop a glyph whichever clone is installed.
THETA_UNIT = "m³/m³"

# Robust limits, since a QRF width has no fixed range: the far tails are a
# handful of pixels and would spend most of the ramp on them.
WIDTH_PCTL = (2.0, 98.0)


def _missing_quantiles(path: Path, names: list[str]) -> str:
    missing = [n for n in (Q025_BAND_NAME, Q975_BAND_NAME) if n not in names]
    return (
        f"{path} carries bands {names} and is missing {missing}. The prediction "
        "interval is the released q025/q975 pair, so this map needs a run made "
        "with `swapstress-predict --release-quantiles` against a model trained "
        "with `swapstress-train --quantile`. Rasters from the 0.2 release carry "
        "the median alone."
    )


def read_interval_width(path: Path):
    """Interval width over CONUS, with its transform and CRS.

    Returns ``(width, transform, crs)`` where *width* is ``q975 - q025`` in
    log10 units and NaN wherever the day has no retrieval.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No prediction raster at {path}. Fig 6 maps the released quantile "
            "pair, which comes from the 0.3 release run "
            "(`swapstress-predict --release-quantiles`)."
        )

    with rasterio.open(path) as src:
        names = list(src.descriptions)
        if not {Q025_BAND_NAME, Q975_BAND_NAME} <= set(names):
            raise ValueError(_missing_quantiles(path, names))

        window = (
            from_bounds(
                *transform_bounds("EPSG:4326", src.crs, *CONUS_LONLAT),
                transform=src.transform,
            )
            .round_offsets()
            .round_lengths()
        )
        q025 = src.read(names.index(Q025_BAND_NAME) + 1, window=window)
        q975 = src.read(names.index(Q975_BAND_NAME) + 1, window=window)
        nodata = src.nodata
        transform = src.window_transform(window)
        crs = src.crs

    valid = np.isfinite(q025) & np.isfinite(q975)
    if nodata is not None and not np.isnan(nodata):
        valid &= (q025 != nodata) & (q975 != nodata)

    width = np.where(valid, q975 - q025, np.nan).astype(np.float32)
    return width, transform, crs


def read_coverage(error_dir: Path) -> tuple:
    """The two holdout coverage tables written by stage 04."""
    paths = [error_dir / COVERAGE_CSV, error_dir / COVERAGE_BY_THETA_CSV]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing coverage table(s): {', '.join(str(p) for p in missing)}. "
            "Panels b and c report PICP on the spatial holdout; run "
            "`swapstress-validate --analysis quantile-coverage --model-dir "
            "<quantile model>` to produce them."
        )
    return pd.read_csv(paths[0]), pd.read_csv(paths[1])


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


def width_limits(width):
    """Robust colour limits, rounded to something a tick label can say."""
    finite = width[np.isfinite(width)]
    if finite.size == 0:
        raise ValueError(
            "The interval width is empty over CONUS: the day has no valid "
            "retrievals in the window, so there is nothing to map."
        )
    lo, hi = np.percentile(finite, WIDTH_PCTL)
    if hi <= lo:
        # A constant field is degenerate for a colour ramp but not an error;
        # open a small window around it so the bar has an axis.
        lo, hi = lo - 0.05, hi + 0.05
    return float(np.floor(lo * 20) / 20), float(np.ceil(hi * 20) / 20)


def draw_width_map(ax, mesh_x, mesh_y, width, states, extent):
    """Panel a: the interval width under recessive state outlines."""
    vmin, vmax = width_limits(width)
    mesh = ax.pcolormesh(
        mesh_x,
        mesh_y,
        width,
        cmap=style.SEQUENTIAL,
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


def draw_calibration(ax, coverage):
    """Panel b: empirical coverage against nominal, with the released level."""
    coverage = coverage.sort_values("nominal")
    nominal = coverage["nominal"].values
    picp = coverage["picp"].values

    ax.plot(
        [nominal.min(), 1.0],
        [nominal.min(), 1.0],
        linestyle=(0, (3, 2)),
        linewidth=0.6,
        color=REFERENCE_COLOR,
        zorder=1,
    )
    ax.plot(
        nominal,
        picp,
        marker="o",
        markersize=2.6,
        linewidth=0.9,
        color=style.CATEGORICAL[0],
        zorder=2,
    )

    released = coverage[np.isclose(coverage["nominal"], RELEASE_NOMINAL)]
    if len(released):
        row = released.iloc[0]
        ax.plot(
            row["nominal"],
            row["picp"],
            marker="o",
            markersize=4.2,
            markerfacecolor="none",
            markeredgewidth=0.8,
            color=style.CATEGORICAL[1],
            zorder=3,
        )
        # In the empty corner under the diagonal rather than beside the point:
        # the curve hugs the 1:1 line, so there is no room next to it.
        ax.text(
            0.97,
            0.04,
            f"released {row['nominal']:.0%}\nPICP = {row['picp']:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=NOTE_PT,
            color=style.CATEGORICAL[1],
        )

    # Identical limits on both axes, so the reference line is the identity and
    # a point above it is over-coverage no matter where on the sweep it sits.
    lo = min(nominal.min(), picp.min()) - 0.03
    ax.set_xlim(lo, 1.01)
    ax.set_ylim(lo, 1.01)
    # No aspect constraint. A square box would be the textbook calibration plot,
    # but this cell is half the figure height and most of a column wide, so a
    # square leaves the panel a third the width of the one beneath it. The
    # dashed 1:1 line is drawn, not inferred from the angle, so letting the box
    # fill the cell costs nothing. (A *data* aspect is worse still: it resizes
    # the box after the layout is solved, and the x-label lands on panel c.)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage (PICP)")
    ax.tick_params(length=1.8, width=0.4, pad=1.5)


def draw_theta_coverage(ax, by_theta):
    """Panel c: coverage of the released interval, by theta decile."""
    by_theta = by_theta.sort_values("theta_mid")
    nominal = float(by_theta["nominal"].iloc[0])

    ax.axhline(
        nominal,
        linestyle=(0, (3, 2)),
        linewidth=0.6,
        color=REFERENCE_COLOR,
        zorder=1,
    )
    ax.plot(
        by_theta["theta_mid"],
        by_theta["picp"],
        marker="o",
        markersize=2.6,
        linewidth=0.9,
        color=style.CATEGORICAL[0],
        zorder=2,
    )
    # Headroom above the nominal line, so its label is not pinned to the spine.
    picp = by_theta["picp"].values
    lo, hi = min(picp.min(), nominal), max(picp.max(), nominal)
    pad = 0.18 * max(hi - lo, 1e-3)
    ax.set_ylim(lo - pad, hi + pad)
    ax.annotate(
        f"nominal {nominal:.0%}",
        xy=(by_theta["theta_mid"].max(), nominal),
        xytext=(0, 2),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=NOTE_PT,
        color=style.MUTED_INK,
    )
    ax.set_xlabel(f"θ decile midpoint ({THETA_UNIT})")
    ax.set_ylabel("PICP")
    ax.tick_params(length=1.8, width=0.4, pad=1.5)


def render(width, mesh_x, mesh_y, states, coverage, by_theta, date_str, output_dir):
    """Draw and save the three panels from data already in memory."""
    style.apply()

    fig = plt.figure(
        figsize=style.figsize(style.DOUBLE_COLUMN_MM, FIG_HEIGHT_MM),
        layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.04)
    gs = fig.add_gridspec(2, 2, width_ratios=list(WIDTH_RATIOS))

    ax_map = fig.add_subplot(gs[:, 0])
    ax_cal = fig.add_subplot(gs[0, 1])
    ax_theta = fig.add_subplot(gs[1, 1])

    extent = padded_extent(states.total_bounds)
    mesh = draw_width_map(ax_map, mesh_x, mesh_y, width, states, extent)
    ax_map.set_title(
        "QRF prediction interval width",
        fontsize=style.MAX_TEXT_PT,
        pad=2.5,
    )
    ax_map.text(
        0.02,
        0.03,
        date_str,
        transform=ax_map.transAxes,
        fontsize=NOTE_PT,
        color=style.MUTED_INK,
        ha="left",
        va="bottom",
    )
    style.panel_label(ax_map, "a", dx=0.0, dy=1.0)

    # ``ax=`` rather than a dedicated cell: the bar steals its space from the
    # map, which is the axes it belongs to, and a nested gridspec thin enough
    # to hold a colour bar collapses the constrained layout.
    cbar = fig.colorbar(
        mesh,
        ax=ax_map,
        orientation="horizontal",
        extend="both",
        shrink=0.62,
        aspect=38,
        pad=0.015,
    )
    cbar.set_label(
        f"q975 − q025 ({style.LOG10_ABS_MPA_UNIT})",
        fontsize=NOTE_PT,
        labelpad=1.5,
    )
    cbar.ax.tick_params(labelsize=NOTE_PT, length=1.8, width=0.4, pad=1.5)
    cbar.outline.set_linewidth(0.4)
    cbar.outline.set_edgecolor(style.AXIS_COLOR)

    draw_calibration(ax_cal, coverage)
    style.panel_label(ax_cal, "b", dx=-0.22, dy=1.02)

    draw_theta_coverage(ax_theta, by_theta)
    style.panel_label(ax_theta, "c", dx=-0.22, dy=1.02)

    out = Path(output_dir)
    path = style.save(fig, out / f"fig06_uncertainty_{date_str.replace('-', '')}")
    print(f"Saved: {path}")
    print(f"Saved: {path.with_suffix('.pdf')}")
    return path


def build_figure(date_str, output_dir, pred_dir=PRED_DIR, error_dir=ERROR_DIR):
    coverage, by_theta = read_coverage(Path(error_dir))
    pred_path = Path(pred_dir) / f"suction_{date_str.replace('-', '')}.tif"
    width, transform, crs = read_interval_width(pred_path)
    states = load_conus_states(crs=MAP_CRS)
    mesh_x, mesh_y = albers_corner_mesh(transform, width.shape, crs)
    return render(
        width, mesh_x, mesh_y, states, coverage, by_theta, date_str, output_dir
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Figure 6: released prediction interval width and coverage"
    )
    parser.add_argument("--date", default="2024-07-15")
    parser.add_argument("--output-dir", default="figs/descriptor")
    parser.add_argument("--pred-dir", default=str(PRED_DIR))
    parser.add_argument("--error-dir", default=str(ERROR_DIR))
    args = parser.parse_args(argv)
    build_figure(args.date, args.output_dir, args.pred_dir, args.error_dir)


if __name__ == "__main__":
    main()
