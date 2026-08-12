"""Descriptor Supplementary Fig 1: data coverage of the Level 1 product.

Held the main-text Fig 3 slot until 2026-08-12, when the feature stack and
released-model importance figure took it; the coverage story supports the
gap-filled level's motivation and reads fine from the supplement.

a  A CONUS map of the fraction of days each pixel carries a valid raw
   retrieval over the record.
b  An inset time series of the daily valid-pixel count.

Together these say where and when the raw product is dense or sparse, which is
the argument for shipping the gap-filled level alongside it.

The title is a short identifier; the record dates, raster/day counts and the
in-frame median print to stdout and live in the caption (2026-08-12 handoff:
internal text is not the place for experimental metadata). CONUS land that
never carries a valid retrieval is painted light gray so it cannot be read as
ocean, which stays white -- the same no-data convention as Figs 6 and 7.

The colour ramp ends at the in-frame maximum (52%) and the key is ticked
0-50% in even steps inside that range, so every part of the bar encodes a
value that occurs; the earlier 0/20/40/60 ticks made matplotlib extend the
bar to 60% and left a white 52-60% cap that encoded nothing.

The denominator is **calendar days spanned, not files present**. SMAP's revisit
leaves whole days with no overpass and therefore no raster at all -- 67 of 366
in the 2024 record, spread evenly across every month rather than clustered,
which is the orbital pattern and not a processing failure. Counting only the
files that exist would divide those days out of the statistic and overstate
coverage by about a fifth.

The rasters are global; the map is a CONUS crop of them, and every number the
figure quotes is cut to that same crop. The per-pixel fraction, the inset's
daily count and the median in the header all count only cells whose centres
fall inside the drawn extent, so a reader can check any of them against the map
beside it. A product-wide statistic would be a different figure's number: the
tropics and the high latitudes have their own revisit, and mixing them in would
put a value in the header that nothing on the page can confirm.

The frame is Conus Albers (EPSG:5070), matching Figs 6 and 7. The grid is already
equal-area, and drawing it on raw lon/lat would stretch the north of the
country sideways -- the swath geometry the figure is about would be read
through a distortion that has nothing to do with the satellite.

Usage:
    uv run swapstress-figures --figure coverage
    uv run python -m swapstress.figures.figS1_coverage --source-dir <level1-dir>
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Patch, Rectangle
from pyproj import Transformer
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from swapstress.figures import style
from swapstress.figures.basemap import (
    lakes_shapefile,
    load_conus_states,
    pixel_corner_lonlat,
)

DATE_PATTERN = re.compile(r"_(\d{8})\.tif$")

DEFAULT_SOURCE_DIR = "/nas/soils/swapstress/releases/v03_20260729/inference_conus"
DEFAULT_OUTPUT_DIR = "figs/descriptor"
NODATA_VALUE = -9999.0

# Double column. The orbital striping that carries the figure's message is a
# few 9 km cells wide, and the inset holds a year of daily values: at 89 mm the
# swaths merge and the inset's months fall under the 5 pt floor. The height is
# whatever an equal-area CONUS needs at that width, plus the two-line header.
FIG_WIDTH_MM = style.DOUBLE_COLUMN_MM
FIG_HEIGHT_MM = 115.5

# NAD83 / Conus Albers, matching Figs 6 and 7. The rasters are already on an
# equal-area grid, so drawing them equal-area keeps CONUS the shape readers
# know; plotting straight lon/lat would stretch the north of the country
# sideways.
MAP_CRS = "EPSG:5070"
MAP_PAD = 0.015

# Lon/lat box the mesh is cut to before projecting. Albers is a regional
# projection: handed the whole global grid it sends the far hemisphere to
# nonsense coordinates, so the window comes first. It is generous enough to
# cover the drawn extent and everything outside it would be clipped anyway.
DRAW_BOX = (-134.0, 18.0, -58.0, 56.0)

BOUNDARY_COLOR = "white"
BOUNDARY_WIDTH = 0.3

# In-frame retrieved pixels top out at 0.519 and 99 % sit below 0.46, so the
# ramp ends at the in-frame maximum to spend its full range on values that
# occur. A few high-latitude cells in the mesh overhang outside the frame run
# past the ceiling and clamp to the top color; they sit outside the domain the
# statistics describe, so no "max" arrow is drawn for them.
COLOR_MAX = 0.52
# Ticks stay inside the ramp: a tick beyond vmax (the old 60%) makes the bar
# extend past the data and leaves an unencoded cap.
COLOR_TICKS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)

# CONUS land that never sees a valid retrieval, distinct from white water --
# the shared no-data convention across the map figures, defined in ``style``.
NO_RETRIEVAL_GRAY = style.NO_DATA_GRAY

# Inset plot box, and the white backing that carries its title and tick labels,
# both in axes fractions. The corner is the Pacific dead space off southern
# California: the only block of the frame wide enough for a year of daily
# values that holds no CONUS.
INSET_RECT = (0.043, 0.062, 0.285, 0.150)
INSET_BACKING = (0.018, 0.018, 0.315, 0.226)


def parse_date(path: Path) -> date:
    match = DATE_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Cannot read a YYYYMMDD date from {path.name}")
    stamp = match.group(1)
    return date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:]))


def source_paths(source_dir: str, prefix: str) -> List[Path]:
    paths = sorted(Path(source_dir).glob(f"{prefix}_*.tif"))
    if not paths:
        raise FileNotFoundError(f"No {prefix}_*.tif rasters under {source_dir}")
    return paths


class MapFrame:
    """The slice of the global grid the map draws, and its Albers mesh.

    ``window`` is deliberately larger than the drawn extent: Albers curves the
    graticule, so a lon/lat box cut to CONUS exactly would leave the corners of
    the rectangular frame empty. ``inside`` is the subset of those cells whose
    centres land within the extent -- the cells a reader can actually see, and
    therefore the only ones the figure's statistics count. Keeping the two apart
    means the drawn mesh still overhangs the frame edge, so no hairline of blank
    cells appears along the border.
    """

    def __init__(self, profile, states):
        self.states = states
        self.window = _draw_window(profile)
        self.mesh_x, self.mesh_y = _albers_mesh(profile, self.window)
        self.extent = _frame_extent(states)
        self.inside = _cells_inside(self.mesh_x, self.mesh_y, self.extent)

    @property
    def shape(self) -> tuple:
        return int(self.window.height), int(self.window.width)


def open_frame(
    source_dir: str, prefix: str = "suction", boundaries_root=None
) -> MapFrame:
    """Read the grid geometry off the first raster and build the drawn frame."""
    with rasterio.open(source_paths(source_dir, prefix)[0]) as src:
        profile = src.profile.copy()
    return MapFrame(profile, load_conus_states(boundaries_root, crs=MAP_CRS))


class Coverage:
    """Per-pixel valid-day counts and the daily in-frame total.

    ``valid_days`` spans the whole mesh window so the map has something to draw
    at its edges; ``daily_counts`` is restricted to ``frame.inside``, as is the
    median the header quotes.
    """

    def __init__(self, valid_days, calendar, daily_counts, n_files):
        self.valid_days = valid_days  # (row, col) int32, whole mesh window
        self.calendar = calendar  # every day in the span, including absent ones
        self.daily_counts = daily_counts  # valid in-frame px per day, 0 if absent
        self.n_files = n_files

    @property
    def n_days(self) -> int:
        return len(self.calendar)

    @property
    def fraction(self) -> np.ndarray:
        """Valid fraction, NaN where the pixel is never valid (off-swath/ocean)."""
        frac = self.valid_days.astype(np.float32) / float(self.n_days)
        return np.where(self.valid_days > 0, frac, np.nan)

    @property
    def absent_days(self) -> int:
        return self.n_days - self.n_files


def compute_coverage(
    source_dir: str, frame: MapFrame, prefix: str = "suction"
) -> Coverage:
    """Accumulate valid-retrieval counts across a directory of daily rasters.

    Only ``frame.window`` is read from each file -- the rasters are global and
    the figure is not -- and one day is held at a time; only the count array
    persists.
    """
    paths = source_paths(source_dir, prefix)
    dates = [parse_date(p) for p in paths]
    span = (max(dates) - min(dates)).days + 1
    calendar = [min(dates) + timedelta(days=i) for i in range(span)]
    by_date = {d: 0 for d in calendar}

    valid_days = np.zeros(frame.shape, dtype=np.int32)
    for path, day in zip(paths, dates):
        with rasterio.open(path) as src:
            data = src.read(1, window=frame.window)
            nodata = src.nodata if src.nodata is not None else NODATA_VALUE
        valid = np.isfinite(data) & (data != nodata)
        valid_days += valid
        by_date[day] = int((valid & frame.inside).sum())

    print(
        f"{len(paths)} rasters over {span} calendar days ({span - len(paths)} absent)"
    )
    return Coverage(
        valid_days=valid_days,
        calendar=calendar,
        daily_counts=[by_date[d] for d in calendar],
        n_files=len(paths),
    )


def _draw_window(profile, box=DRAW_BOX) -> Window:
    """The sub-window of the global grid that covers *box*.

    The grid is a cylindrical equal-area one, so longitude varies only along
    columns and latitude only along rows; each edge can be located on its own
    one-dimensional axis instead of searching the full mesh.
    """
    transform, height, width = (
        profile["transform"],
        profile["height"],
        profile["width"],
    )
    to_wgs84 = Transformer.from_crs(profile["crs"], "EPSG:4326", always_xy=True)
    xs = transform.c + np.arange(width + 1) * transform.a
    ys = transform.f + np.arange(height + 1) * transform.e
    lon, _ = to_wgs84.transform(xs, np.zeros_like(xs))
    _, lat = to_wgs84.transform(np.zeros_like(ys), ys)

    west, south, east, north = box
    cols = np.flatnonzero((lon >= west) & (lon <= east))
    rows = np.flatnonzero((lat >= south) & (lat <= north))
    return Window.from_slices(
        (int(rows[0]), int(rows[-1])), (int(cols[0]), int(cols[-1]))
    )


def _albers_mesh(profile, window: Window):
    """Pixel *corner* mesh of the window, in Conus Albers.

    Corners rather than centres: flat shading wants one more node than cells in
    each direction, and centres would shift the image half a 9 km pixel.
    """
    lon, lat = pixel_corner_lonlat(
        window_transform(window, profile["transform"]),
        (int(window.height), int(window.width)),
        profile["crs"],
    )
    to_albers = Transformer.from_crs("EPSG:4326", MAP_CRS, always_xy=True)
    return to_albers.transform(lon, lat)


def _frame_extent(states) -> tuple:
    """Albers ``(x0, y0, x1, y1)`` the map is drawn to: CONUS plus a hair."""
    x0, y0, x1, y1 = states.total_bounds
    dx, dy = (x1 - x0) * MAP_PAD, (y1 - y0) * MAP_PAD
    return x0 - dx, y0 - dy, x1 + dx, y1 + dy


def _cells_inside(mesh_x, mesh_y, extent) -> np.ndarray:
    """Which cells of a corner mesh have their centre inside *extent*.

    The centre is the mean of the four corners, which is exact enough at 9 km
    and avoids projecting a second mesh. Corners outside the Albers domain come
    back as ``inf``; the comparisons drop those, which is the wanted answer.
    """
    centre_x = 0.25 * (
        mesh_x[:-1, :-1] + mesh_x[1:, :-1] + mesh_x[:-1, 1:] + mesh_x[1:, 1:]
    )
    centre_y = 0.25 * (
        mesh_y[:-1, :-1] + mesh_y[1:, :-1] + mesh_y[:-1, 1:] + mesh_y[1:, 1:]
    )
    x0, y0, x1, y1 = extent
    return (centre_x >= x0) & (centre_x <= x1) & (centre_y >= y0) & (centre_y <= y1)


def _frame_map(ax, frame: MapFrame) -> None:
    """Equal-area CONUS frame: state outlines, extent, no axis furniture.

    The outlines are white because the ramp is dark at the low end, where a grey
    hairline disappears. There are no ticks: projected metres mean nothing to a
    reader, and the state outlines already say where everything is.

    The states are filled ``NO_RETRIEVAL_GRAY`` *under* the mesh: a valid cell
    covers its patch of gray, so gray survives only where CONUS land never
    carries a retrieval, and it cannot be confused with white water. Lakes are
    painted back white on top for the same reason in reverse.
    """
    frame.states.plot(ax=ax, facecolor=NO_RETRIEVAL_GRAY, edgecolor="none", zorder=0.5)
    lakes = gpd.read_file(lakes_shapefile())
    lakes = lakes[lakes["scalerank"] <= 3].to_crs(MAP_CRS)
    lakes.plot(ax=ax, facecolor="white", edgecolor="none", zorder=2)
    frame.states.boundary.plot(
        ax=ax, edgecolor=BOUNDARY_COLOR, linewidth=BOUNDARY_WIDTH, alpha=0.85, zorder=3
    )
    ax.legend(
        handles=[Patch(facecolor=NO_RETRIEVAL_GRAY, edgecolor="none")],
        labels=["land, never retrieved"],
        loc="lower right",
        frameon=False,
        fontsize=style.MIN_TEXT_PT + 1,
        handlelength=1.0,
        handleheight=1.0,
        handletextpad=0.4,
    )
    x0, y0, x1, y1 = frame.extent
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_axis_off()


def _add_colorbar(fig, mesh, ax) -> None:
    """Colour key, in per cent so it reads against the median quoted above."""
    bar = fig.colorbar(mesh, ax=ax, shrink=0.86, aspect=24, pad=0.012)
    bar.set_label(
        "Valid retrieval days (%)",
        fontsize=style.MAX_TEXT_PT,
        labelpad=3,
    )
    bar.set_ticks(list(COLOR_TICKS))
    bar.set_ticklabels([f"{tick:.0%}".rstrip("%") for tick in COLOR_TICKS])
    bar.ax.tick_params(labelsize=style.MAX_TEXT_PT - 1, length=2.0, width=0.5, pad=1.5)
    bar.outline.set_linewidth(0.5)
    bar.outline.set_edgecolor(style.AXIS_COLOR)


def _panel_backing(ax) -> None:
    """White card behind the inset, sized to hold its title and tick labels too.

    Without it the inset's own axes patch clips at the plot box and the month
    labels fall straight onto the mesh, which reads as text lost on the map.
    """
    x0, y0, width, height = INSET_BACKING
    ax.add_patch(
        Rectangle(
            (x0, y0),
            width,
            height,
            transform=ax.transAxes,
            facecolor="white",
            edgecolor=style.AXIS_COLOR,
            linewidth=0.4,
            zorder=4,
        )
    )


def _add_daily_inset(ax, coverage: Coverage) -> None:
    """Daily valid-pixel count, over the Pacific/Mexico corner the map leaves empty.

    Counted over the drawn frame, like everything else on the figure. Most drops
    to zero are calendar days with no overpass at all, so the comb is the
    record's cadence rather than noise -- the same 67 days the map's denominator
    refuses to discard. One is not: 23 September 2024 has a raster, but its
    granules only reach the eastern hemisphere, so nothing lands in frame. That
    is exactly the kind of hole a product-wide count would hide.
    """
    _panel_backing(ax)
    counts = np.array(coverage.daily_counts) / 1e3
    inset = ax.inset_axes(INSET_RECT)
    # A light fill rather than a solid block: the opaque version carried more
    # visual mass than its physical size and competed with the coverage map
    # the inset supports. No line on top -- eleven years of daily steps at any
    # weight shade themselves back into a block.
    inset.fill_between(
        coverage.calendar,
        counts,
        step="mid",
        color=style.CATEGORICAL[0],
        alpha=0.35,
        linewidth=0.0,
    )
    inset.set_title(
        "Valid pixels per day, in frame (thousands)",
        fontsize=style.MIN_TEXT_PT + 1,
        color=style.AXIS_COLOR,
        pad=2.0,
        loc="left",
    )
    # The record is eleven years; month ticks (the single-year original) shred
    # into each other at this width, so tick alternate Januaries instead.
    inset.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    inset.xaxis.set_major_locator(mdates.YearLocator(base=2))
    inset.set_yticks([0, 30, 60])
    inset.set_ylim(0.0, max(72.0, counts.max() * 1.08))
    inset.margins(x=0.01)
    inset.tick_params(labelsize=style.MIN_TEXT_PT + 0.5, length=1.5, width=0.4, pad=1.0)
    inset.set_facecolor("none")
    for side, spine in inset.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_linewidth(0.4)
        spine.set_edgecolor(style.AXIS_COLOR)


def render(coverage: Coverage, frame: MapFrame, output_dir: str) -> Path:
    """Draw the coverage map with its daily-count inset."""
    style.apply()

    # Drawn over the whole mesh window so the frame edge has no blank hairline;
    # quoted over ``frame.inside`` only, which is what a reader can see.
    fraction = coverage.fraction
    mesh_x, mesh_y = frame.mesh_x, frame.mesh_y
    quoted = fraction[frame.inside & np.isfinite(fraction)]

    fig = plt.figure(
        figsize=style.figsize(FIG_WIDTH_MM, FIG_HEIGHT_MM), layout="constrained"
    )
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.0, hspace=0.0)
    ax = fig.add_subplot()

    mesh = ax.pcolormesh(
        mesh_x,
        mesh_y,
        fraction,
        cmap=style.SEQUENTIAL,
        vmin=0.0,
        vmax=COLOR_MAX,
        shading="flat",
        zorder=1,
        # Rasterises the 9 km cells only. Outlines, ticks and every label stay
        # vector, which is what the artwork guide actually asks for.
        rasterized=True,
    )
    _frame_map(ax, frame)
    _add_colorbar(fig, mesh, ax)

    # The record's metadata goes to the caption, not the artwork; print it so
    # the caption writer has the exact numbers this render was cut from.
    first, last = coverage.calendar[0], coverage.calendar[-1]
    print(
        f"For the caption: {first:%-d %b %Y} – {last:%-d %b %Y}; "
        f"{coverage.n_files} daily rasters over {coverage.n_days} calendar days "
        f"({coverage.absent_days} with no overpass); {quoted.size} retrieved "
        f"pixels in frame; median {np.median(quoted):.1%} of calendar days "
        "per pixel"
    )
    # Centered, so the short title clears the panel letter at top-left.
    ax.set_title(
        "Valid Level 1 retrieval days",
        fontsize=style.MAX_TEXT_PT,
        color=style.AXIS_COLOR,
        pad=3.0,
    )
    style.panel_label(ax, "a", dx=0.0, dy=1.005)

    _add_daily_inset(ax, coverage)
    # The inset is panel b: its label sits at the top-left of the white
    # backing card, clear of the mesh underneath.
    bx, by, _, bh = INSET_BACKING
    ax.text(
        bx + 0.008,
        by + bh - 0.012,
        "b",
        transform=ax.transAxes,
        fontsize=style.PANEL_LABEL_PT,
        fontweight="bold",
        va="top",
        ha="left",
        zorder=6,
    )

    return style.save(fig, Path(output_dir) / "figS1_coverage")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="figS1_coverage",
        description="Descriptor Supplementary Fig 1: Level 1 retrieval coverage.",
    )
    parser.add_argument(
        "--source-dir",
        default=DEFAULT_SOURCE_DIR,
        help="Directory of Level 1 <prefix>_YYYYMMDD.tif rasters.",
    )
    parser.add_argument(
        "--prefix",
        default="suction",
        help="Filename prefix before _YYYYMMDD.tif (default: suction).",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--boundaries-root",
        default=None,
        help="Root holding boundaries/states/...; defaults to "
        "SWAPSTRESS_BOUNDARIES_ROOT or /nas.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    frame = open_frame(
        args.source_dir, prefix=args.prefix, boundaries_root=args.boundaries_root
    )
    coverage = compute_coverage(args.source_dir, frame, prefix=args.prefix)
    path = render(coverage, frame, args.output_dir)
    print(f"Saved to {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
