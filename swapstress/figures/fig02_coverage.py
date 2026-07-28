"""Figure 2: data coverage of the Level 1 product.

A CONUS map of the fraction of days each pixel carries a valid raw retrieval
over the record, with an inset time series of the daily national valid-pixel
count. Together these say where and when the raw product is dense or sparse,
which is the argument for shipping the gap-filled level alongside it.

The denominator is **calendar days spanned, not files present**. SMAP's revisit
leaves whole days with no overpass and therefore no raster at all -- 67 of 366
in the 2024 record, spread evenly across every month rather than clustered,
which is the orbital pattern and not a processing failure. Counting only the
files that exist would divide those days out of the statistic and overstate
coverage by about a fifth.

Usage:
    uv run swapstress-figures --figure coverage
    uv run python -m swapstress.figures.fig02_coverage --source-dir <level1-dir>
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
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from swapstress.figures.basemap import (
    load_conus_states,
    pixel_corner_lonlat,
    style_conus_axis,
)

DATE_PATTERN = re.compile(r"_(\d{8})\.tif$")

DEFAULT_SOURCE_DIR = (
    "/nas/soils/swapstress/releases/global_pruned_refresh_20260520/inference_l3"
)
DEFAULT_OUTPUT_DIR = "figs/descriptor"
NODATA_VALUE = -9999.0


def parse_date(path: Path) -> date:
    match = DATE_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Cannot read a YYYYMMDD date from {path.name}")
    stamp = match.group(1)
    return date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:]))


class Coverage:
    """Per-pixel valid-day counts and the daily national total."""

    def __init__(self, valid_days, calendar, daily_counts, profile, n_files):
        self.valid_days = valid_days  # (row, col) int32
        self.calendar = calendar  # every day in the span, including absent ones
        self.daily_counts = daily_counts  # valid px per calendar day, 0 if absent
        self.profile = profile
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


def compute_coverage(source_dir: str, prefix: str = "suction") -> Coverage:
    """Accumulate valid-retrieval counts across a directory of daily rasters.

    One day is held at a time; only the count array persists.
    """
    paths = sorted(Path(source_dir).glob(f"{prefix}_*.tif"))
    if not paths:
        raise FileNotFoundError(f"No {prefix}_*.tif rasters under {source_dir}")

    dates = [parse_date(p) for p in paths]
    span = (max(dates) - min(dates)).days + 1
    calendar = [min(dates) + timedelta(days=i) for i in range(span)]
    by_date = {d: 0 for d in calendar}

    valid_days = None
    profile = None
    for path, day in zip(paths, dates):
        with rasterio.open(path) as src:
            data = src.read(1)
            nodata = src.nodata if src.nodata is not None else NODATA_VALUE
            if profile is None:
                profile = src.profile.copy()
                valid_days = np.zeros(data.shape, dtype=np.int32)
        valid = np.isfinite(data) & (data != nodata)
        valid_days += valid
        by_date[day] = int(valid.sum())

    print(
        f"{len(paths)} rasters over {span} calendar days ({span - len(paths)} absent)"
    )
    return Coverage(
        valid_days=valid_days,
        calendar=calendar,
        daily_counts=[by_date[d] for d in calendar],
        profile=profile,
        n_files=len(paths),
    )


def render(coverage: Coverage, output_dir: str, boundaries_root=None) -> Path:
    """Draw the coverage map with its daily-count inset."""
    fraction = coverage.fraction
    lon, lat = pixel_corner_lonlat(
        coverage.profile["transform"],
        (coverage.profile["height"], coverage.profile["width"]),
        coverage.profile["crs"],
    )
    states = load_conus_states(boundaries_root)

    fig, ax = plt.subplots(figsize=(11.0, 6.4))
    mesh = ax.pcolormesh(
        lon,
        lat,
        fraction,
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        shading="flat",
        zorder=1,
        rasterized=True,
    )
    style_conus_axis(ax, states=states, root=boundaries_root)

    bar = fig.colorbar(mesh, ax=ax, shrink=0.72, pad=0.02)
    bar.set_label("Fraction of days with a valid Level 1 retrieval", fontsize=9)
    bar.ax.tick_params(labelsize=8)

    first, last = coverage.calendar[0], coverage.calendar[-1]
    covered = np.isfinite(fraction)
    ax.set_title(
        f"Level 1 retrieval coverage, {first:%Y-%m-%d} to {last:%Y-%m-%d}\n"
        f"{coverage.n_files} daily rasters over {coverage.n_days} calendar days "
        f"({coverage.absent_days} with no overpass); "
        f"median {np.nanmedian(fraction[covered]):.0%} of days per pixel",
        fontsize=10,
    )

    # Inset over the Pacific/Mexico corner, where the map carries no data. Each
    # drop to zero is a day with no overpass at all, so the comb is the record's
    # cadence rather than noise.
    inset = ax.inset_axes([0.025, 0.045, 0.30, 0.155])
    inset.fill_between(
        coverage.calendar,
        np.array(coverage.daily_counts) / 1e6,
        step="mid",
        color="#2b6a99",
        linewidth=0.0,
        alpha=0.9,
    )
    inset.set_ylabel("valid px (M)", fontsize=6.0, labelpad=2)
    inset.tick_params(labelsize=5.5, length=2, pad=1)
    inset.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    inset.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    inset.margins(x=0.01)
    inset.set_facecolor("white")
    inset.patch.set_alpha(0.82)
    for side in ("top", "right"):
        inset.spines[side].set_visible(False)
    inset.set_title(
        "Daily national valid-pixel count", fontsize=6.0, pad=1.5, loc="left"
    )

    fig.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / "fig02_coverage"
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return Path(f"{stem}.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fig02_coverage",
        description="Figure 2: Level 1 retrieval coverage.",
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
    coverage = compute_coverage(args.source_dir, prefix=args.prefix)
    path = render(coverage, args.output_dir, boundaries_root=args.boundaries_root)
    print(f"Saved to {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
