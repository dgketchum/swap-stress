"""Supporting analysis (was Fig 3): daily matric potential at representative pixels across a gradient.

For each site, the Level 1 retrievals are drawn as points and the Level 2
gap-filled series as a line beneath them, so a reuser can see three things at
once: the true observation cadence, where the gaps are, and the drydown/rewet
dynamics the product resolves.

Both are drawn in the descriptor's presentation unit, ``log10|psi|`` with psi
in MPa, with a linear MPa axis on the right; the pipeline's internal
``log10_suction_cm`` never reaches the page.

The Level 2 line is not recomputed here. It comes from
``swapstress.inference.gapfill.interpolate_pixel`` -- the same function stage 06
applies to every pixel -- run over the same calendar span the stage would use
(first through last available raster). The curve is therefore the one the
released Level 2 raster carries at that pixel, not a lookalike.

Where Level 2 is held flat rather than interpolated -- before the first
retrieval and after the last, which ``np.interp`` clamps -- the line is shaded,
and the legend names the shading rather than leaving it to the caption. That is
the product's weakest region and the figure should not hide it. The shading
covers each clamped day's full width, so a one-day clamp still draws: at the
Humid Southeast and Arid Southwest pixels that is all there is, a single day at
either end of the year, and it should read as the sliver it actually is.

Usage:
    uv run swapstress-figures --figure pixel-series
    uv run python -m swapstress.figures.pixel_series --source-dir <dir>
"""

from __future__ import annotations

import argparse
import os
import string
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import rasterio
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pyproj import Transformer

from swapstress.figures import style
from swapstress.inference.gapfill import (
    NODATA_VALUE,
    discover_source_rasters,
    interpolate_pixel,
)
from swapstress.units import log10_suction_cm_to_log10_abs_mpa

DEFAULT_SOURCE_DIR = (
    "/nas/soils/swapstress/releases/global_pruned_refresh_20260520/inference_l3"
)
DEFAULT_PREFIX = "suction"
DEFAULT_OUTPUT_DIR = "figs/descriptor"

# Ordered wet to dry so the panels read as a gradient.
SITES = [
    ("Humid Southeast", "Athens, GA", -83.4, 33.9),
    ("Arid Southwest", "Tucson, AZ", -110.9, 32.2),
    ("Great Plains", "Hays, KS", -99.3, 38.9),
]

# The two data layers take the first two slots of the validated categorical
# palette. The clamp shading is deliberately achromatic: it marks a region of
# the record, not a third series, and must not compete with either layer.
LEVEL1_COLOR = style.CATEGORICAL[0]
LEVEL2_COLOR = style.CATEGORICAL[1]
CLAMP_COLOR = "#d5d5d5"

# Round matric-potential values to label the right-hand axis with. The panels
# are drawn in log10 of the same magnitude, so these land on exact positions.
MPA_TICKS = (0.01, 0.1, 1.0, 10.0)

# Double column across, because a year of daily values needs the width. The
# depth is set by the panels: the three share one y scale so the climate
# offsets stay comparable, which costs each panel some empty range, and the
# stack has to stay tall enough that the drydown and rewet swings still read.
FIGURE_HEIGHT_MM = 140.0


def site_series(source_dir: str, prefix: str) -> tuple[pd.DatetimeIndex, list[dict]]:
    """Level 1 and Level 2 series at each site over the record's calendar span.

    One raster is opened at a time and sampled at the site pixels, so this costs
    a few hundred small windowed reads rather than a stack of global arrays.

    The rasters carry ``log10_suction_cm``; the returned series are in
    ``log10|psi|`` with psi in MPa, the descriptor's presentation unit. The
    conversion is the exact additive shift from ``swapstress.units``, applied
    *after* the gap-fill so the sentinel ``NODATA_VALUE`` is never shifted --
    and, being additive, it commutes with the linear interpolation anyway, so
    the drawn Level 2 line is still the one the released raster carries.
    """
    rasters = discover_source_rasters(source_dir, prefix)
    if not rasters:
        raise FileNotFoundError(f"No {prefix}_YYYYMMDD.tif rasters in {source_dir}")

    dates = sorted(rasters)
    with rasterio.open(rasters[dates[0]]) as src:
        to_grid = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        index = [src.index(*to_grid.transform(lon, lat)) for _, _, lon, lat in SITES]

    # The stage spans first through last raster and writes every calendar day in
    # between, so absent days must be present here as NaN, not skipped.
    span = pd.date_range(dates[0], dates[-1], freq="D")
    position = {d.date(): i for i, d in enumerate(span)}
    raw = np.full((len(SITES), len(span)), np.nan, dtype=np.float32)

    for day in dates:
        with rasterio.open(rasters[day]) as src:
            for s, (row, col) in enumerate(index):
                value = src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]
                if value != NODATA_VALUE and np.isfinite(value):
                    raw[s, position[day]] = value

    day_index = np.arange(len(span), dtype=np.float32)
    out = []
    for s, (region, place, lon, lat) in enumerate(SITES):
        observed = ~np.isnan(raw[s])
        if not observed.any():
            raise ValueError(
                f"{region} ({place}) has no valid retrieval in the whole record; "
                "it is not a land pixel in this product."
            )
        filled = interpolate_pixel(day_index, raw[s], day_index)
        filled = np.where(filled == NODATA_VALUE, np.nan, filled)
        first, last = np.flatnonzero(observed)[[0, -1]]
        out.append(
            {
                "region": region,
                "place": place,
                "lon": lon,
                "lat": lat,
                "raw": log10_suction_cm_to_log10_abs_mpa(raw[s]),
                "filled": log10_suction_cm_to_log10_abs_mpa(filled),
                "observed": observed,
                "clamped": (day_index < first) | (day_index > last),
            }
        )
    return span, out


def render(span: pd.DatetimeIndex, series: list[dict], output_dir: str) -> Path:
    """Draw the three panels at Nature's double-column width.

    Sizing is declared once, in millimetres, and survives to disk: the figure is
    written at exactly its declared size rather than cropped to its content, so
    it drops into a 183 mm column without rescaling the type.
    """
    style.apply()
    fig, axes = plt.subplots(
        len(series),
        1,
        figsize=style.figsize(style.DOUBLE_COLUMN_MM, FIGURE_HEIGHT_MM),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    axes = np.atleast_1d(axes)

    low = min(np.nanmin(s["filled"]) for s in series) - 0.25
    high = max(np.nanmax(s["filled"]) for s in series) + 0.25

    # A clamped run of a single day is still a real day of the record, so the
    # shading covers each day's full width rather than collapsing to no width.
    half_day = pd.Timedelta(hours=12)

    mpa_axes = []
    for ax, letter, s in zip(axes, string.ascii_lowercase, series):
        ax.set_axisbelow(True)
        ax.grid(axis="y")

        for start, stop in clamped_spans(s["clamped"]):
            ax.axvspan(
                span[start] - half_day,
                span[stop] + half_day,
                facecolor=CLAMP_COLOR,
                edgecolor="none",
                zorder=0,
            )

        ax.plot(span, s["filled"], color=LEVEL2_COLOR, lw=0.6, zorder=2)
        ax.scatter(
            span[s["observed"]],
            s["raw"][s["observed"]],
            s=3.0,
            color=LEVEL1_COLOR,
            linewidths=0,
            zorder=3,
        )

        ax.set_ylim(low, high)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(1.0))

        style.panel_label(ax, letter, dx=0.0, dy=1.02)
        ax.text(
            0.017,
            1.02,
            f"{s['region']} · {s['place']} · "
            f"{s['observed'].sum()} of {len(span)} days retrieved",
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=style.MAX_TEXT_PT,
            color=style.AXIS_COLOR,
        )

        # Linear matric potential on the right, against the log10 magnitude on
        # the left. Same measure, same unit, two readings of one scale, so
        # these ticks are exact rather than approximated.
        #
        # A secondary axis rather than a twin: ``set_yticks`` widens a twin's
        # limits to reach its outermost tick, which silently slides the MPa
        # labels off the values they name. A secondary axis re-derives its
        # limits from the parent at every draw, so it cannot come loose.
        mpa = ax.secondary_yaxis("right", functions=(lambda y: y, lambda y: y))
        labelled = [(t, np.log10(t)) for t in MPA_TICKS]
        labelled = [(t, y) for t, y in labelled if low <= y <= high]
        mpa.set_yticks([y for _, y in labelled])
        mpa.set_yticklabels([f"\N{MINUS SIGN}{t:g}" for t, _ in labelled])
        mpa.tick_params(length=0, labelsize=style.MAX_TEXT_PT - 1)
        mpa.spines["right"].set_visible(False)
        mpa_axes.append(mpa)

    # One unit label per side for the whole stack -- centred on the middle
    # panel on the right, figure-level on the left -- rather than three times.
    mpa_axes[len(mpa_axes) // 2].set_ylabel(
        style.MPA_AXIS, rotation=270, va="bottom", labelpad=7
    )
    fig.supylabel(style.LOG10_ABS_MPA_AXIS, fontsize=style.MAX_TEXT_PT)

    # Month names sit mid-month between boundary ticks, so a tick means the
    # first of the month and no label straddles the year's ends.
    bottom = axes[-1]
    bottom.set_xlim(span[0], span[-1])
    bottom.xaxis.set_major_locator(mdates.MonthLocator())
    bottom.xaxis.set_major_formatter(mticker.NullFormatter())
    bottom.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=16))
    bottom.xaxis.set_minor_formatter(mdates.DateFormatter("%b"))
    bottom.tick_params(axis="x", which="minor", length=0)
    bottom.set_xlabel(str(span[0].year))

    fig.legend(
        handles=legend_handles(),
        loc="outside lower center",
        ncol=3,
        columnspacing=1.8,
        handletextpad=0.5,
    )

    return style.save(fig, Path(output_dir) / "pixel_series")


def legend_handles() -> list:
    """Legend proxies, so the shading is named rather than left to the caption."""
    return [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=2.0,
            color=LEVEL1_COLOR,
            label="Level 1 retrieval",
        ),
        Line2D([], [], color=LEVEL2_COLOR, lw=0.8, label="Level 2 gap-filled"),
        Patch(
            facecolor=CLAMP_COLOR,
            edgecolor="none",
            label="Level 2 held flat outside the observed range",
        ),
    ]


def clamped_spans(clamped: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [start, stop] index runs where Level 2 is extrapolated flat."""
    spans, start = [], None
    for i, flag in enumerate(clamped):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            spans.append((start, i - 1))
            start = None
    if start is not None:
        spans.append((start, len(clamped) - 1))
    return spans


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pixel_series",
        description=(
            "Supporting analysis: daily matric potential at pixels across a climate gradient."
        ),
    )
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    span, series = site_series(args.source_dir, args.prefix)
    print(f"{len(span)} calendar days, {span[0].date()} to {span[-1].date()}")
    for s in series:
        print(
            f"  {s['region']:18} {s['observed'].sum():3d} retrieval days  "
            f"log10|psi| MPa {np.nanmin(s['raw']):.2f}-{np.nanmax(s['raw']):.2f}"
        )
    path = render(span, series, args.output_dir)
    print(f"Saved to {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
