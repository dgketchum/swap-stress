"""Figure 3: daily suction at representative pixels across a climate gradient.

For each site, the Level 1 retrievals are drawn as points and the Level 2
gap-filled series as a line beneath them, so a reuser can see three things at
once: the true observation cadence, where the gaps are, and the drydown/rewet
dynamics the product resolves.

The Level 2 line is not recomputed here. It comes from
``swapstress.inference.gapfill.interpolate_pixel`` -- the same function stage 06
applies to every pixel -- run over the same calendar span the stage would use
(first through last available raster). The curve is therefore the one the
released Level 2 raster carries at that pixel, not a lookalike.

Where Level 2 is held flat rather than interpolated -- before the first
retrieval and after the last, which ``np.interp`` clamps -- the line is shaded.
That is the product's weakest region and the figure should not hide it.

Usage:
    uv run swapstress-figures --figure pixel-series
    uv run python -m swapstress.figures.fig03_pixel_series --source-dir <dir>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

from swapstress.inference.gapfill import (
    NODATA_VALUE,
    discover_source_rasters,
    interpolate_pixel,
)
from swapstress.inference.product import MPA_TO_CM

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

LEVEL1_COLOR = "#1f4e79"
LEVEL2_COLOR = "#c0562a"
CLAMP_COLOR = "#b0b0b0"

# Round matric-potential values to label the right-hand axis with. The
# conversion is an exact shift in log space, so these land on exact positions.
MPA_TICKS = (0.01, 0.1, 1.0, 10.0)


def site_series(source_dir: str, prefix: str) -> tuple[pd.DatetimeIndex, list[dict]]:
    """Level 1 and Level 2 series at each site over the record's calendar span.

    One raster is opened at a time and sampled at the site pixels, so this costs
    a few hundred small windowed reads rather than a stack of global arrays.
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
        first, last = np.flatnonzero(observed)[[0, -1]]
        out.append(
            {
                "region": region,
                "place": place,
                "lon": lon,
                "lat": lat,
                "raw": raw[s],
                "filled": np.where(filled == NODATA_VALUE, np.nan, filled),
                "observed": observed,
                "clamped": (day_index < first) | (day_index > last),
            }
        )
    return span, out


def render(span: pd.DatetimeIndex, series: list[dict], output_dir: str) -> Path:
    fig, axes = plt.subplots(
        len(series), 1, figsize=(10.5, 2.5 * len(series)), sharex=True
    )
    axes = np.atleast_1d(axes)

    low = min(np.nanmin(s["filled"]) for s in series) - 0.25
    high = max(np.nanmax(s["filled"]) for s in series) + 0.25

    for ax, s in zip(axes, series):
        for start, stop in clamped_spans(s["clamped"]):
            ax.axvspan(
                span[start], span[stop], color=CLAMP_COLOR, alpha=0.18, lw=0, zorder=0
            )

        ax.plot(
            span,
            s["filled"],
            color=LEVEL2_COLOR,
            lw=1.3,
            zorder=2,
            label="Level 2 (gap-filled)",
        )
        ax.scatter(
            span[s["observed"]],
            s["raw"][s["observed"]],
            s=9,
            color=LEVEL1_COLOR,
            zorder=3,
            label="Level 1 (retrieval days)",
        )

        ax.set_ylim(low, high)
        ax.set_ylabel(r"$\log_{10}\,\psi$ (cm)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.15)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.text(
            0.008,
            0.94,
            f"{s['region']}  ·  {s['place']}  ·  "
            f"{s['observed'].sum()} of {len(span)} days",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
        )

        # Matric potential on the right. log10(cm) -> log10(MPa) is a constant
        # offset, so these ticks are exact rather than approximated.
        mpa = ax.twinx()
        mpa.set_ylim(low, high)
        mpa.set_yticks([np.log10(t * MPA_TO_CM) for t in MPA_TICKS])
        mpa.set_yticklabels([f"-{t:g}" for t in MPA_TICKS], fontsize=7)
        mpa.set_ylabel("MPa", fontsize=8, rotation=270, labelpad=11)
        mpa.tick_params(length=0)
        for side in ("top", "left"):
            mpa.spines[side].set_visible(False)

    # The wettest panel is the one with headroom, the axes being on a shared
    # scale so the climate offsets stay comparable.
    axes[0].legend(fontsize=8, loc="upper right", framealpha=0.9, ncol=2)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].set_xlim(span[0], span[-1])
    fig.suptitle(
        f"Daily suction across a climate gradient — {span[0].year}\n"
        "shaded: Level 2 held flat outside the observed range",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / "fig03_pixel_series"
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return Path(f"{stem}.png")


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
        prog="fig03_pixel_series",
        description="Figure 3: daily suction at pixels across a climate gradient.",
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
            f"log10 psi {np.nanmin(s['raw']):.2f}-{np.nanmax(s['raw']):.2f}"
        )
    path = render(span, series, args.output_dir)
    print(f"Saved to {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
