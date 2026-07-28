"""Animated MP4 of CONUS log10 suction (source + gap-filled merged).

Gap-filled frames are labelled in the title. Output written to /tmp/.

Usage:
    uv run python viz/inferences/animate_conus_suction.py \
        --start 20150401 --end 20260415

    # Defaults to full period of record if no dates given.
"""

from __future__ import annotations

import argparse
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.animation import FFMpegWriter, FuncAnimation

SOURCE_DIR = Path(
    "/nas/soils/swapstress/releases/global_pruned_refresh_20260520/inference"
)
GAPFILL_DIR = Path(
    "/nas/soils/swapstress/releases/global_pruned_refresh_20260520/gapfill"
)
OUT_DIR = Path("/tmp")

NODATA = -9999.0
VMIN, VMAX = 1.2, 4.8  # log10 suction cm
FPS = 24
CMAP = "RdBu_r"  # blue = wet (low suction), red = dry (high suction)
HEIGHT, WIDTH = 295, 634

_RE = re.compile(r"^suction_(\d{8})\.tif$")


def _discover(start: date, end: date) -> dict[date, tuple[Path, bool]]:
    """Return {date: (path, is_gap_filled)} for all available rasters in range.

    Reads from GAPFILL_DIR (spatially + temporally complete); falls back to
    SOURCE_DIR for any dates not yet gap-filled.
    """
    result: dict[date, tuple[Path, bool]] = {}
    for directory, is_gf in [(GAPFILL_DIR, True), (SOURCE_DIR, False)]:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("suction_*.tif")):
            m = _RE.match(path.name)
            if not m:
                continue
            d = datetime.strptime(m.group(1), "%Y%m%d").date()
            if not (start <= d <= end):
                continue
            if d not in result:
                result[d] = (path, is_gf)
    return dict(sorted(result.items()))


def _load(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
    data[data == NODATA] = np.nan
    data[~np.isfinite(data)] = np.nan
    return data.reshape(HEIGHT, WIDTH)


def _blank() -> np.ndarray:
    return np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)


def _year_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """Split a date range into per-year chunks."""
    chunks = []
    cur = start
    while cur <= end:
        year_end = min(date(cur.year, 12, 31), end)
        chunks.append((cur, year_end))
        cur = date(cur.year + 1, 1, 1)
    return chunks


def animate_year(
    year_start: date,
    year_end: date,
    rasters: dict[date, tuple[Path, bool]],
    out_path: Path,
) -> None:
    """Write one MP4 for the given year range."""
    n_days = (year_end - year_start).days + 1
    all_dates = [year_start + timedelta(days=i) for i in range(n_days)]

    year_rasters = {d: v for d, v in rasters.items() if year_start <= d <= year_end}
    n_source = sum(1 for _, (_, gf) in year_rasters.items() if not gf)
    n_gf = sum(1 for _, (_, gf) in year_rasters.items() if gf)
    print(
        f"\n{year_start.year}: {len(year_rasters)} rasters "
        f"(source: {n_source}, gap-filled: {n_gf}), {n_days} frames"
    )

    blank = _blank()
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=100)
    fig.subplots_adjust(bottom=0.05, top=0.90)

    im = ax.imshow(
        blank,
        cmap=CMAP,
        vmin=VMIN,
        vmax=VMAX,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_axis_off()

    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("log\u2081\u2080 suction (cm)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    title_text = ax.set_title("", fontsize=11, pad=4)

    def update(frame_idx: int):
        d = all_dates[frame_idx]
        if d in year_rasters:
            path, is_gf = year_rasters[d]
            data = _load(path)
            label = d.strftime("%Y-%m-%d")
            if is_gf:
                label += "  [gap-filled]"
        else:
            data = blank
            label = d.strftime("%Y-%m-%d") + "  [no data]"

        im.set_data(data)
        title_text.set_text(label)

        if (frame_idx + 1) % 60 == 0 or (frame_idx + 1) == len(all_dates):
            print(f"  rendered {frame_idx + 1}/{len(all_dates)}", flush=True)

        return (im, title_text)

    ani = FuncAnimation(
        fig, update, frames=len(all_dates), interval=1000 // FPS, blit=True
    )

    writer = FFMpegWriter(
        fps=FPS,
        codec="libx264",
        extra_args=["-crf", "20", "-preset", "fast", "-pix_fmt", "yuv420p"],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {out_path} ...", flush=True)
    ani.save(str(out_path), writer=writer)
    plt.close(fig)
    size_mb = out_path.stat().st_size / 1e6
    print(f"Wrote {out_path}  ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Animate CONUS suction predictions (per-year MP4s)",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Start date YYYYMMDD (default: earliest available)",
    )
    parser.add_argument(
        "--end", default=None, help="End date YYYYMMDD (default: latest available)"
    )
    parser.add_argument(
        "--out-dir", default=str(OUT_DIR), help=f"Output directory (default: {OUT_DIR})"
    )
    args = parser.parse_args()

    # Discover all rasters to determine date range
    scan_start = date(2015, 1, 1)
    scan_end = date(2030, 12, 31)
    all_rasters = _discover(scan_start, scan_end)

    if not all_rasters:
        raise FileNotFoundError("No suction rasters found")

    first_date = min(all_rasters)
    last_date = max(all_rasters)

    start = datetime.strptime(args.start, "%Y%m%d").date() if args.start else first_date
    end = datetime.strptime(args.end, "%Y%m%d").date() if args.end else last_date

    rasters = {d: v for d, v in all_rasters.items() if start <= d <= end}
    n_total = len(rasters)
    n_gf = sum(1 for _, gf in rasters.values() if gf)
    print(f"Period: {start} to {end}")
    print(f"Total rasters: {n_total}  (source: {n_total - n_gf}, gap-filled: {n_gf})")

    out_dir = Path(args.out_dir)
    for year_start, year_end in _year_chunks(start, end):
        out_path = out_dir / f"conus_suction_{year_start.year}.mp4"
        animate_year(year_start, year_end, rasters, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
