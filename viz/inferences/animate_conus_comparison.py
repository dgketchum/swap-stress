"""Three-panel MP4: raw SMAP VWC / swath suction / gap-filled suction, per year.

Panels:
  1. Raw SMAP L3 soil moisture (viridis_r) — orbital swath gaps
  2. Inferred log10 suction, swath-level (RdBu_r) — same gaps as SMAP
  3. Inferred log10 suction, gap-filled (RdBu_r) — temporally complete

Output: /tmp/conus_comparison_YYYY.mp4  (H.264, scrubbable)

Usage:
    uv run python viz/inferences/animate_conus_comparison.py \
        --start 20150401 --end 20260415
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

SUCTION_DIR = Path(
    "/nas/soils/swapstress/releases/global_pruned_refresh_20260520/inference"
)
SUCTION_GF_DIR = Path(
    "/nas/soils/swapstress/releases/global_pruned_refresh_20260520/gapfill"
)
SMAP_RAW_DIR = Path("/nas/soils/smap/SPL3SMP_E/daily_tif")

SUCTION_VMIN, SUCTION_VMAX = 1.2, 4.8
VWC_VMIN, VWC_VMAX = 0.02, 0.55

FPS = 8
HEIGHT, WIDTH = 295, 634
NODATA = -9999.0
OUT_DIR = Path("/tmp")

_RE_SUCTION = re.compile(r"^suction_(\d{8})\.tif$")
_RE_SMAP = re.compile(r"^smap_sm_(\d{8})\.tif$")


def _discover(
    directory: Path, pattern: re.Pattern, start: date, end: date
) -> dict[date, Path]:
    if not directory.exists():
        return {}
    result: dict[date, Path] = {}
    prefix = pattern.pattern.lstrip("^").split("_")[0]
    for path in sorted(directory.glob(f"{prefix}*.tif")):
        m = pattern.match(path.name)
        if not m:
            continue
        d = datetime.strptime(m.group(1), "%Y%m%d").date()
        if start <= d <= end:
            result[d] = path
    return result


def _load(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nd = src.nodata
    if nd is not None and np.isfinite(nd):
        data[data == nd] = np.nan
    data[data == NODATA] = np.nan
    data[~np.isfinite(data)] = np.nan
    return data.reshape(HEIGHT, WIDTH)


def _blank() -> np.ndarray:
    return np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)


def _year_chunks(start: date, end: date) -> list[tuple[date, date]]:
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
    smap_raw: dict[date, Path],
    suction_raw: dict[date, Path],
    suction_gf: dict[date, Path],
    out_path: Path,
    max_frames: int | None = None,
) -> None:
    """Three-panel animation for one year.

    Layout: left column has SMAP raw (top) and swath suction (bottom) at
    smaller size; right column has gap-filled suction at full height.
    """
    n_days = (year_end - year_start).days + 1
    all_dates = [year_start + timedelta(days=i) for i in range(n_days)]
    if max_frames is not None:
        all_dates = all_dates[:max_frames]
        n_days = len(all_dates)

    print(
        f"\n{year_start.year}: SMAP raw={len(smap_raw)}, "
        f"suction swath={len(suction_raw)}, suction gf={len(suction_gf)}, "
        f"frames={n_days}"
    )

    blank = _blank()

    # Layout: 2 rows × 2 cols; left col = small input panels, right col = large output
    fig = plt.figure(figsize=(16, 8), dpi=150)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1, 1.6],
        height_ratios=[1, 1],
        hspace=0.15,
        wspace=0.08,
        left=0.03,
        right=0.97,
        top=0.92,
        bottom=0.03,
    )

    ax_smap = fig.add_subplot(gs[0, 0])
    ax_swath = fig.add_subplot(gs[1, 0])
    ax_gf = fig.add_subplot(gs[:, 1])

    for ax in [ax_smap, ax_swath, ax_gf]:
        ax.set_axis_off()

    # Left top: SMAP raw
    im0 = ax_smap.imshow(
        blank,
        cmap="viridis_r",
        vmin=VWC_VMIN,
        vmax=VWC_VMAX,
        interpolation="nearest",
        aspect="equal",
    )
    ax_smap.set_title("SMAP L3 soil moisture (raw)", fontsize=9, pad=3)
    cb0 = fig.colorbar(im0, ax=ax_smap, fraction=0.03, pad=0.02, shrink=0.8)
    cb0.set_label(r"VWC (m$^3$/m$^3$)", fontsize=7)
    cb0.ax.tick_params(labelsize=6)

    # Left bottom: swath suction
    im1 = ax_swath.imshow(
        blank,
        cmap="RdBu_r",
        vmin=SUCTION_VMIN,
        vmax=SUCTION_VMAX,
        interpolation="nearest",
        aspect="equal",
    )
    ax_swath.set_title("Inferred suction (swath)", fontsize=9, pad=3)
    cb1 = fig.colorbar(im1, ax=ax_swath, fraction=0.03, pad=0.02, shrink=0.8)
    cb1.set_label(r"log$_{10}$ suction (cm H$_2$O)", fontsize=7)
    cb1.ax.tick_params(labelsize=6)

    # Right: gap-filled suction (full height)
    im2 = ax_gf.imshow(
        blank,
        cmap="RdBu_r",
        vmin=SUCTION_VMIN,
        vmax=SUCTION_VMAX,
        interpolation="nearest",
        aspect="equal",
    )
    ax_gf.set_title("Inferred suction (gap-filled)", fontsize=11, pad=3)
    cb2 = fig.colorbar(im2, ax=ax_gf, fraction=0.02, pad=0.01, shrink=0.6)
    cb2.set_label(r"log$_{10}$ suction (cm H$_2$O)", fontsize=8)
    cb2.ax.tick_params(labelsize=7)

    date_text = ax_gf.text(
        0.5,
        -0.03,
        "",
        transform=ax_gf.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
    )

    def update(frame_idx: int):
        d = all_dates[frame_idx]

        r0 = _load(smap_raw[d]) if d in smap_raw else blank
        r1 = _load(suction_raw[d]) if d in suction_raw else blank
        r2 = _load(suction_gf[d]) if d in suction_gf else blank

        im0.set_data(r0)
        im1.set_data(r1)
        im2.set_data(r2)
        date_text.set_text(d.strftime("%B %-d, %Y"))

        if (frame_idx + 1) % 60 == 0 or (frame_idx + 1) == n_days:
            print(f"  rendered {frame_idx + 1}/{n_days}", flush=True)

        return im0, im1, im2

    ani = FuncAnimation(fig, update, frames=n_days, interval=1000 // FPS, blit=True)
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
        description="Animate CONUS SMAP + suction comparison (per-year MP4s)",
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
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit to N frames (for layout testing)",
    )
    args = parser.parse_args()

    scan_start, scan_end = date(2015, 1, 1), date(2030, 12, 31)
    all_suction = _discover(SUCTION_DIR, _RE_SUCTION, scan_start, scan_end)
    all_suction_gf = _discover(SUCTION_GF_DIR, _RE_SUCTION, scan_start, scan_end)
    all_smap = _discover(SMAP_RAW_DIR, _RE_SMAP, scan_start, scan_end)

    if not all_suction:
        raise FileNotFoundError("No suction rasters found")

    first_date = min(min(all_suction), min(all_smap))
    last_date = max(max(all_suction), max(all_smap))

    start = datetime.strptime(args.start, "%Y%m%d").date() if args.start else first_date
    end = datetime.strptime(args.end, "%Y%m%d").date() if args.end else last_date

    print(f"Period: {start} to {end}")
    print(f"SMAP raw:       {len(all_smap)}")
    print(f"Suction swath:  {len(all_suction)}")
    print(f"Suction gf:     {len(all_suction_gf)}")

    out_dir = Path(args.out_dir)

    for year_start, year_end in _year_chunks(start, end):
        smap_raw = {d: p for d, p in all_smap.items() if year_start <= d <= year_end}
        suction_raw = {
            d: p for d, p in all_suction.items() if year_start <= d <= year_end
        }
        suction_gf = {
            d: p for d, p in all_suction_gf.items() if year_start <= d <= year_end
        }
        out_path = out_dir / f"conus_comparison_{year_start.year}.mp4"
        animate_year(
            year_start,
            year_end,
            smap_raw,
            suction_raw,
            suction_gf,
            out_path,
            max_frames=args.max_frames,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
