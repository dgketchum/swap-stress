"""
Temporal gap-filling for CONUS inference rasters.

Fills missing calendar days (SMAP L3 gaps) in a directory of daily
``suction_YYYYMMDD.tif`` rasters via pixel-wise linear interpolation in
log10-suction space between the nearest valid bracketing rasters.

Algorithm
---------
For each gap date G with predecessor P (last valid date ≤ G) and
successor S (first valid date ≥ G):

    weight = (G - P).days / (S - P).days
    output[px] = P[px] + weight * (S[px] - P[px])

- NODATA (-9999) propagates: any pixel NODATA in either bracket → NODATA output
- Boundary: no P → forward-fill from S; no S → back-fill from P

Usage
-----
    uv run python -m map.inference.gapfill_conus \\
        --source-dir /nas/soils/swapstress/inference/predictions/direct_rf_9km_global_pruned \\
        --output-dir /nas/soils/swapstress/inference/predictions/direct_rf_9km_global_pruned_gapfilled \\
        --start-date 20240101 --end-date 20241231
"""

from __future__ import annotations

import argparse
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio

SUCTION_FILENAME_RE = re.compile(r"^suction_(\d{8})\.tif$")
NODATA_VALUE: float = -9999.0
DEFAULT_SOURCE_DIR = (
    "/nas/soils/swapstress/inference/predictions/direct_rf_9km_global_pruned"
)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_source_rasters(
    source_dir: str | Path,
    start: date | None = None,
    end: date | None = None,
) -> dict[date, Path]:
    """Return {date: path} for all suction rasters in *source_dir*, filtered by range."""
    source_path = Path(source_dir).expanduser().resolve()
    result: dict[date, Path] = {}
    for path in sorted(source_path.glob("suction_*.tif")):
        m = SUCTION_FILENAME_RE.match(path.name)
        if not m:
            continue
        d = datetime.strptime(m.group(1), "%Y%m%d").date()
        if start and d < start:
            continue
        if end and d > end:
            continue
        result[d] = path
    return result


def find_gap_dates(
    source_dates: set[date],
    start: date,
    end: date,
) -> list[date]:
    """Return all calendar days in [start, end] that have no source raster."""
    gaps: list[date] = []
    current = start
    while current <= end:
        if current not in source_dates:
            gaps.append(current)
        current += timedelta(days=1)
    return gaps


# ---------------------------------------------------------------------------
# Raster I/O
# ---------------------------------------------------------------------------


def load_raster(path: Path) -> tuple[np.ndarray, dict]:
    """Read a suction raster; return (flat float32 array, rasterio profile)."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32, copy=False)
        profile = src.profile.copy()
    return data.reshape(-1), profile


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


def interpolate_gap(
    prev_arr: np.ndarray,
    next_arr: np.ndarray,
    weight: float,
) -> np.ndarray:
    """
    Linear interpolation between two flat log10-suction arrays.

    NODATA (-9999) in either bracket propagates to the output.
    """
    prev_nodata = prev_arr == NODATA_VALUE
    next_nodata = next_arr == NODATA_VALUE

    result = prev_arr + weight * (next_arr - prev_arr)

    # Propagate NODATA where either bracket is missing
    result[prev_nodata | next_nodata] = NODATA_VALUE
    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_gapfilled_raster(
    out_path: Path,
    data: np.ndarray,
    profile: dict,
    gap_date: date,
    prev_date: date | None,
    next_date: date | None,
    weight: float,
) -> None:
    """Write a gap-filled raster with metadata tags."""
    height = profile["height"]
    width = profile["width"]

    out_profile = profile.copy()
    out_profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=NODATA_VALUE,
        compress="zstd",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(data.reshape(height, width), 1)
        dst.set_band_description(1, "log10_suction_cm")
        dst.update_tags(
            gap_fill_method="linear_interp",
            prev_date=prev_date.strftime("%Y%m%d") if prev_date else "none",
            next_date=next_date.strftime("%Y%m%d") if next_date else "none",
            interp_weight=f"{weight:.6f}",
            is_gap_filled="true",
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_gapfill(
    source_dir: str,
    output_dir: str,
    start_date: str | None,
    end_date: str | None,
    overwrite: bool,
) -> None:
    """Main orchestration: discover gaps, interpolate, write outputs."""
    start = datetime.strptime(start_date, "%Y%m%d").date() if start_date else None
    end = datetime.strptime(end_date, "%Y%m%d").date() if end_date else None

    source_rasters = discover_source_rasters(source_dir, start, end)
    if not source_rasters:
        raise FileNotFoundError(f"No suction rasters found in {source_dir}")

    source_dates = set(source_rasters.keys())
    effective_start = start or min(source_dates)
    effective_end = end or max(source_dates)

    gap_dates = find_gap_dates(source_dates, effective_start, effective_end)
    if not gap_dates:
        print("No gap dates found — nothing to fill.")
        return

    sorted_source_dates = sorted(source_dates)
    output_path = Path(output_dir).expanduser().resolve()

    print(f"Source dir:  {Path(source_dir).expanduser().resolve()}")
    print(f"Output dir:  {output_path}")
    print(f"Source rasters found: {len(source_rasters)}")
    print(f"Gap dates to fill:    {len(gap_dates)}")

    # Cache: date → flat array (keep both brackets of last gap in memory)
    raster_cache: dict[date, np.ndarray] = {}
    ref_profile: dict | None = None

    def get_array(d: date) -> np.ndarray:
        if d not in raster_cache:
            arr, profile = load_raster(source_rasters[d])
            nonlocal ref_profile
            if ref_profile is None:
                ref_profile = profile
            raster_cache[d] = arr
        return raster_cache[d]

    for gap in gap_dates:
        out_name = f"suction_{gap.strftime('%Y%m%d')}.tif"
        out_file = output_path / out_name

        if out_file.exists() and not overwrite:
            print(f"SKIP {out_name} (exists)")
            continue

        # Find bracketing dates
        prev_date = max((d for d in sorted_source_dates if d <= gap), default=None)
        next_date = min((d for d in sorted_source_dates if d >= gap), default=None)

        if prev_date is None and next_date is None:
            print(f"SKIP {out_name} (no bracketing rasters)")
            continue

        if prev_date is None:
            # Forward-fill from next
            arr = get_array(next_date).copy()
            weight = 0.0
            used_prev, used_next = next_date, next_date
        elif next_date is None:
            # Back-fill from prev
            arr = get_array(prev_date).copy()
            weight = 1.0
            used_prev, used_next = prev_date, prev_date
        else:
            span = (next_date - prev_date).days
            weight = (gap - prev_date).days / span
            arr = interpolate_gap(get_array(prev_date), get_array(next_date), weight)
            used_prev, used_next = prev_date, next_date

        # Load profile from a neighbouring source raster if not yet cached
        if ref_profile is None:
            _, ref_profile = load_raster(source_rasters[next_date or prev_date])

        write_gapfilled_raster(
            out_path=out_file,
            data=arr,
            profile=ref_profile,
            gap_date=gap,
            prev_date=used_prev,
            next_date=used_next,
            weight=weight,
        )
        print(f"WROTE {out_name}  (prev={used_prev}, next={used_next}, w={weight:.4f})")

        # Evict cache entries that are no longer needed
        # Keep only prev/next for the current gap (they may be reused for the next gap)
        keep = {used_prev, used_next}
        stale = [k for k in raster_cache if k not in keep]
        for k in stale:
            del raster_cache[k]

    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gap-fill CONUS suction rasters by linear interpolation in log10 space",
    )
    parser.add_argument(
        "--source-dir",
        default=DEFAULT_SOURCE_DIR,
        help=f"Directory of source suction_YYYYMMDD.tif rasters (default: {DEFAULT_SOURCE_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for gap-filled rasters",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Inclusive start date YYYYMMDD",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Inclusive end date YYYYMMDD",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output rasters",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_gapfill(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
