"""
Temporal and spatial gap-filling for gridded inference rasters.

Fills all NODATA values (temporal gaps = days with no raster; spatial gaps =
SMAP swath NODATA within a raster) in a directory of daily
``<prefix>_YYYYMMDD.tif`` rasters via per-pixel linear interpolation along
the time axis.

Algorithm
---------
All source rasters are loaded and stacked into a (n_days, n_pixels) array
with NaN wherever a pixel has no valid observation (SMAP swath miss or
missing day).  For each pixel, ``np.interp`` is called along the time axis:

- Interpolates linearly between any two valid observations.
- Back-fills (flat) before the first valid observation.
- Forward-fills (flat) after the last valid observation.
- Pixels with zero valid observations across the full period (ocean /
  permanent mask) remain NODATA.

All output rasters are written (not just the temporal gap days), so the
output directory is a spatially and temporally complete daily series.

Usage
-----
    # Suction inference
    uv run python -m swapstress.inference.gapfill \\
        --config /home/dgketchum/code/swap-stress/configs/gapfill_9km_global_pruned.toml

    # SMAP L3 soil moisture
    uv run python -m swapstress.inference.gapfill \\
        --source-dir /nas/soils/smap/SPL3SMP_E/daily_tif \\
        --output-dir /nas/soils/smap/SPL3SMP_E/daily_tif_gapfilled \\
        --prefix smap_sm --band-description soil_moisture_m3m3 \\
        --start-date 20240101 --end-date 20241231
"""

from __future__ import annotations

import argparse
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio

NODATA_VALUE: float = -9999.0
DEFAULT_SOURCE_DIR = (
    "/nas/soils/swapstress/releases/global_pruned_refresh_20260520/inference"
)
DEFAULT_PREFIX = "suction"
DEFAULT_BAND_DESCRIPTION = "log10_suction_cm"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_source_rasters(
    source_dir: str | Path,
    prefix: str,
    start: date | None = None,
    end: date | None = None,
) -> dict[date, Path]:
    """Return {date: path} for all <prefix>_YYYYMMDD.tif rasters in source_dir."""
    source_path = Path(source_dir).expanduser().resolve()
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d{{8}})\.tif$")
    result: dict[date, Path] = {}
    for path in sorted(source_path.glob(f"{prefix}_*.tif")):
        m = pattern.match(path.name)
        if not m:
            continue
        d = datetime.strptime(m.group(1), "%Y%m%d").date()
        if start and d < start:
            continue
        if end and d > end:
            continue
        result[d] = path
    return result


# ---------------------------------------------------------------------------
# Raster I/O
# ---------------------------------------------------------------------------


def load_raster(path: Path) -> tuple[np.ndarray, dict]:
    """Read a raster; return (flat float32 array, rasterio profile)."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32, copy=False)
        profile = src.profile.copy()
    return data.reshape(-1), profile


def write_raster(
    out_path: Path,
    data: np.ndarray,
    profile: dict,
    out_date: date,
    is_gap_filled: bool,
    band_description: str,
) -> None:
    """Write an output raster with metadata tags."""
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
        dst.set_band_description(1, band_description)
        dst.update_tags(
            gap_fill_method="time_axis_interp",
            date=out_date.strftime("%Y%m%d"),
            is_gap_filled=str(is_gap_filled).lower(),
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
    prefix: str = DEFAULT_PREFIX,
    band_description: str = DEFAULT_BAND_DESCRIPTION,
    config_dict: dict | None = None,
) -> None:
    """Stack all source rasters, interpolate all pixels for all days, write output."""
    start = datetime.strptime(start_date, "%Y%m%d").date() if start_date else None
    end = datetime.strptime(end_date, "%Y%m%d").date() if end_date else None

    source_rasters = discover_source_rasters(source_dir, prefix, start, end)
    if not source_rasters:
        raise FileNotFoundError(
            f"No {prefix}_YYYYMMDD.tif rasters found in {source_dir}"
        )

    source_dates = set(source_rasters.keys())
    effective_start = start or min(source_dates)
    effective_end = end or max(source_dates)

    n_days = (effective_end - effective_start).days + 1
    all_dates = [effective_start + timedelta(days=i) for i in range(n_days)]
    day_idx = np.arange(n_days, dtype=np.float32)

    output_path = Path(output_dir).expanduser().resolve()
    print(f"Prefix:      {prefix}")
    print(f"Source dir:  {Path(source_dir).expanduser().resolve()}")
    print(f"Output dir:  {output_path}")
    print(f"Source rasters: {len(source_rasters)}")
    print(f"Output days:    {n_days}")

    def out_name(d: date) -> str:
        return f"{prefix}_{d.strftime('%Y%m%d')}.tif"

    days_to_write = [
        d for d in all_dates if overwrite or not (output_path / out_name(d)).exists()
    ]
    if not days_to_write:
        print("All output files already exist — nothing to write.")
        return
    print(f"Days to write:  {len(days_to_write)}", flush=True)

    sorted_source_dates = sorted(source_dates)

    # Load all source rasters
    print("Loading source rasters...", flush=True)
    ref_profile: dict | None = None
    n_pixels: int | None = None
    source_arrays: dict[date, np.ndarray] = {}
    for i, d in enumerate(sorted_source_dates):
        arr, profile = load_raster(source_rasters[d])
        if ref_profile is None:
            ref_profile = profile
            n_pixels = arr.size
        source_arrays[d] = arr
        if (i + 1) % 50 == 0 or (i + 1) == len(sorted_source_dates):
            print(f"  loaded {i + 1}/{len(sorted_source_dates)}", flush=True)

    # Assemble (n_days, n_pixels) stack — NaN for all NODATA (temporal + spatial)
    print("Assembling time stack...", flush=True)
    stack = np.full((n_days, n_pixels), np.nan, dtype=np.float32)
    for d, arr in source_arrays.items():
        t = (d - effective_start).days
        row = arr.copy()
        row[row == NODATA_VALUE] = np.nan  # -9999 → NaN; NaN nodata passes through
        stack[t] = row

    # Per-pixel interpolation along time axis for all output days
    write_t_indices = np.array(
        [(d - effective_start).days for d in days_to_write], dtype=np.float32
    )
    n_write = len(days_to_write)

    print(f"Interpolating {n_pixels:,} pixels × {n_write} days...", flush=True)

    CHUNK = 10_000
    out_stack = np.full((n_write, n_pixels), NODATA_VALUE, dtype=np.float32)

    for col_start in range(0, n_pixels, CHUNK):
        col_end = min(col_start + CHUNK, n_pixels)
        chunk = stack[:, col_start:col_end]

        for local_px in range(col_end - col_start):
            col = chunk[:, local_px]
            valid_mask = ~np.isnan(col)
            if not valid_mask.any():
                continue  # permanent NODATA (ocean etc.) — stays NODATA
            xp = day_idx[valid_mask]
            fp = col[valid_mask]
            out_stack[:, col_start + local_px] = np.interp(write_t_indices, xp, fp)

        if (col_start // CHUNK + 1) % 5 == 0 or col_end == n_pixels:
            print(f"  pixels {col_end:,}/{n_pixels:,}", flush=True)

    # Write all output rasters
    print("Writing output rasters...", flush=True)
    for i, d in enumerate(days_to_write):
        out_file = output_path / out_name(d)
        arr = out_stack[i]
        n_valid = (arr != NODATA_VALUE).sum()
        write_raster(
            out_file, arr, ref_profile, d, d not in source_dates, band_description
        )
        print(f"  WROTE {out_name(d)}  valid_px={n_valid:,}")

    # Write provenance artifact
    if config_dict is not None:
        from swapstress.config import write_provenance

        upstream_prov = Path(source_dir).expanduser().resolve() / "provenance.json"
        prov_path = write_provenance(
            output_dir=str(output_path),
            config=config_dict,
            run_type="gapfill",
            extras={
                "inputs": {
                    "n_source_rasters": len(source_rasters),
                },
                "outputs": {
                    "n_days_written": len(days_to_write),
                    "date_range": f"{effective_start} to {effective_end}",
                },
                "upstream": {
                    "source_provenance": str(upstream_prov)
                    if upstream_prov.exists()
                    else None,
                },
            },
        )
        print(f"Saved provenance to {prov_path}")

    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    from swapstress.cli import add_common_args

    parser = argparse.ArgumentParser(
        prog="swapstress-gapfill",
        description="Stage 06: gap-fill an aligned raster series along the time axis",
    )
    add_common_args(parser)
    parser.add_argument(
        "--source-dir",
        default=None,
        help=f"Directory of source <prefix>_YYYYMMDD.tif rasters (default: {DEFAULT_SOURCE_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for gap-filled rasters",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help=f"Filename prefix before _YYYYMMDD.tif (default: {DEFAULT_PREFIX})",
    )
    parser.add_argument(
        "--band-description",
        default=None,
        help=f"Band description tag written to output rasters (default: {DEFAULT_BAND_DESCRIPTION})",
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
        default=None,
        help="Overwrite existing output rasters",
    )
    return parser


def main(argv=None) -> None:
    from swapstress.cli import report_paths, resolve

    config = resolve(build_parser(), argv, required=["output_dir"])

    if config["dry_run"]:
        report_paths(
            "06 gapfill",
            {"predictions": config.get("source_dir", DEFAULT_SOURCE_DIR)},
            {"gap-filled": config["output_dir"]},
        )
        return

    run_gapfill(
        source_dir=config.get("source_dir", DEFAULT_SOURCE_DIR),
        output_dir=config["output_dir"],
        start_date=config.get("start_date"),
        end_date=config.get("end_date"),
        overwrite=config.get("overwrite", False),
        prefix=config.get("prefix", DEFAULT_PREFIX),
        band_description=config.get("band_description", DEFAULT_BAND_DESCRIPTION),
        config_dict=config,
    )


if __name__ == "__main__":
    main()
