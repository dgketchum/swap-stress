"""
Download SMAP L4 (SPL4SMGP v008) 9 km soil moisture and convert to daily CONUS GeoTIFFs.

Downloads SPL4SMGP 3-hourly HDF5 files from NASA Earthdata via earthaccess, selects
the ~06Z time step (closest to SMAP's AM descending pass over CONUS), extracts surface
soil moisture, subsets to CONUS on the native 9 km EASE-Grid2, and writes one GeoTIFF
per day.

HDF5 path: /Geophysical_Data/sm_surface
Grid: EASE-Grid2 M09 — 3856 cols x 1624 rows, 9008.055 m cell size (same as L3)
Fill value: -9999.0

Usage:
    python -m research.sensors.smap_l4_download \
        --output-dir /nas/soils/smap/SPL4SMGP \
        --start 2022-01-01 --end 2023-12-31
"""

import argparse
import os
import re
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from swapstress.features.smap_download import (
    EASE2_CRS,
    _conus_slice,
    _conus_transform,
)

_SM_SURFACE_PATH = "Geophysical_Data/sm_surface"
_SM_FILL = -9999.0


def _parse_l4_datetime(filename: str) -> Optional[str]:
    """Extract YYYYMMDDTHHMMSS from L4 filename like SMAP_L4_SM_gph_20220615T013000_..."""
    m = re.search(r"_(\d{8}T\d{6})_", filename)
    if m:
        return m.group(1)
    return None


def _is_am_pass(filename: str) -> bool:
    """Check if this L4 file is the 10:30 UTC step (closest to SMAP AM over CONUS).

    SMAP descending (AM) pass crosses equator at ~6 AM local.
    CONUS spans UTC-5 to UTC-8, so 6 AM local = 11-14 UTC.
    The 10:30 UTC L4 step is the closest match.
    """
    dt_str = _parse_l4_datetime(filename)
    if dt_str is None:
        return False
    hour = int(dt_str[9:11])
    return hour == 10


def download_smap_l4(output_dir: str, start: str, end: str) -> list:
    """Download SPL4SMGP v008 HDF5 files from NASA Earthdata.

    Parameters
    ----------
    output_dir : str
        Root directory. HDF5 files go in ``output_dir/hdf5/``.
    start, end : str
        Date range (YYYY-MM-DD).

    Returns
    -------
    list
        Paths to downloaded HDF5 files.
    """
    import earthaccess

    hdf_dir = os.path.join(output_dir, "hdf5")
    os.makedirs(hdf_dir, exist_ok=True)

    earthaccess.login()

    results = earthaccess.search_data(
        short_name="SPL4SMGP",
        version="008",
        temporal=(start, end),
    )
    print(f"Found {len(results)} SPL4SMGP granules for {start} to {end}")

    if not results:
        return []

    # Filter to only download 10:30 UTC files (closest to SMAP AM pass over CONUS)
    filtered = []
    for r in results:
        native_id = r["meta"]["native-id"]
        if _is_am_pass(native_id):
            filtered.append(r)

    print(f"Filtered to {len(filtered)} files (10:30 UTC AM-pass time steps)")

    if not filtered:
        print("No 06Z files found, downloading all and filtering during conversion")
        filtered = results

    downloaded = earthaccess.download(filtered, hdf_dir)
    print(f"Downloaded {len(downloaded)} files to {hdf_dir}")
    return [str(p) for p in downloaded]


def convert_l4_hdf5_to_geotiff(
    hdf5_path: str,
    tif_dir: str,
    row_slice,
    col_slice,
    transform,
    overwrite: bool = False,
) -> Optional[str]:
    """Extract surface SM from L4 HDF5 and write CONUS GeoTIFF."""
    import rasterio

    fname = os.path.basename(hdf5_path)
    dt_str = _parse_l4_datetime(fname)
    if dt_str is None:
        return None

    date_str = dt_str[:8]
    out_name = f"smap_l4_sm_{date_str}.tif"
    out_path = os.path.join(tif_dir, out_name)

    if not overwrite and os.path.exists(out_path):
        return out_path

    try:
        with h5py.File(hdf5_path, "r") as f:
            if _SM_SURFACE_PATH not in f:
                print(f"  {_SM_SURFACE_PATH} not in {fname}, skipping")
                return None
            sm = f[_SM_SURFACE_PATH][row_slice, col_slice].astype(np.float32)
    except Exception as e:
        print(f"  Error reading {fname}: {e}")
        return None

    sm[sm == _SM_FILL] = np.nan
    sm[(sm < 0) | (sm > 1.0)] = np.nan

    height, width = sm.shape
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=EASE2_CRS,
        transform=transform,
        nodata=np.nan,
        compress="zstd",
    ) as dst:
        dst.write(sm, 1)
        dst.set_band_description(1, "sm_surface")

    return out_path


def run(output_dir: str, start: str, end: str, overwrite: bool = False) -> None:
    """Full pipeline: download L4 HDF5 -> convert to CONUS GeoTIFF."""
    output_dir = str(Path(output_dir).expanduser())
    tif_dir = os.path.join(output_dir, "daily_tif")
    os.makedirs(tif_dir, exist_ok=True)

    hdf5_files = download_smap_l4(output_dir, start, end)

    hdf_dir = os.path.join(output_dir, "hdf5")
    if os.path.isdir(hdf_dir):
        existing = [
            os.path.join(hdf_dir, f) for f in os.listdir(hdf_dir) if f.endswith(".h5")
        ]
        all_files = sorted(set(hdf5_files + existing))
    else:
        all_files = sorted(hdf5_files)

    if not all_files:
        print("No HDF5 files to convert.")
        return

    row_sl, col_sl = _conus_slice()
    transform = _conus_transform(row_sl, col_sl)
    print(
        f"CONUS subset: rows {row_sl.start}:{row_sl.stop}, "
        f"cols {col_sl.start}:{col_sl.stop}"
    )

    converted = 0
    for hdf_path in all_files:
        result = convert_l4_hdf5_to_geotiff(
            hdf_path, tif_dir, row_sl, col_sl, transform, overwrite
        )
        if result:
            converted += 1

    print(f"Converted {converted}/{len(all_files)} files to {tif_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download SMAP L4 SPL4SMGP and convert to CONUS GeoTIFFs",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--start", type=str, default="2022-01-01")
    parser.add_argument("--end", type=str, default="2023-12-31")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run(args.output_dir, args.start, args.end, args.overwrite)
