"""
Download SMAP Enhanced L3 9 km soil moisture and convert to daily CONUS GeoTIFFs.

Downloads SPL3SMP_E (v005/v006) HDF5 files from NASA Earthdata via earthaccess,
extracts the AM descending-pass soil_moisture field, subsets to CONUS on the
native 9 km EASE-Grid2 (EPSG:6933), and writes one GeoTIFF per day.

HDF5 path: /Soil_Moisture_Retrieval_Data_AM/soil_moisture
Grid: EASE-Grid2 M09 — 3856 cols x 1624 rows, 9008.055 m cell size
Fill value: -9999.0

Usage:
    python -m map.data.smap_download \
        --output-dir /nas/soils/smap/SPL3SMP_E \
        --start 2015-04-01 --end 2026-02-15

    # Download only (skip GeoTIFF conversion)
    python -m map.data.smap_download \
        --output-dir /nas/soils/smap/SPL3SMP_E \
        --start 2015-04-01 --end 2026-02-15 \
        --download-only
"""

import argparse
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

# ---------------------------------------------------------------------------
# EASE-Grid2 M09 parameters (Brodzik et al., 2012)
# ---------------------------------------------------------------------------
MAP_SCALE = 9008.055210146  # metres per pixel
GLOBAL_COLS = 3856
GLOBAL_ROWS = 1624
S0 = (GLOBAL_COLS - 1) / 2.0  # 1927.5  column of prime meridian
R0 = (GLOBAL_ROWS - 1) / 2.0  # 811.5   row of equator

# Upper-left pixel edge of the full global grid
_X_UL = -(S0 + 0.5) * MAP_SCALE
_Y_UL = (R0 + 0.5) * MAP_SCALE

EASE2_CRS = CRS.from_epsg(6933)

# HDF5 dataset path for AM soil moisture
_SM_AM_PATH = "Soil_Moisture_Retrieval_Data_AM/soil_moisture"
_SM_PM_PATH = "Soil_Moisture_Retrieval_Data_PM/soil_moisture_pm"
_SM_FILL = -9999.0

# CONUS bounding box (WGS84)
_CONUS_LON_MIN, _CONUS_LON_MAX = -125.0, -66.0
_CONUS_LAT_MIN, _CONUS_LAT_MAX = 24.0, 50.0


def _lonlat_to_ease2_colrow(lon: float, lat: float) -> Tuple[float, float]:
    """Convert WGS84 lon/lat to fractional EASE-Grid2 M09 col/row."""
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
    x, y = transformer.transform(lon, lat)
    col = (x - _X_UL) / MAP_SCALE
    row = (_Y_UL - y) / MAP_SCALE
    return col, row


def _conus_slice() -> Tuple[slice, slice]:
    """Row and column slices for the CONUS bounding box on the M09 grid."""
    c0, r0 = _lonlat_to_ease2_colrow(_CONUS_LON_MIN, _CONUS_LAT_MAX)
    c1, r1 = _lonlat_to_ease2_colrow(_CONUS_LON_MAX, _CONUS_LAT_MIN)
    row_start = max(0, int(np.floor(r0)))
    row_end = min(GLOBAL_ROWS, int(np.ceil(r1)) + 1)
    col_start = max(0, int(np.floor(c0)))
    col_end = min(GLOBAL_COLS, int(np.ceil(c1)) + 1)
    return slice(row_start, row_end), slice(col_start, col_end)


def _conus_transform(row_slice: slice, col_slice: slice) -> Affine:
    """Build rasterio Affine for the CONUS subset."""
    x_origin = _X_UL + col_slice.start * MAP_SCALE
    y_origin = _Y_UL - row_slice.start * MAP_SCALE
    return Affine(MAP_SCALE, 0, x_origin, 0, -MAP_SCALE, y_origin)


def _parse_date_from_filename(filename: str) -> Optional[str]:
    """Extract YYYYMMDD from SMAP filename like SMAP_L3_SM_P_E_20150401_..."""
    m = re.search(r"_(\d{8})_", filename)
    if m:
        return m.group(1)
    return None


def download_smap(
    output_dir: str,
    start: str,
    end: str,
) -> list:
    """Download SPL3SMP_E HDF5 files from NASA Earthdata.

    Tries v006 first (covers 2015-04-01 to present with latest processing),
    falls back to v005 for any gaps.

    Parameters
    ----------
    output_dir : str
        Directory to store downloaded HDF5 files.
    start : str
        Start date (YYYY-MM-DD).
    end : str
        End date (YYYY-MM-DD).

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
        short_name="SPL3SMP_E",
        version="006",
        temporal=(start, end),
        bounding_box=(_CONUS_LON_MIN, _CONUS_LAT_MIN, _CONUS_LON_MAX, _CONUS_LAT_MAX),
    )

    if not results:
        print("No v006 results, trying v005...")
        results = earthaccess.search_data(
            short_name="SPL3SMP_E",
            version="005",
            temporal=(start, end),
            bounding_box=(
                _CONUS_LON_MIN,
                _CONUS_LAT_MIN,
                _CONUS_LON_MAX,
                _CONUS_LAT_MAX,
            ),
        )

    print(f"Found {len(results)} granules for {start} to {end}")

    if not results:
        return []

    downloaded = earthaccess.download(results, hdf_dir)
    print(f"Downloaded {len(downloaded)} files to {hdf_dir}")
    return [str(p) for p in downloaded]


def convert_hdf5_to_geotiff(
    hdf5_path: str,
    tif_dir: str,
    row_slice: slice,
    col_slice: slice,
    transform: Affine,
    overwrite: bool = False,
) -> Optional[str]:
    """Extract AM soil moisture from HDF5 and write CONUS GeoTIFF.

    Parameters
    ----------
    hdf5_path : str
        Path to SMAP HDF5 file.
    tif_dir : str
        Output directory for GeoTIFFs.
    row_slice, col_slice : slice
        CONUS subset indices on the global grid.
    transform : Affine
        Rasterio affine for the subset.
    overwrite : bool
        If False, skip files that already exist.

    Returns
    -------
    str or None
        Path to output GeoTIFF, or None if skipped/failed.
    """
    fname = os.path.basename(hdf5_path)
    date_str = _parse_date_from_filename(fname)
    if date_str is None:
        print(f"  Could not parse date from {fname}, skipping")
        return None

    out_name = f"smap_sm_{date_str}.tif"
    out_path = os.path.join(tif_dir, out_name)

    if not overwrite and os.path.exists(out_path):
        return out_path

    try:
        with h5py.File(hdf5_path, "r") as f:
            if _SM_AM_PATH not in f:
                print(f"  {_SM_AM_PATH} not in {fname}, skipping")
                return None
            sm = f[_SM_AM_PATH][row_slice, col_slice].astype(np.float32)
    except Exception as e:
        print(f"  Error reading {fname}: {e}")
        return None

    # Mask fill values
    sm[sm == _SM_FILL] = np.nan
    # Mask physically invalid values
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
        dst.set_band_description(1, "soil_moisture_am")

    return out_path


def build_index(tif_dir: str, index_path: str) -> None:
    """Write a CSV index mapping date to GeoTIFF path.

    Parameters
    ----------
    tif_dir : str
        Directory containing smap_sm_YYYYMMDD.tif files.
    index_path : str
        Output CSV path.
    """
    import pandas as pd

    records = []
    for fname in sorted(os.listdir(tif_dir)):
        if not fname.endswith(".tif"):
            continue
        date_str = _parse_date_from_filename(fname)
        if date_str is None:
            continue
        records.append(
            {
                "date": datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d"),
                "file": os.path.join(tif_dir, fname),
            }
        )

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    df.to_csv(index_path, index=False)
    print(f"Index: {len(df)} files written to {index_path}")


def run(
    output_dir: str,
    start: str,
    end: str,
    download_only: bool = False,
    overwrite: bool = False,
) -> None:
    """Full pipeline: download HDF5 -> convert to CONUS GeoTIFF -> build index."""
    output_dir = str(Path(output_dir).expanduser())
    tif_dir = os.path.join(output_dir, "daily_tif")
    os.makedirs(tif_dir, exist_ok=True)

    # Download
    hdf5_files = download_smap(output_dir, start, end)

    if download_only:
        print("Download-only mode, stopping.")
        return

    # Also pick up any previously-downloaded files
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

    # Compute CONUS grid parameters once
    row_sl, col_sl = _conus_slice()
    transform = _conus_transform(row_sl, col_sl)
    print(
        f"CONUS subset: rows {row_sl.start}:{row_sl.stop}, "
        f"cols {col_sl.start}:{col_sl.stop} "
        f"({row_sl.stop - row_sl.start} x {col_sl.stop - col_sl.start} pixels)"
    )

    # Convert
    converted = 0
    for hdf_path in all_files:
        result = convert_hdf5_to_geotiff(
            hdf_path, tif_dir, row_sl, col_sl, transform, overwrite=overwrite
        )
        if result:
            converted += 1

    print(f"Converted {converted}/{len(all_files)} files to {tif_dir}")

    # Build index
    index_path = os.path.join(output_dir, "smap_daily_index.csv")
    build_index(tif_dir, index_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download SMAP SPL3SMP_E and convert to CONUS GeoTIFFs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Root output directory (hdf5/ and daily_tif/ subdirs created).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2015-04-01",
        help="Start date YYYY-MM-DD (default: 2015-04-01).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date YYYY-MM-DD (default: today).",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download HDF5 files but skip GeoTIFF conversion.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing GeoTIFFs.",
    )
    args = parser.parse_args()

    run(
        output_dir=args.output_dir,
        start=args.start,
        end=args.end,
        download_only=args.download_only,
        overwrite=args.overwrite,
    )

# ========================= EOF ====================================================================
