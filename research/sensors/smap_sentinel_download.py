"""
Download SMAP/Sentinel-1 L2 3 km soil moisture (SPL2SMAP_S v003) HDF5 granules.

Downloads SPL2SMAP_S granules from NASA Earthdata via earthaccess. Unlike the
gridded L3 product, SPL2SMAP_S is a swath-level (L2) product with scene-based
HDF5 files. Each granule covers one Sentinel-1 scene footprint.

HDF5 group: /Soil_Moisture_Retrieval_Data_3km/
Key fields: soil_moisture_3km, EASE_row_index_3km, EASE_column_index_3km,
            retrieval_qual_flag_3km, latitude_3km, longitude_3km
Grid: EASE-Grid2 M03 — 11568 cols x 4872 rows, 3002.685 m cell size
Fill value: -9999.0

Usage:
    python -m research.sensors.smap_sentinel_download \
        --output-dir /nas/soils/smap/SPL2SMAP_S \
        --start 2022-01-01 --end 2023-12-31
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

from pyproj import Transformer

# ---------------------------------------------------------------------------
# EASE-Grid2 M03 parameters (Brodzik et al., 2012)
# ---------------------------------------------------------------------------
M03_SCALE = 3002.685070049  # metres per pixel
M03_COLS = 11568
M03_ROWS = 4872
_M03_S0 = (M03_COLS - 1) / 2.0  # 5783.5  column of prime meridian
_M03_R0 = (M03_ROWS - 1) / 2.0  # 2435.5  row of equator

_M03_X_UL = -(_M03_S0 + 0.5) * M03_SCALE
_M03_Y_UL = (_M03_R0 + 0.5) * M03_SCALE

# CONUS bounding box (WGS84) — same as L3 pipeline
CONUS_LON_MIN, CONUS_LON_MAX = -125.0, -66.0
CONUS_LAT_MIN, CONUS_LAT_MAX = 24.0, 50.0

_TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)


def lonlat_to_m03_colrow(lon: float, lat: float) -> Tuple[int, int]:
    """Convert WGS84 lon/lat to integer EASE-Grid2 M03 (col, row).

    Returns 0-based indices matching SPL2SMAP_S EASE_row_index_3km /
    EASE_column_index_3km fields.
    """
    x, y = _TRANSFORMER.transform(lon, lat)
    col = int(round((x - _M03_X_UL) / M03_SCALE - 0.5))
    row = int(round((_M03_Y_UL - y) / M03_SCALE - 0.5))
    return col, row


def download_smap_sentinel(output_dir: str, start: str, end: str) -> list:
    """Download SPL2SMAP_S v003 HDF5 granules from NASA Earthdata.

    Parameters
    ----------
    output_dir : str
        Root output directory. HDF5 files go in ``output_dir/hdf5/``.
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
        short_name="SPL2SMAP_S",
        version="003",
        temporal=(start, end),
        bounding_box=(CONUS_LON_MIN, CONUS_LAT_MIN, CONUS_LON_MAX, CONUS_LAT_MAX),
    )

    print(f"Found {len(results)} SPL2SMAP_S granules for {start} to {end}")

    if not results:
        return []

    downloaded = earthaccess.download(results, hdf_dir)
    print(f"Downloaded {len(downloaded)} files to {hdf_dir}")
    return [str(p) for p in downloaded]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download SMAP/Sentinel-1 SPL2SMAP_S HDF5 granules",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Root output directory (hdf5/ subdir created).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2022-01-01",
        help="Start date YYYY-MM-DD (default: 2022-01-01).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2023-12-31",
        help="End date YYYY-MM-DD (default: 2023-12-31).",
    )
    args = parser.parse_args()

    output_dir = str(Path(args.output_dir).expanduser())
    download_smap_sentinel(output_dir, args.start, args.end)
