"""
Download SMAP soil moisture products and convert to GeoTIFFs on EASE-Grid2 M09.

Supports two products:
  - SPL3SMP_E (L3 Enhanced): daily AM soil moisture at 9 km
  - SPL4SMGP (L4 Global): 3-hourly surface + root-zone SM at 9 km,
    aggregated to daily-mean GeoTIFFs with two bands

HDF5 paths:
  L3: /Soil_Moisture_Retrieval_Data_AM/soil_moisture
  L4: /Geophysical_Data/sm_surface, /Geophysical_Data/sm_rootzone

Grid: EASE-Grid2 M09 — 3856 cols x 1624 rows, 9008.055 m cell size
Fill value: -9999.0

Usage:
    # L3, CONUS (original workflow)
    python -m swapstress.features.smap_download \
        --product L3 --grid-scope conus \
        --output-dir /nas/soils/smap/SPL3SMP_E \
        --start 2015-04-01 --end 2026-02-15

    # L4, global, full period of record
    python -m swapstress.features.smap_download \
        --product L4 --grid-scope global \
        --output-dir /nas/soils/smap/SPL4SMGP \
        --start 2015-03-31 --end 2026-06-01

    # Download only (skip GeoTIFF conversion)
    python -m swapstress.features.smap_download \
        --product L4 --grid-scope global \
        --output-dir /nas/soils/smap/SPL4SMGP \
        --start 2015-03-31 --end 2026-06-01 \
        --download-only
"""

import argparse
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

# ---------------------------------------------------------------------------
# Product configurations
# ---------------------------------------------------------------------------
_PRODUCTS = {
    "L3": {
        "short_name": "SPL3SMP_E",
        "versions": ["006", "005"],
        "hdf5_datasets": {
            "soil_moisture_am": "Soil_Moisture_Retrieval_Data_AM/soil_moisture",
        },
        "fill_value": -9999.0,
        "global_granules": False,
    },
    "L4": {
        "short_name": "SPL4SMGP",
        "versions": ["008", "007"],
        "hdf5_datasets": {
            "sm_surface": "Geophysical_Data/sm_surface",
            "sm_rootzone": "Geophysical_Data/sm_rootzone",
        },
        "fill_value": -9999.0,
        "global_granules": True,
    },
}

# Legacy constants kept for backward compatibility with external imports
_SM_AM_PATH = "Soil_Moisture_Retrieval_Data_AM/soil_moisture"
_SM_PM_PATH = "Soil_Moisture_Retrieval_Data_PM/soil_moisture_pm"
_SM_FILL = -9999.0

# CONUS bounding box (WGS84)
_CONUS_LON_MIN, _CONUS_LON_MAX = -125.0, -66.0
_CONUS_LAT_MIN, _CONUS_LAT_MAX = 24.0, 50.0
VALID_GRID_SCOPES = ("conus", "global")


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


def global_slice() -> Tuple[slice, slice]:
    """Row and column slices for the full global EASE-Grid2 M09 domain."""
    return slice(0, GLOBAL_ROWS), slice(0, GLOBAL_COLS)


def ease2_transform(row_slice: slice, col_slice: slice) -> Affine:
    """Build a rasterio Affine for any EASE-Grid2 M09 window."""
    x_origin = _X_UL + col_slice.start * MAP_SCALE
    y_origin = _Y_UL - row_slice.start * MAP_SCALE
    return Affine(MAP_SCALE, 0, x_origin, 0, -MAP_SCALE, y_origin)


def resolve_ease2_grid(grid_scope: str) -> Tuple[slice, slice, Affine]:
    """Return the row slice, col slice, and affine for a named grid scope."""
    if grid_scope == "conus":
        row_slice, col_slice = _conus_slice()
    elif grid_scope == "global":
        row_slice, col_slice = global_slice()
    else:
        raise ValueError(
            f"grid_scope must be one of {VALID_GRID_SCOPES}, got {grid_scope!r}"
        )

    return row_slice, col_slice, ease2_transform(row_slice, col_slice)


def _conus_transform(row_slice: slice, col_slice: slice) -> Affine:
    """Build rasterio Affine for the CONUS subset."""
    return ease2_transform(row_slice, col_slice)


def _parse_date_from_filename(filename: str) -> Optional[str]:
    """Extract YYYYMMDD from SMAP filename.

    Handles both L3 (SMAP_L3_SM_P_E_20150401_...) and
    L4 (SMAP_L4_SM_gph_20150401T013000_...) patterns.
    """
    m = re.search(r"_(\d{8})[T_.]", filename)
    if m:
        return m.group(1)
    return None


def download_smap(
    output_dir: str,
    start: str,
    end: str,
    product: str = "L3",
    grid_scope: str = "conus",
) -> list:
    """Download SMAP HDF5 files from NASA Earthdata.

    Parameters
    ----------
    output_dir : str
        Directory to store downloaded HDF5 files.
    start : str
        Start date (YYYY-MM-DD).
    end : str
        End date (YYYY-MM-DD).
    product : str
        "L3" for SPL3SMP_E or "L4" for SPL4SMGP.
    grid_scope : str
        "conus" to restrict search to CONUS bbox, "global" for no bbox filter.
        Ignored for L4 (granules are always global).

    Returns
    -------
    list
        Paths to downloaded HDF5 files.
    """
    import earthaccess

    cfg = _PRODUCTS[product]
    hdf_dir = os.path.join(output_dir, "hdf5")
    os.makedirs(hdf_dir, exist_ok=True)

    earthaccess.login()

    search_kwargs: dict = {}
    if not cfg["global_granules"] and grid_scope == "conus":
        search_kwargs["bounding_box"] = (
            _CONUS_LON_MIN,
            _CONUS_LAT_MIN,
            _CONUS_LON_MAX,
            _CONUS_LAT_MAX,
        )

    results = []
    for version in cfg["versions"]:
        results = earthaccess.search_data(
            short_name=cfg["short_name"],
            version=version,
            temporal=(start, end),
            **search_kwargs,
        )
        if results:
            print(f"Found {len(results)} {cfg['short_name']} v{version} granules")
            break
        print(f"No {cfg['short_name']} v{version} results, trying next version...")

    if not results:
        print(f"No granules found for {product} from {start} to {end}")
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
    """Extract AM soil moisture from L3 HDF5 and write a single-band GeoTIFF.

    Parameters
    ----------
    hdf5_path : str
        Path to SMAP L3 HDF5 file.
    tif_dir : str
        Output directory for GeoTIFFs.
    row_slice, col_slice : slice
        Subset indices on the global grid.
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


def _convert_one_l4_day(
    date_str: str,
    hdf_paths: List[str],
    tif_dir: str,
    row_slice: slice,
    col_slice: slice,
    transform: Affine,
    overwrite: bool = False,
) -> bool:
    """Convert a single day of 3-hourly L4 granules to one 2-band GeoTIFF."""
    out_path = os.path.join(tif_dir, f"smap_l4_{date_str}.tif")
    if not overwrite and os.path.exists(out_path):
        return True

    cfg = _PRODUCTS["L4"]
    fill = cfg["fill_value"]
    ds_paths = cfg["hdf5_datasets"]
    band_names = list(ds_paths.keys())
    n_bands = len(band_names)
    height = row_slice.stop - row_slice.start
    width = col_slice.stop - col_slice.start

    band_accum: Dict[str, list] = {name: [] for name in band_names}

    for hdf_path in sorted(hdf_paths):
        try:
            with h5py.File(hdf_path, "r") as f:
                for name, ds_path in ds_paths.items():
                    if ds_path not in f:
                        continue
                    arr = f[ds_path][row_slice, col_slice].astype(np.float32)
                    arr[arr == fill] = np.nan
                    arr[(arr < 0) | (arr > 1.0)] = np.nan
                    band_accum[name].append(arr)
        except Exception as e:
            print(f"  Error reading {os.path.basename(hdf_path)}: {e}")

    bands: Dict[str, np.ndarray] = {}
    for name in band_names:
        if band_accum[name]:
            with np.errstate(all="ignore"):
                bands[name] = np.nanmean(np.stack(band_accum[name], axis=0), axis=0)

    if not bands:
        return False

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=n_bands,
        dtype="float32",
        crs=EASE2_CRS,
        transform=transform,
        nodata=np.nan,
        compress="zstd",
    ) as dst:
        for i, name in enumerate(band_names, 1):
            if name in bands:
                dst.write(bands[name], i)
            else:
                dst.write(np.full((height, width), np.nan, dtype=np.float32), i)
            dst.set_band_description(i, name)

    return True


def convert_l4_daily(
    hdf5_files: List[str],
    tif_dir: str,
    row_slice: slice,
    col_slice: slice,
    transform: Affine,
    overwrite: bool = False,
    n_jobs: int = 1,
) -> int:
    """Aggregate 3-hourly L4 HDF5 granules to daily-mean 2-band GeoTIFFs.

    Bands: (1) sm_surface [0-5 cm], (2) sm_rootzone [0-100 cm].

    Parameters
    ----------
    hdf5_files : list of str
        All L4 HDF5 file paths.
    tif_dir : str
        Output directory for daily GeoTIFFs.
    row_slice, col_slice : slice
        Subset indices on the global grid.
    transform : Affine
        Rasterio affine for the subset.
    overwrite : bool
        If False, skip dates with existing GeoTIFFs.
    n_jobs : int
        Number of parallel workers (1 = sequential).

    Returns
    -------
    int
        Number of daily GeoTIFFs written.
    """
    by_date: Dict[str, List[str]] = defaultdict(list)
    for path in hdf5_files:
        date_str = _parse_date_from_filename(os.path.basename(path))
        if date_str:
            by_date[date_str].append(path)

    dates_sorted = sorted(by_date.keys())
    print(f"{len(dates_sorted)} unique dates to convert (n_jobs={n_jobs})")

    if n_jobs == 1:
        results = [
            _convert_one_l4_day(
                d, by_date[d], tif_dir, row_slice, col_slice, transform, overwrite
            )
            for d in dates_sorted
        ]
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_convert_one_l4_day)(
                d, by_date[d], tif_dir, row_slice, col_slice, transform, overwrite
            )
            for d in dates_sorted
        )

    return sum(1 for r in results if r)


def build_index(tif_dir: str, index_path: str) -> None:
    """Write a CSV index mapping date to GeoTIFF path.

    Parameters
    ----------
    tif_dir : str
        Directory containing smap_*.tif files.
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
    product: str = "L3",
    grid_scope: str = "conus",
    download_only: bool = False,
    overwrite: bool = False,
) -> None:
    """Full pipeline: download HDF5 -> convert to GeoTIFF -> build index."""
    output_dir = str(Path(output_dir).expanduser())
    tif_dir = os.path.join(output_dir, "daily_tif")
    os.makedirs(tif_dir, exist_ok=True)

    hdf5_files = download_smap(
        output_dir, start, end, product=product, grid_scope=grid_scope
    )

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

    row_sl, col_sl, transform = resolve_ease2_grid(grid_scope)
    print(
        f"{grid_scope.upper()} subset: rows {row_sl.start}:{row_sl.stop}, "
        f"cols {col_sl.start}:{col_sl.stop} "
        f"({row_sl.stop - row_sl.start} x {col_sl.stop - col_sl.start} pixels)"
    )

    if product == "L4":
        written = convert_l4_daily(
            all_files, tif_dir, row_sl, col_sl, transform, overwrite=overwrite
        )
        print(f"Wrote {written} daily L4 GeoTIFFs to {tif_dir}")
    else:
        converted = 0
        for hdf_path in all_files:
            result = convert_hdf5_to_geotiff(
                hdf_path, tif_dir, row_sl, col_sl, transform, overwrite=overwrite
            )
            if result:
                converted += 1
        print(f"Converted {converted}/{len(all_files)} files to {tif_dir}")

    index_path = os.path.join(output_dir, "smap_daily_index.csv")
    build_index(tif_dir, index_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download SMAP products and convert to GeoTIFFs",
    )
    parser.add_argument(
        "--product",
        type=str,
        choices=["L3", "L4"],
        default="L3",
        help="SMAP product: L3 (SPL3SMP_E) or L4 (SPL4SMGP). Default: L3.",
    )
    parser.add_argument(
        "--grid-scope",
        type=str,
        choices=["conus", "global"],
        default="conus",
        help="Spatial extent: 'conus' or 'global'. Default: conus.",
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
        default=None,
        help="Download HDF5 files but skip GeoTIFF conversion.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
        help="Overwrite existing GeoTIFFs.",
    )
    args = parser.parse_args()

    run(
        output_dir=args.output_dir,
        start=args.start,
        end=args.end,
        product=args.product,
        grid_scope=args.grid_scope,
        download_only=args.download_only,
        overwrite=args.overwrite,
    )

# ========================= EOF ====================================================================
