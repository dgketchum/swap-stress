"""
Download SMOS-IC V2 soil moisture and convert to EASE-Grid2 GeoTIFFs.

Downloads daily NetCDF files from the CATDS HTTPS server, extracts
Soil_Moisture for the AM pass (ascending = 6 AM local), applies quality
flags (Scene_Flags <= 1), and writes GeoTIFFs on the native 25 km
EASE-Grid2.

Source grid: EASE-Grid2 M25 — 1388 cols × 584 rows, 25025.26 m cell size
CRS: EPSG:6933 (same family as SMAP 9 km, just coarser)
Period: 2010-03 to 2021 (V2, ASC pass)

Data source:
    https://data.catds.fr/cecsm/Land_products/L3_SMOS_IC_Soil_Moisture/

NetCDF variables used:
    Soil_Moisture   — surface SM (m³/m³)
    Scene_Flags     — QC: 0=nominal, 1=scene dependent, >=4 = bad

Usage:
    python -m map.data.smos_ic_download \
        --output-dir /nas/soils/smos/SMOS_IC \
        --start 2010-03-01 --end 2021-12-31

    # Download only (skip GeoTIFF conversion)
    python -m map.data.smos_ic_download \
        --output-dir /nas/soils/smos/SMOS_IC \
        --start 2015-01-01 --end 2021-12-31 \
        --download-only
"""

import argparse
import os
import re
import urllib.request
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

# ---------------------------------------------------------------------------
# EASE-Grid2 M25 parameters
# ---------------------------------------------------------------------------
M25_SCALE = 25025.26000  # metres per pixel
M25_COLS = 1388
M25_ROWS = 584

EASE2_CRS = CRS.from_epsg(6933)

_BASE_URL = "https://data.catds.fr/cecsm/Land_products/L3_SMOS_IC_Soil_Moisture"
_SM_VAR = "Soil_Moisture"
_SF_VAR = "Scene_Flags"
_MAX_GOOD_FLAG = 1
_SM_FILL_SENTINEL = -999.0

_FILENAME_RE = re.compile(
    r"SM_\w+_MIR_CDF3S([AD])_(\d{8})T\d{6}_(\d{8})T\d{6}_.*\.DBL\.nc"
)


class _LinkParser(HTMLParser):
    """Extract href links from an Apache directory listing."""

    def __init__(self):
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, val in attrs:
                if name == "href" and val and not val.startswith("?"):
                    self.links.append(val)


def _list_remote_nc(year_url: str) -> List[str]:
    """List .nc files from a CATDS year directory."""
    try:
        with urllib.request.urlopen(year_url, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  Error listing {year_url}: {e}")
        return []

    parser = _LinkParser()
    parser.feed(html)
    return [link for link in parser.links if link.endswith(".DBL.nc")]


def _parse_date_from_filename(filename: str) -> Optional[str]:
    """Extract YYYYMMDD from SMOS-IC filename."""
    m = _FILENAME_RE.search(filename)
    if m:
        return m.group(2)
    return None


def _ease2_m25_transform() -> Tuple[Affine, float, float]:
    """Compute the rasterio Affine for the full EASE2 M25 global grid.

    The lat/lon arrays in the NetCDF are cell centres on a regular EASE2
    grid.  We compute the upper-left pixel edge from the first coordinate.
    """
    from pyproj import Transformer

    t = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
    # SW corner cell centre (from the NetCDF)
    x_sw, y_sw = t.transform(-179.8703, -83.51714)
    # The grid is stored S→N in the NetCDF; for GeoTIFF we flip to N→S
    x_ul = x_sw - 0.5 * M25_SCALE
    y_ul = y_sw + (M25_ROWS - 0.5) * M25_SCALE
    return Affine(M25_SCALE, 0, x_ul, 0, -M25_SCALE, y_ul)


def download_smos_ic(
    output_dir: str,
    start: str,
    end: str,
    orbit: str = "ASC",
) -> List[str]:
    """Download SMOS-IC NetCDF files from CATDS.

    Parameters
    ----------
    output_dir : str
        Root output directory (nc/ subdir created).
    start, end : str
        Date range (YYYY-MM-DD).
    orbit : str
        "ASC" (6 AM, preferred) or "DES" (6 PM).

    Returns
    -------
    list
        Paths to downloaded files.
    """
    nc_dir = os.path.join(output_dir, "nc")
    os.makedirs(nc_dir, exist_ok=True)

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    years = range(start_dt.year, end_dt.year + 1)
    downloaded: List[str] = []

    for year in years:
        year_url = f"{_BASE_URL}/{orbit}/{year}/"
        print(f"Listing {year_url} ...")
        nc_names = _list_remote_nc(year_url)
        if not nc_names:
            print(f"  No files found for {year}")
            continue

        for nc_name in nc_names:
            date_str = _parse_date_from_filename(nc_name)
            if date_str is None:
                continue
            file_dt = datetime.strptime(date_str, "%Y%m%d")
            if file_dt < start_dt or file_dt > end_dt:
                continue

            out_path = os.path.join(nc_dir, nc_name)
            if os.path.exists(out_path):
                downloaded.append(out_path)
                continue

            file_url = year_url + nc_name
            try:
                urllib.request.urlretrieve(file_url, out_path)
                downloaded.append(out_path)
            except Exception as e:
                print(f"  Error downloading {nc_name}: {e}")
                if os.path.exists(out_path):
                    os.remove(out_path)

        print(
            f"  {year}: {len([d for d in downloaded if f'/{year}/' in d or date_str[:4] == str(year)])} files"
        )

    print(f"Total downloaded: {len(downloaded)} files")
    return downloaded


def convert_nc_to_geotiff(
    nc_path: str,
    tif_dir: str,
    transform: Affine,
    overwrite: bool = False,
) -> Optional[str]:
    """Extract SM from a SMOS-IC NetCDF and write a GeoTIFF.

    Quality filter: Scene_Flags <= 1.
    The NetCDF stores data S→N; the GeoTIFF is written N→S.
    """
    import netCDF4 as nc

    fname = os.path.basename(nc_path)
    date_str = _parse_date_from_filename(fname)
    if date_str is None:
        return None

    out_name = f"smos_ic_{date_str}.tif"
    out_path = os.path.join(tif_dir, out_name)

    if not overwrite and os.path.exists(out_path):
        return out_path

    try:
        ds = nc.Dataset(nc_path)
        sm = ds[_SM_VAR][:].astype(np.float32)
        sf = ds[_SF_VAR][:].astype(np.int8)
        ds.close()
    except Exception as e:
        print(f"  Error reading {fname}: {e}")
        return None

    # Mask bad QC and fill values
    bad = (sf > _MAX_GOOD_FLAG) | (sf < 0) | (sm < 0) | (sm > 1.0)
    if hasattr(sm, "mask"):
        bad = bad | sm.mask
    sm = np.where(bad, np.nan, sm).astype(np.float32)

    # Flip S→N to N→S for GeoTIFF convention
    sm = np.flipud(sm)

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=M25_ROWS,
        width=M25_COLS,
        count=1,
        dtype="float32",
        crs=EASE2_CRS,
        transform=transform,
        nodata=np.nan,
        compress="zstd",
    ) as dst:
        dst.write(sm, 1)
        dst.set_band_description(1, "soil_moisture")

    return out_path


def build_index(tif_dir: str, index_path: str) -> None:
    """Write a CSV index mapping date to GeoTIFF path."""
    import pandas as pd

    records = []
    for fname in sorted(os.listdir(tif_dir)):
        if not fname.endswith(".tif"):
            continue
        date_str = _parse_date_from_filename(fname)
        if date_str is None:
            m = re.search(r"_(\d{8})\.", fname)
            if m:
                date_str = m.group(1)
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
    orbit: str = "ASC",
    download_only: bool = False,
    overwrite: bool = False,
    n_jobs: int = 1,
) -> None:
    """Full pipeline: download NetCDF -> convert to GeoTIFF -> build index."""
    output_dir = str(Path(output_dir).expanduser())
    tif_dir = os.path.join(output_dir, "daily_tif")
    os.makedirs(tif_dir, exist_ok=True)

    nc_files = download_smos_ic(output_dir, start, end, orbit=orbit)

    if download_only:
        print("Download-only mode, stopping.")
        return

    # Also pick up any previously-downloaded files
    nc_dir = os.path.join(output_dir, "nc")
    if os.path.isdir(nc_dir):
        existing = [
            os.path.join(nc_dir, f) for f in os.listdir(nc_dir) if f.endswith(".nc")
        ]
        all_files = sorted(set(nc_files + existing))
    else:
        all_files = sorted(nc_files)

    if not all_files:
        print("No NetCDF files to convert.")
        return

    transform = _ease2_m25_transform()
    print(f"EASE2 M25 grid: {M25_ROWS}×{M25_COLS}, {M25_SCALE:.1f} m/px")

    if n_jobs == 1:
        converted = 0
        for nc_path in all_files:
            result = convert_nc_to_geotiff(nc_path, tif_dir, transform, overwrite)
            if result:
                converted += 1
        print(f"Converted {converted}/{len(all_files)} files to {tif_dir}")
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(convert_nc_to_geotiff)(p, tif_dir, transform, overwrite)
            for p in all_files
        )
        print(f"Converted {sum(1 for r in results if r)}/{len(all_files)} files")

    index_path = os.path.join(output_dir, "smos_ic_daily_index.csv")
    build_index(tif_dir, index_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download SMOS-IC V2 SM and convert to GeoTIFFs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Root output directory (nc/ and daily_tif/ subdirs created).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2010-03-01",
        help="Start date YYYY-MM-DD (default: 2010-03-01).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2021-12-31",
        help="End date YYYY-MM-DD (default: 2021-12-31).",
    )
    parser.add_argument(
        "--orbit",
        type=str,
        choices=["ASC", "DES"],
        default="ASC",
        help="Orbit pass: ASC (6 AM, default) or DES (6 PM).",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download NetCDF files but skip GeoTIFF conversion.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing GeoTIFFs.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel workers for GeoTIFF conversion (default: 1).",
    )
    args = parser.parse_args()

    run(
        output_dir=args.output_dir,
        start=args.start,
        end=args.end,
        orbit=args.orbit,
        download_only=args.download_only,
        overwrite=args.overwrite,
        n_jobs=args.n_jobs,
    )

# ========================= EOF ====================================================================
