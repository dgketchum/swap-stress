"""
Extract AMSR-E/AMSR2 Vegetation Optical Depth (VOD) climatology at training sites.

Reads LPDR v3 NetCDF files (daily VOD at 10.7 GHz, ~25 km EASE-Grid EPSG:3410),
samples nearest-neighbor values at all training site locations, and computes
multi-year seasonal statistics (mean, stddev) per site.

Output: one row per sample_id with 20 columns:
  vod_{asc,desc}_{mean,stddev}_{winter,spring,summer,autumn,annual}

Usage:
    python -m swapstress.features.amsr_extract \
        --amsr-dir /path/to/amsr/nc4s \
        --sites-parquet /path/to/training.parquet \
        --output /path/to/amsr_vod_climatology.parquet
"""

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

# Seasonal DOY ranges matching call_ee.py conventions
_SEASONS = {
    "winter": (335, 59),  # Dec-Feb (wraps year boundary)
    "spring": (60, 151),  # Mar-May
    "summer": (152, 243),  # Jun-Aug
    "autumn": (244, 334),  # Sep-Nov
}


def _doy_in_season(doy: np.ndarray, start: int, end: int) -> np.ndarray:
    """Boolean mask for DOY values within a season range (handles wrap-around)."""
    if start > end:
        return (doy >= start) | (doy <= end)
    return (doy >= start) & (doy <= end)


def _load_sites(sites_parquet: str) -> pd.DataFrame:
    """Load unique (sample_id, lat, lon) from training parquet."""
    df = pd.read_parquet(sites_parquet)
    if df.index.name:
        df = df.reset_index()
    sites = df[["sample_id", "lat", "lon"]].drop_duplicates(subset="sample_id")
    return sites.reset_index(drop=True)


def _reproject_sites(lats: np.ndarray, lons: np.ndarray) -> tuple:
    """Reproject lat/lon (EPSG:4326) to EASE-Grid (EPSG:3410) x/y."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3410", always_xy=True)
    x, y = transformer.transform(lons, lats)
    return x, y


def _find_nearest_indices(
    grid_coords: np.ndarray, query_coords: np.ndarray
) -> np.ndarray:
    """Find nearest grid index for each query coordinate using searchsorted."""
    # grid_coords must be sorted ascending
    idx = np.searchsorted(grid_coords, query_coords)
    idx = np.clip(idx, 1, len(grid_coords) - 1)
    # Pick closer of idx-1 and idx
    left = np.abs(grid_coords[idx - 1] - query_coords)
    right = np.abs(grid_coords[idx] - query_coords)
    idx = np.where(left <= right, idx - 1, idx)
    return idx


def extract_amsr_vod_climatology(
    amsr_dir: str,
    sites_parquet: str,
    output_path: Optional[str] = None,
    start_year: int = 2002,
    end_year: int = 2022,
) -> pd.DataFrame:
    """
    Extract multi-year seasonal VOD climatology from AMSR LPDR v3 NetCDF files.

    Parameters
    ----------
    amsr_dir : str
        Directory containing AMSR-E-2_LPDRv3_Y{year}_{A,D}.nc4 files.
    sites_parquet : str
        Training table parquet (needs sample_id, lat, lon columns).
    output_path : str, optional
        Path to write output parquet.
    start_year : int
        First year to include.
    end_year : int
        Last year to include.

    Returns
    -------
    pd.DataFrame
        One row per sample_id, 20 VOD statistic columns.
    """
    # Load and reproject sites
    sites = _load_sites(sites_parquet)
    print(f"Loaded {len(sites)} unique sites")

    site_x, site_y = _reproject_sites(sites["lat"].values, sites["lon"].values)

    # Collect daily VOD per pass across all years
    # Store as list of (doy, vod_array[n_sites]) tuples per pass
    pass_data = {"asc": [], "desc": []}
    pass_codes = {"A": "asc", "D": "desc"}

    grid_x = None
    grid_y = None
    xi = None
    yi = None

    for year in range(start_year, end_year + 1):
        for code, pass_name in pass_codes.items():
            fname = f"AMSR-E-2_LPDRv3_Y{year}_{code}.nc4"
            fpath = os.path.join(amsr_dir, fname)
            if not os.path.exists(fpath):
                print(f"  Missing: {fname}")
                continue

            ds = xr.open_dataset(fpath)

            # Build grid indices on first file
            if grid_x is None:
                gx = ds.x.values
                gy = ds.y.values
                # y may be descending — sort for searchsorted
                if gy[0] > gy[-1]:
                    gy = gy[::-1]
                    y_flipped = True
                else:
                    y_flipped = False
                grid_x = gx
                grid_y = gy
                xi = _find_nearest_indices(grid_x, site_x)
                yi_raw = _find_nearest_indices(grid_y, site_y)
                if y_flipped:
                    yi = len(grid_y) - 1 - yi_raw
                else:
                    yi = yi_raw

            vod = ds["VOD"].values  # (time, y, x)
            times = pd.DatetimeIndex(ds.time.values)
            doys = times.dayofyear

            # Extract VOD at all sites for all timesteps
            # vod[:, yi, xi] gives (n_time, n_sites)
            site_vod = vod[:, yi, xi]

            # Mask invalid values (outside 0-3 range)
            site_vod[(site_vod < 0) | (site_vod > 3)] = np.nan

            for t in range(len(times)):
                pass_data[pass_name].append((int(doys[t]), site_vod[t, :]))

            ds.close()
            print(f"  Processed {fname}: {len(times)} days")

    # Compute seasonal statistics
    n_sites = len(sites)
    result_cols = {}

    for pass_name in ["asc", "desc"]:
        records = pass_data[pass_name]
        if not records:
            print(f"  Warning: no data for {pass_name} pass")
            for season_name in list(_SEASONS.keys()) + ["annual"]:
                result_cols[f"vod_{pass_name}_mean_{season_name}"] = np.full(
                    n_sites, np.nan
                )
                result_cols[f"vod_{pass_name}_stddev_{season_name}"] = np.full(
                    n_sites, np.nan
                )
            continue

        all_doys = np.array([r[0] for r in records])
        # Stack into (n_days, n_sites)
        all_vod = np.stack([r[1] for r in records], axis=0)

        # Annual stats
        with np.errstate(all="ignore"):
            result_cols[f"vod_{pass_name}_mean_annual"] = np.nanmean(all_vod, axis=0)
            result_cols[f"vod_{pass_name}_stddev_annual"] = np.nanstd(all_vod, axis=0)

        # Seasonal stats
        for season_name, (s_start, s_end) in _SEASONS.items():
            mask = _doy_in_season(all_doys, s_start, s_end)
            season_vod = all_vod[mask, :]

            with np.errstate(all="ignore"):
                result_cols[f"vod_{pass_name}_mean_{season_name}"] = np.nanmean(
                    season_vod, axis=0
                )
                result_cols[f"vod_{pass_name}_stddev_{season_name}"] = np.nanstd(
                    season_vod, axis=0
                )

    result = pd.DataFrame(result_cols)
    result.insert(0, "sample_id", sites["sample_id"].values)

    print(f"\nResult: {len(result)} sites, {len(result.columns)} columns")
    for col in result.columns:
        if col != "sample_id":
            valid = result[col].notna().sum()
            print(
                f"  {col}: {valid}/{len(result)} valid ({100 * valid / len(result):.1f}%)"
            )

    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        result.to_parquet(output_path, index=False)
        print(f"\nSaved to {output_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract AMSR VOD climatology at training sites.",
    )
    parser.add_argument(
        "--amsr-dir",
        type=str,
        required=True,
        help="Directory containing AMSR LPDR v3 nc4 files.",
    )
    parser.add_argument(
        "--sites-parquet",
        type=str,
        required=True,
        help="Training table parquet (needs sample_id, lat, lon).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output parquet path for VOD climatology.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2002,
        help="First year to include (default: 2002).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2022,
        help="Last year to include (default: 2022).",
    )
    args = parser.parse_args()

    extract_amsr_vod_climatology(
        amsr_dir=args.amsr_dir,
        sites_parquet=args.sites_parquet,
        output_path=args.output,
        start_year=args.start_year,
        end_year=args.end_year,
    )
