"""
Reproject EE-exported covariate rasters from EPSG:5070 to SMAP EASE-Grid2.

EE cannot export in EPSG:6933, so covariates are exported in EPSG:5070
(Conus Albers) at 9000 m. This script reprojects each GeoTIFF to the exact
SMAP 9 km EASE-Grid2 grid defined by smap_download.py, ensuring pixel-perfect
alignment with the daily SMAP soil moisture GeoTIFFs.

Usage:
    # Reproject all *_9km.tif files in place (writes *_ease2.tif alongside)
    python -m map.data.reproject_to_ease2 \
        --input-dir /nas/soils/swapstress/inference/conus_features

    # Reproject specific files
    python -m map.data.reproject_to_ease2 \
        --input-dir /nas/soils/swapstress/inference/conus_features \
        --files soilgrids_9km.tif,worldclim_9km.tif

    # Verify alignment against a SMAP daily GeoTIFF
    python -m map.data.reproject_to_ease2 \
        --input-dir /nas/soils/swapstress/inference/conus_features \
        --verify /path/to/smap_sm_20200101.tif
"""

import argparse
import os
import sys

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from map.data.smap_download import (
    EASE2_CRS,
    MAP_SCALE,
    _conus_slice,
    _conus_transform,
)

# Groups that need nearest-neighbor resampling (categorical/discrete values)
NEAREST_GROUPS = {"landcover", "fao_hwsd", "ssurgo"}


def _resampling_for(filename):
    """Pick resampling method based on filename."""
    stem = filename.replace("_9km.tif", "").replace("_9km", "")
    for cat_group in NEAREST_GROUPS:
        if cat_group in stem:
            return Resampling.nearest
    return Resampling.bilinear


def reproject_raster(src_path, dst_path, dst_crs, dst_transform, dst_width, dst_height):
    """Reproject a single raster to the SMAP EASE-Grid2 CONUS grid."""
    resampling = _resampling_for(os.path.basename(src_path))

    with rasterio.open(src_path) as src:
        dst_data = np.empty((src.count, dst_height, dst_width), dtype=np.float32)

        reproject(
            source=rasterio.band(src, list(range(1, src.count + 1))),
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
            dst_nodata=np.nan,
        )

        band_descriptions = [src.descriptions[i] for i in range(src.count)]

    with rasterio.open(
        dst_path,
        "w",
        driver="GTiff",
        height=dst_height,
        width=dst_width,
        count=dst_data.shape[0],
        dtype="float32",
        crs=dst_crs,
        transform=dst_transform,
        nodata=np.nan,
        compress="zstd",
    ) as dst:
        dst.write(dst_data)
        for i, desc in enumerate(band_descriptions):
            if desc:
                dst.set_band_description(i + 1, desc)

    return resampling.name


def verify_alignment(ease2_path, smap_path):
    """Check that a reprojected raster aligns with a SMAP daily GeoTIFF."""
    with rasterio.open(ease2_path) as cov, rasterio.open(smap_path) as smap:
        ok = True

        if cov.crs != smap.crs:
            print(f"  CRS mismatch: {cov.crs} vs {smap.crs}")
            ok = False
        else:
            print(f"  CRS: {cov.crs} OK")

        if cov.transform != smap.transform:
            print(
                f"  Transform mismatch:\n    covariate: {cov.transform}\n    smap:      {smap.transform}"
            )
            ok = False
        else:
            print(f"  Transform: {cov.transform} OK")

        if (cov.width, cov.height) != (smap.width, smap.height):
            print(
                f"  Shape mismatch: ({cov.width}, {cov.height}) vs ({smap.width}, {smap.height})"
            )
            ok = False
        else:
            print(f"  Shape: {cov.width} x {cov.height} OK")

        return ok


def main():
    parser = argparse.ArgumentParser(
        description="Reproject EE covariate rasters to SMAP EASE-Grid2",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing *_9km.tif files from EE export",
    )
    parser.add_argument(
        "--files",
        default=None,
        help="Comma-separated filenames to reproject (default: all *_9km.tif)",
    )
    parser.add_argument(
        "--verify",
        default=None,
        help="Path to a SMAP daily GeoTIFF for alignment verification",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *_ease2.tif files",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a directory")
        sys.exit(1)

    # Compute target grid from smap_download constants
    row_sl, col_sl = _conus_slice()
    dst_transform = _conus_transform(row_sl, col_sl)
    dst_width = col_sl.stop - col_sl.start
    dst_height = row_sl.stop - row_sl.start

    print(f"Target grid: {dst_width} x {dst_height} pixels")
    print("  CRS: EPSG:6933 (EASE-Grid2)")
    print(f"  Pixel size: {MAP_SCALE:.3f} m")
    print(f"  Transform: {dst_transform}")
    print()

    # Find files to reproject
    if args.files:
        filenames = [f.strip() for f in args.files.split(",")]
    else:
        filenames = sorted(f for f in os.listdir(input_dir) if f.endswith("_9km.tif"))

    if not filenames:
        print("No *_9km.tif files found.")
        sys.exit(0)

    # Reproject each file
    for fname in filenames:
        src_path = os.path.join(input_dir, fname)
        dst_name = fname.replace("_9km.tif", "_ease2.tif")
        dst_path = os.path.join(input_dir, dst_name)

        if not os.path.exists(src_path):
            print(f"  SKIP {fname} (not found)")
            continue

        if os.path.exists(dst_path) and not args.overwrite:
            print(f"  SKIP {dst_name} (exists, use --overwrite)")
            continue

        with rasterio.open(src_path) as src:
            nbands = src.count
            src_crs = src.crs

        resamp = reproject_raster(
            src_path, dst_path, EASE2_CRS, dst_transform, dst_width, dst_height
        )
        print(f"  {fname} -> {dst_name}  ({nbands} bands, {src_crs}, {resamp})")

    # Verify alignment if a SMAP reference is provided
    if args.verify:
        print(f"\nVerification against {args.verify}:")
        ease2_files = sorted(
            f for f in os.listdir(input_dir) if f.endswith("_ease2.tif")
        )
        all_ok = True
        for fname in ease2_files:
            print(f"\n  {fname}:")
            ok = verify_alignment(os.path.join(input_dir, fname), args.verify)
            if not ok:
                all_ok = False

        if all_ok:
            print("\nAll files aligned.")
        else:
            print("\nAlignment issues found.")
            sys.exit(1)


if __name__ == "__main__":
    main()
