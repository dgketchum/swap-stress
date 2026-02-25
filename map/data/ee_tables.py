import argparse
import os
from glob import glob
import json

import geopandas as gpd
import pandas as pd

from map.data.source_registry import get_source, DataPaths, VALID_SCALES

CATEGORIES = [
    "hhs_stc",
    "glc10_lc",
    "WRB4",
    "WRB_PHASES",
    "WRB2_CODE",
    "FAO90",
    "KOPPEN",
    "TEXTURE_USDA",
]

DROPCOLS_250M = [
    ".geo",
    "system:index",
    "MGRS_TILE",
    "name",
    "has_swp",
    "source",
    "network",
    "HWSD2_ID",
    "WISE30s_ID",
    "COVERAGE",
    "SHARE",
    "SWCC_class",
    "obs_ct",
    "nwsli_id",
    "mesowest_i",
    "gwic_id",
    "funded",
]

DROPCOLS_9KM = [".geo", "system:index", "constant"]


def concatenate_and_join(
    ee_in_dir,
    out_file,
    network,
    rosetta_pqt=None,
    index_col="site_id",
    categories=None,
    categorical_mappings_json=None,
    dropcols=None,
):
    """
    Concatenates CSVs from Earth Engine extraction, joins with Rosetta data,
    and saves to a single Parquet file.
    """
    if categorical_mappings_json is not None and categories is None:
        raise ValueError

    csv_files = glob(os.path.join(ee_in_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {ee_in_dir}")
        return

    print(f"Found {len(csv_files)} CSV files to concatenate.")
    df_list = []

    for f in sorted(csv_files):
        try:
            c = pd.read_csv(f)
            print(os.path.basename(f), len(c))
        except pd.errors.EmptyDataError:
            print(f"Found empty file {os.path.basename(f)}, removing")
            os.remove(f)
            continue
        if c.empty:
            print(f"Found empty file {os.path.basename(f)}, removing")
            os.remove(f)
            continue
        if "uid" in c.columns:
            print(f"UID in {os.path.basename(f)}, removing")
            os.remove(f)
            continue
        df_list.append(c)

    ee_df = pd.concat(df_list, ignore_index=True)

    if dropcols:
        drop_from_ee = [c for c in dropcols if c in ee_df.columns]
        ee_df = ee_df.drop(columns=drop_from_ee)

    ee_df[index_col] = ee_df[index_col].astype(str)
    ee_df.set_index(index_col, inplace=True)

    try:
        ee_df["elevation"] = ee_df["elevation"].astype(float)
    except KeyError:
        pass

    if rosetta_pqt:
        rosetta_df = pd.read_parquet(rosetta_pqt)
        rosetta_df = rosetta_df.groupby(index_col).first()
        if dropcols:
            drop_from_rose = [c for c in dropcols if c in rosetta_df.columns]
            rosetta_df = rosetta_df.drop(columns=drop_from_rose)

        try:
            rosetta_df.drop(columns=["elevation"], inplace=True)
        except KeyError:
            pass

        for col in ee_df.columns.to_list():
            if col in rosetta_df:
                ee_df = ee_df.drop(columns=[col])

        final_df = ee_df.join(rosetta_df, how="left")
    else:
        final_df = ee_df

    final_df = final_df[sorted(final_df.columns.to_list())]

    if categorical_mappings_json:
        mappings = {}
        for col in categories:
            final_df.loc[:, col] = final_df[col].values.astype(int)
            mappings[col] = {
                int(k): int(v) for v, k in enumerate(final_df[col].unique())
            }

        with open(categorical_mappings_json, "w") as f:
            json.dump(mappings, f, indent=4)
        print(f"Saved categorical mappings to {categorical_mappings_json}")

    out_dir = os.path.dirname(out_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    final_df.index.name = index_col
    final_df.to_parquet(out_file)
    print(f"Saving final concatenated data to {out_file} {len(final_df)} samples")


def convert_single_csv_to_parquet(
    csv_path,
    out_file,
    index_col,
    shapefile_path,
    lat_col=None,
    lon_col=None,
    dropcols=None,
):
    """Convert a single EE CSV (9km extraction) to parquet with lat/lon from shapefile.

    Parameters
    ----------
    csv_path : str
        Path to the single CSV file.
    out_file : str
        Output parquet path.
    index_col : str
        Primary key column name.
    shapefile_path : str
        Path to source shapefile for coordinate lookup.
    lat_col : str or None
        Latitude column name in shapefile. If None, extract from geometry centroid.
    lon_col : str or None
        Longitude column name in shapefile. If None, extract from geometry centroid.
    dropcols : list of str, optional
        Columns to drop from the CSV.
    """
    df = pd.read_csv(csv_path)
    print(f"Read {len(df)} rows from {csv_path}")

    if dropcols:
        drop = [c for c in dropcols if c in df.columns]
        df = df.drop(columns=drop)

    df[index_col] = df[index_col].astype(str)
    df = df.set_index(index_col)

    # Join lat/lon from shapefile
    gdf = gpd.read_file(shapefile_path)
    gdf[index_col] = gdf[index_col].astype(str)
    gdf = gdf.set_index(index_col)

    if lat_col and lon_col:
        coords = gdf[[lat_col, lon_col]].rename(
            columns={lat_col: "lat", lon_col: "lon"}
        )
    else:
        coords = pd.DataFrame(
            {"lat": gdf.geometry.centroid.y, "lon": gdf.geometry.centroid.x},
            index=gdf.index,
        )

    df = df.join(coords, how="left")

    df = df[sorted(df.columns.to_list())]

    out_dir = os.path.dirname(out_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    df.to_parquet(out_file)
    print(f"Saved {len(df)} rows to {out_file}")


# Rosetta parquet paths for 250m workflows (relative to data_root)
_ROSETTA_SUBPATHS = {
    "gshp": "soil_potential_obs/gshp/extracted_rosetta_points.parquet",
    "mt_mesonet": "rosetta/mt_mesonet/extracted_rosetta_points.parquet",
    "reesh": "soil_potential_obs/reesh/extracted_rosetta_points.parquet",
    "ncss": None,
    "lacadian": None,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert EE CSV extracts to training parquets.",
    )
    parser.add_argument(
        "--source",
        type=str,
        nargs="+",
        required=True,
        help="Source name(s) to process (e.g., gshp ncss mt_mesonet reesh lacadian).",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default="250m",
        choices=VALID_SCALES,
        help="Resolution scale (default: 250m).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/nas/soils",
        help="Root data directory (default: /nas/soils).",
    )
    args = parser.parse_args()

    for source_name in args.source:
        print(f"\n=== Processing {source_name} at {args.scale} ===")
        source = get_source(source_name)
        paths = DataPaths(args.data_root, source, scale=args.scale)

        if paths.is_single_csv:
            csv_path = paths.ee_csv_file
            if not os.path.exists(csv_path):
                print(f"  CSV not found: {csv_path}, skipping")
                continue
            shp_path = paths.shapefile
            if not shp_path or not os.path.exists(shp_path):
                print(f"  Shapefile not found: {shp_path}, skipping")
                continue
            convert_single_csv_to_parquet(
                csv_path=csv_path,
                out_file=paths.ee_table,
                index_col=source.index_col,
                shapefile_path=shp_path,
                lat_col=source.lat_col,
                lon_col=source.lon_col,
                dropcols=DROPCOLS_9KM,
            )
        else:
            rosetta_subpath = _ROSETTA_SUBPATHS.get(source_name)
            rosetta_pqt = (
                os.path.join(args.data_root, rosetta_subpath)
                if rosetta_subpath
                else None
            )
            mappings_json = os.path.join(
                args.data_root,
                "swapstress",
                "training",
                f"{source_name}_categorical_mappings_250m.json",
            )
            concatenate_and_join(
                ee_in_dir=paths.ee_extracts_dir,
                out_file=paths.ee_table,
                rosetta_pqt=rosetta_pqt,
                index_col=source.index_col,
                categorical_mappings_json=mappings_json,
                network=source_name,
                categories=CATEGORIES,
                dropcols=DROPCOLS_250M,
            )

# ========================= EOF ====================================================================
