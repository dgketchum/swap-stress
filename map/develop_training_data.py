import os
import sys
from glob import glob
import json

import ee
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../EEMapper/map')))
from map.call_ee import is_authorized, stack_bands_climatology


def _export_tile_data(roi, points, desc, bucket, file_prefix, resolution, index_col):
    """Helper function to run and export data for a given ROI and point set."""

    try:
        stack = stack_bands_climatology(roi, resolution=resolution)
    except ee.ee_exception.EEException as exc:
        print(f'{desc} error: {exc}')
        return

    plot_sample_regions = stack.sampleRegions(
        collection=points,
        properties=['MGRS_TILE', index_col],
        scale=resolution,
        tileScale=16)

    task = ee.batch.Export.table.toCloudStorage(
        plot_sample_regions,
        description=desc,
        bucket=bucket,
        fileNamePrefix=f'{file_prefix}/{desc}',
        fileFormat='CSV')

    task.start()
    print(f'Started export: {file_prefix}/{desc}')


def get_bands(shapefile_path, mgrs_shp_path, bucket, file_prefix, resolution, index_col=None, split_tiles=False,
              check_dir=None):
    """
    Extract climatological data for a set of points from a local shapefile.
    """
    points_df = gpd.read_file(shapefile_path)
    mgrs_gdf = gpd.read_file(mgrs_shp_path)

    if index_col not in points_df.columns:
        raise ValueError(f"Index column '{index_col}' not found in shapefile.")

    mgrs_tiles = points_df['MGRS_TILE'].unique()

    for tile in mgrs_tiles:

        desc = f'swapstress_{tile}'

        if check_dir:
            expected_path = os.path.join(check_dir, f'{desc}.csv')
            if os.path.exists(expected_path):
                print(f'File already exists: {expected_path}. Skipping export.')
                continue

        tile_df = points_df[points_df['MGRS_TILE'] == tile]
        if tile_df.empty:
            continue

        tile_points = ee.FeatureCollection(tile_df.__geo_interface__)

        mgrs_tile_gdf = mgrs_gdf[mgrs_gdf['MGRS_TILE'] == tile]
        if mgrs_tile_gdf.empty:
            print(f'Warning: MGRS tile {tile} not found in {mgrs_shp_path}. Skipping.')
            continue

        if split_tiles:
            min_lon, min_lat, max_lon, max_lat = mgrs_tile_gdf.geometry.iloc[0].bounds
            center_lon = (min_lon + max_lon) / 2
            center_lat = (min_lat + max_lat) / 2

            quadrants = {
                'SW': box(min_lon, min_lat, center_lon, center_lat),
                'SE': box(center_lon, min_lat, max_lon, center_lat),
                'NW': box(min_lon, center_lat, center_lon, max_lat),
                'NE': box(center_lon, center_lat, max_lon, max_lat)
            }

            for name, geom in quadrants.items():
                desc = f'swapstress_{tile}_{name}'
                roi_ee_geom = ee.Geometry(geom.__geo_interface__)
                _export_tile_data(roi=ee.FeatureCollection(roi_ee_geom),
                                  points=tile_points.filterBounds(roi_ee_geom),
                                  desc=desc,
                                  bucket=bucket,
                                  file_prefix=file_prefix,
                                  resolution=resolution,
                                  index_col=index_col)
        else:
            geo_json = mgrs_tile_gdf.geometry.iloc[0].__geo_interface__
            roi_ee_geom = ee.Geometry(geo_json)
            _export_tile_data(roi=ee.FeatureCollection(roi_ee_geom),
                              points=tile_points,
                              desc=desc,
                              bucket=bucket,
                              file_prefix=file_prefix,
                              resolution=resolution,
                              index_col=index_col)


def concatenate_and_join(ee_in_dir, rosetta_pqt, out_file, categorical_mappings_json=None, categories=None):
    """
    Concatenates CSVs from Earth Engine extraction, joins with Rosetta data,
    and saves to a single Parquet file.
    """
    if categorical_mappings_json is not None and categories is None:
        raise ValueError

    csv_files = glob(os.path.join(ee_in_dir, '*.csv'))
    if not csv_files:
        print(f"No CSV files found in {ee_in_dir}")
        return

    print(f"Found {len(csv_files)} CSV files to concatenate.")
    df_list = []

    for f in csv_files:
        try:
            df_list.append(pd.read_csv(f))
        except pd.errors.EmptyDataError:
            print(f'Found empty file {os.path.basename(f)}, removing')
            os.remove(f)
            continue

    ee_df = pd.concat(df_list, ignore_index=True)
    ee_df = ee_df.drop(columns=['.geo', 'system:index', 'MGRS_TILE'])

    if 'site_id' in ee_df.columns:
        ee_df.set_index('site_id', inplace=True)
    else:
        print("Fatal: 'site_id' column not found in Earth Engine data. Cannot join.")
        return

    rosetta_df = pd.read_parquet(rosetta_pqt)
    rosetta_df = rosetta_df.groupby('site_id').first()

    final_df = ee_df.join(rosetta_df, how='left')
    final_df = final_df[sorted(final_df.columns.to_list())]

    if categorical_mappings_json:
        mappings = {}
        for col in categories:
            if final_df[col].dtype == 'object':
                final_df[col] = final_df[col].astype('category')
            mappings[col] = {int(k): int(v) for v, k in enumerate(final_df[col].unique())}

        with open(categorical_mappings_json, 'w') as f:
            json.dump(mappings, f, indent=4)
        print(f"Saved categorical mappings to {categorical_mappings_json}")

    out_dir = os.path.dirname(out_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    final_df.to_parquet(out_file)
    print(f"Saving final concatenated data to {out_file}\n{len(final_df)} samples")


if __name__ == '__main__':

    run_gee_extract = True
    if run_gee_extract:
        is_authorized()
        root = '/home/dgketchum/data/IrrigationGIS'

        shapefile = os.path.join(root, 'soils', 'gis', 'pretraining-roi-10000_mgrs.shp')
        output_prefix = 'swap-stress/training_data'

        # shapefile = '/home/dgketchum/data/IrrigationGIS/soils/soil_potential_obs/mt_mesonet/station_metadata_mgrs.shp'
        # output_prefix = 'swap-stress/mesonet_training_data'

        check_dir_ = os.path.join(root, 'soils', 'swapstress', 'extracts')

        mgrs_shapefile = os.path.join(root, 'boundaries', 'mgrs', 'mgrs_wgs.shp')
        gcs_bucket = 'wudr'

        get_bands(shapefile_path=shapefile,
                  mgrs_shp_path=mgrs_shapefile,
                  bucket=gcs_bucket,
                  file_prefix=output_prefix,
                  resolution=4000,
                  index_col='site_id',
                  split_tiles=True,
                  check_dir=check_dir_)

    run_concatenate = False
    if run_concatenate:
        ee_extract_dir_ = '/home/dgketchum/data/IrrigationGIS/soils/swapstress/extracts/'
        rosetta_file_ = '/home/dgketchum/data/IrrigationGIS/soils/rosetta/extracted_rosetta_points.parquet'
        output_file_ = '/home/dgketchum/data/IrrigationGIS/soils/swapstress/training/training_data.parquet'
        mappings_json_ = '/home/dgketchum/data/IrrigationGIS/soils/swapstress/training/categorical_mappings.json'

        concatenate_and_join(ee_in_dir=ee_extract_dir_,
                             rosetta_pqt=rosetta_file_,
                             out_file=output_file_,
                             categorical_mappings_json=mappings_json_,
                             categories=['cdl_mode', 'nlcd'])

# ========================= EOF ====================================================================
