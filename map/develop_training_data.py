import os
import sys
from glob import glob
import json

import ee
import geopandas as gpd
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../EEMapper/map')))
from map.call_ee import is_authorized
from map.call_ee import stack_bands_climatology


def get_bands(shapefile_path, bucket, file_prefix, resolution):
    """
    Extract climatological data for a set of points from a local shapefile.
    """
    points_df = gpd.read_file(shapefile_path)
    mgrs_ee_tiles = ee.FeatureCollection('users/dgketchum/boundaries/MGRS_TILE')

    mgrs_tiles = points_df['MGRS_TILE'].unique()

    for tile in mgrs_tiles:

        tile_df = points_df[points_df['MGRS_TILE'] == tile]
        tile_points = ee.FeatureCollection(tile_df.__geo_interface__)

        desc = f'swapstress_{tile}'
        roi = mgrs_ee_tiles.filter(ee.Filter.eq('MGRS_TILE', tile))

        try:
            stack = stack_bands_climatology(roi, resolution=resolution)
        except ee.ee_exception.EEException as exc:
            print(f'{desc} error: {exc}')
            continue

        plot_sample_regions = stack.sampleRegions(
            collection=tile_points,
            properties=['MGRS_TILE', 'site_id'],
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
    # TODO: fix extract to properly concat the columns under 'site_id' index, until then:
    rosetta_df = rosetta_df.groupby('site_id').first()

    final_df = ee_df.join(rosetta_df, how='left')
    final_df = final_df[sorted(final_df.columns.to_list())]

    if categorical_mappings_json:
        mappings = {}
        for col in categories:
            if final_df[col].dtype == 'object':
                final_df[col] = final_df[col].astype('category')
            mappings[col] = {k: int(v) for v, k in enumerate(final_df[col].unique())}

        with open(categorical_mappings_json, 'w') as f:
            json.dump(mappings, f, indent=4)
        print(f"Saved categorical mappings to {categorical_mappings_json}")

    out_dir = os.path.dirname(out_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    final_df.to_parquet(out_file)
    print(f"Saving final concatenated data to {out_file}")


if __name__ == '__main__':

    run_gee_extract = False
    if run_gee_extract:
        is_authorized()
        root = '/media/research/IrrigationGIS'
        if not os.path.exists(root):
            root = '/home/dgketchum/data/IrrigationGIS'

        shapefile = os.path.join(root, 'soils', 'gis', 'pretraining-roi-10000_mgrs.shp')
        gcs_bucket = 'wudr'
        output_prefix = 'swap-stress/training_data'

        get_bands(shapefile_path=shapefile,
                  bucket=gcs_bucket,
                  file_prefix=output_prefix,
                  resolution=4000)

    run_concatenate = True
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
