import os
import sys
import ee
import geopandas as gpd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../EEMapper/map')))
from map.call_ee import is_authorized
from map.call_ee import stack_bands_climatology


def get_bands(shapefile_path, bucket, file_prefix, extract_alpha_earth=False):
    """
    Extract climatological data for a set of points from a local shapefile.
    """
    points_df = gpd.read_file(shapefile_path)
    mgrs_ee_tiles = ee.FeatureCollection('users/dgketchum/boundaries/MGRS_TILE')

    mgrs_tiles = points_df['MGRS_TILE'].unique()

    for tile in mgrs_tiles:

        tile_df = points_df[points_df['MGRS_TILE'] == tile]
        tile_points = ee.FeatureCollection(tile_df.__geo_interface__)

        desc = f'climatology_{tile}'
        if extract_alpha_earth:
            desc = f'{desc}_ae'

        roi = mgrs_ee_tiles.filter(ee.Filter.eq('MGRS_TILE', tile))

        try:
            stack = stack_bands_climatology(roi, alpha_earth=extract_alpha_earth)
        except ee.ee_exception.EEException as exc:
            print(f'{desc} error: {exc}')
            continue

        plot_sample_regions = stack.sampleRegions(
            collection=tile_points,
            properties=['MGRS_TILE', 'POINT_TYPE'],
            scale=30,
            tileScale=16)

        task = ee.batch.Export.table.toCloudStorage(
            plot_sample_regions,
            description=desc,
            bucket=bucket,
            fileNamePrefix=f'{file_prefix}/{desc}',
            fileFormat='CSV')

        task.start()
        print(f'Started export: {file_prefix}/{desc}')


if __name__ == '__main__':
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
              extract_alpha_earth=True)

# ========================= EOF ====================================================================
