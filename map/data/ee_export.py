import os

import ee
import os, pyproj
os.environ.setdefault("PROJ_LIB", pyproj.datadir.get_data_dir())


import geopandas as gpd
from shapely.geometry import box

from map.data.call_ee import stack_bands_climatology, is_authorized


def _export_tile_data(roi, points, desc, bucket, file_prefix, resolution, index_col, region):
    """Helper function to run and export data for a given ROI and point set."""
    try:
        # Build stack for this ROI
        stack = stack_bands_climatology(roi, resolution=resolution, region=region)

        # Skip no-op tasks
        if points.size().eq(0).getInfo():
            print(f'{desc}: no points to sample, skipping.')
            return

        # Sample with selected properties
        samples = stack.sampleRegions(
            collection=points,
            properties=['MGRS_TILE', index_col],
            scale=resolution,
            tileScale=16
        )

        # Lock output columns (properties + band names)
        band_names = stack.bandNames()
        selectors = ee.List(['MGRS_TILE', index_col]).cat(band_names)

        task = ee.batch.Export.table.toCloudStorage(
            samples,
            description=desc,
            bucket=bucket,
            fileNamePrefix=f'{file_prefix}/{desc}',
            fileFormat='CSV',
            selectors=selectors
        )
        task.start()
        print(f'Started export: {file_prefix}/{desc} (task: {task.id})')

    except Exception as exc:
        print(f'{desc} error: {exc}')


def get_bands(shapefile_path, mgrs_shp_path, bucket, file_prefix, resolution, index_col=None, split_tiles=False,
              check_dir=None, region='conus'):
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
                q_desc = f'swapstress_{tile}_{name}'
                roi_ee_geom = ee.Geometry(geom.__geo_interface__)

                # Filter points to quadrant bounds (skip empty)
                q_points = tile_points.filterBounds(roi_ee_geom)
                if q_points.size().eq(0).getInfo():
                    print(f'{q_desc}: no points in ROI, skipping.')
                    continue

                _export_tile_data(
                    roi=roi_ee_geom,  # pass Geometry (not FC)
                    points=q_points,
                    desc=q_desc,
                    bucket=bucket,
                    file_prefix=file_prefix,
                    resolution=resolution,
                    index_col=index_col,
                    region=region
                )
        else:
            geo_json = mgrs_tile_gdf.geometry.iloc[0].__geo_interface__
            roi_ee_geom = ee.Geometry(geo_json)

            # Symmetric: ensure points lie in the tile bounds
            tile_points_bounded = tile_points.filterBounds(roi_ee_geom)

            _export_tile_data(
                roi=roi_ee_geom,  # pass Geometry (not FC)
                points=tile_points_bounded,
                desc=desc,
                bucket=bucket,
                file_prefix=file_prefix,
                resolution=resolution,
                index_col=index_col,
                region=region
            )


if __name__ == '__main__':
    """"""
    run_mt_mesonet_workflow = False
    run_general_workflow = False
    run_global_workflow = True

    home = os.path.expanduser('~')
    root_ = os.path.join(home, 'data', 'IrrigationGIS')
    gcs_bucket_ = 'wudr'

    if run_mt_mesonet_workflow:
        extracts_dir_ = os.path.join(root_, 'soils', 'swapstress', 'mt_mesonet_extracts')
        shapefile_ = os.path.join(root_, 'soils', 'soil_potential_obs', 'mt_mesonet', 'station_metadata_mgrs.shp')
        index_ = 'station'
        output_prefix_ = 'swap-stress/mesonet_training_data'
        mgrs_shapefile_ = os.path.join(root_, 'boundaries', 'mgrs', 'mgrs_wgs.shp')

        is_authorized()
        get_bands(shapefile_path=shapefile_,
                  mgrs_shp_path=mgrs_shapefile_,
                  bucket=gcs_bucket_,
                  file_prefix=output_prefix_,
                  resolution=4000,
                  index_col=index_,
                  split_tiles=False,
                  check_dir=extracts_dir_)

    elif run_general_workflow:
        extracts_dir_ = os.path.join(root_, 'soils', 'swapstress', 'extracts')
        shapefile_ = os.path.join(root_, 'soils', 'gis', 'pretraining-roi-10000_mgrs.shp')
        index_ = 'site_id'
        output_prefix_ = 'swap-stress/training_data'
        mgrs_shapefile_ = os.path.join(root_, 'boundaries', 'mgrs', 'mgrs_wgs.shp')

        is_authorized()
        get_bands(shapefile_path=shapefile_,
                  mgrs_shp_path=mgrs_shapefile_,
                  bucket=gcs_bucket_,
                  file_prefix=output_prefix_,
                  resolution=4000,
                  index_col=index_,
                  split_tiles=True,
                  check_dir=extracts_dir_)

    elif run_global_workflow:
        extracts_dir_ = os.path.join(root_, 'soils', 'swapstress', 'global_extracts')
        # shapefile_ = os.path.join(root_, 'soils', 'soil_potential_obs', 'global', 'station_metadata_mgrs.shp')
        shapefile_ = '/home/dgketchum/Downloads/wrc_aggregated_mgrs.shp'
        index_ = 'layer_id'
        output_prefix_ = 'swap-stress/global_training_data'
        # mgrs_shapefile_ = os.path.join(root_, 'boundaries', 'mgrs', 'mgrs_wgs.shp')
        mgrs_shapefile_ = '/home/dgketchum/Downloads/mgrs/mgrs_world_attr.shp'

        is_authorized()
        get_bands(shapefile_path=shapefile_,
                  mgrs_shp_path=mgrs_shapefile_,
                  bucket=gcs_bucket_,
                  file_prefix=output_prefix_,
                  resolution=4000,
                  index_col=index_,
                  split_tiles=False,
                  check_dir=extracts_dir_,
                  region='global')
# ========================= EOF ====================================================================
