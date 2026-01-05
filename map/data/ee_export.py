import os

import ee
import os, pyproj

os.environ.setdefault("PROJ_LIB", pyproj.datadir.get_data_dir())

import geopandas as gpd
from shapely.geometry import box

from map.data.call_ee import stack_bands_climatology, is_authorized


def _export_tile_data(roi, points, desc, bucket, file_prefix, resolution, index_col, region, diagnose=False):
    """Helper function to run and export data for a given ROI and point set."""
    stack = stack_bands_climatology(roi, region=region)

    if points.size().eq(0).getInfo():
        print(f'{desc}: no points to sample, skipping.')
        return

    # Optional diagnostic: probe one point and check band-by-band values
    if diagnose:
        try:
            print(desc)
            filtered = ee.FeatureCollection([points.first()])
            bad_ = []
            bands = stack.bandNames().getInfo()
            for b in bands:
                sel = stack.select([b])
                sample = sel.sampleRegions(collection=filtered, properties=[], scale=resolution).first()
                val = ee.Algorithms.If(sample, ee.Feature(sample).get(b), None)
                try:
                    info = ee.Dictionary({'v': val}).get('v').getInfo()
                    print(b, info)
                    if info is None:
                        bad_.append(b)
                except Exception as e:
                    print(b, 'not there', e)
                    bad_.append(b)
            print('Bands with None or errors:', bad_)
        except Exception as e:
            print(f'Diagnostic failed for {desc}: {e}')
        return

    samples = stack.sampleRegions(
        collection=points,
        properties=['MGRS_TILE', index_col],
        scale=resolution,
        tileScale=16
    )

    band_names = stack.bandNames()
    selectors = ['MGRS_TILE', index_col] + band_names.getInfo()

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


def get_bands(shapefile_path, mgrs_shp_path, bucket, file_prefix, resolution, index_col=None, split_tiles=False,
              check_dir=None, region='conus', diagnose=False):
    """
    Extract climatological data for a set of points from a local shapefile.
    """
    points_df = gpd.read_file(shapefile_path)
    bounds = tuple(points_df.total_bounds)  # (minx, miny, maxx, maxy)
    mgrs_gdf = gpd.read_file(mgrs_shp_path, bbox=bounds)

    if index_col not in points_df.columns:
        raise ValueError(f"Index column '{index_col}' not found in shapefile.")

    mgrs_tiles = points_df['MGRS_TILE'].sample(frac=1).unique()
    print(f'{len(mgrs_tiles)} MGRS tiles to process', flush=True)

    empty_tiles = 0
    pts_identified = 0

    for tile in mgrs_tiles:
        desc = f'swapstress_{tile}'

        if check_dir:
            expected_path = os.path.join(check_dir, f'{desc}.csv')
            if os.path.exists(expected_path):
                print(f'File already exists: {expected_path}. Skipping export.')
                continue

        tile_df = points_df[points_df['MGRS_TILE'] == tile]
        print(f'{desc}.csv', len(tile_df))
        if tile_df.empty:
            empty_tiles += 1
            continue
        else:
            pts_identified += len(tile_df)

        tile_points = ee.FeatureCollection(tile_df.__geo_interface__)

        mgrs_tile_gdf = mgrs_gdf[mgrs_gdf['MGRS_TILE'] == tile]
        if mgrs_tile_gdf.empty:
            print(f'Warning: MGRS tile {tile} not found in {mgrs_shp_path}. Skipping.')
            continue

        if split_tiles:

            mgrs_tile_gdf = mgrs_gdf[mgrs_gdf['MGRS_TILE'] == tile]
            if mgrs_tile_gdf.empty:
                print(f'Warning: MGRS tile {tile} not found in {mgrs_shp_path}. Skipping.')
                continue

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
                    region=region,
                    diagnose=diagnose
                )
        else:
            geo_json = mgrs_tile_gdf.geometry.iloc[0].__geo_interface__
            roi_ee_geom = ee.Geometry(geo_json)

            tile_points_bounded = tile_points.filterBounds(roi_ee_geom)

            _export_tile_data(
                roi=roi_ee_geom,
                points=tile_points_bounded,
                desc=desc,
                bucket=bucket,
                file_prefix=file_prefix,
                resolution=resolution,
                index_col=index_col,
                region=region,
                diagnose=diagnose
            )

    print(f'{pts_identified} points identified for export')
    print(f'{empty_tiles} tiles missing')


if __name__ == '__main__':
    """"""
    run_mt_mesonet_workflow = False
    run_rosetta_workflow = False
    run_gshp_workflow = False
    run_ncss_workflow = False
    run_reesh_workflow = True
    run_ismn_workflow = False

    resolution_ = 250

    home = os.path.expanduser('~')
    root_ = os.path.join(home, 'data', 'IrrigationGIS')
    gcs_bucket_ = 'wudr'

    if run_mt_mesonet_workflow:
        # TODO: re-run MT Mesonet extract with the clean file (has many extra columns right now)
        extracts_dir_ = os.path.join(root_, 'soils', 'swapstress', 'extracts', f'mt_mesonet_extracts_{resolution_}m')
        shapefile_ = os.path.join(root_, 'soils', 'soil_potential_obs', 'mt_mesonet', 'station_metadata_clean_mgrs.shp')
        index_ = 'station'
        output_prefix_ = f'swapstress/mesonet_training_data_{resolution_}m'
        mgrs_shapefile_ = os.path.join(root_, 'boundaries', 'mgrs', 'mgrs_wgs.shp')

        is_authorized()
        get_bands(shapefile_path=shapefile_,
                  mgrs_shp_path=mgrs_shapefile_,
                  bucket=gcs_bucket_,
                  file_prefix=output_prefix_,
                  resolution=resolution_,
                  index_col=index_,
                  split_tiles=False,
                  check_dir=extracts_dir_,
                  region='global')

    elif run_rosetta_workflow:
        extracts_dir_ = os.path.join(root_, 'soils', 'swapstress', 'extracts', 'conus_extracts')
        shapefile_ = os.path.join(root_, 'soils', 'gis', 'pretraining-roi-10000_mgrs.shp')
        index_ = 'site_id'
        output_prefix_ = 'swapstress/training_data'
        mgrs_shapefile_ = os.path.join(root_, 'boundaries', 'mgrs', 'mgrs_wgs.shp')

        is_authorized()
        get_bands(shapefile_path=shapefile_,
                  mgrs_shp_path=mgrs_shapefile_,
                  bucket=gcs_bucket_,
                  file_prefix=output_prefix_,
                  resolution=resolution_,
                  index_col=index_,
                  split_tiles=True,
                  check_dir=extracts_dir_)

    elif run_gshp_workflow:
        extracts_dir_ = os.path.join(root_, 'soils', 'swapstress', 'extracts', f'gshp_extracts_{resolution_}m')
        shapefile_ = os.path.join(root_, 'soils', 'soil_potential_obs', 'gshp', 'wrc_aggregated_mgrs.shp')
        index_ = 'profile_id'
        output_prefix_ = f'swapstress/gshp_training_data_{resolution_}m'
        mgrs_shapefile_ = os.path.join(root_, 'boundaries', 'mgrs', 'mgrs_world_attr.shp')

        is_authorized()
        get_bands(shapefile_path=shapefile_,
                  mgrs_shp_path=mgrs_shapefile_,
                  bucket=gcs_bucket_,
                  file_prefix=output_prefix_,
                  resolution=resolution_,
                  index_col=index_,
                  split_tiles=False,
                  diagnose=False,
                  check_dir=extracts_dir_,
                  region='global')

    elif run_ncss_workflow:
        extracts_dir_ = os.path.join(root_, 'soils', 'swapstress', 'extracts', f'ncss_extracts_{resolution_}m')
        shapefile_ = os.path.join(root_, 'soils', 'soil_potential_obs', 'ncss_labdatasqlite', 'ncss_profiles.shp')
        # out_shp = os.path.join(root_, 'soils', 'soil_potential_obs', 'ncss_labdatasqlite',  'ncss_profiles.shp')
        index_ = 'profile_id'
        output_prefix_ = f'swapstress/ncss_training_data_{resolution_}m'
        mgrs_shapefile_ = os.path.join(root_, 'boundaries', 'mgrs', 'mgrs_world_attr.shp')

        is_authorized()
        get_bands(shapefile_path=shapefile_,
                  mgrs_shp_path=mgrs_shapefile_,
                  bucket=gcs_bucket_,
                  file_prefix=output_prefix_,
                  resolution=resolution_,
                  index_col=index_,
                  split_tiles=False,
                  diagnose=False,
                  check_dir=extracts_dir_,
                  region='global')

    elif run_reesh_workflow:
        extracts_dir_ = os.path.join(root_, 'soils', 'swapstress', 'extracts', f'reesh_extracts_{resolution_}m')
        shapefile_ = os.path.join(root_, 'soils', 'soil_potential_obs', 'reesh', 'shapefile', 'reesh_sites_mgrs.shp')
        index_ = 'site_id'
        output_prefix_ = f'swapstress/reesh_training_data_{resolution_}m'
        mgrs_shapefile_ = os.path.join(root_, 'boundaries', 'mgrs', 'mgrs_world_attr.shp')

        is_authorized()
        get_bands(shapefile_path=shapefile_,
                  mgrs_shp_path=mgrs_shapefile_,
                  bucket=gcs_bucket_,
                  file_prefix=output_prefix_,
                  resolution=resolution_,
                  index_col=index_,
                  split_tiles=False,
                  diagnose=False,
                  check_dir=extracts_dir_,
                  region='global')

    elif run_ismn_workflow:
        extracts_dir_ = os.path.join(root_, 'sfofifls', 'swapstress', 'extracts', f'ismn_extracts_{resolution_}m')
        shapefile_ = os.path.join(root_, 'soils', 'vwc_timeseries', 'ismn', 'ismn_stations_mgrs.shp')
        index_ = 'station_ui'
        output_prefix_ = f'swapstress/ismn_training_data_{resolution_}m'
        mgrs_shapefile_ = os.path.join(root_, 'boundaries', 'mgrs', 'mgrs_world_attr.shp')

        is_authorized()
        get_bands(shapefile_path=shapefile_,
                  mgrs_shp_path=mgrs_shapefile_,
                  bucket=gcs_bucket_,
                  file_prefix=output_prefix_,
                  resolution=resolution_,
                  index_col=index_,
                  split_tiles=False,
                  diagnose=False,
                  check_dir=extracts_dir_,
                  region='global')
# ========================= EOF ====================================================================
