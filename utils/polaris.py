import os
import re
from glob import glob
from typing import Optional, Sequence

import numpy as np
import pandas as pd

import ee
import geopandas as gpd
from shapely.geometry import box  # likely error: unused import

from retention_curve import parse_polaris_depth_from_asset, map_polaris_depth_range_to_rosetta_level


# --------------------------- Concatenate station layers ---------------------------

def _parse_depth_from_path(fp):
    m = re.search(r"_(\d+)_(\d+)(?:\.|$)", os.path.basename(fp))
    assert m is not None
    dmin = float(m.group(1))
    dmax = float(m.group(2))
    return dmin, dmax


def concat_polaris_stations(in_dirs, out_file):
    files = []
    for d in in_dirs:
        files.extend(glob(os.path.join(d, '**', '*.*'), recursive=True))
    files.reverse()
    exts = {'.parquet', '.pq', '.csv'}
    rows = []

    for fp in files:
        ext = os.path.splitext(fp)[1].lower()
        if ext not in exts:
            continue
        if ext in ('.parquet', '.pq'):
            df = pd.read_parquet(fp)
        else:
            df = pd.read_csv(fp)

        df = df.copy()
        if 'station' not in df.columns:
            df = df.rename(columns={'site_id': 'station'})
        if 'station' not in df.columns:
            continue  # likely error: missing station identifier

        df['station'] = df['station'].astype(str).str.lower().str.replace('_', '-', regex=False)

        pat = re.compile(r'^polaris_(alpha|n|theta_r|theta_s)_(\d+)_(\d+)cm$')
        matches = []
        for c in df.columns:
            m = pat.match(c)
            if m:
                var = m.group(1)
                d0 = int(m.group(2))
                d1 = int(m.group(3))
                matches.append((c, var, d0, d1))
        if not matches:
            continue

        buckets = {}
        name_map = {'alpha': 'alpha_mean', 'n': 'n_mean', 'theta_r': 'theta_r_mean', 'theta_s': 'theta_s_mean'}
        for col, var, d0, d1 in matches:
            lvl = map_polaris_depth_range_to_rosetta_level(d0, d1)
            tmp = pd.DataFrame({
                'station': df['station'],
                'depth_min_cm': float(d0),
                'depth_max_cm': float(d1),
                'rosetta_level': lvl,
                name_map[var]: df[col]
            })
            key = (d0, d1, lvl)
            if key in buckets:
                buckets[key] = buckets[key].merge(tmp, on=['station', 'depth_min_cm', 'depth_max_cm', 'rosetta_level'],
                                                  how='outer')
            else:
                buckets[key] = tmp

        for _, frame in buckets.items():
            rows.append(frame)

    if not rows:
        return pd.DataFrame(columns=['station', 'depth_min_cm', 'depth_max_cm', 'rosetta_level',
                                     'alpha_mean', 'n_mean', 'theta_r_mean', 'theta_s_mean'])

    cat = pd.concat(rows, ignore_index=True)
    cat = cat.replace([np.inf, -np.inf], np.nan)
    cat = cat.dropna(subset=['station'])
    cat = cat.groupby(['station', 'rosetta_level', 'depth_min_cm', 'depth_max_cm'], as_index=False).mean()

    out_dir = os.path.dirname(out_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cat.to_parquet(out_file)
    return cat


# --------------------------- Concatenate GSHP layers ---------------------------

def concat_polaris_gshp(in_dirs, out_file):
    files = []
    dirs = in_dirs if isinstance(in_dirs, (list, tuple)) else [in_dirs]
    for d in dirs:
        files.extend(glob(os.path.join(d, '**', '*.*'), recursive=True))
    files.reverse()
    exts = {'.parquet', '.pq', '.csv'}
    rows = []

    for fp in files:
        ext = os.path.splitext(fp)[1].lower()
        if ext not in exts:
            continue
        if ext in ('.parquet', '.pq'):
            df = pd.read_parquet(fp)
        else:
            df = pd.read_csv(fp)
        if df.empty:
            os.remove(fp)
            print(f'{os.path.basename(fp)} empty')

        df = df.copy()
        if 'profile_id' not in df.columns:
            if 'site_id' in df.columns:
                df = df.rename(columns={'site_id': 'profile_id'})
            else:
                continue  # likely error: missing profile identifier

        pat = re.compile(r'^polaris_(alpha|n|theta_r|theta_s)_(\d+)_(\d+)cm$')
        matches = []
        for c in df.columns:
            m = pat.match(c)
            if m:
                var = m.group(1)
                d0 = int(m.group(2))
                d1 = int(m.group(3))
                matches.append((c, var, d0, d1))
        if not matches:
            continue

        buckets = {}
        name_map = {'alpha': 'alpha_mean', 'n': 'n_mean', 'theta_r': 'theta_r_mean', 'theta_s': 'theta_s_mean'}
        for col, var, d0, d1 in matches:
            lvl = map_polaris_depth_range_to_rosetta_level(d0, d1)
            tmp = pd.DataFrame({
                'profile_id': df['profile_id'],
                'depth_min_cm': float(d0),
                'depth_max_cm': float(d1),
                'rosetta_level': lvl,
                name_map[var]: df[col]
            })
            key = (d0, d1, lvl)
            if key in buckets:
                buckets[key] = buckets[key].merge(
                    tmp,
                    on=['profile_id', 'depth_min_cm', 'depth_max_cm', 'rosetta_level'],
                    how='outer'
                )
            else:
                buckets[key] = tmp

        for _, frame in buckets.items():
            rows.append(frame)

    if not rows:
        return pd.DataFrame(columns=['profile_id', 'depth_min_cm', 'depth_max_cm', 'rosetta_level',
                                     'alpha_mean', 'n_mean', 'theta_r_mean', 'theta_s_mean'])

    cat = pd.concat(rows, ignore_index=True)
    cat = cat.replace([np.inf, -np.inf], np.nan)
    cat = cat.dropna(subset=['profile_id'])
    cat = cat.groupby(['profile_id', 'rosetta_level', 'depth_min_cm', 'depth_max_cm'], as_index=False).mean()

    out_dir = os.path.dirname(out_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cat['profile_id'] = cat['profile_id'].astype(str)
    cat.to_parquet(out_file)
    return cat


# --------------------------- Earth Engine export ---------------------------

def is_authorized(project: Optional[str] = 'ee-dgketchum') -> None:
    try:
        ee.Initialize(project=project)
        print('Authorized')
    except Exception as e:
        print(f'You are not authorized: {e}')
        raise


def _polaris_all_depths_image() -> ee.Image:
    root = 'projects/sat-io/open-datasets/polaris'

    vars_ = [
        'bd', 'clay', 'ksat', 'n', 'om', 'ph', 'sand', 'silt',
        'theta_r', 'theta_s', 'lambda', 'hb', 'alpha'
    ]
    depths = [
        ('0_5', '0_5cm'),
        ('5_15', '5_15cm'),
        ('15_30', '15_30cm'),
        ('30_60', '30_60cm'),
        ('60_100', '60_100cm'),
        ('100_200', '100_200cm'),
    ]

    images = []
    for var in vars_:
        var_dir = f'{root}/{var}_mean'
        for depth_code, depth_label in depths:
            asset_id = f'{var_dir}/{var}_{depth_code}'
            img = ee.Image(asset_id).rename(f'polaris_{var}_{depth_label}')
            images.append(img)

    return ee.Image.cat(images)


def _export_tile_data_polaris(
        roi: ee.Geometry,
        points: ee.FeatureCollection,
        desc: str,
        bucket: str,
        file_prefix: str,
        resolution: int,
        index_col: str,
        diagnose: bool = False,
) -> None:
    stack = _polaris_all_depths_image().clip(roi)

    if points.size().eq(0).getInfo():
        print(f'{desc}: no points to sample, skipping.')
        return

    if diagnose:
        try:
            print('diagnose', desc)
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
        tileScale=16,
    )

    band_names = stack.bandNames()
    selectors = ['MGRS_TILE', index_col] + band_names.getInfo()

    task = ee.batch.Export.table.toCloudStorage(
        samples,
        description=desc,
        bucket=bucket,
        fileNamePrefix=f'{file_prefix}/{desc}',
        fileFormat='CSV',
        selectors=selectors,
    )
    task.start()
    print(f'Started export: {file_prefix}/{desc} (task: {task.id})')


def export_polaris_by_mgrs(
        shapefile_path: str,
        mgrs_shp_path: str,
        bucket: str,
        file_prefix: str,
        resolution: int,
        index_col: str,
        check_dir: Optional[str] = None,
        diagnose: bool = False,
        tile_subset: Optional[Sequence[str]] = None,
) -> None:
    points_df = gpd.read_file(shapefile_path)
    mgrs_gdf = gpd.read_file(mgrs_shp_path)

    if index_col not in points_df.columns:
        raise ValueError(f"Index column '{index_col}' not found in shapefile.")
    if 'MGRS_TILE' not in points_df.columns:
        raise ValueError("Shapefile must contain a 'MGRS_TILE' column.")

    mgrs_tiles = points_df['MGRS_TILE'].unique()
    if tile_subset:
        mgrs_tiles = [t for t in mgrs_tiles if t in set(tile_subset)]

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

        geo_json = mgrs_tile_gdf.geometry.iloc[0].__geo_interface__
        roi_ee_geom = ee.Geometry(geo_json)

        tile_points_bounded = tile_points.filterBounds(roi_ee_geom)

        _export_tile_data_polaris(
            roi=roi_ee_geom,
            points=tile_points_bounded,
            desc=desc,
            bucket=bucket,
            file_prefix=file_prefix,
            resolution=resolution,
            index_col=index_col,
            diagnose=diagnose,
        )


if __name__ == '__main__':
    run_mt_mesonet_export = False
    run_reesh_export = False
    run_gshp_export = False
    run_concat = False
    run_concat_gshp = True

    resolution_ = 250
    home_ = os.path.expanduser('~')
    root_ = os.path.join(home_, 'data', 'IrrigationGIS')
    gcs_bucket_ = 'wudr'

    if run_mt_mesonet_export or run_reesh_export or run_gshp_export:
        is_authorized()

    if run_mt_mesonet_export:
        extracts_dir_ = os.path.join(
            root_, 'soils', 'swapstress', 'extracts', f'mt_mesonet_polaris_all_depths_{resolution_}m'
        )
        shapefile_ = os.path.join(
            root_, 'soils', 'soil_potential_obs', 'mt_mesonet', 'station_metadata_clean_mgrs.shp'
        )
        index_ = 'station'
        output_prefix_ = f'swapstress/polaris/mesonet_training_all_depths_{resolution_}m'
        mgrs_shapefile_ = os.path.join(root_, 'boundaries', 'mgrs', 'mgrs_wgs.shp')

        export_polaris_by_mgrs(
            shapefile_path=shapefile_,
            mgrs_shp_path=mgrs_shapefile_,
            bucket=gcs_bucket_,
            file_prefix=output_prefix_,
            resolution=resolution_,
            index_col=index_,
            check_dir=extracts_dir_,
            diagnose=False,
        )

    if run_reesh_export:
        extracts_dir_ = os.path.join(
            root_, 'soils', 'swapstress', 'extracts', f'reesh_polaris_all_depths_{resolution_}m'
        )
        shapefile_ = os.path.join(
            root_, 'soils', 'soil_potential_obs', 'reesh', 'shapefile', 'reesh_sites_mgrs.shp'
        )
        index_ = 'site_id'
        output_prefix_ = f'swapstress/polaris/reesh_training_all_depths_{resolution_}m'
        mgrs_shapefile_ = os.path.join(root_, 'boundaries', 'mgrs', 'mgrs_world_attr.shp')

        export_polaris_by_mgrs(
            shapefile_path=shapefile_,
            mgrs_shp_path=mgrs_shapefile_,
            bucket=gcs_bucket_,
            file_prefix=output_prefix_,
            resolution=resolution_,
            index_col=index_,
            check_dir=extracts_dir_,
            diagnose=False,
        )

    if run_gshp_export:
        extracts_dir_ = os.path.join(
            root_, 'soils', 'swapstress', 'extracts', f'gshp_polaris_all_depths_{resolution_}m'
        )
        shapefile_ = os.path.join(
            root_, 'soils', 'soil_potential_obs', 'gshp', 'wrc_aggregated_mgrs.shp'
        )
        index_ = 'profile_id'
        output_prefix_ = f'swapstress/polaris/gshp_training_all_depths_{resolution_}m'
        mgrs_shapefile_ = os.path.join(root_, 'boundaries', 'mgrs', 'mgrs_world_attr.shp')

        export_polaris_by_mgrs(
            shapefile_path=shapefile_,
            mgrs_shp_path=mgrs_shapefile_,
            bucket=gcs_bucket_,
            file_prefix=output_prefix_,
            resolution=resolution_,
            index_col=index_,
            check_dir=extracts_dir_,
            diagnose=False,
        )

    if run_concat:
        mt_dir_ = os.path.join(root_, 'soils', 'swapstress', 'extracts',
                               f'mt_mesonet_polaris_all_depths_{resolution_}m')
        reesh_dir_ = os.path.join(root_, 'soils', 'swapstress', 'extracts', f'reesh_polaris_all_depths_{resolution_}m')
        out_file_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'polaris', 'polaris_stations.parquet')
        concat_polaris_stations([mt_dir_, reesh_dir_], out_file_)

    if run_concat_gshp:
        gshp_dir_ = os.path.join(root_, 'soils', 'swapstress', 'extracts', f'gshp_polaris_all_depths_{resolution_}m')
        out_gshp_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'polaris', 'polaris_gshp.parquet')
        concat_polaris_gshp(gshp_dir_, out_gshp_)

# ========================= EOF ====================================================================
