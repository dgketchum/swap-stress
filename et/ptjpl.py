import csv
import os
import sys
import time
from subprocess import Popen, PIPE, run

import ee
import geopandas as gpd
from tqdm import tqdm

from openet import ptjpl
from et.et_utils import is_authorized, get_lanid

sys.path.insert(0, os.path.abspath('..'))
sys.setrecursionlimit(5000)

LANDSAT_COLLECTIONS = ['LANDSAT/LT04/C02/T1_L2',
                       'LANDSAT/LT05/C02/T1_L2',
                       'LANDSAT/LE07/C02/T1_L2',
                       'LANDSAT/LC08/C02/T1_L2',
                       'LANDSAT/LC09/C02/T1_L2']

EE = '/home/dgketchum/miniconda/envs/swim/bin/earthengine'
GS = '/home/dgketchum/miniconda/envs/swim/bin/gsutil'

# Irrigation mask sources and regions
IRR = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'
STATES = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']
WEST_STATES = 'users/dgketchum/boundaries/western_11_union'
EAST_STATES = 'users/dgketchum/boundaries/eastern_38_dissolved'

FLUX_3PIX = 'projects/ee-dgketchum/assets/swim/flux_footprints_3p'


def list_gcs_bucket_contents(gcs_path: str) -> list[str]:
    command = [GS, 'ls', gcs_path]
    result = run(command, capture_output=True, text=True, encoding='utf-8')
    if result.returncode == 0 and result.stdout:
        return [line for line in result.stdout.strip().split('\n') if line]
    else:
        return []


def list_assets(location):
    command = 'ls'
    cmd = ['{}'.format(EE), '{}'.format(command), '{}'.format(location)]
    asset_list = Popen(cmd, stdout=PIPE)
    stdout, stderr = asset_list.communicate()
    reader = csv.DictReader(stdout.decode('ascii').splitlines(),
                            delimiter=' ', skipinitialspace=True,
                            fieldnames=['name'])
    assets = [x['name'] for x in reader]
    assets = [x for x in assets if 'Running' not in x]
    return assets


def export_et_fraction(shapefile, bucket, feature_id='FID', select=None, start_yr=2000, end_yr=2024,
                       overwrite=False, check_dir=None, buffer=False):
    """"""
    df = gpd.read_file(shapefile)
    df = df.set_index(feature_id, drop=False)
    df = df.sort_index(ascending=False)

    if buffer:
        df.geometry = df.geometry.buffer(buffer)

    original_crs = df.crs
    if original_crs and not original_crs.srs == 'EPSG:4326':
        df = df.to_crs(4326)

    skipped, exported, existing_images = 0, 0, []

    for fid, row in df.iterrows():

        if row['geometry'].geom_type == 'Point':
            raise ValueError
        elif row['geometry'].geom_type == 'Polygon':
            polygon = ee.Geometry(row.geometry.__geo_interface__)
        else:
            raise ValueError

        if select is not None and fid not in select:
            continue

        coll = ptjpl.Collection(LANDSAT_COLLECTIONS, start_date=f'{start_yr}-01-01',
                                       end_date=f'{end_yr}-12-31', geometry=polygon,
                                       cloud_cover_max=70)

        scenes = coll.get_image_ids()
        scenes = list(set(scenes))
        scenes = sorted(scenes, key=lambda item: item.split('_')[-1])

        if not overwrite:
            dst = os.path.join(f"gs://{bucket}", 'ptjpl', f'{fid}')
            existing = list_gcs_bucket_contents(dst)
            existing_images = [os.path.basename(i) for i in existing]

        with tqdm(scenes, desc=f'Export PTJPL for {fid}', total=len(scenes)) as pbar:
            for img_id in scenes:
                pbar.set_description(f'Export PTJPL for {fid} (Processing: {img_id})')
                splt = img_id.split('/')
                splt = splt[-1].split('_')
                _name = '_'.join(splt[-3:])

                desc = os.path.join('ptjpl', fid, _name)

                if not overwrite and check_dir is not None:
                    target_file = os.path.join(check_dir, fid, f'{_name}.tif')
                    if os.path.isfile(target_file):
                        continue

                if len(existing_images) > 0 and f'{_name}.tif' in existing_images:
                    continue

                ptjpl_kwargs = dict(ta_source='ERA5LAND',
                                    ea_source='ERA5LAND',
                                    windspeed_source='ERA5LAND',
                                    rs_source='ERA5LAND',
                                    LWin_source='ERA5LAND')

                ptjpl_img = ptjpl.Image.from_landsat_c2_sr(img_id,
                                                                  et_reference_source='ERA5LAND',
                                                                  et_reference_band='eto',
                                                                  et_reference_factor=1.0,
                                                                  et_reference_resample='bilinear',
                                                                  **ptjpl_kwargs
                                                                  )
                etf = ptjpl_img.et_fraction

                try:
                    proj = etf.select('et_fraction').getInfo()['bands'][0]
                except ee.ee_exception.EEException as e:
                    print(f'{_name} returned error {e}')
                    continue

                crs = proj['crs']
                crs_transform = proj['crs_transform']

                etf = etf.clip(polygon)

                task = ee.batch.Export.image.toCloudStorage(
                    etf,
                    description=_name,
                    bucket=bucket,
                    fileNamePrefix=desc,
                    crs=crs,
                    crsTransform=crs_transform,
                    region=polygon,
                    scale=30,
                    maxPixels=1e13,
                    formatOptions={'noData': 0.0}
                )

                try:
                    task.start()
                    print(desc)
                except ee.ee_exception.EEException as e:
                    print('{}, waiting on '.format(e), _name, '......')
                    time.sleep(600)
                    task.start()

                exported += 1


def _export_chunked_tables(bands, selectors, polygon, fid, desc, bucket, part, feature_id, scale=30, chunk_size=50):
    fc = ee.FeatureCollection(ee.Feature(polygon, {feature_id: fid}))
    chunk_band_names = selectors[1:]
    img_chunk = bands.select(chunk_band_names)
    data_chunk = img_chunk.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=scale)
    chunk_desc = f"{desc}_chunk_{part:03d}"
    fn_prefix = os.path.join('ptjpl_tables', 'chunked', str(fid), chunk_desc)
    task = ee.batch.Export.table.toCloudStorage(
        data_chunk,
        description=chunk_desc,
        bucket=bucket,
        fileNamePrefix=fn_prefix,
        fileFormat='CSV',
        selectors=[feature_id] + chunk_band_names,
    )
    try:
        task.start()
        print(chunk_desc)
    except ee.ee_exception.EEException as e:
        if "many tasks already in the queue" in str(e):
            time.sleep(600)
            task.start()
        else:
            raise


def export_ptjpl_zonal_stats(shapefile, bucket, feature_id='FID', polygon_asset=None,
                             select=None, start_yr=2000, end_yr=2024, chunk=False, chunk_size=50,
                             mask_type='irr', check_dir=None, state_col='state', buffer=False):
    fc = None
    df = gpd.read_file(shapefile)
    df = df.set_index(feature_id, drop=False)

    if buffer:
        df.geometry = df.geometry.buffer(buffer)

    original_crs = df.crs
    if original_crs and not original_crs.srs == 'EPSG:4326':
        df = df.to_crs(4326)

    irr_coll = ee.ImageCollection(IRR)
    s, e = '1987-01-01', '2024-12-31'
    remap = irr_coll.filterDate(s, e).select('classification').map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gte(5)
    east = ee.FeatureCollection(EAST_STATES)
    lanid = get_lanid()

    for fid, row in tqdm(df.iterrows(), desc='Export PTJPL zonal stats', total=df.shape[0]):
        if row['geometry'].geom_type == 'Point':
            raise ValueError

        elif row['geometry'].geom_type == 'Polygon':
            if polygon_asset:
                "reduce request size, avoid e.g. 'Payload size limit exceeded'"
                fc = ee.FeatureCollection(polygon_asset).filterMetadata(feature_id, 'equals', fid)
                polygon = fc.geometry()
            else:
                polygon = ee.Geometry(row.geometry.__geo_interface__)
                fc = ee.FeatureCollection(ee.Feature(polygon, {feature_id: fid}))

        else:
            continue

        if select is not None and fid not in select:
            continue

        for year in range(start_yr, end_yr + 1):

            # Use fid-specific subdirectory similar to export_et_fraction
            desc = f'ptjpl_etf_{fid}_{mask_type}_{year}'
            fn_prefix = os.path.join('ptjpl_tables', mask_type, str(fid), desc)

            if check_dir:
                f = os.path.join(check_dir, mask_type, str(fid), f'{desc}.csv')
                if os.path.exists(f):
                    # print(f'{f} exists, skipping')
                    continue

            if mask_type in ['irr', 'inv_irr']:
                if state_col in row and row[state_col] in STATES:
                    irr = irr_coll.filterDate(f'{year}-01-01', f'{year}-12-31').select('classification').mosaic()
                    irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))
                else:
                    irr_mask = lanid.select(f'irr_{year}').clip(east)
                    irr = ee.Image(1).subtract(irr_mask)
            else:
                irr, irr_mask = None, None

            coll = ptjpl.Collection(
                LANDSAT_COLLECTIONS,
                start_date=f'{year}-01-01',
                end_date=f'{year}-12-31',
                geometry=polygon,
                cloud_cover_max=70,
            )
            scenes = coll.get_image_ids()
            scenes = list(set(scenes))
            scenes = sorted(scenes, key=lambda item: item.split('_')[-1])

            first, bands, band_ct, chunk_ct = True, [], 0, -1
            selectors = [feature_id]

            for img_id in scenes:
                splt = img_id.split('/')[-1].split('_')
                _name = '_'.join(splt[-3:])
                selectors.append(_name)

                ptjpl_kwargs = dict(
                    ta_source='ERA5LAND',
                    ea_source='ERA5LAND',
                    windspeed_source='ERA5LAND',
                    rs_source='ERA5LAND',
                    LWin_source='ERA5LAND',
                )
                ptjpl_img = ptjpl.Image.from_landsat_c2_sr(
                    img_id,
                    et_reference_source='ERA5LAND',
                    et_reference_band='eto',
                    et_reference_factor=1.0,
                    et_reference_resample='bilinear',
                    **ptjpl_kwargs,
                )
                etf_img = ptjpl_img.et_fraction.rename(_name)

                if mask_type == 'no_mask':
                    etf_img = etf_img.clip(polygon)
                elif mask_type == 'irr':
                    etf_img = etf_img.clip(polygon).mask(irr_mask)
                elif mask_type == 'inv_irr':
                    etf_img = etf_img.clip(polygon).mask(irr.gt(0))

                if first:
                    bands = etf_img
                    first = False
                else:
                    bands = bands.addBands([etf_img])

                band_ct += 1

                if chunk and band_ct == chunk_size:
                    chunk_ct += 1
                    _export_chunked_tables(bands, selectors, polygon, fid, desc, bucket, chunk_ct,
                                           feature_id, scale=30, chunk_size=chunk_size)
                    band_ct = 0
                    bands = None
                    first = True
                    selectors = [feature_id]

            if bands is None:
                continue

            if chunk:
                chunk_ct += 1
                _export_chunked_tables(bands, selectors, polygon, fid, desc, bucket, chunk_ct,
                                       feature_id, scale=30, chunk_size=chunk_size)
                continue

            data = bands.reduceRegions(collection=fc, reducer=ee.Reducer.mean(),
                                       scale=30)

            task = ee.batch.Export.table.toCloudStorage(
                data,
                description=desc,
                bucket=bucket,
                fileNamePrefix=fn_prefix,
                fileFormat='CSV',
                selectors=selectors,
            )

            try:
                task.start()
                print(desc)
            except ee.ee_exception.EEException as e:
                error_message = str(e)

                if "payload size exceeds the limit" in error_message:
                    print(error_message)
                    # skip this year/feature; consider enabling chunking to reduce payload size
                    pass

                elif "many tasks already in the queue" in error_message:
                    print(f"Task queue full. Waiting 10 minutes to retry {desc}...")
                    time.sleep(600)
                    try:
                        task.start()
                        print(desc)
                    except Exception as e2:
                        print(f"Retry failed for {desc}: {e2}")

                elif "already started with the given request_id" in error_message:
                    # Task with same request/description has already been enqueued; treat as success and continue
                    print(f"{desc}: already started, skipping duplicate start.")
                    continue

                else:
                    raise


if __name__ == '__main__':
    pass
# ========================= EOF =======================================================================================
