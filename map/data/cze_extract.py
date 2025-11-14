import os
import time
import calendar
from typing import List, Dict, Any, Tuple

import ee
import geopandas as gpd
import multiprocessing as mp
from multiprocessing.pool import ThreadPool

from ee.ee_exception import EEException

try:
    # Optional, only needed for ERA5 ETo
    from openet.refetgee import Daily as RefETDaily
except Exception:
    RefETDaily = None


def _execute_with_backoff(func, *, tries: int = 6, delay: float = 1.0,
                          backoff: float = 2.0, exceptions: Tuple[type[BaseException], ...] = (Exception,),
                          fallback=None):
    """Execute callable with exponential backoff; return fallback on exhaustion."""
    wait = delay
    for attempt in range(int(tries)):
        try:
            return func()
        except exceptions as exc:  # pragma: no cover - utility guard
            if attempt == int(tries) - 1:
                if fallback is not None:
                    return fallback
                raise
            time.sleep(wait)
            wait = min(wait * backoff, 60.0)


_BIG_DATA_ERROR_SUBSTRINGS = (
    'response too large',
    'payload is too large',
    'more than 50000 elements',
    'exceeds the allowed amount',
)


def _fetch_features_from_expression(expression: ee.FeatureCollection,
                                     use_big_data_api: bool = False,
                                     page_size: int = 5000) -> List[Dict[str, Any]]:
    """Fetch features either via getInfo or the Big Data API with pagination."""

    def _getinfo() -> List[Dict[str, Any]]:
        data = expression.getInfo()
        if isinstance(data, dict):
            return data.get('features', []) or []
        return []

    def _compute_features() -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {'expression': expression, 'pageSize': page_size}
        features: List[Dict[str, Any]] = []
        next_token: str | None = None
        while True:
            if next_token:
                params['pageToken'] = next_token
            resp = ee.data.computeFeatures(params)
            features.extend(resp.get('features', []) or [])
            next_token = resp.get('next_page_token')
            if not next_token:
                break
        return features

    if use_big_data_api:
        return _compute_features()

    try:
        return _getinfo()
    except EEException as exc:
        message = str(exc).lower()
        if any(token in message for token in _BIG_DATA_ERROR_SUBSTRINGS):
            return _compute_features()
        raise


def _build_monthly_era5_image(year: int, month: int) -> Tuple[ee.Image | None, List[str]]:
    """Assemble a multiband ERA5-Land image for the given month."""
    month_start = ee.Date.fromYMD(year, month, 1)
    days_in_month = calendar.monthrange(year, month)[1]
    selectors: List[str] = []
    monthly_images: List[ee.Image] = []
    for day in range(1, days_in_month + 1):
        day_start = ee.Date.fromYMD(year, month, day)
        day_end = day_start.advance(1, 'day')
        day_str = f'{year}{str(month).zfill(2)}{str(day).zfill(2)}'
        coll_day = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY').filterDate(day_start, day_end)

        swe = coll_day.select('snow_depth_water_equivalent').mean().multiply(1000).rename(f'swe_{day_str}')
        swe = swe.set('system:time_start', day_start.millis())

        if RefETDaily is not None:
            eto = RefETDaily.era5_land(coll_day).etr.rename(f'eto_{day_str}')
            eto = eto.set('system:time_start', day_start.millis())
        else:
            eto = ee.Image(0).rename(f'eto_{day_str}')

        t_hourly = coll_day.select('temperature_2m')
        tmean = t_hourly.mean().subtract(273.15).rename(f'tmean_{day_str}')
        tmin = t_hourly.min().subtract(273.15).rename(f'tmin_{day_str}')
        tmax = t_hourly.max().subtract(273.15).rename(f'tmax_{day_str}')
        tmean = tmean.set('system:time_start', day_start.millis())
        tmin = tmin.set('system:time_start', day_start.millis())
        tmax = tmax.set('system:time_start', day_start.millis())

        precip = coll_day.select('total_precipitation_hourly').sum().multiply(1000).rename(f'precip_{day_str}')
        precip = precip.set('system:time_start', day_start.millis())

        srad = coll_day.select('surface_solar_radiation_downwards_hourly').mean().rename(f'srad_{day_str}')
        srad = srad.set('system:time_start', day_start.millis())

        for nm in [f'swe_{day_str}', f'eto_{day_str}', f'tmean_{day_str}', f'tmin_{day_str}',
                   f'tmax_{day_str}', f'precip_{day_str}', f'srad_{day_str}']:
            selectors.append(nm)

        monthly_images.extend([swe, eto, tmean, tmin, tmax, precip, srad])

    if not monthly_images:
        return None, selectors

    combined = monthly_images[0]
    for img in monthly_images[1:]:
        combined = combined.addBands(img)
    return combined, selectors


def _era5_chunk_worker(args: Tuple[ee.Image, ee.List, int, int, float, str, bool]) -> List[Dict[str, Any]]:
    """Reduce ERA5 monthly image over a slice of features."""
    monthly_image, feats_list, start_idx, end_idx, scale, id_col, use_big_data_api = args
    sub_fc = ee.FeatureCollection(feats_list.slice(start_idx, end_idx))
    expression = monthly_image.reduceRegions(
        collection=sub_fc,
        reducer=ee.Reducer.mean(),
        scale=scale,
    )

    def _fetch_rows() -> List[Dict[str, Any]]:
        features = _fetch_features_from_expression(expression, use_big_data_api=use_big_data_api)
        rows_local: List[Dict[str, Any]] = []
        for feat in features:
            props = feat.get('properties', {}) or {}
            if id_col not in props:
                props[id_col] = None
            rows_local.append(props)
        return rows_local

    return _execute_with_backoff(
        _fetch_rows,
        tries=6,
        delay=1.0,
        backoff=2.0,
        exceptions=(Exception,),
        fallback=[],
    )


def ee_init(project: str = 'ee-dgketchum', high_volume: bool = False) -> None:
    """Initialize the Earth Engine client for a given project.

    Parameters
    - project: GEE project id to use for auth.
    - high_volume: if True, use the high-volume API endpoint for parallel requests.
    """
    try:
        if high_volume:
            ee.Initialize(project=project, opt_url='https://earthengine-highvolume.googleapis.com')
        else:
            ee.Initialize(project=project)
        print('Earth Engine initialized')
    except Exception as e:
        raise RuntimeError(f'Failed to initialize Earth Engine: {e}')


def build_buffered_fc_from_points(shapefile: str, id_col: str = 'profile_id',
                                  buffer_m: float = 250.0) -> ee.FeatureCollection:
    """Build an ee.FeatureCollection of 250 m buffers around point features.

    Parameters
    - shapefile: path to input point shapefile (any CRS; reprojected to EPSG:4326).
    - id_col: attribute to carry through as identifier per feature.
    - buffer_m: buffer radius in meters.

    Returns
    - ee.FeatureCollection with buffered geometries and `{id_col}` property.
    """
    gdf = gpd.read_file(shapefile)
    if gdf.crs is None:
        raise ValueError('Input shapefile CRS is undefined')
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    features: List[ee.Feature] = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        if geom.geom_type.lower() != 'point':
            # Fallback: take centroid if not a point
            lon, lat = geom.centroid.x, geom.centroid.y
        else:
            lon, lat = geom.x, geom.y
        props = {id_col: row[id_col] if id_col in row and row[id_col] is not None else None}
        ee_geom = ee.Geometry.Point([float(lon), float(lat)]).buffer(float(buffer_m))
        features.append(ee.Feature(ee_geom, props))

    if not features:
        raise ValueError('No features were built from shapefile')

    return ee.FeatureCollection(features)


def export_era5_land_over_buffers(
        feature_coll: ee.FeatureCollection,
        id_col: str = 'profile_id',
        start_yr: int = 2000,
        end_yr: int = 2024,
        debug: bool = False,
        direct_out_dir: str | None = None,
        project: str = 'ee-dgketchum',
        high_volume: bool = True,
        chunk_size: int = 200,
        workers: int = 8,
        use_big_data_api: bool = True,
) -> None:
    """Write daily ERA5-Land variables reduced over buffered features by month.

    Saves local CSVs per month to `direct_out_dir` with filenames like
    `era5_vars_YYYY_MM.csv` containing columns: `{id_col}`, daily SWE, ETo,
    Tmean/Tmin/Tmax (C), precipitation (mm), and shortwave radiation (W/m^2).
    """
    scale_era5 = 11132  # ~10 km at equator; consistent with reference workflows

    if direct_out_dir is None:
        raise ValueError("Provide 'direct_out_dir' for ERA5 direct export.")

    try:
        import pandas as _pd
    except Exception as e:
        raise RuntimeError('pandas is required for direct ERA5 export') from e

    # Ensure high-volume endpoint init for direct requests
    try:
        if high_volume:
            ee.Initialize(project=project, opt_url='https://earthengine-highvolume.googleapis.com')
        else:
            ee.Initialize(project=project)
    except Exception:
        pass

    try:
        n_feat = int(feature_coll.size().getInfo())
    except Exception as e:
        raise RuntimeError(f'Failed to get feature count: {e}')

    if n_feat == 0:
        print('No features available for ERA5 export')
        return

    feats_list = feature_coll.toList(n_feat)
    dtimes = [(y, m) for y in range(start_yr, end_yr + 1) for m in range(1, 13)]

    pool: ThreadPool | None = ThreadPool(processes=max(1, int(workers))) if workers and workers > 1 else None

    try:
        for year, month in dtimes:
            desc = f'era5_vars_{year}_{str(month).zfill(2)}'
            out_fp = os.path.join(direct_out_dir, f'{desc}.csv')
            if os.path.exists(out_fp):
                if debug:
                    print('Exists, skipping:', out_fp)
                continue

            try:
                monthly_bands_image, selectors = _build_monthly_era5_image(year, month)
            except Exception as e:
                print(f'ERA5 {year}-{str(month).zfill(2)} build failed: {e}')
                continue

            if monthly_bands_image is None:
                print(f'ERA5 {year}-{str(month).zfill(2)} produced no bands, skipping')
                continue

            chunk_args = [
                (monthly_bands_image, feats_list, start, min(start + max(1, int(chunk_size)), n_feat),
                 scale_era5, id_col, use_big_data_api)
                for start in range(0, n_feat, max(1, int(chunk_size)))
            ]

            if not chunk_args:
                continue

            if pool:
                chunk_results = pool.map(_era5_chunk_worker, chunk_args)
            else:
                chunk_results = [_era5_chunk_worker(arg) for arg in chunk_args]

            rows: List[Dict[str, Any]] = []
            for chunk in chunk_results:
                if chunk:
                    rows.extend(chunk)

            if not rows:
                print('No rows for', desc)
                continue

            cols = [id_col] + selectors
            df = _pd.DataFrame(rows)
            df = df.reindex(columns=cols)
            os.makedirs(os.path.dirname(out_fp), exist_ok=True)
            df.to_csv(out_fp, index=False)
            print('Saved:', out_fp)
    finally:
        if pool:
            pool.close()
            pool.join()


def landsat_c2_sr(input_img: ee.Image) -> ee.Image:
    """Scale and mask Landsat C2 SR image; returns optical + thermal bands masked."""
    INPUT_BANDS = ee.Dictionary({
        'LANDSAT_4': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'ST_B6', 'QA_PIXEL', 'QA_RADSAT'],
        'LANDSAT_5': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'ST_B6', 'QA_PIXEL', 'QA_RADSAT'],
        'LANDSAT_7': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'ST_B6', 'QA_PIXEL', 'QA_RADSAT'],
        'LANDSAT_8': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10', 'QA_PIXEL', 'QA_RADSAT'],
        'LANDSAT_9': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10', 'QA_PIXEL', 'QA_RADSAT'],
    })
    OUTPUT_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'QA_PIXEL', 'QA_RADSAT']

    spacecraft_id = ee.String(input_img.get('SPACECRAFT_ID'))
    scaled = input_img.select(INPUT_BANDS.get(spacecraft_id), OUTPUT_BANDS) \
        .multiply([0.0000275, 0.0000275, 0.0000275, 0.0000275, 0.0000275, 0.0000275, 0.00341802, 1, 1]) \
        .add([-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, 149.0, 0, 0])

    def _mask(i: ee.Image) -> ee.Image:
        qa = i.select('QA_PIXEL')
        cloud = qa.rightShift(3).bitwiseAnd(1).neq(0)
        cloud = cloud.Or(qa.rightShift(2).bitwiseAnd(1).neq(0))
        cloud = cloud.Or(qa.rightShift(1).bitwiseAnd(1).neq(0))
        cloud = cloud.Or(qa.rightShift(4).bitwiseAnd(1).neq(0))
        cloud = cloud.Or(qa.rightShift(5).bitwiseAnd(1).neq(0))
        sat = i.select('QA_RADSAT').gt(0)
        mask = cloud.Or(sat).Not().rename('cloud_mask')
        return mask

    mask = _mask(input_img)
    # Drop QA bands from output; keep only spectral/thermal bands
    spectral = scaled.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10'])
    out = spectral.updateMask(mask).copyProperties(input_img, ['system:time_start'])
    return out


def landsat_masked(year: int, roi: ee.FeatureCollection) -> ee.ImageCollection:
    """Return Landsat C2 SR ImageCollection (L4–L9) cloud-masked over ROI for a year."""
    s = f'{year}-01-01'
    e = f'{year + 1}-01-01'
    l4 = ee.ImageCollection('LANDSAT/LT04/C02/T1_L2').filterBounds(roi).filterDate(s, e).map(landsat_c2_sr)
    l5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').filterBounds(roi).filterDate(s, e).map(landsat_c2_sr)
    l7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').filterBounds(roi).filterDate(s, e).map(landsat_c2_sr)
    l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(roi).filterDate(s, e).map(landsat_c2_sr)
    l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').filterBounds(roi).filterDate(s, e).map(landsat_c2_sr)
    return ee.ImageCollection(l7.merge(l8).merge(l9).merge(l5).merge(l4))


def _ensure_ee_in_worker(high_volume: bool, project: str):
    """Initialize EE in a worker process if not already initialized."""
    try:
        # if already initialized, a second initialize is a no-op
        if high_volume:
            ee.Initialize(project=project, opt_url='https://earthengine-highvolume.googleapis.com')
        else:
            ee.Initialize(project=project)
    except Exception:
        # Fallback attempt after short delay
        time.sleep(1)
        if high_volume:
            ee.Initialize(project=project, opt_url='https://earthengine-highvolume.googleapis.com')
        else:
            ee.Initialize(project=project)


def _landsat_scene_worker(args: Tuple[str, str, Dict[str, Any], int, List[str], bool, str, float]) -> Tuple[str, Dict[str, float]]:
    """Worker to compute per-scene band means for a single image ID.

    Args tuple: (job_id, img_id, region_geojson, year, select_bands, high_volume, project, scale)
    Returns: (job_id, mapping from '{scene}_{band}' -> value)
    """
    job_id, img_id, region_geojson, year, select_bands, high_volume, project, scale = args
    _ensure_ee_in_worker(high_volume=high_volume, project=project)

    # Rebuild the filtered collection cheaply and select the one image
    region = ee.Geometry(region_geojson)
    coll = landsat_masked(year, region).select(select_bands)

    def _compute_scene_values() -> Dict[str, float]:
        img = coll.filter(ee.Filter.eq('system:index', img_id)).first()
        sel = img.select(select_bands)
        vals = sel.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=scale,
            maxPixels=1e13,
            bestEffort=True
        ).getInfo() or {}
        parts = img_id.split('_')
        scene_name = '_'.join(parts[-3:])
        out: Dict[str, float] = {}
        for b in select_bands:
            v = vals.get(b)
            if v is not None:
                out[f'{scene_name}_{b}'] = float(v)
        return out

    result = _execute_with_backoff(
        _compute_scene_values,
        tries=8,
        delay=1.0,
        backoff=2.0,
        exceptions=(Exception,),
        fallback={},
    )
    return job_id, result


def export_landsat_bands_over_buffers(
        shapefile: str,
        bucket: str | None = None,  # deprecated, ignored
        gcs_prefix: str | None = None,  # deprecated, ignored
        id_col: str = 'profile_id',
        start_yr: int = 2000,
        end_yr: int = 2024,
        debug: bool = False,
        buffer_m: float = 250.0,
        check_dir: str | None = None,  # deprecated, ignored
        direct_out_dir: str | None = None,
        workers: int = 25,
        project: str = 'ee-dgketchum',
        high_volume: bool = True,
) -> None:
    """Export per-scene Landsat bands using the shapefile directly (per-feature/year).

    Replaces the previous approach that constructed a large FeatureCollection.
    For each feature in the shapefile and each year:
      - Build the masked Landsat collection over only that feature (buffering points)
      - Stack per-scene bands (B2–B7, B10), renamed `{scene}_{band}`
      - Reduce over that single feature and write out one CSV per feature-year

    Output files:
      - direct: `{direct_out_dir}/landsat_bands_{id}_{year}.csv`

    If output file exists, the job is skipped.
    """
    # Read shapefile and normalize CRS
    gdf = gpd.read_file(shapefile)
    if gdf.crs is None:
        raise ValueError('Input shapefile CRS is undefined')
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    select_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10']

    if direct_out_dir is None:
        raise ValueError("Provide 'direct_out_dir' for Landsat direct export.")
    os.makedirs(direct_out_dir, exist_ok=True)

    skipped, exported = 0, 0
    job_meta: Dict[str, Dict[str, Any]] = {}
    job_expected: Dict[str, int] = {}
    job_received: Dict[str, int] = {}
    job_rows: Dict[str, Dict[str, Any]] = {}
    scene_requests: List[Tuple[str, str, Dict[str, Any], int, List[str], bool, str, float]] = []

    try:
        import pandas as _pd
    except Exception as e:
        raise RuntimeError('pandas is required for Landsat direct export') from e

    for idx, row in gdf.iterrows():
        try:
            fid_val = row[id_col] if (id_col in row and row[id_col] is not None) else f'feat_{idx}'

            geom = row.geometry
            if geom is None:
                print(f'Feature {fid_val}: geometry missing, skipping')
                continue

            if geom.geom_type.lower() == 'point':
                lon, lat = float(geom.x), float(geom.y)
                ee_geom = ee.Geometry.Point([lon, lat]).buffer(float(buffer_m)) if buffer_m and buffer_m > 0 else ee.Geometry.Point([lon, lat])
            else:
                coords = None
                try:
                    if geom.geom_type.lower() in ['polygon', 'multipolygon']:
                        if geom.geom_type.lower() == 'polygon':
                            coords = [[list(tup) for tup in list(geom.exterior.coords)]]
                        else:
                            largest = max(geom.geoms, key=lambda g: g.area)
                            coords = [[list(tup) for tup in list(largest.exterior.coords)]]
                        ee_geom = ee.Geometry.Polygon(coords)
                    else:
                        lon, lat = float(geom.centroid.x), float(geom.centroid.y)
                        ee_geom = ee.Geometry.Point([lon, lat]).buffer(float(buffer_m)) if buffer_m and buffer_m > 0 else ee.Geometry.Point([lon, lat])
                except Exception:
                    lon, lat = float(geom.centroid.x), float(geom.centroid.y)
                    ee_geom = ee.Geometry.Point([lon, lat]).buffer(float(buffer_m)) if buffer_m and buffer_m > 0 else ee.Geometry.Point([lon, lat])

            fc_single = ee.FeatureCollection([ee.Feature(ee_geom, {id_col: fid_val})])
            region_geojson = ee_geom.toGeoJSON()

            for year in range(start_yr, end_yr + 1):
                desc = f'landsat_bands_{fid_val}_{year}'
                out_fp = os.path.join(direct_out_dir, f'{desc}.csv')
                if os.path.exists(out_fp):
                    skipped += 1
                    continue

                try:
                    coll = landsat_masked(year, fc_single).select(select_bands)
                    img_ids = coll.aggregate_array('system:index').getInfo() or []
                except Exception as e:
                    print(f'{desc}: failed to list scenes: {e}')
                    continue

                if len(img_ids) == 0:
                    print(f'{desc}: no scenes, skipping')
                    continue

                job_id = f'{fid_val}_{year}'
                job_meta[job_id] = {'out_fp': out_fp, 'id_val': fid_val}
                job_expected[job_id] = len(img_ids)
                job_received[job_id] = 0
                job_rows[job_id] = {id_col: fid_val}

                for img_id in img_ids:
                    scene_requests.append((job_id, img_id, region_geojson, year, select_bands, high_volume, project, 30.0))
        except Exception as e:
            print(f'Feature {idx} failed: {e}')

    if not scene_requests:
        print('No Landsat scene requests to process.')
        if direct_out_dir:
            print(f'Landsat bands (direct): Exported {exported}, skipped {skipped} files found in {direct_out_dir}')
        return

    try:
        if high_volume:
            ee.Initialize(project=project, opt_url='https://earthengine-highvolume.googleapis.com')
        else:
            ee.Initialize(project=project)
    except Exception:
        pass

    with mp.Pool(processes=max(1, int(workers))) as pool:
        for job_id, scene_vals in pool.imap_unordered(_landsat_scene_worker, scene_requests, chunksize=1):
            if job_id not in job_meta:
                continue
            if scene_vals:
                job_rows[job_id].update(scene_vals)
            job_received[job_id] += 1
            if job_received[job_id] >= job_expected[job_id]:
                out_fp = job_meta[job_id]['out_fp']
                try:
                    df = _pd.DataFrame([job_rows[job_id]])
                    df.to_csv(out_fp, index=False)
                    exported += 1
                    print('Saved:', out_fp)
                except Exception as e:
                    print(f'{job_id}: failed to save CSV: {e}')
                finally:
                    job_rows.pop(job_id, None)
                    job_meta.pop(job_id, None)
                    job_expected.pop(job_id, None)
                    job_received.pop(job_id, None)

    if direct_out_dir:
        print(f'Landsat bands (direct): Exported {exported}, skipped {skipped} files found in {direct_out_dir}')


if __name__ == '__main__':
    """"""

    domain = 'pretrain'

    if domain == 'gshp':
        id_col = 'profile_id'
        shapefile = '/home/dgketchum/data/IrrigationGIS/soils/soil_potential_obs/gshp/wrc_aggregated_mgrs.shp'
        check_dir_ = '/home/dgketchum/data/IrrigationGIS/soils/swapstress/cze/extracts/gshp/landsat'

    elif domain == 'pretrain':
        id_col = 'site_id'
        shapefile = '/home/dgketchum/data/IrrigationGIS/soils/gis/pretraining-roi-10000_mgrs.shp'
        check_dir_ = '/home/dgketchum/data/IrrigationGIS/soils/swapstress/cze/extracts/pretrain/landsat'

    else:
        raise ValueError

    bucket = 'wudr'
    prefix = f'cze/{domain}'
    start_year = 2000
    end_year = 2024

    # Initialize EE; use high-volume endpoint if demonstrating direct mode
    ee_init(high_volume=True)

    # ERA5-Land daily variables by month (direct to local CSVs)
    fc_buffers = build_buffered_fc_from_points(shapefile, id_col=id_col, buffer_m=250.0)
    era5_direct_dir_ = f'/home/dgketchum/data/IrrigationGIS/soils/swapstress/cze/direct/{domain}/era5_land'
    export_era5_land_over_buffers(
        feature_coll=fc_buffers,
        id_col=id_col,
        start_yr=start_year,
        end_yr=end_year,
        debug=False,
        direct_out_dir=era5_direct_dir_,
        project='ee-dgketchum',
        high_volume=True,
        chunk_size=200,
        workers=1,
        use_big_data_api=True,
    )

    # Landsat all bands per-scene by year, iterating directly over shapefile
    # Demonstrate the new direct mode (local CSVs) by default
    direct_dir_ = f'/home/dgketchum/data/IrrigationGIS/soils/swapstress/cze/direct/{domain}/landsat'
    export_landsat_bands_over_buffers(
        shapefile=shapefile,
        bucket=bucket,  # deprecated, ignored
        gcs_prefix=prefix,  # deprecated, ignored
        id_col=id_col,
        start_yr=start_year,
        end_yr=end_year,
        debug=False,
        buffer_m=250.0,
        direct_out_dir=direct_dir_,
        workers=12,
        project='ee-dgketchum',
        high_volume=True,
    )
