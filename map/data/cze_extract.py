import os
import time
from typing import List, Dict, Any, Tuple

import ee
import geopandas as gpd
import multiprocessing as mp

try:
    # Optional, only needed for ERA5 ETo
    from openet.refetgee import Daily as RefETDaily
except Exception:
    RefETDaily = None


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
) -> None:
    """Write daily ERA5-Land variables reduced over buffered features by month.

    Saves local CSVs per month to `direct_out_dir` with filenames like
    `era5_vars_YYYY_MM.csv` containing columns: `{id_col}`, daily SWE, ETo,
    Tmean/Tmin/Tmax (C), precipitation (mm), and shortwave radiation (W/m^2).
    """
    era5_land_hourly = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
    scale_era5 = 11132  # ~10 km at equator; consistent with reference workflows

    # Build list of (year, month)
    dtimes = [(y, m) for y in range(start_yr, end_yr + 1) for m in range(1, 13)]

    for year, month in dtimes:
        try:
            first_band_in_month = True
            monthly_bands_image = None
            selectors = [id_col]

            desc = f'era5_vars_{year}_{str(month).zfill(2)}'

            month_start = ee.Date.fromYMD(year, month, 1)
            month_end = month_start.advance(1, 'month')

            # Build list of day start dates for month
            try:
                days_in_month_list = []
                d = month_start
                while d.millis().lt(month_end.millis()).getInfo():
                    days_in_month_list.append(d)
                    d = d.advance(1, 'day')
            except Exception as e:
                print(f'ERA5 {year}-{str(month).zfill(2)} failed building date list: {e}')
                continue

            if not days_in_month_list:
                continue

            for day_date in days_in_month_list:
                try:
                    day_str = day_date.format('YYYYMMdd').getInfo()
                    day_start = day_date
                    day_end = day_date.advance(1, 'day')

                    coll_day = era5_land_hourly.filterDate(day_start, day_end)

                    # SWE mm
                    swe = coll_day.select('snow_depth_water_equivalent').mean().multiply(1000).rename(f'swe_{day_str}')
                    swe = swe.set('system:time_start', day_start.millis())
                    # ETo (alfalfa) via RefET; optional if refetgee missing
                    if RefETDaily is not None:
                        eto = RefETDaily.era5_land(coll_day).etr.rename(f'eto_{day_str}')
                        eto = eto.set('system:time_start', day_start.millis())
                    else:
                        eto = ee.Image(0).rename(f'eto_{day_str}')
                    # Temps C
                    t_hourly = coll_day.select('temperature_2m')
                    tmean = t_hourly.mean().subtract(273.15).rename(f'tmean_{day_str}')
                    tmin = t_hourly.min().subtract(273.15).rename(f'tmin_{day_str}')
                    tmax = t_hourly.max().subtract(273.15).rename(f'tmax_{day_str}')
                    tmean = tmean.set('system:time_start', day_start.millis())
                    tmin = tmin.set('system:time_start', day_start.millis())
                    tmax = tmax.set('system:time_start', day_start.millis())
                    # Precip mm
                    precip = coll_day.select('total_precipitation_hourly').sum().multiply(1000).rename(
                        f'precip_{day_str}')
                    precip = precip.set('system:time_start', day_start.millis())
                    # Shortwave W/m^2
                    srad = coll_day.select('surface_solar_radiation_downwards_hourly').mean().rename(f'srad_{day_str}')
                    srad = srad.set('system:time_start', day_start.millis())

                    for nm in [
                        f'swe_{day_str}', f'eto_{day_str}', f'tmean_{day_str}',
                        f'tmin_{day_str}', f'tmax_{day_str}', f'precip_{day_str}', f'srad_{day_str}'
                    ]:
                        selectors.append(nm)

                    daily_bands = [swe, eto, tmean, tmin, tmax, precip, srad]
                    if first_band_in_month:
                        monthly_bands_image = ee.Image(daily_bands)
                        first_band_in_month = False
                    else:
                        monthly_bands_image = monthly_bands_image.addBands(ee.Image(daily_bands))
                except Exception as e:
                    print(f'ERA5 {year}-{str(month).zfill(2)} day build failed: {e}')
                    continue

            if monthly_bands_image is None:
                continue

            # Direct mode only: request results via getInfo() and save locally per month.
            if direct_out_dir is None:
                raise ValueError("Provide 'direct_out_dir' for ERA5 direct export.")

            out_fp = os.path.join(direct_out_dir, f'{desc}.csv')
            if os.path.exists(out_fp):
                print('Exists, skipping:', out_fp)
                continue

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

            # Chunk the feature collection to keep responses under limits
            try:
                n_feat = int(feature_coll.size().getInfo())
            except Exception as e:
                print(f'ERA5 {desc}: failed to get feature count: {e}')
                continue

            rows: list[Dict[str, Any]] = []
            feats_list = feature_coll.toList(n_feat)
            for start in range(0, n_feat, max(1, int(chunk_size))):
                end = min(start + int(chunk_size), n_feat)
                sub_fc = ee.FeatureCollection(feats_list.slice(start, end))

                # Backoff for each chunk in case of 429s
                delay = 1.0
                for attempt in range(6):
                    try:
                        sub = monthly_bands_image.reduceRegions(
                            collection=sub_fc,
                            reducer=ee.Reducer.mean(),
                            scale=scale_era5,
                        ).getInfo()
                        features = sub.get('features', []) if isinstance(sub, dict) else []
                        for f in features:
                            props = f.get('properties', {})
                            if not props:
                                continue
                            if id_col not in props:
                                props[id_col] = None
                            rows.append(props)
                        break
                    except Exception:
                        time.sleep(delay)
                        delay = min(delay * 2, 30)
                        continue

            if not rows:
                print('No rows for', desc)
                continue

            # Order columns: id first, then bands in the original selectors order
            cols = [id_col] + [c for c in selectors if c != id_col]
            df = _pd.DataFrame(rows)
            df = df.reindex(columns=cols)
            os.makedirs(os.path.dirname(out_fp), exist_ok=True)
            df.to_csv(out_fp, index=False)
            print('Saved:', out_fp)
        except Exception as e:
            print(f'ERA5 {year}-{str(month).zfill(2)} failed: {e}')


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


def _landsat_scene_worker(args: Tuple[str, Dict[str, Any], int, List[str], bool, str, float]) -> Dict[str, float]:
    """Worker to compute per-scene band means for a single image ID.

    Args tuple: (img_id, region_geojson, year, select_bands, high_volume, project, scale)
    Returns: mapping from '{scene}_{band}' -> value
    """
    img_id, region_geojson, year, select_bands, high_volume, project, scale = args
    _ensure_ee_in_worker(high_volume=high_volume, project=project)

    # Rebuild the filtered collection cheaply and select the one image
    region = ee.Geometry(region_geojson)
    coll = landsat_masked(year, region).select(select_bands)

    # Retry loop with exponential backoff for transient errors
    delay = 1.0
    for attempt in range(8):
        try:
            img = coll.filter(ee.Filter.eq('system:index', img_id)).first()
            sel = img.select(select_bands)
            # Compute per-band means over the region
            vals = sel.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=scale,
                maxPixels=1e13,
                bestEffort=True
            ).getInfo()
            if vals is None:
                vals = {}
            # Format keys like export path: '{scene}_{band}'
            parts = img_id.split('_')
            scene_name = '_'.join(parts[-3:])
            out = {}
            for b in select_bands:
                v = vals.get(b)
                if v is not None:
                    out[f'{scene_name}_{b}'] = float(v)
            return out
        except Exception:
            time.sleep(delay)
            delay = min(delay * 2, 30)
            continue
    # If all retries failed, return empty dict so caller can proceed
    return {}


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

    for idx, row in gdf.iterrows():
        try:
            # Determine identifier
            fid_val = row[id_col] if (id_col in row and row[id_col] is not None) else f'feat_{idx}'

            geom = row.geometry
            if geom is None:
                print(f'Feature {fid_val}: geometry missing, skipping')
                continue

            # Build EE geometry: buffer points; otherwise use polygon/geometry centroid buffer if requested
            if geom.geom_type.lower() == 'point':
                lon, lat = float(geom.x), float(geom.y)
                ee_geom = ee.Geometry.Point([lon, lat]).buffer(float(buffer_m)) if buffer_m and buffer_m > 0 else ee.Geometry.Point([lon, lat])
            else:
                # Use the provided geometry; no additional buffering by default
                coords = None
                try:
                    # Prefer polygons; fall back to centroid buffer for non-area types
                    if geom.geom_type.lower() in ['polygon', 'multipolygon']:
                        # ee.Geometry expects lists of lists; take exterior ring for polygon
                        if geom.geom_type.lower() == 'polygon':
                            coords = [[list(tup) for tup in list(geom.exterior.coords)]]
                        else:
                            # Multipolygon: pick the largest polygon's exterior
                            largest = max(geom.geoms, key=lambda g: g.area)
                            coords = [[list(tup) for tup in list(largest.exterior.coords)]]
                        ee_geom = ee.Geometry.Polygon(coords)
                    else:
                        # Lines/others: fallback to centroid with optional buffer
                        lon, lat = float(geom.centroid.x), float(geom.centroid.y)
                        ee_geom = ee.Geometry.Point([lon, lat]).buffer(float(buffer_m)) if buffer_m and buffer_m > 0 else ee.Geometry.Point([lon, lat])
                except Exception:
                    lon, lat = float(geom.centroid.x), float(geom.centroid.y)
                    ee_geom = ee.Geometry.Point([lon, lat]).buffer(float(buffer_m)) if buffer_m and buffer_m > 0 else ee.Geometry.Point([lon, lat])

            fc_single = ee.FeatureCollection([ee.Feature(ee_geom, {id_col: fid_val})])

            for year in range(start_yr, end_yr + 1):
                desc = f'landsat_bands_{fid_val}_{year}'

                out_fp = os.path.join(direct_out_dir, f'{desc}.csv')
                if os.path.exists(out_fp):
                    skipped += 1
                    continue

                # Build minimal request list: image IDs for this year/feature
                try:
                    coll = landsat_masked(year, fc_single).select(select_bands)
                    img_ids = coll.aggregate_array('system:index').getInfo() or []
                except Exception as e:
                    print(f'{desc}: failed to list scenes: {e}')
                    continue
                if len(img_ids) == 0:
                    print(f'{desc}: no scenes, skipping')
                    continue

                # Parallel reduceRegion for each image
                region_geojson = ee_geom.toGeoJSON()
                args = [(img_id, region_geojson, year, select_bands, high_volume, project, 30.0)
                        for img_id in img_ids]

                # Initialize EE once in parent for HV endpoint
                # (workers will re-init in their process)
                try:
                    if high_volume:
                        ee.Initialize(project=project, opt_url='https://earthengine-highvolume.googleapis.com')
                    else:
                        ee.Initialize(project=project)
                except Exception:
                    pass

                with mp.Pool(processes=max(1, int(workers))) as pool:
                    results = pool.map(_landsat_scene_worker, args)

                # Merge per-scene dicts into one row
                row: Dict[str, Any] = {id_col: fid_val}
                for d in results:
                    if d:
                        row.update(d)

                # Persist a single-row CSV
                try:
                    import pandas as _pd  # local import to avoid dependency when unused
                    df = _pd.DataFrame([row])
                    df.to_csv(out_fp, index=False)
                    print('Saved:', out_fp)
                    exported += 1
                except Exception as e:
                    print(f'{desc}: failed to save CSV: {e}')
        except Exception as e:
            print(f'Feature {idx} failed: {e}')

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
        workers=25,
        project='ee-dgketchum',
        high_volume=True,
    )
