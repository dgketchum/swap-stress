import os
import time
from typing import List

import ee
import geopandas as gpd

try:
    # Optional, only needed for ERA5 ETo
    from openet.refetgee import Daily as RefETDaily
except Exception:
    RefETDaily = None


def ee_init(project: str = 'ee-dgketchum') -> None:
    """Initialize the Earth Engine client for a given project."""
    try:
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
        bucket: str,
        gcs_prefix: str = 'cze',
        id_col: str = 'profile_id',
        start_yr: int = 2000,
        end_yr: int = 2024,
        debug: bool = False,
) -> None:
    """Export daily ERA5-Land variables reduced over buffered features by month.

    Exports CSVs to `gs://{bucket}/{gcs_prefix}/` with filenames like
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

            if debug:
                try:
                    sample = monthly_bands_image.reduceRegions(
                        collection=feature_coll.limit(1),
                        reducer=ee.Reducer.mean(),
                        scale=scale_era5,
                    ).getInfo()
                    print('ERA5 sample:', sample)
                except Exception as e:
                    print(f'ERA5 {year}-{str(month).zfill(2)} debug sample failed: {e}')

            data = monthly_bands_image.reduceRegions(
                collection=feature_coll,
                reducer=ee.Reducer.mean(),
                scale=scale_era5,
            )

            task = ee.batch.Export.table.toCloudStorage(
                collection=data,
                description=desc,
                bucket=bucket,
                fileNamePrefix=f'{gcs_prefix}/era5_land/{desc}',
                fileFormat='CSV',
                selectors=selectors,
            )
            try:
                task.start()
                print('Started:', desc)
            except ee.ee_exception.EEException as e:
                print('ERA5 export start failed, will retry:', desc, e)
                time.sleep(600)
                try:
                    task.start()
                    print('Started on retry:', desc)
                except Exception as e2:
                    print(f'ERA5 export start retry failed for {desc}: {e2}')
            except Exception as e:
                print(f'ERA5 unexpected error starting export {desc}: {e}')
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


def export_landsat_bands_over_buffers(
        shapefile: str,
        bucket: str,
        gcs_prefix: str = 'cze',
        id_col: str = 'profile_id',
        start_yr: int = 2000,
        end_yr: int = 2024,
        debug: bool = False,
        buffer_m: float = 250.0,
        check_dir: str | None = None,
) -> None:
    """Export per-scene Landsat bands using the shapefile directly (per-feature/year).

    Replaces the previous approach that constructed a large FeatureCollection.
    For each feature in the shapefile and each year:
      - Build the masked Landsat collection over only that feature (buffering points)
      - Stack per-scene bands (B2–B7, B10), renamed `{scene}_{band}`
      - Reduce over that single feature and export one CSV per feature-year

    Output files: `gs://{bucket}/{gcs_prefix}/landsat_bands_{id}_{year}.csv`.
    If `check_dir` is provided, skips exports when `{check_dir}/landsat_bands_{id}_{year}.csv` exists.
    """
    # Read shapefile and normalize CRS
    gdf = gpd.read_file(shapefile)
    if gdf.crs is None:
        raise ValueError('Input shapefile CRS is undefined')
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    select_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10']

    if check_dir:
        if not os.path.isdir(check_dir):
            raise ValueError(f'File checking on but directory does not exist: {check_dir}')

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

                # Skip if local CSV exists
                if check_dir:
                    fpath = os.path.join(check_dir, f'{desc}.csv')
                    if os.path.exists(fpath):
                        skipped += 1
                        continue

                # Masked Landsat over this feature and year
                coll = landsat_masked(year, fc_single).select(select_bands)

                # Skip if no scenes
                try:
                    n_img = coll.size().getInfo()
                except Exception as e:
                    print(f'{desc}: size check failed: {e}')
                    continue
                if n_img == 0:
                    print(f'{desc}: no scenes, skipping')
                    continue

                # Build stacked image of renamed scene bands
                try:
                    scene_hist = coll.aggregate_histogram('system:index').getInfo()
                except Exception as e:
                    print(f'{desc}: failed to list scenes: {e}')
                    continue

                first = True
                stacked = None
                selectors = [id_col]

                for img_id in scene_hist:
                    parts = img_id.split('_')
                    scene_name = '_'.join(parts[-3:])

                    # Append column names for this scene
                    for b in select_bands:
                        selectors.append(f'{scene_name}_{b}')

                    img = coll.filterMetadata('system:index', 'equals', img_id).first()
                    sel = img.select(select_bands)
                    renamed = sel.rename([f'{scene_name}_{b}' for b in select_bands])

                    if first:
                        stacked = renamed
                        first = False
                    else:
                        stacked = stacked.addBands(renamed)

                if stacked is None:
                    print(f'{desc}: no bands assembled, skipping')
                    continue

                if debug:
                    try:
                        sample = stacked.sample(fc_single, 30).first().toDictionary().getInfo()
                        print(f'{desc}: sample keys (truncated):', list(sample.keys())[:10])
                    except Exception as e:
                        print(f'{desc}: debug sampling failed: {e}')

                try:
                    data = stacked.reduceRegions(collection=fc_single,
                                                 reducer=ee.Reducer.mean(),
                                                 scale=30)
                except Exception as e:
                    print(f'{desc}: reduceRegions failed: {e}')
                    continue

                task = ee.batch.Export.table.toCloudStorage(
                    collection=data,
                    description=desc,
                    bucket=bucket,
                    fileNamePrefix=f'{gcs_prefix}/landsat/{desc}',
                    fileFormat='CSV',
                    selectors=selectors,
                )
                try:
                    task.start()
                    print('Started:', desc)
                    exported += 1
                except ee.ee_exception.EEException as e:
                    print('Landsat export start failed, will retry:', desc, e)
                    time.sleep(600)
                    try:
                        task.start()
                        print('Started on retry:', desc)
                        exported += 1
                    except Exception as e2:
                        print(f'Landsat export start retry failed for {desc}: {e2}')
                except Exception as e:
                    print(f'Landsat unexpected error starting export {desc}: {e}')
        except Exception as e:
            print(f'Feature {idx} failed: {e}')

    if check_dir:
        print(f'Landsat bands: Exported {exported}, skipped {skipped} files found in {check_dir}')


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
    prefix = 'cze'
    start_year = 2000
    end_year = 2024

    ee_init()

    # ERA5-Land daily variables by month (still supports full collection input)
    fc_buffers = build_buffered_fc_from_points(shapefile, id_col=id_col, buffer_m=250.0)
    export_era5_land_over_buffers(
        feature_coll=fc_buffers,
        bucket=bucket,
        gcs_prefix=prefix,
        id_col=id_col,
        start_yr=start_year,
        end_yr=end_year,
        debug=False)

    # Landsat all bands per-scene by year, iterating directly over shapefile
    export_landsat_bands_over_buffers(
        shapefile=shapefile,
        bucket=bucket,
        gcs_prefix=prefix,
        id_col=id_col,
        start_yr=start_year,
        end_yr=end_year,
        debug=False,
        buffer_m=250.0,
        check_dir=check_dir_,
    )
