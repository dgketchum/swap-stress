import ee

from map.ee_utils import landsat_composites
from map.cdl import get_cdl


def stack_bands_climatology(roi, start_yr=1991, end_yr=2020, resolution=4000):
    """
    Create a stack of climatological bands for the roi specified.
    """
    years = ee.List.sequence(start_yr, end_yr)

    # spring: 60-121, late spring: 121-196, summer: 196-273, fall: 273-365
    periods = [('gs', 121, 273),
               ('1', 60, 121),
               ('2', 121, 196),
               ('3', 196, 273),
               ('4', 273, 365)]

    band_list = []
    for name, start_doy, end_doy in periods:
        landsat_stats = landsat_composites(start_yr, end_yr, start_doy, end_doy, roi, name)
        band_list.append(landsat_stats)

    input_bands = ee.Image.cat(band_list)
    proj = input_bands.select(0).projection()

    ae_bands = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
                .filterDate(f'2017-01-01', f'2024-12-31').filterBounds(roi).mean())

    input_bands = input_bands.addBands([ae_bands])

    # 4-Season GridMET Climatology
    seasons = [('winter', 335, 59),
               ('spring', 60, 151),
               ('summer', 152, 243),
               ('autumn', 244, 334)]

    gridmet_coll = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").filterDate(f'{start_yr}-01-01', f'{end_yr}-12-31')

    for name, start_doy, end_doy in seasons:
        if start_doy > end_doy:  # Handle winter case
            season_filter = ee.Filter.Or(
                ee.Filter.calendarRange(start_doy, 365, 'day_of_year'),
                ee.Filter.calendarRange(1, end_doy, 'day_of_year')
            )
        else:
            season_filter = ee.Filter.calendarRange(start_doy, end_doy, 'day_of_year')

        seasonal_gridmet = gridmet_coll.filter(season_filter)

        reducers = ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', True)
        climate_stats = (seasonal_gridmet.select(['pr', 'eto', 'tmmn', 'tmmx'])
                         .reduce(reducers))

        mean_temp = (climate_stats.select('tmmn_mean').add(climate_stats.select('tmmx_mean'))
                     .divide(2).rename('tmean_mean'))

        climate_stats = climate_stats.addBands(mean_temp)

        new_names = climate_stats.bandNames().map(lambda b: ee.String(b).cat('_').cat(name))
        seasonal_clim_image = climate_stats.rename(new_names)

        input_bands = input_bands.addBands(seasonal_clim_image)

    s1_coll = (ee.ImageCollection("COPERNICUS/S1_GRD")
               .filterDate(ee.Date.fromYMD(2015, 1, 1), ee.Date.fromYMD(2024, 12, 31))
               .filterBounds(roi)
               .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
               .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
               .filter(ee.Filter.eq('instrumentMode', 'IW')))

    def add_ratio(img):
        vh_vv = img.select('VH').divide(img.select('VV')).rename('VH_VV')
        return img.addBands(vh_vv)

    s1_coll_with_ratio = s1_coll.map(add_ratio).select(['VV', 'VH', 'VH_VV'])

    s1_mean = s1_coll_with_ratio.mean().rename(['VV_mean', 'VH_mean', 'VH_VV_mean'])
    s1_stdDev = s1_coll_with_ratio.reduce(ee.Reducer.stdDev()).rename(['VV_stdDev', 'VH_stdDev', 'VH_VV_stdDev'])
    s1_clim = ee.Image.cat([s1_mean, s1_stdDev])
    input_bands = input_bands.addBands(s1_clim)

    smap_l3_coll = (ee.ImageCollection("NASA/SMAP/SPL3SMP_E/005")
                    .filterDate('2015-03-31T12:00:00', '2023-12-03T12:00:00')
                    .filterBounds(roi))

    smap_l3_am_good = smap_l3_coll.filter(ee.Filter.eq('retrieval_qual_flag_am', 0))
    smap_l3_pm_good = smap_l3_coll.filter(ee.Filter.eq('retrieval_qual_flag_pm', 0))

    smap_l3_am_clim = (smap_l3_am_good.select(['soil_moisture_am', 'vegetation_water_content_am'])
                       .reduce(ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', True)))
    smap_l3_pm_clim = (smap_l3_pm_good.select(['soil_moisture_pm', 'vegetation_water_content_pm'])
                       .reduce(ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', True)))

    smap_l3_clim = ee.Image.cat([smap_l3_am_clim, smap_l3_pm_clim])
    input_bands = input_bands.addBands(smap_l3_clim)

    smap_l4_bands = ['sm_surface', 'sm_rootzone', 'sm_profile', 'surface_temp', 'leaf_area_index']
    smap_l4_coll = (ee.ImageCollection("NASA/SMAP/SPL4SMGP/008")
                    .filterDate('2015-03-31T00:00:00', '2025-08-15T22:30:00')
                    .filterBounds(roi)
                    .select(smap_l4_bands))

    smap_l4_clim = smap_l4_coll.reduce(ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', True))
    input_bands = input_bands.addBands(smap_l4_clim)

    coords = ee.Image.pixelLonLat().rename(['lon', 'lat'])
    ned = ee.Image('USGS/3DEP/10m')
    terrain = ee.Terrain.products(ned).select('elevation', 'slope', 'aspect')
    elev = terrain.select('elevation')
    tpi_250 = elev.subtract(elev.focal_mean(250, 'circle', 'meters')).add(0.5).rename('tpi_250')
    tpi_500 = elev.subtract(elev.focal_mean(500, 'circle', 'meters')).add(0.5).rename('tpi_500')
    tpi_1250 = elev.subtract(elev.focal_mean(1250, 'circle', 'meters')).add(0.5).rename('tpi_1250')
    tpi_2500 = elev.subtract(elev.focal_mean(2500, 'circle', 'meters')).add(0.5).rename('tpi_2500')
    tpi_10000 = elev.subtract(elev.focal_mean(10000, 'circle', 'meters')).add(0.5).rename('tpi_10000')
    tpi_22500 = elev.subtract(elev.focal_mean(22500, 'circle', 'meters')).add(0.5).rename('tpi_22500')

    nlcd = ee.ImageCollection('USGS/NLCD_RELEASES/2019_REL/NLCD').select('landcover').mosaic().rename('nlcd')

    def get_cdl_simple(y):
        return get_cdl(y)[2]

    cdl_mode = ee.ImageCollection(years.map(get_cdl_simple)).mode().rename('cdl_mode')

    gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').gt(0).unmask(0).rename('gsw')

    ssurgo = ee.Image.cat([
        ee.Image('projects/earthengine-legacy/assets/projects/openet/soil/ssurgo_AWC_WTA_0to152cm_composite').rename(
            'ssurgo_awc'),
        ee.Image('projects/earthengine-legacy/assets/projects/openet/soil/ssurgo_Clay_WTA_0to152cm_composite').rename(
            'ssurgo_clay'),
        ee.Image('projects/earthengine-legacy/assets/projects/openet/soil/ssurgo_Ksat_WTA_0to152cm_composite').rename(
            'ssurgo_ksat'),
        ee.Image('projects/earthengine-legacy/assets/projects/openet/soil/ssurgo_Sand_WTA_0to152cm_composite').rename(
            'ssurgo_sand')
    ])

    prism = ee.ImageCollection("OREGONSTATE/PRISM/Norm91m").mean()

    additional_terrain = ee.Image.cat([
        ee.Image("users/zhoylman/CONUS_TWI_epsg5072_30m"),
        ee.Image("CSP/ERGo/1_0/US/topoDiversity")
    ])

    polaris = ee.Image.cat([
        ee.ImageCollection('projects/sat-io/open-datasets/polaris/bd_mean').mean().rename('bd_mean'),
        ee.ImageCollection('projects/sat-io/open-datasets/polaris/clay_mean').mean().rename('clay_mean'),
        ee.ImageCollection('projects/sat-io/open-datasets/polaris/ksat_mean').mean().rename('ksat_mean'),
        ee.ImageCollection('projects/sat-io/open-datasets/polaris/n_mean').mean().rename('n_mean'),
        ee.ImageCollection('projects/sat-io/open-datasets/polaris/om_mean').mean().rename('om_mean'),
        ee.ImageCollection('projects/sat-io/open-datasets/polaris/ph_mean').mean().rename('ph_mean'),
        ee.ImageCollection('projects/sat-io/open-datasets/polaris/sand_mean').mean().rename('sand_mean'),
        ee.ImageCollection('projects/sat-io/open-datasets/polaris/silt_mean').mean().rename('silt_mean'),
        ee.ImageCollection('projects/sat-io/open-datasets/polaris/theta_r_mean').mean().rename('theta_r_mean'),
        ee.ImageCollection('projects/sat-io/open-datasets/polaris/theta_s_mean').mean().rename('theta_s_mean'),
        ee.ImageCollection('projects/sat-io/open-datasets/polaris/lambda_mean').mean().rename('lambda_mean'),
        ee.ImageCollection('projects/sat-io/open-datasets/polaris/hb_mean').mean().rename('hb_mean'),
        ee.ImageCollection('projects/sat-io/open-datasets/polaris/alpha_mean').mean().rename('alpha_mean')
    ])


    static_bands = [coords,
                    terrain,
                    tpi_1250,
                    tpi_250,
                    tpi_500,
                    tpi_2500,
                    tpi_10000,
                    tpi_22500,
                    nlcd,
                    cdl_mode,
                    gsw,
                    ssurgo,
                    prism,
                    additional_terrain,
                    polaris]


    input_bands = input_bands.addBands(static_bands).resample('bilinear').reproject(crs=proj.crs(),
                                                                                    scale=resolution)

    return input_bands.clip(roi)


def is_authorized():
    try:
        ee.Initialize(project='ee-dgketchum')
        print('Authorized')
    except Exception as e:
        print('You are not authorized: {}'.format(e))
        exit(1)
    return None


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
