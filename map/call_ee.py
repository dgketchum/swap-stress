import ee

from map.ee_utils import get_world_climate, landsat_composites
from map.cdl import get_cdl


def stack_bands_climatology(roi, start_yr=1991, end_yr=2020, alpha_earth=False):
    """
    Create a stack of climatological bands for the roi specified.
    """

    if alpha_earth:
        dataset = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
                   .filterDate(f'{start_yr}-01-01', f'{end_yr}-12-31').filterBounds(roi).mean())
        dataset = dataset.clip(roi)
        return dataset

    years = ee.List.sequence(start_yr, end_yr)

    spring_s, spring_e = '03-01', '05-01'
    late_spring_s, late_spring_e = '05-01', '07-15'
    summer_s, summer_e = '07-15', '09-30'
    fall_s, fall_e = '09-30', '12-31'

    periods = [('gs', spring_e, fall_s),
               ('1', spring_s, spring_e),
               ('2', late_spring_s, late_spring_e),
               ('3', summer_s, summer_e),
               ('4', fall_s, fall_e)]

    band_list = []
    for name, start, end in periods:
        def annual_composite(y):
            y = ee.Number(y).toInt()
            s = ee.String(y.format()).cat('-').cat(start)
            e = ee.String(y.format()).cat('-').cat(end)
            return landsat_composites(y, s, e, roi, name, composites_only=False)

        collection = ee.ImageCollection.fromImages(years.map(annual_composite))
        first_image_bands = collection.first().bandNames()
        mean = collection.mean().rename(first_image_bands.map(lambda b: ee.String(b).cat('_mean')))
        std_dev = collection.reduce(ee.Reducer.stdDev()).rename(
            first_image_bands.map(lambda b: ee.String(b).cat('_stdDev')))
        band_list.append(mean)
        band_list.append(std_dev)

    input_bands = ee.Image(band_list)
    proj = input_bands.select(0).projection()

    gridmet_coll = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").filterDate(f'{start_yr}-01-01', f'{end_yr}-12-31')

    for s_month, e_month, n, m in [(3, 5, 'spr', (3, 5)),
                                   (6, 9, 'smr', (6, 9))]:
        gridmet = gridmet_coll.filter(ee.Filter.calendarRange(s_month, e_month, 'month'))

        def annual_mean_temp(y):
            return (gridmet.filter(ee.Filter.calendarRange(y, y, 'year'))
                    .select(['tmmn', 'tmmx']).mean()
                    .expression('(b("tmmn") + b("tmmx")) / 2').rename(f'tmp_{n}'))

        temp_collection = ee.ImageCollection(years.map(annual_mean_temp))
        mean_temp = temp_collection.mean()
        std_dev_temp = temp_collection.reduce(ee.Reducer.stdDev()).rename(f'tmp_{n}_stdDev')

        def annual_sum(y):
            return gridmet.filter(ee.Filter.calendarRange(y, y, 'year')).select(['pr', 'eto']).sum()

        annual_sum_collection = ee.ImageCollection(years.map(annual_sum))
        mean_annual_sum = annual_sum_collection.mean().rename(f'prec_tot_{n}', f'pet_tot_{n}')
        std_dev_annual_sum = (annual_sum_collection.reduce(ee.Reducer.stdDev())
                              .rename(f'prec_tot_{n}_stdDev', f'pet_tot_{n}_stdDev'))

        wd_estimate = mean_annual_sum.select(f'prec_tot_{n}').subtract(
            mean_annual_sum.select(f'pet_tot_{n}')).rename(f'cwd_{n}')

        worldclim_prec = get_world_climate(proj=proj, months=m, param='prec')
        anom_prec = mean_annual_sum.select(f'prec_tot_{n}').subtract(worldclim_prec).rename(f'an_prec_{n}')
        worldclim_temp = get_world_climate(proj=proj, months=m, param='tavg')
        anom_temp = mean_temp.subtract(worldclim_temp).rename(f'an_temp_{n}')

        input_bands = input_bands.addBands([mean_temp, std_dev_temp, mean_annual_sum, std_dev_annual_sum,
                                        wd_estimate, anom_temp, anom_prec])

    s1_start_yr = ee.Number.max(start_yr, 2015)
    s1_coll = (ee.ImageCollection("COPERNICUS/S1_GRD")
               .filterDate(ee.Date.fromYMD(s1_start_yr.toInt(), 1, 1), ee.Date.fromYMD(ee.Number.parse(end_yr), 12, 31))
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

    nlcd = ee.Image('USGS/NLCD/NLCD2019').select('landcover').rename('nlcd')

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

    landform = ee.Image('projects/usgs-gap/landform').rename('landform')

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
                    polaris,
                    landform]

    input_bands = input_bands.addBands(static_bands).resample('bilinear').reproject(crs=proj)

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
