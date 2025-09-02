import ee

from map.data.ee_utils import landsat_composites
from map.data.cdl import get_cdl


def _prefixed(img, prefix):
    names = img.bandNames()
    new = names.map(lambda b: ee.String(prefix).cat('_').cat(ee.String(b)))
    return ee.Image(img).rename(new)


def get_world_climate(months, param='prec', band_name=None):
    if band_name is None:
        band_name = param

    if months[0] > months[1]:
        month_numbers = list(range(months[0], 13)) + list(range(1, months[1] + 1))
    else:
        month_numbers = list(range(months[0], months[1] + 1))

    if param in ['prec', 'tmin', 'tavg', 'tmax']:
        collection = ee.ImageCollection('WORLDCLIM/V1/MONTHLY')
        monthly_images = []
        for m in month_numbers:
            img = collection.filter(ee.Filter.eq('month', m)).first().select(band_name)
            monthly_images.append(img)

        image_coll = ee.ImageCollection.fromImages(monthly_images)

        if param == 'prec':
            agg_image = image_coll.sum()
        else:
            agg_image = image_coll.mean()

    elif param == 'eto':
        monthly_root = 'projects/sat-io/open-datasets/global_et0/global_et0_monthly'
        monthly_images = []
        for m in month_numbers:
            if m in list(range(5, 10)):
                mi = ee.Image(f'{monthly_root}/et0_V3_{str(m).rjust(2, "0")}')
            else:
                mi = ee.Image(f'{monthly_root}/et0_v3_{str(m).rjust(2, "0")}')
            monthly_images.append(mi)
        image_coll = ee.ImageCollection.fromImages(monthly_images)
        agg_image = image_coll.sum()

    else:
        raise ValueError

    return agg_image


def stack_bands_climatology(roi, start_yr=1991, end_yr=2020, subselection=None, region='conus'):
    years = ee.List.sequence(start_yr, end_yr)

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

    ae_bands = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
                .filterDate(f'2017-01-01', f'2024-12-31').filterBounds(roi).mean())

    input_bands = input_bands.addBands([ae_bands])

    if region == 'conus':
        seasons = [('winter', 335, 59),
                   ('spring', 60, 151),
                   ('summer', 152, 243),
                   ('autumn', 244, 334)]

        gridmet_coll = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").filterDate(f'{start_yr}-01-01', f'{end_yr}-12-31')

        for name, start_doy, end_doy in seasons:
            if start_doy > end_doy:
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

    elif region == 'global':
        seasons = [('winter', (12, 2)),
                   ('spring', (3, 5)),
                   ('summer', (6, 8)),
                   ('autumn', (9, 11))]

        for s_name, s_months in seasons:
            wc_params = ['prec', 'tavg', 'tmin', 'tmax']
            for p_name in wc_params:
                clim_band = get_world_climate(s_months, param=p_name)
                band_name = f'wc_{p_name}_{s_name}'
                input_bands = input_bands.addBands(clim_band.rename(band_name))

            eto_band = get_world_climate(s_months, param='eto', band_name='b1')
            band_name = f'eto_{s_name}'
            input_bands = input_bands.addBands(eto_band.rename(band_name))

        et_yearly = ee.Image("projects/sat-io/open-datasets/global_et0/global_et0_yearly").rename('eto_yearly_mean')
        et_yearly_sd = ee.Image("projects/sat-io/open-datasets/global_et0/global_et0_yearly_sd").rename('eto_yearly_sd')
        input_bands = input_bands.addBands(et_yearly).addBands(et_yearly_sd)

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
                    .filterDate('2018-01-01', '2023-12-31')
                    .filterBounds(roi))

    # TODO: Figure out why these won't go
    # vegetation_water_content_am
    # vegetation_water_content_pm

    smap_l3_am_clim = (smap_l3_coll.select(['vegetation_water_content_am'])
                       .reduce(ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', True)))
    smap_l3_pm_clim = (smap_l3_coll.select(['vegetation_water_content_pm'])
                       .reduce(ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', True)))

    smap_l3_clim = ee.Image.cat([smap_l3_am_clim, smap_l3_pm_clim])
    input_bands = input_bands.addBands(smap_l3_clim)

    smap_l4_bands = ['sm_surface', 'sm_rootzone', 'sm_profile', 'surface_temp', 'leaf_area_index']
    smap_l4_coll = (ee.ImageCollection("NASA/SMAP/SPL4SMGP/008")
                    .filterDate('2018-01-01', '2023-12-31')
                    .filterBounds(roi)
                    .select(smap_l4_bands))

    smap_l4_clim = smap_l4_coll.reduce(ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', True))
    input_bands = input_bands.addBands(smap_l4_clim)

    coords = ee.Image.pixelLonLat().rename(['lon', 'lat'])

    # annoying treatment of international DEM
    if region == 'conus':
        ned = ee.Image('USGS/3DEP/10m')
        terrain = ee.Terrain.products(ned).select('elevation', 'slope', 'aspect')
    elif region == 'global':
        dem = ee.Image("projects/sat-io/open-datasets/ASTER/GDEM").select('b1').rename('elevation')
        terrain = ee.Terrain.products(dem).select('slope', 'aspect').addBands(dem)
    else:
        raise ValueError("region must be one of 'conus' or 'global'")

    tpi_10000 = dem.subtract(dem.focal_mean(10000, 'circle', 'meters')).add(0.5).rename('tpi_10000')
    tpi_22500 = dem.subtract(dem.focal_mean(22500, 'circle', 'meters')).add(0.5).rename('tpi_22500')

    gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').gt(0).unmask(0).rename('gsw')

    hihydro_bands = ee.Image.cat([
        ee.ImageCollection("projects/sat-io/open-datasets/HiHydroSoilv2_0/ksat").mean().rename('hhs_ksat'),
        ee.ImageCollection("projects/sat-io/open-datasets/HiHydroSoilv2_0/sat-field").mean().rename('hhs_sat_field'),
        ee.ImageCollection("projects/sat-io/open-datasets/HiHydroSoilv2_0/N").mean().rename('hhs_n'),
        ee.ImageCollection("projects/sat-io/open-datasets/HiHydroSoilv2_0/alpha").mean().rename('hhs_alpha'),
        ee.ImageCollection("projects/sat-io/open-datasets/HiHydroSoilv2_0/crit-wilt").mean().rename('hhs_crit_wilt'),
        ee.ImageCollection("projects/sat-io/open-datasets/HiHydroSoilv2_0/field-crit").mean().rename('hhs_field_crit'),
        ee.ImageCollection("projects/sat-io/open-datasets/HiHydroSoilv2_0/ormc").mean().rename('hhs_ormc'),
        ee.ImageCollection("projects/sat-io/open-datasets/HiHydroSoilv2_0/stc").mode().rename('hhs_stc'),
        ee.ImageCollection("projects/sat-io/open-datasets/HiHydroSoilv2_0/wcavail").mean().rename('hhs_wcavail'),
        ee.ImageCollection("projects/sat-io/open-datasets/HiHydroSoilv2_0/wcpf2").mean().rename('hhs_wcpf2'),
        ee.ImageCollection("projects/sat-io/open-datasets/HiHydroSoilv2_0/wcpf3").mean().rename('hhs_wcpf3'),
        ee.ImageCollection("projects/sat-io/open-datasets/HiHydroSoilv2_0/wcpf4-2").mean().rename('hhs_wcpf4_2'),
        ee.ImageCollection("projects/sat-io/open-datasets/HiHydroSoilv2_0/wcres").mean().rename('hhs_wcres'),
        ee.ImageCollection("projects/sat-io/open-datasets/HiHydroSoilv2_0/wcsat").mean().rename('hhs_wcsat'),
    ])

    isric_bands = ee.Image.cat([
        ee.Image("projects/soilgrids-isric/bdod_mean"),
        ee.Image("projects/soilgrids-isric/cec_mean"),
        ee.Image("projects/soilgrids-isric/cfvo_mean"),
        ee.Image("projects/soilgrids-isric/clay_mean"),
        ee.Image("projects/soilgrids-isric/sand_mean"),
        ee.Image("projects/soilgrids-isric/silt_mean"),
        ee.Image("projects/soilgrids-isric/nitrogen_mean"),
        ee.Image("projects/soilgrids-isric/phh2o_mean"),
        ee.Image("projects/soilgrids-isric/soc_mean"),
        ee.Image("projects/soilgrids-isric/ocd_mean"),
        ee.Image("projects/soilgrids-isric/ocs_mean"),
    ])

    hwsd2_bands = ee.Image("projects/sat-io/open-datasets/FAO/HWSD_V2_SMU")
    hwsd2_bands = hwsd2_bands.select(
        ['HWSD2_ID', 'WISE30s_ID', 'COVERAGE', 'SHARE', 'WRB4', 'WRB_PHASES', 'WRB2_CODE', 'FAO90', 'KOPPEN',
         'TEXTURE_USDA', 'REF_BULK_DENSITY', 'BULK_DENSITY', 'DRAINAGE', 'ROOT_DEPTH', 'AWC'])

    c3s_lc_coll = (ee.ImageCollection("projects/sat-io/open-datasets/ESA/C3S-LC-L4-LCCS")
                   .filterDate(f'{start_yr}-01-01', f'{end_yr}-12-31')
                   .select('lccs_class'))
    c3s_lc_mode = c3s_lc_coll.mode().rename('c3s_lccs_class_mode')

    from_glc10_img = (ee.ImageCollection('projects/sat-io/open-datasets/FROM-GLC10')
                      .mosaic().rename('glc10_lc'))

    nlcd = None
    cdl_bands = None
    ssurgo = None
    prism = None
    additional_terrain = None
    us_lith = None
    polaris = None

    if region == 'conus':
        us_lith = ee.Image('CSP/ERGo/1_0/US/lithology').rename('us_lith')

        nlcd = ee.ImageCollection('USGS/NLCD_RELEASES/2019_REL/NLCD').select('landcover') \
            .mosaic().rename('nlcd')

        def get_all_cdl_bands(y):
            return ee.Image.cat(get_cdl(y))

        cdl_collection = ee.ImageCollection(years.map(get_all_cdl_bands))
        cdl_bands = cdl_collection.mode().rename(
            ['cdl_cultivated_mode', 'cdl_crop_mode', 'cdl_simple_crop_mode']
        )

        ssurgo = ee.Image.cat([
            ee.Image('projects/earthengine-legacy/assets/projects/openet/soil/ssurgo_AWC_WTA_0to152cm_composite')
            .rename('ssurgo_awc'),
            ee.Image('projects/earthengine-legacy/assets/projects/openet/soil/ssurgo_Clay_WTA_0to152cm_composite')
            .rename('ssurgo_clay'),
            ee.Image('projects/earthengine-legacy/assets/projects/openet/soil/ssurgo_Ksat_WTA_0to152cm_composite')
            .rename('ssurgo_ksat'),
            ee.Image('projects/earthengine-legacy/assets/projects/openet/soil/ssurgo_Sand_WTA_0to152cm_composite')
            .rename('ssurgo_sand')
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
                    tpi_10000,
                    tpi_22500,
                    gsw,
                    hihydro_bands,
                    isric_bands,
                    hwsd2_bands,
                    c3s_lc_mode,
                    from_glc10_img]

    if region == 'conus':
        static_bands.extend([nlcd, cdl_bands, ssurgo, prism, additional_terrain, us_lith, polaris])

    input_bands = input_bands.addBands(static_bands)

    if subselection:
        input_bands = input_bands.select(subselection)

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
