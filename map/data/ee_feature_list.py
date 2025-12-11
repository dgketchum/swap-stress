import re
import pandas as pd


_BAND_LABELS = {
    'B2': 'Landsat B2 (blue)',
    'B3': 'Landsat B3 (green)',
    'B4': 'Landsat B4 (red)',
    'B5': 'Landsat B5 (NIR)',
    'B6': 'Landsat B6 (SWIR1)',
    'B7': 'Landsat B7 (SWIR2)',
    'B10': 'Landsat B10 (thermal)'
}

_INDEX_LABELS = {
    'nd': 'NDVI',
    'nw': 'NDWI',
    'evi': 'EVI',
    'gi': 'Green Index'
}

_SEASON_LABELS = {
    'gs': 'growing season',
    '1': 'Q1', '2': 'Q2', '3': 'Q3', '4': 'Q4',
    'winter': 'winter', 'spring': 'spring', 'summer': 'summer', 'autumn': 'autumn'
}

_STAT_LABELS = {'mean': 'mean', 'stdDev': 'std dev', 'sd': 'std dev'}

_GRIDMET_VARS = {
    'pr': 'precipitation',
    'eto': 'ET0',
    'tmmn': 'tmin',
    'tmmx': 'tmax',
    'tmean': 'tmean'
}

_SMAP_L4 = {
    'sm_surface': 'Surface soil moisture',
    'sm_rootzone': 'Rootzone soil moisture',
    'sm_profile': 'Profile soil moisture',
    'surface_temp': 'Surface temperature',
    'leaf_area_index': 'Leaf area index'
}

_SOILGRIDS = {
    'bdod_mean': 'SoilGrids bulk density (mean)',
    'cec_mean': 'SoilGrids CEC (mean)',
    'cfvo_mean': 'SoilGrids coarse fragments (mean)',
    'clay_mean': 'SoilGrids clay (mean)',
    'sand_mean': 'SoilGrids sand (mean)',
    'silt_mean': 'SoilGrids silt (mean)',
    'nitrogen_mean': 'SoilGrids nitrogen (mean)',
    'phh2o_mean': 'SoilGrids pH H2O (mean)',
    'soc_mean': 'SoilGrids SOC (mean)',
    'ocd_mean': 'SoilGrids OCS density (mean)',
    'ocs_mean': 'SoilGrids OCS (mean)'
}

_FAO_SOILS = {
    'HWSD2_ID': 'HWSD v2 soil unit ID',
    'WISE30s_ID': 'WISE30s soil unit ID',
    'COVERAGE': 'HWSD v2 coverage fraction',
    'SHARE': 'HWSD v2 soil share',
    'WRB4': 'WRB4 soil class',
    'WRB_PHASES': 'WRB phases',
    'WRB2_CODE': 'WRB2 soil code',
    'FAO90': 'FAO 1990 soil unit',
    'KOPPEN': 'Köppen climate class',
    'TEXTURE_USDA': 'USDA soil texture class',
    'REF_BULK_DENSITY': 'Reference bulk density',
    'BULK_DENSITY': 'Bulk density',
    'DRAINAGE': 'Soil drainage class',
    'ROOT_DEPTH': 'Rooting depth',
    'AWC': 'Available water capacity',
}
_POLARIS = {
    'bd_mean': 'POLARIS bulk density (mean)',
    'clay_mean': 'POLARIS clay (mean)',
    'ksat_mean': 'POLARIS Ksat (mean)',
    'n_mean': 'POLARIS van Genuchten n (mean)',
    'om_mean': 'POLARIS organic matter (mean)',
    'ph_mean': 'POLARIS pH (mean)',
    'sand_mean': 'POLARIS sand (mean)',
    'silt_mean': 'POLARIS silt (mean)',
    'theta_r_mean': 'POLARIS theta_r (mean)',
    'theta_s_mean': 'POLARIS theta_s (mean)',
    'lambda_mean': 'POLARIS lambda (mean)',
    'hb_mean': 'POLARIS hb (mean)',
    'alpha_mean': 'POLARIS alpha (mean)'
}


def label_feature(n):
    x = str(n)

    m = re.match(r'^(B2|B3|B4|B5|B6|B7|B10|nd|nw|evi|gi)_(mean|stdDev)_(gs|[1234])$', x)
    if m:
        base, stat, per = m.groups()
        base_lbl = _BAND_LABELS.get(base, _INDEX_LABELS.get(base, base))
        s_lbl = _SEASON_LABELS.get(per, per)
        stat_lbl = _STAT_LABELS.get(stat, stat)
        lbl = f'{base_lbl} {stat_lbl} ({s_lbl})'
        return lbl

    m = re.match(r'^(pr|eto|tmmn|tmmx|tmean)_(mean|stdDev)_(winter|spring|summer|autumn)$', x)
    if m:
        v, stat, season = m.groups()
        v_lbl = _GRIDMET_VARS.get(v, v)
        stat_lbl = _STAT_LABELS.get(stat, stat)
        lbl = f'gridMET {v_lbl} {stat_lbl} ({season})'
        return lbl

    m = re.match(r'^(VV|VH|VH_VV)_(mean|stdDev)$', x)
    if m:
        pol, stat = m.groups()
        stat_lbl = _STAT_LABELS.get(stat, stat)
        lbl = f'Sentinel-1 {pol} {stat_lbl}'
        return lbl

    m = re.match(r'^vegetation_water_content_(am|pm)_(mean|stdDev)$', x)
    if m:
        ap, stat = m.groups()
        stat_lbl = _STAT_LABELS.get(stat, stat)
        lbl = f'SMAP L3 VWC ({ap.upper()}) {stat_lbl}'
        return lbl

    m = re.match(r'^(sm_surface|sm_rootzone|sm_profile|surface_temp|leaf_area_index)_(mean|stdDev)$', x)
    if m:
        v, stat = m.groups()
        stat_lbl = _STAT_LABELS.get(stat, stat)
        v_lbl = _SMAP_L4.get(v, v)
        lbl = f'SMAP L4 {v_lbl} {stat_lbl}'
        return lbl

    # SoilGrids depth-resolved layers, e.g., clay_100-200cm_mean
    m = re.match(r'^(bdod|cec|cfvo|clay|sand|silt|nitrogen|phh2o|soc|ocd|ocs)_(\d+-\d+cm)_(mean|stdDev|sd)$', x)
    if m:
        prop, depth, stat = m.groups()
        base_key = f'{prop}_mean'
        base_lbl = _SOILGRIDS.get(base_key, prop)
        base_lbl = base_lbl.replace(' (mean)', '')
        stat_lbl = _STAT_LABELS.get(stat, stat)
        lbl = f'{base_lbl} {stat_lbl} ({depth})'
        return lbl

    m = re.match(r'^wc_(prec|tavg|tmin|tmax)_(winter|spring|summer|autumn)$', x)
    if m:
        v, season = m.groups()
        v_lbl = {'prec': 'precipitation', 'tavg': 'tavg', 'tmin': 'tmin', 'tmax': 'tmax'}.get(v, v)
        lbl = f'WorldClim {v_lbl} ({season})'
        return lbl

    m = re.match(r'^eto_(winter|spring|summer|autumn)$', x)
    if m:
        season = m.groups()[0]
        lbl = f'Global ET0 ({season})'
        return lbl

    if x in ['eto_yearly_mean', 'eto_yearly_sd']:
        stat_lbl = 'mean' if x.endswith('mean') else 'std dev'
        lbl = f'Global ET0 yearly ({stat_lbl})'
        return lbl

    if x == 'lon':
        return 'Longitude'
    if x == 'lat':
        return 'Latitude'
    if x in ['elevation', 'slope', 'aspect']:
        lbl = x.capitalize()
        return lbl
    if x == 'tpi_10000':
        return 'Topographic position index (10 km)'
    if x == 'tpi_22500':
        return 'Topographic position index (22.5 km)'
    if x == 'gsw':
        return 'Global surface water presence'

    if x.startswith('hhs_'):
        lbl = f'HiHydroSoil v2 {x.replace("hhs_", "").replace("_", " ")}'
        return lbl

    if x in _SOILGRIDS:
        return _SOILGRIDS[x]

    if x in _POLARIS:
        return _POLARIS[x]

    if x in _FAO_SOILS:
        return _FAO_SOILS[x]

    if x == 'c3s_lccs_class_mode':
        return 'C3S LCCS class (mode)'
    if x == 'glc10_lc':
        return 'FROM-GLC10 land cover'
    if x == 'nlcd':
        return 'NLCD land cover'
    if x in ['cdl_cultivated_mode', 'cdl_crop_mode', 'cdl_simple_crop_mode']:
        lbl = x.replace('cdl_', 'CDL ').replace('_mode', ' (mode)').replace('_', ' ')
        return lbl
    if x in ['ssurgo_awc', 'ssurgo_clay', 'ssurgo_ksat', 'ssurgo_sand']:
        lbl = x.replace('ssurgo_', 'SSURGO ').replace('_', ' ')
        return lbl
    if x == 'us_lith':
        return 'US lithology'

    prism_bands = {'ppt': 'precipitation', 'tmin': 'tmin', 'tmax': 'tmax', 'tmean': 'tmean', 'tdmean': 'dewpoint mean',
                   'vpdmin': 'vpd min', 'vpdmax': 'vpd max'}
    if x in prism_bands:
        lbl = f'PRISM normals 1991–2020 {prism_bands[x]}'
        return lbl

    if re.match(r'^embedding_\d+$', x):
        i = int(x.split('_')[-1])
        lbl = f'Google Satellite Embedding {i}'
        return lbl

    m = re.match(r'^e(\d+)$', x)
    if m:
        i = int(m.group(1))
        lbl = f'Headwaters Hydrology Project Embedding {i}'
        return lbl

    m = re.match(r'^A(\d+)$', x)
    if m:
        i = int(m.group(1))
        lbl = f'AlphaEarth Foundations Satellite Embedding {i}'
        return lbl

    return x


def build_feature_label_map(features_csv):
    df = pd.read_csv(features_csv)
    col = 'features' if 'features' in df.columns else df.columns[0]
    feats = df[col].dropna().astype(str).tolist()

    bn = [f for f in feats if re.match(r'^b\d+$', f)]
    emb = [f for f in feats if re.match(r'^embedding_\d+$', f)]

    special = {}
    if len(emb) > 0 or len(bn) >= 32:
        for f in bn:
            i = int(f[1:])
            special[f] = f'Google Satellite Embedding {i}'
    else:
        if 'b1' in feats and 'b1_1' in feats:
            special['b1'] = 'Terrain wetness index'  # likely error: ambiguous origin of b1
            special['b1_1'] = 'Topographic diversity'

    mapping = {}
    for f in feats:
        if f in special:
            mapping[f] = special[f]
        else:
            lbl = label_feature(f)
            mapping[f] = lbl

    return mapping

# ========================= EOF ====================================================================
