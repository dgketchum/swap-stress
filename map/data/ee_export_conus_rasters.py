"""
Export static covariate rasters and point extractions from Google Earth Engine.

Exports feature groups as separate GeoTIFFs aligned to the SMAP 9 km EASE-Grid2
(EPSG:6933), or samples the same images at training site locations producing CSVs
with values matching the raster exports exactly.

Usage:
    # Export all raster groups
    python -m map.data.ee_export_conus_rasters --mode rasters --groups all

    # Export specific raster groups
    python -m map.data.ee_export_conus_rasters --mode rasters --groups soilgrids_shallow,worldclim

    # Extract point values at 9km resolution
    python -m map.data.ee_export_conus_rasters --mode points \
        --shapefile /path/to/sites.shp --index-col site_id
"""

import argparse
import os

import ee
import geopandas as gpd

from map.data.call_ee import get_world_climate, is_authorized
from map.data.cdl import remap_cdl
from map.data.ee_utils import landsat_composites
from map.data.smap_download import MAP_SCALE, _conus_slice, _conus_transform

GCS_BUCKET = "wudr"
GCS_PREFIX = "conus_features"
EASE2_CRS = "EPSG:6933"

_NAS_ROOT = "/nas"
_LOCAL_ROOT = os.path.expanduser("~/data/IrrigationGIS")


def _data_root():
    """Return /nas if mounted, else fall back to ~/data/IrrigationGIS."""
    if os.path.isdir(os.path.join(_NAS_ROOT, "soils")):
        return _NAS_ROOT
    return _LOCAL_ROOT


START_YR = 1991
END_YR = 2020


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------


def _grid_params():
    """Compute CONUS EASE-Grid2 9 km export parameters from SMAP grid constants."""
    row_sl, col_sl = _conus_slice()
    transform = _conus_transform(row_sl, col_sl)
    width = col_sl.stop - col_sl.start
    height = row_sl.stop - row_sl.start
    x_origin = transform.c
    y_origin = transform.f
    crs_transform = [MAP_SCALE, 0, x_origin, 0, -MAP_SCALE, y_origin]
    roi = ee.Geometry.Rectangle(
        [
            x_origin,
            y_origin - height * MAP_SCALE,
            x_origin + width * MAP_SCALE,
            y_origin,
        ],
        proj=EASE2_CRS,
        geodesic=False,
    )
    return roi, crs_transform, width, height


# ---------------------------------------------------------------------------
# Feature group builders — each returns an ee.Image
# ---------------------------------------------------------------------------


def build_soilgrids_shallow(roi):
    """SoilGrids ISRIC at 0-5 cm and 5-15 cm depths (~22 bands)."""
    isric = ee.Image.cat(
        [
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
        ]
    )
    shallow_bands = isric.bandNames().filter(
        ee.Filter.Or(
            ee.Filter.stringContains("item", "0-5cm"),
            ee.Filter.stringContains("item", "5-15cm"),
        )
    )
    return isric.select(shallow_bands)


def build_worldclim(roi):
    """WorldClim seasonal climate + annual ET0 (~22 bands)."""
    seasons = [
        ("winter", (12, 2)),
        ("spring", (3, 5)),
        ("summer", (6, 8)),
        ("autumn", (9, 11)),
    ]
    bands = []
    for s_name, s_months in seasons:
        for p_name in ["prec", "tavg", "tmin", "tmax"]:
            clim = get_world_climate(s_months, param=p_name)
            bands.append(clim.rename(f"wc_{p_name}_{s_name}"))
        eto = get_world_climate(s_months, param="eto", band_name="b1")
        bands.append(eto.rename(f"eto_{s_name}"))

    et_yearly = ee.Image(
        "projects/sat-io/open-datasets/global_et0/global_et0_yearly"
    ).rename("eto_yearly_mean")
    et_yearly_sd = ee.Image(
        "projects/sat-io/open-datasets/global_et0/global_et0_yearly_sd"
    ).rename("eto_yearly_sd")
    bands.extend([et_yearly, et_yearly_sd])
    return ee.Image.cat(bands)


def build_smap_l3_vwc_clim(roi):
    """SMAP L3 vegetation water content climatology (4 bands)."""
    smap_l3 = (
        ee.ImageCollection("NASA/SMAP/SPL3SMP_E/005")
        .filterDate("2018-01-01", "2023-12-31")
        .filterBounds(roi)
    )
    reducers = ee.Reducer.mean().combine(ee.Reducer.stdDev(), "", True)
    am = smap_l3.select(["vegetation_water_content_am"]).reduce(reducers)
    pm = smap_l3.select(["vegetation_water_content_pm"]).reduce(reducers)
    return ee.Image.cat([am, pm])


def build_landcover(roi):
    """Land cover composites: C3S, GLC10, GSW, NLCD, CDL (~8 bands)."""
    gsw = (
        ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
        .select("occurrence")
        .gt(0)
        .unmask(0)
        .rename("gsw")
    )
    c3s = (
        ee.ImageCollection("projects/sat-io/open-datasets/ESA/C3S-LC-L4-LCCS")
        .filterDate(f"{START_YR}-01-01", f"{END_YR}-12-31")
        .select("b1")
        .mode()
        .rename("c3s_lccs_class_mode")
    )
    glc10 = (
        ee.ImageCollection("projects/sat-io/open-datasets/FROM-GLC10")
        .mosaic()
        .rename("glc10_lc")
    )
    nlcd = (
        ee.ImageCollection("USGS/NLCD_RELEASES/2019_REL/NLCD")
        .select("landcover")
        .mosaic()
        .rename("nlcd")
    )

    # CDL multi-year mode composites
    cultivated_years = list(range(2013, 2019))
    cdl_years = list(range(2008, 2021))

    cultivated_imgs = [
        ee.Image(f"USDA/NASS/CDL/{y}").select("cultivated").remap([1, 2], [0, 1])
        for y in cultivated_years
    ]
    cultivated = (
        ee.ImageCollection.fromImages(cultivated_imgs)
        .reduce(ee.Reducer.mode())
        .resample("bilinear")
        .rename("cdl_cultivated_mode")
    )

    crop_imgs = [ee.Image(f"USDA/NASS/CDL/{y}").select("cropland") for y in cdl_years]
    crop_mode = (
        ee.ImageCollection.fromImages(crop_imgs)
        .reduce(ee.Reducer.mode())
        .rename("cdl_crop_mode")
    )

    cdl_keys, our_keys = remap_cdl()
    simple_crop = (
        crop_mode.remap(cdl_keys, our_keys)
        .rename("cdl_simple_crop_mode")
        .resample("bilinear")
    )

    return ee.Image.cat([gsw, c3s, glc10, nlcd, cultivated, crop_mode, simple_crop])


def build_fao_hwsd(roi):
    """FAO Harmonized World Soil Database v2 (15 bands)."""
    return ee.Image("projects/sat-io/open-datasets/FAO/HWSD_V2_SMU").select(
        [
            "HWSD2_ID",
            "WISE30s_ID",
            "COVERAGE",
            "SHARE",
            "WRB4",
            "WRB_PHASES",
            "WRB2_CODE",
            "FAO90",
            "KOPPEN",
            "TEXTURE_USDA",
            "REF_BULK_DENSITY",
            "BULK_DENSITY",
            "DRAINAGE",
            "ROOT_DEPTH",
            "AWC",
        ]
    )


def build_landsat_bands(roi):
    """Landsat reflectance composites, 5 periods, bands only (~70 bands)."""
    periods = [
        ("gs", 121, 273),
        ("1", 60, 121),
        ("2", 121, 196),
        ("3", 196, 273),
        ("4", 273, 365),
    ]

    composites = []
    for name, start_doy, end_doy in periods:
        comp = landsat_composites(START_YR, END_YR, start_doy, end_doy, roi, name)
        composites.append(comp)

    all_bands = ee.Image.cat(composites)

    # Keep only reflectance bands (drop derived indices: nd, nw, evi, gi)
    raw_bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B10"]
    stats = ["mean", "stdDev"]
    select_names = []
    for name, _, _ in periods:
        for band in raw_bands:
            for stat in stats:
                select_names.append(f"{band}_{stat}_{name}")

    return all_bands.select(select_names)


def build_prism(roi):
    """PRISM 30-year climate normals (7 bands)."""
    return ee.ImageCollection("OREGONSTATE/PRISM/Norm91m").mean()


def build_ssurgo(roi):
    """SSURGO soil properties (4 bands)."""
    return ee.Image.cat(
        [
            ee.Image(
                "projects/earthengine-legacy/assets/projects/openet/soil/ssurgo_AWC_WTA_0to152cm_composite"
            ).rename("ssurgo_awc"),
            ee.Image(
                "projects/earthengine-legacy/assets/projects/openet/soil/ssurgo_Clay_WTA_0to152cm_composite"
            ).rename("ssurgo_clay"),
            ee.Image(
                "projects/earthengine-legacy/assets/projects/openet/soil/ssurgo_Ksat_WTA_0to152cm_composite"
            ).rename("ssurgo_ksat"),
            ee.Image(
                "projects/earthengine-legacy/assets/projects/openet/soil/ssurgo_Sand_WTA_0to152cm_composite"
            ).rename("ssurgo_sand"),
        ]
    )


def build_polaris(roi):
    """POLARIS soil properties (13 bands)."""
    return ee.Image.cat(
        [
            ee.ImageCollection("projects/sat-io/open-datasets/polaris/bd_mean")
            .mean()
            .rename("bd_mean"),
            ee.ImageCollection("projects/sat-io/open-datasets/polaris/clay_mean")
            .mean()
            .rename("clay_mean"),
            ee.ImageCollection("projects/sat-io/open-datasets/polaris/ksat_mean")
            .mean()
            .rename("ksat_mean"),
            ee.ImageCollection("projects/sat-io/open-datasets/polaris/n_mean")
            .mean()
            .rename("n_mean"),
            ee.ImageCollection("projects/sat-io/open-datasets/polaris/om_mean")
            .mean()
            .rename("om_mean"),
            ee.ImageCollection("projects/sat-io/open-datasets/polaris/ph_mean")
            .mean()
            .rename("ph_mean"),
            ee.ImageCollection("projects/sat-io/open-datasets/polaris/sand_mean")
            .mean()
            .rename("sand_mean"),
            ee.ImageCollection("projects/sat-io/open-datasets/polaris/silt_mean")
            .mean()
            .rename("silt_mean"),
            ee.ImageCollection("projects/sat-io/open-datasets/polaris/theta_r_mean")
            .mean()
            .rename("theta_r_mean"),
            ee.ImageCollection("projects/sat-io/open-datasets/polaris/theta_s_mean")
            .mean()
            .rename("theta_s_mean"),
            ee.ImageCollection("projects/sat-io/open-datasets/polaris/lambda_mean")
            .mean()
            .rename("lambda_mean"),
            ee.ImageCollection("projects/sat-io/open-datasets/polaris/hb_mean")
            .mean()
            .rename("hb_mean"),
            ee.ImageCollection("projects/sat-io/open-datasets/polaris/alpha_mean")
            .mean()
            .rename("alpha_mean"),
        ]
    )


def build_terrain(roi):
    """Terrain: elevation, slope, aspect, TPI, TWI, topoDiversity, lithology (~8 bands)."""
    dem = ee.Image("USGS/3DEP/10m")
    terrain = ee.Terrain.products(dem).select("elevation", "slope", "aspect")
    tpi_10000 = (
        dem.subtract(dem.focal_mean(10000, "circle", "meters"))
        .add(0.5)
        .rename("tpi_10000")
    )
    tpi_22500 = (
        dem.subtract(dem.focal_mean(22500, "circle", "meters"))
        .add(0.5)
        .rename("tpi_22500")
    )
    twi = ee.Image("users/zhoylman/CONUS_TWI_epsg5072_30m")
    topo_div = ee.Image("CSP/ERGo/1_0/US/topoDiversity")
    us_lith = ee.Image("CSP/ERGo/1_0/US/lithology").rename("us_lith")
    return ee.Image.cat([terrain, tpi_10000, tpi_22500, twi, topo_div, us_lith])


def build_sentinel1(roi):
    """Sentinel-1 VV/VH/ratio climatology (6 bands)."""
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterDate("2015-01-01", "2024-12-31")
        .filterBounds(roi)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
    )

    def _add_ratio(img):
        return img.addBands(img.select("VH").divide(img.select("VV")).rename("VH_VV"))

    s1 = s1.map(_add_ratio).select(["VV", "VH", "VH_VV"])
    s1_mean = s1.mean().rename(["VV_mean", "VH_mean", "VH_VV_mean"])
    s1_std = s1.reduce(ee.Reducer.stdDev()).rename(
        ["VV_stdDev", "VH_stdDev", "VH_VV_stdDev"]
    )
    return ee.Image.cat([s1_mean, s1_std])


# ---------------------------------------------------------------------------
# Feature group registry
# ---------------------------------------------------------------------------

FEATURE_GROUPS = {
    "soilgrids_shallow": (build_soilgrids_shallow, "soilgrids_9km"),
    "worldclim": (build_worldclim, "worldclim_9km"),
    "smap_l3_vwc_clim": (build_smap_l3_vwc_clim, "smap_l3_clim_9km"),
    "landcover": (build_landcover, "landcover_9km"),
    "fao_hwsd": (build_fao_hwsd, "fao_hwsd_9km"),
    "landsat_bands": (build_landsat_bands, "landsat_bands_9km"),
    "prism": (build_prism, "prism_normals_9km"),
    "ssurgo": (build_ssurgo, "ssurgo_9km"),
    "polaris": (build_polaris, "polaris_9km"),
    "terrain": (build_terrain, "terrain_9km"),
    "sentinel1": (build_sentinel1, "sentinel1_9km"),
}


# ---------------------------------------------------------------------------
# Export modes
# ---------------------------------------------------------------------------


def export_rasters(groups, bucket, prefix):
    """Submit EE Export.image tasks for each feature group."""
    roi, crs_transform, width, height = _grid_params()

    for name, (build_fn, filename) in groups.items():
        image = build_fn(roi)
        task = ee.batch.Export.image.toCloudStorage(
            image=image.clip(roi).toFloat(),
            description=f"conus_{name}_9km",
            bucket=bucket,
            fileNamePrefix=f"{prefix}/{filename}",
            crs=EASE2_CRS,
            crsTransform=crs_transform,
            dimensions=f"{width}x{height}",
            maxPixels=int(1e13),
            fileFormat="GeoTIFF",
        )
        task.start()
        print(f"Started raster export: {prefix}/{filename} (task: {task.id})")


def export_points(groups, shapefile, index_col, bucket, prefix):
    """Sample feature groups at point locations on the 9 km grid and export CSV."""
    roi, _, _, _ = _grid_params()

    gdf = gpd.read_file(shapefile)
    if index_col not in gdf.columns:
        raise ValueError(f"Column '{index_col}' not found in {shapefile}")

    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    points = ee.FeatureCollection(gdf.__geo_interface__)
    stack = ee.Image.cat([build_fn(roi) for build_fn, _ in groups.values()])

    samples = stack.sampleRegions(
        collection=points,
        properties=[index_col],
        scale=MAP_SCALE,
        crs=EASE2_CRS,
        tileScale=16,
    )

    task = ee.batch.Export.table.toCloudStorage(
        samples,
        description="conus_point_extract_9km",
        bucket=bucket,
        fileNamePrefix=f"{prefix}/point_extract_9km",
        fileFormat="CSV",
    )
    task.start()
    print(f"Started point export: {prefix}/point_extract_9km (task: {task.id})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_groups(group_str):
    """Parse --groups flag into a dict subset of FEATURE_GROUPS."""
    if group_str == "all":
        return dict(FEATURE_GROUPS)
    names = [g.strip() for g in group_str.split(",")]
    resolved = {}
    for n in names:
        if n not in FEATURE_GROUPS:
            available = ", ".join(FEATURE_GROUPS.keys())
            raise ValueError(f"Unknown group '{n}'. Available: {available}")
        resolved[n] = FEATURE_GROUPS[n]
    return resolved


def main():
    parser = argparse.ArgumentParser(
        description="Export EE static covariates aligned to SMAP 9 km EASE-Grid2",
    )
    parser.add_argument("--mode", required=True, choices=["rasters", "points"])
    parser.add_argument(
        "--groups", default="all", help="Comma-separated group names or 'all'"
    )
    parser.add_argument(
        "--shapefile", help="Point shapefile (required for points mode)"
    )
    parser.add_argument(
        "--index-col", help="ID column in shapefile (required for points mode)"
    )
    parser.add_argument("--project", default="ee-dgketchum", help="EE project ID")
    parser.add_argument("--bucket", default=GCS_BUCKET)
    parser.add_argument("--prefix", default=GCS_PREFIX)
    args = parser.parse_args()

    if args.mode == "points" and (not args.shapefile or not args.index_col):
        parser.error("--shapefile and --index-col required for points mode")

    root = _data_root()
    print(f"Data root: {root}")

    is_authorized(project=args.project)
    groups = _resolve_groups(args.groups)
    print(f"Groups: {', '.join(groups.keys())}")

    if args.mode == "rasters":
        export_rasters(groups, args.bucket, args.prefix)
    else:
        export_points(groups, args.shapefile, args.index_col, args.bucket, args.prefix)


if __name__ == "__main__":
    main()

# ========================= EOF ====================================================================
