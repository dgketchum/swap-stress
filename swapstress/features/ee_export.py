import os

import ee
import geopandas as gpd
from shapely.geometry import box

from swapstress.features.call_ee import stack_bands_climatology, is_authorized


def _export_tile_data(
    roi,
    points,
    desc,
    bucket,
    file_prefix,
    resolution,
    index_col,
    region,
    diagnose=False,
):
    """Helper function to run and export data for a given ROI and point set."""
    stack = stack_bands_climatology(roi, region=region)

    if points.size().eq(0).getInfo():
        print(f"{desc}: no points to sample, skipping.")
        return

    # Optional diagnostic: probe one point and check band-by-band values
    if diagnose:
        try:
            print(desc)
            filtered = ee.FeatureCollection([points.first()])
            bad_ = []
            bands = stack.bandNames().getInfo()
            for b in bands:
                sel = stack.select([b])
                sample = sel.sampleRegions(
                    collection=filtered, properties=[], scale=resolution
                ).first()
                val = ee.Algorithms.If(sample, ee.Feature(sample).get(b), None)
                try:
                    info = ee.Dictionary({"v": val}).get("v").getInfo()
                    print(b, info)
                    if info is None:
                        bad_.append(b)
                except Exception as e:
                    print(b, "not there", e)
                    bad_.append(b)
            print("Bands with None or errors:", bad_)
        except Exception as e:
            print(f"Diagnostic failed for {desc}: {e}")
        return

    samples = stack.sampleRegions(
        collection=points,
        properties=["MGRS_TILE", index_col],
        scale=resolution,
        tileScale=16,
    )

    band_names = stack.bandNames()
    selectors = ["MGRS_TILE", index_col] + band_names.getInfo()

    task = ee.batch.Export.table.toCloudStorage(
        samples,
        description=desc,
        bucket=bucket,
        fileNamePrefix=f"{file_prefix}/{desc}",
        fileFormat="CSV",
        selectors=selectors,
    )
    task.start()
    print(f"Started export: {file_prefix}/{desc} (task: {task.id})")


def get_bands(
    shapefile_path,
    mgrs_shp_path,
    bucket,
    file_prefix,
    resolution,
    index_col=None,
    split_tiles=False,
    check_dir=None,
    region="conus",
    diagnose=False,
):
    """
    Extract climatological data for a set of points from a local shapefile.
    """
    points_df = gpd.read_file(shapefile_path)
    bounds = tuple(points_df.total_bounds)  # (minx, miny, maxx, maxy)
    mgrs_gdf = gpd.read_file(mgrs_shp_path, bbox=bounds)

    if index_col not in points_df.columns:
        raise ValueError(f"Index column '{index_col}' not found in shapefile.")

    mgrs_tiles = points_df["MGRS_TILE"].sample(frac=1).unique()
    print(f"{len(mgrs_tiles)} MGRS tiles to process", flush=True)

    empty_tiles = 0
    pts_identified = 0

    for tile in mgrs_tiles:
        desc = f"swapstress_{tile}"

        if check_dir:
            expected_path = os.path.join(check_dir, f"{desc}.csv")
            if os.path.exists(expected_path):
                print(f"File already exists: {expected_path}. Skipping export.")
                continue

        tile_df = points_df[points_df["MGRS_TILE"] == tile]
        print(f"{desc}.csv", len(tile_df))
        if tile_df.empty:
            empty_tiles += 1
            continue
        else:
            pts_identified += len(tile_df)

        tile_points = ee.FeatureCollection(tile_df.__geo_interface__)

        mgrs_tile_gdf = mgrs_gdf[mgrs_gdf["MGRS_TILE"] == tile]
        if mgrs_tile_gdf.empty:
            print(f"Warning: MGRS tile {tile} not found in {mgrs_shp_path}. Skipping.")
            continue

        if split_tiles:
            mgrs_tile_gdf = mgrs_gdf[mgrs_gdf["MGRS_TILE"] == tile]
            if mgrs_tile_gdf.empty:
                print(
                    f"Warning: MGRS tile {tile} not found in {mgrs_shp_path}. Skipping."
                )
                continue

            min_lon, min_lat, max_lon, max_lat = mgrs_tile_gdf.geometry.iloc[0].bounds
            center_lon = (min_lon + max_lon) / 2
            center_lat = (min_lat + max_lat) / 2

            quadrants = {
                "SW": box(min_lon, min_lat, center_lon, center_lat),
                "SE": box(center_lon, min_lat, max_lon, center_lat),
                "NW": box(min_lon, center_lat, center_lon, max_lat),
                "NE": box(center_lon, center_lat, max_lon, max_lat),
            }

            for name, geom in quadrants.items():
                q_desc = f"swapstress_{tile}_{name}"
                roi_ee_geom = ee.Geometry(geom.__geo_interface__)

                # Filter points to quadrant bounds (skip empty)
                q_points = tile_points.filterBounds(roi_ee_geom)
                if q_points.size().eq(0).getInfo():
                    print(f"{q_desc}: no points in ROI, skipping.")
                    continue

                _export_tile_data(
                    roi=roi_ee_geom,  # pass Geometry (not FC)
                    points=q_points,
                    desc=q_desc,
                    bucket=bucket,
                    file_prefix=file_prefix,
                    resolution=resolution,
                    index_col=index_col,
                    region=region,
                    diagnose=diagnose,
                )
        else:
            geo_json = mgrs_tile_gdf.geometry.iloc[0].__geo_interface__
            roi_ee_geom = ee.Geometry(geo_json)

            tile_points_bounded = tile_points.filterBounds(roi_ee_geom)

            _export_tile_data(
                roi=roi_ee_geom,
                points=tile_points_bounded,
                desc=desc,
                bucket=bucket,
                file_prefix=file_prefix,
                resolution=resolution,
                index_col=index_col,
                region=region,
                diagnose=diagnose,
            )

    print(f"{pts_identified} points identified for export")
    print(f"{empty_tiles} tiles missing")


# Stage 01: swapstress-extract
#
# Two steps. 'export' starts one Earth Engine batch task per MGRS tile, writing
# CSVs to Cloud Storage; those have to be synced down before 'tables' folds them
# into per-source parquets. They are one stage because neither is useful alone,
# but they are separately runnable because the wait between them is manual.
#
# Everything that used to be hardcoded per source -- shapefile, MGRS index,
# region, tile splitting, output prefix -- now comes from the registry.


def build_parser():
    import argparse

    from swapstress.cli import add_common_args
    from swapstress.sources.registry import DEFAULT_SOURCES, SOURCES, VALID_SCALES

    parser = argparse.ArgumentParser(
        prog="swapstress-extract",
        description="Stage 01: sample the covariate stack at every site, then "
        "fold the exports into per-source feature tables.",
    )
    add_common_args(parser)
    parser.add_argument(
        "--step",
        choices=["export", "tables", "all"],
        default=None,
        help="'export' starts the Earth Engine tasks; 'tables' converts the "
        "downloaded CSVs (default: export).",
    )
    parser.add_argument(
        "--sources",
        type=str,
        nargs="+",
        default=None,
        choices=sorted(SOURCES),
        help=f"Sources to extract (default: {' '.join(DEFAULT_SOURCES)}).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root data directory (default: /nas/soils).",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default=None,
        choices=VALID_SCALES,
        help="Resolution scale for the tables step (default: 9km_global).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Sampling resolution in metres (default: 250).",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=None,
        help="Cloud Storage bucket to export to (default: wudr).",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        default=None,
        help="Probe one point band by band and report nulls instead of exporting.",
    )
    return parser


def main(argv=None):
    from swapstress.cli import report_paths, resolve, stage_provenance
    from swapstress.features.ee_tables import build_tables
    from swapstress.sources.registry import DEFAULT_SOURCES, DataPaths, get_source

    config = resolve(build_parser(), argv)
    config.setdefault("step", "export")
    config.setdefault("sources", DEFAULT_SOURCES)
    config.setdefault("data_root", "/nas/soils")
    config.setdefault("scale", "9km_global")
    config.setdefault("resolution", 250)
    config.setdefault("bucket", "wudr")

    do_export = config["step"] in ("export", "all")
    do_tables = config["step"] in ("tables", "all")

    if config["dry_run"]:
        for name in config["sources"]:
            paths = DataPaths(config["data_root"], get_source(name), config["scale"])
            report_paths(
                f"01 extract [{name}]",
                {"sites": paths.shapefile, "mgrs index": paths.mgrs_shapefile},
                {
                    "gcs prefix": f"gs://{config['bucket']}/"
                    f"{paths.ee_output_prefix(config['resolution'])}",
                    "local extracts": paths.ee_extracts_dir,
                    "features table": paths.ee_table,
                },
            )
        return

    if do_export:
        is_authorized()
        for name in config["sources"]:
            source = get_source(name)
            paths = DataPaths(config["data_root"], source, config["scale"])
            print(f"\n=== Exporting {name} at {config['resolution']} m ===")
            get_bands(
                shapefile_path=paths.shapefile,
                mgrs_shp_path=paths.mgrs_shapefile,
                bucket=config["bucket"],
                file_prefix=paths.ee_output_prefix(config["resolution"]),
                resolution=config["resolution"],
                index_col=source.index_col,
                split_tiles=source.ee_split_tiles,
                check_dir=paths.ee_extracts_dir,
                region=source.ee_region,
                diagnose=config["diagnose"],
            )

    if do_tables:
        tables = build_tables(
            config["sources"],
            scale=config["scale"],
            data_root=config["data_root"],
        )
        if tables:
            stage_provenance(
                os.path.dirname(tables[0]),
                config,
                run_type="extract",
                extras={"outputs": tables},
            )


if __name__ == "__main__":
    main()

# ========================= EOF ====================================================================
