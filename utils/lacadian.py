import os

import pandas as pd
import geopandas as gpd


def build_lacadian_shapefile(metadata_csv, mgrs_shp_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    meta = pd.read_csv(metadata_csv)
    if meta.empty:
        print("No stations in metadata CSV")
        return

    geometry = gpd.points_from_xy(
        meta["longitude"].astype(float), meta["latitude"].astype(float)
    )
    gdf = gpd.GeoDataFrame(meta, geometry=geometry, crs="EPSG:4326")

    mgrs = gpd.read_file(mgrs_shp_path)
    if mgrs.crs != gdf.crs:
        mgrs = mgrs.to_crs(gdf.crs)

    joined = gpd.sjoin(
        gdf, mgrs[["MGRS_TILE", "geometry"]], how="inner", predicate="intersects"
    ).drop(columns=["index_right"])

    shp_path = os.path.join(out_dir, "lacadian_stations_mgrs.shp")
    csv_path = os.path.join(out_dir, "lacadian_stations_mgrs.csv")

    joined.to_file(shp_path)
    joined.drop(columns=["geometry"]).to_csv(csv_path, index=False)
    print(f"wrote {shp_path}")
    print(f"{joined.shape[0]} stations")


if __name__ == "__main__":
    root_ = "/nas"

    metadata_csv_ = os.path.join(
        root_, "soils", "soil_potential_obs", "lacadian", "station_metadata.csv"
    )
    mgrs_shp_path_ = os.path.join(root_, "boundaries", "mgrs", "mgrs_world_attr.shp")
    out_dir_ = os.path.join(root_, "soils", "soil_potential_obs", "lacadian")

    build_lacadian_shapefile(metadata_csv_, mgrs_shp_path_, out_dir_)

# ========================= EOF ====================================================================
