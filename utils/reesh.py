import os
from glob import glob

import pandas as pd
import geopandas as gpd


def build_reesh_shapefile(in_dir, mgrs_shp_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob(os.path.join(in_dir, '*_SoilCharacteristics.csv')))  # assumes ReESH metadata naming
    if not files:
        return

    rows = []
    for fp in files:
        try:
            d = pd.read_csv(fp)
        except Exception:
            print(f'failed to read {fp}')
            continue
        if 'Latitude' not in d.columns or 'Longitude' not in d.columns:
            continue
        r0 = d.iloc[0].copy()
        r0 = r0[['Latitude', 'Longitude']].copy()
        r0['source'] = os.path.basename(fp)
        r0['site_id'] = os.path.basename(fp).replace('_SoilCharacteristics.csv', '')
        rows.append(r0)

    if not rows:
        return

    df = pd.DataFrame(rows)
    geometry = gpd.points_from_xy(df['Longitude'].astype(float), df['Latitude'].astype(float))
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

    mgrs = gpd.read_file(mgrs_shp_path)
    if mgrs.crs != gdf.crs:
        mgrs = mgrs.to_crs(gdf.crs)

    joined = gpd.sjoin(
        gdf,
        mgrs[['MGRS_TILE', 'geometry']],
        how='inner',
        predicate='intersects'
    ).drop(columns=['index_right'])

    shp_path = os.path.join(out_dir, 'reesh_sites_mgrs.shp')
    csv_path = os.path.join(out_dir, 'reesh_sites_mgrs.csv')

    joined.to_file(shp_path)
    joined.drop(columns=['geometry']).to_csv(csv_path, index=False)


if __name__ == '__main__':
    home_ = os.path.expanduser('~')
    root_ = os.path.join(home_, 'data', 'IrrigationGIS')

    in_dir_ = os.path.join(root_, 'soils', 'soil_potential_obs', 'reesh')
    mgrs_shp_path_ = os.path.join(root_, 'boundaries', 'mgrs', 'mgrs_world_attr.shp')
    out_dir_ = os.path.join(root_, 'soils', 'soil_potential_obs', 'reesh', 'shapefile')

    build_reesh_shapefile(in_dir_, mgrs_shp_path_, out_dir_)

# ========================= EOF ====================================================================
