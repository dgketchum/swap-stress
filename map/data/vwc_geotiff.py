import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import geopandas as gpd
import pandas as pd
import rasterio
from tqdm import tqdm
import pandas as pd
import random


def _list_vwc_files(vwc_dir):
    files = glob(os.path.join(vwc_dir, 'vwc_*.tif'))
    mapping = {}
    for fp in files:
        name = os.path.basename(fp)
        try:
            date_str = name.split('vwc_')[1].split('.tif')[0]
        except Exception:
            continue  # likely error: unexpected filename pattern
        mapping[pd.to_datetime(date_str)] = fp
    return mapping


def _sample_value(raster_path, x, y):
    with rasterio.open(raster_path) as src:
        val = list(src.sample([(x, y)]))[0]
    return float(val[0])


def _worker_point(args):
    pid, x, y, shallow_map, middle_map, out_dir, overwrite = args

    out_fp = os.path.join(out_dir, f"{pid}.parquet")
    if os.path.exists(out_fp) and not overwrite:
        return out_fp

    all_dates = sorted(set(shallow_map.keys()) | set(middle_map.keys()))
    if not all_dates:
        return out_fp

    order = list(all_dates)
    random.shuffle(order)

    s_vals = []
    m_vals = []
    for d in order:
        sp = shallow_map.get(d)
        mp = middle_map.get(d)
        sv = float('nan') if sp is None else _sample_value(sp, x, y)
        mv = float('nan') if mp is None else _sample_value(mp, x, y)
        s_vals.append(sv)
        m_vals.append(mv)

    df = pd.DataFrame({'shallow': s_vals, 'middle': m_vals}, index=pd.DatetimeIndex(order, name='date'))
    df = df.sort_index()
    if df[['shallow', 'middle']].isna().any().any():
        return out_fp
    df.to_parquet(out_fp)
    return out_fp


def extract_vwc_timeseries(points_shp, shallow_dir, middle_dir, out_dir, index_col,
                           num_workers=4, overwrite=False, debug=False):
    print("Reading points and reprojecting to EPSG:4326...")
    gdf = gpd.read_file(points_shp)
    gdf = gdf.to_crs(4326)

    if index_col not in gdf.columns:
        raise ValueError(f"Index column '{index_col}' not found in shapefile.")

    ids = gdf[index_col].astype(str).to_list()
    xs = gdf.geometry.x.to_list()
    ys = gdf.geometry.y.to_list()

    os.makedirs(out_dir, exist_ok=True)

    print("Indexing VWC rasters (shallow and middle)...")
    shallow_map = _list_vwc_files(shallow_dir)
    middle_map = _list_vwc_files(middle_dir)
    print(f"Found {len(shallow_map)} shallow and {len(middle_map)} middle rasters.")

    tasks = [(pid, x, y, shallow_map, middle_map, out_dir, overwrite) for pid, x, y in zip(ids, xs, ys)]

    print(f"Extracting VWC rasters to points in {points_shp}")

    if debug:
        for t in tqdm(tasks):
            _worker_point(t)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = {ex.submit(_worker_point, t): t[0] for t in tasks}
            for f in tqdm(as_completed(futures), total=len(futures)):
                try:
                    _ = f.result()
                except Exception as e:
                    print(f"A point generated an exception: {e}")

    print("Done.")


if __name__ == '__main__':
    run_mt_mesonet_workflow = True
    run_rosetta_workflow = True
    run_gshp_workflow = True
    run_reesh_workflow = True

    home = os.path.expanduser('~')
    root_ = os.path.join(home, 'data', 'IrrigationGIS')

    vwc_ = '/data/ssd2/swapstress/vwc'
    out_dir_ = os.path.join(vwc_, 'hhp')
    gridmet_dir = os.path.join(vwc_, 'gridmet')

    clean_empty = True

    shallow_dir_ = '/data/ssd4/soil-moisture-ml-inference/predictions-smoothed-daily-shallow/'
    middle_dir_ = '/data/ssd4/soil-moisture-ml-inference/predictions-smoothed-daily-middle/'

    gridmet_vars_ = ['pr', 'pet', 'vpd', 'srad', 'tmmx', 'tmmn']
    gridmet_ = '/data/ssd2/gridmet/pr_raw/pr_2001.nc'

    if run_mt_mesonet_workflow:
        points_shp_ = os.path.join(root_, 'soils', 'soil_potential_obs', 'mt_mesonet', 'station_metadata_mgrs.shp')
        vwc_out_ = os.path.join(out_dir_, 'mt_mesonet')
        extract_vwc_timeseries(points_shp=points_shp_,
                               shallow_dir=shallow_dir_,
                               middle_dir=middle_dir_,
                               out_dir=vwc_out_,
                               index_col='station',
                               num_workers=36,
                               overwrite=False,
                               debug=False)

    if run_gshp_workflow:
        points_shp_ = os.path.join(root_, 'soils', 'soil_potential_obs', 'gshp', 'wrc_aggregated_mgrs.shp')
        vwc_out_ = os.path.join(out_dir_, 'gshp')
        extract_vwc_timeseries(points_shp=points_shp_,
                               shallow_dir=shallow_dir_,
                               middle_dir=middle_dir_,
                               out_dir=vwc_out_,
                               index_col='profile_id',
                               num_workers=36,
                               overwrite=False,
                               debug=False)

    if run_reesh_workflow:
        points_shp_ = os.path.join(root_, 'soils', 'soil_potential_obs', 'reesh', 'shapefile', 'reesh_sites_mgrs.shp')
        vwc_out_ = os.path.join(out_dir_, 'reesh')
        extract_vwc_timeseries(points_shp=points_shp_,
                               shallow_dir=shallow_dir_,
                               middle_dir=middle_dir_,
                               out_dir=vwc_out_,
                               index_col='site_id',
                               num_workers=36,
                               overwrite=False,
                               debug=False)

    if run_rosetta_workflow:
        points_shp_ = os.path.join(root_, 'soils', 'gis', 'pretraining-roi-10000_mgrs.shp')
        vwc_out_ = os.path.join(out_dir_, 'rosetta')
        extract_vwc_timeseries(points_shp=points_shp_,
                               shallow_dir=shallow_dir_,
                               middle_dir=middle_dir_,
                               out_dir=vwc_out_,
                               index_col='site_id',
                               num_workers=36,
                               overwrite=False,
                               debug=False)

# ========================= EOF ====================================================================
