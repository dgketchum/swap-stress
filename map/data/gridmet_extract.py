import os
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm
from map.data.thredds import GridMet


def _list_gridmet_files(base_dir, var):
    pattern = os.path.join(base_dir, f"{var}_raw", f"{var}_*.nc")
    files = glob(pattern)
    mapping = {}
    for fp in files:
        name = os.path.basename(fp)
        try:
            year = int(name.split(f"{var}_")[1].split('.nc')[0])
        except Exception:
            continue  # likely error: unexpected filename pattern
        mapping[year] = fp
    return mapping


def _get_data_var(ds, var_code):
    # Prefer variable whose long_name or standard_name matches var_code
    for v in ds.data_vars:
        attrs = ds[v].attrs
        if attrs.get('long_name') == var_code or attrs.get('standard_name') == var_code:
            return v
    # Fallback: first non-ancillary variable
    for v in ds.data_vars:
        if v.lower() != 'crs':
            return v
    return list(ds.data_vars)[0]


def _decode_times(ds):
    if 'day' in ds.variables or 'day' in ds.coords:
        day = ds['day']
        if np.issubdtype(day.dtype, np.datetime64):
            return pd.to_datetime(day.values)
        units = str(day.attrs.get('units', 'days since 1900-01-01'))
        # simple CF-like parsing for origin date
        origin = '1900-01-01'
        if 'since' in units:
            try:
                origin = units.split('since')[1].strip().split(' ')[0]
            except Exception:
                pass
        return pd.to_datetime(origin) + pd.to_timedelta(day.values, unit='D')
    # likely error: no 'day' coordinate present
    return None


def _extract_point_series(x, y, files_map, var_list):
    data = {}
    for var in var_list:
        year_map = files_map.get(var, {})
        if not year_map:
            continue
        parts = []
        for yr in sorted(year_map):
            fp = year_map[yr]
            ds = xr.open_dataset(fp, decode_times=True)
            vname = _get_data_var(ds, var)
            da = ds[vname]
            loc = da.sel(lon=x, lat=y, method='nearest')
            times = _decode_times(ds)
            if times is None:
                ds.close()
                continue
            vals = loc.values
            s = pd.Series(vals, index=pd.DatetimeIndex(times, name='date'))
            parts.append(s)
            ds.close()
        if parts:
            full = pd.concat(parts)
            full = full[~full.index.duplicated(keep='first')].sort_index()
            data[var] = full
    if not data:
        return None
    df = pd.DataFrame(data)
    df.index.name = 'date'
    return df


def _worker_point(args):
    pid, x, y, files_map, var_list, out_dir, overwrite = args
    out_fp = os.path.join(out_dir, f"{pid}.parquet")
    if os.path.exists(out_fp) and not overwrite:
        return out_fp
    df = _extract_point_series(x, y, files_map, var_list)
    if df is None:
        return out_fp
    df.to_parquet(out_fp)
    return out_fp


def extract_gridmet_timeseries_ee(points_shp, base_dir, variables, out_dir, index_col,
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

    print("Indexing gridMET NetCDFs for variables...")
    files_map = {v: _list_gridmet_files(base_dir, v) for v in variables}
    counts = {v: len(files_map.get(v, {})) for v in variables}
    print(f"Found yearly files: " + ", ".join([f"{k}={counts[k]}" for k in variables]))

    tasks = [(pid, x, y, files_map, variables, out_dir, overwrite) for pid, x, y in zip(ids, xs, ys)]

    print(f"Extracting gridMET series to points in {points_shp}")

    if debug:
        for t in tqdm(tasks):
            _worker_point(t)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = {ex.submit(_worker_point, t): t[0] for t in tasks}
            for _ in tqdm(as_completed(futures), total=len(futures)):
                _

    print("Done.")


def get_gridmet_point_timeseries_thredds(lon: float, lat: float, start_date: str, end_date: str,
                                         variables=('pet', 'pr')) -> pd.DataFrame:
    frames = []
    for var in variables:
        g = GridMet(variable=var, start=start_date, end=end_date, lat=lat, lon=lon)
        dfv = g.get_point_timeseries()
        frames.append(dfv)
    df = pd.concat(frames, axis=1)
    df.index.name = 'date'
    return df


if __name__ == '__main__':
    run_mt_mesonet_workflow = True
    run_rosetta_workflow = True
    run_gshp_workflow = True
    run_reesh_workflow = True

    home = os.path.expanduser('~')
    root_ = os.path.join(home, 'data', 'IrrigationGIS')
    out_dir_ = os.path.join(root_, 'soils', 'swapstress', 'vwc', 'gridmet')

    base_dir_ = '/data/ssd2/gridmet'
    gridmet_vars_ = ['pr', 'pet', 'vpd', 'srad', 'tmmx', 'tmmn']

    if run_mt_mesonet_workflow:
        points_shp_ = os.path.join(root_, 'soils', 'soil_potential_obs', 'mt_mesonet', 'station_metadata_mgrs.shp')
        extract_gridmet_timeseries_ee(points_shp=points_shp_,
                                      base_dir=base_dir_,
                                      variables=gridmet_vars_,
                                      out_dir=os.path.join(out_dir_, 'mt_mesonet'),
                                      index_col='station',
                                      num_workers=36,
                                      overwrite=False,
                                      debug=False)

    if run_gshp_workflow:
        points_shp_ = os.path.join(root_, 'soils', 'soil_potential_obs', 'gshp', 'wrc_aggregated_mgrs.shp')
        extract_gridmet_timeseries_ee(points_shp=points_shp_,
                                      base_dir=base_dir_,
                                      variables=gridmet_vars_,
                                      out_dir=os.path.join(out_dir_, 'gshp'),
                                      index_col='profile_id',
                                      num_workers=36,
                                      overwrite=False,
                                      debug=False)

    if run_reesh_workflow:
        points_shp_ = os.path.join(root_, 'soils', 'soil_potential_obs', 'reesh', 'shapefile', 'reesh_sites_mgrs.shp')
        extract_gridmet_timeseries_ee(points_shp=points_shp_,
                                      base_dir=base_dir_,
                                      variables=gridmet_vars_,
                                      out_dir=os.path.join(out_dir_, 'reesh'),
                                      index_col='site_id',
                                      num_workers=36,
                                      overwrite=False,
                                      debug=False)

    if run_rosetta_workflow:
        points_shp_ = os.path.join(root_, 'soils', 'gis', 'pretraining-roi-10000_mgrs.shp')
        extract_gridmet_timeseries_ee(points_shp=points_shp_,
                                      base_dir=base_dir_,
                                      variables=gridmet_vars_,
                                      out_dir=os.path.join(out_dir_, 'rosetta'),
                                      index_col='site_id',
                                      num_workers=36,
                                      overwrite=False,
                                      debug=False)
# ========================= EOF ====================================================================
