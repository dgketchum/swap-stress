import os
import json
from glob import glob

import numpy as np
import pandas as pd

from retention_curve import ROSETTA_LEVEL_DEPTHS


def _depth_to_rosetta_level(depth_cm):
    try:
        d = float(depth_cm)
    except Exception:
        return None
    for lvl, (lo, hi) in ROSETTA_LEVEL_DEPTHS.items():
        if lo <= d < hi:
            return int(lvl)
    centers = {lvl: (rng[0] + rng[1]) / 2.0 for lvl, rng in ROSETTA_LEVEL_DEPTHS.items()}
    levels = np.array(list(centers.keys()), dtype=int)
    vals = np.array(list(centers.values()), dtype=float)
    idx = int(np.argmin(np.abs(vals - d)))
    return int(levels[idx])


def extract_station_fit_params(results_dir, networks):
    """
    Extracts fitted VG parameters from station JSON summaries into a long table.

    Returns a DataFrame with columns:
      ['station', 'depth', 'rosetta_level', 'theta_r', 'theta_s', 'alpha', 'n']
    """

    files = []
    for sd in networks:
        p = os.path.join(results_dir, sd)
        files.extend(glob(os.path.join(p, '**', '*.json'), recursive=True))

    rows = []

    for fp in files:

        try:
            with open(fp, 'r') as f:
                data = json.load(f)
                meta = data.pop('metadata')
        except Exception:
            continue

        for depth_str, res in data.items():

            try:
                if res.get('status') != 'Success':
                    continue
            except AttributeError:
                continue

            try:
                depth_cm = float(depth_str)
            except Exception:
                continue

            params = res.get('parameters', {})
            depth_meta = meta.get(depth_str, {})
            station = depth_meta['station']
            profile_id = depth_meta['profile_id']

            try:
                tr = float(params['theta_r']['value'])
                ts = float(params['theta_s']['value'])
                a = float(params['alpha']['value'])
                n = float(params['n']['value'])
            except Exception:
                continue

            row = {
                'station': station,
                'profile_id': profile_id,
                'depth': depth_cm,
                'rosetta_level': _depth_to_rosetta_level(depth_cm),
                'theta_r': tr,
                'theta_s': ts,
                'alpha': a,
                'n': n,
            }

            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=['station', 'sample', 'depth', 'rosetta_level',
                                     'theta_r', 'theta_s', 'alpha', 'n'])

    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['theta_r', 'theta_s', 'alpha', 'n', 'depth'])
    return df


def build_station_training_table(ee_stations_pqt, results_dir, out_file, include_subdirs=None):
    params_df = extract_station_fit_params(results_dir, include_subdirs)
    if params_df.empty:
        return params_df

    ee_df = pd.read_parquet(ee_stations_pqt)
    if ee_df.index.name == 'station' and 'station' not in ee_df.columns:
        ee_df = ee_df.reset_index()

    params_df['station'] = params_df['station'].astype(str)
    params_df['profile_id'] = params_df['profile_id'].astype(str)
    params_df['station'] = [s.lower().replace('_', '-') for s in params_df['station']]

    ee_df['station'] = ee_df['station'].astype(str)
    ee_df['station'] = [s.lower().replace('_', '-') for s in ee_df['station']]

    ee_df = ee_df.drop_duplicates(subset='station')

    merged = params_df.merge(
        ee_df,
        on='station',
        how='left',
        validate='m:1',
        suffixes=('', '_ee'))

    missing = merged['station'][merged[ee_df.columns.difference(['station'])].isna().all(axis=1)].unique()
    if len(missing):
        print(f"Stations missing from ee_df: {missing.tolist()}")
    if out_file:
        out_dir = os.path.dirname(out_file)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        merged.to_parquet(out_file)
        print(f"Saving station training table to {out_file} {len(merged)} samples")


if __name__ == '__main__':
    run_stations_workflow = True

    home_ = os.path.expanduser('~')
    root_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils')

    if run_stations_workflow:
        results_dir_ = os.path.join(root_, 'soil_potential_obs', 'curve_fits')
        ee_stations_pqt_ = os.path.join(root_, 'swapstress', 'training', 'stations_ee_data_250m.parquet')
        out_file_ = os.path.join(root_, 'swapstress', 'training', 'stations_training_table_250m.parquet')
        reesh_replicates = os.path.join(root_, 'soil_potential_obs', 'reesh', 'replicate_key.csv')

        include_ = ('mt_mesonet', 'reesh')
        build_station_training_table(ee_stations_pqt_, results_dir_, out_file_, include_subdirs=include_)
# ========================= EOF ====================================================================
