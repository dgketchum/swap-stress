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


def build_station_training_table(ee_stations_pqt, results_dir, out_file, include_subdirs=None,
                                 embeddings=None, features_csv=None):
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

    if embeddings:
        rows = {}
        for k, d in embeddings.items():
            emb_files = glob(os.path.join(d, '*.parquet'))
            for fp in emb_files:
                sid = os.path.splitext(os.path.basename(fp))[0].lower().replace('_', '-')
                try:
                    df = pd.read_parquet(fp)
                    if len(df) >= 1:
                        rows[str(sid)] = df.iloc[0]
                except Exception:
                    continue
        if rows:
            emb_df = pd.DataFrame.from_dict(rows, orient='index')
            emb_df['station'] = emb_df.index
            merged = merged.merge(emb_df, on='station', how='left')

    if out_file:
        out_dir = os.path.dirname(out_file)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        merged.to_parquet(out_file)
        print(f"Saving station training table to {out_file} {len(merged)} samples")

    # Optionally write current features (EE features + embeddings if present)
    if features_csv:
        feat_dir = os.path.dirname(features_csv)
        if feat_dir and not os.path.exists(feat_dir):
            os.makedirs(feat_dir)
        ee_feature_cols = [c for c in ee_df.columns if c != 'station']
        # Heuristic: embedding columns start with 'e' followed by 2 digits (e00..e63)
        emb_cols = [c for c in merged.columns if c.startswith('e') and len(c) in (3, 4)]
        features = pd.DataFrame(data=ee_feature_cols + emb_cols, columns=['features'])
        features.to_csv(features_csv, index=False)
        print(f"Saved current features list to {features_csv}")


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
        features_csv_ = os.path.join(root_, 'swapstress', 'training', 'current_features.csv')

        vwc_root_ = '/data/ssd2/swapstress/vwc'
        embeddings_map_ = {
            'reesh': os.path.join(vwc_root_, 'embeddings', 'reesh'),
            'mt_mesonet': os.path.join(vwc_root_, 'embeddings', 'mt_mesonet'),
        }

        build_station_training_table(ee_stations_pqt_, results_dir_, out_file_, include_subdirs=include_,
                                     embeddings=embeddings_map_, features_csv=features_csv_)
# ========================= EOF ====================================================================
