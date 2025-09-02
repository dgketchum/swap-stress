import os
import json
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from retention_curve.swrc import SWRC
from utils.compare_gshp_rosetta_params import find_rosetta_param_columns, choose_rosetta_columns


def _sanitize_uid(val):
    s = str(val) if pd.notnull(val) else ''
    s = re.sub(r"\s+", "", s)
    s = s.replace('.', '_')
    s = re.sub(r"_+", "_", s)
    return s


def vg_theta(psi_cm, theta_r, theta_s, alpha, n):
    if n <= 1:
        return np.full_like(psi_cm, np.nan, dtype=float)
    m = 1.0 - 1.0 / n
    psi_safe = np.maximum(psi_cm, 1e-9)
    term = 1.0 + (alpha * psi_safe) ** n
    return theta_r + (theta_s - theta_r) / (term ** m)


def fit_gshp_curve(csv_path, results_dir, method='slsqp'):
    """
    Prepare GSHP lab data for SWRC fitting by converting/renaming columns:
      - lab_head_m -> suction [cm]
      - lab_wrc   -> theta
      - depth     -> mean(hzn_bottom, hzn_top) [cm]
    Grouping key for fitting is 'layer_id' so each layer is fit independently.
    """
    os.makedirs(results_dir, exist_ok=True)

    try:
        df = pd.read_csv(csv_path, encoding='latin1', low_memory=False)
    except Exception:
        df = pd.read_csv(csv_path)

    # create sanitized uid from layer_id
    if 'layer_id' in df.columns:
        df['uid'] = df['layer_id'].apply(_sanitize_uid)

    df = df.rename(columns={'latitude_decimal_degrees': 'lat', 'longitude_decimal_degrees': 'lon'})
    total_stations = len(df.groupby(['lat', 'lon']))

    df = df[df['data_flag'] == 'good quality estimate']
    df = df[df['SWCC_classes'] == 'YWYD']

    keep = ['layer_id', 'uid', 'lab_head_m', 'lab_wrc', 'hzn_bot', 'hzn_top', 'lat', 'lon']
    present = [c for c in keep if c in df.columns]
    if len(present) < 5:
        raise ValueError

    d = df[present].copy()
    d = d.dropna(subset=['lab_head_m', 'lab_wrc'])
    d['suction'] = (d['lab_head_m'].astype(float) * 100.0).abs()
    d['theta'] = d['lab_wrc'].astype(float)
    if 'hzn_bot' in d.columns and 'hzn_top' in d.columns:
        d['depth'] = (d['hzn_bot'].astype(float) + d['hzn_top'].astype(float)) / 2.0
    else:
        d['depth'] = np.nan

    filtered_stations = len(d.groupby(['depth', 'layer_id']))

    print(f'Analyzing {filtered_stations} of {total_stations} total stations')
    print(f'{len(d)} vwc/psi data points')

    # Instantiate SWRC using DataFrame, grouping by uid to keep per-layer fits
    for i, r in d.groupby(['lat', 'lon']):
        fitter = SWRC(df=r, depth_col='depth')
        fitter.fit(report=False, method=method)
        fitter.save_results(output_dir=results_dir, output_filename=f'{i[0]}.json')


def load_swrc_results(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    rows = []
    raw_points: Dict[str, Dict[str, List[float]]] = {}
    for key, info in data.items():
        # Skip global status keys if present
        if not isinstance(info, dict):
            continue
        status = info.get('status')
        if status is None:
            continue
        if status != 'Success':
            # keep data blob for plotting even if fit failed
            uid_key = info.get('meta', {}).get('uid') or _sanitize_uid(key)
            if 'data' in info:
                raw_points[uid_key] = {
                    'suction': info['data'].get('suction', []),
                    'theta': info['data'].get('theta', []),
                }
            continue
        params = info.get('parameters', {})
        uid_key = info.get('meta', {}).get('uid') or _sanitize_uid(key)
        row = {
            'uid': uid_key,
            'theta_r': params.get('theta_r', {}).get('value'),
            'theta_s': params.get('theta_s', {}).get('value'),
            'alpha': params.get('alpha', {}).get('value'),
            'n': params.get('n', {}).get('value'),
        }
        rows.append(row)
        if 'data' in info:
            raw_points[uid_key] = {
                'suction': info['data'].get('suction', []),
                'theta': info['data'].get('theta', []),
            }
    return pd.DataFrame(rows), raw_points


def load_rosetta_params(parquet_path, prefix_regex=None):
    r = pd.read_parquet(parquet_path)
    if 'uid' not in r.columns and 'layer_id' in r.columns:
        r['uid'] = r['layer_id'].apply(_sanitize_uid)
    r = r.groupby('uid').first().copy()
    r['uid'] = r.index
    cand_map = find_rosetta_param_columns(r)
    chosen = choose_rosetta_columns(cand_map, prefix_regex)
    keep: List[str] = []
    for p, col in chosen.items():
        if col is not None and col in r.columns:
            keep.append(col)
    keep.append('uid')
    r = r[keep].copy()
    r.columns = [c if c == 'uid' else c for c in r.columns]
    return r, chosen


def plot_curves(gshp_json, rosetta_parquet, out_dir, rosetta_prefix_regex=None,
                sample_uids: Optional[List[str]] = None):
    os.makedirs(out_dir, exist_ok=True)

    swrc_df, raw_points = load_swrc_results(gshp_json)
    ros_df, chosen = load_rosetta_params(rosetta_parquet, rosetta_prefix_regex)

    merged = swrc_df.merge(ros_df, on='uid', how='inner')
    if merged.empty:
        return

    # Resolve Rosetta param columns
    ros_cols: Dict[str, Optional[str]] = chosen

    # Build psi grid (cm)
    psi = np.logspace(-2, 6, 400)

    # Subset to requested UIDs if provided
    if sample_uids:
        merged = merged[merged['uid'].isin(sample_uids)]

    for _, row in merged.iterrows():
        uid = row['uid']
        tr = float(row['theta_r']) if pd.notnull(row['theta_r']) else np.nan
        ts = float(row['theta_s']) if pd.notnull(row['theta_s']) else np.nan
        a_fit = float(row['alpha']) if pd.notnull(row['alpha']) else np.nan
        n_fit = float(row['n']) if pd.notnull(row['n']) else np.nan

        # Rosetta params may be in log10; detect by column name
        def get_ros(p):
            col = ros_cols.get(p)
            if col is None or col not in row.index:
                return np.nan
            val = row[col]
            if isinstance(col, str) and ('log10_' in col or col.endswith('_log10_alpha') or col.endswith('_log10_n')):
                return float(10 ** val)
            if p == 'alpha' and 'log10' in col:
                return float(10 ** val)
            if p == 'n' and 'log10' in col:
                return float(10 ** val)
            return float(val)

        a_ros = get_ros('alpha')
        n_ros = get_ros('n')
        tr_ros = get_ros('theta_r')
        ts_ros = get_ros('theta_s')

        # Plot
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(7, 6))

        theta_fit = vg_theta(psi, tr, ts, a_fit, n_fit)
        theta_ros = vg_theta(psi, tr_ros, ts_ros, a_ros, n_ros)

        # plot raw GSHP points if available
        raw = raw_points.get(uid)
        if raw is not None and raw.get('suction') and raw.get('theta'):
            ax.scatter(raw['theta'], raw['suction'], s=12, alpha=0.5, label='GSHP points')

        ax.plot(theta_fit, psi, label='GSHP fit', lw=2)
        ax.plot(theta_ros, psi, label='Rosetta', lw=2)

        ax.set_yscale('log')
        ax.set_ylim(1e-2, 1e6)
        ax.set_xlim(0, 0.7)
        ax.set_xlabel('Volumetric Water Content (cm3/cm3)')
        ax.set_ylabel('Soil Water Potential (cm)')
        ax.set_title(f' SWRC: {uid}')
        ax.legend()
        plt.tight_layout()

        out_fp = os.path.join(out_dir, f'swrc_compare_{uid}.png')
        plt.savefig(out_fp, dpi=300)
        plt.close(fig)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    root_ = os.path.join(home, 'data', 'IrrigationGIS')
    gshp_dir_ = os.path.join(root_, 'soils', 'vg_paramaer_databases', 'wrc')

    gshp_csv_ = os.path.join(gshp_dir_, 'WRC_dataset_surya_et_al_2021_final.csv')
    rosetta_parquet_ = os.path.join(gshp_dir_, 'extracted_rosetta_points.parquet')
    fits_json_ = os.path.join(gshp_dir_, 'local_fits')

    fit_gshp_curve(gshp_csv_, fits_json_)

    plots_dir_ = os.path.join(gshp_dir_, 'swrc_curve_plots')
    os.makedirs(plots_dir_, exist_ok=True)

    # plot_curves(fits_json_, rosetta_parquet_, plots_dir_, rosetta_prefix_regex=None, sample_uids=None)
# ========================= EOF ====================================================================
