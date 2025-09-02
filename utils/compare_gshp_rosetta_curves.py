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
    s = s.strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace('/', '_').replace('\\', '_')
    s = s.replace('.', '_')
    s = re.sub(r"[^A-Za-z0-9_-]", "_", s)
    s = re.sub(r"_+", "_", s).strip('_')
    return s[:80]


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

    # Instantiate SWRC using DataFrame, grouping by site (lat, lon)
    for (lat, lon), r in d.groupby(['lat', 'lon']):
        fitter = SWRC(df=r, depth_col='depth')
        fitter.fit(report=False, method=method)
        try:
            fitter.save_results(output_dir=results_dir, output_filename=f'{r.iloc[0]['uid']}.json')
        except FileNotFoundError:
            continue


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


def plot_curves(gshp_results, rosetta_parquet, out_dir, rosetta_prefix_regex=None,
                sample_uids: Optional[List[str]] = None):
    os.makedirs(out_dir, exist_ok=True)

    # Preload Rosetta params once
    ros_df, chosen = load_rosetta_params(rosetta_parquet, rosetta_prefix_regex)
    ros_cols: Dict[str, Optional[str]] = chosen

    psi = np.logspace(-2, 6, 400)

    # Determine JSON files to plot
    if os.path.isdir(gshp_results):
        json_files = [os.path.join(gshp_results, f) for f in os.listdir(gshp_results) if f.endswith('.json')]
    else:
        json_files = [gshp_results]

    for jfp in json_files:
        try:
            with open(jfp, 'r') as f:
                data = json.load(f)
        except Exception:
            continue

        depth_items = []
        raw_points_map: Dict[str, Dict[str, List[float]]] = {}
        uids_for_site: List[str] = []
        for key, info in data.items():
            if not isinstance(info, dict) or 'status' not in info:
                continue
            uid_key = info.get('meta', {}).get('uid')
            if uid_key:
                uids_for_site.append(uid_key)
            if 'data' in info and uid_key:
                raw_points_map[uid_key] = {
                    'suction': info['data'].get('suction', []),
                    'theta': info['data'].get('theta', []),
                }
            if info.get('status') != 'Success':
                continue
            params = info.get('parameters', {})
            try:
                depth_val = float(key)
            except Exception:
                depth_val = key
            depth_items.append({
                'depth': depth_val,
                'theta_r': params.get('theta_r', {}).get('value'),
                'theta_s': params.get('theta_s', {}).get('value'),
                'alpha': params.get('alpha', {}).get('value'),
                'n': params.get('n', {}).get('value'),
                'uid': uid_key,
            })

        if not depth_items:
            continue

        if sample_uids:
            depth_items = [d for d in depth_items if d['uid'] in sample_uids]
            if not depth_items:
                continue

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(8, 7))
        depths_sorted = sorted(depth_items, key=lambda x: (np.nan if x['depth'] is None else x['depth']))
        colors = plt.cm.get_cmap('plasma')(np.linspace(0, 0.85, len(depths_sorted)))

        for color, item in zip(colors, depths_sorted):
            tr = float(item['theta_r']) if pd.notnull(item['theta_r']) else np.nan
            ts = float(item['theta_s']) if pd.notnull(item['theta_s']) else np.nan
            a_fit = float(item['alpha']) if pd.notnull(item['alpha']) else np.nan
            n_fit = float(item['n']) if pd.notnull(item['n']) else np.nan
            theta_fit = vg_theta(psi, tr, ts, a_fit, n_fit)
            label_prefix = f"Depth {item['depth']} cm" if isinstance(item['depth'], (int, float)) else str(item['depth'])
            rp = raw_points_map.get(item['uid'])
            if rp is not None and rp.get('suction') and rp.get('theta'):
                ax.scatter(rp['theta'], rp['suction'], s=12, alpha=0.5, color=color, label=f"{label_prefix} points")
            ax.plot(theta_fit, psi, '-', color=color, lw=2, label=f"{label_prefix} fit")

        # Rosetta overlay using first matching uid
        a_ros = n_ros = tr_ros = ts_ros = np.nan
        for uid in uids_for_site:
            row = ros_df[ros_df['uid'] == uid]
            if not row.empty:
                row = row.iloc[0]
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
                break
        if not np.isnan(a_ros) and not np.isnan(n_ros) and not np.isnan(tr_ros) and not np.isnan(ts_ros):
            theta_ros = vg_theta(psi, tr_ros, ts_ros, a_ros, n_ros)
            ax.plot(theta_ros, psi, 'k--', lw=2, label='Rosetta')

        ax.set_yscale('log')
        ax.set_ylim(1e-2, 1e6)
        ax.set_xlim(0, 0.7)
        ax.set_xlabel('Volumetric Water Content (cm3/cm3)')
        ax.set_ylabel('Soil Water Potential (cm)')
        ax.set_title(f'SWRC: {os.path.basename(jfp).replace(".json", "")}')
        ax.legend(fontsize=9)
        plt.tight_layout()

        out_fp = os.path.join(out_dir, os.path.basename(jfp).replace('.json', '_compare.png'))
        plt.savefig(out_fp, dpi=300)
        plt.close(fig)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    root_ = os.path.join(home, 'data', 'IrrigationGIS')
    gshp_dir_ = os.path.join(root_, 'soils', 'vg_paramaer_databases', 'wrc')

    gshp_csv_ = os.path.join(gshp_dir_, 'WRC_dataset_surya_et_al_2021_final.csv')
    rosetta_parquet_ = os.path.join(gshp_dir_, 'extracted_rosetta_points.parquet')
    fits_dir_ = os.path.join(gshp_dir_, 'local_fits')

    fit_gshp_curve(gshp_csv_, fits_dir_)

    plots_dir_ = os.path.join(gshp_dir_, 'swrc_curve_plots')
    os.makedirs(plots_dir_, exist_ok=True)

    plot_curves(fits_dir_, rosetta_parquet_, plots_dir_, rosetta_prefix_regex=None, sample_uids=None)
# ========================= EOF ====================================================================
