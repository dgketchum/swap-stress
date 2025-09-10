import os
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from retention_curve.swrc import SWRC
from retention_curve.gshp_swrc import GshpSWRC
from retention_curve import ROSETTA_LEVEL_DEPTHS
from viz.compare_gshp_rosetta_params import find_rosetta_param_columns


"""Compare GSHP SWRC fits to Rosetta curves.

Reads GSHP lab data, fits van Genuchten curves per profile/layer, and
plots fitted curves alongside Rosetta level curves for the same profile.
Saves per-profile comparison figures; optional training PTF for theta_s.
"""

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

    total_stations = len(df.groupby(['latitude_decimal_degrees', 'longitude_decimal_degrees']))

    df = df[df['data_flag'] == 'good quality estimate']

    missing_wet_ptf = GshpSWRC.estimate_theta_s_ptf(df)

    df = df.rename(columns={'latitude_decimal_degrees': 'lat', 'longitude_decimal_degrees': 'lon',
                            'alpha': 'alpha_pub', 'n': 'n_pub', 'thetar': 'theta_r_pub',
                            'thetas': 'theta_s_pub'})

    df['alpha_pub'] /= 100
    df['alpha_pub'][df['alpha_pub'] < -5] = np.nan

    keep = [
        'profile_id',
        'layer_id',
        'lab_head_m',
        'lab_wrc',
        'hzn_bot',
        'hzn_top',
        'lat',
        'lon',
        'alpha_pub',
        'n_pub',
        'theta_r_pub',
        'theta_s_pub',
        'SWCC_classes',
    ]

    present = [c for c in keep if c in df.columns]

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
    for idx, r in d.groupby(['profile_id']):

        pid = idx[0]
        if np.all(np.isnan(r['depth'])):
            print(f'invalid depth for {pid}')
            continue

        if np.any(np.isnan(r['depth'])):
            print(f'some invalid depth for {pid}')
            r = r[~pd.isna(r['depth'])]

        filename = f'{pid}.json'
        fitter = GshpSWRC(df=r, depth_col='depth')

        if r.iloc[0]['SWCC_classes'] == 'NWYD':
            fitter.set_theta_s_ptf(missing_wet_ptf)

        fitter.fit(report=False, method=method)

        additional_data = r.groupby('depth').first()[['alpha_pub',
                                                      'n_pub',
                                                      'theta_r_pub',
                                                      'theta_s_pub',
                                                      'SWCC_classes']]

        additional_data = additional_data.to_dict(orient='index')
        try:
            fitter.save_results(output_dir=results_dir, output_filename=filename, add_data=additional_data)
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
            pid = info.get('meta', {}).get('profile_id') or str(key)
            if 'data' in info:
                raw_points[pid] = {
                    'suction': info['data'].get('suction', []),
                    'theta': info['data'].get('theta', []),
                }
            continue
        params = info.get('parameters', {})
        pid = info.get('meta', {}).get('profile_id') or str(key)
        row = {
            'profile_id': pid,
            'theta_r': params.get('theta_r', {}).get('value'),
            'theta_s': params.get('theta_s', {}).get('value'),
            'alpha': params.get('alpha', {}).get('value'),
            'n': params.get('n', {}).get('value'),
        }
        rows.append(row)
        if 'data' in info:
            raw_points[pid] = {
                'suction': info['data'].get('suction', []),
                'theta': info['data'].get('theta', []),
            }
    return pd.DataFrame(rows), raw_points


def plot_curves(gshp_results, rosetta_parquet, out_dir,
                sample_uids: Optional[List[str]] = None):
    os.makedirs(out_dir, exist_ok=True)

    # Preload Rosetta params once (keep all columns)
    ros_df = pd.read_parquet(rosetta_parquet).groupby('profile_id').first().reset_index()
    ros_df = ros_df.dropna()
    ros_cand_map = find_rosetta_param_columns(ros_df)

    # Define Rosetta level centers (cm) for nearest-depth matching using ROSETTA_LEVEL_DEPTHS
    level_centers_cm = {lvl: (rng[0] + rng[1]) / 2.0 for lvl, rng in ROSETTA_LEVEL_DEPTHS.items()}

    def nearest_level(depth_cm: float) -> int:
        vals = np.array(list(level_centers_cm.values()), dtype=float)
        keys = np.array(list(level_centers_cm.keys()), dtype=int)
        idx = int(np.argmin(np.abs(vals - float(depth_cm))))
        return int(keys[idx])

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
        pub_by_depth: Dict[float, Dict[str, float]] = {}
        pid = None

        for key, info in data.items():
            if not isinstance(info, dict) or 'status' not in info:
                continue

            if pid is None:
                pid = info.get('meta', {}).get('profile_id')

            if 'data' in info and pid:
                raw_points_map[key] = {
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

            try:
                dv = str(depth_val)
                pub_by_depth[dv] = {
                    'theta_r': float(data[dv].get('theta_r_pub')) if data[dv].get(
                        'theta_r_pub') is not None else np.nan,
                    'theta_s': float(data[dv].get('theta_s_pub')) if data[dv].get(
                        'theta_s_pub') is not None else np.nan,
                    'alpha': float(data[dv].get('alpha_pub')) if data[dv].get('alpha_pub') is not None else np.nan,
                    'n': float(data[dv].get('n_pub')) if data[dv].get('n_pub') is not None else np.nan,
                }
            except Exception:
                pass

            depth_items.append({
                'depth': depth_val,
                'theta_r': params.get('theta_r', {}).get('value'),
                'theta_s': params.get('theta_s', {}).get('value'),
                'alpha': params.get('alpha', {}).get('value'),
                'n': params.get('n', {}).get('value'),
                'profile_id': pid,
            })

        if not depth_items:
            continue

        if sample_uids:
            depth_items = [d for d in depth_items if d['profile_id'] in sample_uids]
            if not depth_items:
                continue

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(8, 7))
        depths_sorted = sorted(depth_items, key=lambda x: (np.nan if x['depth'] is None else x['depth']))
        cmap = plt.get_cmap('plasma')
        colors = cmap(np.linspace(0, 0.85, len(depths_sorted)))

        scatters, gshp_params = 0, None
        for color, item in zip(colors, depths_sorted):
            tr = float(item['theta_r']) if pd.notnull(item['theta_r']) else np.nan
            ts = float(item['theta_s']) if pd.notnull(item['theta_s']) else np.nan
            a_fit = float(item['alpha']) if pd.notnull(item['alpha']) else np.nan
            n_fit = float(item['n']) if pd.notnull(item['n']) else np.nan
            theta_fit = vg_theta(psi, tr, ts, a_fit, n_fit)
            label_prefix = f"Depth {item['depth']} cm" if isinstance(item['depth'], (int, float)) else str(
                item['depth'])
            rp = raw_points_map.get(str(item['depth']))
            if rp is not None and rp.get('suction') and rp.get('theta'):
                if len(np.unique(rp['suction'])) < 4:
                    continue
                ax.scatter(rp['theta'], rp['suction'], s=12, alpha=0.5, color=color, label=f"{label_prefix} points")
                scatters += 1

            ax.plot(theta_fit, psi, '-', color=color, lw=2, label=f"{label_prefix} fit")

        if scatters == 0:
            plt.close()
            continue

        # For each depth, add a Rosetta curve using nearest level for that profile_id
        rosetta_found = False
        rosetta_by_depth: Dict[float, Dict[str, float]] = {}
        for color, item in zip(colors, depths_sorted):
            depth_cm = item['depth']
            if depth_cm is None:
                continue
            row = ros_df[ros_df['profile_id'] == pid]
            if row.empty:
                continue

            rosetta_found = True
            row = row.iloc[0]
            level = nearest_level(depth_cm)

            def get_ros_at_level(p: str):
                cands = ros_cand_map.get(p, [])
                # find a candidate containing the level token
                token = f"L{level}"
                for col in cands:
                    if token in col and col in row.index:
                        val = row[col]
                        # handle log10 columns
                        if isinstance(col, str) and (
                            'log10_' in col or col.endswith('_log10_alpha') or col.endswith('_log10_n')):
                            return float(10 ** val)
                        if p in ('alpha', 'n') and 'log10' in col:
                            return float(10 ** val)
                        return float(val)
                return np.nan

            a_ros = get_ros_at_level('alpha')
            n_ros = get_ros_at_level('n')
            tr_ros = get_ros_at_level('theta_r')
            ts_ros = get_ros_at_level('theta_s')
            if not (np.isnan(a_ros) or np.isnan(n_ros) or np.isnan(tr_ros) or np.isnan(ts_ros)):
                theta_ros = vg_theta(psi, tr_ros, ts_ros, a_ros, n_ros)
                rng = ROSETTA_LEVEL_DEPTHS.get(level)
                if rng:
                    lbl = f"Rosetta L{level} ({rng[0]:.0f}-{rng[1]:.0f} cm)"
                else:
                    lbl = f"Rosetta L{level}"
                ax.plot(theta_ros, psi, '--', color=color, lw=1.6, label=lbl)
                try:
                    rosetta_by_depth[float(depth_cm)] = {
                        'theta_r': float(tr_ros), 'theta_s': float(ts_ros), 'alpha': float(a_ros), 'n': float(n_ros),
                        'level': level,
                    }
                except Exception:
                    pass

        if not rosetta_found:
            plt.close()
            continue

        ax.set_yscale('log')
        # Set y-limits to bracket measured suction by ±0.5 decades
        suctions = []
        for _pid, rp in raw_points_map.items():
            if rp is None:
                continue
            s = rp.get('suction')
            if s is None:
                continue
            suctions.extend([float(v) for v in s if v is not None and np.isfinite(v) and v > 0])
        if suctions:
            sf = np.sqrt(10.0)
            y_min = max(min(suctions) / sf, 1e-6)
            y_max = max(suctions) * sf
            if y_max > y_min:
                ax.set_ylim(y_min, y_max)

        ax.set_xlim(0, 0.7)
        ax.set_xlabel('Volumetric Water Content (cm3/cm3)')
        ax.set_ylabel('Soil Water Potential (cm)')
        ax.set_title(f'SWRC: {os.path.basename(jfp).replace(".json", "")}')

        shallow_items = [d for d in depths_sorted if isinstance(d['depth'], (int, float, np.floating))]
        if shallow_items:
            shallow = shallow_items[0]
            sh_depth = float(shallow['depth'])
            sh_color = colors[0]

            rows = []
            row_labels = []

            pub = pub_by_depth.get(sh_depth) or pub_by_depth.get(str(sh_depth))
            if pub and not all(np.isnan(list(pub.values()))):
                rows.append(
                    [f"{pub['theta_r']:.3f}", f"{pub['theta_s']:.3f}", f"{pub['alpha']:.3f}", f"{pub['n']:.3f}"])
                row_labels.append('Published')

            rows.append([f"{float(shallow['theta_r']):.3f}", f"{float(shallow['theta_s']):.3f}",
                         f"{float(shallow['alpha']):.3f}", f"{float(shallow['n']):.3f}"])
            row_labels.append('Fit')
            # Rosetta
            ros = rosetta_by_depth.get(sh_depth)
            if ros:
                rows.append(
                    [f"{ros['theta_r']:.3f}", f"{ros['theta_s']:.3f}", f"{ros['alpha']:.3f}", f"{ros['n']:.3f}"])
                lvl = ros.get('level')
                lbl = f"Rosetta L{lvl}"
                row_labels.append(lbl)
            if rows:
                tbl = ax.table(cellText=rows,
                               rowLabels=row_labels,
                               colLabels=['θr', 'θs', 'α', 'n'],
                               cellLoc='center', colLoc='center',
                               bbox=[0.10, 0.05, 0.30, 0.17])
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(9)
                tbl.scale(0.3, 0.6)
                try:
                    tbl.set_zorder(10)
                except Exception:
                    pass
                for key_cell, cell in tbl.get_celld().items():
                    cell.set_edgecolor('none')
                    cell.set_facecolor((1, 1, 1, 0.0))
                    try:
                        cell.set_zorder(11)
                    except Exception:
                        pass
                    if hasattr(cell, 'get_text'):
                        try:
                            cell.get_text().set_color(sh_color)
                            cell.get_text().set_zorder(12)
                        except Exception:
                            pass
        ax.legend(fontsize=9)
        plt.tight_layout()

        out_fp = os.path.join(out_dir, os.path.basename(jfp).replace('.json', '_compare.png'))
        plt.savefig(out_fp, dpi=300)
        plt.close(fig)
        print(out_fp)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    root_ = os.path.join(home, 'data', 'IrrigationGIS')
    gshp_dir_ = os.path.join(root_, 'soils', 'soil_potential_obs', 'gshp')

    gshp_csv_ = os.path.join(gshp_dir_, 'WRC_dataset_surya_et_al_2021_final.csv')
    rosetta_parquet_ = os.path.join(gshp_dir_, 'extracted_rosetta_points.parquet')
    fits_dir_ = os.path.join(gshp_dir_, 'local_fits')

    fit_gshp_curve(gshp_csv_, fits_dir_)

    plots_dir_ = os.path.join(gshp_dir_, 'swrc_curve_plots')

    plot_curves(fits_dir_, rosetta_parquet_, plots_dir_, sample_uids=None)
# ========================= EOF ====================================================================

