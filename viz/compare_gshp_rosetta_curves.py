import os
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from retention_curve import ROSETTA_LEVEL_DEPTHS
from viz.compare_gshp_rosetta_params import find_rosetta_param_columns

"""Compare GSHP SWRC fits to Rosetta curves.

Reads GSHP lab data, plots published curves alongside Rosetta level curves
for the same profile. Saves per-profile comparison figures.
"""


def vg_theta(psi_cm, theta_r, theta_s, alpha, n):
    if n <= 1:
        return np.full_like(psi_cm, np.nan, dtype=float)
    m = 1.0 - 1.0 / n
    psi_safe = np.maximum(psi_cm, 1e-9)
    term = 1.0 + (alpha * psi_safe) ** n
    return theta_r + (theta_s - theta_r) / (term ** m)


def plot_curves(gshp_csv, rosetta_parquet, out_dir,
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

    df = pd.read_csv(gshp_csv, encoding='latin1', low_memory=False)
    df = df[df['data_flag'] == 'good quality estimate']
    df = df.rename(columns={'latitude_decimal_degrees': 'lat', 'longitude_decimal_degrees': 'lon',
                            'alpha': 'alpha_pub', 'n': 'n_pub', 'thetar': 'theta_r_pub',
                            'thetas': 'theta_s_pub'})
    df['alpha_pub'] /= 100
    df['alpha_pub'][df['alpha_pub'] < -5] = np.nan

    keep = [
        'profile_id', 'layer_id', 'lab_head_m', 'lab_wrc', 'hzn_bot', 'hzn_top',
        'lat', 'lon', 'alpha_pub', 'n_pub', 'theta_r_pub', 'theta_s_pub'
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

    for idx, r in d.groupby(['profile_id']):
        pid = idx[0] if isinstance(idx, tuple) else idx
        if np.all(np.isnan(r['depth'])):
            continue
        if np.any(np.isnan(r['depth'])):
            r = r[~pd.isna(r['depth'])]

        raw_points_map: Dict[str, Dict[str, List[float]]] = {}
        for depth_val, rr in r.groupby('depth'):
            raw_points_map[str(depth_val)] = {
                'suction': rr['suction'].astype(float).tolist(),
                'theta': rr['theta'].astype(float).tolist(),
            }

        pub_params = r.groupby('depth').first()[['alpha_pub', 'n_pub', 'theta_r_pub', 'theta_s_pub']]
        pub_by_depth: Dict[str, Dict[str, float]] = {}
        for depth_val, row in pub_params.iterrows():
            pub_by_depth[str(depth_val)] = {
                'theta_r': float(row.get('theta_r_pub')) if row.get('theta_r_pub') is not None else np.nan,
                'theta_s': float(row.get('theta_s_pub')) if row.get('theta_s_pub') is not None else np.nan,
                'alpha': float(row.get('alpha_pub')) if row.get('alpha_pub') is not None else np.nan,
                'n': float(row.get('n_pub')) if row.get('n_pub') is not None else np.nan,
            }

        depths = sorted(r['depth'].unique())
        if sample_uids and pid not in set(sample_uids):
            continue

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(8, 7))
        cmap = plt.get_cmap('plasma')
        colors = cmap(np.linspace(0, 0.85, len(depths)))

        scatters = 0
        for color, depth_cm in zip(colors, depths):
            label_prefix = f"Depth {depth_cm} cm"
            rp = raw_points_map.get(str(depth_cm))
            if rp is not None and rp.get('suction') and rp.get('theta'):
                if len(np.unique(rp['suction'])) < 4:
                    continue
                ax.scatter(rp['theta'], rp['suction'], s=12, alpha=0.5, color=color, label=f"{label_prefix} points")
                scatters += 1

            pub = pub_by_depth.get(str(depth_cm))
            if pub:
                tr = float(pub.get('theta_r')) if pub.get('theta_r') is not None else np.nan
                ts = float(pub.get('theta_s')) if pub.get('theta_s') is not None else np.nan
                a_pub = float(pub.get('alpha')) if pub.get('alpha') is not None else np.nan
                n_pub = float(pub.get('n')) if pub.get('n') is not None else np.nan
                if not (np.isnan(tr) or np.isnan(ts) or np.isnan(a_pub) or np.isnan(n_pub)):
                    theta_pub = vg_theta(psi, tr, ts, a_pub, n_pub)
                    ax.plot(theta_pub, psi, '-', color=color, lw=2, label=f"{label_prefix} published")

        if scatters == 0:
            plt.close()
            continue

        # Rosetta curves by nearest level for this profile
        rosetta_found = False
        rosetta_by_depth: Dict[float, Dict[str, float]] = {}
        row = ros_df[ros_df['profile_id'] == pid]
        if not row.empty:
            row = row.iloc[0]
            for color, depth_cm in zip(colors, depths):
                level = nearest_level(depth_cm)

                def get_ros_at_level(p: str):
                    cands = ros_cand_map.get(p, [])
                    token = f"L{level}"
                    for col in cands:
                        if token in col and col in row.index:
                            val = row[col]
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
                    rosetta_found = True
                    theta_ros = vg_theta(psi, tr_ros, ts_ros, a_ros, n_ros)
                    rng = ROSETTA_LEVEL_DEPTHS.get(level)
                    if rng:
                        lbl = f"Rosetta L{level} ({rng[0]:.0f}-{rng[1]:.0f} cm)"
                    else:
                        lbl = f"Rosetta L{level}"
                    ax.plot(theta_ros, psi, '--', color=color, lw=1.6, label=lbl)
                    try:
                        rosetta_by_depth[float(depth_cm)] = {
                            'theta_r': float(tr_ros), 'theta_s': float(ts_ros), 'alpha': float(a_ros),
                            'n': float(n_ros),
                            'level': level,
                        }
                    except Exception:
                        pass

        if not rosetta_found:
            plt.close()
            continue

        ax.set_yscale('log')
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
        ax.set_title(f'SWRC: {pid}')

        shallow_vals = [v for v in depths if isinstance(v, (int, float, np.floating))]
        if shallow_vals:
            sh_depth = float(shallow_vals[0])
            sh_color = colors[0]

            rows = []
            row_labels = []

            pub = pub_by_depth.get(str(sh_depth))
            if pub and not all(np.isnan(list(pub.values()))):
                rows.append(
                    [f"{pub['theta_r']:.3f}", f"{pub['theta_s']:.3f}", f"{pub['alpha']:.3f}", f"{pub['n']:.3f}"])
                row_labels.append('Published')

            ros = rosetta_by_depth.get(sh_depth)
            if ros:
                rows.append(
                    [f"{ros['theta_r']:.3f}", f"{ros['theta_s']:.3f}", f"{ros['alpha']:.3f}", f"{ros['n']:.3f}"])
                lvl = ros.get('level')
                row_labels.append(f"Rosetta L{lvl}")

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

        out_fp = os.path.join(out_dir, f"{pid}_compare.png")
        plt.savefig(out_fp, dpi=300)
        plt.close(fig)
        print(out_fp)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    root_ = os.path.join(home, 'data', 'IrrigationGIS')
    gshp_dir_ = os.path.join(root_, 'soils', 'soil_potential_obs', 'gshp')

    gshp_csv_ = os.path.join(gshp_dir_, 'WRC_dataset_surya_et_al_2021_final.csv')
    rosetta_parquet_ = os.path.join(gshp_dir_, 'extracted_rosetta_points.parquet')
    plots_dir_ = os.path.join(gshp_dir_, 'swrc_curve_plots')

    plot_curves(gshp_csv_, rosetta_parquet_, plots_dir_, sample_uids=None)
# ========================= EOF ====================================================================
