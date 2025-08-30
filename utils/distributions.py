import os
import re
from glob import glob

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from retention_curve import PARAM_SYMBOLS, EMPIRICAL_TO_ROSETTA_LEVEL_MAP


def _load_empirical_results(results_dir):
    """Loads successful empirical fit results and pivots to station_L{level}_{param} columns."""
    json_files = glob(os.path.join(results_dir, '**', '*_fit_results.json'), recursive=True)
    if not json_files:
        raise FileNotFoundError(f"No empirical result files found in {results_dir}")

    rows = []
    for f in json_files:
        station_name = os.path.basename(f).replace('_fit_results.json', '')
        try:
            data = pd.read_json(f, typ='series').to_dict()
        except ValueError:
            # Fallback: read via std json
            import json as _json
            with open(f, 'r') as jf:
                data = _json.load(jf)
        for depth, res in data.items():
            try:
                if res.get('status') != 'Success':
                    continue
            except AttributeError:
                continue
            try:
                d = int(depth)
            except Exception:
                continue
            row = {'station': station_name, 'depth': d}
            params = res.get('parameters', {})
            for p, v in params.items():
                row[f'station_{p}'] = v.get('value')
            rows.append(row)

    if not rows:
        raise ValueError('No successful empirical fit results found')

    df = pd.DataFrame(rows)
    df['rosetta_level'] = df['depth'].map(EMPIRICAL_TO_ROSETTA_LEVEL_MAP)
    params = ['theta_r', 'theta_s', 'alpha', 'n']
    df_m = df.melt(id_vars=['station', 'rosetta_level'], value_vars=[f'station_{p}' for p in params],
                   var_name='param', value_name='value')
    df_m['level_param'] = df_m.apply(lambda r: f"L{int(r['rosetta_level'])}_{r['param']}", axis=1)
    wide = df_m.pivot_table(index='station', columns='level_param', values='value', aggfunc='mean').reset_index()
    # Rename "Lx_station_param" to "station_Lx_param"
    wide.columns = [re.sub(r'^L([1-7])_station_', r'station_L\1_', c) for c in wide.columns]
    return wide


def _load_rosetta_params(rosetta_parquet):
    df = pd.read_parquet(rosetta_parquet)
    if 'station' in df.columns:
        df = df.groupby('station').first().reset_index()
    else:
        # Assume index is station id
        df = df.copy()
        df['station'] = df.index.astype(str)
        df.reset_index(drop=True, inplace=True)

    # Keep only US_R3H3* columns and station
    keep_cols = ['station'] + [c for c in df.columns if c.startswith('US_R3H3_')]
    df = df[keep_cols]
    # Rename to rosetta_Lx_* pattern
    df.columns = [re.sub(r'US_R3H3_L([1-7])_VG_', r'rosetta_L\1_', c) for c in df.columns]
    return df


def _load_polaris_params(training_parquet):
    """Loads POLARIS VG parameter estimates from training data if present."""
    df = pd.read_parquet(training_parquet)
    polaris_map = {
        'theta_r': 'theta_r_mean',
        'theta_s': 'theta_s_mean',
        'alpha': 'alpha_mean',
        'n': 'n_mean',
    }
    df.loc[df['alpha_mean'] > 0.2, 'alpha_mean'] = np.nan
    available = {p: c for p, c in polaris_map.items() if c in df.columns}
    if not available:
        # Try alternate names
        alt_map = {p: p for p in ['theta_r', 'theta_s', 'alpha', 'n']}
        available = {p: c for p, c in alt_map.items() if c in df.columns}
    polaris = {p: df[c].dropna().values for p, c in available.items()}
    return polaris


def _load_gshp_params(csv_path):
    """
    Load GSHP (Surya et al., 2021) parameters grouped by 'layer_id' taking the first
    value per group to avoid duplicates. Maps columns ['alpha','thetar','thetas','n']
    to keys ['alpha','theta_r','theta_s','n'].
    """
    if not csv_path or not os.path.exists(csv_path):
        return {}
    try:
        df = pd.read_csv(csv_path, encoding='latin1')
    except Exception:
        df = pd.read_csv(csv_path)
    needed = ['layer_id', 'alpha', 'thetar', 'thetas', 'n']
    present = [c for c in needed if c in df.columns]
    if 'layer_id' not in present:
        return {}
    df = df[df['data_flag'] == 'good quality estimate']
    g = df[present].groupby('layer_id').agg({c: 'first' for c in present if c != 'layer_id'}).reset_index()
    return {
        'theta_r': g['thetar'].dropna().values if 'thetar' in g.columns else np.array([]),
        'theta_s': g['thetas'].dropna().values if 'thetas' in g.columns else np.array([]),
        'alpha': g['alpha'].dropna().values if 'alpha' in g.columns else np.array([]),
        'n': g['n'].dropna().values if 'n' in g.columns else np.array([]),
    }

def compare_parameter_distributions_combined(empirical_results_dir, rosetta_parquet, training_parquet,
                                             output_dir, bins=30, gshp_csv=None):
    """
    Combines all depths/levels into a single population per parameter and
    produces a single 2x2 figure with panels for [theta_r, theta_s, alpha, n].

    - For alpha and n, compares distributions in log10 space across sources.
    - Sources: Station empirical fits, Rosetta extracts, POLARIS (overall).
    """
    os.makedirs(output_dir, exist_ok=True)

    station_df = _load_empirical_results(empirical_results_dir)
    ros_df = _load_rosetta_params(rosetta_parquet)
    polaris_vals = _load_polaris_params(training_parquet)
    gshp_vals = _load_gshp_params(gshp_csv) if gshp_csv else {}

    params = ['theta_r', 'theta_s', 'alpha', 'n']
    need_log = {'alpha', 'n'}
    palette = {'Station': '#1f77b4', 'Rosetta': '#ff7f0e', 'POLARIS': '#2ca02c', 'GSHP': '#9467bd'}

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for ax, param in zip(axes, params):
        # Station: gather all station_Lx_param columns
        st_cols = [c for c in station_df.columns if re.match(fr'^station_L\d+_{param}$', c)]
        st_series = pd.concat([station_df[c] for c in st_cols], ignore_index=True) if st_cols else pd.Series(dtype=float)

        # Rosetta: gather linear and log10 columns
        ros_lin_cols = [c for c in ros_df.columns if re.match(fr'^rosetta_L2_{param}$', c)]
        ros_log_cols = [c for c in ros_df.columns if re.match(fr'^rosetta_L2_log10_{param}$', c)]
        ros_vals = []
        if ros_lin_cols:
            lin = pd.concat([ros_df[c] for c in ros_lin_cols], ignore_index=True)
            # Remove Rosetta sentinel values
            lin = lin[lin > -9999]
            if param in need_log:
                lin = lin[lin > 0]
                lin = np.log10(lin)
            ros_vals.append(lin)
        if ros_log_cols and param in need_log:
            log_ser = pd.concat([ros_df[c] for c in ros_log_cols], ignore_index=True)
            # Remove Rosetta sentinel values (already in log10 space)
            log_ser = log_ser[log_ser > -9999]
            ros_vals.append(log_ser)
        ros_series = pd.concat(ros_vals, ignore_index=True) if ros_vals else pd.Series(dtype=float)

        # POLARIS (overall)
        pol_series = pd.Series(polaris_vals.get(param, []))
        if param in need_log and len(pol_series) > 0:
            if param == 'alpha':
                pass
            else:
                pol_series = np.log10(pol_series)

        # Transform station if needed
        if param in need_log and len(st_series) > 0:
            st_series = np.log10(st_series)

        # Build long-form
        sources = []
        if len(st_series) > 0:
            sources.append(pd.DataFrame({'Source': 'Station', 'Value': st_series.dropna().values}))
        if len(ros_series) > 0:
            sources.append(pd.DataFrame({'Source': 'Rosetta', 'Value': ros_series.dropna().values}))
        if len(pol_series) > 0:
            sources.append(pd.DataFrame({'Source': 'POLARIS', 'Value': pol_series.dropna().values}))
        # GSHP (overall)
        g_series = pd.Series(gshp_vals.get(param, [])) if gshp_vals else pd.Series(dtype=float)
        if param in need_log and len(g_series) > 0:
            if param == 'alpha':
                pass
            else:
                g_series = np.log10(g_series)
        if len(g_series) > 0:
            sources.append(pd.DataFrame({'Source': 'GSHP', 'Value': g_series.dropna().values}))
        if not sources:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(PARAM_SYMBOLS.get(param, param))
            continue

        print(f'Param {param}')
        [print(s.iloc[0]['Source'], s['Value'].mean()) for s in sources]
        long_df = pd.concat(sources, ignore_index=True)
        sns.histplot(data=long_df, x='Value', hue='Source', bins=bins, element='step', stat='density',
                     common_bins=True, alpha=0.45, palette=palette, legend=False, ax=ax)

        # Labels
        xlab = f"log10({PARAM_SYMBOLS.get(param, param)})" if param in need_log else PARAM_SYMBOLS.get(param, param)
        ax.set_xlabel(xlab)
        ax.set_ylabel('Density')
        ax.set_title(PARAM_SYMBOLS.get(param, param), fontsize=13)
        ax.grid(True, which='major', alpha=0.3)

    # Figure-level legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=palette['Station'], edgecolor=palette['Station'], alpha=0.45, label='Station (MT)'),
               Patch(facecolor=palette['Rosetta'], edgecolor=palette['Rosetta'], alpha=0.45, label='Rosetta (10k CONUS)')]
    # Include POLARIS/GSHP if any param had values
    if any(len(v) > 0 for v in (_load_polaris_params(training_parquet).values())):
        handles.append(Patch(facecolor=palette['POLARIS'], edgecolor=palette['POLARIS'], alpha=0.45, label='POLARIS (10k CONUS)'))
    if gshp_vals and any(len(v) > 0 for v in gshp_vals.values()):
        handles.append(Patch(facecolor=palette['GSHP'], edgecolor=palette['GSHP'], alpha=0.45, label='GSHP (Global)'))
    fig.legend(handles=handles, loc='upper right')
    fig.suptitle('Combined Distribution Comparison (Rosetta L2; Others All Depths)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 0.98, 0.92])
    out_path = os.path.join(output_dir, 'vg_param_distributions_all_sources_combined.png')
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved combined distribution comparison to {out_path}")


if __name__ == '__main__':
    # Example wiring (edit paths as needed)
    home = os.path.expanduser('~')
    root = os.path.join(home, 'data', 'IrrigationGIS', 'soils')

    empirical_dir = os.path.join(root, 'soil_potential_obs', 'mt_mesonet', 'results_by_station')
    rosetta_pqt = os.path.join(root, 'rosetta', 'extracted_rosetta_points.parquet')
    training_pqt = os.path.join(root, 'swapstress', 'training', 'training_data.parquet')
    out_dir = os.path.join(root, 'swapstress', 'comparison_plots')

    # Combined all-depths, single figure with 4 panels
    gshp_csv = os.path.join(root, 'vg_paramaer_databases', 'wrc', 'WRC_dataset_surya_et_al_2021_final.csv')
    compare_parameter_distributions_combined(empirical_dir, rosetta_pqt, training_pqt, out_dir, bins=30,
                                             gshp_csv=gshp_csv)

# ========================= EOF ====================================================================
