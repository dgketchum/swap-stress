"""Build 4-panel boxplots comparing VG parameter distributions across sources.

Combines Station empirical fits, Rosetta extractions (linear + log columns),
POLARIS means, GSHP published values, Rosetta (Train), and optional Rosetta
SWRC fits into four box-and-whisker panels: θr, θs, log10(α), log10(n).
"""
import os
import re
from glob import glob

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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


def _load_rosetta_swrc_params(results_dir):
    """Load fitted Rosetta SWRC parameter values from saved JSONs."""
    json_files = glob(os.path.join(results_dir, '**', '*_fit_results.json'), recursive=True)
    if not json_files:
        return {}
    vals = {'theta_r': [], 'theta_s': [], 'alpha': [], 'n': []}
    for f in json_files:
        try:
            data = pd.read_json(f, typ='series').to_dict()
        except ValueError:
            import json as _json
            with open(f, 'r') as jf:
                data = _json.load(jf)
        for depth, res in data.items():
            try:
                if res.get('status') != 'Success':
                    continue
            except AttributeError:
                continue
            params = res.get('parameters', {})
            for p in list(vals.keys()):
                v = params.get(p, {}).get('value')
                if v is not None:
                    vals[p].append(v)
    return {k: np.array(v, dtype=float) for k, v in vals.items()}


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


def _load_rosetta_training_params(parquet_path):
    """
    Load original Rosetta training data parameters.
    """
    if not parquet_path or not os.path.exists(parquet_path):
        return {}
    df = pd.read_parquet(parquet_path)
    dct = {}
    for p in ['theta_r', 'theta_s', 'alpha', 'npar']:
        vals = df[p].values
        if p == 'npar':
            dct['n'] = vals
        else:
            dct[p] = vals
    return dct


def compare_parameter_distributions_combined(empirical_results_dir, rosetta_parquet, training_parquet,
                                             output_dir, gshp_csv=None, rosetta_training_parquet=None,
                                             rosetta_swrc_results_dir=None):
    """
    Combines all depths/levels into a single population per parameter and
    produces a single figure with boxplots for [theta_r, theta_s, alpha, n].

    - For alpha and n, compares distributions in log10 space across sources.
    - Sources: Station empirical fits, Rosetta extracts, POLARIS (overall), GSHP, and Rosetta Training.
    """
    os.makedirs(output_dir, exist_ok=True)

    station_df = _load_empirical_results(empirical_results_dir)
    ros_df = _load_rosetta_params(rosetta_parquet)
    polaris_vals = _load_polaris_params(training_parquet)
    gshp_vals = _load_gshp_params(gshp_csv) if gshp_csv else {}
    rosetta_training_vals = _load_rosetta_training_params(rosetta_training_parquet) if rosetta_training_parquet else {}
    rosetta_swrc_vals = _load_rosetta_swrc_params(rosetta_swrc_results_dir) if rosetta_swrc_results_dir else {}

    params = ['theta_r', 'theta_s', 'alpha', 'n']
    need_log = {'alpha', 'n'}
    palette = {'Station': '#1f77b4', 'Rosetta': '#ff7f0e', 'POLARIS': '#2ca02c', 'GSHP': '#9467bd',
               'Rosetta (Train)': '#d62728'}

    plt.style.use('seaborn-v0_8-whitegrid')

    all_sources_dfs = []
    for param in params:
        # Station: gather all station_Lx_param columns
        st_cols = [c for c in station_df.columns if re.match(fr'^station_L\d+_{param}$', c)]
        st_series = pd.concat([station_df[c] for c in st_cols], ignore_index=True) if st_cols else pd.Series(
            dtype=float)

        # Rosetta: gather linear and log10 columns
        ros_lin_cols = [c for c in ros_df.columns if re.match(fr'^rosetta_L2_{param}$', c)]
        ros_log_cols = [c for c in ros_df.columns if re.match(fr'^rosetta_L2_log10_{param}$', c)]
        ros_vals = []
        if ros_lin_cols:
            lin = pd.concat([ros_df[c] for c in ros_lin_cols], ignore_index=True)
            lin = lin[lin > -9999]
            if param in need_log:
                lin = lin[lin > 0]
                lin = np.log10(lin)
            ros_vals.append(lin)
        if ros_log_cols:
            log_ser = pd.concat([ros_df[c] for c in ros_log_cols], ignore_index=True)
            log_ser = log_ser[log_ser > -9999]
            # Cast Rosetta log10 columns back to non-log space for consistency
            if param in need_log:
                lin_from_log = 10 ** log_ser
                lin_from_log = lin_from_log[lin_from_log > 0]
                if param in need_log:
                    lin_from_log = np.log10(lin_from_log)
                ros_vals.append(lin_from_log)
            else:
                # likely error: non-log param should not have a log10 column
                pass
        ros_series = pd.concat(ros_vals, ignore_index=True) if ros_vals else pd.Series(dtype=float)

        # POLARIS (overall)
        pol_series = pd.Series(polaris_vals.get(param, []))
        if param in need_log and len(pol_series) > 0:
            pol_series = pol_series[pol_series > 0]
            pol_series = np.log10(pol_series)

        # Transform station if needed
        if param in need_log and len(st_series) > 0:
            st_series = np.log10(st_series[st_series > 0])

        # GSHP (overall)
        g_series = pd.Series(gshp_vals.get(param, [])) if gshp_vals else pd.Series(dtype=float)
        if param in need_log and len(g_series) > 0:
            if param == 'alpha':
                g_series = g_series / 100.0
                g_series = g_series[g_series > 0]
                g_series = np.log10(g_series)
            else:
                g_series = g_series[g_series > 0]
                g_series = np.log10(g_series)

        # Rosetta Training (new)
        rt_series = pd.Series(rosetta_training_vals.get(param, [])) if rosetta_training_vals else pd.Series(
            dtype=float)
        if param in need_log and len(rt_series) > 0:
            rt_series = np.log10(rt_series[rt_series > 0])

        # Rosetta (SWRC fits)
        rs_series = pd.Series(rosetta_swrc_vals.get(param, [])) if rosetta_swrc_vals else pd.Series(dtype=float)
        if param in need_log and len(rs_series) > 0:
            rs_series = np.log10(rs_series[rs_series > 0])

        param_symbol = PARAM_SYMBOLS.get(param, param)
        param_label = f"log10({param_symbol})" if param in need_log else param_symbol

        sources_data = {
            'Station': st_series,
            'Rosetta': ros_series,
            'POLARIS': pol_series,
            'GSHP': g_series,
            'Rosetta (Train)': rt_series
        }

        for source_name, data_series in sources_data.items():
            if len(data_series) > 0:
                df = pd.DataFrame({'Value': data_series.dropna().values})
                df['Source'] = source_name
                df['Parameter'] = param_label
                all_sources_dfs.append(df)

    if not all_sources_dfs:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Parameter Distributions by Source')
    else:
        long_df = pd.concat(all_sources_dfs, ignore_index=True)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = np.array(axes).reshape(2, 2)

        label_map = {
            'theta_r': PARAM_SYMBOLS.get('theta_r', 'theta_r'),
            'theta_s': PARAM_SYMBOLS.get('theta_s', 'theta_s'),
            'alpha': f"log10({PARAM_SYMBOLS.get('alpha', 'alpha')})",
            'n': f"log10({PARAM_SYMBOLS.get('n', 'n')})",
        }
        param_order = ['theta_r', 'theta_s', 'alpha', 'n']

        for i, p in enumerate(param_order):
            r, c = divmod(i, 2)
            ax = axes[r, c]
            lab = label_map[p]
            sdf = long_df[long_df['Parameter'] == lab]
            if sdf.empty:
                ax.axis('off')
                continue
            order = [s for s in palette.keys() if s in sdf['Source'].unique()]
            sns.boxplot(data=sdf, x='Source', y='Value', order=order, palette=palette, ax=ax)
            ax.set_title(str(lab))
            ax.set_xlabel('Source')
            ax.set_ylabel('Value')
            ax.grid(True, which='major', alpha=0.3)
            for tick in ax.get_xticklabels():
                tick.set_rotation(30)
                tick.set_ha('right')

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'vg_param_distributions_all_sources_combined.png')
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved combined distribution comparison to {out_path}")


if __name__ == '__main__':
    # Example wiring (edit paths as needed)
    home = os.path.expanduser('~')
    rosetta_soil_proj = os.path.join(home, 'PycharmProjects', 'rosetta-soil')
    root = os.path.join(home, 'data', 'IrrigationGIS', 'soils')

    empirical_dir = os.path.join(root, 'soil_potential_obs', 'mt_mesonet', 'results_by_station')
    rosetta_pqt = os.path.join(root, 'rosetta', 'extracted_rosetta_points.parquet')
    training_pqt = os.path.join(root, 'swapstress', 'training', 'training_data.parquet')
    out_dir = os.path.join(root, 'swapstress', 'comparison_plots')
    rosetta_training_pqt = os.path.join(rosetta_soil_proj, 'rosetta', 'db', 'Data.parquet')

    # Combined all-depths, single figure with 4 panels
    gshp_csv = os.path.join(root, 'soil_potential_obs', 'gshp', 'WRC_dataset_surya_et_al_2021_final.csv')
    compare_parameter_distributions_combined(empirical_dir, rosetta_pqt, training_pqt, out_dir, bins=30,
                                             gshp_csv=gshp_csv, rosetta_training_parquet=rosetta_training_pqt)

# ========================= EOF ====================================================================
