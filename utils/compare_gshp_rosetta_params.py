import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

PARAMS = ['theta_r', 'theta_s', 'alpha', 'n']
LOG10_PARAMS = {'alpha', 'n'}


def _sanitize_uid(val):
    s = str(val) if pd.notnull(val) else ''
    s = re.sub(r"\s+", "", s)
    s = s.replace('.', '_')
    s = re.sub(r"_+", "_", s)
    return s


def load_gshp_groupped(csv_path):
    """Load GSHP CSV and aggregate one row per layer_id with parameter columns present.

    Returns a DataFrame with columns: uid, layer_id, [theta_r, theta_s, alpha, n], latitude, longitude, data_flag
    Only rows with good quality estimates are kept when possible.
    """
    try:
        df = pd.read_csv(csv_path, encoding='latin1', low_memory=False)
    except Exception:
        df = pd.read_csv(csv_path)

    # Filter to good quality if column exists
    if 'data_flag' in df.columns:
        mask_good = df['data_flag'].astype(str).str.lower().str.contains('good')
        if mask_good.any():
            df = df[mask_good]

    needed = ['layer_id', 'alpha', 'thetar', 'thetas', 'n']
    present = [c for c in needed if c in df.columns]
    if 'layer_id' not in present:
        raise ValueError("GSHP CSV missing 'layer_id' column.")

    # Aggregate one row per layer_id (first value per column)
    agg_spec = {c: 'first' for c in present if c != 'layer_id'}
    g = df[present].groupby('layer_id').agg(agg_spec).reset_index()

    # Coordinates if available
    lat_cols = ['latitude', 'latitude_decimal_degrees']
    lon_cols = ['longitude', 'longitude_decimal_degrees']
    lat_src = next((c for c in lat_cols if c in df.columns), None)
    lon_src = next((c for c in lon_cols if c in df.columns), None)
    if lat_src and lon_src:
        coords = df[['layer_id', lat_src, lon_src]].groupby('layer_id').agg('first').reset_index()
        coords = coords.rename(columns={lat_src: 'latitude', lon_src: 'longitude'})
        g = g.merge(coords, on='layer_id', how='left')

    # Rename params to our standard
    rename_map = {'thetar': 'theta_r', 'thetas': 'theta_s'}
    g = g.rename(columns=rename_map)

    # Sanitize uid from layer_id
    g['uid'] = g['layer_id'].apply(_sanitize_uid)
    return g


def find_rosetta_param_columns(df):
    """Return a mapping from param -> list of candidate Rosetta columns that end with _{param}.

    This supports flexible raster base prefixes in the extracted parquet.
    """
    mapping: Dict[str, List[str]] = {}
    for p in PARAMS:
        suffix = f'_{p}'
        cols = [c for c in df.columns if c.lower().endswith(suffix)]
        mapping[p] = cols
    return mapping


def choose_rosetta_columns(mapping, prefix_regex):
    """Choose a single Rosetta column per param.

    - If prefix_regex is provided, select first column matching the regex.
    - Else if exactly one candidate exists, select it.
    - Else return None for that param (caller can decide to aggregate or skip).
    """
    chosen: Dict[str, Optional[str]] = {}
    for p, cands in mapping.items():
        sel = None
        if prefix_regex:
            pat = re.compile(prefix_regex)
            for c in cands:
                if pat.search(c):
                    sel = c
                    break
        elif len(cands) == 1:
            sel = cands[0]
        chosen[p] = sel
    return chosen


def r2_score_np(y_true, y_pred):
    if y_true.size == 0:
        return float('nan')
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / ss_tot if ss_tot != 0 else np.nan)


def rmse_np(y_true, y_pred):
    if y_true.size == 0:
        return float('nan')
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compare_gshp_to_rosetta(
        gshp_csv,
        rosetta_parquet,
        out_dir,
        join_key='uid',
        rosetta_prefix_regex=None,
        make_scatter=True,
        bins=40,
):
    os.makedirs(out_dir, exist_ok=True)

    g = load_gshp_groupped(gshp_csv)
    r = pd.read_parquet(rosetta_parquet)

    # Try to propagate uid/layer_id in Rosetta DF if possible
    if 'uid' not in r.columns and 'layer_id' in r.columns:
        r['uid'] = r['layer_id'].apply(_sanitize_uid)

    r = r.groupby('uid').first()
    r['uid'] = r.index
    r.index = pd.Index(range(len(r)))

    # Choose Rosetta columns to use
    cand_map = find_rosetta_param_columns(r)
    chosen = choose_rosetta_columns(cand_map, rosetta_prefix_regex)

    # If some param has multiple candidates and none chosen, average across them
    rosetta_cols: Dict[str, str] = {}
    for p, sel in chosen.items():
        if sel is not None:
            rosetta_cols[p] = sel
        else:
            cands = cand_map[p]
            if len(cands) >= 1:
                # Average across candidate columns, ignoring missing/sentinel values
                r[f'_avg_{p}'] = r[cands].replace(-9999, np.nan).mean(axis=1)
                rosetta_cols[p] = f'_avg_{p}'

    # Merge on join_key if possible
    merged = None
    if join_key in g.columns and join_key in r.columns:
        merged = g.merge(r, on=join_key, how='left', suffixes=('_gshp', '_ros'))

    # Plot distributions regardless of join success
    for p in PARAMS:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(8, 6))

        # GSHP series
        if p in g.columns:
            s_g = g[p].copy()
            if p == 'alpha':
                s_g /= 100
                s_g = np.log10(s_g)
                s_g[s_g < -5] = np.nan
            s_g = s_g.dropna()

            sns.histplot(pd.Series(s_g), bins=bins, stat='density', color='#1f77b4', alpha=0.45, label='GSHP', ax=ax)

        # Rosetta series (all rows)
        if p in rosetta_cols:
            s_r = r[rosetta_cols[p]].copy()
            if p == 'alpha':
                s_r[s_r < -5] = np.nan
            if p == 'n':
                s_r[s_r < 0] = np.nan
            if 'theta' in p:
                s_r[s_r > 1] = np.nan
                s_r[s_r < 0] = np.nan
            s_r[s_r <= -9999] = np.nan
            s_r = s_r.dropna()
            sns.histplot(pd.Series(s_r), bins=bins, stat='density', color='#ff7f0e', alpha=0.45, label='Rosetta', ax=ax)

        xlab = f'log10({p})' if p in LOG10_PARAMS else p
        ax.set_xlabel(xlab)
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution Comparison: {p}')
        ax.legend(loc='best')
        plt.tight_layout()
        out_fp = os.path.join(out_dir, f'{p}_distribution_gshp_vs_rosetta.png')
        plt.savefig(out_fp, dpi=300)
        plt.close(fig)

    # If merged is available, compute metrics and optional scatter
    if merged is not None and not merged.empty:
        rows = []
        for p in PARAMS:
            if p not in g.columns or p not in rosetta_cols:
                continue

            y_true = merged[p].copy()
            y_pred = merged[rosetta_cols[p]].copy()
            # Handle sentinels
            dfp = pd.DataFrame({'true': y_true, 'pred': y_pred}).dropna()
            dfp[dfp <= -9999] = np.nan

            if dfp.empty:
                continue

            if 'theta' in p:
                dfp[dfp > 1] = np.nan
                dfp[dfp < 0] = np.nan

            if p == 'alpha':
                dfp['true'] /= 100
                dfp['true'] = np.log10(dfp['true'])
                vals = dfp.values
                vals[vals < -5.] = np.nan
                dfp.loc[:, dfp.columns] = vals

            if p == 'n':
                dfp['true'] = np.log10(dfp['true'])
                dfp[dfp < 0] = np.nan

            dfp = dfp.dropna()
            r2 = r2_score_np(dfp['true'].values, dfp['pred'].values)
            rmse = rmse_np(dfp['true'].values, dfp['pred'].values)
            bias = float((dfp['pred'] - dfp['true']).mean())

            rows.append({'param': p, 'n': len(dfp), 'r2': r2, 'rmse': rmse, 'bias': bias})

            if make_scatter:
                plt.style.use('seaborn-v0_8-whitegrid')
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(dfp['true'], dfp['pred'], s=12, alpha=0.7)
                vmin = min(dfp['true'].min(), dfp['pred'].min())
                vmax = max(dfp['true'].max(), dfp['pred'].max())
                ax.plot([vmin, vmax], [vmin, vmax], 'r--', lw=1)
                ax.set_xlabel(f"GSHP {'log10(' + p + ')' if p in LOG10_PARAMS else p}")
                ax.set_ylabel(f"Rosetta {'log10(' + p + ')' if p in LOG10_PARAMS else p}")
                ax.set_title(f"GSHP vs Rosetta: {p} (R2={r2:.2f}, RMSE={rmse:.3f})")
                plt.tight_layout()
                out_fp = os.path.join(out_dir, 'plots', f'{p}_scatter_gshp_vs_rosetta.png')
                plt.savefig(out_fp, dpi=300)
                plt.close(fig)

        if rows:
            metrics_df = pd.DataFrame(rows)
            metrics_df.to_csv(os.path.join(out_dir, 'gshp_rosetta_metrics.csv'), index=False)

        # Save merged compact table of chosen Rosetta columns + GSHP
        keep_cols = ['uid', 'layer_id'] if 'layer_id' in merged.columns else ['uid']
        for p in PARAMS:
            if p in merged.columns:
                keep_cols.append(p)
            if p in rosetta_cols and rosetta_cols[p] in merged.columns:
                keep_cols.append(rosetta_cols[p])
        merged[keep_cols].to_parquet(os.path.join(out_dir, 'gshp_rosetta_merged.parquet'), index=False)


if __name__ == '__main__':
    home = os.path.expanduser('~')
    root_ = os.path.join(home, 'data', 'IrrigationGIS')
    gshp_directory_ = os.path.join(root_, 'soils', 'vg_paramaer_databases', 'wrc')

    output_csv_ = os.path.join(root_, 'soils', 'vg_paramaer_databases', 'wrc', 'extracted_rosetta_points.parquet')

    soil_csv_path_ = os.path.join(gshp_directory_, 'WRC_dataset_surya_et_al_2021_final.csv')

    compare_gshp_to_rosetta(soil_csv_path_, output_csv_, out_dir=gshp_directory_, make_scatter=True)
# ======================== EOF ====================================================================
