"""
DEPRECATED: This module is superseded by build_training_table.py

Use build_training_table.py with source='gshp' instead:

    from map.data.build_training_table import build_unified_table
    build_unified_table(['gshp'], data_root, output_path, include_embeddings=True)

This file is retained for backward compatibility only.
"""
import os
import warnings
from tqdm import tqdm
from glob import glob
import numpy as np
import pandas as pd

from retention_curve.depth_utils import depth_to_rosetta_level

LABELS_COLS_KEEP = ['alpha', 'data_flag', 'n', 'theta_r', 'theta_s']


def _deprecation_warning():
    warnings.warn(
        "gshp_training_table is deprecated. Use build_training_table.py instead.",
        DeprecationWarning,
        stacklevel=3
    )

def build_gshp_training_table(ee_features_pqt, labels_csv, out_file, index_col='profile_id', filter_good_quality=True,
                              embeddings=None, features_path=None):
    """DEPRECATED: Use build_training_table.build_unified_table() instead."""
    _deprecation_warning()
    if not features_path or not os.path.exists(features_path):
        raise ValueError('features_path is required and must point to stations current_features.csv')
    # for "simplicity", all EE extracts concatenated in ee_tables.py will by indexed with 'station'
    ee_df = pd.read_parquet(ee_features_pqt)
    ee_df.index.name = index_col

    labels_df = pd.read_csv(labels_csv, encoding='latin1') if labels_csv.endswith('.csv') else pd.read_parquet(labels_csv)
    if index_col in labels_df.columns:
        labels_df[index_col] = labels_df[index_col].astype(str)
        labels_df = labels_df.set_index(index_col)
    if filter_good_quality and 'data_flag' in labels_df.columns:
        labels_df = labels_df[labels_df['data_flag'] == 'good quality estimate']

    # derive Rosetta vertical level from available depth info
    depth_series = None
    if 'hzn_top' in labels_df.columns and 'hzn_bot' in labels_df.columns:
        depth_series = (labels_df['hzn_top'].astype(float) + labels_df['hzn_bot'].astype(float)) / 2.0
    elif 'depth_cm' in labels_df.columns:
        depth_series = labels_df['depth_cm']
    elif 'depth' in labels_df.columns:
        depth_series = labels_df['depth']
    if depth_series is not None:
        labels_df['rosetta_level'] = [depth_to_rosetta_level(v) for v in depth_series]
    else:
        labels_df['rosetta_level'] = np.nan

    keep = [c for c in LABELS_COLS_KEEP + ['rosetta_level'] if c in labels_df.columns]
    labels_df = labels_df[keep]

    if any([c for c in LABELS_COLS_KEEP if c in ee_df.columns]):
       for c in  LABELS_COLS_KEEP:
           if c in ee_df.columns:
               ee_df.drop(columns=[c], inplace=True)

    final_df = ee_df.join(labels_df, how='left').dropna(subset=['theta_r', 'theta_s', 'alpha', 'n'])

    if embeddings and os.path.isdir(embeddings):
        emb_files = glob(os.path.join(embeddings, '*.parquet'))
        if emb_files:
            rows = {}
            for fp in tqdm(emb_files, desc='Adding Embedding Features', total=len(emb_files)):
                pid = os.path.splitext(os.path.basename(fp))[0]
                try:
                    df = pd.read_parquet(fp)
                    vec = df.iloc[0] if len(df) >= 1 else None
                except Exception:
                    vec = None  # likely error: malformed embedding file
                if vec is not None:
                    rows[str(pid)] = vec
            if rows:
                emb_df = pd.DataFrame.from_dict(rows, orient='index')
                emb_df.index.name = index_col
                final_df = final_df.join(emb_df, how='left')

    # Enforce uniform feature set (from station_training_table current_features.csv)
    if features_path.endswith('.csv'):
        feats_df = pd.read_csv(features_path)
        col = 'features' if 'features' in feats_df.columns else feats_df.columns[0]
        listed = feats_df[col].dropna().astype(str).tolist()
    else:
        raise ValueError('Unsupported features file; expected CSV list of features')
    missing = [c for c in listed if c not in final_df.columns]
    if missing:
        raise ValueError(f'Missing required features in GSHP table: {missing}')
    label_cols = [c for c in LABELS_COLS_KEEP if c in final_df.columns]
    if 'rosetta_level' in final_df.columns:
        label_cols.append('rosetta_level')
    final_df = final_df[listed + label_cols]

    out_dir = os.path.dirname(out_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # final_df.dropna(inplace=True)
    final_df.to_parquet(out_file)
    print(f'{len(final_df)} x {final_df.shape[1]} rows GSHP training samples written to {out_file}')


if __name__ == '__main__':
    run_gshp_workflow = True

    home_ = os.path.expanduser('~')
    root_ = os.path.join(home_, 'data', 'IrrigationGIS')

    if run_gshp_workflow:
        gshp_directory_ = os.path.join(root_, 'soils', 'soil_potential_obs', 'gshp')
        labels_csv_ = os.path.join(gshp_directory_, 'WRC_dataset_surya_et_al_2021_final_clean.csv')
        ee_features_pqt_ = os.path.join(root_, 'soils', 'swapstress', 'training', 'gshp_ee_data_250m.parquet')
        out_file_ = os.path.join(root_, 'soils', 'swapstress', 'training', 'gshp_training_data_emb_250m.parquet')

        vwc_root_ = '/data/ssd2/swapstress/vwc'
        embeddings_ = os.path.join(vwc_root_, 'embeddings', 'gshp')

        features_csv_ = os.path.join(root_, 'soils', 'swapstress', 'training', 'current_features.csv')
        build_gshp_training_table(
            ee_features_pqt=ee_features_pqt_,
            labels_csv=labels_csv_,
            out_file=out_file_,
            index_col='profile_id',
            filter_good_quality=True,
            embeddings=embeddings_,
            features_path=features_csv_,
        )
# ========================= EOF ====================================================================
