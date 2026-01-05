"""
Unified training table builder for soil hydraulic parameter estimation.

This module combines Earth Engine features with VG parameters from multiple
sources into a single training table with consistent schema.

Output schema:
    - sample_id: Unique identifier ({source}_{original_id}_{depth})
    - source: Data source name
    - rosetta_level: Depth mapped to Rosetta levels 1-7
    - theta_r, theta_s, alpha, n: VG parameters (natural scale)
    - [EE features]: All extracted geospatial features

Usage:
    from map.data.build_training_table import build_unified_table
    df = build_unified_table(['gshp', 'ncss'], output_path='training.parquet')
"""
import os
import json
from glob import glob
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from map.data.source_registry import (
    SOURCES, get_source, DataSource, DataPaths,
    VG_PARAMS_NATURAL, VG_PARAMS_LOG10, TRAINING_TABLE_DROP_COLS,
)
from retention_curve.depth_utils import depth_to_rosetta_level


def _count_obs_from_preprocessed(preprocessed_dir: str, profile_id: str, depth_cm: float) -> int:
    """Count observations from a preprocessed CSV file for a given profile and depth."""
    csv_path = os.path.join(preprocessed_dir, f'{profile_id}.csv')
    if not os.path.exists(csv_path):
        return -9999
    try:
        df = pd.read_csv(csv_path)
        # Filter to matching depth (within tolerance)
        depth_tol = 1.0
        depth_mask = (df['depth_cm'] - depth_cm).abs() <= depth_tol
        return int(depth_mask.sum())
    except Exception:
        return -9999


def load_vg_params_from_csv(
        labels_path: str,
        source: DataSource,
        preprocessed_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load VG parameters from a CSV file (for sources like GSHP).

    Parameters
    ----------
    labels_path : str
        Path to labels CSV file.
    source : DataSource
        Source configuration.
    preprocessed_dir : str, optional
        Directory containing preprocessed observation CSVs for data_ct lookup.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [index_col, depth_cm, rosetta_level, data_ct, theta_r, theta_s, alpha, n]
    """
    if labels_path.endswith('.parquet'):
        df = pd.read_parquet(labels_path)
    else:
        df = pd.read_csv(labels_path, encoding='latin1')

    # Apply quality filter if configured
    if source.quality_filter_col and source.quality_filter_value:
        if source.quality_filter_col in df.columns:
            df = df[df[source.quality_filter_col] == source.quality_filter_value]

    # Ensure index column is string
    if source.index_col in df.columns:
        df[source.index_col] = df[source.index_col].astype(str)

    # Derive depth
    if source.depth_from_horizon and 'hzn_top' in df.columns and 'hzn_bot' in df.columns:
        df['depth_cm'] = (df['hzn_top'].astype(float) + df['hzn_bot'].astype(float)) / 2.0
    elif 'depth_cm' not in df.columns and 'depth' in df.columns:
        df['depth_cm'] = df['depth'].astype(float)

    # Add rosetta_level
    if 'depth_cm' in df.columns:
        df['rosetta_level'] = df['depth_cm'].apply(depth_to_rosetta_level)
    else:
        df['rosetta_level'] = np.nan

    # Add data_ct: check columns first, then try preprocessed files
    if 'data_ct' not in df.columns and 'obs_ct' in df.columns:
        df['data_ct'] = df['obs_ct'].astype(int)
    elif 'data_ct' not in df.columns and preprocessed_dir and os.path.isdir(preprocessed_dir):
        df['data_ct'] = df.apply(
            lambda row: _count_obs_from_preprocessed(
                preprocessed_dir, row[source.index_col], row.get('depth_cm', 0)
            ), axis=1
        )
    elif 'data_ct' not in df.columns:
        df['data_ct'] = -9999

    # Keep only needed columns
    vg_cols = [c for c in VG_PARAMS_NATURAL if c in df.columns]
    keep_cols = [source.index_col, 'depth_cm', 'rosetta_level', 'data_ct'] + vg_cols
    keep_cols = [c for c in keep_cols if c in df.columns]

    return df[keep_cols].copy()


def load_vg_params_from_json(
        fit_results_dir: str,
        source: DataSource,
        fit_method: str = 'bayes',
) -> pd.DataFrame:
    """
    Load VG parameters from fitted JSON files.

    Parameters
    ----------
    fit_results_dir : str
        Directory containing fitted JSON files.
    source : DataSource
        Source configuration.
    fit_method : list of str, optional
        Subdirectories to search within fit_results_dir.
    fit_method : str
        Fitting method to filter files by (e.g., 'nelder').

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [index_col, depth_cm, rosetta_level, theta_r, theta_s, alpha, n]
    """
    files = []
    subpath = os.path.join(fit_results_dir, fit_method)
    [files.append(os.path.join(subpath, f)) for f in os.listdir(subpath) if f.endswith('.json')]
    if len(files) == 0:
        raise ValueError('No fitted JSON files found.')

    rows = []
    for fp in files:
        try:
            with open(fp, 'r') as f:
                data = json.load(f)
                meta = data.pop('metadata', {})
        except Exception:
            continue

        for depth_str, res in data.items():
            if not isinstance(res, dict):
                continue
            if res.get('status') != 'Success':
                continue

            try:
                depth_cm = float(depth_str)
            except (TypeError, ValueError):
                continue

            params = res.get('parameters', {})
            depth_meta = meta.get(depth_str, {})

            # Get identifier based on source config
            identifier = depth_meta.get(source.index_col) or depth_meta.get('station') or depth_meta.get('profile_id')
            if not identifier:
                # Fall back to filename
                identifier = os.path.splitext(os.path.basename(fp))[0]

            try:
                data_ct = len(res.get('data', {}).get('theta', []))
                row = {
                    source.index_col: str(identifier),
                    'depth_cm': depth_cm,
                    'rosetta_level': depth_to_rosetta_level(depth_cm),
                    'data_ct': data_ct if data_ct > 0 else -9999,
                    'theta_r': float(params['theta_r']['value']),
                    'theta_s': float(params['theta_s']['value']),
                    'alpha': float(params['alpha']['value']),
                    'n': float(params['n']['value']),
                }
                rows.append(row)
            except (KeyError, TypeError, ValueError):
                continue

    if not rows:
        return pd.DataFrame(columns=[source.index_col, 'depth_cm', 'rosetta_level'] + VG_PARAMS_NATURAL[:4])

    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['theta_r', 'theta_s', 'alpha', 'n'])
    return df


def normalize_vg_params(
        df: pd.DataFrame,
        source_format: str,
        target_format: str = 'natural',
) -> pd.DataFrame:
    """
    Convert VG parameters between natural and log10 formats.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with VG parameter columns.
    source_format : str
        Current format: 'natural' or 'log10'.
    target_format : str
        Desired format: 'natural' or 'log10'.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized VG parameters.
    """
    if source_format == target_format:
        return df

    df = df.copy()

    if source_format == 'log10' and target_format == 'natural':
        if 'log10_alpha' in df.columns:
            df['alpha'] = 10 ** df['log10_alpha']
            df.drop(columns=['log10_alpha'], inplace=True)
        if 'log10_n' in df.columns:
            df['n'] = 10 ** df['log10_n']
            df.drop(columns=['log10_n'], inplace=True)
        if 'log10_Ks' in df.columns:
            df['Ks'] = 10 ** df['log10_Ks']
            df.drop(columns=['log10_Ks'], inplace=True)

    elif source_format == 'natural' and target_format == 'log10':
        if 'alpha' in df.columns:
            df['log10_alpha'] = np.log10(df['alpha'].clip(lower=1e-10))
            df.drop(columns=['alpha'], inplace=True)
        if 'n' in df.columns:
            df['log10_n'] = np.log10(df['n'].clip(lower=1e-10))
            df.drop(columns=['n'], inplace=True)

    return df


def load_embeddings(
        embeddings_dir: str,
        index_col: str,
) -> pd.DataFrame:
    """
    Load embeddings from per-site parquet files.

    Parameters
    ----------
    embeddings_dir : str
        Directory containing embedding parquet files.
    index_col : str
        Name of index column.

    Returns
    -------
    pd.DataFrame
        DataFrame with embeddings, indexed by site/station ID.
    """
    if not embeddings_dir or not os.path.isdir(embeddings_dir):
        return pd.DataFrame()

    emb_files = glob(os.path.join(embeddings_dir, '*.parquet'))
    if not emb_files:
        return pd.DataFrame()

    rows = {}
    for fp in tqdm(emb_files, desc='Loading embeddings', leave=False):
        identifier = os.path.splitext(os.path.basename(fp))[0]
        try:
            df = pd.read_parquet(fp)
            if len(df) >= 1:
                rows[str(identifier)] = df.iloc[0]
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    emb_df = pd.DataFrame.from_dict(rows, orient='index')
    emb_df.index.name = index_col
    return emb_df


def load_source_data(
        source: DataSource,
        data_root: str,
        fit_method: str = 'bayes',
        include_embeddings: bool = False,
) -> pd.DataFrame:
    """
    Load and join EE features with VG parameters for a single source.

    Parameters
    ----------
    source : DataSource
        Source configuration.
    data_root : str
        Root data directory.
    fit_method : str
        Fitting method for JSON files.
    include_embeddings : bool
        Whether to include embeddings.

    Returns
    -------
    pd.DataFrame
        Combined features and labels with standardized columns.
    """
    paths = DataPaths(data_root, source)

    # Load EE features
    ee_table = paths.ee_table
    if not os.path.exists(ee_table):
        raise FileNotFoundError(f"EE features not found: {ee_table}")

    ee_df = pd.read_parquet(ee_table)

    # EE tables always have index named 'station' (see ee_tables.py line 95)
    # but the values correspond to the source's actual index_col
    # Reset index and rename to the expected index_col
    if ee_df.index.name is not None:
        ee_df = ee_df.reset_index()

    ee_df[source.index_col] = ee_df[source.index_col].astype(str)

    # Normalize station names (lowercase, underscore to hyphen) for station-based sources
    if source.index_col == 'station':
        ee_df[source.index_col] = ee_df[source.index_col].str.lower().str.replace('_', '-')

    # Load VG parameters based on source type
    if source.vg_source == 'labels_csv':
        labels_path = paths.labels_file
        if not labels_path or not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        params_df = load_vg_params_from_csv(labels_path, source, preprocessed_dir=paths.preprocessed_dir)

    elif source.vg_source == 'fitted_json':
        fit_dir = paths.fit_results_dir
        if not fit_dir or not os.path.isdir(fit_dir):
            raise FileNotFoundError(f"Fit results not found: {fit_dir}")

        params_df = load_vg_params_from_json(fit_dir, source, fit_method)

    elif source.vg_source == 'rosetta_join':
        # Rosetta params are already joined in the EE table
        params_df = ee_df.copy()
        params_df['data_ct'] = -9999  # No measured data for PTF-derived params

    else:
        raise ValueError(f"Unknown vg_source: {source.vg_source}")

    # Normalize station names in params_df
    if source.index_col in ['station'] and source.index_col in params_df.columns:
        params_df[source.index_col] = params_df[source.index_col].str.lower().str.replace('_', '-')

    # Normalize VG params to natural scale
    params_df = normalize_vg_params(params_df, source.vg_param_format, 'natural')

    # Join params with EE features
    if source.vg_source != 'rosetta_join':
        # Drop any existing VG columns from EE data
        vg_cols_to_drop = [c for c in VG_PARAMS_NATURAL + VG_PARAMS_LOG10 if c in ee_df.columns]
        if vg_cols_to_drop:
            ee_df = ee_df.drop(columns=vg_cols_to_drop)

        # Merge
        merged = params_df.merge(
            ee_df.drop_duplicates(subset=source.index_col),
            on=source.index_col,
            how='left',
        )
    else:
        merged = params_df

    # Add embeddings if requested
    if include_embeddings:
        emb_dir = paths.embeddings_dir
        if emb_dir and os.path.isdir(emb_dir):
            emb_df = load_embeddings(emb_dir, source.index_col)
            if not emb_df.empty:
                emb_df = emb_df.reset_index()
                merged = merged.merge(emb_df, on=source.index_col, how='left')

    # Add source identifier
    merged['source'] = source.name

    # Create unique sample_id
    if 'depth_cm' in merged.columns:
        merged['sample_id'] = (
                source.name + '_' +
                merged[source.index_col].astype(str) + '_' +
                merged['depth_cm'].astype(str)
        )
    else:
        merged['sample_id'] = source.name + '_' + merged[source.index_col].astype(str)

    return merged


def build_unified_table(
        sources: List[str],
        data_root: str,
        output_path: Optional[str] = None,
        fit_method: str = 'bayes',
        include_embeddings: bool = False,
        vg_format: str = 'natural',
) -> pd.DataFrame:
    """
    Build a unified training table from multiple data sources.

    Parameters
    ----------
    sources : list of str
        Source names to include (e.g., ['gshp', 'ncss', 'mt_mesonet']).
    data_root : str
        Root data directory.
    output_path : str, optional
        Path to save output parquet file.
    fit_method : str
        Fitting method for JSON files.
    include_embeddings : bool
        Whether to include embeddings.
    vg_format : str
        Output VG parameter format: 'natural' or 'log10'.

    Returns
    -------
    pd.DataFrame
        Unified training table with consistent schema.
    """
    frames = []

    for source_name in sources:
        source = get_source(source_name)
        print(f"Loading {source_name}...")

        try:
            df = load_source_data(
                source,
                data_root,
                fit_method=fit_method,
                include_embeddings=include_embeddings,
            )

            # Convert VG format if needed
            if vg_format != 'natural':
                df = normalize_vg_params(df, 'natural', vg_format)

            frames.append(df)
            print(f"  Loaded {len(df)} samples from {source_name}")

        except FileNotFoundError as e:
            print(f"  Warning: Skipping {source_name} - {e}")
            continue

    if not frames:
        raise ValueError("No data loaded from any source")

    # Combine all sources
    combined = pd.concat(frames, ignore_index=True)

    # Set sample_id as index
    combined = combined.set_index('sample_id')

    # Drop rows with missing VG params
    vg_cols = VG_PARAMS_NATURAL[:4] if vg_format == 'natural' else ['theta_r', 'theta_s', 'log10_alpha', 'log10_n']
    vg_cols = [c for c in vg_cols if c in combined.columns]
    combined = combined.dropna(subset=vg_cols)

    # Ensure depth columns are present and valid
    if 'depth_cm' not in combined.columns:
        print("  Warning: depth_cm column missing")
    if 'rosetta_level' not in combined.columns:
        print("  Warning: rosetta_level column missing")
    elif combined['rosetta_level'].isna().all():
        print("  Warning: rosetta_level is all NaN")

    print(f"\nCombined table: {len(combined)} samples, {combined.shape[1]} columns")
    print(f"Sources: {combined['source'].value_counts().to_dict()}")
    if 'rosetta_level' in combined.columns:
        level_counts = combined['rosetta_level'].value_counts().sort_index().to_dict()
        print(f"Rosetta levels: {level_counts}")
    if 'depth_cm' in combined.columns:
        print(f"Depth range: {combined['depth_cm'].min():.1f} - {combined['depth_cm'].max():.1f} cm")

    # Drop metadata/duplicate columns before saving
    drop_cols = [c for c in TRAINING_TABLE_DROP_COLS if c in combined.columns]
    combined = combined.drop(columns=drop_cols)

    # Reorder columns with important ones first
    priority_cols = ['source', 'rosetta_level', 'depth_cm', 'data_ct', 'theta_r', 'theta_s', 'alpha', 'n']
    priority_cols = [c for c in priority_cols if c in combined.columns]
    other_cols = [c for c in combined.columns if c not in priority_cols]
    combined = combined[priority_cols + other_cols]

    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        combined.to_parquet(output_path)
        print(f"Saved to {output_path}")

    return combined


if __name__ == '__main__':

    home_ = os.path.expanduser('~')
    data_root_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils')
    output_dir_ = os.path.join(data_root_, 'swapstress', 'training')

    # Workflow flags
    build_gshp_ncss = False
    build_stations = False
    build_all_sources = True
    include_embeddings_ = True

    if build_gshp_ncss:
        sources_ = ['gshp', 'ncss']
        output_path_ = os.path.join(output_dir_, 'unified_gshp_ncss_250m.parquet')
        build_unified_table(
            sources=sources_,
            data_root=data_root_,
            output_path=output_path_,
            fit_method='bayes',
            include_embeddings=False,
            vg_format='natural',
        )

    if build_stations:
        sources_ = ['mt_mesonet', 'reesh']
        output_path_ = os.path.join(output_dir_, 'unified_stations_250m.parquet')
        build_unified_table(
            sources=sources_,
            data_root=data_root_,
            output_path=output_path_,
            fit_method='bayes',
            include_embeddings=include_embeddings_,
            vg_format='natural',
        )

    if build_all_sources:
        sources_ = ['gshp', 'ncss', 'mt_mesonet', 'reesh']
        output_path_ = os.path.join(output_dir_, 'unified_training_250m.parquet')
        if include_embeddings_:
            output_path_ = os.path.join(output_dir_, 'unified_training_emb_250m.parquet')
        build_unified_table(
            sources=sources_,
            data_root=data_root_,
            output_path=output_path_,
            fit_method='bayes',
            include_embeddings=include_embeddings_,
            vg_format='natural',
        )

# ========================= EOF ====================================================================
