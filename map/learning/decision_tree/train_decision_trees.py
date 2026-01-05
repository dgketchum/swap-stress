import json
import os
import re
from datetime import datetime
from pprint import pprint

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

from map.data import ee_feature_list

# Target columns (VG parameters)
VG_PARAMS = ['theta_r', 'theta_s', 'log10_alpha', 'log10_n', 'log10_Ks']
GSHP_PARAMS = ['theta_r', 'theta_s', 'alpha', 'n']

# Columns that are NOT features - these are either targets, identifiers, or metadata
# Everything else in the training table is assumed to be a feature
NON_FEATURE_COLS = {
    # Targets
    'theta_r', 'theta_s', 'alpha', 'n', 'Ks',
    'log10_alpha', 'log10_n', 'log10_Ks',
    # Identifiers
    'sample_id', 'profile_id', 'station', 'site_id',
    # Source/metadata
    'source', 'data_flag', 'SWCC_class', 'SWCC_classes',
    # Network metadata
    'nwsli_id', 'network', 'mesowest_i', 'obs_ct',
    # Spatial identifiers (not features)
    'MGRS_TILE', 'lat', 'lon', 'latitude', 'longitude',
}

# Depth columns - can be optionally included as features
DEPTH_COLS = {'rosetta_level', 'depth_cm', 'depth'}

# Legacy compatibility
DROP_FEATURES = list(NON_FEATURE_COLS | DEPTH_COLS)
ID_FEATURES = list(NON_FEATURE_COLS)
DEPTH_FEATURES = list(DEPTH_COLS)

_EMBEDDING_PATTERNS = [
    re.compile(r'^embedding_\d+$'),
    re.compile(r'^e\d+$'),
    re.compile(r'^A\d+$'),
    re.compile(r'^b\d+$'),
    re.compile(r'^US_R3H3_'),  # Rosetta layers
]


def filter_base_data_features(feature_cols):
    """Return only non-modeled 'base data' features.

    This removes learned/embedded features (e.g., satellite embeddings) so that
    the Random Forest is trained or analyzed using only surface reflectance,
    soil properties, and meteorology/climatology inputs.
    """
    polaris_keys = set(ee_feature_list._POLARIS.keys())
    smap_keys = set(ee_feature_list._SMAP_L4.keys())

    base_cols = []
    for c in feature_cols:
        name = str(c)
        # Drop embedded / modeled / Rosetta features
        if any(pat.match(name) for pat in _EMBEDDING_PATTERNS):
            continue
        # Drop POLARIS hydraulic properties (and any simple suffix forms)
        if name in polaris_keys:
            continue
        if any(name.startswith(k + '_') for k in polaris_keys):
            continue
        # Drop SMAP L4 variables (and any simple suffix forms)
        if name in smap_keys:
            continue
        if any(name.startswith(k + '_') for k in smap_keys):
            continue
        base_cols.append(c)
    return base_cols


def filter_soil_features(feature_cols):
    """Remove soil information features (SoilGrids + FAO/HWSD bands)."""
    soilgrids_keys = set(ee_feature_list._SOILGRIDS.keys())
    fao_keys = set(ee_feature_list._FAO_SOILS.keys())

    soil_prop_prefixes = sorted({k.split('_')[0] for k in soilgrids_keys})

    filtered = []
    for c in feature_cols:
        name = str(c)
        if name in soilgrids_keys:
            continue
        if any(name.startswith(p + '_') for p in soil_prop_prefixes):
            continue
        if name in fao_keys:
            continue
        filtered.append(c)
    return filtered


# Feature group definitions for selective exclusion
FEATURE_GROUPS = {
    'landsat': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'nd', 'nw', 'evi', 'gi'],
    'sentinel1': ['VV', 'VH', 'VH_VV'],
    'smap': list(ee_feature_list._SMAP_L4.keys()),
    'gridmet': list(ee_feature_list._GRIDMET_VARS.keys()),
    'soilgrids': list(ee_feature_list._SOILGRIDS.keys()),
    'fao': list(ee_feature_list._FAO_SOILS.keys()),
    'polaris': list(ee_feature_list._POLARIS.keys()),
    'terrain': ['elevation', 'slope', 'aspect', 'tpi_10000', 'tpi_22500', 'topoDiversity', 'b1'],
    'coords': ['lat', 'lon'],
    'embeddings': [],  # Handled by pattern matching in filter function
}


def filter_feature_groups(feature_cols, exclude_groups):
    """
    Remove features belonging to specified groups.

    Parameters
    ----------
    feature_cols : list
        List of feature column names.
    exclude_groups : list of str
        Group names to exclude (e.g., ['landsat', 'sentinel1', 'smap']).

    Returns
    -------
    list
        Filtered feature columns.
    """
    if not exclude_groups:
        return feature_cols

    exclude_prefixes = set()
    exclude_exact = set()

    for group in exclude_groups:
        if group not in FEATURE_GROUPS:
            print(f"Warning: Unknown feature group '{group}'. Available: {list(FEATURE_GROUPS.keys())}")
            continue

        group_features = FEATURE_GROUPS[group]
        for f in group_features:
            exclude_exact.add(f)
            exclude_prefixes.add(f + '_')

    # Special handling for embeddings (pattern-based)
    filter_embeddings = 'embeddings' in exclude_groups

    filtered = []
    for c in feature_cols:
        name = str(c)

        # Check exact match
        if name in exclude_exact:
            continue

        # Check prefix match (e.g., 'B2_mean_gs' starts with 'B2_')
        if any(name.startswith(p) for p in exclude_prefixes):
            continue

        # Check embedding patterns
        if filter_embeddings and any(pat.match(name) for pat in _EMBEDDING_PATTERNS):
            continue

        filtered.append(c)

    return filtered


def train_rf_rosetta(f, levels=None, base_data=False, include_soils=True):
    if levels is None:
        levels = list(range(1, 8))

    df = pd.read_parquet(f)
    rosetta_cols = [c for c in df.columns if any(p in c for p in VG_PARAMS) or c in GSHP_PARAMS]
    feature_cols = [c for c in df.columns if c not in rosetta_cols and c not in DROP_FEATURES]

    if base_data:
        feature_cols = filter_base_data_features(feature_cols)
    if not include_soils:
        feature_cols = filter_soil_features(feature_cols)

    all_metrics = {}
    for vert_level in levels:
        targets = [f'US_R3H3_L{vert_level}_VG_{p}' for p in VG_PARAMS]
        if not all(t in df.columns for t in targets):
            continue

        level_id = f'L{vert_level}_VG_combined'
        print(f"\n--- Training combined RF for Level {vert_level} ---")

        data = df[targets + feature_cols].copy()
        initial_len = len(data)
        for t in targets:
            data[data[t] <= -9999] = np.nan  # likely error: this sets entire rows to NaN
        data.dropna(subset=targets, inplace=True)
        print(f'Dropped {initial_len - len(data)} NaN records for Level {vert_level}')

        features = data[feature_cols]
        y = data[targets]

        x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        metrics = {}
        for i, target in enumerate(targets):
            param_name = VG_PARAMS[i]
            metrics[param_name] = {
                'r2': r2_score(y_test.iloc[:, i], y_pred[:, i]),
                'rmse': root_mean_squared_error(y_test.iloc[:, i], y_pred[:, i]),
                'mae': mean_absolute_error(y_test.iloc[:, i], y_pred[:, i]),
                'mean_val': y[target].mean().item(),
                'std_val': y[target].std().item(),
            }

        all_metrics[level_id] = metrics
        print(f"Metrics for combined RF model at level {vert_level}: {metrics}")

    return all_metrics


def train_rf_gshp(f, model_dir=None, features_csv=None, base_data=False,
                  include_soils=True, split_type='grouped'):
    df = pd.read_parquet(f)
    rosetta_cols = [c for c in df.columns if any(p in c for p in VG_PARAMS) or c in GSHP_PARAMS]

    if features_csv:
        feature_cols = pd.read_csv(features_csv)['features'].to_list()
    else:
        feature_cols = [c for c in df.columns if c not in rosetta_cols and c not in DROP_FEATURES]
        feature_cols = [c for c in feature_cols if not (len(c) == 3 and c.startswith('e'))]

    if base_data:
        feature_cols = filter_base_data_features(feature_cols)
    if not include_soils:
        feature_cols = filter_soil_features(feature_cols)

    all_metrics = {}
    targets = [p for p in GSHP_PARAMS if p in df.columns]
    if not targets:
        return all_metrics

    level_id = 'GSHP_VG_combined'
    print(f"\n--- Training RF for GSHP with {len(df)} Samples ---")

    # you already dropped these!
    # if 'data_flag' in df.columns:
    #     df = df[df['data_flag'] == 'good quality estimate']
    #     df.drop(columns=['data_flag'], inplace=True)
    #     if 'data_flag' in feature_cols:
    #         feature_cols.remove('data_flag')

    data = df[targets + feature_cols].copy()
    data = data.sort_index()
    # data = data.dropna()
    initial_len = len(data)

    target_info = {t: data[t].mean() for t in targets}
    print(f'Raw Target Means')
    [print(f'{k}: {v:.3f}') for k, v in target_info.items()]

    for t in targets:
        data[data[t] <= -9999] = np.nan

    data.dropna(subset=targets, inplace=True)
    print(f'Dropped {initial_len - len(data)} NaN records for GSHP combined')

    features = data[feature_cols]
    y = data[targets]

    target_info = {t: data[t].mean() for t in targets}
    print(f'Transformed Target Means')
    [print(f'{k}: {v:.3f}') for k, v in target_info.items()]

    target_info = {t: data[t].std() for t in targets}
    print(f'Transformed Target Std')
    [print(f'{k}: {v:.3f}') for k, v in target_info.items()]

    # Split strategy:
    # - 'naive': standard random row-wise split
    # - 'grouped': sequential split along the sorted index (default)
    if split_type == 'naive':
        x_train, x_test, y_train, y_test = train_test_split(
            features, y, test_size=0.3, random_state=42
        )
    elif split_type == 'grouped':
        split_idx = int(len(features) * 0.7)
        x_train, x_test = features[:split_idx], features[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
    else:
        raise ValueError(f"Unsupported split_type '{split_type}'. Use 'naive' or 'grouped'.")

    print(f'{len(x_train)} training, {len(x_test)} test samples')

    model = RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)

    if model_dir:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, 'rf_gshp_model.joblib')
        joblib.dump(model, model_path)
        print(f'Saved model to {model_path}')
        features_path = os.path.join(model_dir, 'rf_gshp_features.json')
        with open(features_path, 'w') as f:
            json.dump(feature_cols, f, indent=4)
        print(f'Saved features to {features_path}')

    y_pred = model.predict(x_test)

    metrics = {}
    for i, target in enumerate(targets):
        param_name = targets[i]
        metrics[param_name] = {
            'r2': r2_score(y_test.iloc[:, i], y_pred[:, i]),
            'rmse': root_mean_squared_error(y_test.iloc[:, i], y_pred[:, i]),
            'mae': mean_absolute_error(y_test.iloc[:, i], y_pred[:, i]),
            'mean_val': y[target].mean().item(),
            'std_val': y[target].std().item(),
        }
    all_metrics[level_id] = metrics
    pprint(f"Metrics for combined RF model GSHP: ")
    [print(f"{k}: {v['r2']}") for k, v in metrics.items()]

    return all_metrics


def train_rf_stations(table_path, model_dir=None, base_data=False,
                      include_soils=True, split_type='grouped'):
    df = pd.read_parquet(table_path)
    rosetta_cols = [c for c in df.columns if any(p in c for p in VG_PARAMS) or c in GSHP_PARAMS]
    feature_cols = [c for c in df.columns if c not in rosetta_cols and c not in DROP_FEATURES]

    if base_data:
        feature_cols = filter_base_data_features(feature_cols)
    if not include_soils:
        feature_cols = filter_soil_features(feature_cols)

    targets = [p for p in GSHP_PARAMS if p in df.columns]
    station_ids = df['station'].astype(str)

    data = df[targets + feature_cols].copy()
    initial_len = len(data)
    for t in targets:
        data[data[t] <= -9999] = np.nan
    data.dropna(subset=targets, inplace=True)
    print(f'Dropped {initial_len - len(data)} NaN records for Stations combined')

    features = data[feature_cols]
    y = data[targets]

    # Split strategy:
    # - 'naive': standard random row-wise split
    # - 'grouped': group-aware split by station (default, preserves station integrity)
    if split_type == 'naive':
        x_train, x_test, y_train, y_test = train_test_split(
            features, y, test_size=0.2, random_state=42
        )
    elif split_type == 'grouped':
        station_ids = station_ids.loc[data.index]
        unique_stations = station_ids.dropna().unique()
        tr_stations, te_stations = train_test_split(
            unique_stations, test_size=0.2, random_state=42
        )
        train_mask = station_ids.isin(tr_stations).values
        test_mask = station_ids.isin(te_stations).values

        x_train, x_test = features[train_mask], features[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
    else:
        raise ValueError(f"Unsupported split_type '{split_type}'. Use 'naive' or 'grouped'.")
    print(f'{len(x_train)} training, {len(x_test)} test samples')

    model = RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)

    if model_dir:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, 'rf_stations_model.joblib')
        joblib.dump(model, model_path)
        print(f'Saved model to {model_path}')
        features_path = os.path.join(model_dir, 'rf_stations_features.json')
        with open(features_path, 'w') as f:
            json.dump(feature_cols, f, indent=4)
        print(f'Saved features to {features_path}')

    y_pred = model.predict(x_test)

    metrics = {}
    for i, target in enumerate(targets):
        param_name = targets[i]
        metrics[param_name] = {
            'r2': r2_score(y_test.iloc[:, i], y_pred[:, i]),
            'rmse': root_mean_squared_error(y_test.iloc[:, i], y_pred[:, i]),
            'mae': mean_absolute_error(y_test.iloc[:, i], y_pred[:, i]),
            'mean_val': y[target].mean().item(),
            'std_val': y[target].std().item(),
        }

    return metrics


def _train_rf_per_level(
        table_path,
        sources=None,
        targets=None,
        model_dir=None,
        base_data=False,
        include_soils=True,
        split_type='grouped',
        group_col='source',
        test_size=0.2,
        n_estimators=250,
        filter_levels=None,
        min_data_ct=None,
        exclude_features=None,
        exclude_feature_groups=None,
        return_importance=False,
):
    """
    Train separate RF models for each Rosetta depth level.

    Internal helper called by train_rf() when depth_handling='per_level'.
    """
    if targets is None:
        targets = GSHP_PARAMS

    df = pd.read_parquet(table_path)
    print(f"Loaded {len(df)} samples from {table_path}")

    if sources and 'source' in df.columns:
        df = df[df['source'].isin(sources)]
        print(f"Filtered to sources {sources}: {len(df)} samples")

    # Filter by minimum data count
    if min_data_ct is not None and 'data_ct' in df.columns:
        df = df[(df['data_ct'] >= min_data_ct) & (df['data_ct'] != -9999)]
        print(f"Filtered to min_data_ct >= {min_data_ct}: {len(df)} samples")

    if 'rosetta_level' not in df.columns:
        raise ValueError("rosetta_level column required for per_level training")

    levels = filter_levels if filter_levels else sorted(df['rosetta_level'].dropna().unique())
    levels = [int(lvl) for lvl in levels]

    all_metrics = {}

    for level in levels:
        level_df = df[df['rosetta_level'] == level].copy()
        if len(level_df) < 10:
            print(f"\nLevel {level}: Skipping (only {len(level_df)} samples)")
            continue

        print(f"\n--- Training RF for Level {level} ({len(level_df)} samples) ---")

        # Identify feature columns (exclude depth since we're splitting by level)
        rosetta_cols = [c for c in level_df.columns if any(p in c for p in VG_PARAMS) or c in GSHP_PARAMS]
        feature_cols = [c for c in level_df.columns if c not in rosetta_cols and c not in DROP_FEATURES]
        feature_cols = [c for c in feature_cols if c not in ['source']]

        if base_data:
            feature_cols = filter_base_data_features(feature_cols)
        if not include_soils:
            feature_cols = filter_soil_features(feature_cols)

        # Apply feature group exclusions
        if exclude_feature_groups:
            feature_cols = filter_feature_groups(feature_cols, exclude_feature_groups)

        # Apply individual feature exclusions
        if exclude_features:
            feature_cols = [c for c in feature_cols if c not in exclude_features]

        level_targets = [t for t in targets if t in level_df.columns]
        if not level_targets:
            print(f"  No target columns found, skipping")
            continue

        data = level_df[level_targets + feature_cols].copy()
        initial_len = len(data)
        for t in level_targets:
            data.loc[data[t] <= -9999, t] = np.nan
        data = data.dropna(subset=level_targets)
        print(f"  Dropped {initial_len - len(data)} rows with invalid targets")

        if len(data) < 10:
            print(f"  Skipping (only {len(data)} valid samples)")
            continue

        features = data[feature_cols]
        y = data[level_targets]

        # Split
        if split_type == 'naive':
            x_train, x_test, y_train, y_test = train_test_split(
                features, y, test_size=test_size, random_state=42
            )
        elif split_type == 'grouped' and group_col in level_df.columns:
            group_ids = level_df.loc[data.index, group_col].astype(str)
            unique_groups = group_ids.dropna().unique()
            if len(unique_groups) < 2:
                x_train, x_test, y_train, y_test = train_test_split(
                    features, y, test_size=test_size, random_state=42
                )
            else:
                tr_groups, te_groups = train_test_split(
                    unique_groups, test_size=test_size, random_state=42
                )
                train_mask = group_ids.isin(tr_groups).values
                test_mask = group_ids.isin(te_groups).values
                x_train, x_test = features[train_mask], features[test_mask]
                y_train, y_test = y[train_mask], y[test_mask]
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                features, y, test_size=test_size, random_state=42
            )

        print(f"  Split: {len(x_train)} train, {len(x_test)} test")

        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        model.fit(x_train, y_train)

        if model_dir:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            source_str = '_'.join(sources) if sources else 'unified'
            model_path = os.path.join(model_dir, f'rf_{source_str}_L{level}_model.joblib')
            joblib.dump(model, model_path)
            print(f"  Saved model to {model_path}")

        y_pred = model.predict(x_test)

        level_metrics = {'n_train': len(x_train), 'n_test': len(x_test), 'n_features': len(feature_cols), 'targets': {}}
        for i, target in enumerate(level_targets):
            if len(level_targets) > 1:
                y_true = y_test.iloc[:, i]
                y_p = y_pred[:, i]
            else:
                y_true = y_test.iloc[:, 0]
                y_p = y_pred[:, 0] if y_pred.ndim > 1 else y_pred

            level_metrics['targets'][target] = {
                'r2': float(r2_score(y_true, y_p)),
                'rmse': float(root_mean_squared_error(y_true, y_p)),
                'mae': float(mean_absolute_error(y_true, y_p)),
            }
            print(f"  {target}: R²={level_metrics['targets'][target]['r2']:.3f}")

        # Add feature importance if requested
        if return_importance:
            importances = model.feature_importances_
            importance_dict = {
                feature_cols[i]: float(importances[i])
                for i in np.argsort(importances)[::-1]
            }
            level_metrics['feature_importance'] = importance_dict

        all_metrics[f'L{level}'] = level_metrics

    return all_metrics


def train_rf(
        table_path,
        sources=None,
        targets=None,
        model_dir=None,
        base_data=False,
        include_soils=True,
        split_type='grouped',
        group_col='source',
        test_size=0.2,
        n_estimators=250,
        include_rosetta=False,
        rosetta_table=None,
        rosetta_levels=None,
        rosetta_weight=1.0,
        depth_handling='none',
        filter_levels=None,
        min_data_ct=None,
        exclude_features=None,
        exclude_feature_groups=None,
        return_importance=False,
):
    """
    Train Random Forest on unified training data with flexible source selection.

    This function works with the unified training table produced by
    map.data.build_training_table and supports train-time source filtering,
    multiple split strategies, and optional Rosetta pre-training data.

    Parameters
    ----------
    table_path : str
        Path to unified training parquet file.
    sources : list of str, optional
        Filter to specific sources (e.g., ['gshp', 'ncss']). If None, use all.
    targets : list of str, optional
        Target columns. If None, uses ['theta_r', 'theta_s', 'alpha', 'n'].
    model_dir : str, optional
        Directory to save model and feature list.
    base_data : bool
        If True, remove embeddings/POLARIS/SMAP features.
    include_soils : bool
        If True, include SoilGrids/FAO features.
    split_type : str
        Split strategy:
        - 'naive': random row-wise split
        - 'grouped': group-aware split by group_col
        - 'sequential': sequential split (first N% train, rest test)
    group_col : str
        Column to group by for 'grouped' split (default: 'source').
    test_size : float
        Fraction of data for testing (default: 0.2).
    n_estimators : int
        Number of trees in forest (default: 250).
    include_rosetta : bool
        If True, add Rosetta data for pre-training.
    rosetta_table : str, optional
        Path to Rosetta training data parquet.
    rosetta_levels : list of int, optional
        Which Rosetta levels to include (default: all 1-7).
    rosetta_weight : float
        Sample weight for Rosetta data (default: 1.0).
    depth_handling : str
        How to handle depth in training:
        - 'none': exclude depth from features (default, legacy behavior)
        - 'feature': include rosetta_level as a categorical feature
        - 'continuous': include depth_cm as a continuous feature
        - 'both': include both rosetta_level and depth_cm
        - 'per_level': train separate models for each Rosetta level
    filter_levels : list of int, optional
        Only include samples from these Rosetta levels (1-7).
    min_data_ct : int, optional
        Minimum number of observations required per sample. Samples with
        data_ct < min_data_ct (or data_ct == -9999) are excluded.
    exclude_features : list of str, optional
        Specific feature column names to exclude from training.
    exclude_feature_groups : list of str, optional
        Feature group names to exclude. Available groups:
        'landsat', 'sentinel1', 'smap', 'gridmet', 'soilgrids', 'fao',
        'polaris', 'terrain', 'coords', 'embeddings'.
    return_importance : bool
        If True, include feature importances in the returned metrics dict.

    Returns
    -------
    dict
        Metrics dictionary with R², RMSE, MAE per target.
        If depth_handling='per_level', returns dict keyed by level.
        If return_importance=True, includes 'feature_importance' key.
    """
    if targets is None:
        targets = GSHP_PARAMS

    # Handle per-level training separately
    if depth_handling == 'per_level':
        return _train_rf_per_level(
            table_path=table_path,
            sources=sources,
            targets=targets,
            model_dir=model_dir,
            base_data=base_data,
            include_soils=include_soils,
            split_type=split_type,
            group_col=group_col,
            test_size=test_size,
            n_estimators=n_estimators,
            filter_levels=filter_levels,
            min_data_ct=min_data_ct,
            exclude_features=exclude_features,
            exclude_feature_groups=exclude_feature_groups,
            return_importance=return_importance,
        )

    # Load unified table
    df = pd.read_parquet(table_path)
    print(f"Loaded {len(df)} samples from {table_path}")

    # Filter to specific sources if requested
    if sources and 'source' in df.columns:
        df = df[df['source'].isin(sources)]
        print(f"Filtered to sources {sources}: {len(df)} samples")

    # Filter to specific Rosetta levels if requested
    if filter_levels and 'rosetta_level' in df.columns:
        df = df[df['rosetta_level'].isin(filter_levels)]
        print(f"Filtered to levels {filter_levels}: {len(df)} samples")

    # Filter by minimum data count (observations used in VG fitting)
    if min_data_ct is not None and 'data_ct' in df.columns:
        # Exclude samples with insufficient observations or missing count (-9999)
        df = df[(df['data_ct'] >= min_data_ct) & (df['data_ct'] != -9999)]
        print(f"Filtered to min_data_ct >= {min_data_ct}: {len(df)} samples")

    # Optionally add Rosetta pre-training data
    if include_rosetta and rosetta_table and os.path.exists(rosetta_table):
        rosetta_df = pd.read_parquet(rosetta_table)
        if rosetta_levels is None:
            rosetta_levels = list(range(1, 8))

        # Rosetta has different column structure - need to reshape
        rosetta_rows = []
        for level in rosetta_levels:
            level_targets = [f'US_R3H3_L{level}_VG_{p}' for p in VG_PARAMS]
            if not all(t in rosetta_df.columns for t in level_targets):
                continue

            level_df = rosetta_df.copy()
            # Map Rosetta columns to standard names
            for i, param in enumerate(VG_PARAMS):
                src_col = f'US_R3H3_L{level}_VG_{param}'
                if param == 'log10_alpha':
                    level_df['alpha'] = 10 ** level_df[src_col]
                elif param == 'log10_n':
                    level_df['n'] = 10 ** level_df[src_col]
                elif param == 'log10_Ks':
                    level_df['Ks'] = 10 ** level_df[src_col]
                elif param in ['theta_r', 'theta_s']:
                    level_df[param] = level_df[src_col]

            level_df['source'] = 'rosetta'
            level_df['rosetta_level'] = level
            rosetta_rows.append(level_df)

        if rosetta_rows:
            rosetta_combined = pd.concat(rosetta_rows, ignore_index=True)
            # Keep only columns that exist in both
            common_cols = list(set(df.columns) & set(rosetta_combined.columns))
            df = pd.concat([df[common_cols], rosetta_combined[common_cols]], ignore_index=True)
            print(f"Added Rosetta pre-training data: {len(df)} total samples")

    # Identify feature columns: everything NOT in NON_FEATURE_COLS or DEPTH_COLS
    # Also exclude any Rosetta-prefixed columns (those are targets, not features)
    rosetta_target_pattern = re.compile(r'^US_R3H3_L\d+_VG_')
    exclude_cols = NON_FEATURE_COLS | DEPTH_COLS
    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols and not rosetta_target_pattern.match(c)
    ]

    # Handle depth columns based on depth_handling option
    if depth_handling == 'feature':
        # Add rosetta_level as feature
        if 'rosetta_level' in df.columns:
            feature_cols.append('rosetta_level')
    elif depth_handling == 'continuous':
        # Add depth_cm as feature
        if 'depth_cm' in df.columns:
            feature_cols.append('depth_cm')
    elif depth_handling == 'both':
        # Add both depth columns as features
        if 'rosetta_level' in df.columns:
            feature_cols.append('rosetta_level')
        if 'depth_cm' in df.columns:
            feature_cols.append('depth_cm')
    # depth_handling == 'none': don't add any depth columns (default)

    if base_data:
        feature_cols = filter_base_data_features(feature_cols)
    if not include_soils:
        feature_cols = filter_soil_features(feature_cols)

    # Apply feature group exclusions
    if exclude_feature_groups:
        feature_cols = filter_feature_groups(feature_cols, exclude_feature_groups)
        print(f"After excluding groups {exclude_feature_groups}: {len(feature_cols)} features")

    # Apply individual feature exclusions
    if exclude_features:
        feature_cols = [c for c in feature_cols if c not in exclude_features]
        print(f"After excluding {len(exclude_features)} features: {len(feature_cols)} features")

    # Verify targets exist
    targets = [t for t in targets if t in df.columns]
    if not targets:
        raise ValueError(f"No target columns found in data. Available: {df.columns.tolist()}")

    print(f"Training RF with {len(feature_cols)} features, targets: {targets}")

    # Prepare data
    data = df[targets + feature_cols].copy()
    initial_len = len(data)

    # Clean invalid values
    for t in targets:
        data.loc[data[t] <= -9999, t] = np.nan
    data = data.dropna(subset=targets)
    print(f"Dropped {initial_len - len(data)} rows with invalid targets")

    # Get group IDs for splitting - derive from sample_id index
    # sample_id format: {source}_{id}_{depth}, e.g., 'gshp_131_10.5'
    # Group by source + first 2 chars of id, e.g., 'gshp_13'
    if split_type == 'grouped':
        def extract_group(sample_id):
            parts = str(sample_id).split('_')
            if len(parts) >= 2:
                return f"{parts[0]}_{parts[1][:2]}"
            return str(sample_id)
        group_ids = pd.Series([extract_group(idx) for idx in data.index], index=data.index)
    else:
        group_ids = None

    features = data[feature_cols]
    y = data[targets]

    # Split based on strategy
    if split_type == 'naive':
        x_train, x_test, y_train, y_test = train_test_split(
            features, y, test_size=test_size, random_state=42
        )
    elif split_type == 'sequential':
        split_idx = int(len(features) * (1 - test_size))
        x_train, x_test = features.iloc[:split_idx], features.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    elif split_type == 'grouped':
        if group_ids is None:
            raise ValueError(f"group_col '{group_col}' not found for grouped split")
        unique_groups = group_ids.dropna().unique()
        tr_groups, te_groups = train_test_split(
            unique_groups, test_size=test_size, random_state=42
        )
        train_mask = group_ids.isin(tr_groups).values
        test_mask = group_ids.isin(te_groups).values
        x_train, x_test = features[train_mask], features[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
    else:
        raise ValueError(f"Unknown split_type: {split_type}")

    print(f"Split: {len(x_train)} train, {len(x_test)} test samples")

    # Handle sample weights for Rosetta if applicable
    sample_weight = None
    if include_rosetta and rosetta_weight != 1.0 and 'source' in df.columns:
        train_sources = df.loc[x_train.index, 'source']
        sample_weight = np.where(train_sources == 'rosetta', rosetta_weight, 1.0)

    # Train model
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train, sample_weight=sample_weight)

    # Save model if requested
    if model_dir:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Create descriptive model name
        source_str = '_'.join(sources) if sources else 'unified'
        model_path = os.path.join(model_dir, f'rf_{source_str}_model.joblib')
        joblib.dump(model, model_path)
        print(f"Saved model to {model_path}")

        features_path = os.path.join(model_dir, f'rf_{source_str}_features.json')
        with open(features_path, 'w') as fp:
            json.dump(feature_cols, fp, indent=4)
        print(f"Saved features to {features_path}")

    # Compute metrics
    y_pred = model.predict(x_test)

    metrics = {
        'config': {
            'sources': sources,
            'split_type': split_type,
            'group_col': group_col,
            'test_size': test_size,
            'n_train': len(x_train),
            'n_test': len(x_test),
            'n_features': len(feature_cols),
            'base_data_only': base_data,
            'include_soils': include_soils,
            'include_rosetta': include_rosetta,
            'depth_handling': depth_handling,
            'filter_levels': filter_levels,
            'min_data_ct': min_data_ct,
            'exclude_features': exclude_features,
            'exclude_feature_groups': exclude_feature_groups,
        },
        'targets': {},
    }

    for i, target in enumerate(targets):
        if len(targets) > 1:
            y_true = y_test.iloc[:, i]
            y_p = y_pred[:, i]
        else:
            y_true = y_test.iloc[:, 0]
            y_p = y_pred[:, 0] if y_pred.ndim > 1 else y_pred

        metrics['targets'][target] = {
            'r2': float(r2_score(y_true, y_p)),
            'rmse': float(root_mean_squared_error(y_true, y_p)),
            'mae': float(mean_absolute_error(y_true, y_p)),
            'mean_val': float(y[target].mean()),
            'std_val': float(y[target].std()),
        }

    # Add feature importance if requested
    if return_importance:
        importances = model.feature_importances_
        importance_dict = {
            feature_cols[i]: float(importances[i])
            for i in np.argsort(importances)[::-1]
        }
        metrics['feature_importance'] = importance_dict

    print("\nResults:")
    for target, m in metrics['targets'].items():
        print(f"  {target}: R²={m['r2']:.3f}, RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}")

    return metrics


if __name__ == '__main__':

    # Legacy workflows (use existing separate training tables)
    run_gshp_workflow = False
    run_rosetta_workflow = False
    run_stations_workflow = False

    # Unified workflow (uses unified training table from build_training_table.py)
    run_unified_workflow = False

    # Depth experiment workflows
    run_depth_as_feature = True  # Include rosetta_level as categorical feature
    run_depth_continuous = False  # Include depth_cm as continuous feature
    run_per_level_models = False  # Train separate model per Rosetta level

    # Common settings
    base_data_only = False
    include_soils = True
    split_type_ = 'grouped'
    min_data_ct_ = 4
    include_rosetta_ = True

    home_ = os.path.expanduser('~')
    root_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'swapstress')

    features_csv_ = os.path.join(root_, 'training', 'current_features.csv')

    metrics_dir_ = os.path.join(root_, 'training', 'metrics')
    models_dir_ = os.path.join(root_, 'training', 'models')
    for d in [metrics_dir_, models_dir_]:
        if not os.path.exists(d):
            os.makedirs(d)

    unified_table_ = os.path.join(root_, 'training', 'unified_training_emb_250m.parquet')
    rosetta_table_ = os.path.join(root_, 'training', 'training_data.parquet')
    sources_ = ['gshp', 'ncss', 'mt_mesonet', 'reesh']

    if run_unified_workflow:
        # Train single model across all depths (legacy behavior, no depth info)
        unified_metrics_ = train_rf(
            table_path=unified_table_,
            sources=sources_,
            model_dir=models_dir_,
            min_data_ct=min_data_ct_,
            base_data=base_data_only,
            include_soils=include_soils,
            split_type=split_type_,
            group_col='source',
            test_size=0.2,
            depth_handling='none',
        )

        unified_dst_ = os.path.join(metrics_dir_, 'learn_unified')
        if not os.path.exists(unified_dst_):
            os.makedirs(unified_dst_)
        ts_ = datetime.now().strftime('%Y%m%d_%H%M%S')
        source_str_ = '_'.join(sources_) if sources_ else 'all'
        with open(os.path.join(unified_dst_, f'RandomForest_unified_{source_str_}_{ts_}.json'), 'w') as f:
            json.dump(unified_metrics_, f, indent=4)
        print(f'Wrote unified RF metrics')

    if run_depth_as_feature:
        # Train single model with rosetta_level as a categorical feature
        depth_metrics_ = train_rf(
            table_path=unified_table_,
            sources=sources_,
            model_dir=models_dir_,
            min_data_ct=min_data_ct_,
            base_data=base_data_only,
            include_soils=include_soils,
            include_rosetta=include_rosetta_,
            split_type=split_type_,
            group_col='source',
            test_size=0.2,
            depth_handling='feature',  # rosetta_level as feature
        )

        depth_dst_ = os.path.join(metrics_dir_, 'learn_unified_depth_feature')
        if not os.path.exists(depth_dst_):
            os.makedirs(depth_dst_)
        ts_ = datetime.now().strftime('%Y%m%d_%H%M%S')
        source_str_ = '_'.join(sources_) if sources_ else 'all'
        with open(os.path.join(depth_dst_, f'RandomForest_depth_feature_{source_str_}_{ts_}.json'), 'w') as f:
            json.dump(depth_metrics_, f, indent=4)
        print(f'Wrote depth-as-feature RF metrics')

    if run_depth_continuous:
        # Train single model with depth_cm as a continuous feature
        depth_metrics_ = train_rf(
            table_path=unified_table_,
            sources=sources_,
            model_dir=models_dir_,
            base_data=base_data_only,
            include_soils=include_soils,
            split_type=split_type_,
            group_col='source',
            test_size=0.2,
            depth_handling='continuous',  # depth_cm as feature
        )

        depth_dst_ = os.path.join(metrics_dir_, 'learn_unified_depth_continuous')
        if not os.path.exists(depth_dst_):
            os.makedirs(depth_dst_)
        ts_ = datetime.now().strftime('%Y%m%d_%H%M%S')
        source_str_ = '_'.join(sources_) if sources_ else 'all'
        with open(os.path.join(depth_dst_, f'RandomForest_depth_continuous_{source_str_}_{ts_}.json'), 'w') as f:
            json.dump(depth_metrics_, f, indent=4)
        print(f'Wrote depth-continuous RF metrics')

    if run_per_level_models:
        # Train separate model for each Rosetta level (like Rosetta approach)
        level_metrics_ = train_rf(
            table_path=unified_table_,
            sources=sources_,
            model_dir=models_dir_,
            base_data=base_data_only,
            include_soils=include_soils,
            split_type=split_type_,
            group_col='source',
            test_size=0.2,
            depth_handling='per_level',  # Separate models per level
            filter_levels=None,  # All levels, or e.g., [2, 3, 4] for specific levels
        )

        level_dst_ = os.path.join(metrics_dir_, 'learn_unified_per_level')
        if not os.path.exists(level_dst_):
            os.makedirs(level_dst_)
        ts_ = datetime.now().strftime('%Y%m%d_%H%M%S')
        source_str_ = '_'.join(sources_) if sources_ else 'all'
        with open(os.path.join(level_dst_, f'RandomForest_per_level_{source_str_}_{ts_}.json'), 'w') as f:
            json.dump(level_metrics_, f, indent=4)
        print(f'Wrote per-level RF metrics')

    elif run_gshp_workflow:
        gshp_file_ = os.path.join(root_, 'training', 'gshp_training_data_emb_250m.parquet')
        gshp_metrics_ = train_rf_gshp(
            gshp_file_,
            model_dir=models_dir_,
            features_csv=features_csv_,
            base_data=base_data_only,
            include_soils=include_soils,
            split_type=split_type_,
        )
        gshp_dst_ = os.path.join(metrics_dir_, 'learn_gshp')
        if not os.path.exists(gshp_dst_):
            os.makedirs(gshp_dst_)
        ts_ = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(os.path.join(gshp_dst_, f'RandomForest_GSHP_{ts_}.json'), 'w') as f:
            json.dump(gshp_metrics_, f, indent=4)
        print(f'Wrote GSHP RF metrics')

    elif run_rosetta_workflow:
        rosetta_file_ = os.path.join(root_, 'training', 'training_data.parquet')
        rosetta_metrics_ = train_rf_rosetta(
            rosetta_file_,
            levels=list(range(1, 8)),
            base_data=base_data_only,
            include_soils=include_soils,
        )
        rosetta_dst_ = os.path.join(metrics_dir_, 'learn_rosetta')
        if not os.path.exists(rosetta_dst_):
            os.makedirs(rosetta_dst_)
        ts_ = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(os.path.join(rosetta_dst_, f'RandomForest_Rosetta_{ts_}.json'), 'w') as f:
            json.dump(rosetta_metrics_, f, indent=4)
        print(f'Wrote Rosetta RF metrics')

    elif run_stations_workflow:
        training_table = os.path.join(root_, 'training', 'stations_training_table_250m.parquet')
        station_metrics_ = train_rf_stations(
            training_table,
            model_dir=models_dir_,
            base_data=base_data_only,
            include_soils=include_soils,
            split_type='grouped',
        )
        stations_dst_ = os.path.join(metrics_dir_, 'learn_stations')
        if not os.path.exists(stations_dst_):
            os.makedirs(stations_dst_)
        ts_ = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(os.path.join(stations_dst_, f'RandomForest_Stations_{ts_}.json'), 'w') as f:
            json.dump(station_metrics_, f, indent=4)
        print(f'Wrote Stations RF metrics')

# ========================= EOF ====================================================================
