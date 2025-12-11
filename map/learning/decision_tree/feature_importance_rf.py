import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from map.learning.decision_tree.train_decision_trees import (
    DROP_FEATURES,
    VG_PARAMS,
    GSHP_PARAMS,
    filter_base_data_features,
    filter_soil_features,
)


def rf_feature_importance_gshp(f, features_csv=None, n_estimators=250,
                               random_state=42, base_data=False, include_soils=True):
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

    targets = [p for p in GSHP_PARAMS if p in df.columns]
    if not targets:
        return []

    data = df[targets + feature_cols].copy()
    initial_len = len(data)
    for t in targets:
        data.loc[data[t] <= -9999, t] = np.nan
    data.dropna(subset=targets, inplace=True)
    print(f'Dropped {initial_len - len(data)} NaN records for GSHP importance')

    features = data[feature_cols]
    y = data[targets]

    x_train, _, y_train, _ = train_test_split(features, y, test_size=0.2, random_state=random_state)

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(x_train, y_train)

    importances = model.feature_importances_
    pairs = list(zip(feature_cols, importances))
    pairs.sort(key=lambda x: x[1], reverse=True)

    result = [{'feature': n, 'importance': float(v)} for n, v in pairs]
    return result


def rf_feature_importance_rosetta(f, levels=None, n_estimators=250,
                                  random_state=42, base_data=False, include_soils=True):
    if levels is None:
        levels = list(range(1, 8))

    df = pd.read_parquet(f)
    rosetta_cols = [c for c in df.columns if any(p in c for p in VG_PARAMS) or c in GSHP_PARAMS]
    feature_cols = [c for c in df.columns if c not in rosetta_cols and c not in DROP_FEATURES]

    if base_data:
        feature_cols = filter_base_data_features(feature_cols)
    if not include_soils:
        feature_cols = filter_soil_features(feature_cols)

    all_importances = {}
    for vert_level in levels:
        targets = [f'US_R3H3_L{vert_level}_VG_{p}' for p in VG_PARAMS]
        if not all(t in df.columns for t in targets):
            continue

        level_id = f'L{vert_level}_VG_combined'
        print(f'--- Computing RF importance for Level {vert_level} ---')

        data = df[targets + feature_cols].copy()
        initial_len = len(data)
        for t in targets:
            data.loc[data[t] <= -9999, t] = np.nan
        data.dropna(subset=targets, inplace=True)
        print(f'Dropped {initial_len - len(data)} NaN records for Level {vert_level}')

        features = data[feature_cols]
        y = data[targets]

        x_train, _, y_train, _ = train_test_split(features, y, test_size=0.2, random_state=random_state)

        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        model.fit(x_train, y_train)

        importances = model.feature_importances_
        pairs = list(zip(feature_cols, importances))
        pairs.sort(key=lambda x: x[1], reverse=True)

        all_importances[level_id] = [{'feature': n, 'importance': float(v)} for n, v in pairs]

    return all_importances


if __name__ == '__main__':
    run_gshp_importance = True
    run_rosetta_importance = False
    base_data_only = True
    include_soils = False

    home_ = os.path.expanduser('~')
    root_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'swapstress')

    features_csv_ = os.path.join(root_, 'training', 'current_features.csv')

    metrics_dir_ = os.path.join(root_, 'training', 'metrics')
    out_dir_ = os.path.join(metrics_dir_, 'feature_importance')
    if not os.path.exists(out_dir_):
        os.makedirs(out_dir_)

    if run_gshp_importance:
        gshp_file_ = os.path.join(root_, 'training', 'gshp_training_data_emb_250m.parquet')
        importances_ = rf_feature_importance_gshp(gshp_file_, features_csv=features_csv_,
                                                  base_data=base_data_only,
                                                  include_soils=include_soils)
        ts_ = datetime.now().strftime('%Y%m%d_%H%M%S')
        suffix_parts = []
        if base_data_only:
            suffix_parts.append('BaseData')
        if not include_soils:
            suffix_parts.append('NoSoils')
        suffix_ = f"_{'_'.join(suffix_parts)}" if suffix_parts else ''
        out_path_ = os.path.join(out_dir_, f'RF_Importance_GSHP{suffix_}_{ts_}.json')
        with open(out_path_, 'w') as f:
            json.dump(importances_, f, indent=4)
        print(f'Wrote GSHP RF feature importances')

    elif run_rosetta_importance:
        rosetta_file_ = os.path.join(root_, 'training', 'training_data.parquet')
        importances_by_level_ = rf_feature_importance_rosetta(rosetta_file_, levels=list(range(1, 8)),
                                                              base_data=base_data_only,
                                                              include_soils=include_soils)
        ts_ = datetime.now().strftime('%Y%m%d_%H%M%S')
        suffix_parts = []
        if base_data_only:
            suffix_parts.append('BaseData')
        if not include_soils:
            suffix_parts.append('NoSoils')
        suffix_ = f"_{'_'.join(suffix_parts)}" if suffix_parts else ''
        out_path_ = os.path.join(out_dir_, f'RF_Importance_Rosetta{suffix_}_{ts_}.json')
        with open(out_path_, 'w') as f:
            json.dump(importances_by_level_, f, indent=4)
        print(f'Wrote Rosetta RF feature importances')

# ========================= EOF ====================================================================
