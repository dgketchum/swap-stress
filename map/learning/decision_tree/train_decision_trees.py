import json
import os
from datetime import datetime
from pprint import pprint

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

DROP_FEATURES = ['MGRS_TILE', 'station', 'rosetta_level', 'profile_id',
                 'nwsli_id', 'network', 'mesowest_i', 'data_flag', 'obs_ct', 'SWCC_class',
                 ]

VG_PARAMS = ['theta_r', 'theta_s', 'log10_alpha', 'log10_n', 'log10_Ks']
GSHP_PARAMS = ['theta_r', 'theta_s', 'alpha', 'n']


def train_rf_rosetta(f, levels=None):
    if levels is None:
        levels = list(range(1, 8))

    df = pd.read_parquet(f)
    rosetta_cols = [c for c in df.columns if any(p in c for p in VG_PARAMS) or c in GSHP_PARAMS]
    feature_cols = [c for c in df.columns if c not in rosetta_cols and c not in DROP_FEATURES]

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


def train_rf_gshp(f, model_dir=None, features_csv=None):
    df = pd.read_parquet(f)
    rosetta_cols = [c for c in df.columns if any(p in c for p in VG_PARAMS) or c in GSHP_PARAMS]

    if features_csv:
        feature_cols = pd.read_csv(features_csv)['features'].to_list()
    else:
        feature_cols = [c for c in df.columns if c not in rosetta_cols and c not in DROP_FEATURES]

    all_metrics = {}
    targets = [p for p in GSHP_PARAMS if p in df.columns]
    if not targets:
        return all_metrics

    level_id = 'GSHP_VG_combined'
    print(f"\n--- Training RF for GSHP with {len(df)} Samples ---")
    if 'data_flag' in df.columns:
        df = df[df['data_flag'] == 'good quality estimate']
        df.drop(columns=['data_flag'], inplace=True)
        if 'data_flag' in feature_cols:
            feature_cols.remove('data_flag')

    data = df[targets + feature_cols].copy()
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

    x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)
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


def train_rf_stations(table_path, model_dir=None):
    df = pd.read_parquet(table_path)
    rosetta_cols = [c for c in df.columns if any(p in c for p in VG_PARAMS) or c in GSHP_PARAMS]
    feature_cols = [c for c in df.columns if c not in rosetta_cols and c not in DROP_FEATURES]

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

    # Group-aware split: ensure all depths from a station stay in the same split
    station_ids = station_ids.loc[data.index]
    unique_stations = station_ids.dropna().unique()
    tr_stations, te_stations = train_test_split(unique_stations, test_size=0.2, random_state=42)
    train_mask = station_ids.isin(tr_stations)
    test_mask = station_ids.isin(te_stations)

    x_train, x_test = features.loc[train_mask], features.loc[test_mask]
    y_train, y_test = y.loc[train_mask], y.loc[test_mask]
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


if __name__ == '__main__':

    run_gshp_workflow = True
    run_rosetta_workflow = False
    run_stations_workflow = False

    home_ = os.path.expanduser('~')
    root_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'swapstress')

    # features_csv_ = os.path.join(root_, 'training', 'current_features.csv')
    features_csv_ = None

    metrics_dir_ = os.path.join(root_, 'training', 'metrics')
    models_dir_ = os.path.join(root_, 'training', 'models')
    for d in [metrics_dir_, models_dir_]:
        if not os.path.exists(d):
            os.makedirs(d)

    if run_gshp_workflow:
        gshp_file_ = os.path.join(root_, 'training', 'gshp_training_data_emb_250m.parquet')
        gshp_metrics_ = train_rf_gshp(gshp_file_, model_dir=models_dir_, features_csv=features_csv_)
        gshp_dst_ = os.path.join(metrics_dir_, 'learn_gshp')
        if not os.path.exists(gshp_dst_):
            os.makedirs(gshp_dst_)
        ts_ = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(os.path.join(gshp_dst_, f'RandomForest_GSHP_{ts_}.json'), 'w') as f:
            json.dump(gshp_metrics_, f, indent=4)
        print(f'Wrote GSHP RF metrics')

    elif run_rosetta_workflow:

        rosetta_file_ = os.path.join(root_, 'training', 'training_data.parquet')
        rosetta_metrics_ = train_rf_rosetta(rosetta_file_, levels=list(range(1, 8)))
        rosetta_dst_ = os.path.join(metrics_dir_, 'learn_rosetta')
        if not os.path.exists(rosetta_dst_):
            os.makedirs(rosetta_dst_)
        ts_ = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(os.path.join(rosetta_dst_, f'RandomForest_Rosetta_{ts_}.json'), 'w') as f:
            json.dump(rosetta_metrics_, f, indent=4)
        print(f'Wrote Rosetta RF metrics')

    elif run_stations_workflow:

        training_table = os.path.join(root_, 'training', 'stations_training_table_250m.parquet')
        station_metrics_ = train_rf_stations(training_table, model_dir=models_dir_)
        stations_dst_ = os.path.join(metrics_dir_, 'learn_stations')
        ts_ = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(os.path.join(stations_dst_, f'RandomForest_Stations_{ts_}.json'), 'w') as f:
            json.dump(station_metrics_, f, indent=4)
        print(f'Wrote Stations RF metrics')

# ========================= EOF ====================================================================
