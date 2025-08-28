import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

DROP_FEATURES = ['MGRS_TILE', 'lat', 'lon']
VG_PARAMS = ['theta_r', 'theta_s', 'log10_alpha', 'log10_n', 'log10_Ks']


def run_rf_training(f, mode='single', levels=None):
    if levels is None:
        levels = list(range(1, 8))

    df = pd.read_parquet(f)
    rosetta_cols = [c for c in df.columns if any(p in c for p in VG_PARAMS)]
    feature_cols = [c for c in df.columns if c not in rosetta_cols and c not in DROP_FEATURES]

    all_metrics = {}

    if mode == 'single':
        all_metrics = {p: {vl: None for vl in range(1, 8)} for p in VG_PARAMS}
        for param in VG_PARAMS:
            for vert_level in levels:
                target = f'US_R3H3_L{vert_level}_VG_{param}'
                if target not in df.columns:
                    continue

                print(f"\n--- Training RF for {target} ---")
                data = df[[target] + feature_cols].copy()
                initial_len = len(data)
                data[data[target] <= -9999] = np.nan
                data.dropna(subset=[target], inplace=True)
                print(f'Dropped {initial_len - len(data)} NaN records for {target}')

                features = data[feature_cols]
                y = data[target]

                x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

                model = RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)
                model.fit(x_train, y_train)

                y_pred = model.predict(x_test)

                metrics = {
                    'r2': r2_score(y_test, y_pred),
                    'mse': mean_squared_error(y_test, y_pred),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'rmse': root_mean_squared_error(y_test, y_pred),
                    'mean_val': y.mean().item(),
                    'std_val': y.std().item(),
                }
                all_metrics[param][vert_level] = metrics

    elif mode == 'combined':
        for vert_level in levels:
            targets = [f'US_R3H3_L{vert_level}_VG_{p}' for p in VG_PARAMS]
            if not all(t in df.columns for t in targets):
                continue

            level_id = f'L{vert_level}_VG_combined'
            print(f"\n--- Training combined RF for Level {vert_level} ---")

            data = df[targets + feature_cols].copy()
            initial_len = len(data)
            for t in targets:
                data[data[t] <= -9999] = np.nan
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

def run_xgb_training(f, mode='single', levels=None):
    if levels is None:
        levels = list(range(1, 8))

    df = pd.read_parquet(f)
    rosetta_cols = [c for c in df.columns if any(p in c for p in VG_PARAMS)]
    feature_cols = [c for c in df.columns if c not in rosetta_cols and c not in DROP_FEATURES]

    all_metrics = {}

    if mode == 'single':
        all_metrics = {p: {vl: None for vl in range(1, 8)} for p in VG_PARAMS}
        for param in VG_PARAMS:
            for vert_level in levels:
                target = f'US_R3H3_L{vert_level}_VG_{param}'
                if target not in df.columns:
                    continue

                print(f"\n--- Training XGB for {target} ---")
                data = df[[target] + feature_cols].copy()
                initial_len = len(data)
                data[data[target] <= -9999] = np.nan
                data.dropna(subset=[target], inplace=True)
                print(f'Dropped {initial_len - len(data)} NaN records for {target}')

                features = data[feature_cols]
                y = data[target]

                x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

                model = xgb.XGBRegressor(n_estimators=250, random_state=42, n_jobs=-1,
                                         objective='reg:squarederror', learning_rate=0.1,
                                         max_depth=5, subsample=0.8, colsample_bytree=0.8)
                model.fit(x_train, y_train)

                y_pred = model.predict(x_test)

                metrics = {
                    'r2': r2_score(y_test, y_pred),
                    'mse': mean_squared_error(y_test, y_pred),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'rmse': root_mean_squared_error(y_test, y_pred),
                    'mean_val': y.mean().item(),
                    'std_val': y.std().item(),
                }
                all_metrics[param][vert_level] = metrics

    elif mode == 'combined':
        for vert_level in levels:
            targets = [f'US_R3H3_L{vert_level}_VG_{p}' for p in VG_PARAMS]
            if not all(t in df.columns for t in targets):
                continue

            level_id = f'L{vert_level}_VG_combined'
            print(f"\n--- Training combined XGB for Level {vert_level} ---")

            data = df[targets + feature_cols].copy()
            initial_len = len(data)
            for t in targets:
                data[data[t] <= -9999] = np.nan
            data.dropna(subset=targets, inplace=True)
            print(f'Dropped {initial_len - len(data)} NaN records for Level {vert_level}')

            features = data[feature_cols]
            y = data[targets]

            x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

            xgb_model = xgb.XGBRegressor(n_estimators=250, random_state=42, n_jobs=-1,
                                         objective='reg:squarederror', learning_rate=0.1,
                                         max_depth=5, subsample=0.8, colsample_bytree=0.8)
            model = MultiOutputRegressor(xgb_model, n_jobs=-1)
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
            print(f"Metrics for combined XGB model at level {vert_level}: {metrics}")

    return all_metrics


if __name__ == '__main__':


    home_ = os.path.expanduser('~')
    root_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'swapstress', 'training')

    f_ = os.path.join(root_, 'training_data.parquet')
    metrics_dir_ = os.path.join(root_, 'metrics')

    if not os.path.exists(metrics_dir_):
        os.makedirs(metrics_dir_)

    # Combined mode
    print("=" * 50)
    print("RUNNING RANDOM FOREST in combined mode")
    all_metrics_ = run_rf_training(f_, mode='combined',  levels=(2, 3, 5, 6))

    timestamp_ = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_json_ = os.path.join(metrics_dir_, f'RandomForest_combined_{timestamp_}.json')

    with open(metrics_json_, 'w') as f:
        json.dump(all_metrics_, f, indent=4)

    print(f'Wrote RandomForest metrics to {metrics_json_}')

    # XGBoost
    print("=" * 50)
    print("RUNNING XGBOOST in combined mode")
    all_metrics_xgb_ = run_xgb_training(f_, mode='combined', levels=(2, 3, 5, 6))

    timestamp_xgb_ = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_json_xgb_ = os.path.join(metrics_dir_, f'XGBoost_combined_{timestamp_xgb_}.json')

    with open(metrics_json_xgb_, 'w') as f:
        json.dump(all_metrics_xgb_, f, indent=4)

    print(f'Wrote XGBoost metrics to {metrics_json_xgb_}')


# ========================= EOF ====================================================================
