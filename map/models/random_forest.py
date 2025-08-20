import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error

DROP_FEATURES = ['MGRS_TILE', 'lat', 'lon']


def train_random_forest_vg(f):
    df = pd.read_parquet(f)
    vg_params = ['theta_r', 'theta_s', 'log10_alpha', 'log10_n', 'log10_Ks']
    rosetta_cols = [c for c in df.columns if any(p in c for p in vg_params)]

    all_metrics = {p: {vl: None for vl in range(1, 8)} for p in vg_params}

    for param in vg_params:
        for vert_level in range(1, 8):
            target = f'US_R3H3_L{vert_level}_VG_{param}'

            assert target in df.columns

            feature_cols = [c for c in df.columns if c not in rosetta_cols and c not in DROP_FEATURES]

            data = df[[target] + feature_cols].copy()
            initial_len = len(data)
            data[data[target] <= -9999] = np.nan
            data.dropna(subset=[target], inplace=True)
            print(f'Dropped {initial_len - len(data)} NaN records')

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

    return all_metrics


if __name__ == '__main__':
    f = '/home/dgketchum/data/IrrigationGIS/soils/swapstress/training/training_data.parquet'
    all_metrics_ = train_random_forest_vg(f)
# ========================= EOF ====================================================================
