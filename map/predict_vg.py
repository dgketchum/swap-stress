import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

DROP_FEATURES = ['MGRS_TILE', 'lat', 'lon']


def train_random_forest_vg(f):
    df = pd.read_parquet(f)
    vg_params = ['theta_r', 'theta_s', 'log10_alpha', 'log10_n', 'log10_Ks']
    rosetta_cols = [c for c in df.columns if any(p in c for p in vg_params)]

    for vert_level in range(1,8):
        for param in vg_params:
            target = f'US_R3H3_L{vert_level}_VG_{param}'

            assert target in df.columns

            feature_cols = [c for c in df.columns if c not in rosetta_cols]

            features = df[feature_cols]

            all_metrics = {}

            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            metrics = {
                'r2_score': r2_score(y_test, y_pred),
                'mean_squared_error': mean_squared_error(y_test, y_pred),
                'mean_absolute_error': mean_absolute_error(y_test, y_pred)
            }
            all_metrics[target] = metrics

    return all_metrics


if __name__ == '__main__':
    f = '/home/dgketchum/data/IrrigationGIS/soils/swapstress/training/training_data.parquet'
    all_metrics_ = train_random_forest_vg(f)
# ========================= EOF ====================================================================