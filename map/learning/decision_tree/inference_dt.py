import os
import json

import joblib
import pandas as pd
import numpy as np

GSHP_PARAMS = ['theta_r', 'theta_s', 'alpha', 'n']


def run_inference(model_path, features_path, data_path, output_path):
    """
    Run inference on a dataset using a trained scikit-learn model.
    Args:
        model_path (str): Path to the trained model file (e.g., .joblib).
        features_path (str): Path to the JSON file containing the list of feature columns used for training.
        data_path (str): Path to the input data file (e.g., .parquet) for inference.
        output_path (str): Path to save the output predictions (e.g., .parquet).
    """
    print(f'Loading model from {model_path}')
    model = joblib.load(model_path)

    print(f'Loading features from {features_path}')
    training_features = None
    if features_path.endswith('.json'):
        with open(features_path, 'r') as f:
            training_features = json.load(f)
    elif features_path.endswith('.csv'):
        feats_df = pd.read_csv(features_path)
        col = 'features' if 'features' in feats_df.columns else feats_df.columns[0]
        training_features = feats_df[col].dropna().astype(str).tolist()
    else:
        raise ValueError(f'Unsupported features file: {features_path}')

    print(f'Loading data from {data_path}')
    inference_df = pd.read_parquet(data_path)
    inference_df.index = inference_df['station']

    missing_cols = [col for col in training_features if col not in inference_df.columns]
    if missing_cols:
        raise ValueError(f'Input data is missing columns: {missing_cols}')

    extra_cols = [col for col in inference_df.columns if col not in training_features]
    print(f'Input data has extra columns that will be ignored: {extra_cols}')

    inference_features = inference_df[training_features]

    print('Running inference...')
    predictions = model.predict(inference_features)

    pred_df = pd.DataFrame(predictions, columns=GSHP_PARAMS, index=inference_df.index)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    pred_df.to_parquet(output_path)
    print(f'Saved predictions to {output_path}')


if __name__ == '__main__':
    home_ = os.path.expanduser('~')
    root_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'swapstress')
    model_dir_ = os.path.join(root_, 'training', 'models')
    data_dir_ = os.path.join(root_, 'training')
    output_dir_ = os.path.join(root_, 'training', 'predictions')

    model_path_ = os.path.join(model_dir_, 'rf_gshp_model.joblib')
    # Use the unified station training table and the features list produced by station_training_table.py
    features_path_ = os.path.join(data_dir_, 'current_features.csv')
    data_path_ = os.path.join(data_dir_, 'stations_training_table_250m.parquet')

    os.makedirs(output_dir_, exist_ok=True)
    if os.path.exists(data_path_):
        print('\n--- Running inference for stations training table ---')
        output_path_ = os.path.join(output_dir_, 'stations_predictions.parquet')
        run_inference(model_path=model_path_, features_path=features_path_,
                      data_path=data_path_, output_path=output_path_)
    else:
        print(f'Input data not found: {data_path_}')

# ========================= EOF ====================================================================
