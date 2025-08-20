import json
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import rtdl
import torch
from pytorch_lightning.callbacks import TQDMProgressBar
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, random_split

from tabular_nn import TabularDataset, TabularDatasetVanilla, VanillaMLP, MLPWithEmbeddings, TabularLightningModule

from map.models import VG_PARAMS, DEVICE, BATCH_SIZE, EPOCHS, DROP_FEATURES


def prepare_data(df, target_col, feature_cols, cat_cols, mappings):
    data = df[[target_col] + feature_cols].copy()
    data[data[target_col] <= -9999] = np.nan
    data.dropna(subset=[target_col], inplace=True)
    for col in feature_cols:
        if col in cat_cols:
            data[col].fillna(data[col].mode()[0], inplace=True)
        else:
            data[col].fillna(data[col].mean(), inplace=True)
    y = data[target_col].values
    features_df = data[feature_cols]
    num_cols = [c for c in features_df.columns if c not in cat_cols]
    cat_cardinalities = [len(mappings[col]) for col in cat_cols]
    scaler = StandardScaler()
    features_df[num_cols] = scaler.fit_transform(features_df[num_cols])
    X_train, X_test, y_train, y_test = train_test_split(features_df, y, test_size=0.2, random_state=42)
    train_dataset = TabularDataset(X_train[num_cols].values, X_train[cat_cols].values, y_train)
    test_dataset = TabularDataset(X_test[num_cols].values, X_test[cat_cols].values, y_test)
    return train_dataset, test_dataset, len(num_cols), cat_cardinalities, {'mean': y.mean(), 'std': y.std()}


def prepare_data_vanilla(df, target_col, feature_cols, cat_cols):
    data = df[[target_col] + feature_cols].copy()
    data[data[target_col] <= -9999] = np.nan
    data.dropna(subset=[target_col], inplace=True)
    for col in feature_cols:
        if col in cat_cols:
            data[col].fillna(data[col].mode()[0], inplace=True)
        else:
            data[col].fillna(data[col].mean(), inplace=True)
    y = data[target_col].values
    features_df = data[feature_cols]
    num_cols = [c for c in features_df.columns if c not in cat_cols]
    features_df = pd.get_dummies(features_df, columns=cat_cols, dummy_na=False)
    scaler = StandardScaler()
    features_df[num_cols] = scaler.fit_transform(features_df[num_cols])
    X_train, X_test, y_train, y_test = train_test_split(features_df.values, y, test_size=0.2, random_state=42)
    train_dataset = TabularDatasetVanilla(X_train, y_train)
    test_dataset = TabularDatasetVanilla(X_test, y_test)
    return train_dataset, test_dataset, X_train.shape[1], {'mean': y.mean(), 'std': y.std()}


def run_training(f, model_type, mappings_json):
    df = pd.read_parquet(f)
    with open(mappings_json, 'r') as fj:
        mappings = json.load(fj)
    cat_cols = list(mappings.keys())
    rosetta_cols = [c for c in df.columns if any(p in c for p in VG_PARAMS)]
    feature_cols = [c for c in df.columns if c not in rosetta_cols and c not in DROP_FEATURES]
    all_metrics = {p: {vl: None for vl in range(1, 8)} for p in VG_PARAMS}

    for param in VG_PARAMS:
        for vert_level in range(1, 8):
            target = f'US_R3H3_L{vert_level}_VG_{param}'
            if target not in df.columns: continue

            print(f"\n--- Training {model_type} for {target} ---")

            if model_type == 'Vanilla MLP':
                train_ds, test_ds, n_features, target_stats = prepare_data_vanilla(df, target, feature_cols, cat_cols)
                model = VanillaMLP(n_features=n_features)
            else:
                train_ds, test_ds, n_num, cat_cards, target_stats = prepare_data(df, target, feature_cols, cat_cols, mappings)
                if model_type == 'MLP-Embeddings':
                    model = MLPWithEmbeddings(n_num_features=n_num, cat_cardinalities=cat_cards)
                elif model_type == 'FT-Transformer':
                    model = rtdl.FTTransformer.make_baseline(
                        n_num_features=n_num, cat_cardinalities=cat_cards, d_token=192,
                        n_blocks=3, attention_dropout=0.2, ffn_dropout=0.2, d_out=1,
                    )

            train_size = int(0.8 * len(train_ds))
            val_size = len(train_ds) - train_size
            train_ds, val_ds = random_split(train_ds, [train_size, val_size])

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=4)

            pl_module = TabularLightningModule(model)

            progress_bar = TQDMProgressBar(refresh_rate=10)

            trainer = pl.Trainer(
                max_epochs=EPOCHS,
                accelerator=DEVICE,
                devices=1,
                callbacks=[progress_bar],
                logger=False,
                enable_checkpointing=False
            )

            trainer.fit(pl_module, train_loader, val_loader)
            trainer.test(pl_module, test_loader)

            y_pred = np.concatenate(pl_module.test_preds).flatten()
            y_test = np.concatenate(pl_module.test_targets).flatten()

            metrics = {
                'r2': r2_score(y_test, y_pred),
                'rmse': root_mean_squared_error(y_test, y_pred),
                'mean_val': target_stats['mean'],
                'std_val': target_stats['std'],
            }
            all_metrics[param][vert_level] = metrics
            print(f"Metrics for {target}: R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")

    return all_metrics


if __name__ == '__main__':
    f = '/home/dgketchum/data/IrrigationGIS/soils/swapstress/training/training_data.parquet'
    mappings_json = '/home/dgketchum/data/IrrigationGIS/soils/swapstress/training/categorical_mappings.json'

    for model_name in ['Vanilla MLP', 'MLP-Embeddings', 'FT-Transformer']:
        print("\n\n" + "=" * 50)
        print(f"RUNNING {model_name.upper()}")
        print("=" * 50)
        metrics = run_training(f, model_name, mappings_json)
        print(f"\n{model_name} Final Metrics (R2):")
        print(pd.DataFrame(metrics).applymap(lambda x: x['r2'] if x else None))

# ========================= EOF ====================================================================
