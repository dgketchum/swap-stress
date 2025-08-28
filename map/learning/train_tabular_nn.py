import json
import os.path
from datetime import datetime

import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import rtdl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, random_split

from map.learning.dataset import TabularDataset, TabularDatasetVanilla

from map.learning.tabular_nn import VanillaMLP, MLPWithEmbeddings, TabularLightningModule
from map.learning import VG_PARAMS, DEVICE, DROP_FEATURES

torch.set_float32_matmul_precision('medium')
EPOCHS = 200
BATCH_SIZE = 32


def prepare_data(df, target_cols, feature_cols, cat_cols, mappings=None, use_one_hot=False, unscale_predictions=False):
    if isinstance(target_cols, str):
        target_cols = [target_cols]

    data = df[target_cols + feature_cols].copy()
    for col in target_cols:
        data[data[col] <= -9999] = np.nan
    data.dropna(subset=target_cols, inplace=True)

    for col in feature_cols:
        if col in cat_cols:
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            data[col] = data[col].fillna(data[col].mean())

    y = data[target_cols].values
    features_df = data[feature_cols]

    if not use_one_hot:
        for col in cat_cols:
            int_map = {int(k): int(v) for k, v in mappings[col].items()}
            features_df.loc[:, col] = features_df[col].map(int_map)

    num_cols = [c for c in features_df.columns if c not in cat_cols]

    scaler = StandardScaler()
    unscaled_vals = features_df[num_cols].copy().values
    features_df.loc[:, num_cols] = scaler.fit_transform(unscaled_vals)

    target_stats = {col: {'mean': data[col].mean(), 'std': data[col].std()} for col in target_cols}

    if use_one_hot:
        features_df = pd.get_dummies(features_df, columns=cat_cols, dummy_na=False, dtype=int)
        x_train, x_test, y_train, y_test = train_test_split(features_df.values, y, test_size=0.2, random_state=42)
        train_dataset = TabularDatasetVanilla(x_train, y_train)
        test_dataset = TabularDatasetVanilla(x_test, y_test)
        return train_dataset, test_dataset, x_train.shape[1], None, target_stats
    else:
        cat_cardinalities = [len(mappings[col]) for col in cat_cols]
        x_train, x_test, y_train, y_test = train_test_split(features_df, y, test_size=0.2, random_state=42)
        train_dataset = TabularDataset(x_train[num_cols].values, x_train[cat_cols].values, y_train)
        test_dataset = TabularDataset(x_test[num_cols].values, x_test[cat_cols].values, y_test)
        return train_dataset, test_dataset, len(num_cols), cat_cardinalities, target_stats


def run_training(f, model_type, mappings_json, checkpoint_dir, metrics_dir, mode='single', levels=None,
                 unscale_predictions=False):
    if levels is None:
        levels = [2]

    df = pd.read_parquet(f)
    with open(mappings_json, 'r') as fj:
        mappings = json.load(fj)

    cat_cols = list(mappings.keys())
    rosetta_cols = [c for c in df.columns if any(p in c for p in VG_PARAMS)]
    feature_cols = [c for c in df.columns if c not in rosetta_cols and c not in DROP_FEATURES]
    all_metrics = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if mode == 'single':
        all_metrics = {p: {vl: None for vl in range(1, 8)} for p in VG_PARAMS}
        for param in VG_PARAMS:
            for vert_level in levels:
                target = f'US_R3H3_L{vert_level}_VG_{param}'
                if target not in df.columns:
                    continue

                print(f"\n--- Training {model_type} for {target} ---")

                if model_type == 'MLP':
                    train_ds, test_ds, n_features, _, target_stats = prepare_data(
                        df, target, feature_cols, cat_cols, use_one_hot=True, unscale_predictions=unscale_predictions)
                    model = VanillaMLP(n_features=n_features, n_outputs=1)
                else:
                    train_ds, test_ds, n_num, cat_cards, target_stats = prepare_data(
                        df, target, feature_cols, cat_cols, mappings=mappings, use_one_hot=False,
                        unscale_predictions=unscale_predictions)
                    if model_type == 'MLPEmbeddings':
                        model = MLPWithEmbeddings(n_num_features=n_num, cat_cardinalities=cat_cards, n_outputs=1)
                    elif model_type == 'FTTransformer':
                        model = rtdl.FTTransformer.make_baseline(
                            n_num_features=n_num, cat_cardinalities=cat_cards, d_token=256,
                            ffn_d_hidden=32, residual_dropout=0.0,
                            n_blocks=3, attention_dropout=0.2, ffn_dropout=0.2, d_out=1,
                        )

                train_size = int(0.8 * len(train_ds))
                val_size = len(train_ds) - train_size
                train_ds, val_ds = random_split(train_ds, [train_size, val_size])

                train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
                val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4)
                test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=4)

                pl_module = TabularLightningModule(model, n_outputs=1)

                checkpoint_callback = ModelCheckpoint(
                    dirpath=f'{checkpoint_dir}/{target}',
                    filename=f'{model_type}-{timestamp}-{{epoch:02d}}-{{val_r2:.2f}}',
                    save_top_k=1, verbose=True, monitor='val_r2', mode='max'
                )

                trainer = pl.Trainer(max_epochs=EPOCHS, accelerator=DEVICE, devices=1, callbacks=[checkpoint_callback])
                trainer.fit(pl_module, train_loader, val_loader)
                trainer.test(pl_module, test_loader)

                y_pred = np.concatenate(pl_module.test_preds).flatten()
                y_test = np.concatenate(pl_module.test_targets).flatten()

                metrics = {
                    'r2': r2_score(y_test, y_pred),
                    'rmse': root_mean_squared_error(y_test, y_pred).item(),
                    'mean_val': list(target_stats.values())[0]['mean'].item(),
                    'std_val': list(target_stats.values())[0]['std'].item(),
                }
                all_metrics[param][vert_level] = metrics
                print(f"Metrics for {target}: R2={{metrics['r2']:.4f}}, RMSE={{metrics['rmse']:.4f}}")

    elif mode == 'combined':
        for vert_level in levels:
            targets = [f'US_R3H3_L{vert_level}_VG_{p}' for p in VG_PARAMS]
            n_outputs = len(targets)
            target_name_for_path = f'L{vert_level}_VG_combined'

            print(f"\n--- Training combined {model_type} for all VG params ---")

            if model_type == 'MLP':
                train_ds, test_ds, n_features, _, target_stats = prepare_data(
                    df, targets, feature_cols, cat_cols, use_one_hot=True, unscale_predictions=unscale_predictions)
                model = VanillaMLP(n_features=n_features, n_outputs=n_outputs)
            else:
                train_ds, test_ds, n_num, cat_cards, target_stats = prepare_data(
                    df, targets, feature_cols, cat_cols, mappings=mappings, use_one_hot=False,
                    unscale_predictions=unscale_predictions)
                if model_type == 'MLPEmbeddings':
                    model = MLPWithEmbeddings(n_num_features=n_num, cat_cardinalities=cat_cards, n_outputs=n_outputs)
                elif model_type == 'FTTransformer':
                    model = rtdl.FTTransformer.make_baseline(
                        n_num_features=n_num, cat_cardinalities=cat_cards, d_token=256,
                        ffn_d_hidden=32, residual_dropout=0.0,
                        n_blocks=3, attention_dropout=0.2, ffn_dropout=0.2, d_out=n_outputs,
                    )

            train_size = int(0.8 * len(train_ds))
            val_size = len(train_ds) - train_size
            train_ds, val_ds = random_split(train_ds, [train_size, val_size])

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=4)

            pl_module = TabularLightningModule(model, n_outputs=n_outputs)

            checkpoint_callback = ModelCheckpoint(
                dirpath=f'{checkpoint_dir}/{target_name_for_path}',
                filename=f'{model_type}-{timestamp}-{{epoch:02d}}-{{val_r2:.2f}}',
                save_top_k=1, verbose=True, monitor='val_r2', mode='max'
            )

            trainer = pl.Trainer(max_epochs=EPOCHS, accelerator=DEVICE, devices=1, callbacks=[checkpoint_callback])
            trainer.fit(pl_module, train_loader, val_loader)
            trainer.test(pl_module, test_loader)

            y_pred = np.concatenate(pl_module.test_preds)
            y_test = np.concatenate(pl_module.test_targets)

            metrics = {}
            for i, target in enumerate(targets):
                param_name = VG_PARAMS[i]
                r2 = r2_score(y_test[:, i], y_pred[:, i])
                rmse = root_mean_squared_error(y_test[:, i], y_pred[:, i])
                metrics[param_name] = {
                    'r2': r2,
                    'rmse': rmse.item(),
                    'mean_val': target_stats[target]['mean'].item(),
                    'std_val': target_stats[target]['std'].item(),
                }
            all_metrics[target_name_for_path] = metrics
            print(f"Metrics for combined model: {metrics}")

        metrics_json = os.path.join(metrics_dir, f'{model_type}_{mode}_{timestamp}.json')
        with open(metrics_json, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        print(f'Wrote {model_type} ({mode}) metrics to {metrics_json}')


if __name__ == '__main__':

    home = os.path.expanduser('~')
    root = os.path.join(home, 'data', 'IrrigationGIS', 'soils', 'swapstress', 'training')

    f = os.path.join(root, 'training_data.parquet')
    mappings_json = os.path.join(root, 'categorical_mappings.json')
    checkpoint_dir_ = os.path.join(root, 'checkpoints')
    metrics_ = os.path.join(root, 'metrics')

    for model_name in ['MLP', 'MLPEmbeddings', 'FTTransformer']:
        # print("\n\n" + "=" * 50)
        # print(f"RUNNING {model_name.upper()} in single-target mode")
        # run_training(f, model_name, mappings_json, checkpoint_dir_, metrics_, mode='single')

        print("\n\n" + "=" * 50)
        print(f"RUNNING {model_name.upper()} in combined-target mode")
        run_training(f, model_name, mappings_json, checkpoint_dir_, metrics_, mode='combined',
                     levels=(2, 3, 5, 6), unscale_predictions=False)

# ========================= EOF ====================================================================
