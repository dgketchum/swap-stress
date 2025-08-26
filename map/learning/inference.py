import json
import os
import re
from glob import glob

import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import rtdl
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from map.learning import DEVICE, DROP_FEATURES, VG_PARAMS
from map.learning.dataset import TabularDataset, TabularDatasetVanilla
from map.learning.tabular_nn import MLPWithEmbeddings, TabularLightningModule, VanillaMLP

torch.set_float32_matmul_precision('medium')


def find_best_model_checkpoint(checkpoint_dir, target, model_type, use_finetuned=False):
    """
    Finds the best model checkpoint file based on validation R2 in the filename.
    """
    if use_finetuned:
        search_path = os.path.join(checkpoint_dir, f'{target}_{model_type}-finetuned-*.ckpt')
    else:
        search_path = os.path.join(checkpoint_dir, target, f'{model_type}-*.ckpt')

    checkpoints = glob(search_path)
    if not checkpoints:
        return None, -float('inf')

    r2_pattern = re.compile(r"val_r2=([-]?\d+\.\d+)")
    best_ckpt, max_r2 = None, -float('inf')

    for ckpt in checkpoints:
        match = r2_pattern.search(os.path.basename(ckpt))

        if match:
            r2 = float(match.group(1))
            if r2 > max_r2:
                max_r2 = r2
                best_ckpt = ckpt

    if not best_ckpt:
        print(f"Warning: No checkpoints found for {target} with model {model_type}")
        return None, -float('inf')

    return best_ckpt, max_r2


def run_inference_on_empirical_data(station_data_pqt, training_data_pqt, rosetta_mappings_json,
                                    rosetta_checkpoint_dir, output_pqt, unscale_predictions=False,
                                    use_finetuned=False):
    """
    Uses models trained on Rosetta to predict VG parameters for empirical sites.
    """
    station_data = pd.read_parquet(station_data_pqt)
    training_df = pd.read_parquet(training_data_pqt)

    with open(rosetta_mappings_json, 'r') as f:
        mappings = json.load(f)

    cat_cols = list(mappings.keys())
    rosetta_cols = [c for c in training_df.columns if any(p in c for p in VG_PARAMS)]
    feature_cols = [c for c in training_df.columns if c not in rosetta_cols and c not in DROP_FEATURES]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    cat_cardinalities = [len(mappings[col]) for col in cat_cols]

    for col in rosetta_cols:
        training_df[training_df[col] <= -9999] = np.nan

    scaler = StandardScaler().fit(training_df[num_cols])

    inference_features = station_data[feature_cols].copy()
    for col in feature_cols:
        if col in cat_cols:
            inference_features[col] = inference_features[col].fillna(training_df[col].mode()[0])
        else:
            inference_features[col] = inference_features[col].fillna(training_df[col].mean())

    train_onehot_df = pd.get_dummies(training_df[feature_cols], columns=cat_cols, dummy_na=False, dtype=int)

    checkpoint_dir = os.path.join(rosetta_checkpoint_dir, 'fine_tuned') if use_finetuned else rosetta_checkpoint_dir

    model_types = ['MLP', 'MLPEmbeddings', 'FTTransformer']
    print("Searching for the best overall model type...")
    all_checkpoints = glob(os.path.join(checkpoint_dir, '**', '*.ckpt'), recursive=True)
    r2_pattern = re.compile(r"val_r2=([-]?\d+\.\d+)")
    best_r2 = -float('inf')
    overall_best_model_type = None

    for ckpt in all_checkpoints:
        basename = os.path.basename(ckpt)
        match = r2_pattern.search(basename)
        if match:
            r2 = float(match.group(1))
            if r2 > best_r2:
                best_r2 = r2
                for mt in model_types:
                    if f'{mt}-' in basename:
                        overall_best_model_type = mt
                        break

    if not overall_best_model_type:
        raise ValueError("Could not determine the best model type. No valid checkpoints found.")

    print(f"Found best overall model type: {overall_best_model_type} (R2={best_r2:.4f})")

    all_ckpts = [os.path.basename(x) for x in glob(os.path.join(checkpoint_dir, '*'))]
    levels = sorted([int(x) for x in set(re.findall(r'L(\d+)_VG_combined', ' '.join(all_ckpts)))])

    predictions_df = station_data[['latitude', 'longitude']].copy()
    n_outputs = len(VG_PARAMS)

    for level in levels:
        target_name = f'L{level}_VG_combined'
        best_model_type = overall_best_model_type
        best_ckpt, max_r2 = find_best_model_checkpoint(checkpoint_dir, target_name, best_model_type,
                                                       use_finetuned=use_finetuned)

        if best_ckpt:
            print(f"Using {best_model_type} for {target_name} (R2={max_r2:.4f})")

            level_targets = [f'US_R3H3_L{level}_VG_{p}' for p in VG_PARAMS]
            target_stats = {t: {'mean': training_df[t].mean(), 'std': training_df[t].std()} for t in level_targets}

            if best_model_type == 'MLP':
                inference_onehot = pd.get_dummies(inference_features, columns=cat_cols,
                                                  dummy_na=False, dtype=int)
                missing_cols = set(train_onehot_df.columns) - set(inference_onehot.columns)
                for c in missing_cols:
                    inference_onehot[c] = 0
                inference_onehot = inference_onehot[train_onehot_df.columns]
                x_inference = inference_onehot.values
                inf_dataset = TabularDatasetVanilla(x_inference, np.zeros((x_inference.shape[0], n_outputs)))
                model = VanillaMLP(n_features=x_inference.shape[1], n_outputs=n_outputs)

            else:
                inf_feats_emb = inference_features.copy()
                for col in cat_cols:
                    int_map = {int(k): int(v) for k, v in mappings[col].items()}
                    inf_feats_emb[col] = inf_feats_emb[col].map(int_map)
                x_inf_num = scaler.transform(inf_feats_emb[num_cols])
                x_inf_cat = inf_feats_emb[cat_cols].values
                inf_dataset = TabularDataset(x_inf_num, x_inf_cat, np.zeros((x_inf_num.shape[0], n_outputs)))

                if best_model_type == 'MLPEmbeddings':
                    model = MLPWithEmbeddings(n_num_features=len(num_cols), cat_cardinalities=cat_cardinalities,
                                              n_outputs=n_outputs)
                else:
                    model = rtdl.FTTransformer.make_baseline(n_num_features=len(num_cols),
                                                             cat_cardinalities=cat_cardinalities,
                                                             d_token=256, ffn_d_hidden=32,
                                                             residual_dropout=0.0, n_blocks=3,
                                                             attention_dropout=0.2, ffn_dropout=0.2,
                                                             d_out=n_outputs)

            pl_module = TabularLightningModule.load_from_checkpoint(best_ckpt, model=model)
            inf_loader = DataLoader(inf_dataset, batch_size=32, shuffle=False)
            trainer = pl.Trainer(accelerator=DEVICE, devices=1)
            preds = trainer.predict(pl_module, inf_loader)
            scaled_preds = np.concatenate(preds)

            if unscale_predictions:
                unscaled_preds = np.zeros_like(scaled_preds)
                for i, target in enumerate(level_targets):
                    mean = target_stats[target]['mean']
                    std = target_stats[target]['std']
                    unscaled_preds[:, i] = scaled_preds[:, i] * std + mean
                for i, target in enumerate(level_targets):
                    predictions_df[target] = unscaled_preds[:, i]
            else:
                for i, target in enumerate(level_targets):
                    predictions_df[target] = scaled_preds[:, i]

    predictions_df.to_parquet(output_pqt)

    print(f"Inference complete. Predictions saved to {output_pqt}")


if __name__ == '__main__':
    """"""
    home = os.path.expanduser('~')

    root = os.path.join(home, 'data', 'IrrigationGIS', 'soils', 'swapstress')
    training_ = os.path.join(root, 'training')
    inference_ = os.path.join(root, 'inference')

    mesonet_training_data_ = os.path.join(training_, 'mt_training_data.parquet')
    training_data_ = os.path.join(training_, 'training_data.parquet')
    mappings_json = os.path.join(training_, 'categorical_mappings.json')
    checkpoint_dir_ = os.path.join(training_, 'checkpoints', 'combined_params')

    mode, finetuned = 'finetuned', True
    # mode, finetuned = 'pretrained', False

    run_inference_on_empirical_data(
        station_data_pqt=mesonet_training_data_,
        training_data_pqt=training_data_,
        rosetta_mappings_json=mappings_json,
        rosetta_checkpoint_dir=checkpoint_dir_,
        output_pqt=os.path.join(inference_, f'{mode}_predictions.parquet'),
        unscale_predictions=False,
        use_finetuned=finetuned
    )
# ========================= EOF ====================================================================
