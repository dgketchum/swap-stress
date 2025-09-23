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
from map.learning.tabular_nn.dataset import TabularDataset, TabularDatasetVanilla
from map.learning.tabular_nn.tabular_nn import MLPWithEmbeddings, TabularLightningModule, VanillaMLP

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


def run_inference_on_station_table(station_table_pqt, gshp_training_pqt, gshp_mappings_json,
                                   gshp_checkpoint_dir, output_pqt, features_path):
    station_df = pd.read_parquet(station_table_pqt)
    train_df = pd.read_parquet(gshp_training_pqt)
    with open(gshp_mappings_json, 'r') as f:
        mappings = json.load(f)

    targets = ['theta_r', 'theta_s', 'alpha', 'n']

    feats_df = pd.read_csv(features_path)
    col = 'features' if 'features' in feats_df.columns else feats_df.columns[0]
    training_features = feats_df[col].dropna().astype(str).tolist()

    # Mirror training feature handling
    rosetta_cols = [c for c in train_df.columns if any(p in c for p in VG_PARAMS) or c in targets]
    feature_cols = [c for c in training_features if c in train_df.columns and c not in rosetta_cols and c not in DROP_FEATURES]
    if 'SWCC_classes' in feature_cols:
        feature_cols.remove('SWCC_classes')
    if 'data_flag' in feature_cols:
        feature_cols.remove('data_flag')
    missing_train = [c for c in feature_cols if c not in train_df.columns]
    if missing_train:
        raise ValueError(f'Missing required training features: {missing_train}')
    missing_station = [c for c in feature_cols if c not in station_df.columns]
    if missing_station:
        raise ValueError(f'Missing required station features: {missing_station}')

    cat_cols = [c for c in mappings.keys() if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    train_num = train_df[num_cols].copy()
    train_num = train_num.fillna(train_num.mean())
    scaler = StandardScaler().fit(train_num)
    cat_cardinalities = [len(mappings[col]) for col in cat_cols]

    inf_feats = station_df[feature_cols].copy()
    for col in feature_cols:
        if col in cat_cols:
            inf_feats[col] = inf_feats[col].fillna(train_df[col].mode()[0])
        else:
            inf_feats[col] = inf_feats[col].fillna(train_num[col].mean())

    onehot_train = pd.get_dummies(train_df[feature_cols], columns=cat_cols, dummy_na=False, dtype=int)

    ckpt_root = os.path.join(gshp_checkpoint_dir, 'GSHP_VG_combined')
    all_ckpts = glob(os.path.join(ckpt_root, '*.ckpt'))
    if not all_ckpts:
        raise FileNotFoundError(f'No checkpoints found under {ckpt_root}')
    r2_pattern = re.compile(r"val_r2=([-]?\d+\.\d+)")
    best_ckpt, best_type, best_r2 = None, None, -float('inf')
    for ckpt in all_ckpts:
        m = r2_pattern.search(os.path.basename(ckpt))
        if not m:
            continue
        r2 = float(m.group(1))
        if r2 > best_r2:
            best_r2 = r2
            best_ckpt = ckpt
            best_type = 'MLPEmbeddings' if 'MLPEmbeddings-' in ckpt else (
                'FTTransformer' if 'FTTransformer-' in ckpt else 'MLP')

    if best_ckpt is None:
        raise ValueError('Could not determine best checkpoint for GSHP_VG_combined')

    n_outputs = len(targets)
    # Sanity check: ensure numeric feature count matches checkpoint expectations for transformer/embeddings
    if best_type != 'MLP':
        try:
            ckpt = torch.load(best_ckpt, map_location='cpu')
            w = ckpt['state_dict'].get('model.feature_tokenizer.num_tokenizer.weight')
            if w is not None:
                expected_num = int(w.shape[0])
                if expected_num != len(num_cols):
                    raise ValueError(
                        f'Numeric feature count mismatch: checkpoint expects {expected_num}, got {len(num_cols)}. '
                        f'Ensure GSHP training table matches current_features.csv and retrain.'
                    )
        except Exception as e:
            # likely error: checkpoint format unexpected
            raise e
    if best_type == 'MLP':
        inf_onehot = pd.get_dummies(inf_feats, columns=cat_cols, dummy_na=False, dtype=int)
        missing = set(onehot_train.columns) - set(inf_onehot.columns)
        for c in missing:
            inf_onehot[c] = 0
        inf_onehot = inf_onehot[onehot_train.columns]
        x_inf = inf_onehot.values
        dataset = TabularDatasetVanilla(x_inf, np.zeros((x_inf.shape[0], n_outputs)))
        model = VanillaMLP(n_features=x_inf.shape[1], n_outputs=n_outputs, num_hidden_layers=2)
    else:
        for col in cat_cols:
            int_map = {int(k): int(v) for k, v in mappings[col].items()}
            inf_feats[col] = inf_feats[col].map(int_map)
        x_num = scaler.transform(inf_feats[num_cols])
        x_cat = inf_feats[cat_cols].values
        dataset = TabularDataset(x_num, x_cat, np.zeros((x_num.shape[0], n_outputs)))
        if best_type == 'MLPEmbeddings':
            model = MLPWithEmbeddings(n_num_features=len(num_cols), cat_cardinalities=cat_cardinalities,
                                      n_outputs=n_outputs, num_hidden_layers=2)
        else:
            model = rtdl.FTTransformer.make_baseline(n_num_features=len(num_cols), cat_cardinalities=cat_cardinalities,
                                                     d_token=256, ffn_d_hidden=32, residual_dropout=0.0, n_blocks=3,
                                                     attention_dropout=0.2, ffn_dropout=0.2, d_out=n_outputs)

    pl_module = TabularLightningModule.load_from_checkpoint(best_ckpt, model=model)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    trainer = pl.Trainer(accelerator=DEVICE, devices=1)
    preds = trainer.predict(pl_module, loader)
    y = np.concatenate(preds)

    out = station_df[['station', 'profile_id', 'depth',
                      'rosetta_level']].copy() if 'station' in station_df.columns else station_df.index.to_frame(
        index=False)
    out = out.reset_index(drop=True)
    out['theta_r'] = y[:, 0]
    out['theta_s'] = y[:, 1]
    out['alpha'] = 10 ** y[:, 2]
    out['n'] = 10 ** y[:, 3]
    out.to_parquet(output_pqt)
    print(f'Saved NN station predictions to {output_pqt}')


if __name__ == '__main__':
    home = os.path.expanduser('~')
    base = os.path.join(home, 'data', 'IrrigationGIS', 'soils', 'swapstress', 'training')
    station_table_ = os.path.join(base, 'stations_training_table_250m.parquet')
    gshp_train_ = os.path.join(base, 'gshp_training_data_emb_250m.parquet')
    gshp_maps_ = os.path.join(base, 'gshp_categorical_mappings_250m.json')
    ckpts_ = os.path.join(base, 'checkpoints_gshp')
    out_pq_ = os.path.join(base, 'predictions', 'stations_predictions_nn.parquet')
    feats_csv_ = os.path.join(base, 'current_features.csv')
    run_inference_on_station_table(station_table_, gshp_train_, gshp_maps_, ckpts_, out_pq_, features_path=feats_csv_)
# ========================= EOF ====================================================================
