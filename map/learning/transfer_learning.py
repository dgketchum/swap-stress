import json
import os
import re
from glob import glob

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import rtdl
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, random_split

from map.learning import DEVICE, DROP_FEATURES, VG_PARAMS
from map.learning.dataset import TabularDataset, TabularDatasetVanilla
from map.learning.tabular_nn import TabularLightningModule, VanillaMLP, MLPWithEmbeddings

torch.set_float32_matmul_precision('medium')


def find_best_model_checkpoint(checkpoint_dir, target, model_type):
    """
    Finds the best model checkpoint file based on validation R2 in the filename.
    """
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


def prepare_finetune_data(empirical_df, training_df, feature_cols, cat_cols, num_cols, target_cols, mappings,
                          use_one_hot=False):
    """
    Prepares the empirical dataset for fine-tuning, using statistics (mean, std, mode)
    from the original training data for consistent feature scaling and imputation.
    """
    data = empirical_df[target_cols + feature_cols].copy()
    for col in target_cols:
        data[data[col] <= -9999] = np.nan
    data.dropna(subset=target_cols, inplace=True)

    for col in feature_cols:
        if col in cat_cols:
            fill_val = training_df[col].mode()[0]
            data[col] = data[col].fillna(fill_val)
        else:
            fill_val = training_df[col].mean()
            data[col] = data[col].fillna(fill_val)

    y = data[target_cols].values
    features_df = data[feature_cols]

    if not use_one_hot:
        for col in cat_cols:
            int_map = {int(k): int(v) for k, v in mappings[col].items()}
            features_df.loc[:, col] = features_df[col].map(int_map)

    scaler = StandardScaler().fit(training_df[num_cols])
    features_df.loc[:, num_cols] = scaler.transform(features_df[num_cols])

    if use_one_hot:
        # Ensure one-hot columns match original training data
        train_onehot_df = pd.get_dummies(training_df[feature_cols], columns=cat_cols, dummy_na=False, dtype=int)
        features_df = pd.get_dummies(features_df, columns=cat_cols, dummy_na=False, dtype=int)
        missing_cols = set(train_onehot_df.columns) - set(features_df.columns)
        for c in missing_cols:
            features_df[c] = 0
        features_df = features_df[train_onehot_df.columns]
        dataset = TabularDatasetVanilla(features_df.values.astype(np.float32), y)
    else:
        dataset = TabularDataset(features_df[num_cols].values.astype(np.float32),
                                 features_df[cat_cols].values,
                                 y)

    return dataset, data


class FineTuner:
    def __init__(self, base_model, train_dataset, val_dataset, config=None):
        self.model = base_model
        self.train_ds = train_dataset
        self.val_ds = val_dataset
        self.config = config or {}
        self._setup_model_for_finetuning()

    def _setup_model_for_finetuning(self):
        """
        Freezes layers of the model based on its type (MLP or FTTransformer).
        """

        inner_model = self.model.model
        num_blocks_to_freeze = self.config.get('freeze_layers', 0)

        if num_blocks_to_freeze > 0:
            if isinstance(inner_model, (VanillaMLP, MLPWithEmbeddings)):
                print(f"Freezing first {num_blocks_to_freeze} blocks of MLP-style model.")
                children = list(inner_model.layers.children())

                block_count = 0
                for layer in children:
                    if isinstance(layer, torch.nn.Linear):
                        block_count += 1

                    if block_count <= num_blocks_to_freeze:
                        for param in layer.parameters():
                            param.requires_grad = False
                    else:
                        break

            elif isinstance(inner_model, rtdl.FTTransformer):
                print(f"Freezing parts of the FTTransformer.")
                for param in inner_model.feature_tokenizer.parameters():
                    param.requires_grad = False

                num_transformer_blocks = len(inner_model.transformer.blocks)
                blocks_to_freeze_count = min(num_blocks_to_freeze, num_transformer_blocks)

                if blocks_to_freeze_count > 0:
                    for i in range(blocks_to_freeze_count):
                        for param in inner_model.transformer.blocks[i].parameters():
                            param.requires_grad = False
                    print(
                        f"Froze Feature Tokenizer and the first {blocks_to_freeze_count} of {num_transformer_blocks} Transformer blocks.")
                else:
                    print("Froze Feature Tokenizer. All Transformer blocks and the head remain trainable.")
            else:
                print(
                    f"Warning: Model type {type(inner_model)} not recognized for freezing. All layers will be trainable.")

        self.model.learning_rate = self.config.get('lr', 1e-7)

    def run(self, checkpoint_dir, model_name):
        train_loader = DataLoader(self.train_ds, batch_size=self.config.get('batch_size', 16), shuffle=True)
        val_loader = DataLoader(self.val_ds, batch_size=self.config.get('batch_size', 16))

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f'{model_name}-finetuned-{{epoch:02d}}-{{val_r2:.2f}}',
            save_top_k=1, monitor='val_r2', mode='max', verbose=True)

        trainer = pl.Trainer(
            max_epochs=self.config.get('epochs', 50),
            accelerator=DEVICE,
            callbacks=[checkpoint_callback])

        print(f"\n--- Starting fine-tuning for {model_name} ---")
        trainer.fit(self.model, train_loader, val_loader)
        print("Fine-tuning complete.")
        return trainer.checkpoint_callback.best_model_path


def run_finetuning_workflow(rosetta_training_data, empirical_finetune_data, mappings_json, checkpoint_dir, levels):
    training_df_ = pd.read_parquet(rosetta_training_data)
    empirical_df_ = pd.read_parquet(empirical_finetune_data)
    finetuning_split_path = os.path.join(os.path.dirname(empirical_finetune_data), 'finetuning_split_info.json')

    with open(mappings_json, 'r') as f:
        mappings_ = json.load(f)

    model_types_ = ['MLP', 'MLPEmbeddings', 'FTTransformer']

    cat_cols_ = list(mappings_.keys())
    rosetta_cols_ = [c for c in training_df_.columns if any(p in c for p in VG_PARAMS)]
    feature_cols_ = [c for c in training_df_.columns if c not in rosetta_cols_ and c not in DROP_FEATURES]
    num_cols_ = [c for c in feature_cols_ if c not in cat_cols_]
    cat_cardinalities_ = [len(mappings_[col]) for col in cat_cols_]

    for level_to_tune_ in levels:
        target_name_ = f'L{level_to_tune_}_VG_combined'
        targets_ = [f'US_R3H3_L{level_to_tune_}_VG_{p}' for p in VG_PARAMS]

        for model_type_ in model_types_:

            print(f"\n--- Running fine-tuning for model type: {model_type_} ---")
            best_ckpt_, _ = find_best_model_checkpoint(checkpoint_dir, target_name_, model_type_)

            if best_ckpt_:
                print(f"Preparing data for fine-tuning {target_name_}...")
                use_one_hot_ = model_type_ == 'MLP'
                finetune_dataset, finetune_df = prepare_finetune_data(
                    empirical_df_,
                    training_df_,
                    feature_cols_,
                    cat_cols_,
                    num_cols_,
                    targets_,
                    mappings_,
                    use_one_hot=use_one_hot_
                )

                n_outputs_ = len(targets_)
                if model_type_ == 'MLP':
                    n_features_ = len(
                        pd.get_dummies(training_df_[feature_cols_], columns=cat_cols_, dummy_na=False,
                                       dtype=int).columns)
                    base_model_ = VanillaMLP(n_features=n_features_, n_outputs=n_outputs_)
                elif model_type_ == 'MLPEmbeddings':
                    base_model_ = MLPWithEmbeddings(n_num_features=len(num_cols_), cat_cardinalities=cat_cardinalities_,
                                                    n_outputs=n_outputs_)
                else:
                    base_model_ = rtdl.FTTransformer.make_baseline(
                        n_num_features=len(num_cols_), cat_cardinalities=cat_cardinalities_, d_token=256,
                        ffn_d_hidden=32, residual_dropout=0.0,
                        n_blocks=3, attention_dropout=0.2, ffn_dropout=0.2, d_out=n_outputs_)

                pl_module_ = TabularLightningModule.load_from_checkpoint(best_ckpt_, model=base_model_,
                                                                         n_outputs=n_outputs_)

                finetune_config_ = {'lr': 5e-6, 'epochs': 30, 'freeze_layers': 2, 'batch_size': 8}
                train_ds_, val_ds_ = random_split(finetune_dataset, [0.8, 0.2],
                                                  generator=torch.Generator().manual_seed(42))

                train_indices = train_ds_.indices
                val_indices = val_ds_.indices

                train_info = finetune_df.iloc[train_indices].to_dict(orient='index')
                val_info = finetune_df.iloc[val_indices].to_dict(orient='index')

                # TODO: read this data into the test function
                split_data = {'train': train_info, 'validation': val_info}
                with open(finetuning_split_path, 'w') as f:
                    json.dump(split_data, f, indent=2)
                print(f"Saved fine-tuning split info to {finetuning_split_path}")

                tuner_ = FineTuner(pl_module_, train_ds_, val_ds_, config=finetune_config_)
                finetuned_ckpt_dir_ = os.path.join(checkpoint_dir, 'fine_tuned')
                os.makedirs(finetuned_ckpt_dir_, exist_ok=True)
                finetuned_model_name = f'{target_name_}_{model_type_}'
                tuner_.run(checkpoint_dir=finetuned_ckpt_dir_, model_name=finetuned_model_name)
            else:
                print(f"No checkpoint found for {target_name_} with model type {model_type_}")


if __name__ == '__main__':
    home_ = os.path.expanduser('~')
    root_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'swapstress')
    training_data_root_ = os.path.join(root_, 'training')
    # finetune_data_root_ = os.path.join(root_, 'finetune')

    rosetta_training_data_ = os.path.join(training_data_root_, 'training_data.parquet')
    empirical_finetune_data_ = os.path.join(training_data_root_, 'mt_training_data.parquet')
    mappings_json_ = os.path.join(training_data_root_, 'categorical_mappings.json')
    checkpoint_dir_ = os.path.join(training_data_root_, 'checkpoints', 'combined_params')

    run_finetuning_workflow(
        rosetta_training_data=rosetta_training_data_,
        empirical_finetune_data=empirical_finetune_data_,
        mappings_json=mappings_json_,
        checkpoint_dir=checkpoint_dir_,
        levels=(2, 3, 5, 6))

# ========================= EOF ====================================================================
