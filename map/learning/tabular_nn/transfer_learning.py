import json
import os
import re

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import rtdl
import torch
from sklearn.metrics import r2_score, root_mean_squared_error
from torch.utils.data import DataLoader

from map.learning import DEVICE, DROP_FEATURES, VG_PARAMS
from map.learning.tabular_nn.dataset import prepare_data
from map.learning.tabular_nn.tabular_nn import TabularLightningModule, VanillaMLP, MLPWithEmbeddings

torch.set_float32_matmul_precision('medium')


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

                # Freeze entire MLP blocks: [Linear, ReLU, BatchNorm, Dropout] x N, not the output head
                frozen_linear = 0
                for layer in children:
                    if isinstance(layer, torch.nn.Linear):
                        frozen_linear += 1
                    if frozen_linear <= num_blocks_to_freeze:
                        for param in layer.parameters():
                            param.requires_grad = False
                    # once we pass N hidden Linear layers, leave the rest trainable (including head)

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
            devices=1,
            callbacks=[checkpoint_callback])

        print(f"\n--- Starting fine-tuning for {model_name} ---")
        trainer.fit(self.model, train_loader, val_loader)
        print("Fine-tuning complete.")
        return trainer.checkpoint_callback.best_model_path


def run_finetuning_workflow(rosetta_training_data, empirical_finetune_data, mappings_json,
                            checkpoint_path, metrics_dir, levels, features_path=None):
    training_df_ = pd.read_parquet(rosetta_training_data)
    empirical_df_ = pd.read_parquet(empirical_finetune_data)
    finetuning_split_path = os.path.join(os.path.dirname(empirical_finetune_data), 'finetuning_split_info.json')

    with open(mappings_json, 'r') as f:
        mappings_ = json.load(f)

    # infer model type from provided checkpoint path

    # Enforce uniform feature set (from stations current_features.csv)
    if not features_path or not os.path.exists(features_path):
        raise ValueError('features_path is required and must point to stations current_features.csv')
    if features_path.endswith('.csv'):
        feats_df = pd.read_csv(features_path)
        col_name = 'features' if 'features' in feats_df.columns else feats_df.columns[0]
        listed_feats = feats_df[col_name].dropna().astype(str).tolist()
    else:
        raise ValueError('Unsupported features file; expected CSV list of features')

    # Align with GSHP training: remove GSHP label columns from features
    rosetta_cols_ = [c for c in training_df_.columns if any(p in c for p in VG_PARAMS) or c in ['theta_r', 'theta_s', 'alpha', 'n']]
    gshp_targets_ = [c for c in ['theta_r', 'theta_s', 'alpha', 'n'] if c in training_df_.columns]
    feature_cols_ = [c for c in listed_feats if c in training_df_.columns and c not in rosetta_cols_ and c not in DROP_FEATURES]
    if 'SWCC_classes' in feature_cols_:
        feature_cols_.remove('SWCC_classes')
    if 'data_flag' in feature_cols_:
        feature_cols_.remove('data_flag')

    missing_train_ = [c for c in feature_cols_ if c not in training_df_.columns]
    missing_emp_ = [c for c in feature_cols_ if c not in empirical_df_.columns]
    if missing_train_ or missing_emp_:
        raise ValueError(
            f'Missing required features. train missing: {missing_train_}, finetune missing: {missing_emp_}')
    cat_cols_ = [c for c in mappings_.keys() if c in feature_cols_]

    for level_to_tune_ in levels:
        target_name_ = 'GSHP_VG_combined'
        targets_ = gshp_targets_

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

        model_type_ = 'MLPEmbeddings' if 'MLPEmbeddings-' in checkpoint_path else (
            'FTTransformer' if 'FTTransformer-' in checkpoint_path else 'MLP')

        print(f"\n--- Fine-tuning provided checkpoint as {model_type_} ---")
        print(f"Preparing data for fine-tuning {target_name_}...")
        use_one_hot_ = model_type_ == 'MLP'
        # Use shared preparation to build train/val datasets from empirical_df_
        train_ds_, val_ds_, n_feat_or_num_, cat_cards_or_none_, target_stats_ = prepare_data(
            empirical_df_.copy(),
            targets_,
            feature_cols_,
            cat_cols_,
            mappings=mappings_,
            use_one_hot=use_one_hot_,
            ref_df=training_df_[feature_cols_] if use_one_hot_ else None
        )

        n_outputs_ = len(targets_)
        if model_type_ == 'MLP':
            base_model_ = VanillaMLP(n_features=n_feat_or_num_, n_outputs=n_outputs_)
        elif model_type_ == 'MLPEmbeddings':
            base_model_ = MLPWithEmbeddings(
                n_num_features=n_feat_or_num_,
                cat_cardinalities=cat_cards_or_none_,
                n_outputs=n_outputs_,
            )
        else:
            base_model_ = rtdl.FTTransformer.make_baseline(
                n_num_features=n_feat_or_num_, cat_cardinalities=cat_cards_or_none_, d_token=256,
                ffn_d_hidden=32, residual_dropout=0.0,
                n_blocks=3, attention_dropout=0.2, ffn_dropout=0.2, d_out=n_outputs_)

        pl_module_ = TabularLightningModule.load_from_checkpoint(checkpoint_path, model=base_model_,
                                                                 n_outputs=n_outputs_)

        finetune_config_ = {'lr': 5e-6, 'epochs': 30, 'freeze_layers': 1, 'batch_size': 8}

        # optional: write split info only if indices are present (when using random_split)
        try:
            if hasattr(train_ds_, 'indices') and hasattr(val_ds_, 'indices'):
                print("Saving fine-tuning split info...")
                split_data = {'train': list(train_ds_.indices), 'validation': list(val_ds_.indices)}
                with open(finetuning_split_path, 'w') as f:
                    json.dump(split_data, f, indent=2)
        except Exception:
            pass

        tuner_ = FineTuner(pl_module_, train_ds_, val_ds_, config=finetune_config_)
        finetuned_ckpt_dir_ = os.path.join(os.path.dirname(checkpoint_path), 'fine_tuned')
        os.makedirs(finetuned_ckpt_dir_, exist_ok=True)
        finetuned_model_name = f'{target_name_}_{model_type_}'
        best_ckpt_path_ = tuner_.run(checkpoint_dir=finetuned_ckpt_dir_, model_name=finetuned_model_name)
        try:
            r2m = re.search(r"val_r2=([-]?\d+\.\d+)", os.path.basename(best_ckpt_path_))
            best_val_r2_ = float(r2m.group(1)) if r2m else None
            results_path_ = os.path.join(
                finetuned_ckpt_dir_, f'{finetuned_model_name}_results.json'
            )
            with open(results_path_, 'w') as rf_:
                json.dump({
                    'target': target_name_,
                    'model_type': model_type_,
                    'best_checkpoint': best_ckpt_path_,
                    'best_val_r2': best_val_r2_,
                    'split_info': finetuning_split_path,
                }, rf_, indent=2)
        except Exception:
            pass  # leave silently if results cannot be written

        # Also emit metrics JSON in the same structure as train_tabular_nn.py
        try:
            # Rebuild model skeleton matching training
            if model_type_ == 'MLP':
                eval_model_ = VanillaMLP(n_features=n_feat_or_num_, n_outputs=n_outputs_)
                eval_ds_ = val_ds_
            elif model_type_ == 'MLPEmbeddings':
                eval_model_ = MLPWithEmbeddings(
                    n_num_features=n_feat_or_num_,
                    cat_cardinalities=cat_cards_or_none_,
                    n_outputs=n_outputs_,
                )
                eval_ds_ = val_ds_
            else:
                eval_model_ = rtdl.FTTransformer.make_baseline(
                    n_num_features=n_feat_or_num_, cat_cardinalities=cat_cards_or_none_, d_token=256,
                    ffn_d_hidden=32, residual_dropout=0.0,
                    n_blocks=3, attention_dropout=0.2, ffn_dropout=0.2, d_out=n_outputs_)
                eval_ds_ = val_ds_

            eval_module_ = TabularLightningModule.load_from_checkpoint(
                best_ckpt_path_, model=eval_model_, n_outputs=n_outputs_)
            val_loader_ = DataLoader(val_ds_, batch_size=16)
            eval_trainer_ = pl.Trainer(accelerator=DEVICE, devices=1)
            eval_trainer_.test(eval_module_, val_loader_)

            y_pred_ = np.concatenate(eval_module_.test_preds)
            y_true_ = np.concatenate(eval_module_.test_targets)

            metrics_ = {}
            for i, t in enumerate(targets_):
                p_name = targets_[i]
                r2v = r2_score(y_true_[:, i], y_pred_[:, i])
                rmsev = root_mean_squared_error(y_true_[:, i], y_pred_[:, i])
                metrics_[p_name] = {
                    'r2': r2v,
                    'rmse': rmsev.item(),
                    'mean_val': target_stats_[t]['mean'],
                    'std_val': target_stats_[t]['std'],
                }

            all_metrics_ = {target_name_: metrics_}
            os.makedirs(metrics_dir, exist_ok=True)
            ts_ = re.sub(r'[^0-9]', '', os.path.basename(best_ckpt_path_))[:14] or '00000000000000'
            out_json_ = os.path.join(metrics_dir, f'{model_type_}_combined_finetuned_{ts_}.json')
            with open(out_json_, 'w') as jf_:
                json.dump(all_metrics_, jf_, indent=4)
        except Exception:
            pass


if __name__ == '__main__':
    home_ = os.path.expanduser('~')
    root_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'swapstress')
    training_data_root_ = os.path.join(root_, 'training')
    # finetune_data_root_ = os.path.join(root_, 'finetune')

    base_training_data = os.path.join(training_data_root_, 'gshp_training_data_emb_250m.parquet')
    empirical_finetune_data_ = os.path.join(training_data_root_, 'stations_training_table_250m.parquet')
    mappings_json_ = os.path.join(training_data_root_, 'gshp_categorical_mappings_250m.json')
    checkpoint_path_ = os.path.join(training_data_root_, 'checkpoints_gshp', 'GSHP_VG_combined',
                                    'FTTransformer-20250924_112939-epoch=24-val_r2=0.22.ckpt')

    metrics_ = os.path.join(training_data_root_, 'metrics')

    features_ = os.path.join(training_data_root_, 'current_features.csv')
    metrics_subdir = 'finetune_gshp_embeddings'
    metrics_dst = os.path.join(metrics_, metrics_subdir)

    run_finetuning_workflow(
        rosetta_training_data=base_training_data,
        empirical_finetune_data=empirical_finetune_data_,
        mappings_json=mappings_json_,
        checkpoint_path=checkpoint_path_,
        metrics_dir=metrics_dst,
        levels=(2,),
        features_path=features_)

# ========================= EOF ====================================================================
