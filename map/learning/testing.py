import json
import os
from glob import glob

import pandas as pd
import pytorch_lightning as pl
import rtdl
import torch
from torch.utils.data import DataLoader

from map.learning import DEVICE, DROP_FEATURES, VG_PARAMS
from map.learning.inference import find_best_model_checkpoint
from map.learning.tabular_nn import TabularLightningModule, VanillaMLP, MLPWithEmbeddings
from map.learning.train_tabular_nn import prepare_data

torch.set_float32_matmul_precision('medium')


def test_model(training_data_pqt, mappings_json, checkpoint_dir, model_type, target_name, level,
               use_one_hot):
    """
    Tests the best performing checkpoint for a given model and target.
    """
    df = pd.read_parquet(training_data_pqt)
    with open(mappings_json, 'r') as fj:
        mappings = json.load(fj)

    cat_cols = list(mappings.keys())
    rosetta_cols = [c for c in df.columns if any(p in c for p in VG_PARAMS)]
    feature_cols = [c for c in df.columns if c not in rosetta_cols and c not in DROP_FEATURES]

    targets = [f'US_R3H3_L{level}_VG_{p}' for p in VG_PARAMS]
    n_outputs = len(targets)

    # TODO: use test/validation sites from training
    _, test_ds, n_features, cat_cardinalities, _ = prepare_data(
        df, targets, feature_cols, cat_cols, mappings=mappings, use_one_hot=use_one_hot)

    best_ckpt, max_r2 = find_best_model_checkpoint(checkpoint_dir, target_name, model_type)

    if not best_ckpt:
        print(f"No checkpoint found for {target_name} with model type {model_type}")
        return None

    print(f"Found best checkpoint: {os.path.basename(best_ckpt)} with val_r2: {max_r2:.4f}")

    if model_type == 'MLP':
        model = VanillaMLP(n_features=n_features, n_outputs=n_outputs)
    elif model_type == 'MLPEmbeddings':
        model = MLPWithEmbeddings(n_num_features=n_features, cat_cardinalities=cat_cardinalities,
                                  n_outputs=n_outputs)
    elif model_type == 'FTTransformer':
        model = rtdl.FTTransformer.make_baseline(
            n_num_features=n_features,
            cat_cardinalities=cat_cardinalities,
            d_token=256,
            ffn_d_hidden=32,
            residual_dropout=0.0,
            n_blocks=3,
            attention_dropout=0.2,
            ffn_dropout=0.2,
            d_out=n_outputs,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    pl_module = TabularLightningModule.load_from_checkpoint(best_ckpt, model=model, n_outputs=n_outputs)
    test_loader = DataLoader(test_ds, batch_size=32)
    trainer = pl.Trainer(accelerator=DEVICE, devices=1)

    print(f"\n--- Testing {model_type} for {target_name} ---")
    test_results = trainer.test(pl_module, dataloaders=test_loader, verbose=True)

    print("\nTest results:")
    print(json.dumps(test_results, indent=4))

    return test_results


def export_presentation_csv(metrics_dir, output_csv, use_finetuned=False):
    patterns = {
        'RandomForest': 'RandomForest_combined*_*.json',
        'MLP': 'MLP_combined*_*.json',
        'MLPEmbeddings': 'MLPEmbeddings_combined*_*.json',
        'FTTransformer': 'FTTransformer_combined*_*.json',
    }
    names = {
        'RandomForest': 'Random Forest',
        'MLP': 'Vanilla Neural Network',
        'MLPEmbeddings': 'Embedding MLP',
        'FTTransformer': 'Tabular Transformer',
    }
    cols = ['Model', '', 'r²', 'RMSE', 'STD of Data', 'Mean of Data']
    params = ['theta r', 'theta s', 'log10(alpha)', 'log10(n)']
    rows = []

    for key in ['RandomForest', 'MLP', 'MLPEmbeddings', 'FTTransformer']:
        files = sorted(glob(os.path.join(metrics_dir, patterns[key])))
        metrics = None
        if files:
            with open(files[-1], 'r') as f:
                data = json.load(f)
            if 'GSHP_VG_combined' in data:
                metrics = data['GSHP_VG_combined']
            else:
                ks = [k for k in data.keys() if 'VG_combined' in k]
                if ks:
                    metrics = data[sorted(ks)[0]]
        mapping = {
            'theta r': metrics.get('theta_r') if metrics else None,
            'theta s': metrics.get('theta_s') if metrics else None,
            'log10(alpha)': (metrics.get('log10_alpha') if metrics and 'log10_alpha' in metrics else (
                metrics.get('alpha') if metrics else None)),
            'log10(n)': (metrics.get('log10_n') if metrics and 'log10_n' in metrics else (
                metrics.get('n') if metrics else None)),
        }
        first = True
        for p in params:
            m = mapping[p]
            if not m:
                rows.append([names[key] if first else '', p, '', '', '', ''])
            else:
                r2 = m.get('r2');
                rmse = m.get('rmse');
                stdv = m.get('std_val');
                meanv = m.get('mean_val')
                r2 = r2 if isinstance(r2, (int, float)) else float(r2) if r2 is not None else ''
                rmse = rmse if isinstance(rmse, (int, float)) else float(rmse) if rmse is not None else ''
                stdv = stdv if isinstance(stdv, (int, float)) else float(stdv) if stdv is not None else ''
                meanv = meanv if isinstance(meanv, (int, float)) else float(meanv) if meanv is not None else ''
                rows.append([names[key] if first else '', p, r2, rmse, stdv, meanv])
            first = False

    pd.DataFrame(rows, columns=cols).to_csv(output_csv, index=False)
    print(f'wrote {output_csv}')


if __name__ == '__main__':
    home_ = os.path.expanduser('~')
    root_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'swapstress', 'training')

    f_ = os.path.join(root_, 'training_data.parquet')
    mappings_json_ = os.path.join(root_, 'categorical_mappings.json')
    checkpoint_dir_ = os.path.join(root_, 'checkpoints', 'combined_params')

    # Example usage:
    # level_ = 2
    # target_name_ = f'L{level_}_VG_combined'
    # for model_name_ in ['MLP', 'MLPEmbeddings', 'FTTransformer']:
    #     test_model(
    #         training_data_pqt=f_,
    #         mappings_json=mappings_json_,
    #         checkpoint_dir=checkpoint_dir_,
    #         model_type=model_name_,
    #         target_name=target_name_,
    #         level=level_,
    #         use_one_hot=(model_name_ == 'MLP')
    #     )

    metrics_dir_ = os.path.join(root_, 'metrics')

    pretrained = False
    finetuned = True

    if pretrained:
        desc = 'pretrained'
    elif finetuned:
        desc = 'finetuned'
    else:
        raise ValueError

    metrics_subdir = f'{desc}_rosetta_l2'
    metrics_dst = os.path.join(metrics_dir_, metrics_subdir)

    output_csv_ = os.path.join(root_, f'{desc}_presentation_metrics.csv')
    export_presentation_csv(metrics_dst, output_csv_, use_finetuned=finetuned)

# ========================= EOF ====================================================================
