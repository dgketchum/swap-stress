import os
import json
from glob import glob
from datetime import datetime

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

from map.learning import DEVICE
from map.learning.tabular_nn.dataset import CombinedVwcDataset
from map.learning.mae.mae import VwcMAE


def run_training(data_dir, batch_size=64, epochs=50, mask_ratio=0.3, num_workers=4, seed=42, files=None,
                 gridmet_dir=None, window_len=730, stride=550):
    files = files if files is not None else sorted(glob(os.path.join(data_dir, '*.parquet')))
    rng = np.random.RandomState(seed)
    rng.shuffle(files)
    split = int(0.9 * len(files))
    train_files, val_files = files[:split], files[split:]

    if not train_files:
        return None

    f0 = train_files[0]
    gm0 = os.path.join(gridmet_dir, os.path.basename(f0)) if gridmet_dir else None
    v0 = pd.read_parquet(f0)[['shallow', 'middle']]
    g0 = pd.read_parquet(gm0) if gm0 else pd.DataFrame(index=v0.index)
    common_idx = v0.index.intersection(g0.index) if not g0.empty else v0.index
    T = len(common_idx)
    starts = list(range(0, max(T - window_len + 1, 0), stride))
    n_feat = g0.shape[1] if gm0 else 0

    train_windows = []
    for fp in train_files:
        gmp = os.path.join(gridmet_dir, os.path.basename(fp)) if gridmet_dir else None
        for s in starts:
            train_windows.append({'file': fp, 'gm_file': gmp, 'start': s, 'stop': s + window_len})

    val_windows = []
    for fp in val_files:
        gmp = os.path.join(gridmet_dir, os.path.basename(fp)) if gridmet_dir else None
        for s in starts:
            val_windows.append({'file': fp, 'gm_file': gmp, 'start': s, 'stop': s + window_len})

    train_ds = CombinedVwcDataset(train_windows, zscore=True, mask_mode='mixed')
    val_ds = CombinedVwcDataset(val_windows, zscore=True, mask_mode='mixed')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

    model = VwcMAE(seq_len=window_len, mask_ratio=mask_ratio, n_feat=n_feat, n_channels=2)

    ckpt_root = os.path.join(data_dir, 'checkpoints')
    os.makedirs(ckpt_root, exist_ok=True)
    tag = f"both_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ckpt_dir = os.path.join(ckpt_root, tag)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f'mae-both-{datetime.now().strftime("%Y%m%d")}-' + '{epoch:02d}-{val_loss:.4f}',
        save_top_k=1, verbose=True, monitor='val_loss', mode='min'
    )

    trainer = pl.Trainer(max_epochs=epochs, accelerator=DEVICE, devices=1, callbacks=[checkpoint_callback],
                         default_root_dir=ckpt_dir)
    trainer.fit(model, train_loader, val_loader)

    metrics = {'best_model_path': checkpoint_callback.best_model_path, 'best_val_loss': float(
        checkpoint_callback.best_model_score.cpu().item()) if checkpoint_callback.best_model_score is not None else None,
               'depth': 'both', 'seq_len': window_len, 'n_feat': n_feat}
    with open(os.path.join(ckpt_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
    return ckpt_dir


def load_mae(checkpoint_path):
    model = VwcMAE.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


if __name__ == '__main__':
    run_mt_mesonet_workflow = False
    run_rosetta_workflow = True
    run_gshp_workflow = False
    run_reesh_workflow = False

    # home = os.path.expanduser('~')
    # root_ = os.path.join(home, 'data', 'IrrigationGIS')
    # vwc_ = os.path.join(root_, 'soils', 'swapstress', 'vwc')

    vwc_ = '/data/ssd2/swapstress/vwc'

    epochs_ = 100
    batch_size_ = 128
    workers_ = 4
    mask_ratio_ = 0.3

    if run_mt_mesonet_workflow:
        project_ = 'mt_mesonet'
        data_root_ = os.path.join(vwc_, 'hhp', f'{project_}')
        gridmet_dir_ = os.path.join(vwc_, 'gridmet', f'{project_}')
    elif run_rosetta_workflow:
        project_ = 'rosetta'
        data_root_ = os.path.join(vwc_, 'hhp', f'{project_}')
        gridmet_dir_ = os.path.join(vwc_, 'gridmet', f'{project_}')
    elif run_gshp_workflow:
        project_ = 'gshp'
        data_root_ = os.path.join(vwc_, 'hhp', f'{project_}')
        gridmet_dir_ = os.path.join(vwc_, 'gridmet', f'{project_}')
    elif run_reesh_workflow:
        project_ = 'reesh'
        data_root_ = os.path.join(vwc_, 'hhp', f'{project_}')
        gridmet_dir_ = os.path.join(vwc_, 'gridmet', f'{project_}')
    else:
        exit()

    run_training(data_root_, batch_size=batch_size_, epochs=epochs_, mask_ratio=mask_ratio_,
                 num_workers=workers_, gridmet_dir=gridmet_dir_)

# ========================= EOF ====================================================================
