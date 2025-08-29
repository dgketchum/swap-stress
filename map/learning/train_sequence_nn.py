import os
from datetime import datetime

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from map.learning import DEVICE
from map.learning.dataset import VWCTimeSeriesDataset
from map.learning.sequence_nn import SequenceRegressor


def train_sequence_model(vwc_dir, swrc_results_dir, checkpoint_dir, seq_len=365, batch_size=32, max_epochs=60,
                         seed=42):
    dataset = VWCTimeSeriesDataset(vwc_dir=vwc_dir, swrc_results_dir=swrc_results_dir, seq_len=seq_len)
    if len(dataset) < 10:
        raise ValueError(f"Insufficient samples constructed from {vwc_dir} and {swrc_results_dir}")

    pl.seed_everything(seed, workers=True)

    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(seed))

    # Save split information (station, depth) for reproducibility
    train_indices = train_ds.indices
    val_indices = val_ds.indices
    split_info = {
        'train': {int(i): dataset.meta[i] for i in train_indices},
        'validation': {int(i): dataset.meta[i] for i in val_indices},
    }
    os.makedirs(checkpoint_dir, exist_ok=True)
    split_path = os.path.join(checkpoint_dir, 'sequence_split_info.json')
    with open(split_path, 'w') as f:
        import json as _json
        _json.dump(split_info, f, indent=2)
    print(f"Saved sequence split info to {split_path}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = SequenceRegressor(seq_len=seq_len, n_outputs=4, lr=1e-3)
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'seq1d-{datetime.now().strftime("%Y%m%d_%H%M%S")}-' + '{epoch:02d}-{val_r2:.2f}',
        save_top_k=1,
        monitor='val_r2',
        mode='max',
        save_weights_only=False,
    )

    trainer = pl.Trainer(max_epochs=max_epochs, accelerator=DEVICE, devices=1, callbacks=[checkpoint_cb])
    trainer.fit(model, train_loader, val_loader)

    return checkpoint_cb.best_model_path


if __name__ == '__main__':
    # Define data roots consistent with the rest of the repo
    home = os.path.expanduser('~')
    root = os.path.join(home, 'data', 'IrrigationGIS', 'soils')

    # Inputs
    vwc_dir_ = os.path.join(root, 'vwc_timeseries', 'mt_mesonet')
    swrc_results_dir_ = os.path.join(root, 'soil_potential_obs', 'mt_mesonet', 'results_by_station')

    # Outputs
    checkpoint_dir_ = os.path.join(root, 'swapstress', 'training', 'checkpoints', 'sequence_vwc')
    os.makedirs(checkpoint_dir_, exist_ok=True)

    best_ckpt = train_sequence_model(
        vwc_dir=vwc_dir_,
        swrc_results_dir=swrc_results_dir_,
        checkpoint_dir=checkpoint_dir_,
        seq_len=365,
        batch_size=32,
        max_epochs=60,
    )
    print(f"Best checkpoint: {best_ckpt}")
# ========================= EOF ====================================================================
