import os
import json
from tqdm import tqdm
from glob import glob
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

from map.learning import DEVICE
from map.learning.mae.dataset import CombinedVwcDataset
from map.learning.mae.mae import VwcMAE


def _score_window(v):
    dv = np.diff(v)
    ddv = np.diff(dv) if len(dv) > 1 else np.array([0.0])
    return 0.7 * np.sum(np.abs(dv)) + 0.3 * np.sum(np.abs(ddv))


def _dyn_worker(params):
    fp, gridmet_dir, window_len, stride_days, topk_overall = params
    out = []
    gmp = os.path.join(gridmet_dir, os.path.basename(fp)) if gridmet_dir else None
    if gmp and not os.path.exists(gmp):
        return out
    vdf = pd.read_parquet(fp)[['shallow', 'middle']]
    gdf = pd.read_parquet(gmp) if gmp else pd.DataFrame(index=vdf.index)
    idx = vdf.index.intersection(gdf.index) if not gdf.empty else vdf.index
    idx = pd.DatetimeIndex(idx)
    vdf = vdf.loc[idx]
    if len(idx) < window_len:
        return out
    sig = vdf[['shallow', 'middle']].mean(axis=1).values.astype('float32')
    starts = list(range(0, len(idx) - window_len + 1, stride_days))
    scored = []
    for s0 in starts:
        s1 = s0 + window_len
        wv = sig[s0:s1]
        if len(wv) == window_len:
            scored.append((s0, _score_window(wv)))
    scored.sort(key=lambda x: x[1], reverse=True)
    keep = scored if topk_overall is None else scored[:topk_overall]
    for s0, _ in keep:
        out.append({'file': fp, 'gm_file': gmp, 'start': int(s0), 'stop': int(s0 + window_len)})
    return out


def _build_dynamic_windows(files, gridmet_dir, window_len, stride_days=7,
                           topk_overall=None, num_workers=24):

    windows = []
    if not files:
        return windows
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(_dyn_worker, (fp, gridmet_dir, window_len, stride_days, topk_overall)): fp for fp in files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc='Scoring dynamic windows'):
            res = fut.result()
            if res:
                windows.extend(res)
    return windows


def run_training(data_dir, batch_size=64, epochs=50, mask_ratio=0.3, num_workers=4, seed=42, files=None,
                 gridmet_dir=None, window_len=730, stride=550, static_features_pqt=None, static_id_col=None,
                 static_cols=None, use_dynamic_windows=False, dyn_stride_days=7, dyn_topk_per_year=1):
    files = files if files is not None else sorted(glob(os.path.join(data_dir, '*.parquet')))

    # Build static features map first (if provided)
    static_map = None
    static_ids = None
    if static_features_pqt is not None and os.path.exists(static_features_pqt):
        sdf = pd.read_parquet(static_features_pqt)
        if static_id_col and static_id_col in sdf.columns:
            sdf = sdf.set_index(static_id_col)
        cols = static_cols if static_cols is not None else []
        if cols:
            keep = [c for c in cols if c in sdf.columns]
            sdf = sdf[keep]
        sdf = sdf.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        means = sdf.mean(axis=0)
        stds = sdf.std(axis=0).replace(0, 1.0)
        sdf = (sdf - means) / stds
        static_map = {str(idx): row.values.astype('float32') for idx, row in sdf.iterrows()}
        static_ids = set(static_map.keys())

    # Intersect VWC ids, GRIDMET availability, and static ids
    vwc_ids = [os.path.splitext(os.path.basename(f))[0] for f in files]
    if gridmet_dir:
        gm_exist = {vid for vid in vwc_ids if os.path.exists(os.path.join(gridmet_dir, f'{vid}.parquet'))}
    else:
        gm_exist = set(vwc_ids)
    if static_ids is None:
        eligible = set(vwc_ids) & gm_exist
    else:
        eligible = set(vwc_ids) & gm_exist & static_ids
    files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in eligible]

    rng = np.random.RandomState(seed)
    rng.shuffle(files)
    split = int(0.9 * len(files))
    train_files, val_files = files[:split], files[split:]

    if not train_files:
        return None

    f0 = train_files[0]
    gm0 = os.path.join(gridmet_dir, os.path.basename(f0)) if gridmet_dir else None
    v0 = pd.read_parquet(f0)[['shallow', 'middle']]
    g0 = pd.read_parquet(gm0) if (gm0 and os.path.exists(gm0)) else pd.DataFrame(index=v0.index)
    common_idx = v0.index.intersection(g0.index) if not g0.empty else v0.index
    T = len(common_idx)
    starts = list(range(0, max(T - window_len + 1, 0), stride))
    n_feat = g0.shape[1] if (gm0 and os.path.exists(gm0)) else 0

    if use_dynamic_windows:
        train_windows = _build_dynamic_windows(train_files, gridmet_dir, window_len,
                                               stride_days=dyn_stride_days, topk_overall=10, num_workers=24)
        val_windows = _build_dynamic_windows(val_files, gridmet_dir, window_len,
                                             stride_days=dyn_stride_days, topk_overall=10, num_workers=24)
    else:
        train_windows = []
        for fp in train_files:
            gmp = os.path.join(gridmet_dir, os.path.basename(fp)) if gridmet_dir else None
            if gmp and not os.path.exists(gmp):
                continue
            for s in starts:
                train_windows.append({'file': fp, 'gm_file': gmp, 'start': s, 'stop': s + window_len})

        val_windows = []
        for fp in val_files:
            gmp = os.path.join(gridmet_dir, os.path.basename(fp)) if gridmet_dir else None
            if gmp and not os.path.exists(gmp):
                continue
            for s in starts:
                val_windows.append({'file': fp, 'gm_file': gmp, 'start': s, 'stop': s + window_len})

    train_ds = CombinedVwcDataset(train_windows, zscore=True, mask_mode='mixed', static_map=static_map)
    val_ds = CombinedVwcDataset(val_windows, zscore=True, mask_mode='mixed', static_map=static_map)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

    model = VwcMAE(seq_len=window_len, mask_ratio=mask_ratio, n_feat=n_feat, n_channels=2,
                   contrastive_weight=0.2, temperature=0.1, sim_weight=0.5)

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
    # home = os.path.expanduser('~')
    # root_ = os.path.join(home, 'data', 'IrrigationGIS')
    # vwc_ = os.path.join(root_, 'soils', 'swapstress', 'vwc')

    vwc_ = '/data/ssd2/swapstress/vwc'

    epochs_ = 100
    batch_size_ = 128
    workers_ = 4
    mask_ratio_ = 0.3

    project_ = 'rosetta'
    data_root_ = os.path.join(vwc_, 'hhp', f'{project_}')
    gridmet_dir_ = os.path.join(vwc_, 'gridmet', f'{project_}')
    static_features_pqt_ = os.path.join(os.path.expanduser('~'), 'data', 'IrrigationGIS', 'soils', 'swapstress',
                                        'training', 'training_data.parquet')
    static_id_col_ = 'site_id'

    static_cols_ = ['A00', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13',
                    'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27',
                    'A28', 'A29', 'A30', 'A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'A37', 'A38', 'A39', 'A40', 'A41',
                    'A42', 'A43', 'A44', 'A45', 'A46', 'A47', 'A48', 'A49', 'A50', 'A51', 'A52', 'A53', 'A54', 'A55',
                    'A56', 'A57', 'A58', 'A59', 'A60', 'A61', 'A62', 'A63']

    run_training(data_root_, batch_size=batch_size_, epochs=epochs_, mask_ratio=mask_ratio_,
                 num_workers=workers_, gridmet_dir=gridmet_dir_, static_cols=static_cols_,
                 static_features_pqt=static_features_pqt_, static_id_col=static_id_col_,
                 window_len=180,
                 use_dynamic_windows=True)

# ========================= EOF ====================================================================
