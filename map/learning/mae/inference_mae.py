import os
import re
from glob import glob

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from map.learning import DEVICE
from map.learning.mae.dataset import CombinedVwcDataset
from map.learning.mae.mae import VwcMAE


def find_best_model_checkpoint(checkpoints_root):
    import re
    ckpts = glob(os.path.join(checkpoints_root, '*', '*.ckpt'))
    if not ckpts:
        ckpts = glob(os.path.join(checkpoints_root, '**', '*.ckpt'), recursive=True)
    val_patterns = [
        re.compile(r"val_loss[=_]?(?P<val>-?\d+\.\d+)", re.IGNORECASE),
        re.compile(r"-(?P<val>\d+\.\d+)\.ckpt$"),
    ]
    scored = []
    for p in ckpts:
        base = os.path.basename(p)
        val = None
        for pat in val_patterns:
            m = pat.search(base)
            if m:
                try:
                    val = float(m.group('val'))
                except Exception:
                    val = None
                break
        if val is not None:
            scored.append((val, p))
    if scored:
        scored.sort(key=lambda x: x[0])  # lower val_loss is better
        return scored[0][1]
    if ckpts:
        ckpts.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return ckpts[0]
    return None


def _build_windows(data_dir, gridmet_dir, window_len=730, stride=550):
    files = sorted(glob(os.path.join(data_dir, '*.parquet')))
    if not files:
        return [], []

    f0 = files[0]
    gm0 = os.path.join(gridmet_dir, os.path.basename(f0)) if gridmet_dir else None
    v0 = pd.read_parquet(f0)[['shallow', 'middle']]
    g0 = pd.read_parquet(gm0) if gm0 else pd.DataFrame(index=v0.index)
    idx = v0.index.intersection(g0.index) if not g0.empty else v0.index
    T = len(idx)
    starts = list(range(0, max(T - window_len + 1, 0), stride))

    windows = []
    ids = []
    for fp in files:
        gmp = os.path.join(gridmet_dir, os.path.basename(fp)) if gridmet_dir else None
        for s in starts:
            windows.append({'file': fp, 'gm_file': gmp, 'start': s, 'stop': s + window_len})
            ids.append(os.path.splitext(os.path.basename(fp))[0])
    return windows, ids


def _aggregate_embeddings(ids, emb_array):
    sums = {}
    counts = {}
    for i, key in enumerate(ids):
        if i >= emb_array.shape[0]:
            break
        if key not in sums:
            sums[key] = emb_array[i].copy()
            counts[key] = 1
        else:
            sums[key] += emb_array[i]
            counts[key] += 1
    out = {}
    for key in sums:
        out[key] = (sums[key] / counts[key]).astype(np.float32)
    return out


def run_inference(ckpt_path, data_dir, gridmet_dir, out_dir, batch_size=256, num_workers=4,
                  window_len=730, stride=550):
    os.makedirs(out_dir, exist_ok=True)

    windows, ids = _build_windows(data_dir, gridmet_dir, window_len=window_len, stride=stride)
    if not windows:
        return None

    ds = CombinedVwcDataset(windows, zscore=True, mask_mode='mixed')
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = VwcMAE.load_from_checkpoint(ckpt_path)
    model.eval()
    model.to(DEVICE)

    embs = []
    with torch.no_grad():
        idx_ptr = 0
        for batch in tqdm(loader):
            x = batch[0]
            feats = batch[1]
            x = x.to(DEVICE)
            feats = feats.to(DEVICE)
            e = model.embed(x, feats)
            embs.append(e.detach().cpu().numpy())
            idx_ptr += len(x)

    emb_array = np.vstack(embs) if embs else np.zeros((0, 64), dtype=np.float32)
    agg = _aggregate_embeddings(ids, emb_array)

    cols = [f'e{i:02d}' for i in range(64)]
    print(f'writing {len(agg)} embeddings to {out_dir}')
    for key, vec in agg.items():
        df = pd.DataFrame([vec], columns=cols, index=[key])
        out_fp = os.path.join(out_dir, f'{key}.parquet')
        df.to_parquet(out_fp)
    return out_dir


if __name__ == '__main__':
    run_mt_mesonet_workflow = True
    run_rosetta_workflow = False
    run_gshp_workflow = False
    run_reesh_workflow = False

    vwc_root_ = '/data/ssd2/swapstress/vwc'

    if run_mt_mesonet_workflow:
        project_ = 'mt_mesonet'
    elif run_rosetta_workflow:
        project_ = 'rosetta'
    elif run_gshp_workflow:
        project_ = 'gshp'
    elif run_reesh_workflow:
        project_ = 'reesh'
    else:
        project_ = None

    if project_ is not None:
        data_root_ = os.path.join(vwc_root_, 'hhp', project_)
        gridmet_dir_ = os.path.join(vwc_root_, 'gridmet', project_)

        ckpt_root_ = os.path.join(vwc_root_, 'hhp', 'rosetta', 'checkpoints')
        ckpt_path_ = os.path.join(ckpt_root_, 'both_20250917_162619/mae-both-20250917-epoch=00-val_loss=0.6009.ckpt')
        # ckpt_path_ = os.path.join(ckpt_root_, 'both_20250916_173137',
        #                          'mae-both-20250916-epoch=57-val_loss=0.0011.ckpt')

        if ckpt_path_ is None:
            ckpt_path_ = find_best_model_checkpoint(ckpt_root_)

        out_dir_ = os.path.join(vwc_root_, 'embeddings', project_)
        run_inference(ckpt_path_, data_root_, gridmet_dir_, out_dir_)

# ========================= EOF ====================================================================
