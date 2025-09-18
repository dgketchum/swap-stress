import os
import re
from glob import glob

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

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


def _score_window(v):
    dv = np.diff(v)
    ddv = np.diff(dv) if len(dv) > 1 else np.array([0.0])
    return 0.7 * np.sum(np.abs(dv)) + 0.3 * np.sum(np.abs(ddv))


def _dyn_worker_inf(params):
    fp, gridmet_dir, window_len, dyn_stride_days, dyn_topk_per_year = params
    windows = []
    ids = []
    weights = []
    gmp = os.path.join(gridmet_dir, os.path.basename(fp)) if gridmet_dir else None
    if gmp and not os.path.exists(gmp):
        return windows, ids, weights
    vdf = pd.read_parquet(fp)[['shallow', 'middle']]
    gdf = pd.read_parquet(gmp) if gmp else pd.DataFrame(index=vdf.index)
    idx = vdf.index.intersection(gdf.index) if not gdf.empty else vdf.index
    idx = pd.DatetimeIndex(idx)
    vdf = vdf.loc[idx]
    if len(idx) < window_len:
        return windows, ids, weights
    sig = vdf[['shallow', 'middle']].mean(axis=1).values.astype('float32')
    years = sorted(set(idx.year))
    for y in years:
        year_mask = idx.year == y
        year_idx = np.where(year_mask)[0]
        if len(year_idx) < window_len:
            continue
        starts = []
        s = 0
        while s + window_len <= len(year_idx):
            starts.append(year_idx[s])
            s += dyn_stride_days
        if not starts:
            continue
        scored = []
        for s0 in starts:
            s1 = s0 + window_len
            wv = sig[s0:s1]
            if len(wv) == window_len:
                scored.append((s0, _score_window(wv)))
        if not scored:
            continue
        scored.sort(key=lambda x: x[1], reverse=True)
        for s0, sc in scored[:max(1, dyn_topk_per_year)]:
            windows.append({'file': fp, 'gm_file': gmp, 'start': int(s0), 'stop': int(s0 + window_len)})
            ids.append(os.path.splitext(os.path.basename(fp))[0])
            weights.append(float(sc))
    return windows, ids, weights


def _build_windows(data_dir, gridmet_dir, window_len=730, stride=550, dynamic=False, dyn_stride_days=7,
                   dyn_topk_per_year=1, num_workers=24):
    files = sorted(glob(os.path.join(data_dir, '*.parquet')))
    if not files:
        return [], [], []

    windows = []
    ids = []
    weights = []

    if not dynamic:
        f0 = files[0]
        gm0 = os.path.join(gridmet_dir, os.path.basename(f0)) if gridmet_dir else None
        v0 = pd.read_parquet(f0)[['shallow', 'middle']]
        g0 = pd.read_parquet(gm0) if gm0 else pd.DataFrame(index=v0.index)
        idx = v0.index.intersection(g0.index) if not g0.empty else v0.index
        T = len(idx)
        starts = list(range(0, max(T - window_len + 1, 0), stride))

        for fp in files:
            gmp = os.path.join(gridmet_dir, os.path.basename(fp)) if gridmet_dir else None
            for s in starts:
                windows.append({'file': fp, 'gm_file': gmp, 'start': s, 'stop': s + window_len})
                ids.append(os.path.splitext(os.path.basename(fp))[0])
                weights.append(1.0)
        return windows, ids, weights

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = {
            ex.submit(_dyn_worker_inf, (fp, gridmet_dir, window_len, dyn_stride_days, dyn_topk_per_year)): fp
            for fp in files
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc='Scoring dynamic windows'):
            w, i, wt = fut.result()
            if w:
                windows.extend(w)
                ids.extend(i)
                weights.extend(wt)
    return windows, ids, weights


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
                  window_len=180, stride=550, dynamic=False, dyn_stride_days=7, dyn_topk_per_year=1):
    os.makedirs(out_dir, exist_ok=True)

    windows, ids, weights = _build_windows(data_dir, gridmet_dir, window_len=window_len, stride=stride,
                                           dynamic=dynamic, dyn_stride_days=dyn_stride_days,
                                           dyn_topk_per_year=dyn_topk_per_year)
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
    if weights:
        # normalize weights per id and aggregate weighted mean
        from collections import defaultdict
        wsum = defaultdict(float)
        vecsum = defaultdict(lambda: np.zeros(emb_array.shape[1], dtype=np.float32))
        for i, key in enumerate(ids):
            w = max(1e-6, float(weights[i]))
            vecsum[key] += emb_array[i] * w
            wsum[key] += w
        agg = {k: (vecsum[k] / wsum[k]) for k in vecsum}
    else:
        agg = _aggregate_embeddings(ids, emb_array)

    cols = [f'e{i:02d}' for i in range(64)]
    print(f'writing {len(agg)} embeddings to {out_dir}')
    for key, vec in agg.items():
        df = pd.DataFrame([vec], columns=cols, index=[key])
        out_fp = os.path.join(out_dir, f'{key}.parquet')
        df.to_parquet(out_fp)
    return out_dir


if __name__ == '__main__':
    run_mt_mesonet_workflow = False
    run_rosetta_workflow = False
    run_gshp_workflow = False
    run_reesh_workflow = True

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
        ckpt_path_ = os.path.join(ckpt_root_, 'both_20250917_170218/mae-both-20250917-epoch=99-val_loss=0.1314.ckpt')
        # ckpt_path_ = os.path.join(ckpt_root_, 'both_20250916_173137',
        #                          'mae-both-20250916-epoch=57-val_loss=0.0011.ckpt')

        if ckpt_path_ is None:
            ckpt_path_ = find_best_model_checkpoint(ckpt_root_)

        out_dir_ = os.path.join(vwc_root_, 'embeddings', project_)
        run_inference(ckpt_path_, data_root_, gridmet_dir_, out_dir_,
                      window_len=180,
                      )

# ========================= EOF ====================================================================
