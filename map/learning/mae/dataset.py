import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class CombinedVwcDataset(Dataset):
    def __init__(self, windows, zscore=True, mask_mode='mixed', mask_ratio=0.3, patch_len=30, n_patches=3,
                 end_chunk_len=180, seed=None, static_map=None):
        self.windows = list(windows)
        self.zscore = zscore
        self.mask_mode = mask_mode
        self.mask_ratio = float(mask_ratio)
        self.patch_len = int(patch_len)
        self.n_patches = int(n_patches)
        self.end_chunk_len = int(end_chunk_len)
        self.rng = np.random.RandomState(seed)
        self._id_map = {}
        self.static_map = static_map or {}

    def __len__(self):
        return len(self.windows)

    def _make_mask(self, T):
        mode = self.mask_mode
        if mode == 'mixed':
            mode = self.rng.choice(['speckled', 'patch', 'end'])
        if mode == 'speckled':
            m = self.rng.rand(T) < self.mask_ratio
        elif mode == 'patch':
            m = np.zeros(T, dtype=bool)
            L = min(self.patch_len, T)
            for _ in range(max(1, self.n_patches)):
                if T - L <= 0:
                    m[:] = True
                    break
                s = int(self.rng.randint(0, T - L + 1))
                m[s:s + L] = True
        elif mode == 'end':
            m = np.zeros(T, dtype=bool)
            L = min(self.end_chunk_len, T)
            m[T - L:] = True
        else:
            m = np.zeros(T, dtype=bool)
        return m

    def __getitem__(self, idx):
        w = self.windows[idx]
        f = w['file']
        gmf = w['gm_file']
        s = w['start']
        e = w['stop']

        vdf = pd.read_parquet(f)[['shallow', 'middle']]
        gdf = pd.read_parquet(gmf)
        common_idx = vdf.index.intersection(gdf.index)
        vdf = vdf.loc[common_idx]
        gdf = gdf.loc[common_idx]

        x = vdf.values.astype(np.float32)[s:e]
        if self.zscore:
            mu = np.nanmean(x, axis=0, keepdims=True)
            sd = np.nanstd(x, axis=0, keepdims=True) + 1e-6
            x = (x - mu) / sd
        x = np.nan_to_num(x, nan=0.0)

        gm_vals = gdf.values.astype(np.float32)[s:e]
        if self.zscore:
            gmu = np.nanmean(gm_vals, axis=0, keepdims=True)
            gsd = np.nanstd(gm_vals, axis=0, keepdims=True) + 1e-6
            gm_vals = (gm_vals - gmu) / gsd
        gm_vals = np.nan_to_num(gm_vals, nan=0.0)

        mask = self._make_mask(len(x))

        sid_str = os.path.splitext(os.path.basename(f))[0]
        if sid_str not in self._id_map:
            self._id_map[sid_str] = len(self._id_map)
        sid = self._id_map[sid_str]

        static_vec = self.static_map.get(sid_str)
        if static_vec is not None:
            static_t = torch.tensor(static_vec, dtype=torch.float32)
            return (
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(gm_vals, dtype=torch.float32),
                torch.tensor(mask, dtype=torch.bool),
                torch.tensor(sid, dtype=torch.long),
                static_t,
            )
        else:
            return (
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(gm_vals, dtype=torch.float32),
                torch.tensor(mask, dtype=torch.bool),
                torch.tensor(sid, dtype=torch.long)
            )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
