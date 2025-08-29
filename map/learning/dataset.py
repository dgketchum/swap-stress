import os
import json
from glob import glob

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    """Custom Dataset for tabular data with separate numerical and categorical features."""

    def __init__(self, X_num, X_cat, y):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.int64)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        self.y = y_tensor.unsqueeze(1) if len(y_tensor.shape) == 1 else y_tensor

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.X_num[idx], self.X_cat[idx]), self.y[idx]


class TabularDatasetVanilla(Dataset):
    """Custom Dataset for tabular data with a single flattened feature tensor."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        self.y = y_tensor.unsqueeze(1) if len(y_tensor.shape) == 1 else y_tensor

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class VWCTimeSeriesDataset(Dataset):
    """
    Builds examples from per-station VWC time series and SWRC fitted targets.

    Each example corresponds to a single station-depth pair:
      - X: last `seq_len` daily VWC values for a specific depth (shape [seq_len, 1])
      - y: target vector [theta_r, theta_s, log10_alpha, log10_n]
    """

    def __init__(self, vwc_dir, swrc_results_dir, seq_len=365, min_obs=60):
        self.seq_len = seq_len
        self.samples = []
        self.meta = []
        self.mean_ = None
        self.std_ = None

        # Load fitted targets
        targets = {}
        for fp in glob(os.path.join(swrc_results_dir, '*_fit_results.json')):
            station = os.path.basename(fp).replace('_fit_results.json', '')
            try:
                with open(fp, 'r') as f:
                    res = json.load(f)
            except Exception:
                continue
            # Map depth -> params
            station_targets = {}
            for depth_str, info in res.items():
                try:
                    if info.get('status') != 'Success':
                        continue
                except AttributeError:
                    continue
                try:
                    d = int(depth_str)
                except Exception:
                    continue
                p = info['parameters']
                try:
                    station_targets[d] = {
                        'theta_r': float(p['theta_r']['value']),
                        'theta_s': float(p['theta_s']['value']),
                        'log10_alpha': float(np.log10(max(p['alpha']['value'], 1e-9))),
                        'log10_n': float(np.log10(max(p['n']['value'], 1 + 1e-9))),
                    }
                except Exception:
                    continue
            if station_targets:
                targets[station] = station_targets

        # Build samples from VWC parquet files
        for fp in glob(os.path.join(vwc_dir, '*.parquet')):
            try:
                df = pd.read_parquet(fp)
            except Exception:
                continue
            if df.empty or 'station' not in df.columns:
                continue
            station = str(df['station'].iloc[0])
            if station not in targets:
                continue

            # Sort by time, coerce to daily frequency if needed
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                df = df.dropna(subset=['datetime']).sort_values('datetime')
                df = df.set_index('datetime')
                df = df.asfreq('D')  # regularize to daily, introduces NaNs
            else:
                continue

            # Identify VWC columns
            vwc_cols = [c for c in df.columns if isinstance(c, str) and c.startswith('soil_vwc_') and not c.endswith('_units')]
            if not vwc_cols:
                continue

            # For each VWC depth column, match to same depth params if available
            for col in vwc_cols:
                try:
                    depth_cm = int(col.replace('soil_vwc_', ''))
                except Exception:
                    continue
                if depth_cm not in targets[station]:
                    continue

                series = df[col]
                if series.notna().sum() < min_obs:
                    continue

                # Take last seq_len window, fill missing with median
                tail = series.iloc[-self.seq_len:]
                if len(tail) < self.seq_len:
                    pad_len = self.seq_len - len(tail)
                    pad_index = pd.date_range(end=tail.index[0] - pd.Timedelta(days=1), periods=pad_len, freq='D')
                    pad = pd.Series([np.nan] * pad_len, index=pad_index)
                    tail = pd.concat([pad, tail])

                filled = tail.fillna(tail.median())
                x = filled.values.astype(np.float32).reshape(self.seq_len, 1)
                t = targets[station][depth_cm]
                y = np.array([t['theta_r'], t['theta_s'], t['log10_alpha'], t['log10_n']], dtype=np.float32)
                self.samples.append((x, y))
                self.meta.append({
                    'station': station,
                    'depth_cm': depth_cm,
                    'column': col,
                    'source_file': os.path.basename(fp),
                })

        # Compute normalization stats over all training features
        if self.samples:
            all_vals = np.concatenate([s[0] for s in self.samples], axis=0)
            self.mean_ = float(np.nanmean(all_vals))
            self.std_ = float(np.nanstd(all_vals) + 1e-6)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        if self.mean_ is not None:
            x = (x - self.mean_) / self.std_
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
