"""
PyTorch datasets for the direct suction tabular task.

Two dataset classes:
- FlatDataset: single tensor X (for VanillaMLP with one-hot categoricals)
- SplitDataset: (X_num, X_cat) pair (for MLPWithEmbeddings and FTTransformer)
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class FlatDataset(Dataset):
    """Dataset returning (X, y) as float32 tensors."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class SplitDataset(Dataset):
    """Dataset returning ((X_num, X_cat), y).

    X_num is float32, X_cat is int64.
    """

    def __init__(self, X_num: np.ndarray, X_cat: np.ndarray, y: np.ndarray):
        self.X_num = torch.from_numpy(X_num).float()
        self.X_cat = torch.from_numpy(X_cat).long()
        self.y = torch.from_numpy(y).float().unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X_num)

    def __getitem__(self, idx: int):
        return (self.X_num[idx], self.X_cat[idx]), self.y[idx]
