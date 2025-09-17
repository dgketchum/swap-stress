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


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
