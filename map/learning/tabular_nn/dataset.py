import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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


def prepare_data(df, target_cols, feature_cols, cat_cols, mappings=None, use_one_hot=False, ref_df=None):
    if isinstance(target_cols, str):
        target_cols = [target_cols]

    data = df[target_cols + feature_cols].copy()
    for col in target_cols:
        data[data[col] <= -9999] = np.nan
    data.dropna(subset=target_cols, inplace=True)

    # Transform GSHP targets to log10 where requested
    if 'alpha' in target_cols and 'alpha' in data.columns:
        data.loc[:, 'alpha'] = np.log10(np.clip(data['alpha'].astype(float), 1e-9, None))
    if 'n' in target_cols and 'n' in data.columns:
        data.loc[:, 'n'] = np.log10(np.clip(data['n'].astype(float), 1.0 + 1e-9, None))

    for col in feature_cols:
        if col in cat_cols:
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            try:
                data[col] = data[col].fillna(data[col].mean())
            except TypeError:
                data.drop(columns=[col], inplace=True)
                feature_cols.remove(col)
                # likely error: non-numeric feature in numeric pipeline

    y = data[target_cols].values
    features_df = data[feature_cols]

    if not use_one_hot:
        for col in cat_cols:
            int_map = {int(k): int(v) for k, v in mappings[col].items()}
            features_df.loc[:, col] = features_df[col].values.astype(int)
            features_df.loc[:, col] = features_df[col].map(int_map)

            column_data = features_df.loc[:, col].values
            max_val = column_data.max()
            min_val = column_data.min()
            num_embeddings = len(mappings[col])

            if min_val < 0:
                raise ValueError(f"Error: Column {col} has a negative value: {min_val}")
            if max_val >= num_embeddings:
                raise ValueError(
                    f"Error: Column {col} has max value {max_val}, which is out of bounds "
                    f"for embedding size {num_embeddings}. Valid range is [0, {num_embeddings - 1}]."
                )

    num_cols = [c for c in features_df.columns if c not in cat_cols]

    scaler = StandardScaler()
    unscaled_vals = features_df[num_cols].copy().values.astype(np.float32)
    features_df.loc[:, num_cols] = unscaled_vals
    features_df.loc[:, num_cols] = features_df[num_cols].astype(np.float32)
    features_df.loc[:, num_cols] = scaler.fit_transform(unscaled_vals)

    target_stats = {col: {'mean': data[col].mean(), 'std': data[col].std()} for col in target_cols}

    if use_one_hot:
        # If a reference DataFrame is provided, align one-hot columns to it
        base_cols = None
        if ref_df is not None:
            ref_onehot = pd.get_dummies(ref_df[feature_cols], columns=cat_cols, dummy_na=False, dtype=int)
            base_cols = list(ref_onehot.columns)

        features_df = pd.get_dummies(features_df, columns=cat_cols, dummy_na=False, dtype=int)
        if base_cols is not None:
            missing_cols = set(base_cols) - set(features_df.columns)
            for c in missing_cols:
                features_df[c] = 0
            features_df = features_df[base_cols]
        x_train, x_test, y_train, y_test = train_test_split(features_df.values, y, test_size=0.2, random_state=42)
        train_dataset = TabularDatasetVanilla(x_train, y_train)
        test_dataset = TabularDatasetVanilla(x_test, y_test)
        return train_dataset, test_dataset, x_train.shape[1], None, target_stats
    else:
        cat_cardinalities = [len(mappings[col]) for col in cat_cols]
        x_train, x_test, y_train, y_test = train_test_split(features_df, y, test_size=0.2, random_state=42)
        train_dataset = TabularDataset(x_train[num_cols].values, x_train[cat_cols].values, y_train)
        test_dataset = TabularDataset(x_test[num_cols].values, x_test[cat_cols].values, y_test)
        return train_dataset, test_dataset, len(num_cols), cat_cardinalities, target_stats


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
