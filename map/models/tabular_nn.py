import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

from map.models import LEARNING_RATE, WEIGHT_DECAY


class TabularDataset(Dataset):
    """Custom Dataset for tabular data with separate numerical and categorical features."""

    def __init__(self, X_num, X_cat, y):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.int64)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.X_num[idx], self.X_cat[idx]), self.y[idx]


class TabularDatasetVanilla(Dataset):
    """Custom Dataset for tabular data with a single flattened feature tensor."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLPWithEmbeddings(nn.Module):
    def __init__(self, n_num_features, cat_cardinalities, embedding_dim=10, hidden_dim=128):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings, embedding_dim) for num_embeddings in cat_cardinalities
        ])
        n_cat_features = sum(e.embedding_dim for e in self.embeddings)
        total_features = n_num_features + n_cat_features
        self.layers = nn.Sequential(
            nn.Linear(total_features, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.BatchNorm1d(hidden_dim // 2), nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x_num, x_cat):
        cat_embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_cat_processed = torch.cat(cat_embeds, dim=1)
        x = torch.cat([x_num, x_cat_processed], dim=1)
        return self.layers(x)


class VanillaMLP(nn.Module):
    def __init__(self, n_features, hidden_dim=128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.BatchNorm1d(hidden_dim // 2), nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.layers(x)


class TabularLightningModule(pl.LightningModule):
    """A generic LightningModule for any of our tabular models."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = nn.MSELoss()
        self.test_preds = []
        self.test_targets = []

    def forward(self, x):
        # The input x can be a tuple (x_num, x_cat) or a single tensor
        if isinstance(x, (list, tuple)):
            return self.model(*x)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.test_preds.append(y_hat.cpu().numpy())
        self.test_targets.append(y.cpu().numpy())

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


if __name__ == '__main__':
    pass

# ========================= EOF ====================================================================
