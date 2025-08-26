import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.regression import R2Score

from map.learning import LEARNING_RATE, WEIGHT_DECAY


class MLPWithEmbeddings(nn.Module):
    def __init__(self, n_num_features, cat_cardinalities, embedding_dim=10, hidden_dim=128, n_outputs=1):
        super().__init__()
        self.cat_cardinalities = cat_cardinalities
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings, embedding_dim) for num_embeddings in cat_cardinalities
        ])
        n_cat_features = sum(e.embedding_dim for e in self.embeddings)
        total_features = n_num_features + n_cat_features
        self.layers = nn.Sequential(
            nn.Linear(total_features, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.BatchNorm1d(hidden_dim // 2), nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_outputs)
        )

    def forward(self, x_num, x_cat):

        # Before the embedding lookup in your forward pass
        for i, num_embeddings in enumerate(self.cat_cardinalities):  # Assuming cat_cardinalities is accessible
            column_data = x_cat[:, i]
            max_val = column_data.max()
            min_val = column_data.min()

            if min_val < 0:
                raise ValueError(f"Error: Column {i} has a negative value: {min_val}")
            if max_val >= num_embeddings:
                raise ValueError(
                    f"Error: Column {i} has max value {max_val}, which is out of bounds "
                    f"for embedding size {num_embeddings}. Valid range is [0, {num_embeddings - 1}]."
                )

        cat_embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_cat_processed = torch.cat(cat_embeds, dim=1)
        x = torch.cat([x_num, x_cat_processed], dim=1)
        return self.layers(x)


class VanillaMLP(nn.Module):
    def __init__(self, n_features, hidden_dim=128, n_outputs=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.BatchNorm1d(hidden_dim // 2), nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_outputs)
        )

    def forward(self, x):
        return self.layers(x)


class TabularLightningModule(pl.LightningModule):
    """A generic LightningModule for any of our tabular models."""

    def __init__(self, model, n_outputs=1):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.criterion = nn.MSELoss()
        self.r2_score = R2Score(num_outputs=n_outputs)
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
        r2 = self.r2_score(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_r2', r2, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        r2 = self.r2_score(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_r2', r2, on_step=False, on_epoch=True, prog_bar=True)
        self.test_preds.append(y_hat.cpu().numpy())
        self.test_targets.append(y.cpu().numpy())

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self(x)

    def configure_optimizers(self):
        lr = getattr(self, 'learning_rate', LEARNING_RATE)
        print(f"Configuring optimizer with learning rate: {lr}")

        trainable_params = filter(lambda p: p.requires_grad, self.parameters())

        optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=WEIGHT_DECAY)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5,
                                                                  verbose=False),
                'monitor': 'val_loss',
            }
        }

if __name__ == '__main__':
    pass

# ========================= EOF ====================================================================
