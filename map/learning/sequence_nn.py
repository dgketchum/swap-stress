import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.regression import R2Score


class SequenceRegressor(pl.LightningModule):
    def __init__(self, seq_len=365, n_outputs=4, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_outputs),
        )
        self.criterion = nn.MSELoss()
        self.r2 = R2Score(num_outputs=n_outputs, multioutput='uniform_average')

    def forward(self, x):
        # x: [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)
        feats = self.feature_extractor(x)
        out = self.head(feats)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        r2 = self.r2(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_r2', r2, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        r2 = self.r2(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_r2', r2, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

