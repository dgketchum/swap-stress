"""
PyTorch Lightning module for tabular regression on the direct suction task.

Wraps any of the model architectures in models.py and handles:
- MSE loss
- AdamW optimizer with ReduceLROnPlateau
- RMSE monitoring for early stopping
- Test-time prediction collection
"""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn as nn


class DirectRegressionModule(L.LightningModule):
    """Lightning wrapper for tabular regression models.

    Parameters
    ----------
    model : nn.Module
        One of VanillaMLP, MLPWithEmbeddings, or FTTransformer.
    learning_rate : float
        Initial learning rate.
    weight_decay : float
        AdamW weight decay.
    split_input : bool
        If True, the model expects (x_num, x_cat) instead of a single tensor.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        split_input: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.split_input = split_input
        self.criterion = nn.MSELoss()

        # Collect test predictions
        self._test_preds: list[torch.Tensor] = []
        self._test_targets: list[torch.Tensor] = []

    def forward(self, *args: Any) -> torch.Tensor:
        if self.split_input:
            x_num, x_cat = args[0], args[1]
            return self.model(x_num, x_cat)
        return self.model(args[0])

    def _step(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.split_input:
            (x_num, x_cat), y = batch
            y_hat = self.model(x_num, x_cat)
        else:
            x, y = batch
            y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss, _, _ = self._step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        loss, y_hat, y = self._step(batch)
        self.log("val_loss", loss, prog_bar=True)
        rmse = torch.sqrt(loss)
        self.log("val_rmse", rmse, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        loss, y_hat, y = self._step(batch)
        self.log("test_loss", loss)
        self._test_preds.append(y_hat.detach().cpu())
        self._test_targets.append(y.detach().cpu())

    def on_test_epoch_end(self) -> None:
        self.test_predictions = torch.cat(self._test_preds, dim=0).numpy().ravel()
        self.test_targets = torch.cat(self._test_targets, dim=0).numpy().ravel()
        self._test_preds.clear()
        self._test_targets.clear()

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_rmse",
                "interval": "epoch",
            },
        }
