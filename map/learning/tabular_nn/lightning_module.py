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
        scheduler_monitor: str | None = "val_rmse",
        lambda_bound: float = 0.0,
        bound_lo: float = 0.0,
        bound_hi: float = 7.0,
        lambda_mono: float = 0.0,
        theta_idx: int | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.split_input = split_input
        self.scheduler_monitor = scheduler_monitor
        self.lambda_bound = lambda_bound
        self.bound_lo = bound_lo
        self.bound_hi = bound_hi
        self.lambda_mono = lambda_mono
        self.theta_idx = theta_idx
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

    def _bound_penalty(self, y_hat: torch.Tensor) -> torch.Tensor:
        """Soft penalty for predictions outside [bound_lo, bound_hi]."""
        return (
            torch.relu(y_hat - self.bound_hi).mean()
            + torch.relu(self.bound_lo - y_hat).mean()
        )

    def _mono_step(
        self, batch: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass that also computes d(pred)/d(theta) via autograd.

        Isolates the theta column as a leaf tensor with requires_grad=True,
        rebuilds the input, runs the forward pass, then differentiates the
        output w.r.t. theta.

        Returns (mse_loss, y_hat, y, d_pred_d_theta).
        """
        idx = self.theta_idx

        if self.split_input:
            (x_num, x_cat), y = batch
            # Detach and re-attach theta so it becomes an autograd leaf
            theta_col = x_num[:, idx : idx + 1].detach().requires_grad_(True)
            x_num = torch.cat([x_num[:, :idx], theta_col, x_num[:, idx + 1 :]], dim=1)
            y_hat = self.model(x_num, x_cat)
        else:
            x, y = batch
            theta_col = x[:, idx : idx + 1].detach().requires_grad_(True)
            x = torch.cat([x[:, :idx], theta_col, x[:, idx + 1 :]], dim=1)
            y_hat = self.model(x)

        mse_loss = self.criterion(y_hat, y)

        (d_pred_d_theta,) = torch.autograd.grad(
            outputs=y_hat,
            inputs=theta_col,
            grad_outputs=torch.ones_like(y_hat),
            create_graph=True,
            retain_graph=True,
        )
        return mse_loss, y_hat, y, d_pred_d_theta

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        use_mono = self.lambda_mono > 0 and self.theta_idx is not None

        if use_mono:
            loss, y_hat, _, d_pred_d_theta = self._mono_step(batch)
            mono_loss = torch.relu(d_pred_d_theta).mean()
            self.log("train_mono_loss", mono_loss, prog_bar=False)
            loss = loss + self.lambda_mono * mono_loss
        else:
            loss, y_hat, _ = self._step(batch)

        if self.lambda_bound > 0:
            bound_loss = self._bound_penalty(y_hat)
            self.log("train_bound_loss", bound_loss, prog_bar=False)
            loss = loss + self.lambda_bound * bound_loss

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
        if self.scheduler_monitor is None:
            return {"optimizer": optimizer}

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
                "monitor": self.scheduler_monitor,
                "interval": "epoch",
            },
        }
