import pytorch_lightning as pl
import torch
import torch.nn as nn

from map.learning import LEARNING_RATE, WEIGHT_DECAY
from map.learning.tabular_nn.dataset import PositionalEncoding


class VwcMAE(pl.LightningModule):
    def __init__(self, seq_len, d_model=128, n_heads=4, n_layers=3, emb_dim=64, mask_ratio=0.3, n_feat=0, n_channels=1):
        super().__init__()
        self.seq_len = seq_len
        self.mask_ratio = mask_ratio
        self.token_v = nn.Linear(n_channels, d_model)
        self.token_f = nn.Linear(n_feat, d_model) if n_feat > 0 else None
        self.pos = PositionalEncoding(d_model, seq_len)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        self.decoder = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, n_channels))
        self.proj_embed = nn.Linear(d_model, emb_dim)
        self.criterion = nn.MSELoss(reduction='none')
        self.save_hyperparameters({
            'seq_len': seq_len,
            'd_model': d_model,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'emb_dim': emb_dim,
            'mask_ratio': mask_ratio,
            'n_feat': n_feat,
            'n_channels': n_channels,
        })
        self._tr_se = 0.0
        self._tr_n = 0.0
        self._va_se = 0.0
        self._va_n = 0.0

    def forward(self, x, mask, feats=None):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        z_v = self.token_v(x)
        if self.token_f is not None and feats is not None:
            z_f = self.token_f(feats)
        else:
            z_f = torch.zeros_like(z_v)
        mask_tok = self.mask_token.view(1, 1, -1).expand(z_v.size(0), z_v.size(1), -1)
        z = torch.where(mask.unsqueeze(-1), mask_tok, z_v) + z_f
        z = self.pos(z)
        h = self.encoder(z)
        recon = self.decoder(h)
        emb = self.proj_embed(h).mean(dim=1)
        return recon, emb

    def training_step(self, batch, batch_idx):
        mask = None
        if isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                x, feats, mask = batch
            elif len(batch) == 2:
                x, feats = batch
            else:
                x, feats = batch, None
        else:
            x, feats = batch, None
        b, t = x.size()[:2]
        m = mask if mask is not None else (torch.rand(b, t, device=x.device) < self.mask_ratio)
        y_pred, _ = self(x, m, feats)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        loss_mat = self.criterion(y_pred, x)
        denom = (m.float().sum() * x.size(-1)) + 1e-6
        masked_sqerr = (loss_mat * m.unsqueeze(-1).float()).sum()
        loss = masked_sqerr / denom
        self.log('train_loss', loss, prog_bar=True)
        self._tr_se += float(masked_sqerr.detach().cpu())
        self._tr_n += float(denom.detach().cpu())
        return loss

    def validation_step(self, batch, batch_idx):
        mask = None
        if isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                x, feats, mask = batch
            elif len(batch) == 2:
                x, feats = batch
            else:
                x, feats = batch, None
        else:
            x, feats = batch, None
        b, t = x.size()[:2]
        m = mask if mask is not None else (torch.rand(b, t, device=x.device) < self.mask_ratio)
        y_pred, _ = self(x, m, feats)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        loss_mat = self.criterion(y_pred, x)
        denom = (m.float().sum() * x.size(-1)) + 1e-6
        masked_sqerr = (loss_mat * m.unsqueeze(-1).float()).sum()
        loss = masked_sqerr / denom
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self._va_se += float(masked_sqerr.detach().cpu())
        self._va_n += float(denom.detach().cpu())

    def on_train_epoch_start(self):
        self._tr_se = 0.0
        self._tr_n = 0.0

    def on_validation_epoch_start(self):
        self._va_se = 0.0
        self._va_n = 0.0

    def on_train_epoch_end(self):
        if self._tr_n > 0:
            rmse = (self._tr_se / (self._tr_n + 1e-6)) ** 0.5
            self.log('train_rmse', rmse, prog_bar=True, on_epoch=True)

    def on_validation_epoch_end(self):
        if self._va_n > 0:
            rmse = (self._va_se / (self._va_n + 1e-6)) ** 0.5
            self.log('val_rmse', rmse, prog_bar=True, on_epoch=True)

    @torch.no_grad()
    def embed(self, x, feats=None):
        b, t = x.size()[:2]
        m = torch.zeros(b, t, device=x.device, dtype=torch.bool)
        _, e = self(x, m, feats)
        return e

    def configure_optimizers(self):
        lr = getattr(self, 'learning_rate', LEARNING_RATE)
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=WEIGHT_DECAY)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.1, patience=5, verbose=False
                ),
                'monitor': 'val_loss',
            }
        }


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
