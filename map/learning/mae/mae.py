import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from map.learning import LEARNING_RATE, WEIGHT_DECAY
from map.learning.mae.dataset import PositionalEncoding


class VwcMAE(pl.LightningModule):
    def __init__(self, seq_len, d_model=128, n_heads=4, n_layers=3, emb_dim=64, mask_ratio=0.3, n_feat=0,
                 n_channels=1, contrastive_weight=0.1, temperature=0.1, sim_weight=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.mask_ratio = mask_ratio
        self.contrastive_weight = float(contrastive_weight)
        self.temperature = float(temperature)
        self.sim_weight = float(sim_weight)
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
            'contrastive_weight': self.contrastive_weight,
            'temperature': self.temperature,
            'sim_weight': self.sim_weight,
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
        sids = None
        static_vec = None
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 5:
                x, feats, mask, sids, static_vec = batch
            elif len(batch) == 4:
                x, feats, mask, sids = batch
            elif len(batch) == 3:
                x, feats, mask = batch
            elif len(batch) == 2:
                x, feats = batch
            else:
                x, feats = batch, None
        else:
            x, feats = batch, None
        b, t = x.size()[:2]
        m = mask if mask is not None else (torch.rand(b, t, device=x.device) < self.mask_ratio)
        y_pred, emb = self(x, m, feats)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        loss_mat = self.criterion(y_pred, x)
        denom = (m.float().sum() * x.size(-1)) + 1e-6
        masked_sqerr = (loss_mat * m.unsqueeze(-1).float()).sum()
        rec_loss = masked_sqerr / denom

        closs = torch.tensor(0.0, device=x.device)
        if self.contrastive_weight > 0.0 and sids is not None:
            closs = self._contrastive_loss(emb, sids)
        sim_loss = torch.tensor(0.0, device=x.device)
        if self.sim_weight > 0.0 and static_vec is not None:
            sim_loss = self._similarity_alignment_loss(emb, static_vec)
        loss = rec_loss + self.contrastive_weight * closs + self.sim_weight * sim_loss
        self.log('train_loss', loss, prog_bar=True)
        if self.contrastive_weight > 0.0:
            self.log('train_contrastive', closs, prog_bar=False)
        if self.sim_weight > 0.0:
            self.log('train_sim', sim_loss, prog_bar=False)
        self._tr_se += float(masked_sqerr.detach().cpu())
        self._tr_n += float(denom.detach().cpu())
        return loss

    def validation_step(self, batch, batch_idx):
        mask = None
        sids = None
        static_vec = None
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 5:
                x, feats, mask, sids, static_vec = batch
            elif len(batch) == 4:
                x, feats, mask, sids = batch
            elif len(batch) == 3:
                x, feats, mask = batch
            elif len(batch) == 2:
                x, feats = batch
            else:
                x, feats = batch, None
        else:
            x, feats = batch, None
        b, t = x.size()[:2]
        m = mask if mask is not None else (torch.rand(b, t, device=x.device) < self.mask_ratio)
        y_pred, emb = self(x, m, feats)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        loss_mat = self.criterion(y_pred, x)
        denom = (m.float().sum() * x.size(-1)) + 1e-6
        masked_sqerr = (loss_mat * m.unsqueeze(-1).float()).sum()
        rec_loss = masked_sqerr / denom
        closs = torch.tensor(0.0, device=x.device)
        if self.contrastive_weight > 0.0 and sids is not None:
            closs = self._contrastive_loss(emb, sids)
        sim_loss = torch.tensor(0.0, device=x.device)
        if self.sim_weight > 0.0 and static_vec is not None:
            sim_loss = self._similarity_alignment_loss(emb, static_vec)
        loss = rec_loss + self.contrastive_weight * closs + self.sim_weight * sim_loss
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        if self.contrastive_weight > 0.0:
            self.log('val_contrastive', closs, prog_bar=False, on_step=False, on_epoch=True)
        if self.sim_weight > 0.0:
            self.log('val_sim', sim_loss, prog_bar=False, on_step=False, on_epoch=True)
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

    def _contrastive_loss(self, emb, sids):
        z = F.normalize(emb, dim=1)
        sim = torch.matmul(z, z.t()) / self.temperature
        b = sim.size(0)
        labels = sids.view(-1, 1)
        pos_mask = labels.eq(labels.t()).to(sim.device)
        eye = torch.eye(b, dtype=torch.bool, device=sim.device)
        pos_mask = pos_mask & (~eye)
        exp_sim = torch.exp(sim)
        denom = exp_sim.sum(dim=1) - torch.exp(sim.diag())
        pos_sum = (exp_sim * pos_mask).sum(dim=1)
        valid = pos_sum > 0
        if valid.any():
            loss = -torch.log(pos_sum[valid] / (denom[valid] + 1e-8)).mean()
        else:
            loss = torch.tensor(0.0, device=sim.device)
        return loss

    @staticmethod
    def _similarity_alignment_loss(emb, static_vec):
        z = F.normalize(emb, dim=1)
        s = F.normalize(static_vec, dim=1)
        sim_z = torch.matmul(z, z.t())
        sim_s = torch.matmul(s, s.t())
        b = sim_z.size(0)
        mask = ~torch.eye(b, dtype=torch.bool, device=sim_z.device)
        diff = (sim_z - sim_s)[mask]
        return (diff.pow(2)).mean()


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
