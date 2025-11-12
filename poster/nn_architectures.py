import os
import torch
from PIL import ImageShow
from visualtorch import layered_view
from map.learning.tabular_nn.tabular_nn import VanillaMLP, MLPWithEmbeddings
from map.learning.mae.mae import VwcMAE


class _NoOpViewer(ImageShow.Viewer):
    # Prevent PIL from attempting to xdg-open images during rendering
    def show(self, image, **options):
        return True
    def show_image(self, image, **options):
        return True
    def show_file(self, path, **options):
        return True


ImageShow.register(_NoOpViewer(), order=0)


class _EmbWrapper(torch.nn.Module):
    def __init__(self, model, n_num, n_cat):
        super().__init__()
        self.model = model
        self.n_num = int(n_num)
        self.n_cat = int(n_cat)

    def forward(self, x):
        x_num = x[:, :self.n_num]
        x_cat = x[:, self.n_num:self.n_num + self.n_cat].long()
        return self.model(x_num, x_cat)


class _MAEWrapper(torch.nn.Module):
    def __init__(self, model, seq_len, n_feat):
        super().__init__()
        self.model = model
        self.seq_len = int(seq_len)
        self.n_feat = int(n_feat)

    def forward(self, x):
        b = x.shape[0]
        x_ts = x[:, :self.seq_len].unsqueeze(-1)
        feats = x[:, self.seq_len:self.seq_len + self.n_feat] if self.n_feat > 0 else None
        mask = torch.zeros(b, self.seq_len, dtype=torch.bool)
        recon, emb = self.model(x_ts, mask, feats)
        return emb


def _build_models():
    models = {}
    # Inferred shapes from training setup in map/learning/tabular_nn/train_tabular_nn.py
    # VanillaMLP uses n_features; choose 32 here.
    models['vanilla_mlp'] = (VanillaMLP(n_features=32, hidden_dim=128, n_outputs=4, num_hidden_layers=3), (1, 32))

    # MLPWithEmbeddings takes (x_num, x_cat); wrap to accept a single tensor
    n_num = 16
    cat_cards = [8, 12, 5]
    mlp_emb = MLPWithEmbeddings(
        n_num_features=n_num,
        cat_cardinalities=cat_cards,
        embedding_dim=8,
        hidden_dim=128,
        n_outputs=4,
        num_hidden_layers=2,
    )
    emb_wrap = _EmbWrapper(mlp_emb, n_num=n_num, n_cat=len(cat_cards))
    models['mlp_with_embeddings'] = (emb_wrap, (1, n_num + len(cat_cards)))

    # VwcMAE expects (x, mask, feats); wrap to accept a single tensor
    seq_len = 128
    n_feat = 8
    mae = VwcMAE(seq_len=seq_len, d_model=128, n_heads=4, n_layers=3, emb_dim=64, n_feat=n_feat, n_channels=1)
    mae_wrap = _MAEWrapper(mae, seq_len=seq_len, n_feat=n_feat)
    models['vwc_mae'] = (mae_wrap, (1, seq_len + n_feat))
    return models


def render_nn_architectures(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    models = _build_models()
    for name, tup in models.items():
        model, input_shape = tup
        out_path = os.path.join(out_dir, f'{name}_architecture.png')
        img = layered_view(model, input_shape=input_shape)
        img.save(out_path)


if __name__ == '__main__':
    out_dir_ = os.path.join('poster', 'nn_architectures')
    render_nn_architectures(out_dir_)
# ========================= EOF ====================================================================
