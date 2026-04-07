"""
Neural network architectures for tabular regression.

Three model families:
- VanillaMLP: flat numeric input (one-hot categoricals)
- MLPWithEmbeddings: separate numeric + categorical (learned embeddings)
- FTTransformer: feature tokenizer + transformer encoder
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SafeBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm that passes through single-sample batches unchanged."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.size(0) == 1:
            return x
        return super().forward(x)


def _mlp_block(
    in_dim: int,
    out_dim: int,
    dropout: float = 0.2,
    activation: str = "relu",
) -> nn.Sequential:
    act = nn.GELU() if activation == "gelu" else nn.ReLU()
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        act,
        _SafeBatchNorm1d(out_dim),
        nn.Dropout(dropout),
    )


# ---------------------------------------------------------------------------
# VanillaMLP
# ---------------------------------------------------------------------------


class VanillaMLP(nn.Module):
    """Feed-forward MLP for tabular regression.

    Input is a single flat tensor of numeric features (categorical features
    are expected to be one-hot encoded upstream).

    Parameters
    ----------
    n_features : int
        Input dimension.
    hidden_dim : int
        Width of hidden layers.
    num_hidden_layers : int
        Number of hidden blocks.
    dropout : float
        Dropout probability.
    n_outputs : int
        Output dimension (1 for single-target regression).
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 3,
        dropout: float = 0.2,
        n_outputs: int = 1,
    ):
        super().__init__()
        layers: list[nn.Module] = [_mlp_block(n_features, hidden_dim, dropout)]
        dim = hidden_dim
        for _ in range(num_hidden_layers - 1):
            next_dim = max(dim // 2, 32)
            layers.append(_mlp_block(dim, next_dim, dropout))
            dim = next_dim
        layers.append(nn.Linear(dim, n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# MLPWithEmbeddings
# ---------------------------------------------------------------------------


class MLPWithEmbeddings(nn.Module):
    """MLP with learned embeddings for categorical features.

    Parameters
    ----------
    n_num_features : int
        Number of continuous numeric features.
    cat_cardinalities : list of int
        Vocabulary size for each categorical feature (including unknown bucket).
    embedding_dim : int
        Embedding dimension per categorical feature.
    hidden_dim : int
        Width of hidden layers.
    num_hidden_layers : int
        Number of hidden blocks.
    dropout : float
        Dropout probability.
    n_outputs : int
        Output dimension.
    """

    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: list[int],
        embedding_dim: int = 16,
        hidden_dim: int = 256,
        num_hidden_layers: int = 3,
        dropout: float = 0.2,
        n_outputs: int = 1,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, embedding_dim) for card in cat_cardinalities]
        )
        total_cat_dim = len(cat_cardinalities) * embedding_dim
        input_dim = n_num_features + total_cat_dim

        layers: list[nn.Module] = [_mlp_block(input_dim, hidden_dim, dropout)]
        dim = hidden_dim
        for _ in range(num_hidden_layers - 1):
            next_dim = max(dim // 2, 32)
            layers.append(_mlp_block(dim, next_dim, dropout))
            dim = next_dim
        layers.append(nn.Linear(dim, n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        if len(self.embeddings) > 0:
            cat_embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            x = torch.cat([x_num] + cat_embeds, dim=1)
        else:
            x = x_num
        return self.net(x)


# ---------------------------------------------------------------------------
# FT-Transformer (in-repo implementation)
# ---------------------------------------------------------------------------


class _NumericTokenizer(nn.Module):
    """Project each numeric feature to a d_token-dimensional token."""

    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features)
        # out: (batch, n_features, d_token)
        return x.unsqueeze(2) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class _TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, attn_dropout: float, ff_dropout: float
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4 // 3),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(d_model * 4 // 3, d_model),
            nn.Dropout(ff_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


class FTTransformer(nn.Module):
    """Feature Tokenizer + Transformer for tabular regression.

    Each numeric feature is projected to a d_token-dimensional token via a
    learned linear map.  Categorical features get embedding tables.  A
    learnable [CLS] token is prepended and used for the final prediction.

    Parameters
    ----------
    n_num_features : int
        Number of numeric features.
    cat_cardinalities : list of int
        Vocabulary size per categorical feature.
    d_token : int
        Token embedding dimension.
    n_blocks : int
        Number of transformer blocks.
    n_heads : int
        Number of attention heads.
    attn_dropout : float
        Attention dropout.
    ff_dropout : float
        Feed-forward dropout.
    n_outputs : int
        Output dimension.
    """

    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: list[int],
        d_token: int = 192,
        n_blocks: int = 3,
        n_heads: int = 8,
        attn_dropout: float = 0.2,
        ff_dropout: float = 0.1,
        n_outputs: int = 1,
    ):
        super().__init__()
        self.num_tokenizer = _NumericTokenizer(n_num_features, d_token)
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(card, d_token) for card in cat_cardinalities]
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))

        self.blocks = nn.ModuleList(
            [
                _TransformerBlock(d_token, n_heads, attn_dropout, ff_dropout)
                for _ in range(n_blocks)
            ]
        )
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, n_outputs)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        tokens = [self.num_tokenizer(x_num)]

        for i, emb in enumerate(self.cat_embeddings):
            tokens.append(emb(x_cat[:, i]).unsqueeze(1))

        cls = self.cls_token.expand(x_num.size(0), -1, -1)
        x = torch.cat([cls] + tokens, dim=1)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x[:, 0])  # CLS token
        return self.head(x)
