"""Tests for the tabular NN components.

Split out of tests/test_direct_shared.py when map/learning/tabular_nn moved to
research/neural/. Off the reproduction path and not collected by the default
pytest run -- see research/README.md.
"""

import json

import numpy as np
import pytest
import torch

from research.neural.tabular_nn.models import (
    VanillaMLP,
    MLPWithEmbeddings,
    FTTransformer,
)
from research.neural.tabular_nn.dataset import FlatDataset, SplitDataset
from research.neural.tabular_nn.lightning_module import DirectRegressionModule
from research.neural.tabular_nn.compare_runs import build_comparison_table


class TestModels:
    def test_vanilla_mlp_forward(self):
        model = VanillaMLP(n_features=10, hidden_dim=32, num_hidden_layers=2)
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 1)

    def test_mlp_embeddings_forward(self):
        model = MLPWithEmbeddings(
            n_num_features=8,
            cat_cardinalities=[10, 5],
            embedding_dim=4,
            hidden_dim=32,
            num_hidden_layers=2,
        )
        x_num = torch.randn(4, 8)
        x_cat = torch.randint(0, 5, (4, 2))
        out = model(x_num, x_cat)
        assert out.shape == (4, 1)

    def test_mlp_embeddings_no_cats(self):
        model = MLPWithEmbeddings(
            n_num_features=8,
            cat_cardinalities=[],
            hidden_dim=32,
            num_hidden_layers=2,
        )
        x_num = torch.randn(4, 8)
        x_cat = torch.zeros(4, 0, dtype=torch.long)
        out = model(x_num, x_cat)
        assert out.shape == (4, 1)

    def test_ft_transformer_forward(self):
        model = FTTransformer(
            n_num_features=8,
            cat_cardinalities=[10, 5],
            d_token=32,
            n_blocks=2,
            n_heads=4,
        )
        x_num = torch.randn(4, 8)
        x_cat = torch.randint(0, 5, (4, 2))
        out = model(x_num, x_cat)
        assert out.shape == (4, 1)

    def test_ft_transformer_no_cats(self):
        model = FTTransformer(
            n_num_features=8,
            cat_cardinalities=[],
            d_token=32,
            n_blocks=2,
            n_heads=4,
        )
        x_num = torch.randn(4, 8)
        x_cat = torch.zeros(4, 0, dtype=torch.long)
        out = model(x_num, x_cat)
        assert out.shape == (4, 1)


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------


class TestDatasets:
    def test_flat_dataset(self):
        X = np.random.rand(20, 5).astype(np.float32)
        y = np.random.rand(20).astype(np.float32)
        ds = FlatDataset(X, y)
        assert len(ds) == 20
        x_, y_ = ds[0]
        assert x_.shape == (5,)
        assert y_.shape == (1,)

    def test_split_dataset(self):
        X_num = np.random.rand(20, 5).astype(np.float32)
        X_cat = np.random.randint(0, 3, (20, 2))
        y = np.random.rand(20).astype(np.float32)
        ds = SplitDataset(X_num, X_cat, y)
        assert len(ds) == 20
        (xn, xc), y_ = ds[0]
        assert xn.shape == (5,)
        assert xc.shape == (2,)


# ---------------------------------------------------------------------------
# Lightning module test
# ---------------------------------------------------------------------------


class TestLightningModule:
    def test_training_step(self):
        model = VanillaMLP(n_features=5, hidden_dim=16, num_hidden_layers=1)
        lit = DirectRegressionModule(model, split_input=False)
        x = torch.randn(4, 5)
        y = torch.randn(4, 1)
        loss = lit.training_step((x, y), 0)
        assert loss.dim() == 0

    def test_split_training_step(self):
        model = MLPWithEmbeddings(
            n_num_features=5,
            cat_cardinalities=[3],
            embedding_dim=4,
            hidden_dim=16,
            num_hidden_layers=1,
        )
        lit = DirectRegressionModule(model, split_input=True)
        x_num = torch.randn(4, 5)
        x_cat = torch.randint(0, 3, (4, 1))
        y = torch.randn(4, 1)
        loss = lit.training_step(((x_num, x_cat), y), 0)
        assert loss.dim() == 0

    def test_configure_optimizers_without_scheduler(self):
        model = VanillaMLP(n_features=5, hidden_dim=16, num_hidden_layers=1)
        lit = DirectRegressionModule(
            model,
            split_input=False,
            scheduler_monitor=None,
        )
        optimizers = lit.configure_optimizers()
        assert "optimizer" in optimizers
        assert "lr_scheduler" not in optimizers

    def test_bound_penalty_zero_when_in_range(self):
        model = VanillaMLP(n_features=5, hidden_dim=16, num_hidden_layers=1)
        lit = DirectRegressionModule(
            model,
            split_input=False,
            lambda_bound=1.0,
            bound_lo=0.0,
            bound_hi=7.0,
        )
        y_hat = torch.tensor([[1.0], [3.5], [6.9]])
        penalty = lit._bound_penalty(y_hat)
        assert penalty.item() == pytest.approx(0.0)

    def test_bound_penalty_positive_when_out_of_range(self):
        model = VanillaMLP(n_features=5, hidden_dim=16, num_hidden_layers=1)
        lit = DirectRegressionModule(
            model,
            split_input=False,
            lambda_bound=1.0,
            bound_lo=0.0,
            bound_hi=7.0,
        )
        # One prediction below 0, one above 7
        y_hat = torch.tensor([[-1.0], [8.0], [3.0]])
        penalty = lit._bound_penalty(y_hat)
        # relu(-1 - 7)=0, relu(8 - 7)=1, relu(3 - 7)=0 -> mean = 1/3
        # relu(0 - (-1))=1, relu(0 - 8)=0, relu(0 - 3)=0 -> mean = 1/3
        assert penalty.item() == pytest.approx(1.0 / 3.0 + 1.0 / 3.0)

    def test_bound_penalty_in_training_step(self):
        model = VanillaMLP(n_features=5, hidden_dim=16, num_hidden_layers=1)
        lit = DirectRegressionModule(
            model,
            split_input=False,
            lambda_bound=0.1,
            bound_lo=0.0,
            bound_hi=7.0,
        )
        x = torch.randn(4, 5)
        y = torch.randn(4, 1)
        loss_with = lit.training_step((x, y), 0)
        assert loss_with.dim() == 0

    def test_mono_penalty_flat_input(self):
        """Monotonicity penalty runs on VanillaMLP (flat input)."""
        model = VanillaMLP(n_features=5, hidden_dim=16, num_hidden_layers=1)
        # theta is at index 3 in the 5-feature input
        lit = DirectRegressionModule(
            model,
            split_input=False,
            lambda_mono=0.1,
            theta_idx=3,
        )
        x = torch.randn(4, 5)
        y = torch.randn(4, 1)
        loss = lit.training_step((x, y), 0)
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_mono_penalty_split_input(self):
        """Monotonicity penalty runs on MLPWithEmbeddings (split input)."""
        model = MLPWithEmbeddings(
            n_num_features=5,
            cat_cardinalities=[3],
            embedding_dim=4,
            hidden_dim=16,
            num_hidden_layers=1,
        )
        # theta is at index 2 in the 5-column numeric tensor
        lit = DirectRegressionModule(
            model,
            split_input=True,
            lambda_mono=0.5,
            theta_idx=2,
        )
        x_num = torch.randn(4, 5)
        x_cat = torch.randint(0, 3, (4, 1))
        y = torch.randn(4, 1)
        loss = lit.training_step(((x_num, x_cat), y), 0)
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_mono_penalty_off_when_zero(self):
        """lambda_mono=0 should use the regular _step path, not _mono_step."""
        model = VanillaMLP(n_features=5, hidden_dim=16, num_hidden_layers=1)
        lit = DirectRegressionModule(
            model,
            split_input=False,
            lambda_mono=0.0,
            theta_idx=3,
        )
        x = torch.randn(4, 5)
        y = torch.randn(4, 1)
        loss = lit.training_step((x, y), 0)
        assert loss.dim() == 0

    def test_mono_and_bound_combined(self):
        """Both penalties active simultaneously."""
        model = VanillaMLP(n_features=5, hidden_dim=16, num_hidden_layers=1)
        lit = DirectRegressionModule(
            model,
            split_input=False,
            lambda_mono=0.1,
            theta_idx=3,
            lambda_bound=0.1,
            bound_lo=0.0,
            bound_hi=7.0,
        )
        x = torch.randn(4, 5)
        y = torch.randn(4, 1)
        loss = lit.training_step((x, y), 0)
        assert loss.dim() == 0
        assert loss.requires_grad


# ---------------------------------------------------------------------------
# Compare runs test
# ---------------------------------------------------------------------------


class TestCompareRuns:
    def test_build_comparison_table(self, tmp_path):
        # Create two fake run directories
        for name, r2 in [("rf_run", 0.55), ("nn_run", 0.52)]:
            d = tmp_path / name
            d.mkdir()
            results = {
                "overall_metrics": {
                    "r2": r2,
                    "rmse": 0.3,
                    "mae": 0.2,
                    "bias": 0.01,
                    "n": 1000,
                },
                "site_weighted_metrics": {
                    "mean_r2": r2 - 0.05,
                    "median_r2": r2,
                    "mean_rmse": 0.31,
                    "n_sites": 50,
                },
                "source_metrics": [],
                "feature_importance": None,
                "config": {"n_features": 30, "n_train": 800, "n_test": 200},
            }
            if name == "nn_run":
                results["model_family"] = "nn"
                results["model_name"] = "mlp"
                results["validation_metrics"] = {"val_rmse": 0.28}
            (d / "direct_model_results.json").write_text(json.dumps(results))

        df = build_comparison_table(
            [str(tmp_path / "rf_run"), str(tmp_path / "nn_run")]
        )
        assert len(df) == 2
        assert set(df["run"]) == {"rf_run", "nn_run"}
        assert df.loc[df["run"] == "rf_run", "r2"].iloc[0] == 0.55


# ---------------------------------------------------------------------------
# Manifest interop regression tests
# ---------------------------------------------------------------------------


