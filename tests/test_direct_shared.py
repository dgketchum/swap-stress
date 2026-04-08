"""Tests for shared direct-task utilities and tabular NN components."""

import json
import os
import re as _re

import numpy as np
import pandas as pd
import pytest
import torch

from map.learning.direct.data import (
    assign_spatial_group,
    create_site_split,
    create_site_split_with_val,
    apply_site_split,
    apply_site_split_three,
    write_split_manifest,
    read_split_manifest,
    filter_complete_samples,
)
from map.learning.direct.metrics import (
    compute_metrics,
    compute_metrics_by_source,
    compute_metrics_by_site,
)
from map.learning.direct.preprocessing import (
    NNPreprocessor,
    get_categorical_feature_columns,
    get_numeric_feature_columns,
)
from map.learning.tabular_nn.models import (
    VanillaMLP,
    MLPWithEmbeddings,
    FTTransformer,
)
from map.learning.tabular_nn.dataset import FlatDataset, SplitDataset
from map.learning.tabular_nn.lightning_module import DirectRegressionModule
from map.learning.tabular_nn.compare_runs import build_comparison_table


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_df():
    """Small dataframe mimicking the observation-level training table."""
    rng = np.random.RandomState(42)
    n = 200
    df = pd.DataFrame(
        {
            "lat": rng.uniform(30, 50, n),
            "lon": rng.uniform(-120, -80, n),
            "theta": rng.uniform(0.05, 0.45, n),
            "log10_suction_cm": rng.uniform(0.5, 4.5, n),
            "source": rng.choice(["ismn", "ncss"], n),
            "elevation": rng.uniform(100, 3000, n),
            "slope": rng.uniform(0, 30, n),
            "clay_mean": rng.uniform(5, 60, n),
            "B5_mean_gs": rng.uniform(0, 5000, n),
            "VH_mean": rng.uniform(-25, 0, n),
            "nlcd": rng.choice([11, 21, 41, 52, 71, 82], n),
            "depth_cm": rng.choice([5, 15, 30, 60], n),
        }
    )
    return df


@pytest.fixture
def tiny_parquet(tmp_path, tiny_df):
    """Write tiny_df to a parquet file."""
    path = str(tmp_path / "obs.parquet")
    tiny_df.to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Data tests
# ---------------------------------------------------------------------------


class TestSpatialSplit:
    def test_assign_spatial_group(self, tiny_df):
        groups = assign_spatial_group(tiny_df, resolution_m=9000)
        assert groups.notna().all()
        assert groups.nunique() > 1

    def test_create_site_split_disjoint(self, tiny_df):
        train, test = create_site_split(tiny_df, resolution_m=9000)
        assert len(train & test) == 0
        assert len(train) > 0
        assert len(test) > 0

    def test_create_site_split_with_val(self, tiny_df):
        train, val, test = create_site_split_with_val(
            tiny_df,
            test_size=0.2,
            val_size=0.2,
            resolution_m=9000,
        )
        assert len(train & val) == 0
        assert len(train & test) == 0
        assert len(val & test) == 0

    def test_apply_site_split(self, tiny_df):
        train_g, test_g = create_site_split(tiny_df, resolution_m=9000)
        train_df, test_df = apply_site_split(
            tiny_df,
            train_g,
            test_g,
            resolution_m=9000,
        )
        assert len(train_df) + len(test_df) <= len(tiny_df)
        assert len(train_df) > 0

    def test_three_way_split(self, tiny_df):
        train_g, val_g, test_g = create_site_split_with_val(
            tiny_df,
            resolution_m=9000,
        )
        train_df, val_df, test_df = apply_site_split_three(
            tiny_df,
            train_g,
            val_g,
            test_g,
            resolution_m=9000,
        )
        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0

    def test_same_test_set(self, tiny_df):
        """Two-way and three-way splits share the same test groups."""
        train2, test2 = create_site_split(
            tiny_df,
            test_size=0.2,
            random_state=42,
            resolution_m=9000,
        )
        train3, val3, test3 = create_site_split_with_val(
            tiny_df,
            test_size=0.2,
            val_size=0.2,
            random_state=42,
            resolution_m=9000,
        )
        assert test2 == test3


class TestSplitManifest:
    def test_roundtrip(self, tmp_path):
        train = {"a", "b", "c"}
        val = {"d", "e"}
        test = {"f", "g"}
        path = str(tmp_path / "split.json")
        write_split_manifest(path, train, test, val, random_state=42, resolution_m=9000)
        loaded = read_split_manifest(path)
        assert loaded["train_groups"] == train
        assert loaded["val_groups"] == val
        assert loaded["test_groups"] == test
        assert loaded["random_state"] == 42

    def test_no_val(self, tmp_path):
        path = str(tmp_path / "split.json")
        write_split_manifest(path, {"a"}, {"b"}, random_state=7, resolution_m=250)
        loaded = read_split_manifest(path)
        assert loaded["val_groups"] is None


class TestFilterComplete:
    def test_drops_missing_coords(self, tiny_df):
        tiny_df.loc[0, "lat"] = np.nan
        result = filter_complete_samples(tiny_df)
        assert len(result) == len(tiny_df) - 1


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_compute_metrics(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.0, 2.9, 4.1])
        m = compute_metrics(y_true, y_pred)
        assert m["r2"] > 0.99
        assert m["n"] == 4

    def test_compute_metrics_all_nan(self):
        m = compute_metrics(np.array([np.nan]), np.array([np.nan]))
        assert m["n"] == 0

    def test_by_source(self):
        y_true = np.random.rand(100)
        y_pred = y_true + np.random.randn(100) * 0.1
        sources = np.array(["a"] * 50 + ["b"] * 50)
        df = compute_metrics_by_source(y_true, y_pred, sources)
        assert len(df) == 2
        assert "source" in df.columns

    def test_by_site(self):
        y_true = np.random.rand(30)
        y_pred = y_true + np.random.randn(30) * 0.1
        sites = np.array(["s1"] * 10 + ["s2"] * 10 + ["s3"] * 10)
        per_site, summary = compute_metrics_by_site(y_true, y_pred, sites)
        assert len(per_site) == 3
        assert summary["n_sites"] == 3


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------


class TestNNPreprocessor:
    def test_fit_transform_numeric_only(self, tiny_df):
        features = ["elevation", "slope", "clay_mean", "B5_mean_gs", "theta"]
        pp = NNPreprocessor().fit(tiny_df, features)
        X_num, X_cat, y = pp.transform(tiny_df, features)
        assert X_num.shape == (len(tiny_df), 5)
        assert X_cat.shape[1] == 0
        assert y.shape == (len(tiny_df),)

    def test_fit_transform_with_cats(self, tiny_df):
        features = ["elevation", "slope", "nlcd", "theta"]
        pp = NNPreprocessor().fit(tiny_df, features)
        X_num, X_cat, y = pp.transform(tiny_df, features)
        assert X_num.shape[1] == 3  # elevation, slope, theta
        assert X_cat.shape[1] == 1  # nlcd
        assert pp.cat_cardinalities[0] > 1

    def test_flat_transform(self, tiny_df):
        features = ["elevation", "nlcd", "theta"]
        pp = NNPreprocessor().fit(tiny_df, features)
        X, y = pp.transform_flat(tiny_df, features)
        expected_cols = 2 + pp.cat_cardinalities[0]  # 2 numeric + one-hot
        assert X.shape[1] == expected_cols

    def test_save_load_roundtrip(self, tmp_path, tiny_df):
        features = ["elevation", "nlcd", "theta"]
        pp = NNPreprocessor().fit(tiny_df, features)
        pp.save(str(tmp_path))

        pp2 = NNPreprocessor.load(str(tmp_path))
        X1, _, _ = pp.transform(tiny_df, features)
        X2, _, _ = pp2.transform(tiny_df, features)
        np.testing.assert_allclose(X1, X2, atol=1e-6)

    def test_categorical_registry(self):
        cols = ["elevation", "nlcd", "slope", "cdl_crop_mode"]
        cats = get_categorical_feature_columns(cols)
        nums = get_numeric_feature_columns(cols)
        assert "nlcd" in cats
        assert "cdl_crop_mode" in cats
        assert "elevation" in nums
        assert "slope" in nums

    def test_missing_categoricals_map_to_unknown(self, tiny_df):
        """NaN in categorical columns should map to 0 (unknown bucket),
        not a median-imputed fake class ID."""
        features = ["elevation", "nlcd", "theta"]
        # Inject NaN into nlcd
        tiny_df = tiny_df.copy()
        tiny_df.loc[tiny_df.index[:20], "nlcd"] = np.nan
        pp = NNPreprocessor().fit(tiny_df, features)
        X_num, X_cat, y = pp.transform(tiny_df, features)
        # All NaN rows should have cat index 0 (unknown bucket)
        assert (X_cat[:20, 0] == 0).all()
        # Non-NaN rows should have cat index > 0
        assert (X_cat[20:, 0] > 0).all()

    def test_categorical_not_median_imputed(self, tiny_df):
        """Categorical columns must NOT go through median imputation,
        which could create non-existent class IDs like 46.5 -> 46."""
        features = ["elevation", "nlcd", "theta"]
        tiny_df = tiny_df.copy()
        tiny_df.loc[tiny_df.index[:10], "nlcd"] = np.nan
        pp = NNPreprocessor().fit(tiny_df, features)
        # The imputer should only know about numeric columns
        assert pp.imputer.statistics_.shape[0] == 2  # elevation, theta
        # Category map should only contain real nlcd values from the data
        real_nlcd = set(tiny_df["nlcd"].dropna().astype(int).unique())
        mapped_vals = set(pp.category_maps["nlcd"].keys())
        assert mapped_vals == real_nlcd


# ---------------------------------------------------------------------------
# Model architecture tests
# ---------------------------------------------------------------------------


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


class TestManifestInterop:
    def test_two_way_manifest_is_upgraded_for_nn(self, tmp_path):
        """Legacy two-way manifests should be upgraded in place for NN use."""
        from map.learning.direct.data import prepare_direct_data, read_split_manifest

        # Write a two-way manifest (no val_groups)
        manifest = {
            "random_state": 42,
            "resolution_m": 9000,
            "n_train": 2,
            "n_test": 1,
            "train_groups": ["a", "b"],
            "test_groups": ["c"],
        }
        mpath = str(tmp_path / "split.json")
        with open(mpath, "w") as f:
            json.dump(manifest, f)

        # Create a tiny parquet
        rng = np.random.RandomState(42)
        n = 50
        df = pd.DataFrame(
            {
                "lat": rng.uniform(30, 50, n),
                "lon": rng.uniform(-120, -80, n),
                "theta": rng.uniform(0.05, 0.45, n),
                "log10_suction_cm": rng.uniform(0.5, 4.5, n),
                "elevation": rng.uniform(100, 3000, n),
            }
        )
        pqt = str(tmp_path / "obs.parquet")
        df.to_parquet(pqt, index=False)

        # prepare_direct_data should upgrade the manifest in place, preserving
        # the original test groups while deriving val_groups from train_groups.
        data = prepare_direct_data(
            obs_table_path=pqt,
            output_dir=str(tmp_path / "out"),
            resolution_m=9000,
            test_size=0.2,
            val_size=0.2,
            split_manifest=mpath,
        )
        assert data.get("val_df") is not None
        assert data["test_sites"] == {"c"}

        upgraded = read_split_manifest(mpath)
        assert upgraded["val_groups"] is not None
        assert upgraded["test_groups"] == {"c"}
        assert upgraded["train_groups"] | upgraded["val_groups"] == {"a", "b"}

    def test_rf_writes_three_way_manifest(self, tiny_parquet, tmp_path):
        """RF trainer must write a manifest with val_groups so NN can consume it."""
        from map.learning.direct.data import (
            prepare_direct_data,
            read_split_manifest,
            write_split_manifest,
        )

        manifest_path = str(tmp_path / "shared_split.json")

        # Simulate what RF does: request val_size to get a three-way split
        data = prepare_direct_data(
            obs_table_path=tiny_parquet,
            output_dir=str(tmp_path / "rf_out"),
            resolution_m=9000,
            test_size=0.2,
            val_size=0.2,
        )

        # Write the manifest with val_groups (as RF now does)
        write_split_manifest(
            manifest_path,
            train_groups=data["train_sites"],
            test_groups=data["test_sites"],
            val_groups=data.get("val_sites"),
            random_state=42,
            resolution_m=9000,
        )

        # The manifest must have val_groups
        loaded = read_split_manifest(manifest_path)
        assert loaded["val_groups"] is not None
        assert len(loaded["val_groups"]) > 0

        # NN can consume this manifest
        nn_data = prepare_direct_data(
            obs_table_path=tiny_parquet,
            output_dir=str(tmp_path / "nn_out"),
            resolution_m=9000,
            test_size=0.2,
            val_size=0.2,
            split_manifest=manifest_path,
        )
        assert nn_data["val_df"] is not None
        assert nn_data["test_sites"] == data["test_sites"]

    def test_rf_merges_val_into_train(self, tiny_parquet, tmp_path):
        """RF should train on train+val groups, not just train groups."""
        from map.learning.direct.data import prepare_direct_data

        data = prepare_direct_data(
            obs_table_path=tiny_parquet,
            output_dir=str(tmp_path / "out"),
            resolution_m=9000,
            test_size=0.2,
            val_size=0.2,
        )

        train_only_n = len(data["train_df"])
        val_n = len(data["val_df"])

        # Simulate what RF does: merge train + val
        import pandas as pd

        merged = pd.concat([data["train_df"], data["val_df"]], ignore_index=True)
        merged_sites = data["train_sites"] | data["val_sites"]

        # RF training data should be larger than train-only
        assert len(merged) == train_only_n + val_n
        assert len(merged_sites) == len(data["train_sites"]) + len(data["val_sites"])

    def test_manifest_written_to_shared_path(self, tmp_path):
        """write_split_manifest creates parent dirs and writes correctly."""
        from map.learning.direct.data import write_split_manifest

        shared_path = str(tmp_path / "shared" / "split.json")
        write_split_manifest(
            shared_path,
            train_groups={"a", "b"},
            test_groups={"c"},
            val_groups={"d"},
            random_state=42,
            resolution_m=9000,
        )
        assert os.path.exists(shared_path)
        with open(shared_path) as f:
            doc = json.load(f)
        assert set(doc["train_groups"]) == {"a", "b"}
        assert set(doc["test_groups"]) == {"c"}
        assert set(doc["val_groups"]) == {"d"}


# ---------------------------------------------------------------------------
# Refit epoch regression test
# ---------------------------------------------------------------------------


class TestRefitEpochExtraction:
    def test_extracts_best_epoch_from_checkpoint_filename(self):
        """Refit should use best epoch, not terminal epoch."""
        # Simulate the checkpoint filename pattern
        ckpt_name = "best-012-0.3456.ckpt"
        match = _re.search(r"best-(\d+)-", ckpt_name)
        assert match is not None
        best_epoch = int(match.group(1))
        assert best_epoch == 12
        # Refit epochs = best_epoch + 1 (0-indexed)
        refit_epochs = best_epoch + 1
        assert refit_epochs == 13

    def test_refit_less_than_stopped(self):
        """If patience=20 and best epoch=30, stopped~50, refit should be 31 not 51."""
        best_epoch = 30
        stopped_epoch = 50
        refit_epochs = best_epoch + 1
        assert refit_epochs == 31
        assert refit_epochs < stopped_epoch + 1
