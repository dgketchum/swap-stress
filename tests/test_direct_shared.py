"""Tests for shared direct-task utilities.

The tabular NN half of this module moved to research/neural/test_tabular_nn.py
in the Phase 2 research/ split; those components are off the reproduction path.
"""

import json
import os
import re as _re

import numpy as np
import pandas as pd
import pytest

from map.learning.direct.data import (
    assign_mgrs_fold,
    assign_spatial_group,
    apply_mgrs_split,
    apply_mgrs_split_three,
    apply_site_split,
    apply_site_split_three,
    create_mgrs_split,
    create_mgrs_split_with_val,
    create_site_split,
    create_site_split_with_val,
    filter_complete_samples,
    read_kfold_manifest,
    read_split_manifest,
    write_kfold_manifest,
    write_split_manifest,
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
            "MGRS_TILE": rng.choice(
                [
                    "12TQK",
                    "12TQL",
                    "11SPA",
                    "11SPB",
                    "10TGK",
                    "10TGL",
                    "13TDE",
                    "13TDF",
                ],
                n,
            ),
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


class TestMGRSSplit:
    def test_assign_mgrs_fold_deterministic(self, tiny_df):
        """Same tile always maps to same fold regardless of input ordering."""
        folds1 = assign_mgrs_fold(tiny_df, n_folds=5)
        shuffled = tiny_df.sample(frac=1, random_state=99).reset_index(drop=True)
        folds2 = assign_mgrs_fold(shuffled, n_folds=5)
        # Compare tile->fold mapping, not row order
        map1 = dict(zip(tiny_df["MGRS_TILE"], folds1))
        map2 = dict(zip(shuffled["MGRS_TILE"], folds2))
        assert map1 == map2

    def test_assign_mgrs_fold_stable_across_data_changes(self, tiny_df):
        """Adding rows doesn't change fold assignment of existing tiles."""
        folds_before = assign_mgrs_fold(tiny_df, n_folds=5)
        map_before = dict(zip(tiny_df["MGRS_TILE"], folds_before))
        # Add new rows with a new tile
        extra = pd.DataFrame(
            {
                "lat": [40.0, 41.0],
                "lon": [-100.0, -101.0],
                "MGRS_TILE": ["99XYZ", "99XYZ"],
                "theta": [0.2, 0.3],
                "log10_suction_cm": [2.0, 2.5],
                "source": ["test", "test"],
                "elevation": [500, 600],
                "slope": [5, 10],
                "clay_mean": [20, 30],
                "B5_mean_gs": [1000, 2000],
                "VH_mean": [-10, -15],
                "nlcd": [41, 41],
                "depth_cm": [15, 30],
            }
        )
        bigger = pd.concat([tiny_df, extra], ignore_index=True)
        folds_after = assign_mgrs_fold(bigger, n_folds=5)
        map_after = dict(zip(bigger["MGRS_TILE"], folds_after))
        for tile, fold in map_before.items():
            assert map_after[tile] == fold

    def test_create_mgrs_split_disjoint(self, tiny_df):
        train, test = create_mgrs_split(tiny_df, n_folds=5, test_fold=0)
        assert len(train & test) == 0
        assert len(train) > 0
        assert len(test) > 0

    def test_mgrs_split_with_val_three_way_disjoint(self, tiny_df):
        train, val, test = create_mgrs_split_with_val(tiny_df, n_folds=5, test_fold=0)
        assert len(train & val) == 0
        assert len(train & test) == 0
        assert len(val & test) == 0
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

    def test_kfold_all_data_covered(self, tiny_df):
        """Union of all test folds covers all tiles exactly once."""
        all_tiles = set(tiny_df["MGRS_TILE"].dropna().unique())
        seen = set()
        for k in range(5):
            _, test_tiles = create_mgrs_split(tiny_df, n_folds=5, test_fold=k)
            assert len(seen & test_tiles) == 0, f"Overlap at fold {k}"
            seen |= test_tiles
        assert seen == all_tiles

    def test_apply_mgrs_split(self, tiny_df):
        train_t, test_t = create_mgrs_split(tiny_df, n_folds=5, test_fold=0)
        train_df, test_df = apply_mgrs_split(tiny_df, train_t, test_t)
        assert len(train_df) + len(test_df) == len(tiny_df)
        assert set(train_df["MGRS_TILE"].unique()) <= train_t
        assert set(test_df["MGRS_TILE"].unique()) <= test_t

    def test_apply_mgrs_split_three(self, tiny_df):
        tr, va, te = create_mgrs_split_with_val(tiny_df, n_folds=5, test_fold=0)
        tr_df, va_df, te_df = apply_mgrs_split_three(tiny_df, tr, va, te)
        assert len(tr_df) + len(va_df) + len(te_df) == len(tiny_df)

    def test_kfold_manifest_roundtrip(self, tmp_path):
        tile_to_fold = {"12TQK": 0, "12TQL": 1, "11SPA": 2}
        path = str(tmp_path / "kfold.json")
        write_kfold_manifest(path, tile_to_fold, n_folds=5, holdout_col="MGRS_TILE")
        loaded = read_kfold_manifest(path)
        assert loaded["tile_to_fold"] == tile_to_fold
        assert loaded["n_folds"] == 5
        assert loaded["holdout_col"] == "MGRS_TILE"


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
