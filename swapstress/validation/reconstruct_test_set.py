"""
Reconstruct the full test set for the global-pruned RF model.

The saved predictions.parquet has only (observed, predicted, source).
This module reproduces the exact spatial split by calling
``prepare_direct_data`` with the training-time ``resolution_m`` read from
``reprod.json``, then predicts with the saved model+imputer.

The result is cached as ``test_set_full.parquet`` so downstream consumers
never re-derive the split.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from map.learning.direct.data import prepare_direct_data

MODEL_DIR = "/nas/soils/swapstress/models/direct_rf_9km_global_pruned"

TEST_SET_CACHE = "test_set_full.parquet"

BECK_KOPPEN_TIF = "/nas/soils/swapstress/ancillary/Beck_KG_V1_present_0p0083.tif"

BECK_LABELS = {
    1: "Af",
    2: "Am",
    3: "Aw",
    4: "BWh",
    5: "BWk",
    6: "BSh",
    7: "BSk",
    8: "Csa",
    9: "Csb",
    10: "Csc",
    11: "Cwa",
    12: "Cwb",
    13: "Cwc",
    14: "Cfa",
    15: "Cfb",
    16: "Cfc",
    17: "Dsa",
    18: "Dsb",
    19: "Dsc",
    20: "Dsd",
    21: "Dwa",
    22: "Dwb",
    23: "Dwc",
    24: "Dwd",
    25: "Dfa",
    26: "Dfb",
    27: "Dfc",
    28: "Dfd",
    29: "ET",
    30: "EF",
}


def sample_beck_koppen(
    lat: np.ndarray,
    lon: np.ndarray,
    tif_path: str = BECK_KOPPEN_TIF,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample Beck et al. (2018) Koppen-Geiger at lat/lon points.

    Returns (code, label, major_zone) arrays, all length n.
    """
    import rasterio

    with rasterio.open(tif_path) as src:
        xy = list(zip(lon, lat))
        vals = list(src.sample(xy))
        codes = np.array([v[0] for v in vals], dtype=np.uint8)

    labels = np.array([BECK_LABELS.get(c, "unknown") for c in codes])
    major = np.array([lbl[0] if lbl != "unknown" else "unknown" for lbl in labels])
    return codes, labels, major


def _get_resolution_m(model_dir: str) -> float:
    """Read the training-time resolution_m from model artifacts.

    Checks (in order):
      1. direct_model_results.json  config.resolution_m
      2. reprod.json  training.spatial_holdout.resolution_m
    """
    model_path = Path(model_dir)

    # Primary: saved config (written by train_direct.py)
    results_path = model_path / "direct_model_results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        res = results.get("config", {}).get("resolution_m")
        if res is not None:
            return float(res)

    # Fallback: reprod.json (hand-written provenance)
    reprod_path = model_path / "reprod.json"
    if reprod_path.exists():
        with open(reprod_path) as f:
            reprod = json.load(f)
        res = reprod.get("training", {}).get("spatial_holdout", {}).get("resolution_m")
        if res is not None:
            return float(res)

    raise FileNotFoundError(
        f"Cannot determine resolution_m from {model_dir}: "
        "neither config.resolution_m in direct_model_results.json "
        "nor training.spatial_holdout.resolution_m in reprod.json found"
    )


def _build_test_set(model_dir: str) -> pd.DataFrame:
    """Reproduce the spatial split and predict with the saved model.

    Reads resolution_m from reprod.json, calls prepare_direct_data to
    reproduce the exact train/test split, then predicts with the saved
    model+imputer. The result is saved to ``<model_dir>/test_set_full.parquet``.
    """
    model_path = Path(model_dir)

    with open(model_path / "direct_model_results.json") as f:
        results = json.load(f)
    config = results["config"]
    n_expected = results["config"].get("n_test", 0)

    with open(model_path / "direct_rf_features.json") as f:
        all_features = json.load(f)

    resolution_m = _get_resolution_m(model_dir)
    print(f"Reproducing split with resolution_m={resolution_m}")

    data = prepare_direct_data(
        obs_table_path=config["obs_table"],
        output_dir="/tmp/reconstruct_scratch",
        exclude_groups=config.get("exclude_groups"),
        drop_blocking_features=config.get("drop_blocking_features", True),
        resolution_m=resolution_m,
        test_size=config.get("test_size", 0.2),
        random_state=config.get("random_state", 42),
    )

    test_df = data["test_df"].copy()
    if n_expected and len(test_df) != n_expected:
        raise ValueError(
            f"Reproduced split has {len(test_df)} test rows, "
            f"expected {n_expected} from config"
        )

    model = joblib.load(model_path / "direct_rf_model.joblib")
    imputer = joblib.load(model_path / "direct_rf_imputer.joblib")

    X = test_df[all_features].values.astype(np.float32)
    X = imputer.transform(X)
    test_df["predicted"] = model.predict(X)
    test_df["observed"] = test_df["log10_suction_cm"].values

    # Verify against saved metrics
    r2_ours = 1 - np.sum((test_df["observed"] - test_df["predicted"]) ** 2) / np.sum(
        (test_df["observed"] - test_df["observed"].mean()) ** 2
    )
    r2_saved = results["overall_metrics"]["r2"]
    print(f"Test set: {len(test_df)} rows, R2={r2_ours:.4f} (saved: {r2_saved:.4f})")
    if abs(r2_ours - r2_saved) > 0.001:
        print("WARNING: R2 mismatch — split may not be fully reproduced")

    cache_path = model_path / TEST_SET_CACHE
    test_df.to_parquet(cache_path, index=False)
    print(f"Cached to {cache_path}")

    return test_df


def _validate_cache(cache_path: Path, model_dir: str) -> pd.DataFrame | None:
    """Load and validate a cached test set. Returns None if stale."""
    model_path = Path(model_dir)
    results_path = model_path / "direct_model_results.json"
    if not results_path.exists():
        return None

    with open(results_path) as f:
        results = json.load(f)

    n_expected = results["config"].get("n_test", 0)
    r2_saved = results["overall_metrics"]["r2"]

    test_df = pd.read_parquet(cache_path)

    if n_expected and len(test_df) != n_expected:
        print(f"Cache stale: {len(test_df)} rows, expected {n_expected}. Rebuilding.")
        return None

    if "observed" not in test_df.columns or "predicted" not in test_df.columns:
        print("Cache stale: missing observed/predicted columns. Rebuilding.")
        return None

    r2_cache = 1 - np.sum((test_df["observed"] - test_df["predicted"]) ** 2) / np.sum(
        (test_df["observed"] - test_df["observed"].mean()) ** 2
    )
    if abs(r2_cache - r2_saved) > 0.001:
        print(f"Cache stale: R2={r2_cache:.4f} vs saved {r2_saved:.4f}. Rebuilding.")
        return None

    print(f"Loaded cached test set: {len(test_df)} rows from {cache_path}")
    return test_df


def reconstruct(model_dir: str = MODEL_DIR) -> pd.DataFrame:
    """Return the test DataFrame with observed, predicted, and all metadata.

    On first call, reproduces the spatial split using the training-time
    resolution_m and saves the result to ``<model_dir>/test_set_full.parquet``.
    Subsequent calls load and validate the cache (row count and R2 must match
    the saved model artifacts); a stale cache triggers a rebuild.

    Returns
    -------
    pd.DataFrame
        Test-set rows with columns: observed, predicted, theta, depth_cm,
        source, sample_id, lat, lon, plus all feature columns.
    """
    cache_path = Path(model_dir) / TEST_SET_CACHE
    if cache_path.exists():
        cached = _validate_cache(cache_path, model_dir)
        if cached is not None:
            return cached

    return _build_test_set(model_dir)


if __name__ == "__main__":
    # Force rebuild (delete cache first).
    cache = Path(MODEL_DIR) / TEST_SET_CACHE
    if cache.exists():
        cache.unlink()
        print(f"Deleted existing cache {cache}")

    test = reconstruct()
    print(f"Rows: {len(test)}")
    print(f"Columns: {test.columns.tolist()[:15]}...")
