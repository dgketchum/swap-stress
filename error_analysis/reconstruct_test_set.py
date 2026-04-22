"""
Reconstruct the full test set for the global-pruned RF model.

The saved predictions.parquet has only (observed, predicted, source).
This module re-runs prepare_direct_data() with the same config to recover
theta, depth_cm, lat, lon, spatial_group, and all features for the test set.
It also loads the trained model to regenerate predictions in the same row order.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from map.learning.direct.data import assign_spatial_group, prepare_direct_data

MODEL_DIR = "/nas/soils/swapstress/models/direct_rf_9km_global_pruned"

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


def reconstruct(model_dir: str = MODEL_DIR) -> pd.DataFrame:
    """Return the test DataFrame with observed, predicted, and all metadata.

    Returns
    -------
    pd.DataFrame
        Test-set rows with columns: observed, predicted, theta, depth_cm,
        source, sample_id, lat, lon, spatial_group, plus all feature columns.
    """
    model_path = Path(model_dir)

    with open(model_path / "direct_model_results.json") as f:
        results = json.load(f)

    config = results["config"]
    obs_table = config["obs_table"]
    exclude_groups = config.get("exclude_groups")
    test_size = config.get("test_size", 0.2)
    random_state = config.get("random_state", 42)
    resolution_m = config.get("resolution_m") or 250

    split_manifest = str(model_path / "spatial_split.json")
    manifest_exists = Path(split_manifest).exists()

    data = prepare_direct_data(
        obs_table_path=obs_table,
        output_dir=model_dir,
        exclude_groups=exclude_groups,
        drop_blocking_features=config.get("drop_blocking_features", True),
        resolution_m=resolution_m,
        test_size=test_size,
        random_state=random_state,
        split_manifest=split_manifest if manifest_exists else None,
    )

    test_df = data["test_df"].copy()
    all_features = data["all_features"]

    # Load model and imputer, regenerate predictions
    model = joblib.load(model_path / "direct_rf_model.joblib")
    imputer = joblib.load(model_path / "direct_rf_imputer.joblib")

    X_test = test_df[all_features].values.astype(np.float32)
    X_test = imputer.transform(X_test)
    y_pred = model.predict(X_test).astype(np.float32)

    test_df["observed"] = test_df["log10_suction_cm"].values
    test_df["predicted"] = y_pred
    test_df["spatial_group"] = assign_spatial_group(test_df, resolution_m=resolution_m)

    return test_df


if __name__ == "__main__":
    df = reconstruct()
    out = Path(MODEL_DIR) / "test_set_full.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")
    print(f"Columns: {df.columns.tolist()[:15]}...")
    print(
        f"R2 check: {1 - np.sum((df['observed'] - df['predicted']) ** 2) / np.sum((df['observed'] - df['observed'].mean()) ** 2):.4f}"
    )
