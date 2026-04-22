"""
Reconstruct the full test set for the global-pruned RF model.

The saved predictions.parquet has only (observed, predicted, source).
This module identifies test rows by running the saved model on all candidate
rows and matching the output to predictions.parquet.  This avoids dependence
on reproducing the exact spatial split, which requires parameters
(resolution_m) that were not persisted in early model configs.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

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


def load_table_and_model(
    model_dir: str = MODEL_DIR,
) -> tuple[pd.DataFrame, list[str], object, object, dict]:
    """Load the full observation table, saved feature list, model, and imputer.

    Returns (df, all_features, model, imputer, config).
    """
    model_path = Path(model_dir)

    with open(model_path / "direct_model_results.json") as f:
        results = json.load(f)
    config = results["config"]

    with open(model_path / "direct_rf_features.json") as f:
        all_features = json.load(f)

    model = joblib.load(model_path / "direct_rf_model.joblib")
    imputer = joblib.load(model_path / "direct_rf_imputer.joblib")

    df = pd.read_parquet(config["obs_table"])
    df = df.dropna(subset=["theta", "log10_suction_cm", "lat", "lon"])

    return df, all_features, model, imputer, config


def reconstruct(model_dir: str = MODEL_DIR) -> pd.DataFrame:
    """Return the test DataFrame with observed, predicted, and all metadata.

    Identifies test rows by predicting on every candidate row with the saved
    model+imputer and matching (predicted, observed, source) triples to the
    stored predictions.parquet.  This is deterministic and independent of
    the spatial-split parameters.

    Returns
    -------
    pd.DataFrame
        Test-set rows with columns: observed, predicted, theta, depth_cm,
        source, sample_id, lat, lon, plus all feature columns.
    """
    model_path = Path(model_dir)
    predictions = pd.read_parquet(model_path / "predictions.parquet")
    n_expected = len(predictions)

    df, all_features, model, imputer, config = load_table_and_model(model_dir)

    # Predict on every row with the saved model and imputer.
    # Keep float64 to match the precision stored in predictions.parquet.
    X = df[all_features].values.astype(np.float32)
    X = imputer.transform(X)
    preds = model.predict(X)  # float64
    df["predicted"] = preds
    df["observed"] = df["log10_suction_cm"].values

    # Build a set of target triples from predictions.parquet for matching.
    # Round to 5 decimal places to absorb any float32/64 storage mismatch.
    def _key(pred, obs, src):
        return (round(float(pred), 5), round(float(obs), 5), src)

    target_keys = {}
    for _, row in predictions.iterrows():
        k = _key(row["predicted"], row["observed"], row["source"])
        target_keys[k] = target_keys.get(k, 0) + 1

    # Match rows in the full table to the target set
    match_counts = {}
    mask = np.zeros(len(df), dtype=bool)
    pred_arr = df["predicted"].values
    obs_arr = df["observed"].values
    src_arr = df["source"].values

    for i in range(len(df)):
        k = _key(pred_arr[i], obs_arr[i], src_arr[i])
        if k in target_keys:
            used = match_counts.get(k, 0)
            if used < target_keys[k]:
                mask[i] = True
                match_counts[k] = used + 1

    test_df = df[mask].copy()
    if len(test_df) != n_expected:
        print(
            f"WARNING: matched {len(test_df)}/{n_expected} test rows "
            f"(delta={len(test_df) - n_expected})"
        )
    else:
        print(f"Matched all {n_expected} test rows")

    return test_df


if __name__ == "__main__":
    test = reconstruct()
    out = Path(MODEL_DIR) / "test_set_full.parquet"
    test.to_parquet(out, index=False)
    print(f"Wrote {len(test)} rows to {out}")
    print(f"Columns: {test.columns.tolist()[:15]}...")
    print(
        f"R2 check: {1 - np.sum((test['observed'] - test['predicted']) ** 2) / np.sum((test['observed'] - test['observed'].mean()) ** 2):.4f}"
    )
