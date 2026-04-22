"""
Partial dependence plots for the direct suction RF model.

Computes 1D PDPs, 2D interaction PDPs, and ICE curves for selected features,
using the held-out test set reconstructed via the saved split manifest.

Usage:
    uv run python -m map.learning.decision_tree.partial_dependence \
        --config configs/pdp_9km_global_pruned.toml
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence

from error_analysis.reconstruct_test_set import reconstruct
from map.config import load_config, write_provenance
from map.data.ee_feature_list import label_feature

_EXTRA_LABELS = {
    "theta": r"$\theta$ (volumetric water content)",
    "depth_cm": "Measurement depth (cm)",
    "rosetta_level": "Rosetta depth level",
}


def _label(feature_name: str) -> str:
    if feature_name in _EXTRA_LABELS:
        return _EXTRA_LABELS[feature_name]
    return label_feature(feature_name)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_model_and_data(
    model_dir: str,
    max_samples: int = 5000,
    random_state: int = 42,
) -> tuple[object, object, np.ndarray, list[str]]:
    """Load the trained RF, imputer, and pre-imputed test set.

    Returns
    -------
    model : RandomForestRegressor
    imputer : SimpleImputer
    X : np.ndarray, shape (n_samples, n_features)
        Pre-imputed test feature matrix, subsampled to *max_samples*.
    feature_names : list[str]
    """
    model_path = Path(model_dir)
    model = joblib.load(model_path / "direct_rf_model.joblib")
    imputer = joblib.load(model_path / "direct_rf_imputer.joblib")
    with open(model_path / "direct_rf_features.json") as f:
        feature_names = json.load(f)

    test_df = reconstruct(model_dir)

    X_raw = test_df[feature_names].values.astype(np.float32)
    X = imputer.transform(X_raw)

    rng = np.random.RandomState(random_state)
    if len(X) > max_samples:
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X = X[idx]

    return model, imputer, X, feature_names


# ---------------------------------------------------------------------------
# 1D PDP computation
# ---------------------------------------------------------------------------


def compute_1d_pdps(
    model: object,
    X: np.ndarray,
    feature_names: list[str],
    features: list[str],
    grid_resolution: int = 100,
) -> list[dict]:
    """Compute 1D partial dependence for each feature.

    Returns list of dicts with keys: feature, grid, pdp, idx.
    """
    results = []
    for feat in features:
        idx = feature_names.index(feat)
        pd_result = partial_dependence(
            model,
            X,
            features=[idx],
            grid_resolution=grid_resolution,
            kind="average",
        )
        results.append(
            {
                "feature": feat,
                "idx": idx,
                "grid": pd_result["grid_values"][0],
                "pdp": pd_result["average"][0],
            }
        )
    return results


# ---------------------------------------------------------------------------
# ICE computation
# ---------------------------------------------------------------------------


def compute_ice(
    model: object,
    X: np.ndarray,
    feature_names: list[str],
    feature: str,
    grid_resolution: int = 100,
) -> dict:
    """Compute ICE curves and average PDP for a single feature.

    Returns dict with keys: feature, grid, ice (n_samples x grid_res), pdp.
    """
    idx = feature_names.index(feature)
    pd_result = partial_dependence(
        model,
        X,
        features=[idx],
        grid_resolution=grid_resolution,
        kind="both",
    )
    return {
        "feature": feature,
        "idx": idx,
        "grid": pd_result["grid_values"][0],
        "ice": pd_result["individual"][0],
        "pdp": pd_result["average"][0],
    }


# ---------------------------------------------------------------------------
# 2D PDP computation
# ---------------------------------------------------------------------------


def compute_2d_pdp(
    model: object,
    X: np.ndarray,
    feature_names: list[str],
    pair: list[str],
    grid_resolution: int = 50,
) -> dict:
    """Compute 2D partial dependence for a feature pair.

    Returns dict with keys: features, grid_x, grid_y, pdp_2d.
    """
    idx_a = feature_names.index(pair[0])
    idx_b = feature_names.index(pair[1])
    pd_result = partial_dependence(
        model,
        X,
        features=[(idx_a, idx_b)],
        grid_resolution=grid_resolution,
        kind="average",
    )
    return {
        "features": pair,
        "grid_x": pd_result["grid_values"][0],
        "grid_y": pd_result["grid_values"][1],
        "pdp_2d": pd_result["average"][0],
    }


# ---------------------------------------------------------------------------
# Plotting — 1D grid
# ---------------------------------------------------------------------------


def plot_1d_grid(
    pdp_results: list[dict],
    X: np.ndarray,
    output_path: str,
) -> str:
    """Plot all 1D PDPs in a 2-column grid."""
    n = len(pdp_results)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.8 * nrows))
    axes = axes.flatten()

    for i, res in enumerate(pdp_results):
        ax = axes[i]
        feat = res["feature"]
        grid = res["grid"]
        pdp = res["pdp"]
        col_vals = X[:, res["idx"]]

        lw = 2.5 if feat == "theta" else 1.5
        color = "#c0392b" if feat == "theta" else "#2c3e50"
        ax.plot(grid, pdp, color=color, linewidth=lw)
        ax.plot(
            col_vals,
            np.full_like(col_vals, pdp.min()),
            "|",
            color="#999999",
            alpha=0.08,
            markersize=4,
        )
        ax.set_xlabel(_label(feat), fontsize=9)
        ax.set_ylabel(r"Partial dependence (log$_{10}$ suction)", fontsize=9)
        ax.tick_params(labelsize=8)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Plotting — ICE
# ---------------------------------------------------------------------------


def plot_ice(
    ice_result: dict,
    output_path: str,
    max_curves: int = 300,
    random_state: int = 42,
) -> str:
    """Plot ICE curves with PDP overlay."""
    grid = ice_result["grid"]
    ice = ice_result["ice"]
    pdp = ice_result["pdp"]
    feat = ice_result["feature"]

    rng = np.random.RandomState(random_state)
    n = ice.shape[0]
    if n > max_curves:
        idx = rng.choice(n, size=max_curves, replace=False)
        ice_sub = ice[idx]
    else:
        ice_sub = ice

    fig, ax = plt.subplots(figsize=(10, 6))
    for row in ice_sub:
        ax.plot(grid, row, color="#bdc3c7", alpha=0.06, linewidth=0.5)
    ax.plot(grid, pdp, color="#2c3e50", linewidth=2.5, label="PDP (average)")

    ax.set_xlabel(_label(feat), fontsize=11)
    ax.set_ylabel(r"Predicted log$_{10}$ suction (cm H$_2$O)", fontsize=11)
    ax.legend(fontsize=10, framealpha=0.8)
    ax.tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Plotting — 2D
# ---------------------------------------------------------------------------


def plot_2d(
    result_2d: dict,
    output_path: str,
) -> str:
    """Plot 2D PDP as a filled contour."""
    gx = result_2d["grid_x"]
    gy = result_2d["grid_y"]
    Z = result_2d["pdp_2d"]
    feat_x, feat_y = result_2d["features"]

    fig, ax = plt.subplots(figsize=(8, 6))
    XX, YY = np.meshgrid(gx, gy)
    cf = ax.contourf(XX, YY, Z, levels=20, cmap="plasma")
    cs = ax.contour(
        XX, YY, Z, levels=[1.5, 2.0, 2.5, 3.0, 3.5], colors="white", linewidths=0.8
    )
    ax.clabel(cs, fmt="%.1f", fontsize=8)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(r"Partial dependence (log$_{10}$ suction)", fontsize=10)

    ax.set_xlabel(_label(feat_x), fontsize=11)
    ax.set_ylabel(_label(feat_y), fontsize=11)
    ax.tick_params(labelsize=9)

    if feat_y == "depth_cm":
        ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_pdp_analysis(config: dict) -> dict:
    """Run partial dependence analysis from a merged config dict."""
    model_dir = config["model_dir"]
    output_dir = config["output_dir"]
    max_samples = config.get("max_samples", 5000)
    grid_resolution = config.get("grid_resolution", 100)
    random_state = config.get("random_state", 42)
    features_1d = config.get("features_1d", [])
    features_2d = config.get("features_2d", [])
    ice_features = config.get("ice_features", [])

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model and test data from {model_dir}")
    model, imputer, X, feature_names = load_model_and_data(
        model_dir, max_samples=max_samples, random_state=random_state
    )
    print(f"  Test samples: {X.shape[0]}, features: {X.shape[1]}")

    saved = {}

    # --- 1D PDPs ---
    if features_1d:
        missing = [f for f in features_1d if f not in feature_names]
        if missing:
            raise ValueError(f"Features not in model: {missing}")

        print(f"Computing 1D PDPs for {len(features_1d)} features ...")
        pdp_results = compute_1d_pdps(
            model, X, feature_names, features_1d, grid_resolution
        )
        path = os.path.join(output_dir, "pdp_1d_grid.png")
        plot_1d_grid(pdp_results, X, path)
        print(f"  Saved {path}")

        saved["features_1d"] = {
            r["feature"]: {
                "grid": r["grid"].tolist(),
                "pdp": r["pdp"].tolist(),
            }
            for r in pdp_results
        }

    # --- ICE ---
    for feat in ice_features:
        if feat not in feature_names:
            raise ValueError(f"ICE feature not in model: {feat}")

        print(f"Computing ICE for {feat} ...")
        ice_result = compute_ice(model, X, feature_names, feat, grid_resolution)
        path = os.path.join(output_dir, f"ice_{feat}.png")
        plot_ice(ice_result, path, random_state=random_state)
        print(f"  Saved {path}")

        saved.setdefault("ice", {})[feat] = {
            "grid": ice_result["grid"].tolist(),
            "pdp": ice_result["pdp"].tolist(),
            "n_curves": ice_result["ice"].shape[0],
        }

    # --- 2D PDPs ---
    for pair in features_2d:
        for f in pair:
            if f not in feature_names:
                raise ValueError(f"2D feature not in model: {f}")

        print(f"Computing 2D PDP for {pair[0]} x {pair[1]} ...")
        result_2d = compute_2d_pdp(
            model, X, feature_names, pair, grid_resolution=min(grid_resolution, 50)
        )
        safe_name = f"{pair[0]}_x_{pair[1]}"
        path = os.path.join(output_dir, f"pdp_2d_{safe_name}.png")
        plot_2d(result_2d, path)
        print(f"  Saved {path}")

        saved.setdefault("features_2d", {})[safe_name] = {
            "grid_x": result_2d["grid_x"].tolist(),
            "grid_y": result_2d["grid_y"].tolist(),
        }

    # --- Save numerical results ---
    results_path = os.path.join(output_dir, "pdp_results.json")
    with open(results_path, "w") as f:
        json.dump(saved, f, indent=2)
    print(f"  Saved {results_path}")

    write_provenance(output_dir, config, run_type="pdp")

    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Partial dependence plots for direct suction model.",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--grid-resolution", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, vars(args))
    run_pdp_analysis(cfg)
