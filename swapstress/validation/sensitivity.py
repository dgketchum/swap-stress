"""
Section 5.5: Local sensitivity to theta.

Computes the finite-difference Jacobian d(log10|psi|)/d(theta) for the trained
RF at each held-out observation.  High sensitivity means small errors in theta
propagate into large errors in predicted suction -- a physically meaningful
caution layer even without a full retrieval-error model.

Produces:
    - sensitivity_stats.csv        (per-observation theta, predicted, jacobian)
    - sensitivity_histogram.png    (distribution of |jacobian|)
    - sensitivity_vs_theta.png     (jacobian vs theta, colored by source)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from error_analysis.reconstruct_test_set import MODEL_DIR, reconstruct


def compute_jacobian(
    model,
    imputer,
    feature_matrix: np.ndarray,
    theta_col_idx: int,
    delta: float = 0.005,
) -> np.ndarray:
    """Finite-difference Jacobian d(log10|psi|)/d(theta).

    Parameters
    ----------
    model : fitted sklearn RF
    imputer : fitted SimpleImputer
    feature_matrix : (n, p) raw feature array (pre-imputation)
    theta_col_idx : int
        Column index of theta in feature_matrix.
    delta : float
        Half-width of the central difference step.

    Returns
    -------
    np.ndarray
        (n,) array of d(pred)/d(theta) at each sample.
    """
    X_plus = feature_matrix.copy()
    X_minus = feature_matrix.copy()
    X_plus[:, theta_col_idx] += delta
    X_minus[:, theta_col_idx] -= delta

    # Clamp to physical range
    X_plus[:, theta_col_idx] = np.clip(X_plus[:, theta_col_idx], 0.001, 0.999)
    X_minus[:, theta_col_idx] = np.clip(X_minus[:, theta_col_idx], 0.001, 0.999)

    pred_plus = model.predict(imputer.transform(X_plus))
    pred_minus = model.predict(imputer.transform(X_minus))

    actual_delta = X_plus[:, theta_col_idx] - X_minus[:, theta_col_idx]
    # Avoid division by zero at boundaries
    actual_delta = np.maximum(actual_delta, 1e-6)

    return (pred_plus - pred_minus) / actual_delta


def plot_sensitivity_histogram(
    jacobian: np.ndarray,
    output_dir: str,
    filename: str = "sensitivity_histogram.png",
) -> str:
    """Histogram of absolute Jacobian values."""
    abs_j = np.abs(jacobian)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(abs_j, bins=60, edgecolor="k", linewidth=0.3, alpha=0.7)
    ax.axvline(
        np.median(abs_j),
        color="C1",
        linewidth=1.5,
        linestyle="--",
        label=f"median = {np.median(abs_j):.2f}",
    )
    ax.set_xlabel(r"|d log$_{10}$|$\psi$| / d$\theta$|")
    ax.set_ylabel("Count")
    ax.set_title("RF Sensitivity to Theta (held-out)")
    ax.legend()

    fig.tight_layout()
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def plot_sensitivity_vs_theta(
    theta: np.ndarray,
    jacobian: np.ndarray,
    source: np.ndarray | None,
    output_dir: str,
    filename: str = "sensitivity_vs_theta.png",
) -> str:
    """Scatter of Jacobian vs theta, colored by source."""
    fig, ax = plt.subplots(figsize=(7, 5))

    if source is not None:
        for src in sorted(set(source)):
            mask = source == src
            ax.scatter(theta[mask], jacobian[mask], s=2, alpha=0.2, label=src)
        ax.legend(markerscale=5, fontsize=8)
    else:
        ax.scatter(theta, jacobian, s=2, alpha=0.2)

    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlabel(r"$\theta$ (VWC)")
    ax.set_ylabel(r"d log$_{10}$|$\psi$| / d$\theta$")
    ax.set_title("Local Sensitivity to Theta")

    fig.tight_layout()
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Section 5.5: local sensitivity to theta"
    )
    parser.add_argument(
        "--model-dir",
        default=MODEL_DIR,
        help="Path to trained model directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <model-dir>/error_analysis/).",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.005,
        help="Half-width of finite-difference step (default: 0.005).",
    )
    args = parser.parse_args()

    model_path = Path(args.model_dir)
    output_dir = args.output_dir or os.path.join(args.model_dir, "error_analysis")
    os.makedirs(output_dir, exist_ok=True)

    print("Reconstructing test set...")
    test_df = reconstruct(args.model_dir)

    with open(model_path / "direct_rf_features.json") as f:
        all_features = json.load(f)

    theta_col_idx = all_features.index("theta")
    print(f"Theta is feature column {theta_col_idx}")

    model = joblib.load(model_path / "direct_rf_model.joblib")
    imputer = joblib.load(model_path / "direct_rf_imputer.joblib")

    X_raw = test_df[all_features].values.astype(np.float32)

    print(f"Computing Jacobian (delta={args.delta})...")
    jacobian = compute_jacobian(model, imputer, X_raw, theta_col_idx, delta=args.delta)

    # Save per-observation results
    stats_df = pd.DataFrame(
        {
            "theta": test_df["theta"].values,
            "predicted": test_df["predicted"].values,
            "jacobian": jacobian,
            "abs_jacobian": np.abs(jacobian),
            "source": test_df["source"].values
            if "source" in test_df.columns
            else "unknown",
        }
    )
    stats_path = os.path.join(output_dir, "sensitivity_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved {stats_path}")

    # Summary
    print(f"\nSensitivity summary (n={len(jacobian)}):")
    print(f"  Jacobian mean:   {np.mean(jacobian):.3f}")
    print(f"  Jacobian median: {np.median(jacobian):.3f}")
    print(f"  |Jacobian| mean:   {np.mean(np.abs(jacobian)):.3f}")
    print(f"  |Jacobian| median: {np.median(np.abs(jacobian)):.3f}")
    print(f"  |Jacobian| 90th:   {np.percentile(np.abs(jacobian), 90):.3f}")
    print(f"  |Jacobian| 95th:   {np.percentile(np.abs(jacobian), 95):.3f}")

    # Physically: Jacobian should be negative (more water -> less suction)
    frac_positive = np.mean(jacobian > 0)
    print(f"  Fraction with positive Jacobian: {frac_positive:.3f}")
    if frac_positive > 0.05:
        print(
            "  WARNING: >5% of samples have non-monotonic response (positive Jacobian)"
        )

    # Plots
    plot_sensitivity_histogram(jacobian, output_dir)
    source_arr = test_df["source"].values if "source" in test_df.columns else None
    plot_sensitivity_vs_theta(test_df["theta"].values, jacobian, source_arr, output_dir)


if __name__ == "__main__":
    main()
