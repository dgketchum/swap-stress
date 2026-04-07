"""
Reporting utilities for the direct suction prediction task.

Writes result JSON, predictions, scatter plots, and provenance artifacts
in a format shared between RF and NN trainers.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import numpy as np
import pandas as pd

from map.learning.direct.data import assign_spatial_group
from map.learning.direct.metrics import (
    compute_metrics,
    compute_metrics_by_site,
    compute_metrics_by_source,
)


def evaluate_and_report(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    test_df: pd.DataFrame,
    output_dir: str,
    all_features: list[str],
    train_df: pd.DataFrame,
    train_sites: set,
    test_sites: set,
    resolution_m: float = 250,
    feature_importance: Dict[str, float] | None = None,
    extra_config: Dict[str, Any] | None = None,
    model_family: str = "rf",
    model_name: str | None = None,
    validation_metrics: Dict[str, float] | None = None,
    training_summary: Dict[str, Any] | None = None,
) -> Dict:
    """Run standard evaluation and write all artifacts.

    Parameters
    ----------
    y_test, y_pred : np.ndarray
        Test targets and predictions (log10 scale).
    test_df : pd.DataFrame
        Test dataframe (for source and spatial group info).
    output_dir : str
        Where to write artifacts.
    all_features : list of str
        Feature names used by the model.
    train_df : pd.DataFrame
        Training dataframe (for counts).
    train_sites, test_sites : set
        Spatial group labels.
    resolution_m : float
        Grid cell size for site-level metrics.
    feature_importance : dict, optional
        Feature-level MDI importance (RF only).
    extra_config : dict, optional
        Additional config entries for the results JSON.
    model_family : str
        'rf' or 'nn'.
    model_name : str, optional
        Architecture name (e.g. 'mlp', 'ft_transformer').
    validation_metrics : dict, optional
        NN validation metrics.
    training_summary : dict, optional
        NN training history summary.

    Returns
    -------
    dict
        Full results dictionary (also written to direct_model_results.json).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Overall metrics
    metrics = compute_metrics(y_test, y_pred)
    print(
        f"\nOverall: R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, "
        f"MAE={metrics['mae']:.4f}"
    )

    # Source-level metrics
    source_metrics = None
    if "source" in test_df.columns:
        source_metrics = compute_metrics_by_source(
            y_test,
            y_pred,
            test_df["source"].values,
        )
        print("\nBy source:")
        for _, row in source_metrics.iterrows():
            print(
                f"  {row['source']}: R2={row['r2']:.4f}, "
                f"RMSE={row['rmse']:.4f} (n={row['n']:.0f})"
            )

    # Site-level metrics
    spatial_groups = assign_spatial_group(test_df, resolution_m=resolution_m).values
    site_metrics, site_summary = compute_metrics_by_site(y_test, y_pred, spatial_groups)
    print(
        f"\nSite-weighted: mean R2={site_summary['mean_r2']:.4f}, "
        f"median R2={site_summary['median_r2']:.4f} "
        f"({site_summary['n_sites']} spatial groups)"
    )

    # Build results dict (RF-compatible schema)
    config_block = {
        "n_features": len(all_features),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_train_sites": len(train_sites),
        "n_test_sites": len(test_sites),
    }
    if extra_config:
        config_block.update(extra_config)

    results: Dict[str, Any] = {
        "overall_metrics": metrics,
        "site_weighted_metrics": site_summary,
        "source_metrics": source_metrics.to_dict("records")
        if source_metrics is not None
        else [],
        "feature_importance": feature_importance,
        "config": config_block,
    }

    # NN-specific fields
    if model_family == "nn":
        results["model_family"] = "nn"
        results["model_name"] = model_name
        results["validation_metrics"] = validation_metrics
        results["training_summary"] = training_summary

    # Write results JSON
    results_path = os.path.join(output_dir, "direct_model_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to {results_path}")

    # Predictions parquet
    pred_df = pd.DataFrame({"observed": y_test, "predicted": y_pred})
    if "source" in test_df.columns:
        pred_df["source"] = test_df["source"].values
    pred_df.to_parquet(os.path.join(output_dir, "predictions.parquet"), index=False)

    # Site and source CSVs
    if len(site_metrics) > 0:
        site_metrics.to_csv(
            os.path.join(output_dir, "metrics_by_site.csv"),
            index=False,
        )
    if source_metrics is not None:
        source_metrics.to_csv(
            os.path.join(output_dir, "metrics_by_source.csv"),
            index=False,
        )

    # Scatter plot
    write_scatter(pred_df, metrics, output_dir)

    return results


def write_scatter(
    pred_df: pd.DataFrame,
    metrics: Dict[str, float],
    output_dir: str,
    filename: str = "scatter_direct.png",
) -> str:
    """Write observed-vs-predicted scatter plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    if "source" in pred_df.columns:
        for src in sorted(pred_df["source"].unique()):
            mask = pred_df["source"] == src
            ax.scatter(
                pred_df.loc[mask, "observed"],
                pred_df.loc[mask, "predicted"],
                s=2,
                alpha=0.3,
                label=src,
            )
        ax.legend(markerscale=4, fontsize=8)
    else:
        ax.scatter(pred_df["observed"], pred_df["predicted"], s=2, alpha=0.3)

    lo = min(pred_df["observed"].min(), pred_df["predicted"].min())
    hi = max(pred_df["observed"].max(), pred_df["predicted"].max())
    ax.plot([lo, hi], [lo, hi], "k-", lw=0.8)
    ax.set_xlabel("Observed log$_{10}$(suction) [cm]")
    ax.set_ylabel("Predicted log$_{10}$(suction) [cm]")
    ax.set_title("Direct Model — Spatial-Group Holdout")
    ax.text(
        0.05,
        0.95,
        f"R² = {metrics['r2']:.3f}\n"
        f"RMSE = {metrics['rmse']:.3f}\n"
        f"MAE = {metrics['mae']:.3f}\n"
        f"n = {metrics['n']}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        family="monospace",
    )
    fig.tight_layout()
    scatter_path = os.path.join(output_dir, filename)
    fig.savefig(scatter_path, dpi=200)
    plt.close(fig)
    print(f"Saved scatter plot to {scatter_path}")
    return scatter_path
