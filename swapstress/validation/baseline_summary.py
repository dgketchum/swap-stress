"""
Section 5.1: Baseline RF evaluation summary.

Reads existing RF artifacts and produces a formatted summary of held-out
performance: headline metrics, per-source table, scatter + residual histogram.

Produces:
    - baseline_summary.txt         (headline metrics and per-source table)
    - scatter_residual.png         (scatter + residual histogram side by side)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_DIR = "/nas/soils/swapstress/models/direct_rf_9km_global_pruned"


def load_artifacts(model_dir: str) -> dict:
    """Load the three core artifacts from a trained RF directory."""
    model_path = Path(model_dir)

    with open(model_path / "direct_model_results.json") as f:
        results = json.load(f)

    predictions = pd.read_parquet(model_path / "predictions.parquet")

    source_metrics = None
    source_path = model_path / "metrics_by_source.csv"
    if source_path.exists():
        source_metrics = pd.read_csv(source_path)

    site_metrics = None
    site_path = model_path / "metrics_by_site.csv"
    if site_path.exists():
        site_metrics = pd.read_csv(site_path)

    return {
        "results": results,
        "predictions": predictions,
        "source_metrics": source_metrics,
        "site_metrics": site_metrics,
    }


def format_summary(artifacts: dict) -> str:
    """Build a formatted text summary from artifacts."""
    r = artifacts["results"]
    om = r["overall_metrics"]
    config = r.get("config", {})

    lines = []
    lines.append("=" * 60)
    lines.append("Baseline RF Evaluation Summary")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Model: {config.get('obs_table', 'unknown')}")
    lines.append(f"Features: {config.get('n_features', '?')}")
    lines.append(f"Train samples: {config.get('n_train', '?')}")
    lines.append(f"Test samples: {config.get('n_test', '?')}")
    lines.append("")
    lines.append("--- Headline Metrics (spatial holdout) ---")
    lines.append(f"  R2:   {om['r2']:.4f}")
    lines.append(f"  RMSE: {om['rmse']:.4f} log10(cm)")
    lines.append(f"  MAE:  {om['mae']:.4f} log10(cm)")
    lines.append(f"  Bias: {om['bias']:+.4f} log10(cm)")
    lines.append(f"  n:    {om['n']}")
    lines.append("")

    sm = artifacts["source_metrics"]
    if sm is not None:
        lines.append("--- Per-Source Metrics ---")
        for _, row in sm.iterrows():
            lines.append(
                f"  {row['source']:15s}  R2={row['r2']:.3f}  "
                f"RMSE={row['rmse']:.3f}  MAE={row['mae']:.3f}  "
                f"bias={row['bias']:+.3f}  n={row['n']:.0f}"
            )
        lines.append("")

    site = artifacts["site_metrics"]
    if site is not None:
        lines.append("--- Site-Level Summary ---")
        lines.append(f"  n_sites:    {len(site)}")
        lines.append(f"  median R2:  {site['r2'].median():.3f}")
        lines.append(f"  median RMSE:{site['rmse'].median():.3f}")
        lines.append(f"  mean R2:    {site['r2'].mean():.3f}")
        lines.append(f"  mean RMSE:  {site['rmse'].mean():.3f}")
        lines.append("")

    swm = r.get("site_weighted_metrics", {})
    if swm:
        lines.append("--- Site-Weighted Metrics ---")
        for k, v in sorted(swm.items()):
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")
        lines.append("")

    return "\n".join(lines)


def plot_scatter_residual(
    predictions: pd.DataFrame,
    metrics: dict,
    output_dir: str,
    filename: str = "scatter_residual.png",
) -> str:
    """Scatter plot + residual histogram side by side."""
    obs = predictions["observed"].values
    pred = predictions["predicted"].values
    resid = pred - obs

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter
    ax = axes[0]
    if "source" in predictions.columns:
        for src in sorted(predictions["source"].unique()):
            mask = predictions["source"] == src
            ax.scatter(obs[mask], pred[mask], s=2, alpha=0.3, label=src)
        ax.legend(markerscale=4, fontsize=8)
    else:
        ax.scatter(obs, pred, s=2, alpha=0.3)

    lo = min(obs.min(), pred.min())
    hi = max(obs.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "k-", lw=0.8)
    ax.set_xlabel(r"Observed log$_{10}$|$\psi$| (cm)")
    ax.set_ylabel(r"Predicted log$_{10}$|$\psi$| (cm)")
    ax.set_title("Spatial-Holdout Predictions")
    ax.text(
        0.05,
        0.95,
        f"R² = {metrics['r2']:.3f}\nRMSE = {metrics['rmse']:.3f}\nn = {metrics['n']}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        family="monospace",
    )

    # Residual histogram
    ax = axes[1]
    ax.hist(resid, bins=80, edgecolor="k", linewidth=0.3, alpha=0.7)
    ax.axvline(0, color="k", linewidth=0.8, linestyle="--")
    ax.axvline(
        np.mean(resid),
        color="C1",
        linewidth=1.5,
        linestyle="-",
        label=f"mean = {np.mean(resid):+.3f}",
    )
    ax.set_xlabel(r"Residual: $\hat{y} - y$ (log$_{10}$ cm)")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")
    ax.legend(fontsize=9)

    fig.tight_layout()
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Section 5.1: baseline RF evaluation summary"
    )
    parser.add_argument(
        "--model-dir",
        default=MODEL_DIR,
        help="Path to trained model directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <model-dir>/swapstress.validation/).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.model_dir, "error_analysis")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading RF artifacts...")
    artifacts = load_artifacts(args.model_dir)

    summary = format_summary(artifacts)
    print(summary)

    summary_path = os.path.join(output_dir, "baseline_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"Saved {summary_path}")

    plot_scatter_residual(
        artifacts["predictions"],
        artifacts["results"]["overall_metrics"],
        output_dir,
    )


if __name__ == "__main__":
    main()
