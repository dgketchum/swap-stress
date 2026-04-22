"""
Section 5.6: Within-pixel variance from existing replicates.

For every 9 km pixel containing >=2 training stations, computes within-pixel
variance of observed suction and RF-predicted suction at matched theta bins.
This is a proxy for subpixel heterogeneity, not a formal ANOVA partition.

Matching is done within (theta_bin, depth_stratum, source) cells to isolate
spatial representativeness from moisture-state variation, depth offsets, and
measurement-method differences.

Produces:
    - within_pixel_variance.csv        (per-pixel variance stats)
    - within_pixel_histogram.png       (distribution of within-pixel SD)
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

from error_analysis.reconstruct_test_set import MODEL_DIR


def compute_within_pixel_stats(
    df: pd.DataFrame,
    all_features: list[str],
    model,
    imputer,
    theta_bins: int = 5,
) -> pd.DataFrame:
    """Compute within-pixel variance of observed and predicted suction.

    Groups by (spatial_group, theta_bin, rosetta_level, source), then
    computes variance across distinct sites within each cell.  Only cells
    with >=2 sites contribute.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: spatial_group, sample_id, theta, log10_suction_cm,
        and all columns in ``all_features``.
    all_features : list[str]
        Feature columns for model prediction.
    model : fitted RF
    imputer : fitted SimpleImputer
    theta_bins : int
        Number of theta quantile bins.

    Returns
    -------
    pd.DataFrame
        Per-pixel summary with columns: spatial_group, n_sites, n_cells,
        mean_obs_var, mean_pred_var, median_obs_sd, median_pred_sd.
    """
    df = df.copy()
    df["theta_bin"] = pd.qcut(df["theta"], theta_bins, labels=False, duplicates="drop")

    # Generate predictions
    X = df[all_features].values.astype(np.float32)
    X = imputer.transform(X)
    df["predicted"] = model.predict(X).astype(np.float32)

    has_depth = "rosetta_level" in df.columns
    has_source = "source" in df.columns

    strata_cols = ["theta_bin"]
    if has_depth:
        strata_cols.append("rosetta_level")
    if has_source:
        strata_cols.append("source")

    pixel_rows = []
    for sg, pixel_df in df.groupby("spatial_group"):
        n_sites = pixel_df["sample_id"].nunique()
        if n_sites < 2:
            continue

        obs_vars = []
        pred_vars = []

        for _key, cell_df in pixel_df.groupby(strata_cols):
            if cell_df["sample_id"].nunique() < 2:
                continue

            # Variance across sites within this matched cell
            site_obs_means = cell_df.groupby("sample_id")["log10_suction_cm"].mean()
            site_pred_means = cell_df.groupby("sample_id")["predicted"].mean()

            if len(site_obs_means) >= 2:
                obs_vars.append(site_obs_means.var())
                pred_vars.append(site_pred_means.var())

        if not obs_vars:
            continue

        pixel_rows.append(
            {
                "spatial_group": sg,
                "n_sites": n_sites,
                "n_cells": len(obs_vars),
                "mean_obs_var": np.mean(obs_vars),
                "mean_pred_var": np.mean(pred_vars),
                "median_obs_sd": np.sqrt(np.median(obs_vars)),
                "median_pred_sd": np.sqrt(np.median(pred_vars)),
            }
        )

    return pd.DataFrame(pixel_rows)


def plot_within_pixel_histogram(
    pixel_stats: pd.DataFrame,
    output_dir: str,
    filename: str = "within_pixel_histogram.png",
) -> str:
    """Histogram of within-pixel SD for observed and predicted suction."""
    obs_sd = pixel_stats["median_obs_sd"].dropna()
    pred_sd = pixel_stats["median_pred_sd"].dropna()

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, max(obs_sd.max(), pred_sd.max()) * 1.05, 35)
    ax.hist(
        obs_sd, bins=bins, alpha=0.6, label="Observed", edgecolor="k", linewidth=0.3
    )
    ax.hist(
        pred_sd,
        bins=bins,
        alpha=0.6,
        label="RF predicted",
        edgecolor="k",
        linewidth=0.3,
    )
    ax.axvline(
        obs_sd.median(),
        color="C0",
        linestyle="--",
        linewidth=1.5,
        label=f"obs median = {obs_sd.median():.3f}",
    )
    ax.axvline(
        pred_sd.median(),
        color="C1",
        linestyle="--",
        linewidth=1.5,
        label=f"pred median = {pred_sd.median():.3f}",
    )
    ax.set_xlabel(
        r"Within-pixel $\sigma$ of log$_{10}$|$\psi$| (theta/depth/source matched)"
    )
    ax.set_ylabel("Count (pixels)")
    ax.set_title(f"Subpixel heterogeneity proxy ({len(pixel_stats)} multi-site pixels)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Section 5.6: within-pixel variance from existing replicates"
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
        "--resolution-m",
        type=float,
        default=250,
        help="Spatial grouping resolution in meters.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_dir)
    output_dir = args.output_dir or os.path.join(args.model_dir, "error_analysis")
    os.makedirs(output_dir, exist_ok=True)

    with open(model_path / "direct_model_results.json") as f:
        model_results = json.load(f)
    config = model_results["config"]

    with open(model_path / "direct_rf_features.json") as f:
        all_features = json.load(f)

    model = joblib.load(model_path / "direct_rf_model.joblib")
    imputer = joblib.load(model_path / "direct_rf_imputer.joblib")

    print("Loading training table...")
    df = pd.read_parquet(config["obs_table"])
    df = df.dropna(subset=["theta", "log10_suction_cm", "lat", "lon"])

    from map.learning.direct.data import assign_spatial_group

    df["spatial_group"] = assign_spatial_group(df, resolution_m=args.resolution_m)
    df = df.dropna(subset=["spatial_group"])
    print(f"  {len(df)} observations, {df['spatial_group'].nunique()} spatial groups")

    sites_per_pixel = df.groupby("spatial_group")["sample_id"].nunique()
    multi_site = sites_per_pixel[sites_per_pixel >= 2]
    print(f"  {len(multi_site)} pixels with >=2 sites")

    if "rosetta_level" not in df.columns and "depth_cm" in df.columns:
        from retention_curve.depth_utils import depth_to_rosetta_level

        df["rosetta_level"] = df["depth_cm"].apply(
            lambda d: depth_to_rosetta_level(d) if pd.notna(d) else None
        )

    print("Computing within-pixel variance (theta/depth/source matched)...")
    pixel_stats = compute_within_pixel_stats(
        df, all_features, model, imputer, theta_bins=5
    )
    pixel_stats.to_csv(
        os.path.join(output_dir, "within_pixel_variance.csv"), index=False
    )
    print(f"  {len(pixel_stats)} multi-site pixels with usable matched cells")

    if len(pixel_stats) > 0:
        print("\nWithin-pixel heterogeneity summary:")
        print(f"  Observed median SD:  {pixel_stats['median_obs_sd'].median():.3f}")
        print(f"  Predicted median SD: {pixel_stats['median_pred_sd'].median():.3f}")
        print(f"  Observed mean var:   {pixel_stats['mean_obs_var'].mean():.4f}")
        print(f"  Predicted mean var:  {pixel_stats['mean_pred_var'].mean():.4f}")
        plot_within_pixel_histogram(pixel_stats, output_dir)
    else:
        print("  No multi-site pixels with usable matched cells found.")


if __name__ == "__main__":
    main()
