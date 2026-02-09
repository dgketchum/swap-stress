"""
ReESH holdout evaluation: train direct and VG models on all non-ReESH data, predict on ReESH sites.

Generates paginated 3×3 retention curve plots for all ReESH sites with ≥5 observations,
with multi-depth coloring using plasma colormap.

Usage:
    python -m map.learning.decision_tree.reesh_holdout
    python -m map.learning.decision_tree.reesh_holdout --obs-table /path/to/obs.parquet
"""

import argparse
import json
import os
import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from map.learning.decision_tree.compare_approaches import (
    VGExperimentConfig,
    audit_dataset,
    compute_metrics,
    compute_metrics_by_site,
    extract_site_id,
    filter_blocking_features,
    get_feature_cols_for_config,
    get_feature_columns,
    load_obs_table,
    load_vg_table,
    predict_direct_approach,
    predict_vg_approach,
    train_direct_model,
    train_vg_with_depth_handling,
)


def _extract_depth(sample_id):
    """Extract depth value from sample_id suffix (e.g., 'reesh_IN_Martell_50' -> 50)."""
    if not isinstance(sample_id, str):
        return None
    m = re.search(r"_(\d+)$", sample_id)
    return int(m.group(1)) if m else None


def plot_retention_curves_per_site(
    test_obs_df,
    y_pred,
    y_pred_vg,
    output_dir,
    prefix,
):
    """Plot one retention curve PNG per ReESH site.

    Each figure shows depths colored using plasma colormap.
    Scatter = observed, dashed = direct prediction, solid = VG prediction.

    Parameters
    ----------
    test_obs_df : pd.DataFrame
        Test observations with sample_id, theta, log10_suction_cm.
    y_pred : np.ndarray
        Direct-model predicted log10(suction_cm) for each test row.
    y_pred_vg : np.ndarray
        VG-model predicted log10(suction_cm) for each test row.
    output_dir : str
        Directory to save PNG files.
    prefix : str
        Filename prefix for output PNGs.
    """
    site_ids = test_obs_df["sample_id"].apply(extract_site_id)
    site_counts = site_ids.value_counts()
    valid_sites = sorted(site_counts[site_counts >= 5].index)

    if not valid_sites:
        print("No sites with >=5 observations for curve plotting")
        return

    site_dir = os.path.join(output_dir, prefix)
    os.makedirs(site_dir, exist_ok=True)
    print(f"  Plotting {len(valid_sites)} sites to {site_dir}/")

    plt.style.use("seaborn-v0_8-whitegrid")

    for site in valid_sites:
        site_mask = (site_ids == site).values

        site_sample_ids = test_obs_df.loc[site_mask, "sample_id"].values
        unique_samples = sorted(set(site_sample_ids))

        depths = []
        for sid in unique_samples:
            d = _extract_depth(sid)
            depths.append(d if d is not None else 0)
        depth_order = np.argsort(depths)
        unique_samples = [unique_samples[i] for i in depth_order]
        depths = [depths[i] for i in depth_order]

        n_depths = len(unique_samples)
        colors = plt.cm.plasma(np.linspace(0, 0.85, max(n_depths, 1)))

        fig, ax = plt.subplots(figsize=(6, 5))

        for di, (sid, depth) in enumerate(zip(unique_samples, depths)):
            sample_mask = (test_obs_df["sample_id"] == sid).values
            color = colors[di]

            theta = test_obs_df.loc[sample_mask, "theta"].values
            y_true = test_obs_df.loc[sample_mask, "log10_suction_cm"].values
            y_direct = y_pred[sample_mask]
            y_vg = y_pred_vg[sample_mask]

            order = np.argsort(theta)
            label = f"{depth} cm"

            ax.plot(
                theta[order],
                y_vg[order],
                "-",
                color=color,
                lw=2.0,
                alpha=0.9,
                zorder=2,
            )
            ax.scatter(theta, y_true, c=[color], s=4, alpha=0.5, label=label, zorder=3)
            ax.plot(
                theta[order],
                y_direct[order],
                "--",
                color=color,
                lw=2.5,
                alpha=1.0,
                zorder=5,
            )

        ax.set_xlabel("θ")
        ax.set_ylabel("log₁₀(ψ) (cm)")
        ax.set_title(site)
        ax.set_xlim(0, 0.8)
        ax.set_ylim(bottom=0)

        depth_legend = ax.legend(loc="upper right", fontsize=8, title="Depth")
        ax.add_artist(depth_legend)

        style_handles = [
            Line2D([0], [0], marker="o", color="gray", ls="", ms=3),
            Line2D([0], [0], color="gray", ls="--", lw=1.5),
            Line2D([0], [0], color="gray", ls="-", lw=1.5),
        ]
        ax.legend(
            style_handles,
            ["Observed", "Direct", "VG"],
            loc="lower left",
            fontsize=8,
        )

        plt.tight_layout()
        safe_name = site.replace("/", "_").replace(" ", "_")
        out_path = os.path.join(site_dir, f"{safe_name}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

    print(f"  Saved {len(valid_sites)} site plots")


def plot_scatter_obs_vs_pred(y_true, y_pred, metrics, output_path, title):
    """Plot observed (x) vs predicted (y) scatter with 1:1 line and metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Observed log10(suction_cm).
    y_pred : np.ndarray
        Predicted log10(suction_cm).
    metrics : dict
        Output of compute_metrics (must contain rmse, mae, r2, bias, n).
    output_path : str
        Path to save PNG.
    title : str
        Plot title.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    ax.scatter(y_true, y_pred, s=3, alpha=0.3, color="steelblue", edgecolors="none")

    lo = min(y_true.min(), y_pred.min()) - 0.2
    hi = max(y_true.max(), y_pred.max()) + 0.2
    ax.plot([lo, hi], [lo, hi], "k-", lw=1, zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    ax.set_xlabel("Observed log$_{10}$(\\u03c8) [cm]")
    ax.set_ylabel("Predicted log$_{10}$(\\u03c8) [cm]")
    ax.set_title(title)

    text = (
        f"RMSE = {metrics['rmse']:.3f}\n"
        f"MAE  = {metrics['mae']:.3f}\n"
        f"R\\u00b2   = {metrics['r2']:.3f}\n"
        f"Bias = {metrics['bias']:.3f}\n"
        f"n    = {metrics['n']}"
    )
    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        family="monospace",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray", boxstyle="round"),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")


def run_reesh_holdout(obs_table_path, output_dir, vg_table_path=None):
    """Run ReESH holdout evaluation with both direct and VG models.

    Parameters
    ----------
    obs_table_path : str
        Path to observation-level parquet table.
    output_dir : str
        Directory for output files.
    vg_table_path : str, optional
        Path to VG parameter-level parquet table.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load and filter
    print("Loading observation table...")
    obs_df = load_obs_table(obs_table_path)
    obs_df = obs_df.dropna(subset=["lat", "lon"])
    print(f"  {len(obs_df)} observations with lat/lon")

    # Split by source
    train_df = obs_df[obs_df["source"] != "reesh"].copy()
    test_df = obs_df[obs_df["source"] == "reesh"].copy().reset_index(drop=True)

    print(
        f"  Train: {len(train_df)} obs (sources: {sorted(train_df['source'].unique())})"
    )
    print(f"  Test:  {len(test_df)} obs (ReESH only)")

    if len(test_df) == 0:
        print("ERROR: No ReESH observations found")
        return

    # --- Direct model ---
    feature_cols = get_feature_columns(train_df)
    print(f"  {len(feature_cols)} initial features")

    audit = audit_dataset(train_df, feature_cols, output_dir=output_dir)
    feature_cols = filter_blocking_features(feature_cols, audit["blocking_features"])
    print(f"  {len(feature_cols)} features after removing blocking features")

    print("\nTraining direct model on non-ReESH data...")
    model, preprocessor = train_direct_model(train_df, feature_cols)

    print("\nPredicting on ReESH observations (direct)...")
    y_pred = predict_direct_approach(
        model, test_df, feature_cols, preprocessor=preprocessor
    )
    y_true = test_df["log10_suction_cm"].values

    # Direct metrics
    overall_direct = compute_metrics(y_true, y_pred)
    print("\nDirect model - Overall metrics (log10 scale):")
    print(f"  RMSE: {overall_direct['rmse']:.4f}")
    print(f"  MAE:  {overall_direct['mae']:.4f}")
    print(f"  R2:   {overall_direct['r2']:.4f}")
    print(f"  Bias: {overall_direct['bias']:.4f}")
    print(f"  N:    {overall_direct['n']}")

    site_ids = test_df["sample_id"].apply(extract_site_id).values
    per_site_direct_df, site_weighted_direct = compute_metrics_by_site(
        y_true, y_pred, site_ids
    )

    print(
        f"\nDirect model - Site-weighted metrics ({site_weighted_direct['n_sites']} sites):"
    )
    print(f"  Mean RMSE:   {site_weighted_direct['mean_rmse']:.4f}")
    print(f"  Median RMSE: {site_weighted_direct['median_rmse']:.4f}")
    print(f"  Mean R2:     {site_weighted_direct['mean_r2']:.4f}")
    print(f"  Median R2:   {site_weighted_direct['median_r2']:.4f}")

    # --- VG model ---
    if vg_table_path is None:
        default_dir = os.path.dirname(obs_table_path)
        vg_table_path = os.path.join(default_dir, "unified_training_emb_250m.parquet")

    print(f"\nLoading VG table from {vg_table_path}...")
    vg_df = load_vg_table(vg_table_path)
    vg_train_df = vg_df[vg_df["source"] != "reesh"].copy()
    print(
        f"  VG train: {len(vg_train_df)} samples (sources: {sorted(vg_train_df['source'].unique())})"
    )

    vg_config = VGExperimentConfig(
        name="vg_log_alpha",
        depth_handling="feature",
        log_transform_params=["alpha"],
    )

    vg_feature_cols = get_feature_cols_for_config(vg_train_df, vg_config)
    vg_audit = audit_dataset(vg_train_df, vg_feature_cols, output_dir=output_dir)
    vg_feature_cols = filter_blocking_features(
        vg_feature_cols, vg_audit["blocking_features"]
    )
    print(f"  {len(vg_feature_cols)} VG features after removing blocking features")

    print("\nTraining VG model (vg_log_alpha) on non-ReESH data...")
    vg_result = train_vg_with_depth_handling(vg_train_df, vg_feature_cols, vg_config)

    print("\nPredicting on ReESH observations (VG)...")
    y_pred_vg = predict_vg_approach(
        vg_result["models"],
        test_df,
        vg_feature_cols,
        log_transform_params=vg_result["log_transform_params"],
        preprocessor=vg_result["preprocessor"],
    )

    # VG metrics
    overall_vg = compute_metrics(y_true, y_pred_vg)
    print("\nVG model - Overall metrics (log10 scale):")
    print(f"  RMSE: {overall_vg['rmse']:.4f}")
    print(f"  MAE:  {overall_vg['mae']:.4f}")
    print(f"  R2:   {overall_vg['r2']:.4f}")
    print(f"  Bias: {overall_vg['bias']:.4f}")
    print(f"  N:    {overall_vg['n']}")

    per_site_vg_df, site_weighted_vg = compute_metrics_by_site(
        y_true, y_pred_vg, site_ids
    )

    print(f"\nVG model - Site-weighted metrics ({site_weighted_vg['n_sites']} sites):")
    print(f"  Mean RMSE:   {site_weighted_vg['mean_rmse']:.4f}")
    print(f"  Median RMSE: {site_weighted_vg['median_rmse']:.4f}")
    print(f"  Mean R2:     {site_weighted_vg['mean_r2']:.4f}")
    print(f"  Median R2:   {site_weighted_vg['median_r2']:.4f}")

    # Plot
    print("\nGenerating retention curve plots...")
    plot_retention_curves_per_site(
        test_df, y_pred, y_pred_vg, output_dir, "retention_curves_reesh"
    )

    print("\nGenerating scatter plots...")
    plot_scatter_obs_vs_pred(
        y_true,
        y_pred,
        overall_direct,
        os.path.join(output_dir, "scatter_direct.png"),
        "Direct Model - ReESH Holdout",
    )
    plot_scatter_obs_vs_pred(
        y_true,
        y_pred_vg,
        overall_vg,
        os.path.join(output_dir, "scatter_vg.png"),
        "VG Model - ReESH Holdout",
    )

    # Save predictions
    pred_df = test_df[["sample_id", "theta", "log10_suction_cm"]].copy()
    pred_df["predicted_log10_suction_cm_direct"] = y_pred
    pred_df["predicted_log10_suction_cm_vg"] = y_pred_vg
    pred_path = os.path.join(output_dir, "reesh_holdout_predictions.parquet")
    pred_df.to_parquet(pred_path, index=False)
    print(f"\nSaved predictions to {pred_path}")

    # Save results JSON
    def _serialize(d):
        return {
            k: float(v) if isinstance(v, (np.floating, float)) else v
            for k, v in d.items()
        }

    results = {
        "direct_overall_metrics": _serialize(overall_direct),
        "direct_site_weighted_metrics": _serialize(site_weighted_direct),
        "direct_per_site_metrics": per_site_direct_df.to_dict(orient="records"),
        "vg_overall_metrics": _serialize(overall_vg),
        "vg_site_weighted_metrics": _serialize(site_weighted_vg),
        "vg_per_site_metrics": per_site_vg_df.to_dict(orient="records"),
        "n_train_obs": len(train_df),
        "n_train_vg": len(vg_train_df),
        "n_test": len(test_df),
        "n_direct_features": len(feature_cols),
        "n_vg_features": len(vg_feature_cols),
        "train_sources": sorted(train_df["source"].unique().tolist()),
    }
    results_path = os.path.join(output_dir, "reesh_holdout_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    default_obs = os.path.join(
        os.path.expanduser("~"),
        "data",
        "IrrigationGIS",
        "soils",
        "swapstress",
        "training",
        "obs_level_training_emb_250m.parquet",
    )
    default_out = os.path.join(
        os.path.expanduser("~"),
        "data",
        "IrrigationGIS",
        "soils",
        "swapstress",
        "training",
        "reesh_holdout",
    )

    parser = argparse.ArgumentParser(description="ReESH holdout evaluation")
    parser.add_argument("--obs-table", default=default_obs, help="Path to obs parquet")
    parser.add_argument("--output-dir", default=default_out, help="Output directory")
    parser.add_argument(
        "--vg-table",
        default=None,
        help="Path to VG parameter parquet (default: unified_training_emb_250m.parquet in same dir as obs table)",
    )
    args = parser.parse_args()

    run_reesh_holdout(args.obs_table, args.output_dir, args.vg_table)
