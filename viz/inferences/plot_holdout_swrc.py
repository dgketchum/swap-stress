"""
Plot observed vs predicted SWRCs for held-out MT-Mesonet and REESH sites.

Loads the saved direct RF model, reproduces the spatial train/test split,
and generates one PNG per site showing observed data points and a smooth
predicted retention curve for each depth.

Usage:
    python -m viz.inferences.plot_holdout_swrc \
        --obs-table ~/data/.../obs_level_training_emb_250m.parquet \
        --model-dir ~/data/.../direct_spatial_split/ \
        --out-dir ~/data/.../holdout_swrc_plots/ \
        --sources mt_mesonet reesh
"""

import argparse
import json
import os
import re

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from map.data.features import get_feature_columns
from map.learning.direct.data import (
    apply_site_split,
    audit_dataset,
    create_site_split,
    filter_blocking_features,
    filter_complete_samples,
)
from map.learning.direct.preprocessing import build_preprocessor


def extract_site_name(sample_id: str) -> str:
    """Strip trailing _<digits> depth suffix from sample_id to get site name."""
    return re.sub(r"_\d+$", "", sample_id)


def plot_site_swrc(
    site_name: str,
    site_df: pd.DataFrame,
    model,
    feature_cols: list,
    imputer,
    out_dir: str,
):
    """Plot observed vs predicted SWRC for one site (all depths on one axes)."""
    all_features = feature_cols + ["theta"]
    depths = sorted(site_df["depth_cm"].unique())
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(depths), 1)))

    fig, ax = plt.subplots(figsize=(7, 6))

    site_r2_obs, site_r2_pred = [], []

    for i, depth in enumerate(depths):
        color = cmap[i]
        depth_df = site_df[site_df["depth_cm"] == depth].copy()
        depth_df = depth_df.dropna(subset=["theta", "log10_suction_cm"])
        if len(depth_df) < 2:
            continue

        obs_theta = depth_df["theta"].values
        obs_suction = 10 ** depth_df["log10_suction_cm"].values

        # Predict on actual observed rows
        X_obs = imputer.transform(depth_df[all_features].values)
        y_obs_pred = model.predict(X_obs)
        pred_suction = 10**y_obs_pred

        # Observed as round dots
        ax.scatter(
            obs_theta,
            obs_suction,
            color=color,
            marker="o",
            s=25,
            zorder=5,
            label=f"{depth:.0f} cm obs",
            edgecolors="k",
            linewidths=0.3,
        )

        # Inferred as x markers
        ax.scatter(
            obs_theta,
            pred_suction,
            color=color,
            marker="x",
            s=25,
            zorder=4,
            label=f"{depth:.0f} cm pred",
            linewidths=0.8,
        )

        site_r2_obs.append(depth_df["log10_suction_cm"].values)
        site_r2_pred.append(y_obs_pred)

    if not site_r2_obs:
        plt.close(fig)
        return

    # Site-level metrics
    all_obs = np.concatenate(site_r2_obs)
    all_pred = np.concatenate(site_r2_pred)
    r2 = r2_score(all_obs, all_pred)
    rmse = np.sqrt(mean_squared_error(all_obs, all_pred))

    ax.set_yscale("log")
    ax.set_xlabel("Volumetric Water Content [cm³/cm³]")
    ax.set_ylabel("Suction [cm H₂O]")

    source = site_df["source"].iloc[0] if "source" in site_df.columns else ""
    ax.set_title(f"{site_name}  ({source})")
    ax.text(
        0.95,
        0.95,
        f"R² = {r2:.3f}\nRMSE = {rmse:.3f}\nn = {len(all_obs)}",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        family="monospace",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.grid(True, which="both", ls="-", alpha=0.3)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{site_name}.png"), dpi=200)
    plt.close(fig)
    print(f"  {site_name}: R²={r2:.3f}, RMSE={rmse:.3f}, n={len(all_obs)}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot holdout-site SWRCs (observed vs direct RF model)",
    )
    parser.add_argument(
        "--obs-table", required=True, help="Path to obs-level training parquet"
    )
    parser.add_argument(
        "--model-dir", required=True, help="Directory with saved RF model"
    )
    parser.add_argument(
        "--out-dir", required=True, help="Output directory for PNG plots"
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=["mt_mesonet", "reesh"],
        help="Sources to plot (default: mt_mesonet reesh)",
    )
    parser.add_argument(
        "--max-sites",
        type=int,
        default=None,
        help="Limit number of sites (for testing)",
    )
    args = parser.parse_args()

    # Load data
    print("Loading observation table...")
    df = pd.read_parquet(args.obs_table)
    if df.index.name:
        df = df.reset_index()
    df = filter_complete_samples(df)

    # Reproduce the spatial split
    print("Reproducing spatial split...")
    feature_cols = get_feature_columns(df, include_depth=True, include_embeddings=False)

    audit = audit_dataset(df, feature_cols)
    if audit["blocking_features"]:
        feature_cols = filter_blocking_features(
            feature_cols, audit["blocking_features"]
        )

    train_sites, test_sites = create_site_split(df, "sample_id", 0.2, 42)
    train_df, test_df = apply_site_split(df, train_sites, test_sites, "sample_id")
    train_df = train_df.dropna(subset=["theta", "log10_suction_cm"])
    test_df = test_df.dropna(subset=["theta", "log10_suction_cm"])
    print(f"  Test set: {len(test_df)} obs")

    # Load model and features
    model = joblib.load(os.path.join(args.model_dir, "direct_rf_model.joblib"))
    with open(os.path.join(args.model_dir, "direct_rf_features.json")) as f:
        saved_features = json.load(f)

    # The saved feature list includes theta; separate it
    saved_feature_cols = [c for c in saved_features if c != "theta"]
    all_features = saved_feature_cols + ["theta"]

    # Fit imputer on training data (same as training time)
    print("Fitting imputer on training data...")
    imputer = build_preprocessor(add_indicator=False)
    imputer.fit(train_df[all_features].values)

    # Filter test set to requested sources
    if args.sources:
        test_df = test_df[test_df["source"].isin(args.sources)].copy()
    print(f"  Filtered to sources {args.sources}: {len(test_df)} obs")

    # Extract site names and plot per site
    test_df["site_name"] = test_df["sample_id"].apply(extract_site_name)
    sites = sorted(test_df["site_name"].unique())
    if args.max_sites:
        sites = sites[: args.max_sites]

    print(f"Plotting {len(sites)} held-out sites...")
    for site in sites:
        site_df = test_df[test_df["site_name"] == site]
        plot_site_swrc(site, site_df, model, saved_feature_cols, imputer, args.out_dir)

    print(f"Done. Plots saved to {args.out_dir}")


if __name__ == "__main__":
    main()
