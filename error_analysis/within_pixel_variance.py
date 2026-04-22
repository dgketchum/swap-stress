"""
Phase 1D: Within-pixel variance from existing replicates.

For every 9 km pixel containing >=2 training stations or lab profiles,
computes within-pixel variance of observed suction at matched theta bins
and of RF-predicted suction. Reports nested ANOVA decomposition:
sigma2_between-pixel, sigma2_within-pixel, sigma2_within-site.

Produces:
    - within_pixel_variance.csv        (per-pixel variance stats)
    - nested_anova.csv                 (variance decomposition)
    - within_pixel_histogram.png       (distribution of within-pixel CV)
    - nested_anova_barplot.png         (Figure 4 draft)
"""

from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_within_pixel_stats(
    df: pd.DataFrame,
    theta_bins: int = 5,
) -> pd.DataFrame:
    """Compute within-pixel variance of observed and predicted suction.

    Groups by spatial_group and theta bin, then computes variance across
    samples within each (pixel, theta-bin) cell.  Observations are first
    stratified by Rosetta level (depth matching) and source (method
    matching) so that the "within-pixel" variance isolates spatial
    representativeness rather than absorbing depth or measurement-method
    differences.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: spatial_group, sample_id, theta, log10_suction_cm.
        Should also contain rosetta_level (depth) and source (method).
    theta_bins : int
        Number of theta quantile bins for matching.

    Returns
    -------
    pd.DataFrame
        Per-pixel summary: n_samples, n_sites, mean/var of observed suction,
        mean within-site variance, between-site variance.  Includes a
        ``match_type`` column indicating whether depth+source matching
        was applied.
    """
    df = df.copy()
    df["theta_bin"] = pd.qcut(df["theta"], theta_bins, labels=False, duplicates="drop")

    has_depth = "rosetta_level" in df.columns
    has_source = "source" in df.columns

    pixel_rows = []
    for sg, pixel_df in df.groupby("spatial_group"):
        n_sites = pixel_df["sample_id"].nunique()
        if n_sites < 2:
            continue

        # Try depth+source matched strata first; fall back to depth-only,
        # then unmatched.
        strata_cols = []
        match_type = "unmatched"
        if has_depth and has_source:
            strata_cols = ["rosetta_level", "source"]
            match_type = "depth+source"
        elif has_depth:
            strata_cols = ["rosetta_level"]
            match_type = "depth_only"

        # Within each stratum, compute variance across sites at matched theta
        matched_vars = []
        site_vars_all = []
        site_means_all = []

        stratum_groups = (
            pixel_df.groupby(strata_cols) if strata_cols else [(None, pixel_df)]
        )

        for _key, stratum_df in stratum_groups:
            if stratum_df["sample_id"].nunique() < 2:
                continue

            # Within-site variance in this stratum
            for _, site_df in stratum_df.groupby("sample_id"):
                if len(site_df) >= 3:
                    site_vars_all.append(site_df["log10_suction_cm"].var())
                    site_means_all.append(site_df["log10_suction_cm"].mean())

            # Theta-matched within-pixel variance
            for _, bin_df in stratum_df.groupby("theta_bin"):
                if bin_df["sample_id"].nunique() >= 2:
                    matched_vars.append(bin_df["log10_suction_cm"].var())

        # If depth+source matching yields no usable strata, report as empty
        obs_var = pixel_df["log10_suction_cm"].var()
        obs_mean = pixel_df["log10_suction_cm"].mean()
        within_site_var = np.mean(site_vars_all) if site_vars_all else np.nan
        between_site_var = (
            np.var(site_means_all) if len(site_means_all) >= 2 else np.nan
        )
        matched_var = np.mean(matched_vars) if matched_vars else np.nan

        pixel_rows.append(
            {
                "spatial_group": sg,
                "n_samples": len(pixel_df),
                "n_sites": n_sites,
                "match_type": match_type,
                "obs_mean": obs_mean,
                "obs_var": obs_var,
                "within_site_var": within_site_var,
                "between_site_var": between_site_var,
                "theta_matched_var": matched_var,
            }
        )

    return pd.DataFrame(pixel_rows)


def nested_anova(df: pd.DataFrame) -> dict[str, float]:
    """Decompose total variance into between-pixel, within-pixel, within-site.

    Uses the mean of per-pixel variance estimates for the nested components.
    """
    pixel_stats = compute_within_pixel_stats(df, theta_bins=5)
    pixel_stats = pixel_stats.dropna(subset=["within_site_var", "between_site_var"])

    if len(pixel_stats) == 0:
        return {}

    # Grand variance of all observations
    total_var = df["log10_suction_cm"].var()

    # Between-pixel: variance of pixel means
    pixel_means = df.groupby("spatial_group")["log10_suction_cm"].mean()
    between_pixel_var = pixel_means.var()

    # Within-pixel (average across pixels)
    within_pixel_var = pixel_stats["obs_var"].mean()

    # Within-site (average across pixels of average within-site variance)
    within_site_var = pixel_stats["within_site_var"].mean()

    # Between-site within-pixel (average)
    between_site_var = pixel_stats["between_site_var"].mean()

    return {
        "total_var": total_var,
        "between_pixel_var": between_pixel_var,
        "within_pixel_var": within_pixel_var,
        "between_site_within_pixel_var": between_site_var,
        "within_site_var": within_site_var,
        "n_multi_site_pixels": len(pixel_stats),
        "n_total_pixels": df["spatial_group"].nunique(),
    }


def plot_within_pixel_histogram(
    pixel_stats: pd.DataFrame,
    output_dir: str,
    filename: str = "within_pixel_histogram.png",
) -> str:
    """Histogram of within-pixel standard deviation of log10|psi|."""
    valid = pixel_stats.dropna(subset=["theta_matched_var"])
    within_std = np.sqrt(valid["theta_matched_var"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(within_std, bins=30, edgecolor="k", linewidth=0.5, alpha=0.7)
    ax.axvline(
        within_std.median(),
        color="C1",
        linestyle="--",
        linewidth=1.5,
        label=f"median = {within_std.median():.3f}",
    )
    ax.set_xlabel(r"Within-pixel $\sigma$ of log$_{10}$|$\psi$| (theta-matched)")
    ax.set_ylabel("Count (pixels)")
    ax.set_title(f"Within-pixel variability ({len(valid)} multi-site pixels)")
    ax.legend()

    fig.tight_layout()
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def plot_nested_anova(
    anova: dict[str, float],
    output_dir: str,
    filename: str = "nested_anova_barplot.png",
) -> str:
    """Bar chart of nested ANOVA variance components."""
    components = [
        ("Between pixel", anova.get("between_pixel_var", 0)),
        ("Between site\n(within pixel)", anova.get("between_site_within_pixel_var", 0)),
        ("Within site", anova.get("within_site_var", 0)),
    ]
    labels = [c[0] for c in components]
    values = [c[1] for c in components]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(labels))
    bars = ax.bar(
        x, values, color=["C0", "C1", "C2"], alpha=0.8, edgecolor="k", linewidth=0.5
    )

    total = sum(values)
    for bar, val in zip(bars, values):
        pct = 100 * val / total if total > 0 else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Variance (log$_{10}$ cm)$^2$")
    ax.set_title("Nested ANOVA: log$_{10}$|$\\psi$| variance decomposition")

    fig.tight_layout()
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1D: within-pixel variance from existing replicates"
    )
    parser.add_argument(
        "--obs-table",
        default="/nas/soils/swapstress/training/obs_level_training_9km_global.parquet",
        help="Path to training table parquet.",
    )
    parser.add_argument(
        "--output-dir",
        default="/nas/soils/swapstress/models/direct_rf_9km_global_pruned/error_analysis",
        help="Output directory.",
    )
    parser.add_argument(
        "--resolution-m",
        type=float,
        default=250,
        help="Spatial grouping resolution in meters.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading training table...")
    df = pd.read_parquet(args.obs_table)
    df = df.dropna(subset=["theta", "log10_suction_cm", "lat", "lon"])

    from map.learning.direct.data import assign_spatial_group

    df["spatial_group"] = assign_spatial_group(df, resolution_m=args.resolution_m)
    df = df.dropna(subset=["spatial_group"])
    print(f"  {len(df)} observations, {df['spatial_group'].nunique()} spatial groups")

    # Count multi-site pixels
    sites_per_pixel = df.groupby("spatial_group")["sample_id"].nunique()
    multi_site = sites_per_pixel[sites_per_pixel >= 2]
    print(f"  {len(multi_site)} pixels with >=2 sites")

    # Ensure rosetta_level exists for depth matching
    if "rosetta_level" not in df.columns and "depth_cm" in df.columns:
        from retention_curve.depth_utils import depth_to_rosetta_level

        df["rosetta_level"] = df["depth_cm"].apply(
            lambda d: depth_to_rosetta_level(d) if pd.notna(d) else None
        )

    # Within-pixel stats (depth- and source-matched)
    print("Computing within-pixel variance (depth/source matched)...")
    pixel_stats = compute_within_pixel_stats(df, theta_bins=5)
    pixel_stats.to_csv(
        os.path.join(args.output_dir, "within_pixel_variance.csv"), index=False
    )
    if "match_type" in pixel_stats.columns:
        print(f"  Match types: {pixel_stats['match_type'].value_counts().to_dict()}")
    print(f"  Computed stats for {len(pixel_stats)} multi-site pixels")

    # Nested ANOVA
    print("Computing nested ANOVA...")
    anova = nested_anova(df)
    anova_df = pd.DataFrame([anova])
    anova_df.to_csv(os.path.join(args.output_dir, "nested_anova.csv"), index=False)

    print("\nNested ANOVA results:")
    total = (
        anova.get("between_pixel_var", 0)
        + anova.get("between_site_within_pixel_var", 0)
        + anova.get("within_site_var", 0)
    )
    for key in [
        "between_pixel_var",
        "between_site_within_pixel_var",
        "within_site_var",
    ]:
        val = anova.get(key, 0)
        pct = 100 * val / total if total > 0 else 0
        print(f"  {key}: {val:.4f} ({pct:.1f}%)")
    print(f"  Total variance: {anova.get('total_var', 0):.4f}")
    print(f"  Multi-site pixels: {anova.get('n_multi_site_pixels', 0)}")

    # Plots
    plot_within_pixel_histogram(pixel_stats, args.output_dir)
    plot_nested_anova(anova, args.output_dir)


if __name__ == "__main__":
    main()
