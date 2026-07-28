"""
Phase 1C: Conditional bias by theta-decile.

Bins holdout residuals by theta-decile and reports E[epsilon | decile] and
Var[epsilon | decile], stratified by source. Diagnoses whether prediction
errors concentrate at the wet tail (theta > 0.4), dry tail (theta < 0.05),
or are uniform across the theta domain.

Produces:
    - conditional_bias_by_decile.csv
    - conditional_bias_by_decile.png
    - conditional_bias_by_source.png
"""

from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swapstress.validation.reconstruct_test_set import MODEL_DIR, reconstruct


def compute_conditional_bias(
    test_df: pd.DataFrame,
    n_bins: int = 10,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Bin residuals by theta quantile and compute per-bin statistics.

    Parameters
    ----------
    test_df : pd.DataFrame
        Must contain 'theta', 'observed', 'predicted'.
    n_bins : int
        Number of theta quantile bins.
    group_col : str, optional
        Column to stratify by (e.g. 'source'). If None, computes overall.

    Returns
    -------
    pd.DataFrame
        One row per bin (x group), with columns: theta_lo, theta_hi,
        theta_mid, mean_bias, std_bias, rmse, mae, n, and optionally group.
    """
    df = test_df.copy()
    df["residual"] = df["predicted"] - df["observed"]

    groups = [None] if group_col is None else sorted(df[group_col].unique())
    rows = []

    for g in groups:
        sub = df if g is None else df[df[group_col] == g]
        bins = pd.qcut(sub["theta"], n_bins, duplicates="drop")

        for interval, grp in sub.groupby(bins, observed=True):
            resid = grp["residual"].values
            rows.append(
                {
                    "group": g or "all",
                    "theta_lo": interval.left,
                    "theta_hi": interval.right,
                    "theta_mid": (interval.left + interval.right) / 2,
                    "mean_bias": np.mean(resid),
                    "std_bias": np.std(resid),
                    "rmse": np.sqrt(np.mean(resid**2)),
                    "mae": np.mean(np.abs(resid)),
                    "n": len(resid),
                }
            )

    return pd.DataFrame(rows)


def plot_conditional_bias(
    bias_df: pd.DataFrame,
    output_dir: str,
    filename: str = "conditional_bias_by_decile.png",
) -> str:
    """Plot overall conditional bias with error bars."""
    overall = bias_df[bias_df["group"] == "all"].sort_values("theta_mid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: mean bias by theta decile
    ax = axes[0]
    ax.bar(
        overall["theta_mid"],
        overall["mean_bias"],
        width=overall["theta_hi"] - overall["theta_lo"],
        edgecolor="k",
        linewidth=0.5,
        alpha=0.7,
    )
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlabel(r"$\theta$ (VWC)")
    ax.set_ylabel(r"Mean bias: $\hat{y} - y$ (log$_{10}$ cm)")
    ax.set_title("Conditional bias by theta decile")

    # Right: RMSE by theta decile
    ax = axes[1]
    ax.bar(
        overall["theta_mid"],
        overall["rmse"],
        width=overall["theta_hi"] - overall["theta_lo"],
        edgecolor="k",
        linewidth=0.5,
        alpha=0.7,
        color="C1",
    )
    ax.set_xlabel(r"$\theta$ (VWC)")
    ax.set_ylabel(r"RMSE (log$_{10}$ cm)")
    ax.set_title("Conditional RMSE by theta decile")

    fig.tight_layout()
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def plot_conditional_bias_by_source(
    bias_df: pd.DataFrame,
    output_dir: str,
    filename: str = "conditional_bias_by_source.png",
) -> str:
    """Plot conditional bias per source as line plots."""
    sources = sorted(bias_df["group"].unique())
    sources = [s for s in sources if s != "all"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for src in sources:
        sub = bias_df[bias_df["group"] == src].sort_values("theta_mid")
        axes[0].plot(sub["theta_mid"], sub["mean_bias"], marker="o", ms=4, label=src)
        axes[1].plot(sub["theta_mid"], sub["rmse"], marker="o", ms=4, label=src)

    axes[0].axhline(0, color="k", linewidth=0.8, linestyle="--")
    axes[0].set_xlabel(r"$\theta$ (VWC)")
    axes[0].set_ylabel(r"Mean bias (log$_{10}$ cm)")
    axes[0].set_title("Conditional bias by source")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel(r"$\theta$ (VWC)")
    axes[1].set_ylabel(r"RMSE (log$_{10}$ cm)")
    axes[1].set_title("Conditional RMSE by source")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1C: conditional bias by theta-decile"
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
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of theta quantile bins (default: 10).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.model_dir, "error_analysis")
    os.makedirs(output_dir, exist_ok=True)

    print("Reconstructing test set...")
    test_df = reconstruct(args.model_dir)
    print(f"Test set: {len(test_df)} rows, {test_df['source'].nunique()} sources")

    # Overall conditional bias
    overall_bias = compute_conditional_bias(test_df, n_bins=args.n_bins)
    overall_bias.to_csv(
        os.path.join(output_dir, "conditional_bias_by_decile.csv"), index=False
    )

    # Per-source conditional bias
    source_bias = compute_conditional_bias(
        test_df, n_bins=args.n_bins, group_col="source"
    )
    source_bias.to_csv(
        os.path.join(output_dir, "conditional_bias_by_source.csv"), index=False
    )

    # Combine for CSV output
    all_bias = pd.concat([overall_bias, source_bias], ignore_index=True)
    print("\nConditional bias summary (overall):")
    ov = all_bias[all_bias["group"] == "all"].sort_values("theta_mid")
    for _, row in ov.iterrows():
        print(
            f"  theta [{row['theta_lo']:.3f}, {row['theta_hi']:.3f}]: "
            f"bias={row['mean_bias']:+.3f}, rmse={row['rmse']:.3f}, n={row['n']:.0f}"
        )

    # Plots
    plot_conditional_bias(all_bias, output_dir)
    plot_conditional_bias_by_source(source_bias, output_dir)


if __name__ == "__main__":
    main()
