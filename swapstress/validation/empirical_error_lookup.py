"""
Section 5.7: Empirical error lookup for map interpretation.

Stratifies held-out predictions by inference-available regime variables
(theta bin, climate class, texture class) and computes RMSE, MAE, and
bias within each stratum.  Sparse cells are collapsed until estimates
are stable.  Produces a lookup table mapping regime -> expected error
that can back a simple map-side caution layer.

Produces:
    - empirical_error_lookup.csv      (per-stratum RMSE/MAE/bias)
    - error_by_theta_climate.png      (heatmap: theta bin x climate class)
"""

from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swapstress.validation.reconstruct_test_set import (
    MODEL_DIR,
    reconstruct,
    sample_beck_koppen,
)

# Minimum samples per stratum to report
MIN_STRATUM_N = 30


def assign_theta_bin(theta: pd.Series, n_bins: int = 5) -> pd.Series:
    """Quantile-bin theta into labelled categories."""
    return pd.qcut(theta, n_bins, duplicates="drop")


def assign_climate_class(df: pd.DataFrame) -> pd.Series:
    """Assign climate class using Beck et al. (2018) Koppen-Geiger.

    Falls back to latitude bands if lat/lon unavailable.
    """
    if "lat" in df.columns and "lon" in df.columns:
        valid = df[["lat", "lon"]].notna().all(axis=1)
        if valid.sum() > 0:
            _, labels, _ = sample_beck_koppen(
                df.loc[valid, "lat"].values, df.loc[valid, "lon"].values
            )
            result = pd.Series("unknown", index=df.index)
            result.loc[valid] = labels
            return result

    # Fallback: latitude bands
    if "lat" in df.columns:
        return pd.cut(
            df["lat"],
            bins=[0, 30, 37, 43, 90],
            labels=["south", "mid-south", "mid-north", "north"],
        ).astype(str)

    return pd.Series("unknown", index=df.index)


def assign_texture_class(df: pd.DataFrame) -> pd.Series:
    """Assign a coarse texture class from available soil features.

    Uses sand/clay fraction from SoilGrids features if available.
    """
    sand_cols = [c for c in df.columns if "sand" in c.lower() and "mean" in c.lower()]
    clay_cols = [c for c in df.columns if "clay" in c.lower() and "mean" in c.lower()]

    if not sand_cols or not clay_cols:
        return pd.Series("unknown", index=df.index)

    sand = df[sand_cols[0]]
    clay = df[clay_cols[0]]

    conditions = [
        sand > 65,
        clay > 35,
        (sand <= 65) & (clay <= 35),
    ]
    choices = ["coarse", "fine", "medium"]
    return pd.Series(np.select(conditions, choices, default="unknown"), index=df.index)


def compute_stratum_metrics(
    df: pd.DataFrame,
    group_cols: list[str],
    min_n: int = MIN_STRATUM_N,
) -> pd.DataFrame:
    """Compute RMSE, MAE, bias within each stratum defined by group_cols.

    Strata with fewer than min_n samples are dropped.
    """
    df = df.copy()
    df["residual"] = df["predicted"] - df["observed"]

    rows = []
    for keys, grp in df.groupby(group_cols, observed=True):
        if len(grp) < min_n:
            continue

        resid = grp["residual"].values
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))
        row["rmse"] = float(np.sqrt(np.mean(resid**2)))
        row["mae"] = float(np.mean(np.abs(resid)))
        row["bias"] = float(np.mean(resid))
        row["n"] = len(grp)
        rows.append(row)

    return pd.DataFrame(rows)


def plot_theta_climate_heatmap(
    lookup: pd.DataFrame,
    output_dir: str,
    filename: str = "error_by_theta_climate.png",
) -> str:
    """Heatmap of RMSE by theta bin and climate class."""
    if "theta_bin" not in lookup.columns or "climate_class" not in lookup.columns:
        print("Skipping heatmap: need both theta_bin and climate_class")
        return ""

    pivot = lookup.pivot_table(
        values="rmse",
        index="climate_class",
        columns="theta_bin",
        aggfunc="first",
    )

    if pivot.empty:
        print("Skipping heatmap: no data after pivoting")
        return ""

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(
        [str(c) for c in pivot.columns], rotation=45, ha="right", fontsize=8
    )
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Theta bin")
    ax.set_ylabel("Climate class")
    ax.set_title("RMSE by theta bin and climate class")

    # Annotate cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="RMSE (log10 cm)", shrink=0.8)
    fig.tight_layout()
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Section 5.7: empirical error lookup table"
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
        "--theta-bins",
        type=int,
        default=5,
        help="Number of theta quantile bins (default: 5).",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=MIN_STRATUM_N,
        help="Minimum samples per stratum (default: 30).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.model_dir, "error_analysis")
    os.makedirs(output_dir, exist_ok=True)

    print("Reconstructing test set...")
    test_df = reconstruct(args.model_dir)
    print(f"Test set: {len(test_df)} rows")

    # Assign regime variables
    test_df["theta_bin"] = assign_theta_bin(test_df["theta"], n_bins=args.theta_bins)
    test_df["climate_class"] = assign_climate_class(test_df)
    test_df["texture_class"] = assign_texture_class(test_df)

    print(f"Climate classes: {test_df['climate_class'].value_counts().to_dict()}")
    print(f"Texture classes: {test_df['texture_class'].value_counts().to_dict()}")

    # --- Marginal lookups (one variable at a time) ---
    all_lookups = []

    # By theta bin
    theta_lookup = compute_stratum_metrics(test_df, ["theta_bin"], min_n=args.min_n)
    theta_lookup["stratification"] = "theta_bin"
    all_lookups.append(theta_lookup)

    # By climate class
    climate_lookup = compute_stratum_metrics(
        test_df, ["climate_class"], min_n=args.min_n
    )
    climate_lookup["stratification"] = "climate_class"
    all_lookups.append(climate_lookup)

    # By texture class
    texture_lookup = compute_stratum_metrics(
        test_df, ["texture_class"], min_n=args.min_n
    )
    texture_lookup["stratification"] = "texture_class"
    all_lookups.append(texture_lookup)

    # By source
    source_lookup = compute_stratum_metrics(test_df, ["source"], min_n=args.min_n)
    source_lookup["stratification"] = "source"
    all_lookups.append(source_lookup)

    # --- Cross-tabulated lookups ---

    # Theta x climate
    cross_lookup = compute_stratum_metrics(
        test_df, ["theta_bin", "climate_class"], min_n=args.min_n
    )
    cross_lookup["stratification"] = "theta_bin x climate_class"
    all_lookups.append(cross_lookup)

    # Theta x texture
    cross_tex = compute_stratum_metrics(
        test_df, ["theta_bin", "texture_class"], min_n=args.min_n
    )
    cross_tex["stratification"] = "theta_bin x texture_class"
    all_lookups.append(cross_tex)

    # Combine
    lookup_df = pd.concat(all_lookups, ignore_index=True)
    lookup_path = os.path.join(output_dir, "empirical_error_lookup.csv")
    lookup_df.to_csv(lookup_path, index=False)
    print(f"\nSaved {lookup_path} ({len(lookup_df)} strata)")

    # Print summary for marginal lookups
    for strat in ["theta_bin", "climate_class", "texture_class", "source"]:
        sub = lookup_df[lookup_df["stratification"] == strat]
        if sub.empty:
            continue
        print(f"\n--- {strat} ---")
        for _, row in sub.iterrows():
            key_cols = [
                c
                for c in sub.columns
                if c not in ("rmse", "mae", "bias", "n", "stratification")
            ]
            key_str = ", ".join(
                f"{c}={row[c]}" for c in key_cols if pd.notna(row.get(c))
            )
            print(
                f"  {key_str:40s}  RMSE={row['rmse']:.3f}  MAE={row['mae']:.3f}  bias={row['bias']:+.3f}  n={row['n']:.0f}"
            )

    # Heatmap
    plot_theta_climate_heatmap(cross_lookup, output_dir)


if __name__ == "__main__":
    main()
