"""Figure 8: Training theta vs SMAP theta distribution shift (per source).

Five-panel figure with overlaid density histograms and Wasserstein distance.

Usage:
    python -m research.figures.presentation.fig8_distribution_shift \
        --obs-table /nas/soils/swapstress/training/obs_level_training_9km_global.parquet \
        --output-dir figs/presentation \
        --smap-subsample 10
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SMAP_DIR = "/nas/soils/smap/SPL3SMP_E/daily_tif"

SOURCE_DISPLAY = {
    "gshp": "GSHP",
    "ncss": "NCSS",
    "mt_mesonet": "MT Mesonet",
    "reesh": "ReESH",
    "lacadian": "LaCADIAN",
}

# Order for panels (most obs to least, roughly)
SOURCE_ORDER = ["gshp", "ncss", "mt_mesonet", "reesh", "lacadian"]


def training_theta_by_source(obs_table: str) -> dict[str, np.ndarray]:
    """Load training table and return theta arrays keyed by source."""
    df = pd.read_parquet(obs_table, columns=["source", "theta"])
    df = df.dropna(subset=["theta"])
    return {src: grp["theta"].values for src, grp in df.groupby("source")}


def smap_theta_at_training_pixels(
    obs_table: str,
    smap_dir: str = SMAP_DIR,
    subsample: int = 10,
) -> dict[str, np.ndarray]:
    """Extract SMAP theta at training pixel locations, pooled by source.

    Reads every `subsample`-th SMAP file for speed.
    """
    import rasterio
    from pyproj import Transformer

    df = pd.read_parquet(obs_table, columns=["source", "lat", "lon"])
    df = df.dropna(subset=["lat", "lon"])

    # Get SMAP grid info
    smap_path = Path(smap_dir)
    files = sorted(smap_path.glob("smap_sm_*.tif"))
    files = files[::subsample]
    print(f"  Reading {len(files)} SMAP files (every {subsample}th)...")

    first = files[0]
    with rasterio.open(first) as src:
        transform = src.transform
        height, width = src.height, src.width

    # Unique source-pixel combos
    wgs_to_ease = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
    xs, ys = wgs_to_ease.transform(df["lon"].values, df["lat"].values)
    inv_t = ~transform
    cols, rows = inv_t * (xs, ys)
    df["row"] = np.round(rows).astype(int)
    df["col"] = np.round(cols).astype(int)

    valid = (
        (df["row"] >= 0) & (df["row"] < height) & (df["col"] >= 0) & (df["col"] < width)
    )
    df = df[valid]

    # Unique pixels with their source(s)
    pixel_source = (
        df.groupby(["row", "col"])
        .agg(sources=("source", lambda x: set(x)))
        .reset_index()
    )

    pixel_rows = pixel_source["row"].values
    pixel_cols = pixel_source["col"].values
    pixel_sources = pixel_source["sources"].values
    n_pixels = len(pixel_source)
    print(f"  {n_pixels} unique SMAP pixels to sample")

    # Accumulate per-source
    source_vals = {src: [] for src in df["source"].unique()}

    for i, fpath in enumerate(files):
        if i % 100 == 0:
            print(f"  File {i + 1}/{len(files)}...")
        with rasterio.open(fpath) as src:
            data = src.read(1)

        for r, c, srcs in zip(pixel_rows, pixel_cols, pixel_sources):
            val = data[r, c]
            if np.isfinite(val) and 0 < val < 1:
                for s in srcs:
                    source_vals[s].append(val)

    return {s: np.array(v) for s, v in source_vals.items() if len(v) > 0}


def _kde(arr, grid):
    """Gaussian KDE evaluated on grid, handling edge cases."""
    arr = arr[np.isfinite(arr)]
    if len(arr) < 10 or arr.std() < 1e-8:
        return np.zeros_like(grid)
    kernel = stats.gaussian_kde(arr, bw_method="scott")
    return kernel(grid)


TRAIN_COLOR = "#2166ac"
SMAP_COLOR = "#b2182b"


def plot_distribution_shift(
    train_theta: dict[str, np.ndarray],
    smap_theta: dict[str, np.ndarray],
    output_dir: str,
):
    """Five-panel KDE figure in a 2x3 grid (6th cell = legend)."""
    sources = [s for s in SOURCE_ORDER if s in train_theta and s in smap_theta]

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    grid = np.linspace(0, 0.65, 300)

    for idx, src in enumerate(sources):
        row, col = divmod(idx, 3)
        ax = axes[row, col]
        tt = train_theta[src]
        st = smap_theta[src]

        kde_train = _kde(tt, grid)
        kde_smap = _kde(st, grid)

        ax.plot(grid, kde_train, color=TRAIN_COLOR, linewidth=1.2)
        ax.fill_between(grid, kde_train, alpha=0.25, color=TRAIN_COLOR)
        ax.plot(grid, kde_smap, color=SMAP_COLOR, linewidth=1.2)
        ax.fill_between(grid, kde_smap, alpha=0.25, color=SMAP_COLOR)

        in_situ_sources = {"lacadian"}
        obs_label = "In-Situ Obs" if src in in_situ_sources else "Lab Obs"
        n_lab = f"{obs_label} = {len(tt):,}"
        n_smap = f"SMAP L3 Captures = {len(st):,}"
        ax.text(
            0.97,
            0.95,
            n_lab + "\n" + n_smap,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="0.7", alpha=0.85),
        )

        ax.set_title(SOURCE_DISPLAY.get(src, src), fontsize=11, fontweight="bold")
        ax.set_xlim(0, 0.65)
        ax.set_ylim(bottom=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if col == 0:
            ax.set_ylabel("Density", fontsize=10)
        if row == 1:
            ax.set_xlabel(r"$\theta$ (m$^3$ m$^{-3}$)", fontsize=10)

    # 6th cell: legend
    ax_leg = axes[1, 2]
    ax_leg.axis("off")
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=TRAIN_COLOR, linewidth=2.0),
        Line2D([0], [0], color=SMAP_COLOR, linewidth=2.0),
    ]
    labels = [
        r"Lab-Measured $\theta$–$\psi$ Pairs (Training Data)",
        r"SMAP L3 Captures Over Sample Sites",
    ]
    ax_leg.legend(handles, labels, loc="center", fontsize=11, frameon=False)

    fig.suptitle("Training vs. SMAP L3 Theta Distribution by Source", fontsize=13)
    plt.tight_layout()

    out_png = os.path.join(output_dir, "fig8_distribution_shift.png")
    out_pdf = os.path.join(output_dir, "fig8_distribution_shift.pdf")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


CACHE_PATH = "/tmp/fig8_distshift_cache.npz"


def _save_cache(train_theta, smap_theta):
    arrays = {}
    for s, arr in train_theta.items():
        arrays[f"train_{s}"] = arr
    for s, arr in smap_theta.items():
        arrays[f"smap_{s}"] = arr
    np.savez(CACHE_PATH, **arrays)
    print(f"  Cached to {CACHE_PATH}")


def _load_cache():
    if not os.path.exists(CACHE_PATH):
        return None, None
    data = np.load(CACHE_PATH)
    train, smap = {}, {}
    for key in data.files:
        if key.startswith("train_"):
            train[key[6:]] = data[key]
        elif key.startswith("smap_"):
            smap[key[5:]] = data[key]
    return train, smap


def main():
    parser = argparse.ArgumentParser(
        description="Figure 8: distribution shift histograms"
    )
    parser.add_argument("--obs-table", default=None)
    parser.add_argument("--output-dir", default="figs/presentation")
    parser.add_argument("--smap-dir", default=SMAP_DIR)
    parser.add_argument(
        "--smap-subsample",
        type=int,
        default=10,
        help="Read every Nth SMAP file (default: 10)",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Load cached arrays if available, skip SMAP extraction",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_cache:
        train_theta, smap_theta = _load_cache()
        if train_theta is not None:
            print(f"Loaded cache from {CACHE_PATH}")
        else:
            print("No cache found, running extraction...")
            args.use_cache = False

    if not args.use_cache:
        if not args.obs_table:
            parser.error("--obs-table is required when not using --use-cache")
        print("Loading training theta...")
        train_theta = training_theta_by_source(args.obs_table)
        for s, arr in sorted(train_theta.items()):
            print(f"  {s}: {len(arr)} obs, mean={arr.mean():.3f}")

        print("Extracting SMAP theta at training pixels...")
        smap_theta = smap_theta_at_training_pixels(
            args.obs_table,
            smap_dir=args.smap_dir,
            subsample=args.smap_subsample,
        )
        for s, arr in sorted(smap_theta.items()):
            print(f"  {s}: {len(arr)} SMAP values, mean={arr.mean():.3f}")

        _save_cache(train_theta, smap_theta)

    print("Plotting...")
    plot_distribution_shift(train_theta, smap_theta, args.output_dir)


if __name__ == "__main__":
    main()
