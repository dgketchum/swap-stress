"""
Phase 1A: Distribution-shift confrontation.

At every 9 km pixel containing a training site, extracts the collocated
SMAP L3 time series and compares the marginal distribution of training theta
against the SMAP theta distribution.

Produces:
    - distribution_shift_stats.csv     (per-source KS, Wasserstein, energy distance)
    - cdf_overlay.png                  (marginal CDF: training vs. SMAP by source)
    - qq_plot.png                      (QQ plot colored by source)
    - theta_histograms.png             (histograms of training vs. SMAP theta per source)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SMAP_DIR = "/nas/soils/smap/SPL3SMP_E/daily_tif"
SMAP_RE = re.compile(r"^smap_sm_(\d{8})\.tif$")


def extract_smap_at_pixels(
    pixel_coords: pd.DataFrame,
    smap_dir: str = SMAP_DIR,
    max_files: int | None = None,
) -> dict[str, np.ndarray]:
    """Extract SMAP theta time series at training pixel locations.

    Parameters
    ----------
    pixel_coords : pd.DataFrame
        Must have 'row' and 'col' columns (EASE-Grid2 pixel indices)
        and an index of pixel identifiers.
    smap_dir : str
        Directory of daily SMAP GeoTIFFs.
    max_files : int, optional
        Limit number of files processed (for testing).

    Returns
    -------
    dict
        Mapping pixel_id -> array of valid SMAP theta values across all dates.
    """
    import rasterio

    smap_path = Path(smap_dir)
    files = sorted(smap_path.glob("smap_sm_*.tif"))
    if max_files:
        files = files[:max_files]

    pixel_ids = pixel_coords.index.tolist()
    rows = pixel_coords["row"].values
    cols = pixel_coords["col"].values
    results = {pid: [] for pid in pixel_ids}

    for i, fpath in enumerate(files):
        if i % 500 == 0:
            print(f"  Reading SMAP file {i + 1}/{len(files)}...")
        with rasterio.open(fpath) as src:
            data = src.read(1)

        for pid, r, c in zip(pixel_ids, rows, cols):
            if 0 <= r < data.shape[0] and 0 <= c < data.shape[1]:
                val = data[r, c]
                if np.isfinite(val) and 0 < val < 1:
                    results[pid].append(val)

    return {pid: np.array(vals) for pid, vals in results.items()}


def training_pixels_to_rowcol(
    training_df: pd.DataFrame,
    smap_dir: str = SMAP_DIR,
) -> pd.DataFrame:
    """Convert training lat/lon to SMAP raster row/col indices.

    Returns a DataFrame indexed by spatial_group with columns
    [lat, lon, row, col, source_list].
    """
    import rasterio

    # Get the transform from the first SMAP file
    smap_path = Path(smap_dir)
    first_file = sorted(smap_path.glob("smap_sm_*.tif"))[0]
    with rasterio.open(first_file) as src:
        transform = src.transform
        height = src.height
        width = src.width

    # Aggregate to unique spatial groups
    grouped = training_df.groupby("spatial_group").agg(
        lat=("lat", "mean"),
        lon=("lon", "mean"),
        source_list=("source", lambda x: list(x.unique())),
        n_obs=("theta", "count"),
    )

    # Project lat/lon to EASE-Grid2 and then to pixel row/col
    from pyproj import Transformer

    wgs_to_ease = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
    xs, ys = wgs_to_ease.transform(grouped["lon"].values, grouped["lat"].values)

    inv_transform = ~transform
    cols, rows = inv_transform * (xs, ys)
    grouped["row"] = np.round(rows).astype(int)
    grouped["col"] = np.round(cols).astype(int)

    # Filter to valid pixels
    valid = (
        (grouped["row"] >= 0)
        & (grouped["row"] < height)
        & (grouped["col"] >= 0)
        & (grouped["col"] < width)
    )
    grouped = grouped[valid]

    return grouped


def compute_shift_stats(
    training_theta: np.ndarray,
    smap_theta: np.ndarray,
) -> dict[str, float]:
    """Compute distribution divergence statistics between two theta arrays."""
    ks_stat, ks_p = stats.ks_2samp(training_theta, smap_theta)
    wasserstein = stats.wasserstein_distance(training_theta, smap_theta)

    # Energy distance (Szekely & Rizzo)
    n = len(training_theta)
    m = len(smap_theta)
    if n > 5000:
        rng = np.random.default_rng(42)
        training_theta = rng.choice(training_theta, 5000, replace=False)
        n = 5000
    if m > 5000:
        rng = np.random.default_rng(42)
        smap_theta = rng.choice(smap_theta, 5000, replace=False)
        m = 5000

    # Energy distance = 2*E|X-Y| - E|X-X'| - E|Y-Y'|
    a = training_theta[:, None] - smap_theta[None, :]
    b = training_theta[:, None] - training_theta[None, :]
    c = smap_theta[:, None] - smap_theta[None, :]
    energy = 2 * np.mean(np.abs(a)) - np.mean(np.abs(b)) - np.mean(np.abs(c))

    return {
        "ks_stat": ks_stat,
        "ks_p": ks_p,
        "wasserstein": wasserstein,
        "energy_distance": max(energy, 0.0),
        "train_mean": np.mean(training_theta),
        "smap_mean": np.mean(smap_theta),
        "train_std": np.std(training_theta),
        "smap_std": np.std(smap_theta),
        "n_train": len(training_theta),
        "n_smap": len(smap_theta),
    }


def plot_cdf_overlay(
    source_data: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: str,
    filename: str = "cdf_overlay.png",
) -> str:
    """Plot per-source CDF of training theta vs. SMAP theta."""
    n_sources = len(source_data)
    fig, axes = plt.subplots(1, n_sources, figsize=(4 * n_sources, 4), squeeze=False)

    for i, (source, (train_theta, smap_theta)) in enumerate(
        sorted(source_data.items())
    ):
        ax = axes[0, i]
        t_sorted = np.sort(train_theta)
        s_sorted = np.sort(smap_theta)
        ax.plot(
            t_sorted,
            np.linspace(0, 1, len(t_sorted)),
            label="Training",
            linewidth=1.5,
        )
        ax.plot(
            s_sorted,
            np.linspace(0, 1, len(s_sorted)),
            label="SMAP L3",
            linewidth=1.5,
            linestyle="--",
        )
        ax.set_xlabel(r"$\theta$ (m$^3$/m$^3$)")
        ax.set_ylabel("CDF")
        ax.set_title(source)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 0.6)

    fig.suptitle("Training vs. SMAP theta CDFs", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def plot_qq(
    source_data: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: str,
    filename: str = "qq_plot.png",
) -> str:
    """QQ plot of training vs. SMAP theta quantiles, colored by source."""
    fig, ax = plt.subplots(figsize=(6, 6))
    quantiles = np.linspace(0.01, 0.99, 99)

    for source, (train_theta, smap_theta) in sorted(source_data.items()):
        tq = np.quantile(train_theta, quantiles)
        sq = np.quantile(smap_theta, quantiles)
        ax.scatter(sq, tq, s=10, alpha=0.6, label=source)

    lo, hi = 0, 0.6
    ax.plot([lo, hi], [lo, hi], "k-", linewidth=0.8)
    ax.set_xlabel(r"SMAP L3 $\theta$ quantiles")
    ax.set_ylabel(r"Training $\theta$ quantiles")
    ax.set_title("QQ: Training vs. SMAP theta")
    ax.legend(fontsize=8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    fig.tight_layout()
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def plot_histograms(
    source_data: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: str,
    filename: str = "theta_histograms.png",
) -> str:
    """Side-by-side histograms of training and SMAP theta per source."""
    n_sources = len(source_data)
    fig, axes = plt.subplots(1, n_sources, figsize=(4 * n_sources, 4), squeeze=False)
    bins = np.linspace(0, 0.6, 50)

    for i, (source, (train_theta, smap_theta)) in enumerate(
        sorted(source_data.items())
    ):
        ax = axes[0, i]
        ax.hist(
            train_theta,
            bins=bins,
            density=True,
            alpha=0.5,
            label="Training",
        )
        ax.hist(
            smap_theta,
            bins=bins,
            density=True,
            alpha=0.5,
            label="SMAP L3",
        )
        ax.set_xlabel(r"$\theta$ (m$^3$/m$^3$)")
        ax.set_ylabel("Density")
        ax.set_title(source)
        ax.legend(fontsize=8)

    fig.suptitle("Training vs. SMAP theta distributions", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def main():
    from error_analysis.reconstruct_test_set import MODEL_DIR

    parser = argparse.ArgumentParser(
        description="Section 5.4: distribution-shift confrontation"
    )
    parser.add_argument(
        "--model-dir",
        default=MODEL_DIR,
        help="Path to trained model directory (for config and test-set cache).",
    )
    parser.add_argument(
        "--smap-dir",
        default=SMAP_DIR,
        help="Directory of daily SMAP GeoTIFFs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <model-dir>/error_analysis/).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit SMAP files processed (for testing).",
    )
    parser.add_argument(
        "--resolution-m",
        type=float,
        default=9000,
        help="Spatial grouping resolution in meters (default: 9000 = SMAP pixel).",
    )
    args = parser.parse_args()

    model_path = Path(args.model_dir)
    output_dir = args.output_dir or os.path.join(args.model_dir, "error_analysis")
    os.makedirs(output_dir, exist_ok=True)

    from error_analysis.reconstruct_test_set import _get_resolution_m
    from map.learning.direct.data import assign_spatial_group, prepare_direct_data

    with open(model_path / "direct_model_results.json") as f:
        config = json.load(f)["config"]

    # Reproduce the exact train/test split to get the training subset
    resolution_m = _get_resolution_m(args.model_dir)
    print(f"Reproducing split with resolution_m={resolution_m}...")
    data = prepare_direct_data(
        obs_table_path=config["obs_table"],
        output_dir="/tmp/distshift_scratch",
        exclude_groups=config.get("exclude_groups"),
        drop_blocking_features=config.get("drop_blocking_features", True),
        resolution_m=resolution_m,
        test_size=config.get("test_size", 0.2),
        random_state=config.get("random_state", 42),
    )
    df = data["train_df"].copy()
    print(
        f"  Using {len(df)} training rows (excluded {len(data['test_df'])} test rows)"
    )

    df["spatial_group"] = assign_spatial_group(df, resolution_m=args.resolution_m)
    df = df.dropna(subset=["spatial_group", "theta"])
    print(
        f"  {len(df)} training observations across "
        f"{df['spatial_group'].nunique()} spatial groups"
    )

    print("Mapping training pixels to SMAP row/col...")
    pixel_coords = training_pixels_to_rowcol(df, smap_dir=args.smap_dir)
    print(f"  {len(pixel_coords)} unique spatial groups mapped to SMAP pixels")

    # Deduplicate to unique SMAP pixels (multiple spatial groups may map to same pixel)
    pixel_coords["pixel_key"] = (
        pixel_coords["row"].astype(str) + "_" + pixel_coords["col"].astype(str)
    )
    unique_pixels = pixel_coords.drop_duplicates(subset="pixel_key")
    print(f"  {len(unique_pixels)} unique SMAP pixels to extract")

    print(f"Extracting SMAP time series at {len(unique_pixels)} pixels...")
    smap_ts = extract_smap_at_pixels(
        unique_pixels, smap_dir=args.smap_dir, max_files=args.max_files
    )

    # Map smap timeseries back to spatial groups via pixel_key
    pixel_key_to_smap = {}
    for pid, vals in smap_ts.items():
        pk = pixel_coords.loc[pid, "pixel_key"]
        if pk not in pixel_key_to_smap or len(vals) > len(pixel_key_to_smap[pk]):
            pixel_key_to_smap[pk] = vals

    group_to_smap = {}
    for sg in pixel_coords.index:
        pk = pixel_coords.loc[sg, "pixel_key"]
        if pk in pixel_key_to_smap:
            group_to_smap[sg] = pixel_key_to_smap[pk]

    # Aggregate per source: pool training theta and SMAP theta across all
    # spatial groups that contain that source
    sources = sorted(df["source"].unique())
    source_data = {}
    stats_rows = []

    for source in sources:
        source_df = df[df["source"] == source]
        source_groups = source_df["spatial_group"].unique()

        train_theta = source_df["theta"].values
        smap_values = []
        for sg in source_groups:
            if sg in group_to_smap:
                smap_values.append(group_to_smap[sg])

        if not smap_values:
            print(f"  {source}: no SMAP data at training pixels, skipping")
            continue

        smap_theta = np.concatenate(smap_values)
        source_data[source] = (train_theta, smap_theta)

        shift_stats = compute_shift_stats(train_theta, smap_theta)
        shift_stats["source"] = source
        stats_rows.append(shift_stats)

        print(
            f"  {source}: KS={shift_stats['ks_stat']:.3f}, "
            f"W1={shift_stats['wasserstein']:.4f}, "
            f"train_mean={shift_stats['train_mean']:.3f}, "
            f"smap_mean={shift_stats['smap_mean']:.3f}"
        )

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(
        os.path.join(output_dir, "distribution_shift_stats.csv"), index=False
    )
    print("\nSaved distribution shift statistics")

    # Plots
    plot_cdf_overlay(source_data, output_dir)
    plot_qq(source_data, output_dir)
    plot_histograms(source_data, output_dir)


if __name__ == "__main__":
    main()
