"""Figure 1 Panel C: SMAP L3 theta KDE.

Empirical distribution of SMAP L3 daily soil moisture across CONUS,
rendered as a kernel density estimate. Designed to stack directly below
Panel B with a shared x-axis.

Samples every 10th daily raster (~336 files, ~16.7M pixels).
Total SMAP archive: ~3,358 daily files x ~50K valid pixels ≈ 167M obs.
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

SMAP_DIR = "/nas/soils/smap/SPL3SMP_E/daily_tif"

SMAP_P01 = 0.044
SMAP_P99 = 0.582
SMAP_MEDIAN = 0.197

N_FILES_TOTAL = 3358
PIXELS_PER_FILE = 49834  # 16.7M / 336


def load_smap_values(every_nth=10):
    """Sample every nth daily raster, return flat array of valid theta."""
    files = sorted(glob.glob(f"{SMAP_DIR}/*.tif"))
    files = files[::every_nth]
    vals = []
    for f in files:
        with rasterio.open(f) as src:
            arr = src.read(1).ravel()
            arr = arr[(arr > 0) & (arr < 1) & np.isfinite(arr)]
            vals.append(arr)
    return np.concatenate(vals)


def main(output_dir):
    vals = load_smap_values(every_nth=10)
    n_total_est = N_FILES_TOTAL * PIXELS_PER_FILE

    # Subsample for KDE (full array is too large for gaussian_kde)
    rng = np.random.default_rng(42)
    kde_subset = rng.choice(vals, size=500_000, replace=False)
    kde = gaussian_kde(kde_subset, bw_method=0.012)
    x_grid = np.linspace(0.0, 0.75, 500)
    density = kde(x_grid)

    fig, ax = plt.subplots(figsize=(7.0, 1.8))

    # KDE line and fill — style matches Panel B curves
    ax.plot(x_grid, density, color="#2E86C1", linewidth=1.8, zorder=3)
    ax.fill_between(x_grid, density, color="#2E86C1", alpha=0.15, zorder=2)

    # P01 and P99 lines matching Panel B
    ax.axvline(SMAP_P01, color="#A93226", linewidth=1.0, linestyle="--", zorder=4)
    ax.axvline(SMAP_P99, color="#2E86C1", linewidth=1.0, linestyle="--", zorder=4)

    # Median
    ax.axvline(SMAP_MEDIAN, color="#444444", linewidth=0.8, linestyle=":", zorder=4)
    ymax = density.max()
    ax.text(
        SMAP_MEDIAN + 0.01,
        ymax * 0.92,
        f"median = {SMAP_MEDIAN:.2f}",
        fontsize=7.5,
        color="#444444",
        va="top",
    )

    ax.set_xlim(0.0, 0.75)
    ax.set_ylim(0, None)
    ax.set_xlabel(r"Volumetric Water Content (m$^3$ m$^{-3}$)", fontsize=10)
    ax.set_ylabel("Density", fontsize=9)
    ax.tick_params(labelsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotation: source and sample size
    ax.text(
        0.97,
        0.88,
        f"SMAP L3 daily CONUS  (n $\\approx$ {n_total_est / 1e6:.0f}M pixel-days)",
        fontsize=7.5,
        color="#5F5E5A",
        ha="right",
        va="top",
        transform=ax.transAxes,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig1_panel_c_smap_hist.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out / 'fig1_panel_c_smap_hist.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figure 1 Panel C: SMAP KDE")
    parser.add_argument(
        "--output-dir", default="figs/presentation", help="Output directory for figures"
    )
    args = parser.parse_args()
    main(args.output_dir)
