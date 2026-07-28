"""Figure: SMAP L3 vs SWAP SWP — Spearman rho and OLS slope.

Two-panel CONUS map:
  Left:  Spearman rho (rank correlation) between daily SMAP L3 VWC and SWAP suction
  Right: OLS slope (reused from Pearson cache)

Spearman is the correct metric here because the theta→suction mapping is
monotonic but nonlinear; Pearson underestimates the strength of the association.

Usage:
    uv run python viz/presentation/fig_smap_swap_spearman.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SMAP_DIR = Path("/nas/soils/smap/SPL3SMP_E/daily_tif")
SWAP_DIR = Path(
    "/nas/soils/swapstress/releases/global_pruned_refresh_20260520/inference"
)
STATES_SHP = Path("/tmp/us_states/cb_2022_us_state_20m.shp")
EXCLUDE_STUSPS = {"AK", "HI", "AS", "GU", "MP", "PR", "VI"}

CACHE_PATH = Path("/tmp/smap_swap_spearman_cache.npz")


# ---------------------------------------------------------------------------
# Accumulate all paired observations into 3-D arrays
# ---------------------------------------------------------------------------


def load_paired_stacks(start: str, end: str):
    """Load all co-observed (theta, suction) pairs into dense 3-D arrays."""
    smap_files = sorted(SMAP_DIR.glob("smap_sm_*.tif"))
    swap_lookup = {
        f.stem.replace("suction_", ""): f for f in SWAP_DIR.glob("suction_*.tif")
    }

    # Filter to date range and paired files
    pairs = []
    for sf in smap_files:
        d = sf.stem.replace("smap_sm_", "")
        if d < start or d > end:
            continue
        if d in swap_lookup:
            pairs.append((sf, swap_lookup[d]))

    n_dates = len(pairs)
    with rasterio.open(pairs[0][0]) as src:
        shape = src.shape
        profile = src.profile.copy()

    nrow, ncol = shape
    print(f"  {n_dates} date pairs, grid {nrow}x{ncol}")

    # Pre-allocate NaN arrays (float32 to keep memory reasonable)
    # ~2.5 GB each at 3300 x 295 x 634
    theta = np.full((n_dates, nrow, ncol), np.nan, dtype=np.float32)
    psi = np.full((n_dates, nrow, ncol), np.nan, dtype=np.float32)

    for t, (sf, wf) in enumerate(pairs):
        with rasterio.open(sf) as src:
            x = src.read(1)
        with rasterio.open(wf) as src:
            y = src.read(1)

        # Mask invalid
        valid = np.isfinite(x) & np.isfinite(y) & (y > -9998)
        x[~valid] = np.nan
        y[~valid] = np.nan
        theta[t] = x
        psi[t] = y

        if (t + 1) % 200 == 0:
            print(f"    loaded {t + 1} / {n_dates} ...")

    print(f"    loaded {n_dates} / {n_dates}")
    return theta, psi, profile


# ---------------------------------------------------------------------------
# Compute Spearman rho per pixel
# ---------------------------------------------------------------------------


def compute_spearman(theta, psi, min_obs=30):
    """Pixel-wise Spearman rho from 3-D (time, row, col) stacks."""
    n_dates, nrow, ncol = theta.shape
    rho = np.full((nrow, ncol), np.nan, dtype=np.float32)

    total = nrow * ncol
    done = 0
    for i in range(nrow):
        for j in range(ncol):
            x = theta[:, i, j]
            y = psi[:, i, j]
            mask = np.isfinite(x) & np.isfinite(y)
            n = mask.sum()
            if n < min_obs:
                continue
            r, _ = spearmanr(x[mask], y[mask])
            rho[i, j] = r
        done += ncol
        if (i + 1) % 50 == 0:
            print(f"    rows {i + 1}/{nrow} ({done}/{total} pixels)")

    return rho


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_panels(rho, slope, profile, output_dir: Path):
    """Two-panel CONUS figure: Spearman rho and OLS slope."""
    import geopandas as gpd

    output_dir.mkdir(parents=True, exist_ok=True)

    states = gpd.read_file(STATES_SHP)
    states = states[~states.STUSPS.isin(EXCLUDE_STUSPS)]
    states = states.to_crs(profile["crs"])

    extent = rasterio.transform.array_bounds(
        profile["height"], profile["width"], profile["transform"]
    )
    img_extent = [extent[0], extent[2], extent[1], extent[3]]

    fig, (ax_r, ax_s) = plt.subplots(2, 1, figsize=(10, 10))

    # --- Panel A: Spearman rho ---
    rho_finite = rho[np.isfinite(rho)]
    rho_lo = np.percentile(rho_finite, 1)
    rho_hi = np.percentile(rho_finite, 99)
    im_r = ax_r.imshow(
        rho,
        extent=img_extent,
        origin="upper",
        cmap="RdYlGn",
        vmin=rho_lo,
        vmax=rho_hi,
        interpolation="nearest",
    )
    states.boundary.plot(ax=ax_r, color="0.3", linewidth=0.4)
    _conus_lim(ax_r, profile)
    ax_r.set_title(
        r"(a) Spearman $\rho$:  SMAP L3 VWC vs. SWAP SWP",
        fontsize=12,
        pad=8,
    )
    cb_r = fig.colorbar(im_r, ax=ax_r, shrink=0.7, pad=0.02)
    cb_r.set_label(r"Spearman $\rho$", fontsize=10)

    # --- Panel B: slope (from Pearson cache) ---
    s_finite = slope[np.isfinite(slope)]
    vmin_s = np.percentile(s_finite, 1)
    vmax_s = np.percentile(s_finite, 99)
    im_s = ax_s.imshow(
        slope,
        extent=img_extent,
        origin="upper",
        cmap="RdYlGn",
        vmin=vmin_s,
        vmax=vmax_s,
        interpolation="nearest",
    )
    states.boundary.plot(ax=ax_s, color="0.3", linewidth=0.4)
    _conus_lim(ax_s, profile)
    ax_s.set_title(
        r"(b) Slope:  $\Delta$ log$_{10}$(suction) / $\Delta$ VWC",
        fontsize=12,
        pad=8,
    )
    cb_s = fig.colorbar(im_s, ax=ax_s, shrink=0.7, pad=0.02)
    cb_s.set_label(r"$\Delta$ log$_{10}$(cm H$_2$O) / $\Delta$ VWC", fontsize=10)

    fig.tight_layout()

    for ext in ("png", "pdf"):
        out = output_dir / f"fig_smap_swap_spearman.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Wrote {out}")
    plt.close(fig)


def _conus_lim(ax, profile):
    import pyproj

    proj = pyproj.Transformer.from_crs("EPSG:4326", profile["crs"], always_xy=True)
    x0, y0 = proj.transform(-126, 23)
    x1, y1 = proj.transform(-65, 51)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PEARSON_CACHE = Path("/tmp/smap_swap_corr_cache.npz")


def main():
    parser = argparse.ArgumentParser(description="SMAP vs SWAP Spearman rho and slope")
    parser.add_argument("--start", default="20150401")
    parser.add_argument("--end", default="20260415")
    parser.add_argument("--min-obs", type=int, default=30)
    parser.add_argument("--output-dir", default="figs/presentation")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    with rasterio.open(next(SMAP_DIR.glob("smap_sm_*.tif"))) as src:
        profile = src.profile.copy()

    # Spearman rho
    if not args.no_cache and CACHE_PATH.exists():
        print(f"Loading cached Spearman rho from {CACHE_PATH} ...")
        rho = np.load(CACHE_PATH)["rho"]
    else:
        print("Loading paired stacks ...")
        theta, psi, profile = load_paired_stacks(args.start, args.end)
        print("Computing Spearman rho per pixel ...")
        rho = compute_spearman(theta, psi, min_obs=args.min_obs)
        del theta, psi
        print(f"Caching to {CACHE_PATH} ...")
        np.savez(CACHE_PATH, rho=rho)

    # OLS slope from existing Pearson cache
    print("Loading OLS slope from Pearson cache ...")
    pdata = np.load(PEARSON_CACHE)
    n, sx, sy, sxy, sx2 = (
        pdata["n"],
        pdata["sx"],
        pdata["sy"],
        pdata["sxy"],
        pdata["sx2"],
    )
    safe = n >= args.min_obs
    mean_x = np.where(safe, sx / n, np.nan)
    mean_y = np.where(safe, sy / n, np.nan)
    var_x = np.where(safe, sx2 / n - mean_x**2, np.nan)
    cov_xy = np.where(safe, sxy / n - mean_x * mean_y, np.nan)
    slope = np.where(var_x > 0, cov_xy / var_x, np.nan)

    print("Plotting ...")
    plot_panels(rho, slope.astype(np.float32), profile, Path(args.output_dir))


if __name__ == "__main__":
    main()
