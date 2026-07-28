"""Figure: SMAP L3 vs SWAP SWP temporal correlation and slope.

Two-panel CONUS map:
  Left:  Pearson correlation (r) between daily SMAP L3 VWC and SWAP log10 suction
  Right: Linear regression slope (d(log10 suction) / d(VWC))

Computed pixel-wise over the full swath-level record using online accumulation
(no need to hold all ~3,300 rasters in memory).

Usage:
    uv run python viz/presentation/fig_smap_swap_correlation.py
    uv run python viz/presentation/fig_smap_swap_correlation.py --start 20150401 --end 20260415
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SMAP_DIR = Path("/nas/soils/smap/SPL3SMP_E/daily_tif")
SWAP_DIR = Path(
    "/nas/soils/swapstress/releases/global_pruned_refresh_20260520/inference"
)
STATES_SHP = Path("/tmp/us_states/cb_2022_us_state_20m.shp")
EXCLUDE_STUSPS = {"AK", "HI", "AS", "GU", "MP", "PR", "VI"}


# ---------------------------------------------------------------------------
# Online accumulation
# ---------------------------------------------------------------------------


def accumulate(start: str, end: str):
    """Walk paired SMAP/SWAP tifs and accumulate Pearson statistics online."""
    smap_files = sorted(SMAP_DIR.glob("smap_sm_*.tif"))
    swap_files = {
        f.stem.replace("suction_", ""): f for f in SWAP_DIR.glob("suction_*.tif")
    }

    # Read shape from first file
    with rasterio.open(smap_files[0]) as src:
        shape = src.shape
        profile = src.profile.copy()

    # Accumulators (float64 for precision)
    n = np.zeros(shape, dtype=np.float64)
    sx = np.zeros(shape, dtype=np.float64)
    sy = np.zeros(shape, dtype=np.float64)
    sxy = np.zeros(shape, dtype=np.float64)
    sx2 = np.zeros(shape, dtype=np.float64)
    sy2 = np.zeros(shape, dtype=np.float64)

    count = 0
    for smap_path in smap_files:
        date_str = smap_path.stem.replace("smap_sm_", "")
        if date_str < start or date_str > end:
            continue
        if date_str not in swap_files:
            continue

        with rasterio.open(smap_path) as src:
            x = src.read(1).astype(np.float64)
        with rasterio.open(swap_files[date_str]) as src:
            y = src.read(1).astype(np.float64)

        # Mask invalid pixels
        valid = np.isfinite(x) & np.isfinite(y) & (y > -9998)
        x[~valid] = 0.0
        y[~valid] = 0.0
        v = valid.astype(np.float64)

        n += v
        sx += x * v
        sy += y * v
        sxy += x * y * v
        sx2 += x * x * v
        sy2 += y * y * v

        count += 1
        if count % 200 == 0:
            print(f"  processed {count} date pairs ...")

    print(f"  total: {count} date pairs")
    return n, sx, sy, sxy, sx2, sy2, profile


def compute_stats(n, sx, sy, sxy, sx2, sy2, min_obs=30):
    """Derive Pearson r and OLS slope from online accumulators."""
    safe = n >= min_obs

    mean_x = np.where(safe, sx / n, np.nan)
    mean_y = np.where(safe, sy / n, np.nan)

    var_x = np.where(safe, sx2 / n - mean_x**2, np.nan)
    var_y = np.where(safe, sy2 / n - mean_y**2, np.nan)
    cov_xy = np.where(safe, sxy / n - mean_x * mean_y, np.nan)

    denom = np.sqrt(var_x * var_y)
    r = np.where(denom > 0, cov_xy / denom, np.nan)
    slope = np.where(var_x > 0, cov_xy / var_x, np.nan)

    return r, slope


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_panels(r, slope, profile, output_dir: Path):
    """Two-panel CONUS figure: correlation and slope."""
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

    # --- Panel A: correlation ---
    # Sequential colormap: r is negative nearly everywhere, so a sequential
    # palette reveals spatial structure much better than diverging.
    r_finite = r[np.isfinite(r)]
    r_lo = np.percentile(r_finite, 1)
    r_hi = np.percentile(r_finite, 99)
    im_r = ax_r.imshow(
        r,
        extent=img_extent,
        origin="upper",
        cmap="RdYlGn",
        vmin=r_lo,
        vmax=r_hi,
        interpolation="nearest",
    )
    states.boundary.plot(ax=ax_r, color="0.3", linewidth=0.4)
    _conus_lim(ax_r, profile)
    ax_r.set_title("(a) Pearson r:  SMAP L3 VWC vs. SWAP SWP", fontsize=12, pad=8)
    cb_r = fig.colorbar(im_r, ax=ax_r, shrink=0.7, pad=0.02)
    cb_r.set_label("Pearson r", fontsize=10)

    # --- Panel B: slope ---
    # Sequential colormap: slope is strongly negative everywhere.
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
        out = output_dir / f"fig_smap_swap_correlation.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Wrote {out}")
    plt.close(fig)


def _conus_lim(ax, profile):
    """Zoom axes to CONUS bounds in the raster's CRS (EASE-Grid2 / EPSG:6933)."""
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


CACHE_PATH = Path("/tmp/smap_swap_corr_cache.npz")


def main():
    parser = argparse.ArgumentParser(
        description="SMAP L3 vs SWAP temporal correlation and slope"
    )
    parser.add_argument("--start", default="20150401")
    parser.add_argument("--end", default="20260415")
    parser.add_argument(
        "--min-obs", type=int, default=30, help="Min co-observed days per pixel"
    )
    parser.add_argument("--output-dir", default="figs/presentation")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force re-accumulation even if cache exists",
    )
    args = parser.parse_args()

    # Always need profile from a source raster
    with rasterio.open(next(SMAP_DIR.glob("smap_sm_*.tif"))) as src:
        profile = src.profile.copy()

    if not args.no_cache and CACHE_PATH.exists():
        print(f"Loading cached accumulators from {CACHE_PATH} ...")
        data = np.load(CACHE_PATH)
        n, sx, sy, sxy, sx2, sy2 = (
            data["n"],
            data["sx"],
            data["sy"],
            data["sxy"],
            data["sx2"],
            data["sy2"],
        )
    else:
        print("Accumulating pixel-wise statistics ...")
        n, sx, sy, sxy, sx2, sy2, profile = accumulate(args.start, args.end)
        print(f"Caching accumulators to {CACHE_PATH} ...")
        np.savez(CACHE_PATH, n=n, sx=sx, sy=sy, sxy=sxy, sx2=sx2, sy2=sy2)

    print("Computing correlation and slope ...")
    r, slope = compute_stats(n, sx, sy, sxy, sx2, sy2, min_obs=args.min_obs)

    print("Plotting ...")
    plot_panels(r, slope, profile, Path(args.output_dir))


if __name__ == "__main__":
    main()
