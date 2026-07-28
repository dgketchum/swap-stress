"""Figure 6b: CONUS map of estimated prediction error for a single day.

Total error at each pixel:
    error(x,y) = sqrt( model_rmse(theta)^2 + (|J(theta)| * sigma_SMAP)^2 )

where theta = SMAP L3 at that pixel, model_rmse and |J| are looked up from
the conditional bias and sensitivity analyses, and sigma_SMAP = 0.067
(ISMN-observed ubRMSE at 5 cm, n=811 stations).

Usage:
    uv run swapstress-figures --figure uncertainty
    uv run swapstress-figures --figure uncertainty -- --date 2023-07-15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from scipy.interpolate import interp1d
from swapstress.figures.basemap import states_shapefile

RELEASE_DIR = Path("/nas/soils/swapstress/releases/global_pruned_refresh_20260520")
# Stage 04 writes the error analyses beside the model, not into the release
# tree; the old release path here never existed.
ERROR_DIR = Path(
    "/nas/soils/swapstress/models/direct_rf_9km_global_pruned/error_analysis"
)
SMAP_DIR = Path("/nas/soils/smap/SPL3SMP_E/daily_tif")
# Level 1 rather than gap-filled: the mask only needs the pixels that carry a
# same-day retrieval, and this figure is about the model's own uncertainty.
PRED_DIR = RELEASE_DIR / "inference_l3"
STATES_SHP = Path(states_shapefile())

SIGMA_SMAP_PUB = 0.04  # published SMAP L3 ubRMSE
SIGMA_SMAP_OBS = 0.089  # ISMN-observed RMSE (811 stations, 5 cm, no bias correction)

CONUS_LON = (-127, -65)
CONUS_LAT = (24, 50)


def build_error_lookups():
    """Build interpolation functions for model RMSE and |J| vs theta."""
    cb = pd.read_csv(ERROR_DIR / "conditional_bias_by_decile.csv")
    cb = cb[cb["group"] == "all"].sort_values("theta_mid").reset_index(drop=True)

    ss = pd.read_csv(ERROR_DIR / "sensitivity_stats.csv")
    edges = list(cb["theta_lo"]) + [cb["theta_hi"].iloc[-1]]
    ss["bin"] = pd.cut(ss["theta"], bins=edges, labels=False)
    jac_by_bin = ss.dropna(subset=["bin"]).groupby("bin")["abs_jacobian"].median()

    theta_mid = cb["theta_mid"].values
    model_rmse = cb["rmse"].values

    # Align Jacobian to same bins
    jac_vals = np.array([jac_by_bin.get(i, np.nan) for i in range(len(theta_mid))])
    valid = np.isfinite(jac_vals)

    f_rmse = interp1d(
        theta_mid[valid],
        model_rmse[valid],
        bounds_error=False,
        fill_value=(model_rmse[valid][0], model_rmse[valid][-1]),
    )
    f_jac = interp1d(
        theta_mid[valid],
        jac_vals[valid],
        bounds_error=False,
        fill_value=(jac_vals[valid][0], jac_vals[valid][-1]),
    )
    return f_rmse, f_jac


def load_smap_composite(date_str, window=3):
    """Load SMAP theta, compositing over ±window days to fill orbital gaps."""
    from datetime import datetime, timedelta

    center = datetime.strptime(date_str, "%Y-%m-%d")
    composite = None
    count = None

    for delta in range(-window, window + 1):
        d = center + timedelta(days=delta)
        path = SMAP_DIR / f"smap_sm_{d.strftime('%Y%m%d')}.tif"
        if not path.exists():
            continue
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            crs = src.crs
            transform = src.transform
        valid = (arr > 0) & (arr < 1) & np.isfinite(arr)
        if composite is None:
            composite = np.where(valid, arr, 0.0)
            count = valid.astype(np.float32)
        else:
            composite += np.where(valid, arr, 0.0)
            count += valid.astype(np.float32)

    mean_theta = np.where(count > 0, composite / count, np.nan)
    return mean_theta, crs, transform


def main(argv=None):
    parser = argparse.ArgumentParser(description="Figure 6b: Error map")
    parser.add_argument("--date", default="2024-07-15")
    parser.add_argument("--output-dir", default="figs/descriptor")
    args = parser.parse_args(argv)
    date_str, output_dir = args.date, args.output_dir

    f_rmse, f_jac = build_error_lookups()

    # Load SMAP theta composite (±3 days to fill orbital gaps)
    smap, smap_crs, smap_transform = load_smap_composite(date_str, window=3)

    # Load suction prediction for masking (valid CONUS pixels).
    # The SMAP dailies are a CONUS window of the same EASE-Grid2 raster the
    # predictions are written on -- same CRS, same 9 km cells -- while the
    # prediction covers the globe. Read through the matching window rather than
    # resampling, so the two arrays line up cell for cell.
    pred_path = PRED_DIR / f"suction_{date_str.replace('-', '')}.tif"
    smap_bounds = rasterio.transform.array_bounds(*smap.shape, smap_transform)
    with rasterio.open(pred_path) as src:
        window = (
            from_bounds(*smap_bounds, transform=src.transform)
            .round_offsets()
            .round_lengths()
        )
        pred = src.read(1, window=window).astype(np.float32)
        pred_transform = src.window_transform(window)
        pred_crs = src.crs

    if pred.shape != smap.shape:
        raise ValueError(
            f"Windowed prediction is {pred.shape} but the SMAP composite is "
            f"{smap.shape}; the two rasters are not on the same grid."
        )

    valid_smap = np.isfinite(smap)
    valid_pred = np.isfinite(pred) & (pred > 0)
    valid = valid_smap & valid_pred

    theta = np.where(valid, smap, 0.2)

    model_err = f_rmse(theta)
    jac = f_jac(theta)
    smap_prop_obs = jac * SIGMA_SMAP_OBS
    smap_prop_pub = jac * SIGMA_SMAP_PUB

    total_err_obs = np.sqrt(model_err**2 + smap_prop_obs**2)
    total_err_pub = np.sqrt(model_err**2 + smap_prop_pub**2)
    smap_frac_obs = np.where(total_err_obs > 0, smap_prop_obs / total_err_obs, 0)
    smap_frac_pub = np.where(total_err_pub > 0, smap_prop_pub / total_err_pub, 0)

    # Mask
    total_err_obs = np.where(valid, total_err_obs, np.nan)
    total_err_pub = np.where(valid, total_err_pub, np.nan)
    smap_frac_obs = np.where(valid, smap_frac_obs, np.nan)
    smap_frac_pub = np.where(valid, smap_frac_pub, np.nan)

    # --- Plot in geographic coordinates ---
    from pyproj import Transformer as ProjTransformer

    # Reproject state boundaries to WGS84 for plotting
    states = gpd.read_file(STATES_SHP).to_crs(4326)
    exclude = {"HI", "AK", "AS", "GU", "MP", "PR", "VI"}
    states = states[~states["STUSPS"].isin(exclude)]

    # Build lon/lat grids for pcolormesh
    h, w = pred.shape
    to_wgs84 = ProjTransformer.from_crs(pred_crs, "EPSG:4326", always_xy=True)

    # Corner coordinates of each pixel row/col
    cols = np.arange(w + 1)
    rows = np.arange(h + 1)
    # Pixel corners in projected coords
    px = pred_transform.c + cols * pred_transform.a
    py = pred_transform.f + rows * pred_transform.e
    px_grid, py_grid = np.meshgrid(px, py)
    lon_grid, lat_grid = to_wgs84.transform(px_grid, py_grid)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    def style_map_ax(ax, im, title, cbar_label):
        states.boundary.plot(ax=ax, edgecolor="#aaa", linewidth=0.3, zorder=2)
        ax.set_xlim(-127, -65)
        ax.set_ylim(24, 50)
        ax.set_aspect("auto")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Longitude", fontsize=8)
        ax.set_ylabel("Latitude", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label(cbar_label, fontsize=9)
        cb.ax.tick_params(labelsize=8)

    # Panel A: SMAP error fraction at published spec
    ax = axes[0]
    im = ax.pcolormesh(
        lon_grid,
        lat_grid,
        smap_frac_pub,
        cmap="RdBu_r",
        vmin=0.0,
        vmax=1.0,
        shading="flat",
        zorder=1,
        rasterized=True,
    )
    style_map_ax(
        ax,
        im,
        "SMAP error fraction\n"
        r"$\sigma_{\theta}$ = 0.04 m$^3$m$^{-3}$ (published ubRMSE)",
        "SMAP / Total error",
    )

    # Panel B: SMAP error fraction at ISMN-observed RMSE
    ax = axes[1]
    im2 = ax.pcolormesh(
        lon_grid,
        lat_grid,
        smap_frac_obs,
        cmap="RdBu_r",
        vmin=0.0,
        vmax=1.0,
        shading="flat",
        zorder=1,
        rasterized=True,
    )
    style_map_ax(
        ax,
        im2,
        "SMAP error fraction\n"
        r"$\sigma_{\theta}$ = 0.089 m$^3$m$^{-3}$ (ISMN observed RMSE)",
        "SMAP / Total error",
    )

    fig.suptitle(f"SMAP Input Error Contribution — {date_str}", fontsize=12, y=1.01)
    fig.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fname = f"fig06_uncertainty_{date_str.replace('-', '')}"
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{fname}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {(out / f'{fname}.png').absolute()}")


if __name__ == "__main__":
    main()
