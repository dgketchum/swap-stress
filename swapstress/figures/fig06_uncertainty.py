"""Figure 6: how much of the predicted error is SMAP's, for one example day.

Total error at each pixel is the model's own error and the propagated SMAP
theta error added in quadrature:

    error(x,y) = sqrt( model_rmse(theta)^2 + (|J(theta)| * sigma_SMAP)^2 )

where theta is SMAP L3 at that pixel and model_rmse and |J| come from the
conditional-bias and sensitivity analyses. The figure maps the *share* of that
total attributable to SMAP, under the two sigma assumptions that bracket the
sensor's real performance: the published L3 ubRMSE (0.04) and the RMSE we
actually observe against ISMN (0.089, 811 stations at 5 cm).

The mapped quantity is a fraction in [0, 1] whose midpoint is meaningful --
0.5 is where the SMAP-propagated error equals the model's own -- so it is drawn
on a diverging ramp pinned to 0 and 1 with its neutral at that crossover, and
both panels share one bar so the two assumptions are directly comparable.

The valid region is swath-shaped because the underlying prediction is Level 1:
same-day retrievals only, no gap filling. That is intended. It is *not* fixed
by switching to ``inference_l4`` -- SMAP L4 assimilates PTF-derived hydraulic
parameters, so using it here would be circular.

Usage:
    uv run swapstress-figures --figure uncertainty
    uv run swapstress-figures --figure uncertainty -- --date 2024-04-15

``--date`` has to land inside the released inference year (2024): the example
day is chosen for a wide swath, not for anything the map shows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds
from scipy.interpolate import interp1d

from swapstress.figures import style
from swapstress.figures.basemap import load_conus_states, pixel_corner_lonlat

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

SIGMA_SMAP_PUB = 0.04  # published SMAP L3 ubRMSE
SIGMA_SMAP_OBS = 0.089  # ISMN-observed RMSE (811 stations, 5 cm, no bias correction)

# NAD83 / Conus Albers. The rasters are already on an equal-area grid, so
# drawing them equal-area keeps CONUS the shape readers know; plotting straight
# lon/lat would stretch the north of the country sideways.
MAP_CRS = "EPSG:5070"
MAP_PAD = 0.015

BOUNDARY_COLOR = "#4d4d4d"
BOUNDARY_WIDTH = 0.3

FIG_HEIGHT_MM = 72.0

# Written with a solidus rather than a negative exponent on purpose: the
# Helvetica/Arial clones the guide's typeface resolves to have no U+207B, so a
# literal negative exponent would drop a glyph whichever clone is installed.
THETA_UNIT = "m³/m³"


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

    # ``count`` is zero wherever no day in the window carried a valid retrieval:
    # ocean, and land the swath missed entirely. Those pixels have no mean to
    # take, so divide only where one exists. ``np.where`` would evaluate the
    # division everywhere first and warn on 0/0 before discarding the result.
    mean_theta = np.full(composite.shape, np.nan, dtype=np.float32)
    np.divide(composite, count, out=mean_theta, where=count > 0)
    return mean_theta, crs, transform


def albers_corner_mesh(transform, shape, crs):
    """Pixel *corner* mesh in Conus Albers, for a flat-shaded ``pcolormesh``.

    Corners rather than centres: flat shading wants one more node than cells in
    each direction, and centres would shift the image half a 9 km pixel.
    """
    lon, lat = pixel_corner_lonlat(transform, shape, crs)
    to_albers = Transformer.from_crs("EPSG:4326", MAP_CRS, always_xy=True)
    return to_albers.transform(lon, lat)


def draw_panel(ax, mesh_x, mesh_y, field, states, extent, title):
    """One CONUS panel: the fraction field under recessive state outlines."""
    mesh = ax.pcolormesh(
        mesh_x,
        mesh_y,
        field,
        cmap=style.DIVERGING,
        vmin=0.0,
        vmax=1.0,
        shading="flat",
        zorder=1,
        # Rasterises the 9 km cells only. Outlines, ticks and every label stay
        # vector, which is what the artwork guide actually asks for.
        rasterized=True,
    )
    states.boundary.plot(
        ax=ax, edgecolor=BOUNDARY_COLOR, linewidth=BOUNDARY_WIDTH, zorder=2
    )
    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=style.MAX_TEXT_PT, pad=2.5)
    return mesh


def padded_extent(bounds, pad=MAP_PAD):
    x0, y0, x1, y1 = bounds
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    return x0 - dx, y0 - dy, x1 + dx, y1 + dy


def main(argv=None):
    parser = argparse.ArgumentParser(description="Figure 6: SMAP error share map")
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

    smap_frac_obs = np.where(valid, smap_frac_obs, np.nan)
    smap_frac_pub = np.where(valid, smap_frac_pub, np.nan)

    style.apply()

    states = load_conus_states(crs=MAP_CRS)
    extent = padded_extent(states.total_bounds)
    mesh_x, mesh_y = albers_corner_mesh(pred_transform, pred.shape, pred_crs)

    fig = plt.figure(
        figsize=style.figsize(style.DOUBLE_COLUMN_MM, FIG_HEIGHT_MM),
        layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.0)
    axes = fig.subplots(1, 2)

    panels = (
        (smap_frac_pub, f"Published L3 ubRMSE, σ = {SIGMA_SMAP_PUB:.2f} {THETA_UNIT}"),
        (smap_frac_obs, f"ISMN-observed RMSE, σ = {SIGMA_SMAP_OBS:.3f} {THETA_UNIT}"),
    )
    mesh = None
    for ax, letter, (field, title) in zip(axes, "ab", panels):
        mesh = draw_panel(ax, mesh_x, mesh_y, field, states, extent, title)
        style.panel_label(ax, letter, dx=0.0, dy=1.0)

    axes[0].text(
        0.02,
        0.03,
        date_str,
        transform=axes[0].transAxes,
        fontsize=style.MAX_TEXT_PT - 1,
        color=style.MUTED_INK,
        ha="left",
        va="bottom",
    )

    cbar = fig.colorbar(
        mesh,
        ax=list(axes),
        orientation="horizontal",
        shrink=0.44,
        aspect=36,
        pad=0.012,
    )
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=style.MAX_TEXT_PT - 1, length=1.8, width=0.4, pad=1.5)
    cbar.outline.set_linewidth(0.4)
    # Mark the crossover the ramp's neutral already sits on, so the midpoint
    # reads as a quantity and not just as the pale end of two colours.
    cbar.ax.axvline(0.5, color=style.AXIS_COLOR, linewidth=0.5)
    cbar.set_label(
        "SMAP-attributed share of total prediction error "
        "(0.5 = SMAP and model contribute equally)",
        fontsize=style.MAX_TEXT_PT,
        labelpad=2.0,
    )

    out = Path(output_dir)
    fname = f"fig06_uncertainty_{date_str.replace('-', '')}"
    path = style.save(fig, out / fname)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
