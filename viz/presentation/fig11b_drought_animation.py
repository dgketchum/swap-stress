"""Animated drought time series at an Ameriflux/ReESH site.

Left panel : State-level suction map (daily, from gap-filled RF product)
Right stack: 4 time series panels drawn progressively —
    1. Daily precipitation (bars, inverted axis)
    2. Volumetric water content (observed sensor + SMAP L3)
    3. Soil water potential (observed sensor mean + predicted RF)
    4. Evapotranspiration (observed, mm/day)

Vertical "now" line sweeps across the time series. Output: MP4.

Usage:
    uv run python viz/presentation/fig11b_drought_animation.py --site US-Jo2
    uv run python viz/presentation/fig11b_drought_animation.py --site US-SRM --fps 15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.windows
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from pyproj import Transformer
from rasterio.transform import array_bounds

# ── Site configs ──────────────────────────────────────────────────────
SITE_CONFIGS = {
    "US-SRM": {
        "lon": -110.8661,
        "lat": 31.8214,
        "obs_path": "/nas/soils/swapstress/reesh_site_analysis/data/US_SRM_5cm.parquet",
        "label": "Santa Rita Mesquite, AZ",
        "vwc_label": "In-situ VWC (5 cm)",
        "date_start": "2020-01-01",
        "date_end": "2020-12-31",
        "map_lon": (-115.0, -109.0),
        "map_lat": (31.3, 37.0),
        "psi_ylim": (1.5, 5.5),
        "vwc_ylim": (0.0, 0.16),
        "et_ylim": (0.0, 7.0),
    },
    "US-GLE": {
        "lon": -106.2399,
        "lat": 41.3665,
        "obs_path": "/nas/soils/swapstress/reesh_site_analysis/data/US_GLE_5cm.parquet",
        "label": "Glacier Lakes, WY (alpine)",
        "vwc_label": "In-situ VWC (5 cm)",
        "date_start": "2020-03-01",
        "date_end": "2020-11-15",
        "map_lon": (-111.5, -104.0),
        "map_lat": (40.5, 45.5),
        "psi_ylim": (1.5, 7.0),
        "vwc_ylim": (0.0, 0.55),
        "et_ylim": (0.0, 5.0),
    },
    "US-Ho1": {
        "lon": -68.7402,
        "lat": 45.2041,
        "obs_path": "/nas/soils/swapstress/reesh_site_analysis/data/US_Ho1_5cm.parquet",
        "label": "Howland Forest, ME",
        "vwc_label": "In-situ VWC (5 cm)",
        "date_start": "2020-03-01",
        "date_end": "2020-11-15",
        "map_lon": (-71.5, -67.0),
        "map_lat": (43.0, 47.5),
        "psi_ylim": (1.5, 4.5),
        "vwc_ylim": (0.0, 0.45),
        "et_ylim": (0.0, 5.0),
    },
    "US-Jo2": {
        "lon": -106.6032,
        "lat": 32.5849,
        "obs_path": "/nas/soils/swapstress/reesh_site_analysis/data/US_Jo2_5cm.parquet",
        "label": "Jornada, NM (desert)",
        "vwc_label": "In-situ VWC (5 cm)",
        "date_start": "2018-01-01",
        "date_end": "2018-12-31",
        "map_lon": (-109.5, -103.0),
        "map_lat": (31.0, 37.5),
        "psi_ylim": (1.5, 5.5),
        "vwc_ylim": (0.0, 0.22),
        "et_ylim": (0.0, 5.0),
    },
    "US-UTW": {
        "lon": -110.7290,
        "lat": 39.4454,
        "obs_path": "/nas/soils/swapstress/reesh_site_analysis/data/US_UTW_5cm.parquet",
        "label": "UT Wetland",
        "vwc_label": "In-situ VWC (5 cm)",
        "date_start": "2022-03-01",
        "date_end": "2022-11-15",
        "map_lon": (-114.5, -109.0),
        "map_lat": (37.0, 42.5),
        "psi_ylim": (1.5, 5.0),
        "vwc_ylim": (0.10, 0.45),
        "et_ylim": (0.0, 5.0),
    },
    "US-MMS": {
        "lon": -86.4131,
        "lat": 39.3232,
        "obs_path": "/nas/soils/swapstress/reesh_site_analysis/data/US_MMS.parquet",
        "label": "Morgan-Monroe State Forest, IN",
        "vwc_label": "In-situ VWC (profile avg)",
        "date_start": "2020-03-01",
        "date_end": "2020-11-15",
        "map_lon": (-88.5, -84.5),
        "map_lat": (37.5, 42.0),
        "psi_ylim": (1.8, 4.5),
        "vwc_ylim": (0.05, 0.50),
        "et_ylim": (0.0, 6.0),
    },
}

# ── Shared paths ──────────────────────────────────────────────────────
PRED_DIR = Path("/nas/soils/swapstress/releases/global_pruned_refresh_20260520/gapfill")
SMAP_DIR = Path("/nas/soils/smap/SPL3SMP_E/daily_tif")
STATES_SHP = Path("/tmp/us_states/cb_2022_us_state_20m.shp")

NODATA = -9999.0
VMIN, VMAX = 1.2, 4.8

# Suction thresholds (log10 cm)
FC_LOG = 2.5
PWP_LOG = 4.18

# ── Colormap (same as fig11) ─────────────────────────────────────────
_CMAP_COLORS = [
    (0.00, "#1B4F72"),
    (0.15, "#2E86C1"),
    (0.30, "#27AE60"),
    (0.50, "#D4AC0D"),
    (0.70, "#E67E22"),
    (0.85, "#C0392B"),
    (1.00, "#641E16"),
]
CMAP = LinearSegmentedColormap.from_list("suction", [(p, c) for p, c in _CMAP_COLORS])

FPS = 12
EXCLUDE_STUSPS = {"AK", "HI", "AS", "GU", "MP", "PR", "VI"}


# ══════════════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════════════


def _pixel_coords(raster_path, lon, lat):
    """Return (row, col) for a site in a raster's grid."""
    with rasterio.open(raster_path) as src:
        tfm = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x, y = tfm.transform(lon, lat)
        return src.index(x, y)


def _map_window(raster_path, map_lon, map_lat):
    """Return rasterio Window for the regional map extent."""
    with rasterio.open(raster_path) as src:
        tfm = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x0, y0 = tfm.transform(map_lon[0], map_lat[0])
        x1, y1 = tfm.transform(map_lon[1], map_lat[1])
        r0, c0 = src.index(x0, y1)
        r1, c1 = src.index(x1, y0)
        return rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0), src.crs, r0, c0


def load_observations(obs_path=None):
    """Load observed VWC, SWP, precip, ET from the ReESH parquet."""
    df = pd.read_parquet(obs_path)
    psi_cols = [c for c in df.columns if c.startswith("psi_cm")]
    df["psi_mean"] = df[psi_cols].mean(axis=1)
    df["log_psi"] = np.log10(df["psi_mean"])
    # ET: convert m/day → mm/day
    if "ET" in df.columns:
        df["et_mm"] = df["ET"] * 1000.0
    else:
        df["et_mm"] = np.nan
    return df


def extract_predictions(dates, row, col):
    """Extract predicted log10 suction at MMS pixel for given dates."""
    result = {}
    for d in dates:
        path = PRED_DIR / f"suction_{d.strftime('%Y%m%d')}.tif"
        if path.exists():
            with rasterio.open(path) as src:
                v = src.read(1)[row, col]
                if v != NODATA and np.isfinite(v):
                    result[d] = v
    return pd.Series(result, name="pred_log_psi")


def extract_smap(dates, row, col):
    """Extract SMAP L3 theta at MMS pixel for given dates."""
    result = {}
    for d in dates:
        path = SMAP_DIR / f"smap_sm_{d.strftime('%Y%m%d')}.tif"
        if path.exists():
            with rasterio.open(path) as src:
                v = src.read(1)[row, col]
                if np.isfinite(v) and v > 0:
                    result[d] = float(v)
    return pd.Series(result, name="smap_theta")


def load_map_frame(date, window):
    """Load the regional suction raster window for one date."""
    path = PRED_DIR / f"suction_{date.strftime('%Y%m%d')}.tif"
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        data = src.read(1, window=window).astype(np.float32)
    data[(data == NODATA) | ~np.isfinite(data)] = np.nan
    return data


def load_states(crs):
    """Load CONUS state boundaries reprojected to raster CRS."""
    states = gpd.read_file(STATES_SHP)
    conus = states[~states.STUSPS.isin(EXCLUDE_STUSPS)].copy()
    return conus.to_crs(crs)


# ══════════════════════════════════════════════════════════════════════
#  Animation
# ══════════════════════════════════════════════════════════════════════


def build_animation(site_key, output_dir, fps):
    cfg = SITE_CONFIGS[site_key]
    site_lon, site_lat = cfg["lon"], cfg["lat"]
    obs_path = Path(cfg["obs_path"])
    date_start, date_end = cfg["date_start"], cfg["date_end"]
    map_lon, map_lat = cfg["map_lon"], cfg["map_lat"]
    psi_ylim = cfg["psi_ylim"]
    vwc_ylim = cfg["vwc_ylim"]
    et_ylim = cfg.get("et_ylim", (0.0, 5.0))
    site_label = cfg["label"]
    vwc_label = cfg["vwc_label"]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load all data ─────────────────────────────────────────────────
    print("Loading observations...", flush=True)
    obs_full = load_observations(obs_path)
    obs = obs_full[date_start:date_end].copy()
    all_dates = pd.date_range(date_start, date_end, freq="D")

    ref_path = PRED_DIR / "suction_20200701.tif"
    row, col = _pixel_coords(ref_path, site_lon, site_lat)
    window, crs, r0, c0 = _map_window(ref_path, map_lon, map_lat)

    print("Extracting predictions...", flush=True)
    pred = extract_predictions(all_dates, row, col)

    print("Extracting SMAP...", flush=True)
    smap = extract_smap(all_dates, row, col)

    # Map extent in projected coords
    with rasterio.open(ref_path) as src:
        win_transform = rasterio.windows.transform(window, src.transform)
        h, w = window.height, window.width
        left, bottom, right, top = array_bounds(h, w, win_transform)
        # Site marker in projected coords
        tfm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        site_x, site_y = tfm.transform(site_lon, site_lat)

    states = load_states(crs)

    # ── Figure layout ─────────────────────────────────────────────────
    # Panel order: Precip, VWC, SWP, ET
    fig = plt.figure(figsize=(14, 8), dpi=150)  # 2100x1200, both even
    gs = fig.add_gridspec(
        4,
        2,
        width_ratios=[1, 1.8],
        height_ratios=[1, 2, 3, 1.5],
        hspace=0.08,
        wspace=0.25,
        left=0.06,
        right=0.96,
        top=0.93,
        bottom=0.06,
    )

    # Left: map spans all 4 rows
    ax_map = fig.add_subplot(gs[:, 0])
    # Right: 4 stacked panels (Precip → VWC → SWP → ET)
    ax_pr = fig.add_subplot(gs[0, 1])
    ax_vwc = fig.add_subplot(gs[1, 1], sharex=ax_pr)
    ax_swp = fig.add_subplot(gs[2, 1], sharex=ax_pr)
    ax_et = fig.add_subplot(gs[3, 1], sharex=ax_pr)

    ts_axes = [ax_pr, ax_vwc, ax_swp, ax_et]

    # ── Map panel (static elements) ──────────────────────────────────
    blank_map = np.full((h, w), np.nan, dtype=np.float32)
    map_extent = [left, right, bottom, top]
    im_map = ax_map.imshow(
        blank_map,
        cmap=CMAP,
        vmin=VMIN,
        vmax=VMAX,
        extent=map_extent,
        origin="upper",
        interpolation="nearest",
    )
    states.boundary.plot(ax=ax_map, edgecolor="#2C2C2A", linewidth=0.5)
    ax_map.plot(site_x, site_y, "k^", ms=7, zorder=10)
    ax_map.text(
        site_x + (right - left) * 0.03,
        site_y,
        site_key,
        fontsize=8,
        fontweight="bold",
        va="center",
        zorder=10,
    )
    ax_map.set_xlim(left, right)
    ax_map.set_ylim(bottom, top)
    ax_map.set_aspect("equal")
    ax_map.set_axis_off()
    map_title = ax_map.set_title("", fontsize=10, pad=4)

    cb = fig.colorbar(im_map, ax=ax_map, fraction=0.045, pad=0.02, shrink=0.6)
    cb.set_label(r"log$_{10}$ suction (cm H$_2$O)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # ── Precip panel (row 0) ─────────────────────────────────────────
    ax_pr.set_ylabel("Precip\n(mm)", fontsize=8)
    pr_max = obs["pr"].max() * 1.15 if obs["pr"].notna().any() else 80
    ax_pr.set_ylim(0, pr_max)
    ax_pr.invert_yaxis()
    ax_pr.tick_params(labelsize=7)
    plt.setp(ax_pr.get_xticklabels(), visible=False)

    # Pre-draw all precip bars (revealed progressively)
    pr_bars = ax_pr.bar(
        obs.index,
        obs["pr"].fillna(0),
        width=1.0,
        color="#3498DB",
        alpha=0.0,
        edgecolor="none",
    )

    # ── VWC panel (row 1) ────────────────────────────────────────────
    ax_vwc.set_ylabel("VWC\n(m³/m³)", fontsize=8)
    ax_vwc.set_ylim(*vwc_ylim)
    ax_vwc.tick_params(labelsize=7)
    ax_vwc.grid(axis="y", alpha=0.12)
    plt.setp(ax_vwc.get_xticklabels(), visible=False)

    (line_obs_vwc,) = ax_vwc.plot([], [], color="#2C3E50", lw=1.4, label=vwc_label)
    smap_scatter = ax_vwc.scatter(
        [],
        [],
        s=8,
        color="#2E86C1",
        alpha=0.7,
        zorder=5,
        label="SMAP L3 (0\u20135 cm)",
    )
    ax_vwc.legend(fontsize=6.5, loc="upper right", framealpha=0.85)

    # ── SWP panel (row 2) ────────────────────────────────────────────
    ax_swp.set_ylabel(r"log$_{10}$ $\psi$" + "\n(cm)", fontsize=8)
    ax_swp.set_ylim(*psi_ylim)
    ax_swp.axhline(FC_LOG, color="#27AE60", ls=":", lw=0.8, alpha=0.6)
    ax_swp.axhline(PWP_LOG, color="#C0392B", ls=":", lw=0.8, alpha=0.6)
    ax_swp.text(
        all_dates[1],
        FC_LOG - 0.07,
        "FC",
        fontsize=6.5,
        color="#27AE60",
        va="top",
    )
    ax_swp.text(
        all_dates[1],
        PWP_LOG + 0.07,
        "PWP",
        fontsize=6.5,
        color="#C0392B",
        va="bottom",
    )
    ax_swp.tick_params(labelsize=7)
    ax_swp.grid(axis="y", alpha=0.12)
    plt.setp(ax_swp.get_xticklabels(), visible=False)

    (line_obs_psi,) = ax_swp.plot(
        [], [], color="#2C3E50", lw=1.4, label="Observed (0 cm VG)"
    )
    (line_pred_psi,) = ax_swp.plot(
        [],
        [],
        color="#E74C3C",
        lw=1.4,
        ls="--",
        label="Predicted (RF, 0\u20135 cm)",
    )
    ax_swp.legend(fontsize=6.5, loc="upper right", framealpha=0.85)

    # ── ET panel (row 3) ─────────────────────────────────────────────
    ax_et.set_ylabel("ET\n(mm/day)", fontsize=8)
    ax_et.set_ylim(*et_ylim)
    ax_et.tick_params(labelsize=7)
    ax_et.grid(axis="y", alpha=0.12)
    ax_et.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax_et.xaxis.set_major_locator(mdates.MonthLocator())

    (line_et,) = ax_et.plot([], [], color="#27AE60", lw=1.4, label="Observed ET")
    ax_et.legend(fontsize=6.5, loc="upper right", framealpha=0.85)

    # Vertical now-lines on all ts panels
    vlines = []
    for ax in ts_axes:
        vl = ax.axvline(all_dates[0], color="k", lw=0.6, alpha=0.4, ls="-")
        vlines.append(vl)

    # Set x-limits
    for ax in ts_axes:
        ax.set_xlim(all_dates[0], all_dates[-1])

    year = pd.Timestamp(date_start).year
    fig.suptitle(
        f"{site_key}  \u00b7  {site_label}  \u00b7  {year}",
        fontsize=12,
        fontweight="bold",
    )

    # ── Pre-compute arrays for fast indexing ─────────────────────────
    obs_dates_arr = obs.index.values
    obs_log_psi_arr = obs["log_psi"].values
    obs_theta_arr = obs["theta"].values
    obs_et_arr = obs["et_mm"].values
    pred_dates_arr = pred.index.values
    pred_vals_arr = pred.values
    smap_dates_arr = smap.index.values
    smap_vals_arr = smap.values

    last_map = [blank_map.copy()]

    # ── Animation update ─────────────────────────────────────────────
    def update(frame_idx):
        now = all_dates[frame_idx]
        now_np = now.to_numpy()

        # Map
        map_data = load_map_frame(now, window)
        if map_data is not None:
            last_map[0] = map_data
        im_map.set_data(last_map[0])
        map_title.set_text(now.strftime("%B %-d, %Y"))

        obs_mask = obs_dates_arr <= now_np

        # Precip bars — reveal up to current date
        for bar, bar_date in zip(pr_bars, obs.index):
            bar.set_alpha(0.6 if bar_date <= now else 0.0)

        # VWC lines
        line_obs_vwc.set_data(obs_dates_arr[obs_mask], obs_theta_arr[obs_mask])
        smap_mask = smap_dates_arr <= now_np
        if smap_mask.any():
            offsets = np.column_stack(
                [
                    mdates.date2num(smap_dates_arr[smap_mask]),
                    smap_vals_arr[smap_mask],
                ]
            )
            smap_scatter.set_offsets(offsets)
            smap_scatter.set_alpha(0.7)

        # SWP lines
        line_obs_psi.set_data(obs_dates_arr[obs_mask], obs_log_psi_arr[obs_mask])
        pred_mask = pred_dates_arr <= now_np
        line_pred_psi.set_data(pred_dates_arr[pred_mask], pred_vals_arr[pred_mask])

        # ET line
        line_et.set_data(obs_dates_arr[obs_mask], obs_et_arr[obs_mask])

        # Vertical now-lines
        for vl in vlines:
            vl.set_xdata([now, now])

        if (frame_idx + 1) % 30 == 0 or frame_idx + 1 == len(all_dates):
            print(f"  {frame_idx + 1}/{len(all_dates)}", flush=True)

        return [
            im_map,
            map_title,
            line_obs_psi,
            line_pred_psi,
            line_obs_vwc,
            smap_scatter,
            line_et,
        ] + vlines

    # ── Render ────────────────────────────────────────────────────────
    print(f"Animating {len(all_dates)} frames at {fps} fps...", flush=True)
    ani = FuncAnimation(
        fig,
        update,
        frames=len(all_dates),
        interval=1000 // fps,
        blit=False,
    )
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=["-crf", "22", "-preset", "fast", "-pix_fmt", "yuv420p"],
    )
    out_path = out_dir / f"fig11b_drought_animation_{site_key}.mp4"
    print(f"Writing {out_path} ...", flush=True)
    ani.save(str(out_path), writer=writer)
    plt.close(fig)
    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved to {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Animated drought time series at an Ameriflux site"
    )
    parser.add_argument(
        "--site",
        default="US-Jo2",
        choices=list(SITE_CONFIGS),
        help="Site key (default: US-Jo2)",
    )
    parser.add_argument(
        "--fps", type=int, default=FPS, help=f"Frames per second (default: {FPS})"
    )
    parser.add_argument(
        "--output-dir", default="figs/presentation", help="Output directory"
    )
    args = parser.parse_args()
    build_animation(args.site, args.output_dir, args.fps)
