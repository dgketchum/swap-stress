"""Figure 2: Training data map — 203K observations, 5 sources, ~13.8K sites.

Top panel : Global site locations colored by source.
Bottom panel: CONUS zoom with state boundaries and inset obs-count bar chart.

Style matches Figure 1b (fig1_panel_bc.py): white background, hidden
top/right spines, compact legend, 200 dpi.

Usage:
    uv run python -m viz.presentation.fig2_training_data_map
    uv run python -m viz.presentation.fig2_training_data_map --output-dir figs/presentation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Source shapefiles ────────────────────────────────────────────────
ROOT = Path("/nas/soils/soil_potential_obs")

SOURCES = {
    "GSHP": {
        "path": ROOT / "gshp" / "wrc_aggregated_mgrs.shp",
        "color": "#2ca02c",
        "marker": "o",
        "ms_global": 3,
        "ms_conus": 6,
        "zorder": 2,
    },
    "NCSS": {
        "path": ROOT / "ncss_labdatasqlite" / "ncss_profiles.shp",
        "color": "#1f77b4",
        "marker": "s",
        "ms_global": 5,
        "ms_conus": 8,
        "zorder": 3,
    },
    "MT Mesonet": {
        "path": ROOT / "mt_mesonet" / "station_metadata_clean_mgrs.shp",
        "color": "#ff7f0e",
        "marker": "D",
        "ms_global": 12,
        "ms_conus": 14,
        "zorder": 5,
    },
    "ReESH": {
        "path": ROOT / "reesh" / "shapefile" / "reesh_sites_mgrs.shp",
        "color": "#d62728",
        "marker": "^",
        "ms_global": 12,
        "ms_conus": 16,
        "zorder": 6,
    },
    "LaCADIAN": {
        "path": ROOT / "lacadian" / "lacadian_stations_mgrs.shp",
        "color": "#9467bd",
        "marker": "v",
        "ms_global": 12,
        "ms_conus": 16,
        "zorder": 4,
    },
}

# Observation counts from the 9 km global training table
OBS_COUNTS = {
    "GSHP": 115_379,
    "ReESH": 37_248,
    "MT Mesonet": 31_142,
    "NCSS": 13_864,
    "LaCADIAN": 5_418,
}

SITE_COUNTS = {
    "GSHP": 3_132,
    "NCSS": 5_569,
    "MT Mesonet": 198,
    "ReESH": 45,
    "LaCADIAN": 23,
}

LAND_SHP = Path("/nas/boundaries/natural_earth/ne_110m_land.shp")
STATES_SHP = Path("/tmp/us_states/cb_2022_us_state_20m.shp")

CONUS_XLIM = (-127, -65)
CONUS_YLIM = (24, 50)


def load_sources():
    gdfs = {}
    for name, cfg in SOURCES.items():
        gdf = gpd.read_file(cfg["path"]).to_crs(4326)
        gdfs[name] = gdf
    return gdfs


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(labelsize=8, length=3)


def main(output_dir):
    gdfs = load_sources()
    land = gpd.read_file(LAND_SHP).to_crs(4326)

    fig = plt.figure(figsize=(10, 8))
    # Use gridspec with fixed left/right so both panels share the same width
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[1, 1.2],
        hspace=0.12,
        left=0.06,
        right=0.98,
        top=0.97,
        bottom=0.05,
    )
    ax_g = fig.add_subplot(gs[0])
    ax_c = fig.add_subplot(gs[1])

    # ── Global panel ──────────────────────────────────────────────
    land.plot(ax=ax_g, facecolor="#f0f0f0", edgecolor="#bbb", linewidth=0.3, zorder=1)

    for name, cfg in SOURCES.items():
        gdf = gdfs[name]
        x = gdf.geometry.x.values
        y = gdf.geometry.y.values
        ax_g.scatter(
            x,
            y,
            s=cfg["ms_global"],
            marker=cfg["marker"],
            c=cfg["color"],
            edgecolors="none",
            alpha=0.7,
            zorder=cfg["zorder"],
            rasterized=True,
        )

    # Grey CONUS box to show zoom area
    from matplotlib.patches import Rectangle

    rect = Rectangle(
        (CONUS_XLIM[0], CONUS_YLIM[0]),
        CONUS_XLIM[1] - CONUS_XLIM[0],
        CONUS_YLIM[1] - CONUS_YLIM[0],
        linewidth=1.0,
        edgecolor="#555",
        facecolor="none",
        ls="--",
        zorder=7,
    )
    ax_g.add_patch(rect)

    ax_g.set_xlim(-180, 180)
    ax_g.set_ylim(-60, 82)
    ax_g.set_aspect("auto")
    ax_g.set_ylabel("Latitude", fontsize=9)
    ax_g.set_xlabel("")
    ax_g.set_xticklabels([])
    style_ax(ax_g)

    # Legend (global)
    legend_handles = []
    for name, cfg in SOURCES.items():
        n_sites = SITE_COUNTS[name]
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=cfg["marker"],
                color="none",
                markerfacecolor=cfg["color"],
                markeredgecolor="none",
                markersize=6,
                linestyle="None",
                label=f"{name} ({n_sites:,} sites)",
            )
        )
    ax_g.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=7.5,
        framealpha=0.9,
        edgecolor="#ddd",
        borderpad=0.5,
        handletextpad=0.4,
        labelspacing=0.3,
        ncol=1,
    )

    # ── CONUS panel ───────────────────────────────────────────────
    states = gpd.read_file(STATES_SHP).to_crs(4326)
    exclude = {"HI", "AK", "AS", "GU", "MP", "PR", "VI"}
    states = states[~states["STUSPS"].isin(exclude)]

    states.plot(ax=ax_c, facecolor="#f7f7f7", edgecolor="#bbb", linewidth=0.3, zorder=1)

    for name, cfg in SOURCES.items():
        gdf = gdfs[name]
        x = gdf.geometry.x.values
        y = gdf.geometry.y.values
        mask = (
            (x >= CONUS_XLIM[0])
            & (x <= CONUS_XLIM[1])
            & (y >= CONUS_YLIM[0])
            & (y <= CONUS_YLIM[1])
        )
        ax_c.scatter(
            x[mask],
            y[mask],
            s=cfg["ms_conus"],
            marker=cfg["marker"],
            c=cfg["color"],
            edgecolors="none",
            alpha=0.7,
            zorder=cfg["zorder"],
            rasterized=True,
        )

    ax_c.set_xlim(*CONUS_XLIM)
    ax_c.set_ylim(*CONUS_YLIM)
    ax_c.set_xlabel("Longitude", fontsize=9)
    ax_c.set_ylabel("Latitude", fontsize=9)
    ax_c.set_aspect("auto")
    style_ax(ax_c)

    # ── Save ──────────────────────────────────────────────────────
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig2_training_data_map.{ext}", dpi=200)
    plt.close(fig)
    print(f"Saved to {out / 'fig2_training_data_map.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figure 2: Training data map")
    parser.add_argument("--output-dir", default="figs/presentation")
    args = parser.parse_args()
    main(args.output_dir)
