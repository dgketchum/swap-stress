"""Figure 7. Koppen transferability — three-panel layout.

Left:   Table of per-subclass LOSO accuracy (colored swatches).
Top-R:  CONUS Koppen map with standard Beck colors.
Bot-R:  CONUS choropleth colored by LOSO R² per climate zone.

Usage:
    uv run swapstress-figures --figure spatial-skill
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgba
from rasterio.windows import from_bounds
from swapstress.figures.basemap import lakes_shapefile, states_shapefile

matplotlib.use("Agg")

# ── Paths ────────────────────────────────────────────────────────────
BECK_TIF = Path("/nas/soils/swapstress/ancillary/Beck_KG_V1_present_0p0083.tif")
# The regional CV results are written by stage 04 beside the model, not into
# the release tree -- the old release path here never existed.
MODEL_DIR = Path("/nas/soils/swapstress/models/direct_rf_9km_global_pruned")
CV_CSV = MODEL_DIR / "error_analysis" / "v01" / "regional_cv_results.csv"
STATES_SHP = Path(states_shapefile())
LAKES_SHP = Path(lakes_shapefile())
OUT_DIR = Path("figs/descriptor")

# ── CONUS bounds (EPSG:4326) ─────────────────────────────────────────
LON_MIN, LON_MAX = -125.0, -66.5
LAT_MIN, LAT_MAX = 24.5, 49.5

EXCLUDE_STUSPS = {"AK", "HI", "AS", "GU", "MP", "PR", "VI"}

# ── Beck Koppen labels and colors ────────────────────────────────────
BECK_LABELS = {
    1: "Af",
    2: "Am",
    3: "Aw",
    4: "BWh",
    5: "BWk",
    6: "BSh",
    7: "BSk",
    8: "Csa",
    9: "Csb",
    10: "Csc",
    11: "Cwa",
    12: "Cwb",
    13: "Cwc",
    14: "Cfa",
    15: "Cfb",
    16: "Cfc",
    17: "Dsa",
    18: "Dsb",
    19: "Dsc",
    20: "Dsd",
    21: "Dwa",
    22: "Dwb",
    23: "Dwc",
    24: "Dwd",
    25: "Dfa",
    26: "Dfb",
    27: "Dfc",
    28: "Dfd",
    29: "ET",
    30: "EF",
}

BECK_COLORS = {
    1: "#960000",
    2: "#FF0000",
    3: "#FFCCCC",
    4: "#FFCC00",
    5: "#FFFF64",
    6: "#CC8D14",
    7: "#CCAA54",
    8: "#FFFF00",
    9: "#C8C800",
    10: "#969600",
    11: "#96FF96",
    12: "#63C764",
    13: "#329633",
    14: "#C8FF50",
    15: "#66FF33",
    16: "#33C800",
    17: "#FF00FF",
    18: "#C800C8",
    19: "#963296",
    20: "#966496",
    21: "#ABB1FF",
    22: "#5A77DB",
    23: "#4C51B5",
    24: "#320087",
    25: "#00FFFF",
    26: "#37C8FF",
    27: "#007D7D",
    28: "#00465F",
    29: "#B2B2B2",
    30: "#666666",
}

KOPPEN_DESCRIPTIONS = {
    "Af": "Tropical rainforest",
    "Am": "Tropical monsoon",
    "Aw": "Tropical savanna",
    "BWh": "Hot desert",
    "BWk": "Cold desert",
    "BSh": "Hot semi-arid",
    "BSk": "Cold semi-arid",
    "Csa": "Med. hot summer",
    "Csb": "Med. warm summer",
    "Cfa": "Humid subtropical",
    "Cfb": "Oceanic",
    "Cfc": "Subpolar oceanic",
    "Cwa": "Monsoon humid subtrop.",
    "Cwb": "Subtropical highland",
    "Dfa": "Hot-summer continental",
    "Dfb": "Warm-summer continental",
    "Dfc": "Subarctic",
    "Dfd": "Extreme subarctic",
    "Dsb": "Med. warm-summer cont.",
    "Dsc": "Med. subarctic",
    "Dwa": "Monsoon hot-summer cont.",
    "Dwb": "Monsoon warm-summer cont.",
    "Dwc": "Monsoon subarctic",
    "ET": "Tundra",
}


def load_conus_koppen():
    """Load Beck Koppen raster windowed to CONUS."""
    with rasterio.open(BECK_TIF) as src:
        window = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, src.transform)
        data = src.read(1, window=window)
    return data


def build_figure(output_dir=OUT_DIR):
    cv = pd.read_csv(CV_CSV)
    label_to_code = {v: k for k, v in BECK_LABELS.items()}

    # Sort table by major zone group, then alphabetically within group
    cv["_major"] = cv["held_out_region"].str[0]
    major_order = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    cv["_sort"] = cv["_major"].map(major_order).astype(float) * 100
    cv["_sort"] += cv["held_out_region"].rank(method="dense")
    cv = cv.sort_values("_sort").reset_index(drop=True)

    # ── Load map data ─────────────────────────────────────────────────
    koppen_data = load_conus_koppen()

    # Filter table to classes that appear in the CONUS map
    conus_codes = set(np.unique(koppen_data)) - {0}
    conus_labels = {BECK_LABELS[c] for c in conus_codes if c in BECK_LABELS}
    cv = cv[cv["held_out_region"].isin(conus_labels)].reset_index(drop=True)

    # Standard Beck colormap
    color_list = ["#FFFFFF"]
    for i in range(1, 31):
        color_list.append(BECK_COLORS.get(i, "#FFFFFF"))
    cmap_beck = ListedColormap(color_list)
    bounds_beck = np.arange(-0.5, 31.5, 1)
    norm_beck = BoundaryNorm(bounds_beck, cmap_beck.N)

    koppen_float = koppen_data.astype(np.float32)
    koppen_float[koppen_data == 0] = np.nan

    # ── R² choropleth: map each pixel's code -> LOSO R² ──────────────
    r2_by_label = dict(zip(cv["held_out_region"], cv["r2"]))
    r2_map = np.full_like(koppen_data, np.nan, dtype=np.float32)
    for code, label in BECK_LABELS.items():
        if label in r2_by_label:
            r2_map[koppen_data == code] = r2_by_label[label]

    # Load states and lakes
    states = gpd.read_file(STATES_SHP)
    conus_states = states[~states.STUSPS.isin(EXCLUDE_STUSPS)].copy()

    lakes = gpd.read_file(LAKES_SHP)
    lakes_conus = lakes.cx[LON_MIN:LON_MAX, LAT_MIN:LAT_MAX]
    lakes_conus = lakes_conus[lakes_conus["scalerank"] <= 3].copy()

    extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]

    # ── Figure layout ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9), dpi=200)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1, 1.6],
        height_ratios=[1, 1],
        hspace=0.06,
        wspace=0.04,
        left=0.01,
        right=0.96,
        top=0.94,
        bottom=0.03,
    )

    ax_tbl = fig.add_subplot(gs[:, 0])  # left: table spans both rows
    ax_kop = fig.add_subplot(gs[0, 1])  # top-right: Koppen map
    ax_r2 = fig.add_subplot(gs[1, 1])  # bot-right: R² choropleth

    # ── Left panel: table ─────────────────────────────────────────────
    ax_tbl.set_axis_off()

    col_labels = [
        "",
        "Class",
        "Description",
        "n",
        "RMSE\n(log\u2081\u2080 cm)",
        "R\u00b2",
        "Bias\n(log\u2081\u2080 cm)",
    ]
    n_rows = len(cv)
    n_cols = len(col_labels)

    cell_text = []
    cell_colors = []
    for _, row in cv.iterrows():
        lbl = row["held_out_region"]
        desc = KOPPEN_DESCRIPTIONS.get(lbl, "")
        code = label_to_code.get(lbl)
        beck_color = BECK_COLORS.get(code, "#FFFFFF") if code else "#FFFFFF"

        r2_val = row["r2"]
        # White background for all cells except the swatch column
        white = (1.0, 1.0, 1.0, 1.0)
        swatch = to_rgba(beck_color)
        cell_colors.append([swatch, white, white, white, white, white, white])

        cell_text.append(
            [
                "",
                lbl,
                desc,
                f"{int(row['n_test']):,}",
                f"{row['rmse']:.2f}",
                f"{r2_val:.2f}",
                f"{row['bias']:+.2f}",
            ]
        )

    table = ax_tbl.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=["#D8D8D8"] * n_cols,
        cellLoc="center",
        loc="upper center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    # Header styling
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_text_props(fontweight="bold", fontsize=8.5)
        cell.set_edgecolor("#AAAAAA")
        cell.set_height(0.045)

    # Data cell styling
    for i in range(n_rows):
        for j in range(n_cols):
            cell = table[i + 1, j]
            cell.set_edgecolor("#DDDDDD")
            cell.set_height(0.038)

        # Swatch column: no text, just color
        table[i + 1, 0].set_width(0.04)
        table[0, 0].set_width(0.04)

        # Left-align description
        table[i + 1, 2].set_text_props(ha="left")

        # Highlight poor R²
        r2_val = cv.iloc[i]["r2"]
        if r2_val < 0.25:
            table[i + 1, 5].set_text_props(fontweight="bold", color="#C0392B")
        elif r2_val >= 0.65:
            table[i + 1, 5].set_text_props(fontweight="bold", color="#1A7A2E")

    # Column widths
    col_widths = [0.04, 0.07, 0.38, 0.11, 0.12, 0.12, 0.12]
    for j, w in enumerate(col_widths):
        for i in range(n_rows + 1):
            table[i, j].set_width(w)

    ax_tbl.set_title(
        "Leave-One-Class-Out CV",
        fontsize=11,
        fontweight="bold",
        pad=8,
        loc="center",
    )

    # ── Top-right: Koppen classification map ──────────────────────────
    ax_kop.imshow(
        koppen_float,
        cmap=cmap_beck,
        norm=norm_beck,
        extent=extent,
        origin="upper",
        interpolation="nearest",
    )
    conus_states.boundary.plot(ax=ax_kop, edgecolor="#2C2C2A", linewidth=0.4)
    lakes_conus.plot(ax=ax_kop, facecolor="white", edgecolor="none", zorder=5)
    ax_kop.set_xlim(LON_MIN, LON_MAX)
    ax_kop.set_ylim(LAT_MIN, LAT_MAX)
    ax_kop.set_aspect("equal")
    ax_kop.set_axis_off()
    ax_kop.set_title(
        "K\u00f6ppen-Geiger Classification",
        fontsize=10,
        fontweight="bold",
        pad=4,
    )

    # ── Bottom-right: R² choropleth ───────────────────────────────────
    cmap_r2 = plt.cm.RdYlBu
    im_r2 = ax_r2.imshow(
        r2_map,
        cmap=cmap_r2,
        vmin=0.0,
        vmax=0.85,
        extent=extent,
        origin="upper",
        interpolation="nearest",
    )
    conus_states.boundary.plot(ax=ax_r2, edgecolor="#2C2C2A", linewidth=0.4)
    lakes_conus.plot(ax=ax_r2, facecolor="white", edgecolor="none", zorder=5)
    ax_r2.set_xlim(LON_MIN, LON_MAX)
    ax_r2.set_ylim(LAT_MIN, LAT_MAX)
    ax_r2.set_aspect("equal")
    ax_r2.set_axis_off()
    ax_r2.set_title(
        "LOCO Transferability  (R\u00b2 by climate zone)",
        fontsize=10,
        fontweight="bold",
        pad=4,
    )

    # Colorbar for R² map
    cax = fig.add_axes([0.62, 0.025, 0.22, 0.015])
    cb = fig.colorbar(im_r2, cax=cax, orientation="horizontal")
    cb.set_label("R\u00b2", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # ── Save ──────────────────────────────────────────────────────────
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out_path = out_dir / f"fig05_spatial_skill.{ext}"
        fig.savefig(out_path, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out_path.absolute()}")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Figure 7: Koppen-zone transferability"
    )
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args(argv)
    build_figure(args.output_dir)


if __name__ == "__main__":
    main()
