"""Figure 5. Koppen-zone transferability -- three panels.

a  CONUS Koppen-Geiger classes, in the standard Beck colours.
b  Leave-one-class-out (LOCO) R² painted back onto those classes.
c  The per-class LOCO statistics behind panel b, ordered by decreasing R².

Panel c doubles as the colour key for panel a: every evaluated class carries its
Beck swatch. Sizes and type follow ``swapstress.figures.style`` -- Nature's
183 mm double column, 7 pt ceiling, 8 pt bold panel letters.

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
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle
from rasterio.features import geometry_mask
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import from_bounds

from swapstress.figures import style
from swapstress.figures.basemap import lakes_shapefile, states_shapefile

matplotlib.use("Agg")

# ── Paths ────────────────────────────────────────────────────────────
BECK_TIF = Path("/nas/soils/swapstress/ancillary/Beck_KG_V1_present_0p0083.tif")
# Stage 04 writes the regional CV straight into its ``--output-dir``, making no
# subdirectory of its own, so the 0.3 release run leaves it in the release
# evaluation tree beside the other validation tables. At the default
# ``--level both`` it also writes ``regional_cv_results_major.csv`` and
# ``..._subclass.csv``; the unsuffixed file read here is the two concatenated.
# The CONUS filter below keeps only the sub-class rows, because the major-zone
# rows (A-E) are not Beck class labels and so match nothing on the map.
EVAL_DIR = Path("/nas/soils/swapstress/releases/v03_20260729/evaluation")
CV_CSV = EVAL_DIR / "regional_cv_results.csv"
STATES_SHP = Path(states_shapefile())
LAKES_SHP = Path(lakes_shapefile())
OUT_DIR = Path("figs/descriptor")

# ── CONUS bounds (EPSG:4326) ─────────────────────────────────────────
LON_MIN, LON_MAX = -125.0, -66.5
LAT_MIN, LAT_MAX = 24.5, 49.5

# CONUS Albers. The Beck grid is 0.0083 deg lon/lat, so drawing it on raw
# degrees stretches the country sideways; warping to an equal-area frame gives
# the shape a reader expects and matches the rest of the figure set. 800 m is
# just finer than the source cell (about 740 m of longitude at 37 N).
CONUS_CRS = "EPSG:5070"
CONUS_RES_M = 800.0

EXCLUDE_STUSPS = {"AK", "HI", "AS", "GU", "MP", "PR", "VI"}

# ── R2 encoding ──────────────────────────────────────────────────────
# Sequential, not diverging: 16 of the 17 evaluated classes are positive and the
# single negative one is marginal, so a midpoint at zero would spend half the
# ramp on 0.08 of range. Limits are padded round numbers around the observed
# -0.08 to 0.66, with a tick on zero so "no better than the mean" stays findable.
R2_VMIN, R2_VMAX = -0.10, 0.70
R2_TICKS = (0.0, 0.2, 0.4, 0.6)
NOT_EVALUATED = "#cbcbcb"

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

# ── Table geometry, as fractions of one block's width ────────────────
# Two blocks side by side hold the 17 rows; a single 17-row column would leave
# the bottom third of a double-column figure empty.
BLOCK_W = 0.470
BLOCK_X = (0.0, 0.530)

SWATCH_X0, SWATCH_X1 = 0.000, 0.028
COL_CLASS = 0.052  # left aligned
COL_DESC = 0.145  # left aligned
COL_N = 0.574  # right aligned
COL_RMSE = 0.722  # right aligned
COL_R2 = 0.841  # right aligned
COL_BIAS = 1.000  # right aligned

BODY_PT = 6.5
NOTE_PT = 6.0

# The resolved sans face has no subscript-digit glyphs, so the units line is
# mathtext. Left alone, mathtext sets in DejaVu Sans and the PDF ends up with
# two font families; style.apply points it back at the body face.
LOG_CM = r"(log$_{10}$ cm)"

# Baselines and rules of panel c, as fractions of the table axes.
Y_TITLE = 0.945
Y_TOP_RULE = 0.865
Y_HEAD = 0.805
Y_UNIT = 0.748
Y_HEAD_RULE = 0.700
Y_FIRST = 0.630
Y_BOTTOM_RULE = 0.000
SWATCH_H = 0.052


def load_conus_koppen():
    """Beck Koppen over CONUS, warped to Albers equal area.

    Nearest neighbour, because the values are class codes and must not be
    averaged. Returns the class array and its extent in projected metres.
    """
    with rasterio.open(BECK_TIF) as src:
        window = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, src.transform)
        data = src.read(1, window=window)
        src_transform = src.window_transform(window)
        src_crs = src.crs

    dst_transform, width, height = calculate_default_transform(
        src_crs,
        CONUS_CRS,
        data.shape[1],
        data.shape[0],
        left=LON_MIN,
        bottom=LAT_MIN,
        right=LON_MAX,
        top=LAT_MAX,
        resolution=CONUS_RES_M,
    )
    dst = np.zeros((height, width), dtype=data.dtype)
    reproject(
        source=data,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=CONUS_CRS,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    return dst, dst_transform


def _draw_map(ax, states, lakes, edgecolor, bounds):
    """Common furniture for both map panels."""
    states.boundary.plot(ax=ax, edgecolor=edgecolor, linewidth=0.25)
    lakes.plot(ax=ax, facecolor="white", edgecolor="none", zorder=5)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal")
    ax.set_axis_off()


def _rule(ax, x0, x1, y):
    ax.plot(
        [x0, x1],
        [y, y],
        color=style.AXIS_COLOR,
        linewidth=0.5,
        solid_capstyle="butt",
        clip_on=False,
    )


def _draw_table_block(ax, rows, x0, pitch):
    """Write one block of the statistics table in axes coordinates."""

    def x(frac):
        return x0 + frac * BLOCK_W

    head = dict(fontsize=BODY_PT, fontweight="bold", color=style.AXIS_COLOR)
    unit = dict(fontsize=NOTE_PT, color=style.MUTED_INK)

    ax.text(x(COL_CLASS), Y_HEAD, "Class", ha="left", va="baseline", **head)
    ax.text(x(COL_DESC), Y_HEAD, "Description", ha="left", va="baseline", **head)
    ax.text(x(COL_N), Y_HEAD, "n", ha="right", va="baseline", **head)
    ax.text(x(COL_RMSE), Y_HEAD, "RMSE", ha="right", va="baseline", **head)
    ax.text(x(COL_R2), Y_HEAD, "R²", ha="right", va="baseline", **head)
    ax.text(x(COL_BIAS), Y_HEAD, "Bias", ha="right", va="baseline", **head)
    ax.text(x(COL_RMSE), Y_UNIT, LOG_CM, ha="right", va="baseline", **unit)
    ax.text(x(COL_BIAS), Y_UNIT, LOG_CM, ha="right", va="baseline", **unit)

    for y in (Y_TOP_RULE, Y_HEAD_RULE, Y_BOTTOM_RULE):
        _rule(ax, x(0.0), x(1.0), y)

    body = dict(fontsize=BODY_PT, color=style.AXIS_COLOR)
    for i, row in enumerate(rows):
        y = Y_FIRST - i * pitch
        ax.add_patch(
            Rectangle(
                (x(SWATCH_X0), y - 0.007),
                (SWATCH_X1 - SWATCH_X0) * BLOCK_W,
                SWATCH_H,
                facecolor=BECK_COLORS.get(row["code"], "#ffffff"),
                edgecolor="#9a9a9a",
                linewidth=0.25,
            )
        )
        ax.text(x(COL_CLASS), y, row["label"], ha="left", va="baseline", **body)
        ax.text(x(COL_DESC), y, row["desc"], ha="left", va="baseline", **body)
        ax.text(x(COL_N), y, row["n"], ha="right", va="baseline", **body)
        ax.text(x(COL_RMSE), y, row["rmse"], ha="right", va="baseline", **body)
        ax.text(x(COL_R2), y, row["r2"], ha="right", va="baseline", **body)
        ax.text(x(COL_BIAS), y, row["bias"], ha="right", va="baseline", **body)


def build_figure(output_dir=OUT_DIR):
    style.apply()

    cv = pd.read_csv(CV_CSV)
    label_to_code = {v: k for k, v in BECK_LABELS.items()}

    koppen_data, dst_transform = load_conus_koppen()

    states = gpd.read_file(STATES_SHP)
    conus_states = states[~states.STUSPS.isin(EXCLUDE_STUSPS)].to_crs(CONUS_CRS)

    lakes = gpd.read_file(LAKES_SHP)
    lakes_conus = lakes.cx[LON_MIN:LON_MAX, LAT_MIN:LAT_MAX]
    lakes_conus = lakes_conus[lakes_conus["scalerank"] <= 3].to_crs(CONUS_CRS)

    # Clip to the states. The lon/lat window is a curved quadrilateral once
    # warped, and its arc cutting across Canada reads as a data artefact; the
    # analysis is a CONUS one, so the silhouette should be CONUS.
    inside = geometry_mask(
        conus_states.geometry,
        out_shape=koppen_data.shape,
        transform=dst_transform,
        invert=True,
        all_touched=True,
    )
    koppen_data = np.where(inside, koppen_data, 0)
    rows, cols = koppen_data.shape
    left, bottom, right, top = array_bounds(rows, cols, dst_transform)
    extent = [left, right, bottom, top]

    # Keep the classes that actually appear on the map, best first -- the
    # ranking is the point of the panel, so it is the row order.
    conus_codes = set(np.unique(koppen_data)) - {0}
    conus_labels = {BECK_LABELS[c] for c in conus_codes if c in BECK_LABELS}
    cv = cv[cv["held_out_region"].isin(conus_labels)].copy()
    cv = cv.sort_values("r2", ascending=False).reset_index(drop=True)

    # Standard Beck colormap.
    color_list = ["#FFFFFF"] + [BECK_COLORS.get(i, "#FFFFFF") for i in range(1, 31)]
    cmap_beck = ListedColormap(color_list)
    norm_beck = BoundaryNorm(np.arange(-0.5, 31.5, 1), cmap_beck.N)

    koppen_float = koppen_data.astype(np.float32)
    koppen_float[koppen_data == 0] = np.nan

    # R2 choropleth: map each pixel's class code -> its LOCO R2. Classified
    # pixels whose class was not evaluated are drawn grey rather than left
    # white, so they cannot be misread as ocean.
    r2_by_label = dict(zip(cv["held_out_region"], cv["r2"]))
    r2_map = np.full(koppen_data.shape, np.nan, dtype=np.float32)
    for code, label in BECK_LABELS.items():
        if label in r2_by_label:
            r2_map[koppen_data == code] = r2_by_label[label]
    unevaluated = np.where((koppen_data != 0) & np.isnan(r2_map), 1.0, np.nan)

    x0, y0, x1, y1 = conus_states.total_bounds
    pad = 0.012 * (x1 - x0)
    bounds = [x0 - pad, x1 + pad, y0 - pad, y1 + pad]

    # ── Layout ────────────────────────────────────────────────────────
    fig = plt.figure(
        figsize=style.figsize(style.DOUBLE_COLUMN_MM, 113.0),
        layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0.012, h_pad=0.012, wspace=0.03, hspace=0.01)
    # Rows sized to what they need: map + title, colour key, then the table.
    # The maps are the widest CONUS that fits a column at their 1.6:1 shape;
    # the key row is deep enough that the colour bar is a bar, not a hairline.
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.165, 0.71])

    ax_kop = fig.add_subplot(gs[0, 0])
    ax_r2 = fig.add_subplot(gs[0, 1])
    key_gs = gs[1, 1].subgridspec(1, 2, width_ratios=[1.0, 1.4], wspace=0.08)
    ax_key = fig.add_subplot(key_gs[0, 0])
    cax = fig.add_subplot(key_gs[0, 1])
    ax_tbl = fig.add_subplot(gs[2, :])

    # ── a: Koppen classification ──────────────────────────────────────
    ax_kop.imshow(
        koppen_float,
        cmap=cmap_beck,
        norm=norm_beck,
        extent=extent,
        origin="upper",
        interpolation="nearest",
    )
    _draw_map(ax_kop, conus_states, lakes_conus, "#2c2c2a", bounds)
    ax_kop.set_title("Köppen–Geiger climate class", pad=3)
    style.panel_label(ax_kop, "a", dx=0.0, dy=1.02)

    # ── b: LOCO R2 choropleth ─────────────────────────────────────────
    ax_r2.imshow(
        unevaluated,
        cmap=ListedColormap([NOT_EVALUATED]),
        vmin=0,
        vmax=1,
        extent=extent,
        origin="upper",
        interpolation="nearest",
    )
    im_r2 = ax_r2.imshow(
        r2_map,
        cmap=style.SEQUENTIAL,
        vmin=R2_VMIN,
        vmax=R2_VMAX,
        extent=extent,
        origin="upper",
        interpolation="nearest",
    )
    # Mid grey holds up against both ends of cividis; the near-black used on
    # panel a vanishes into the dark low-skill zones.
    _draw_map(ax_r2, conus_states, lakes_conus, "#7a7a7a", bounds)
    ax_r2.set_title("Leave-one-class-out skill", pad=3)
    style.panel_label(ax_r2, "b", dx=0.0, dy=1.02)

    ax_key.set_axis_off()
    ax_key.legend(
        handles=[Patch(facecolor=NOT_EVALUATED, edgecolor="none")],
        labels=["Class not evaluated"],
        loc="center right",
        frameon=False,
        fontsize=NOTE_PT,
        handlelength=1.0,
        handleheight=1.0,
        handletextpad=0.4,
        borderpad=0.0,
        borderaxespad=0.0,
    )

    cb = fig.colorbar(im_r2, cax=cax, orientation="horizontal", ticks=list(R2_TICKS))
    cb.set_label("Held-out R²", fontsize=NOTE_PT, labelpad=1.5)
    cb.ax.tick_params(labelsize=NOTE_PT, length=1.5, width=0.4, pad=1.5)
    cb.outline.set_linewidth(0.4)
    cb.outline.set_edgecolor(style.AXIS_COLOR)

    # ── c: per-class statistics ───────────────────────────────────────
    ax_tbl.set_axis_off()
    ax_tbl.set_xlim(0, 1)
    ax_tbl.set_ylim(0, 1)

    records = [
        {
            "label": r["held_out_region"],
            "code": label_to_code.get(r["held_out_region"]),
            "desc": KOPPEN_DESCRIPTIONS.get(r["held_out_region"], ""),
            "n": f"{int(r['n_test']):,}",
            "rmse": f"{r['rmse']:.2f}",
            "r2": f"{r['r2']:.2f}",
            "bias": f"{r['bias']:+.2f}",
        }
        for _, r in cv.iterrows()
    ]
    split = (len(records) + 1) // 2
    blocks = [records[:split], records[split:]]

    # Both blocks share one pitch and one bottom rule, so the shorter block
    # still lines up with the taller one.
    pitch = (Y_FIRST - Y_BOTTOM_RULE - 0.045) / max(split - 1, 1)
    for x0, rows in zip(BLOCK_X, blocks):
        _draw_table_block(ax_tbl, rows, x0, pitch)

    ax_tbl.text(
        0.026,
        Y_TITLE,
        "Leave-one-class-out cross-validation, by decreasing R²",
        ha="left",
        va="baseline",
        fontsize=style.MAX_TEXT_PT,
        color=style.AXIS_COLOR,
    )
    style.panel_label(ax_tbl, "c", dx=0.0, dy=Y_TITLE)

    out_dir = Path(output_dir)
    png = style.save(fig, out_dir / "fig05_spatial_skill")
    print(f"Saved: {png}")
    print(f"Saved: {png.with_suffix('.pdf')}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Figure 5: Koppen-zone transferability"
    )
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args(argv)
    build_figure(args.output_dir)


if __name__ == "__main__":
    main()
