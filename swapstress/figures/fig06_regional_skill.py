"""Descriptor Fig 6: CONUS transferability to withheld climate classes.

a  Major Koppen-class leave-one-class-out R2, evaluated only on CONUS
   observations, painted onto the CONUS extent of each evaluated class. One
   score per withheld class: the map's color resolution is the statistical
   resolution, nothing finer.
b  The ranked CONUS-class summary behind the map: R2 as a dot on the shared
   scale, with RMSE, bias, and observation/location support as aligned text.

This is the major-class redesign of 2026-08-12 (figures_handoff.md). The old
three-panel form -- a 20-color Beck classification map, a subclass R2
choropleth at ~800 m texture, and a 20-row table -- read as false density: the
fine map texture implied locally-resolved skill where each class carries one
aggregate score, and the displayed subclasses did not match the manuscript's
Technical Validation narrative, which reports the five major classes. Subclass
results remain in the deposited ``regional_cv_results_subclass.csv``.

The sequential R2 scale spans 0.2-0.7, bracketing the five CONUS scores
(0.27-0.63). Exact values remain printed in panel b, so the truncated scale is
explicit while the map retains enough contrast to distinguish the two broad
performance groups.

Classes B, C, and D are labeled directly on their (large) map patches; A's
CONUS extent is the southern tip of Florida, labeled with a short leader, and
E is scattered alpine tundra too small to label at final size -- the caption
carries that. Class letters identify regions independently of the R2 colors.

Each refit removes one climate class from the global training pool and uses a
standard random forest with the production features and hyperparameters, not
the quantile forest. Only its held-out CONUS observations are scored; the
caption must keep both domain statements explicit.

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
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import from_bounds

from swapstress.figures import style
from swapstress.figures.basemap import lakes_shapefile, states_shapefile

matplotlib.use("Agg")

# ── Paths ────────────────────────────────────────────────────────────
BECK_TIF = Path("/nas/soils/swapstress/ancillary/Beck_KG_V1_present_0p0083.tif")
# The CONUS suffix is deliberate. The older major-class table was scored on
# global observations and must not be painted onto a CONUS-only map.
EVAL_DIR = Path("/nas/soils/swapstress/releases/v03_20260729/evaluation")
CV_CSV = EVAL_DIR / "regional_cv_results_major_conus.csv"
STATES_SHP = Path(states_shapefile())
LAKES_SHP = Path(lakes_shapefile())
OUT_DIR = Path("figs/descriptor")

# ── CONUS bounds (EPSG:4326) ─────────────────────────────────────────
LON_MIN, LON_MAX = -125.0, -66.5
LAT_MIN, LAT_MAX = 24.5, 49.5

# CONUS Albers, matching Figs 3, 4, and 7. The Beck grid is 0.0083 deg
# lon/lat, so drawing it on raw degrees stretches the country sideways. 800 m
# is just finer than the source cell (about 740 m of longitude at 37 N).
CONUS_CRS = "EPSG:5070"
CONUS_RES_M = 800.0

EXCLUDE_STUSPS = {"AK", "HI", "AS", "GU", "MP", "PR", "VI"}

# ── R2 encoding ──────────────────────────────────────────────────────
# All five CONUS scores are positive and span 0.27-0.63. A clean 0.2-0.7
# sequential range makes their relative differences visible; panel b prints
# every value so the non-zero lower bound is unmistakable.
R2_VMIN, R2_VMAX = 0.2, 0.7
R2_TICKS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)

# ── Beck code -> major class ─────────────────────────────────────────
MAJOR_OF_CODE = {
    **{c: "A" for c in range(1, 4)},
    **{c: "B" for c in range(4, 8)},
    **{c: "C" for c in range(8, 17)},
    **{c: "D" for c in range(17, 29)},
    **{c: "E" for c in range(29, 31)},
}
MAJOR_DESCRIPTIONS = {
    "A": "Tropical",
    "B": "Arid",
    "C": "Temperate",
    "D": "Continental",
    "E": "Alpine tundra",
}

# On-map class letters, placed on each class's largest coherent CONUS patch
# (lon, lat). A gets a leader from open Atlantic to its south-Florida sliver;
# E's alpine specks are too small to label and are covered in the caption.
LETTER_LONLAT = {"B": (-116.0, 40.0), "C": (-84.5, 33.0), "D": (-97.5, 46.5)}
A_LETTER_LONLAT = (-78.6, 26.4)
A_TARGET_LONLAT = (-80.9, 25.9)

BOUNDARY_COLOR = "#7a7a7a"
BOUNDARY_WIDTH = 0.25

FIG_HEIGHT_MM = 68.0
WIDTH_RATIOS = (1.32, 0.66, 0.82)

NOTE_PT = 6.0

# Text-column x positions in the summary panel's axes fraction.
COL_RMSE = 0.16
COL_BIAS = 0.43
COL_N = 0.70
COL_LOCATIONS = 1.00
METRIC_HEADER_Y = 0.985
METRIC_UNIT_Y = 0.94


def load_conus_major():
    """Beck Koppen codes over CONUS in Albers.

    Nearest neighbour, because the values are class codes and must not be
    averaged. Returns an array of integer Beck codes (0 outside), plus the
    transform; ``MAJOR_OF_CODE`` provides the major-class aggregation.
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


def read_major_cv(path: Path) -> pd.DataFrame:
    """CONUS-scored major-class LOCO rows, ranked by decreasing R2."""
    if not path.exists():
        raise FileNotFoundError(
            f"No CONUS-scored major-class regional CV table at {path}. Run "
            "`python -m swapstress.validation.regional_cv --level major "
            "--evaluation-domain conus` to produce it."
        )
    cv = pd.read_csv(path)
    required = {
        "held_out_region",
        "r2",
        "rmse",
        "bias",
        "n_test",
        "n_test_locations",
        "training_domain",
        "evaluation_domain",
        "estimator",
        "n_estimators",
        "level",
        "reference_model_dir",
    }
    missing = required - set(cv)
    if missing:
        raise ValueError(f"{path} is missing required column(s) {sorted(missing)}.")
    if not (cv["evaluation_domain"] == "conus").all():
        raise ValueError(f"{path} contains results not evaluated in CONUS.")
    if not (cv["training_domain"] == "global").all():
        raise ValueError(f"{path} contains results not trained on the global pool.")
    if not (cv["estimator"] == "RandomForestRegressor").all():
        raise ValueError(f"{path} contains results from an unexpected estimator.")
    if not (cv["n_estimators"] == 250).all():
        raise ValueError(f"{path} does not contain the specified 250-tree refits.")
    if not (cv["level"] == "major").all():
        raise ValueError(
            f"{path} contains results above or below the major-class level."
        )
    expected_reference = "/nas/soils/swapstress/models/direct_qrf_9km_global_pruned"
    if not (cv["reference_model_dir"] == expected_reference).all():
        raise ValueError(f"{path} does not reference the released QRF configuration.")
    unknown = set(cv["held_out_region"]) - set(MAJOR_DESCRIPTIONS)
    if unknown:
        raise ValueError(f"{path} contains unknown major class(es) {sorted(unknown)}.")
    absent = set(MAJOR_DESCRIPTIONS) - set(cv["held_out_region"])
    if absent:
        raise ValueError(
            f"{path} is missing CONUS-supported major class(es) {sorted(absent)}."
        )
    if cv["held_out_region"].duplicated().any():
        raise ValueError(f"{path} contains duplicate held-out major classes.")
    if cv.empty:
        raise ValueError(f"{path} contains no CONUS climate classes.")
    metric_columns = ["r2", "rmse", "bias", "n_test", "n_test_locations"]
    if not np.isfinite(cv[metric_columns].to_numpy(dtype=float)).all():
        raise ValueError(f"{path} contains non-finite metrics or support counts.")
    return cv.sort_values("r2", ascending=False).reset_index(drop=True)


def build_figure(output_dir=OUT_DIR):
    style.apply()

    cv = read_major_cv(CV_CSV)
    codes, dst_transform = load_conus_major()

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
        out_shape=codes.shape,
        transform=dst_transform,
        invert=True,
        all_touched=True,
    )
    codes = np.where(inside, codes, 0)
    rows, cols = codes.shape
    left, bottom, right, top = array_bounds(rows, cols, dst_transform)
    extent = [left, right, bottom, top]

    # One CONUS score per withheld class, painted onto that class's CONUS
    # extent. The gray base exposes any unmapped land instead of silently
    # making it look like water or background.
    r2_of_major = dict(zip(cv["held_out_region"], cv["r2"]))
    r2_map = np.full(codes.shape, np.nan, dtype=np.float32)
    for code, major in MAJOR_OF_CODE.items():
        if major in r2_of_major:
            r2_map[codes == code] = r2_of_major[major]

    x0, y0, x1, y1 = conus_states.total_bounds
    pad = 0.012 * (x1 - x0)
    bounds = [x0 - pad, x1 + pad, y0 - pad, y1 + pad]

    # ── Layout ────────────────────────────────────────────────────────
    fig = plt.figure(
        figsize=style.figsize(style.DOUBLE_COLUMN_MM, FIG_HEIGHT_MM),
        layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0.012, h_pad=0.012, wspace=0.04)
    gs = fig.add_gridspec(1, 3, width_ratios=list(WIDTH_RATIOS))
    ax_map = fig.add_subplot(gs[0, 0])
    ax_dot = fig.add_subplot(gs[0, 1])
    ax_txt = fig.add_subplot(gs[0, 2])

    # ── a: CONUS-scored major-class LOCO R2 map ──────────────────────
    cmap = matplotlib.colormaps[style.SEQUENTIAL].copy()
    cmap.set_bad(alpha=0.0)
    norm = matplotlib.colors.Normalize(vmin=R2_VMIN, vmax=R2_VMAX)
    gray_cmap = matplotlib.colors.ListedColormap([style.NO_DATA_GRAY])
    gray_cmap.set_bad(alpha=0.0)
    ax_map.imshow(
        np.where(codes > 0, 0.0, np.nan),
        cmap=gray_cmap,
        vmin=0.0,
        vmax=1.0,
        extent=extent,
        origin="upper",
        interpolation="nearest",
    )
    im = ax_map.imshow(
        r2_map,
        cmap=cmap,
        norm=norm,
        extent=extent,
        origin="upper",
        interpolation="nearest",
    )
    conus_states.boundary.plot(
        ax=ax_map, edgecolor=BOUNDARY_COLOR, linewidth=BOUNDARY_WIDTH
    )
    lakes_conus.plot(ax=ax_map, facecolor="white", edgecolor="none", zorder=5)
    ax_map.set_xlim(bounds[0], bounds[1])
    ax_map.set_ylim(bounds[2], bounds[3])
    ax_map.set_aspect("equal")
    ax_map.set_axis_off()
    ax_map.set_title("Held-class R² within CONUS", pad=3)
    style.panel_label(ax_map, "a", dx=0.0, dy=1.02)

    # Letters carry class identity; the fill carries only the score, so
    # near-equal scores (B vs C) legitimately render near-identical colors.
    to_albers = Transformer.from_crs("EPSG:4326", CONUS_CRS, always_xy=True)
    for major, lonlat in LETTER_LONLAT.items():
        if major not in r2_of_major:
            continue
        x, y = to_albers.transform(*lonlat)
        r, g, b, _ = cmap(norm(r2_of_major[major]))
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        ax_map.text(
            x,
            y,
            major,
            fontsize=style.MAX_TEXT_PT,
            fontweight="bold",
            color="white" if luminance < 0.5 else "black",
            ha="center",
            va="center",
            zorder=6,
        )
    if "A" in r2_of_major:
        ax_a = to_albers.transform(*A_LETTER_LONLAT)
        a_target = to_albers.transform(*A_TARGET_LONLAT)
        ax_map.annotate(
            "A",
            xy=a_target,
            xytext=ax_a,
            fontsize=style.MAX_TEXT_PT,
            fontweight="bold",
            color="black",
            ha="center",
            va="center",
            zorder=6,
            arrowprops=dict(
                arrowstyle="-", linewidth=0.4, color=BOUNDARY_COLOR, shrinkB=1.5
            ),
        )

    cbar = fig.colorbar(
        im,
        ax=ax_map,
        orientation="horizontal",
        ticks=list(R2_TICKS),
        shrink=0.62,
        aspect=34,
        pad=0.015,
        extend=(
            "both"
            if cv["r2"].min() < R2_VMIN and cv["r2"].max() > R2_VMAX
            else "min"
            if cv["r2"].min() < R2_VMIN
            else "max"
            if cv["r2"].max() > R2_VMAX
            else "neither"
        ),
    )
    cbar.set_label("Held-out R²", fontsize=NOTE_PT, labelpad=1.5)
    cbar.ax.tick_params(labelsize=NOTE_PT, length=1.8, width=0.4, pad=1.5)
    cbar.outline.set_linewidth(0.4)
    cbar.outline.set_edgecolor(style.AXIS_COLOR)

    # ── b: ranked CONUS summary, dot plus text ────────────────────────
    ypos = np.arange(len(cv))
    ax_dot.grid(axis="x", color=style.GRID_COLOR, linewidth=0.4, zorder=1)
    ax_dot.scatter(
        cv["r2"],
        ypos,
        s=22,
        c=[cmap(norm(v)) for v in cv["r2"]],
        edgecolors=style.AXIS_COLOR,
        linewidths=0.4,
        zorder=3,
    )
    for y, r2 in zip(ypos, cv["r2"]):
        ax_dot.text(
            r2 + 0.035,
            y,
            f"{r2:.2f}",
            fontsize=NOTE_PT,
            color="black",
            ha="left",
            va="center",
        )
    ax_dot.set_yticks(
        ypos,
        labels=[
            f"{r['held_out_region']}  {MAJOR_DESCRIPTIONS[r['held_out_region']]}"
            for _, r in cv.iterrows()
        ],
    )
    ax_dot.set_ylim(len(cv) - 0.4, -0.6)
    ax_dot.set_xlim(R2_VMIN, R2_VMAX)
    ax_dot.set_xticks(list(R2_TICKS))
    ax_dot.set_xlabel("Held-out R²")
    ax_dot.tick_params(axis="y", length=0, colors="black")
    ax_dot.spines["left"].set_visible(False)
    style.panel_label(ax_dot, "b", dx=-0.36, dy=1.02)

    # ── text columns: RMSE, bias, n ───────────────────────────────────
    ax_txt.set_axis_off()
    ax_txt.set_ylim(*ax_dot.get_ylim())
    # Keep the metric header below the panel-letter row. It remains above the
    # first class row without competing with the figure's primary hierarchy.
    head = dict(
        fontsize=NOTE_PT,
        fontweight="bold",
        color="black",
        ha="right",
        va="baseline",
        transform=ax_txt.transAxes,
    )
    ax_txt.text(COL_RMSE, METRIC_HEADER_Y, "RMSE", **head)
    ax_txt.text(COL_BIAS, METRIC_HEADER_Y, "Bias", **head)
    ax_txt.text(COL_N, METRIC_HEADER_Y, "n", **head)
    ax_txt.text(COL_LOCATIONS, METRIC_HEADER_Y, "Locations", **head)
    ax_txt.text(
        (COL_RMSE + COL_BIAS) / 2.0,
        METRIC_UNIT_Y,
        f"({style.LOG10_ABS_MPA_UNIT})",
        fontsize=NOTE_PT,
        color="black",
        ha="center",
        va="baseline",
        transform=ax_txt.transAxes,
    )
    body = dict(fontsize=NOTE_PT, color="black", ha="right", va="center")
    for y, (_, r) in zip(ypos, cv.iterrows()):
        rounded_bias = round(float(r["bias"]), 2)
        bias_text = (
            "0.00" if rounded_bias == 0.0 else f"{rounded_bias:+.2f}".replace("-", "−")
        )
        ax_txt.text(COL_RMSE, y, f"{r['rmse']:.2f}", **body)
        ax_txt.text(COL_BIAS, y, bias_text, **body)
        ax_txt.text(COL_N, y, f"{int(r['n_test']):,}", **body)
        ax_txt.text(COL_LOCATIONS, y, f"{int(r['n_test_locations']):,}", **body)

    out_dir = Path(output_dir)
    png = style.save(fig, out_dir / "fig06_regional_skill")
    print(f"Saved: {png}")
    print(f"Saved: {png.with_suffix('.pdf')}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Descriptor Fig 6: transferability to withheld climate classes"
    )
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args(argv)
    build_figure(args.output_dir)


if __name__ == "__main__":
    main()
