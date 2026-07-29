"""Supporting analysis: 5-fold spatial cross-validation on MGRS tiles.

Three panels:
  a  Pooled density of all five folds' test predictions against observations
  b  CONUS map of the MGRS tiles, coloured by the fold they were held out in
  c  Per-fold metrics

This backs the Technical Validation text rather than being one of the
descriptor's Figs 1-6, so ``swapstress-figures`` renders it on request and not
as part of ``--figure all``. It still follows ``swapstress.figures.style``: the
same 183 mm double column, the same 7 pt ceiling and the same shared ``save``,
so a supporting panel dropped beside a main figure does not arrive in a
different typeface or at a different width.

Usage:
    uv run swapstress-figures --figure kfold
    uv run swapstress-figures --figure kfold --output-dir figs/descriptor
"""

import argparse
import json
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyBboxPatch, Patch
from shapely.geometry import Polygon

from swapstress.figures import style
from swapstress.figures.basemap import states_shapefile
from swapstress.model.data import _tile_to_fold

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

# Stage 04 writes the folds into an ``evaluation/kfold`` subdirectory; the
# release ``evaluation/`` root also holds the L3-vs-L4 comparison, so the fold
# directories are one level further down than they look.
KFOLD_DIR = Path(
    "/nas/soils/swapstress/releases/global_pruned_refresh_20260520/evaluation/kfold"
)
TRAINING_TABLE = Path(
    "/nas/soils/swapstress/training/obs_level_training_9km_global.parquet"
)
STATES_SHP = Path(states_shapefile())

DEFAULT_OUTPUT_DIR = "figs/descriptor"

N_FOLDS = 5
EXCLUDE_STUSPS = {"AK", "HI", "AS", "GU", "MP", "PR", "VI"}

# Fold identity is a nominal label, not a quantity, so it needs a qualitative
# set -- and five of them, which is more than the validated categorical trio in
# ``style`` covers. These are the Tableau 10 leading five: distinguishable
# under the common CVD forms and used nowhere a reader must read a value off
# them, only to tell one holdout block from its neighbour.
FOLD_COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]

# Double column. The scatter is equal-aspect and the map needs the width to
# keep 100 km tiles from merging, so the two sit side by side rather than
# stacked; the depth is what the map plus the metrics table below it need.
FIG_WIDTH_MM = style.DOUBLE_COLUMN_MM
FIG_HEIGHT_MM = 105.0

BODY_PT = style.MAX_TEXT_PT - 1


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_pooled_predictions() -> pd.DataFrame:
    """Pool predictions.parquet from all folds."""
    frames = []
    for k in range(N_FOLDS):
        p = KFOLD_DIR / f"fold_{k}" / "predictions.parquet"
        df = pd.read_parquet(p)
        df["fold"] = k
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_kfold_summary() -> dict:
    with open(KFOLD_DIR / "kfold_summary.json") as f:
        return json.load(f)


def _mgrs_tile_polygon(tile_id: str) -> Polygon | None:
    """Return a Shapely polygon for a 100 km MGRS tile from its 5-char ID."""
    import mgrs

    m = mgrs.MGRS()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sw = m.toLatLon(tile_id)
            se = m.toLatLon(tile_id + "9999900000")
            ne = m.toLatLon(tile_id + "9999999999")
            nw = m.toLatLon(tile_id + "0000099999")
        # (lon, lat) order for Shapely
        return Polygon(
            [
                (sw[1], sw[0]),
                (se[1], se[0]),
                (ne[1], ne[0]),
                (nw[1], nw[0]),
            ]
        )
    except Exception:
        return None


def load_tile_geometries() -> gpd.GeoDataFrame:
    """Build a GeoDataFrame of MGRS tile polygons with fold assignments."""
    df = pd.read_parquet(TRAINING_TABLE, columns=["MGRS_TILE", "lat", "lon"])
    df = df.dropna(subset=["MGRS_TILE"])
    tile_ids = df["MGRS_TILE"].unique()

    records = []
    for tid in tile_ids:
        poly = _mgrs_tile_polygon(tid)
        if poly is not None:
            records.append(
                {
                    "MGRS_TILE": tid,
                    "fold": _tile_to_fold(tid, N_FOLDS),
                    "geometry": poly,
                }
            )
    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    return gdf


def load_conus_states() -> gpd.GeoDataFrame:
    states = gpd.read_file(STATES_SHP)
    return states[~states.STUSPS.isin(EXCLUDE_STUSPS)].copy()


# ---------------------------------------------------------------------------
# Panel a: density scatter
# ---------------------------------------------------------------------------


def draw_scatter(ax, pred_df, summary):
    """Hexbin density of pooled predictions. Returns the mappable for the key."""
    obs = pred_df["observed"].values
    prd = pred_df["predicted"].values

    hb = ax.hexbin(
        obs,
        prd,
        gridsize=80,
        cmap=style.SEQUENTIAL,
        mincnt=1,
        norm=LogNorm(),
        linewidths=0.1,
        edgecolors="face",
        # Rasterises the cells only; the 1:1 line, the axes and every label
        # stay vector, which is what the artwork guide asks for.
        rasterized=True,
    )

    lo = min(obs.min(), prd.min()) - 0.2
    hi = max(obs.max(), prd.max()) + 0.2
    ax.plot([lo, hi], [lo, hi], color="black", lw=0.6, dashes=(2.6, 1.8), zorder=5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    ax.set_xlabel(r"Observed $\log_{10}$ suction (cm)")
    ax.set_ylabel(r"Predicted $\log_{10}$ suction (cm)")

    agg = summary["aggregated"]
    n_total = sum(f["n_test"] for f in summary["per_fold"])
    ax.text(
        0.04,
        0.96,
        f"R$^2$ = {agg['r2']['mean']:.3f} ± {agg['r2']['std']:.3f}\n"
        f"RMSE = {agg['rmse']['mean']:.3f} ± {agg['rmse']['std']:.3f}\n"
        f"n = {n_total:,}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=BODY_PT,
        linespacing=1.35,
        zorder=6,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.2),
    )
    ax.set_title("Observed vs predicted, all holdouts", loc="left", pad=2.5)
    style.panel_label(ax, "a", dx=-0.14, dy=1.02)
    return hb


# ---------------------------------------------------------------------------
# Panel b: CONUS tile map
# ---------------------------------------------------------------------------


def draw_tile_map(ax, tiles_gdf, states):
    """CONUS map of MGRS tiles rendered as true 100 km grid polygons."""
    states.boundary.plot(ax=ax, color="0.65", linewidth=0.3)

    # Filter to CONUS on each tile's bounding-box midpoint. A true centroid of
    # a lon/lat polygon is what geopandas warns about, and this is only a
    # coarse "is the tile in the country" test -- the midpoint answers it
    # without pretending to a planar measure on a geographic CRS.
    box = tiles_gdf.geometry.bounds
    mid_lon = (box["minx"] + box["maxx"]) / 2.0
    mid_lat = (box["miny"] + box["maxy"]) / 2.0
    conus = tiles_gdf[mid_lat.between(24, 50) & mid_lon.between(-125, -66)].copy()

    for k in range(N_FOLDS):
        conus[conus["fold"] == k].plot(
            ax=ax,
            facecolor=FOLD_COLORS[k],
            edgecolor="white",
            linewidth=0.25,
            alpha=0.75,
            zorder=3,
        )

    ax.set_xlim(-126, -65)
    ax.set_ylim(23, 51)
    ax.set_aspect(1.3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(
        handles=[
            Patch(
                facecolor=FOLD_COLORS[k],
                edgecolor="white",
                linewidth=0.25,
                alpha=0.75,
                label=f"Fold {k}",
            )
            for k in range(N_FOLDS)
        ],
        fontsize=style.MIN_TEXT_PT + 1,
        loc="lower left",
        ncol=3,
        handlelength=1.0,
        handleheight=1.0,
        handletextpad=0.3,
        columnspacing=0.8,
        borderpad=0.2,
    )
    ax.set_title(
        f"MGRS tile holdout blocks ({len(conus)} CONUS tiles, 100 km)",
        loc="left",
        pad=2.5,
    )
    style.panel_label(ax, "b", dx=-0.045, dy=1.02)


# ---------------------------------------------------------------------------
# Panel c: metrics table
# ---------------------------------------------------------------------------


def draw_table(ax, summary):
    """Render per-fold metrics as a small typeset table."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    per_fold = summary["per_fold"]
    agg = summary["aggregated"]

    col_x = [0.10, 0.35, 0.58, 0.82]
    row_h = 0.115
    header_y = 0.90

    head = dict(fontsize=BODY_PT, fontweight="bold", color=style.AXIS_COLOR)
    body = dict(fontsize=BODY_PT, color=style.AXIS_COLOR)

    for cx, hdr in zip(col_x, ["Fold", "R²", "RMSE", "n"]):
        ax.text(cx, header_y, hdr, ha="center", va="center", **head)
    ax.plot(
        [0.02, 0.98],
        [header_y - row_h * 0.45] * 2,
        color=style.AXIS_COLOR,
        lw=0.5,
        solid_capstyle="butt",
    )

    for i, f in enumerate(per_fold):
        y = header_y - row_h * (i + 1)
        ax.add_patch(
            FancyBboxPatch(
                (col_x[0] - 0.06, y - 0.022),
                0.03,
                0.044,
                boxstyle="round,pad=0.004",
                facecolor=FOLD_COLORS[f["fold"]],
                edgecolor="none",
            )
        )
        for cx, txt in zip(
            col_x,
            [
                str(f["fold"]),
                f"{f['r2']:.3f}",
                f"{f['rmse']:.3f}",
                f"{f['n_test']:,}",
            ],
        ):
            ax.text(cx, y, txt, ha="center", va="center", **body)

    sep_y = header_y - row_h * (N_FOLDS + 0.55)
    ax.plot(
        [0.02, 0.98],
        [sep_y] * 2,
        color=style.AXIS_COLOR,
        lw=0.5,
        solid_capstyle="butt",
    )
    sum_y = header_y - row_h * (N_FOLDS + 1)
    n_total = sum(f["n_test"] for f in per_fold)
    summary_row = [
        "Mean ± sd",
        f"{agg['r2']['mean']:.3f} ± {agg['r2']['std']:.3f}",
        f"{agg['rmse']['mean']:.3f} ± {agg['rmse']['std']:.3f}",
        f"{n_total:,}",
    ]
    for cx, txt in zip(col_x, summary_row):
        ax.text(cx, sum_y, txt, ha="center", va="center", **head)

    ax.set_title("Per-fold metrics", loc="left", pad=2.5)
    style.panel_label(ax, "c", dx=-0.045, dy=1.02)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def render(pred_df, summary, tiles_gdf, states, output_dir: str) -> Path:
    """Assemble the three panels and write them at the declared size."""
    style.apply()

    fig = plt.figure(
        figsize=style.figsize(FIG_WIDTH_MM, FIG_HEIGHT_MM),
        layout="constrained",
    )
    gs = fig.add_gridspec(2, 2, width_ratios=[0.45, 0.55], height_ratios=[0.62, 0.38])
    ax_scatter = fig.add_subplot(gs[:, 0])
    ax_map = fig.add_subplot(gs[0, 1])
    ax_table = fig.add_subplot(gs[1, 1])

    hb = draw_scatter(ax_scatter, pred_df, summary)
    draw_tile_map(ax_map, tiles_gdf, states)
    draw_table(ax_table, summary)

    # The density is on a log ramp, so it needs a key -- without one the panel
    # says "there are more points here" and never says how many more.
    bar = fig.colorbar(
        hb, ax=ax_scatter, orientation="horizontal", shrink=0.7, aspect=28, pad=0.02
    )
    bar.set_label("Observations per hexagon", fontsize=BODY_PT, labelpad=2.0)
    bar.ax.tick_params(labelsize=BODY_PT, length=1.8, width=0.4, pad=1.5)
    bar.outline.set_linewidth(0.4)
    bar.outline.set_edgecolor(style.AXIS_COLOR)

    fig.suptitle(
        "5-fold spatial cross-validation on MGRS tiles (100 km blocks)",
        fontsize=style.MAX_TEXT_PT,
    )

    return style.save(fig, Path(output_dir) / "kfold_validation")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Supporting analysis: k-fold spatial CV scatter, tile map "
        "and per-fold metrics.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    args = parser.parse_args(argv)

    pred_df = load_pooled_predictions()
    summary = load_kfold_summary()
    tiles_gdf = load_tile_geometries()
    states = load_conus_states()

    print(f"  {len(pred_df):,} pooled predictions")
    print(f"  {len(tiles_gdf)} MGRS tile polygons")

    path = render(pred_df, summary, tiles_gdf, states, args.output_dir)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
