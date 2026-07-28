"""Figure 5: K-fold spatial CV validation — scatter + MGRS tile map + table.

Three-panel 16:9 slide:
  A (left)         Pooled density scatter of all 5-fold test predictions
  B (right top)    CONUS map of MGRS tiles colored by fold assignment
  C (right bottom) Compact per-fold metrics table

Usage:
    uv run python viz/presentation/fig5_kfold_validation.py
    uv run python viz/presentation/fig5_kfold_validation.py --output-dir figs/presentation
"""

import argparse
import json
from pathlib import Path

import warnings

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyBboxPatch
from shapely.geometry import Polygon

from swapstress.model.data import _tile_to_fold

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

KFOLD_DIR = Path(
    "/nas/soils/swapstress/releases/global_pruned_refresh_20260520/evaluation"
)
TRAINING_TABLE = Path(
    "/nas/soils/swapstress/training/obs_level_training_9km_global.parquet"
)
STATES_SHP = Path("/tmp/us_states/cb_2022_us_state_20m.shp")

N_FOLDS = 5
EXCLUDE_STUSPS = {"AK", "HI", "AS", "GU", "MP", "PR", "VI"}

FOLD_COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]


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
# Panel A: density scatter
# ---------------------------------------------------------------------------


def draw_scatter(ax, pred_df, summary):
    """Hexbin density scatter of pooled predictions."""
    obs = pred_df["observed"].values
    prd = pred_df["predicted"].values

    hb = ax.hexbin(
        obs,
        prd,
        gridsize=80,
        cmap="cividis",
        mincnt=1,
        norm=LogNorm(),
        linewidths=0.1,
        edgecolors="face",
    )

    lo = min(obs.min(), prd.min()) - 0.2
    hi = max(obs.max(), prd.max()) + 0.2
    ax.plot([lo, hi], [lo, hi], "k-", lw=0.8, zorder=5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    ax.set_xlabel(r"Observed log$_{10}$(suction) [cm H$_2$O]", fontsize=10)
    ax.set_ylabel(r"Predicted log$_{10}$(suction) [cm H$_2$O]", fontsize=10)

    agg = summary["aggregated"]
    n_total = sum(f["n_test"] for f in summary["per_fold"])
    txt = (
        f"R$^2$ = {agg['r2']['mean']:.3f} $\\pm$ {agg['r2']['std']:.3f}\n"
        f"RMSE = {agg['rmse']['mean']:.3f} $\\pm$ {agg['rmse']['std']:.3f}\n"
        f"n = {n_total:,}"
    )
    ax.text(
        0.05,
        0.95,
        txt,
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="0.7"
        ),
    )
    ax.set_title(
        "(a) Observed vs. Predicted (All Holdouts)", fontsize=11, loc="left", pad=8
    )
    return hb


# ---------------------------------------------------------------------------
# Panel B: CONUS tile map
# ---------------------------------------------------------------------------


def draw_tile_map(ax, tiles_gdf, states):
    """CONUS map of MGRS tiles rendered as true 100 km grid polygons."""
    states.boundary.plot(ax=ax, color="0.65", linewidth=0.4)

    # Filter to CONUS by bounding box of tile centroids
    centroids = tiles_gdf.geometry.centroid
    conus_mask = (
        (centroids.y > 24)
        & (centroids.y < 50)
        & (centroids.x > -125)
        & (centroids.x < -66)
    )
    conus = tiles_gdf[conus_mask].copy()

    from matplotlib.patches import Patch

    for k in range(N_FOLDS):
        fold_gdf = conus[conus["fold"] == k]
        fold_gdf.plot(
            ax=ax,
            facecolor=FOLD_COLORS[k],
            edgecolor="white",
            linewidth=0.3,
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

    legend_handles = [
        Patch(
            facecolor=FOLD_COLORS[k],
            edgecolor="white",
            linewidth=0.3,
            alpha=0.75,
            label=f"Fold {k}",
        )
        for k in range(N_FOLDS)
    ]
    ax.legend(
        handles=legend_handles,
        fontsize=7,
        loc="lower left",
        frameon=True,
        framealpha=0.9,
        edgecolor="0.7",
        ncol=3,
        handletextpad=0.3,
        columnspacing=0.8,
    )
    ax.set_title(
        f"(b) MGRS tile holdout blocks ({len(conus)} CONUS tiles, 100 km)",
        fontsize=11,
        loc="left",
        pad=8,
    )


# ---------------------------------------------------------------------------
# Panel C: metrics table
# ---------------------------------------------------------------------------


def draw_table(ax, summary):
    """Render per-fold metrics as a styled matplotlib table."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    per_fold = summary["per_fold"]
    agg = summary["aggregated"]

    col_headers = ["Fold", "R\u00b2", "RMSE", "n"]
    col_x = [0.10, 0.35, 0.55, 0.78]
    row_h = 0.115
    top_y = 0.92
    header_y = top_y

    # Header
    for cx, hdr in zip(col_x, col_headers):
        ax.text(
            cx, header_y, hdr, fontsize=10, fontweight="bold", ha="center", va="center"
        )
    ax.plot([0.02, 0.98], [header_y - row_h * 0.45] * 2, color="0.4", lw=0.8)

    # Data rows
    for i, f in enumerate(per_fold):
        y = header_y - row_h * (i + 1)
        color = FOLD_COLORS[f["fold"]]

        # Color swatch
        swatch = FancyBboxPatch(
            (col_x[0] - 0.06, y - 0.025),
            0.03,
            0.05,
            boxstyle="round,pad=0.005",
            facecolor=color,
            edgecolor="none",
        )
        ax.add_patch(swatch)
        ax.text(
            col_x[0] + 0.01, y, str(f["fold"]), fontsize=9, ha="center", va="center"
        )

        r2 = f["r2"]
        rmse = f["rmse"]
        n = f["n_test"]
        ax.text(
            col_x[1],
            y,
            f"{r2:.3f}",
            fontsize=9,
            ha="center",
            va="center",
            family="monospace",
        )
        ax.text(
            col_x[2],
            y,
            f"{rmse:.3f}",
            fontsize=9,
            ha="center",
            va="center",
            family="monospace",
        )
        ax.text(
            col_x[3],
            y,
            f"{n:,}",
            fontsize=9,
            ha="center",
            va="center",
            family="monospace",
        )

    # Summary row
    sep_y = header_y - row_h * (N_FOLDS + 0.55)
    ax.plot([0.02, 0.98], [sep_y] * 2, color="0.4", lw=0.8)
    sum_y = header_y - row_h * (N_FOLDS + 1)
    ax.text(
        col_x[0],
        sum_y,
        "Mean\u00b1Std",
        fontsize=8,
        fontweight="bold",
        ha="center",
        va="center",
    )
    r2_str = f"{agg['r2']['mean']:.3f}\u00b1{agg['r2']['std']:.3f}"
    rmse_str = f"{agg['rmse']['mean']:.3f}\u00b1{agg['rmse']['std']:.3f}"
    n_total = sum(f["n_test"] for f in per_fold)
    ax.text(
        col_x[1],
        sum_y,
        r2_str,
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="center",
        family="monospace",
    )
    ax.text(
        col_x[2],
        sum_y,
        rmse_str,
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="center",
        family="monospace",
    )
    ax.text(
        col_x[3],
        sum_y,
        f"{n_total:,}",
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="center",
        family="monospace",
    )

    ax.set_title("(c) Per-fold metrics", fontsize=11, loc="left", pad=8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    pred_df = load_pooled_predictions()
    summary = load_kfold_summary()
    print("  Building MGRS tile polygons...")
    tiles_gdf = load_tile_geometries()
    states = load_conus_states()

    print(f"  {len(pred_df):,} pooled predictions")
    print(f"  {len(tiles_gdf)} MGRS tile polygons")

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[0.45, 0.55],
        height_ratios=[0.6, 0.4],
        wspace=0.08,
        hspace=0.25,
    )
    ax_scatter = fig.add_subplot(gs[:, 0])
    ax_map = fig.add_subplot(gs[0, 1])
    ax_table = fig.add_subplot(gs[1, 1])

    draw_scatter(ax_scatter, pred_df, summary)
    draw_tile_map(ax_map, tiles_gdf, states)
    draw_table(ax_table, summary)

    fig.suptitle(
        "5-fold spatial cross-validation on MGRS tiles (100 km blocks)",
        fontsize=13,
        y=0.98,
    )

    for ext in ("png", "pdf"):
        out = output_dir / f"fig5_kfold_validation.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"  Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Figure 5: K-fold validation scatter + MGRS tile map",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figs/presentation",
        help="Output directory (default: figs/presentation/)",
    )
    args = parser.parse_args()
    main(args.output_dir)
