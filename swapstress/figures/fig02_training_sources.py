"""Descriptor Fig 2: the global training pool behind the CONUS application.

Panel a gives the distribution of paired observations per sample layer as
stacked horizontal bars, so the depth of each source's retention curves is
visible next to its geographic reach; panel b maps every in-model site on a
Robinson world with a dashed rectangle marking panel c's extent; panel c zooms
to the conterminous United States on an equal-area Albers with state outlines,
where most non-GSHP sites cluster. Counts are computed from the training table
at render time, never hard-coded, so the figure cannot drift from Table 1 --
all panels draw from the same rows (in-model = has lat/lon).

Five sources is more than the validated categorical trio in ``style`` covers.
The two extra hues extend that trio in place -- the first three slots are
``style.CATEGORICAL`` unchanged -- and the five-slot set passes the same six
checks the trio did (worst adjacent pair dE 11.0 protan / 17.6 tritan,
normal-vision floor 21.9, all slots >= 3:1 on white). The order below is the
adjacency the validation ran on; it is also Table 1's descending-pairs order,
so color assignment is fixed by entity, never by draw order.

GSHP rows ingested without layer metadata (null ``sample_id`` and depth)
cannot be assigned to a sample and are excluded from panel a only; they remain
in the maps and legend counts. The exclusion is printed at render time and
stated in the caption.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import box

from swapstress.figures import style
from swapstress.figures.basemap import boundaries_root, load_conus_states

TRAINING_TABLE = Path(
    "/nas/soils/swapstress/training/obs_level_training_9km_global.parquet"
)
LAND_SUBPATH = "boundaries/natural_earth/ne_110m_land.shp"

OUT_DIR = Path("figs/descriptor")
STEM = "fig02_training_sources"

# Fixed source order (Table 1, descending pairs) and the five-slot palette
# validated in that adjacency. Keys are the training table's source labels.
SOURCES = {
    "gshp": ("GSHP", "#2166AC"),
    "reesh": ("ReESH", "#D55E00"),
    "mt_mesonet": ("MT Mesonet", "#7B3294"),
    "ncss": ("NCSS", "#117733"),
    "lacadian": ("LaCADIAN", "#BB8800"),
}

ROBINSON = "+proj=robin +lon_0=0 +datum=WGS84 +units=m +no_defs"
ALBERS_CONUS = 5070

# Trim empty polar ocean and Antarctica (no training sites south of -55).
MAP_LON = (-180.0, 180.0)
MAP_LAT = (-60.0, 84.0)

# Paired observations per sample: single-count bins from the standardization
# minimum of 4 through 10, then widening ranges. Lab curves sit low (NCSS 4-8,
# GSHP median 11); the in-situ sensor sources accumulate hundreds of pairs per
# probe depth. The 8 samples (19 pairs) that slipped below 4 past the
# standardization gate fall outside the bins and are counted in the render
# printout and the caption, not the bars.
BIN_EDGES = [3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]
BIN_EDGES += [14.5, 19.5, 29.5, 49.5, 99.5, np.inf]
BIN_LABELS = [str(i) for i in range(4, 11)]
BIN_LABELS += ["11–14", "15–19", "20–29", "30–49", "50–99", "100+"]

FIG_WIDTH_MM = style.DOUBLE_COLUMN_MM
FIG_HEIGHT_MM = 128.0


def load_rows() -> pd.DataFrame:
    """In-model rows (lat/lon present) with their sample-layer identifier."""
    df = pd.read_parquet(TRAINING_TABLE, columns=["source", "sample_id", "lat", "lon"])
    df = df[df[["lat", "lon"]].notna().all(axis=1)]
    df["lat_r"] = df["lat"].round(5)
    df["lon_r"] = df["lon"].round(5)
    return df


def build_figure(output_dir=OUT_DIR) -> Path:
    style.apply()

    df = load_rows()
    site_counts = df.groupby("source").apply(
        lambda g: g[["lat_r", "lon_r"]].drop_duplicates().shape[0],
        include_groups=False,
    )
    sites = df.drop_duplicates(["source", "lat_r", "lon_r"])

    # -- panel a data: samples per pairs-per-sample bin, by source ---------
    with_id = df[df["sample_id"].notna()]
    dropped = len(df) - len(with_id)
    per_sample = (
        with_id.groupby(["source", "sample_id"]).size().rename("pairs").reset_index()
    )
    # Filter explicitly rather than letting sub-minimum rows fall out as NaN
    # bins: pandas 3.0's groupby/unstack folds NaN-binned rows into the last
    # category, which silently inflated the top bin.
    below_min = per_sample[per_sample["pairs"] < BIN_EDGES[0]]
    per_sample = per_sample[per_sample["pairs"] >= BIN_EDGES[0]]
    per_sample["bin"] = pd.cut(per_sample["pairs"], bins=BIN_EDGES, labels=BIN_LABELS)
    bin_table = (
        per_sample.groupby(["bin", "source"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=list(SOURCES), fill_value=0)
    )
    n_samples = int(bin_table.to_numpy().sum())
    print(
        f"Panel a: {n_samples:,} samples from {len(with_id):,} pairs; "
        f"{dropped:,} GSHP pairs without a sample_id and "
        f"{len(below_min)} samples below 4 pairs ({int(below_min['pairs'].sum())} "
        f"pairs) are excluded from panel a only."
    )

    # -- map geometry ------------------------------------------------------
    land = gpd.read_file(Path(boundaries_root()) / LAND_SUBPATH)
    frame = box(MAP_LON[0], MAP_LAT[0], MAP_LON[1], MAP_LAT[1])
    land = gpd.clip(land, frame).to_crs(ROBINSON)
    frame_robin = (
        gpd.GeoSeries([frame.boundary], crs=4326).to_crs(ROBINSON).total_bounds
    )

    states = load_conus_states()
    states_albers = states.to_crs(ALBERS_CONUS)
    conus_union = states.union_all()

    pts = gpd.GeoDataFrame(
        sites[["source", "lat_r", "lon_r"]],
        geometry=gpd.points_from_xy(sites["lon_r"], sites["lat_r"]),
        crs=4326,
    )
    pts_robin = pts.to_crs(ROBINSON)
    in_conus = pts[pts.within(conus_union)]
    pts_albers = in_conus.to_crs(ALBERS_CONUS)

    # Dashed rectangle on the world map marking panel c's extent: the CONUS
    # lon/lat bounding box, densified so it curves correctly under Robinson.
    lon0, lat0, lon1, lat1 = states.total_bounds + np.array([-1, -1, 1, 1])
    edge = np.linspace(0.0, 1.0, 60)
    ring_lon = np.concatenate(
        [
            lon0 + (lon1 - lon0) * edge,
            np.full_like(edge, lon1),
            lon1 - (lon1 - lon0) * edge,
            np.full_like(edge, lon0),
        ]
    )
    ring_lat = np.concatenate(
        [
            np.full_like(edge, lat0),
            lat0 + (lat1 - lat0) * edge,
            np.full_like(edge, lat1),
            lat1 - (lat1 - lat0) * edge,
        ]
    )
    ring = gpd.GeoSeries(gpd.points_from_xy(ring_lon, ring_lat), crs=4326).to_crs(
        ROBINSON
    )

    # -- layout: histogram spans the left column; the two maps stack on the
    # right with row heights matched to their projected aspect ratios so
    # neither map letterboxes.
    world_aspect = (frame_robin[3] - frame_robin[1]) / (frame_robin[2] - frame_robin[0])
    ab = states_albers.total_bounds
    conus_aspect = (ab[3] - ab[1]) / (ab[2] - ab[0])

    fig = plt.figure(
        figsize=style.figsize(FIG_WIDTH_MM, FIG_HEIGHT_MM), layout="constrained"
    )
    # The world row gets a small boost over its raw aspect: its title and the
    # constrained-layout padding eat height before the equal-aspect map is
    # placed, and without the boost the map letterboxes inside its cell.
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(0.55, 1.0),
        height_ratios=(world_aspect * 1.22, conus_aspect),
        wspace=0.04,
        hspace=0.02,
    )
    ax_hist = fig.add_subplot(grid[:, 0])
    ax_world = fig.add_subplot(grid[0, 1])
    ax_conus = fig.add_subplot(grid[1, 1])

    # -- a: paired observations per sample ---------------------------------
    y = np.arange(len(BIN_LABELS))[::-1]
    left = np.zeros(len(BIN_LABELS))
    for key, (_, color) in SOURCES.items():
        counts = bin_table[key].to_numpy(dtype=float)
        ax_hist.barh(y, counts, left=left, color=color, height=0.72)
        left += counts
    ax_hist.set_yticks(y, BIN_LABELS)
    ax_hist.set_ylim(-0.6, len(BIN_LABELS) - 0.4)
    ax_hist.set_xlim(0, left.max() * 1.04)
    ax_hist.set_xlabel("Samples")
    ax_hist.set_ylabel("Paired observations per sample")
    ax_hist.set_title(f"Retention data per sample (n = {n_samples:,})")
    ax_hist.tick_params(axis="y", length=0)
    ax_hist.spines["left"].set_visible(False)
    style.panel_label(ax_hist, "a", dx=-0.14)

    # -- b: global site map ------------------------------------------------
    land.plot(ax=ax_world, facecolor="#e8e8e6", edgecolor="none", zorder=1)
    # Map draw order is descending site count -- the biggest network is painted
    # first so the sparse ones land on top of it in the crowded CONUS cluster.
    # Only the painting order changes: the legend and histogram keep the fixed
    # SOURCES order, and color stays assigned by entity.
    draw_order = sorted(SOURCES, key=lambda k: int(site_counts[k]), reverse=True)
    for key in draw_order:
        sel = pts_robin[pts_robin["source"] == key]
        ax_world.scatter(
            sel.geometry.x,
            sel.geometry.y,
            s=2.5,
            c=SOURCES[key][1],
            linewidths=0,
            zorder=2,
            rasterized=True,
        )
    ax_world.plot(
        ring.x, ring.y, color=style.AXIS_COLOR, lw=0.6, ls=(0, (4, 2)), zorder=3
    )
    ax_world.set_xlim(frame_robin[0], frame_robin[2])
    ax_world.set_ylim(frame_robin[1], frame_robin[3])
    ax_world.set_aspect("equal")
    ax_world.set_axis_off()

    # Global-unique locations, not the per-source sum: co-located sites shared
    # between sources (NCSS rows GSHP ingested) would otherwise double-count,
    # and the figure must agree with Table 1's unique-location total. The
    # legend's per-source counts are within-source sites.
    total_sites = df[["lat_r", "lon_r"]].drop_duplicates().shape[0]
    ax_world.set_title(f"Training sites ({total_sites:,} unique locations)")

    handles = [
        plt.Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=3.5,
            markeredgewidth=0,
            color=color,
            label=f"{label} ({site_counts[key]:,} sites)",
        )
        for key, (label, color) in SOURCES.items()
    ]
    ax_world.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.02),
        handletextpad=0.2,
        borderaxespad=0.0,
        labelspacing=0.25,
    )
    style.panel_label(ax_world, "b", dx=0.0)

    # -- c: CONUS zoom -----------------------------------------------------
    states_albers.plot(
        ax=ax_conus, facecolor="#e8e8e6", edgecolor="white", linewidth=0.4, zorder=1
    )
    for key in draw_order:
        sel = pts_albers[pts_albers["source"] == key]
        ax_conus.scatter(
            sel.geometry.x,
            sel.geometry.y,
            s=4.0,
            c=SOURCES[key][1],
            linewidths=0,
            zorder=2,
            rasterized=True,
        )
    pad = 40_000.0
    ax_conus.set_xlim(ab[0] - pad, ab[2] + pad)
    ax_conus.set_ylim(ab[1] - pad, ab[3] + pad)
    ax_conus.set_aspect("equal")
    ax_conus.set_axis_off()
    # Unique locations, matching panel b's definition -- counting per-source
    # site rows here would double-count locations shared between sources.
    conus_locs = in_conus[["lat_r", "lon_r"]].drop_duplicates().shape[0]
    ax_conus.set_title(f"Conterminous United States ({conus_locs:,} unique locations)")
    style.panel_label(ax_conus, "c", dx=0.0)

    out = style.save(fig, Path(output_dir) / STEM)
    print(f"Saved: {out}")
    print(f"Saved: {out.with_suffix('.pdf')}")
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Descriptor Fig 2: pairs per sample, global training sites, and "
            "the CONUS cluster."
        )
    )
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args(argv)
    build_figure(args.output_dir)


if __name__ == "__main__":
    main()
