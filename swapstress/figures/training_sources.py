"""Descriptor Fig 2: the global training pool behind the CONUS application.

Panel a maps every in-model site on a Robinson world, colored by source; panel
b gives the paired-observation count per source as horizontal bars in the same
colors, so the bar panel doubles as the map's value key. Counts are computed
from the training table at render time, never hard-coded, so the figure cannot
drift from Table 1 -- both draw from the same rows (in-model = has lat/lon).

Five sources is more than the validated categorical trio in ``style`` covers.
The two extra hues extend that trio in place -- the first three slots are
``style.CATEGORICAL`` unchanged -- and the five-slot set passes the same six
checks the trio did (worst adjacent pair dE 11.0 protan / 17.6 tritan,
normal-vision floor 21.9, all slots >= 3:1 on white). The order below is the
adjacency the validation ran on; it is also Table 1's descending-pairs order,
so color assignment is fixed by entity, never by draw order.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import box

from swapstress.figures import style
from swapstress.figures.basemap import boundaries_root

TRAINING_TABLE = Path(
    "/nas/soils/swapstress/training/obs_level_training_9km_global.parquet"
)
LAND_SUBPATH = "boundaries/natural_earth/ne_110m_land.shp"

OUT_DIR = Path("figs/descriptor")
STEM = "fig_training_sources"

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

# Trim empty polar ocean and Antarctica (no training sites south of -55).
MAP_LON = (-180.0, 180.0)
MAP_LAT = (-60.0, 84.0)

FIG_WIDTH_MM = style.DOUBLE_COLUMN_MM
FIG_HEIGHT_MM = 72.0


def load_sites() -> pd.DataFrame:
    """In-model rows (lat/lon present), one row per site with its pair count."""
    df = pd.read_parquet(TRAINING_TABLE, columns=["source", "lat", "lon"])
    df = df[df[["lat", "lon"]].notna().all(axis=1)]
    df["lat_r"] = df["lat"].round(5)
    df["lon_r"] = df["lon"].round(5)
    return df


def build_figure(output_dir=OUT_DIR) -> Path:
    style.apply()

    df = load_sites()
    pairs = df["source"].value_counts()
    site_counts = df.groupby("source").apply(
        lambda g: g[["lat_r", "lon_r"]].drop_duplicates().shape[0],
        include_groups=False,
    )
    sites = df.drop_duplicates(["source", "lat_r", "lon_r"])

    land = gpd.read_file(Path(boundaries_root()) / LAND_SUBPATH)
    frame = box(MAP_LON[0], MAP_LAT[0], MAP_LON[1], MAP_LAT[1])
    land = gpd.clip(land, frame).to_crs(ROBINSON)
    frame_robin = (
        gpd.GeoSeries([frame.boundary], crs=4326).to_crs(ROBINSON).total_bounds
    )

    pts = gpd.GeoDataFrame(
        sites[["source"]],
        geometry=gpd.points_from_xy(sites["lon_r"], sites["lat_r"]),
        crs=4326,
    ).to_crs(ROBINSON)

    fig = plt.figure(
        figsize=style.figsize(FIG_WIDTH_MM, FIG_HEIGHT_MM), layout="constrained"
    )
    ax_map, ax_bar = fig.subplots(
        1, 2, width_ratios=(2.3, 1.0), gridspec_kw={"wspace": 0.02}
    )

    # -- a: site map -------------------------------------------------------
    land.plot(ax=ax_map, facecolor="#e8e8e6", edgecolor="none", zorder=1)
    # Largest source first so the sparse ones stay visible on top of it.
    for key, (_, color) in SOURCES.items():
        sel = pts[pts["source"] == key]
        ax_map.scatter(
            sel.geometry.x,
            sel.geometry.y,
            s=2.5,
            c=color,
            linewidths=0,
            zorder=2,
            rasterized=True,
        )
    ax_map.set_xlim(frame_robin[0], frame_robin[2])
    ax_map.set_ylim(frame_robin[1], frame_robin[3])
    ax_map.set_aspect("equal")
    ax_map.set_axis_off()

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
    ax_map.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.02),
        handletextpad=0.2,
        borderaxespad=0.0,
        labelspacing=0.25,
    )
    style.panel_label(ax_map, "a", dx=0.0)

    # -- b: pairs per source ----------------------------------------------
    keys = list(SOURCES)  # already descending by pairs
    y = range(len(keys))[::-1]
    ax_bar.barh(
        list(y),
        [pairs[k] for k in keys],
        color=[SOURCES[k][1] for k in keys],
        height=0.62,
    )
    for yi, k in zip(y, keys):
        ax_bar.text(
            pairs[k] + pairs.max() * 0.02,
            yi,
            f"{pairs[k]:,}",
            va="center",
            ha="left",
            fontsize=style.MAX_TEXT_PT - 1,
            color=style.AXIS_COLOR,
        )
    ax_bar.set_yticks(list(y), [SOURCES[k][0] for k in keys])
    ax_bar.set_xlim(0, pairs.max() * 1.18)
    ax_bar.set_xticks([0, 50_000, 100_000], ["0", "50k", "100k"])
    ax_bar.tick_params(axis="y", length=0)
    ax_bar.spines["left"].set_visible(False)
    ax_bar.set_xlabel("Paired observations")
    style.panel_label(ax_bar, "b", dx=-0.32)

    # Global-unique sites, not the per-source sum: co-located sites shared
    # between sources (NCSS rows GSHP ingested) would otherwise double-count,
    # and the figure must agree with Table 1's 2,607.
    total_pairs = int(pairs.sum())
    total_sites = df[["lat_r", "lon_r"]].drop_duplicates().shape[0]
    ax_bar.set_title(
        f"{total_pairs:,} pairs at {total_sites:,} sites",
        fontsize=style.MAX_TEXT_PT - 1,
        color=style.MUTED_INK,
        loc="right",
    )

    out = style.save(fig, Path(output_dir) / STEM)
    print(f"Saved: {out}")
    print(f"Saved: {out.with_suffix('.pdf')}")
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Descriptor Fig 2: global training sites and pairs per source."
    )
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args(argv)
    build_figure(args.output_dir)


if __name__ == "__main__":
    main()
