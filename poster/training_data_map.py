import os
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from shapely.geometry import box


def _load_sources(include_rosetta=True):
    home_ = os.path.expanduser('~')
    root_ = os.path.join(home_, 'data', 'IrrigationGIS')
    soils_ = os.path.join(root_, 'soils')
    reesh_shp = os.path.join(soils_, 'soil_potential_obs', 'reesh', 'shapefile', 'reesh_sites_mgrs.shp')
    mesonet_shp = os.path.join(soils_, 'soil_potential_obs', 'mt_mesonet', 'station_metadata_clean_mgrs.shp')
    gshp_shp = os.path.join(soils_, 'soil_potential_obs', 'gshp', 'wrc_aggregated_mgrs.shp')
    rosetta_pts = os.path.join(soils_, 'gis', 'pretraining-roi-10000_mgrs.shp')
    reesh = gpd.read_file(reesh_shp).to_crs(4326)
    mesonet = gpd.read_file(mesonet_shp).to_crs(4326)
    gshp = gpd.read_file(gshp_shp).to_crs(4326)
    rosetta = gpd.read_file(rosetta_pts).to_crs(4326) if include_rosetta else None
    return reesh, mesonet, gshp, rosetta


def _find_land_shapefile():
    home_ = os.path.expanduser('~')
    shp_ = os.path.join(home_, 'data', 'IrrigationGIS', 'boundaries', 'natural_earth', 'ne_110m_land.shp')
    return shp_


def _apply_modern_style():
    sns.set_theme(style='white', context='talk', font_scale=0.9)
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#9aa0a6',
        'axes.linewidth': 0.6,
        'xtick.color': '#4d4d4d',
        'ytick.color': '#4d4d4d',
        'legend.frameon': True,
        'legend.facecolor': 'white',
        'legend.edgecolor': '#dadce0',
    })


def plot_training_data_map(out_path, figsize=(10, 6), dpi=300, include_rosetta=True):
    _apply_modern_style()
    reesh, mesonet, gshp, rosetta = _load_sources(include_rosetta=include_rosetta)

    land_shp = _find_land_shapefile()
    land = gpd.read_file(land_shp).to_crs(4326)

    # Load CONUS outline polygon
    home_ = os.path.expanduser('~')
    conus_shp = os.path.join(home_, 'data', 'IrrigationGIS', 'boundaries', 'world_countries', 'united_states_conus.shp')
    conus = gpd.read_file(conus_shp).to_crs(4326)

    # Build figure with two panels stacked vertically: Global (top) and CONUS (bottom)
    fig, (ax_g, ax_c) = plt.subplots(2, 1, figsize=(figsize[0], figsize[1] * 2.2), dpi=dpi)

    # Global panel
    ax_g.set_facecolor('#eef5fa')
    land.plot(ax=ax_g, facecolor='white', edgecolor='#9aa0a6', linewidth=0.6, zorder=1)
    ax_g.set_aspect('auto')
    # Build labels with total counts
    lbl_reesh = f"ReESH (n = {len(reesh)})"
    lbl_mesonet = f"MT Mesonet (n = {len(mesonet)})"
    lbl_gshp = f"GSHP (n = {len(gshp)})"
    lbl_rosetta = f"Rosetta3 Points (n = {len(rosetta)})" if include_rosetta and rosetta is not None else None
    # Exclude CONUS points from global view
    reesh_in = gpd.sjoin(reesh, conus[['geometry']], predicate='intersects', how='inner')
    mesonet_in = gpd.sjoin(mesonet, conus[['geometry']], predicate='intersects', how='inner')
    gshp_in = gpd.sjoin(gshp, conus[['geometry']], predicate='intersects', how='inner')
    rosetta_in = gpd.sjoin(rosetta, conus[['geometry']], predicate='intersects', how='inner') if include_rosetta and rosetta is not None else None
    reesh_g = reesh.loc[~reesh.index.isin(reesh_in.index)]
    mesonet_g = mesonet.loc[~mesonet.index.isin(mesonet_in.index)]
    gshp_g = gshp.loc[~gshp.index.isin(gshp_in.index)]
    if not reesh_g.empty:
        reesh_g.plot(ax=ax_g, markersize=18, marker='^', color='#d62728', linewidth=0.2, zorder=3)
    if not mesonet_g.empty:
        mesonet_g.plot(ax=ax_g, markersize=16, marker='o', color='#1f77b4', linewidth=0.2, zorder=3)
    if not gshp_g.empty:
        gshp_g.plot(ax=ax_g, markersize=16, marker='s', color='#2ca02c', linewidth=0.2, zorder=3)
    # Rosetta points are omitted from the global panel
    # Grey-conus overlay matching outline
    conus.plot(ax=ax_g, facecolor='#bfbfbf', edgecolor='none', alpha=0.35, zorder=4)
    ax_g.set_xlim(-180, 180)
    ax_g.set_ylim(-60, 85)
    ax_g.set_xlabel('Longitude')
    ax_g.set_ylabel('Latitude')
    legend_elems = [
        Line2D([0], [0], marker='^', color='none', markerfacecolor='#d62728', markeredgecolor='#d62728', markersize=7,
               linestyle='None', label=lbl_reesh),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='#1f77b4', markeredgecolor='#1f77b4', markersize=7,
               linestyle='None', label=lbl_mesonet),
        Line2D([0], [0], marker='s', color='none', markerfacecolor='#2ca02c', markeredgecolor='#2ca02c', markersize=7,
               linestyle='None', label=lbl_gshp),
    ]
    ax_g.legend(
        handles=legend_elems,
        title='Global Training Data',
        frameon=True,
        framealpha=0.95,
        fancybox=False,
        edgecolor='#dadce0',
        loc='lower left',
        borderpad=0.6,
        handletextpad=0.6,
        fontsize=9,
        title_fontsize=10,
    )
    ax_g.grid(False)

    # CONUS panel
    ax_c.set_facecolor('#eef5fa')
    land.plot(ax=ax_c, facecolor='white', edgecolor='#9aa0a6', linewidth=0.6, zorder=1)
    ax_c.set_aspect('auto')
    # Include only points in CONUS bounding box
    xmin, ymin, xmax, ymax = conus.total_bounds
    reesh_c = reesh.cx[xmin:xmax, ymin:ymax]
    mesonet_c = mesonet.cx[xmin:xmax, ymin:ymax]
    gshp_c = gshp.cx[xmin:xmax, ymin:ymax]
    rosetta_c = rosetta.cx[xmin:xmax, ymin:ymax] if include_rosetta and rosetta is not None else None
    if not reesh_c.empty:
        reesh_c.plot(ax=ax_c, markersize=24, marker='^', color='#d62728', linewidth=0.2, zorder=3)
    if not mesonet_c.empty:
        mesonet_c.plot(ax=ax_c, markersize=16, marker='o', color='#1f77b4', linewidth=0.2, zorder=3)
    if not gshp_c.empty:
        gshp_c.plot(ax=ax_c, markersize=16, marker='s', color='#2ca02c', linewidth=0.2, zorder=3)
    if rosetta_c is not None and not rosetta_c.empty:
        rosetta_c.plot(ax=ax_c, markersize=3, marker='x', color='#ff7f0e', linewidth=0.2, zorder=3)

    # Draw CONUS outline
    conus.boundary.plot(ax=ax_c, edgecolor='#9aa0a6', linewidth=0.8, zorder=4)

    # Zoom to CONUS extent
    ax_c.set_xlim(xmin, xmax)
    ax_c.set_ylim(ymin, ymax)
    ax_c.set_xlabel('Longitude (CONUS)')
    ax_c.set_ylabel('Latitude (CONUS)')
    ax_c.grid(False)

    ax_g.set_title('Global', fontsize=12, pad=6)
    ax_c.set_title('CONUS', fontsize=12, pad=6)
    for _ax in (ax_g, ax_c):
        _ax.spines['top'].set_visible(False)
        _ax.spines['right'].set_visible(False)
        _ax.spines['left'].set_linewidth(0.6)
        _ax.spines['bottom'].set_linewidth(0.6)
    fig.tight_layout(h_pad=2.0)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    out_path_ = os.path.join('poster', 'training_data_world_map_no_rosetta.png')
    plot_training_data_map(out_path_, include_rosetta=False)
# ========================= EOF ====================================================================
