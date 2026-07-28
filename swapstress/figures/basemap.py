"""Cartographic assets shared by the descriptor's map figures.

The map figures used to read state and lake outlines from ``/tmp``, which does
not survive a reboot -- so a clean machine could not re-render them, and the
referenced vintages (``cb_2022``, ``ne_10m``) are not the ones actually
archived. These resolve against the same boundaries root the source registry
uses for the MGRS index, with the same "beside the soils tree" convention.

Grid rasters are on EASE-Grid2, so drawing them under WGS84 vector boundaries
means reprojecting *pixel corners* rather than centres -- ``pcolormesh`` with
``shading="flat"`` wants an (h+1, w+1) mesh, and using centres would shift the
image half a cell.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

# Matches DataPaths.boundaries_root: the boundaries tree sits beside the soils
# tree, not inside it, and subpaths are relative to its parent.
DEFAULT_BOUNDARIES_ROOT = "/nas"
STATES_SUBPATH = "boundaries/states/us_state_20m/cb_2016_us_state_20m.shp"
LAKES_SUBPATH = "boundaries/natural_earth/ne_110m_lakes.shp"

# Territories that are not CONUS and would blow out the map extent.
NON_CONUS = {"HI", "AK", "AS", "GU", "MP", "PR", "VI"}
CONUS_LON = (-127.0, -65.0)
CONUS_LAT = (24.0, 50.0)


def boundaries_root(root: Optional[str] = None) -> str:
    """Root holding the cartographic boundaries.

    Explicit argument wins, then ``SWAPSTRESS_BOUNDARIES_ROOT``, then the
    default -- so a figure can be re-rendered off a different mount without
    editing code.
    """
    return os.path.expanduser(
        root or os.environ.get("SWAPSTRESS_BOUNDARIES_ROOT") or DEFAULT_BOUNDARIES_ROOT
    )


def states_shapefile(root: Optional[str] = None) -> str:
    return os.path.join(boundaries_root(root), STATES_SUBPATH)


def lakes_shapefile(root: Optional[str] = None) -> str:
    return os.path.join(boundaries_root(root), LAKES_SUBPATH)


def load_conus_states(root: Optional[str] = None, crs=4326):
    """CONUS state outlines, territories dropped, reprojected to *crs*."""
    import geopandas as gpd

    path = states_shapefile(root)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"State boundaries not found at {path}. Set "
            f"SWAPSTRESS_BOUNDARIES_ROOT to the root holding '{STATES_SUBPATH}'."
        )
    states = gpd.read_file(path).to_crs(crs)
    column = "STUSPS" if "STUSPS" in states.columns else "STATE_ABBR"
    return states[~states[column].isin(NON_CONUS)]


def pixel_corner_lonlat(transform, shape: Tuple[int, int], crs) -> Tuple:
    """Longitude/latitude mesh of pixel *corners* for ``pcolormesh``.

    Corners, not centres: a flat-shaded mesh needs one more node than cells in
    each direction, and centres would offset the image by half a pixel.
    """
    from pyproj import Transformer

    height, width = shape
    x = transform.c + np.arange(width + 1) * transform.a
    y = transform.f + np.arange(height + 1) * transform.e
    x_grid, y_grid = np.meshgrid(x, y)
    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return to_wgs84.transform(x_grid, y_grid)


def style_conus_axis(ax, states=None, root: Optional[str] = None) -> None:
    """Apply the shared CONUS map framing to *ax*."""
    if states is None:
        states = load_conus_states(root)
    states.boundary.plot(ax=ax, edgecolor="#999", linewidth=0.35, zorder=3)
    ax.set_xlim(*CONUS_LON)
    ax.set_ylim(*CONUS_LAT)
    ax.set_aspect("auto")
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
