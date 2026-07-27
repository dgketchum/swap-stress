"""Compatibility wrapper for :mod:`map.inference.gapfill_rasters`.

New release-facing code should invoke ``map.inference.gapfill_rasters``
directly. This module remains to avoid breaking notebooks and references that
still import the older CONUS-named entry point.
"""

from map.inference.gapfill_rasters import (
    DEFAULT_BAND_DESCRIPTION,
    DEFAULT_PREFIX,
    DEFAULT_SOURCE_DIR,
    build_parser,
    discover_source_rasters,
    load_raster,
    main,
    run_gapfill,
    write_raster,
)

__all__ = [
    "DEFAULT_BAND_DESCRIPTION",
    "DEFAULT_PREFIX",
    "DEFAULT_SOURCE_DIR",
    "build_parser",
    "discover_source_rasters",
    "load_raster",
    "main",
    "run_gapfill",
    "write_raster",
]


if __name__ == "__main__":
    main()
