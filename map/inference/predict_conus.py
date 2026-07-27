"""Compatibility wrapper for :mod:`map.inference.predict_rasters`.

New release-facing code should invoke ``map.inference.predict_rasters``
directly. This module remains to avoid breaking notebooks and ad hoc helpers
that still import the older CONUS-named entry point.
"""

from map.inference.predict_rasters import (
    DEFAULT_MODEL_DIR,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SMAP_DIR,
    DEFAULT_STATIC_DIR,
    FIXED_FEATURES,
    NODATA_VALUE,
    GridSpec,
    ModelArtifacts,
    SMAP_FILENAME_RE,
    StaticRasterStack,
    build_feature_matrix,
    build_output_cube,
    build_parser,
    infer_output_dir,
    iter_smap_files,
    main,
    parse_date,
    predict_in_batches,
    read_theta,
    run_prediction,
    validate_feature_contract,
    write_prediction_raster,
)

__all__ = [
    "DEFAULT_MODEL_DIR",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_SMAP_DIR",
    "DEFAULT_STATIC_DIR",
    "FIXED_FEATURES",
    "GridSpec",
    "ModelArtifacts",
    "NODATA_VALUE",
    "SMAP_FILENAME_RE",
    "StaticRasterStack",
    "build_feature_matrix",
    "build_output_cube",
    "build_parser",
    "infer_output_dir",
    "iter_smap_files",
    "main",
    "parse_date",
    "predict_in_batches",
    "read_theta",
    "run_prediction",
    "validate_feature_contract",
    "write_prediction_raster",
]


if __name__ == "__main__":
    main()
