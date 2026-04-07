"""
Run the trained direct model over aligned CONUS 9 km rasters.

This module loads a saved Random Forest model, its fitted imputer, and the
ordered feature list produced by ``train_direct.py``. It then assembles the
matching feature matrix from:

- daily SMAP L3 soil moisture GeoTIFFs (``theta``)
- pre-aligned static covariate rasters (band descriptions define feature names)
- fixed metadata features required by the model (``depth_cm``,
  ``rosetta_level``)

Usage:
    uv run python -m map.inference.predict_conus \
        --model-dir /nas/soils/swapstress/models/direct_rf_9km_conus \
        --start-date 20200101 --end-date 20200131
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import rasterio
from rasterio.transform import Affine

from retention_curve.depth_utils import depth_to_rosetta_level

SMAP_FILENAME_RE = re.compile(r"^smap_sm_(\d{8})\.tif$")
DEFAULT_MODEL_DIR = "/nas/soils/swapstress/models/direct_rf_9km_conus"
DEFAULT_STATIC_DIR = "/nas/soils/swapstress/inference/conus_features"
DEFAULT_SMAP_DIR = "/nas/soils/smap/SPL3SMP_E/daily_tif"
DEFAULT_OUTPUT_ROOT = "/nas/soils/swapstress/inference/predictions"
FIXED_FEATURES = {"depth_cm", "rosetta_level"}
NODATA_VALUE = -9999.0


@dataclass
class ModelArtifacts:
    """Saved model assets required for prediction."""

    model: object
    imputer: object
    feature_names: list[str]
    model_dir: Path

    @classmethod
    def load(cls, model_dir: str | Path) -> "ModelArtifacts":
        model_path = Path(model_dir).expanduser().resolve()
        with open(model_path / "direct_rf_features.json", encoding="utf-8") as f:
            feature_names = json.load(f)

        return cls(
            model=joblib.load(model_path / "direct_rf_model.joblib"),
            imputer=joblib.load(model_path / "direct_rf_imputer.joblib"),
            feature_names=feature_names,
            model_dir=model_path,
        )


@dataclass
class GridSpec:
    """Raster grid metadata shared across aligned inputs."""

    width: int
    height: int
    crs: object
    transform: Affine


@dataclass
class StaticRasterStack:
    """Flattened static covariates keyed by band description."""

    data: np.ndarray
    feature_to_index: dict[str, int]
    grid: GridSpec

    @classmethod
    def load(
        cls,
        static_dir: str | Path,
        required_features: Iterable[str],
    ) -> "StaticRasterStack":
        static_path = Path(static_dir).expanduser().resolve()
        raster_paths = sorted(static_path.glob("*_ease2.tif"))
        if not raster_paths:
            raise FileNotFoundError(f"No *_ease2.tif files found in {static_path}")

        required = set(required_features)
        bands: list[np.ndarray] = []
        feature_to_index: dict[str, int] = {}
        grid: GridSpec | None = None

        for raster_path in raster_paths:
            with rasterio.open(raster_path) as src:
                current_grid = GridSpec(
                    width=src.width,
                    height=src.height,
                    crs=src.crs,
                    transform=src.transform,
                )
                if grid is None:
                    grid = current_grid
                elif current_grid != grid:
                    raise ValueError(
                        f"Static raster grid mismatch for {raster_path}: "
                        f"{current_grid} != {grid}"
                    )

                descriptions = list(src.descriptions)
                missing_desc = [i + 1 for i, desc in enumerate(descriptions) if not desc]
                if missing_desc:
                    raise ValueError(
                        f"Missing band descriptions in {raster_path}: bands {missing_desc}"
                    )

                data = src.read().astype(np.float32, copy=False)
                if src.nodata is not None and not np.isnan(src.nodata):
                    data[data == src.nodata] = np.nan

                for band_idx, name in enumerate(descriptions):
                    if name not in required:
                        continue
                    if name in feature_to_index:
                        raise ValueError(f"Duplicate static feature band: {name}")
                    feature_to_index[name] = len(bands)
                    bands.append(data[band_idx].reshape(-1))

        if grid is None:
            raise ValueError(f"Could not derive raster grid from {static_path}")

        missing = sorted(required - set(feature_to_index))
        if missing:
            raise ValueError(
                "Required static features missing from raster stack: "
                + ", ".join(missing)
            )

        if bands:
            matrix = np.vstack(bands)
        else:
            matrix = np.empty((0, grid.height * grid.width), dtype=np.float32)

        return cls(data=matrix, feature_to_index=feature_to_index, grid=grid)


def parse_date(value: str | None) -> datetime | None:
    """Parse YYYYMMDD dates for CLI filtering."""
    if value is None:
        return None
    return datetime.strptime(value, "%Y%m%d")


def infer_output_dir(model_dir: Path, output_dir: str | None) -> Path:
    """Pick a stable default output location based on the model directory name."""
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    return Path(DEFAULT_OUTPUT_ROOT).expanduser().resolve() / model_dir.name


def iter_smap_files(
    smap_dir: str | Path,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> list[tuple[datetime, Path]]:
    """Discover SMAP daily rasters and filter them by date range."""
    smap_path = Path(smap_dir).expanduser().resolve()
    matches: list[tuple[datetime, Path]] = []

    for path in sorted(smap_path.glob("smap_sm_*.tif")):
        match = SMAP_FILENAME_RE.match(path.name)
        if not match:
            continue
        date = datetime.strptime(match.group(1), "%Y%m%d")
        if start_date and date < start_date:
            continue
        if end_date and date > end_date:
            continue
        matches.append((date, path))

    return matches


def read_theta(path: Path, expected_grid: GridSpec) -> tuple[np.ndarray, dict]:
    """Load a daily SMAP raster and verify it matches the static stack grid."""
    with rasterio.open(path) as src:
        current_grid = GridSpec(
            width=src.width,
            height=src.height,
            crs=src.crs,
            transform=src.transform,
        )
        if current_grid != expected_grid:
            raise ValueError(
                f"SMAP grid mismatch for {path}: {current_grid} != {expected_grid}"
            )

        theta = src.read(1).astype(np.float32, copy=False)
        if src.nodata is not None and not np.isnan(src.nodata):
            theta[theta == src.nodata] = np.nan

        profile = src.profile.copy()

    return theta.reshape(-1), profile


def build_feature_matrix(
    feature_names: list[str],
    static_stack: StaticRasterStack,
    theta_flat: np.ndarray,
    valid_idx: np.ndarray,
    depth_cm: float,
    rosetta_level: int,
) -> np.ndarray:
    """Assemble the model input matrix in saved feature order."""
    n_rows = int(valid_idx.size)
    n_cols = len(feature_names)
    matrix = np.empty((n_rows, n_cols), dtype=np.float32)

    for col_idx, feature in enumerate(feature_names):
        if feature == "theta":
            matrix[:, col_idx] = theta_flat[valid_idx]
        elif feature == "depth_cm":
            matrix[:, col_idx] = depth_cm
        elif feature == "rosetta_level":
            matrix[:, col_idx] = rosetta_level
        else:
            static_idx = static_stack.feature_to_index[feature]
            matrix[:, col_idx] = static_stack.data[static_idx, valid_idx]

    return matrix


def predict_in_batches(
    model_artifacts: ModelArtifacts,
    feature_matrix: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Apply the fitted imputer and RF model in chunks."""
    n_rows = feature_matrix.shape[0]
    predictions = np.empty(n_rows, dtype=np.float32)

    for start in range(0, n_rows, batch_size):
        stop = min(start + batch_size, n_rows)
        chunk = feature_matrix[start:stop]
        transformed = model_artifacts.imputer.transform(chunk)
        predictions[start:stop] = model_artifacts.model.predict(transformed).astype(
            np.float32,
            copy=False,
        )

    return predictions


def build_output_cube(
    predictions: np.ndarray,
    valid_idx: np.ndarray,
    shape: tuple[int, int],
    write_linear: bool,
) -> np.ndarray:
    """Expand flat valid-pixel predictions back to raster form."""
    height, width = shape
    log10_grid = np.full(height * width, NODATA_VALUE, dtype=np.float32)
    log10_grid[valid_idx] = predictions

    bands = [log10_grid.reshape(height, width)]
    if write_linear:
        linear_grid = np.full(height * width, NODATA_VALUE, dtype=np.float32)
        linear_grid[valid_idx] = np.power(10.0, predictions).astype(
            np.float32,
            copy=False,
        )
        bands.append(linear_grid.reshape(height, width))

    return np.stack(bands, axis=0)


def write_prediction_raster(
    out_path: Path,
    profile: dict,
    data: np.ndarray,
    write_linear: bool,
    depth_cm: float,
    rosetta_level: int,
    model_dir: Path,
    source_path: Path,
) -> None:
    """Write prediction raster(s) with useful metadata."""
    profile.update(
        driver="GTiff",
        dtype="float32",
        count=data.shape[0],
        nodata=NODATA_VALUE,
        compress="zstd",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)
        dst.set_band_description(1, "log10_suction_cm")
        if write_linear:
            dst.set_band_description(2, "suction_cm")

        dst.update_tags(
            model_dir=str(model_dir),
            theta_source=str(source_path),
            depth_cm=str(depth_cm),
            rosetta_level=str(rosetta_level),
        )


def validate_feature_contract(model_artifacts: ModelArtifacts) -> tuple[set[str], set[str]]:
    """Split saved features into raster-derived and fixed categories."""
    feature_set = set(model_artifacts.feature_names)
    if "theta" not in feature_set:
        raise ValueError("Saved feature list is missing required feature 'theta'")

    static_features = feature_set - {"theta"} - FIXED_FEATURES
    fixed_features = feature_set & FIXED_FEATURES
    unsupported_fixed = fixed_features - FIXED_FEATURES
    if unsupported_fixed:
        raise ValueError(
            "Unsupported fixed features in model artifact: "
            + ", ".join(sorted(unsupported_fixed))
        )

    return static_features, fixed_features


def run_prediction(
    model_dir: str,
    static_dir: str,
    smap_dir: str,
    output_dir: str | None,
    start_date: str | None,
    end_date: str | None,
    depth_cm: float,
    rosetta_level: int | None,
    batch_size: int,
    overwrite: bool,
    write_linear: bool,
) -> None:
    """Run daily CONUS predictions over the requested SMAP date range."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    model_artifacts = ModelArtifacts.load(model_dir)
    static_features, _ = validate_feature_contract(model_artifacts)
    static_stack = StaticRasterStack.load(static_dir, static_features)

    start = parse_date(start_date)
    end = parse_date(end_date)
    if start and end and start > end:
        raise ValueError("start_date must be <= end_date")

    smap_files = iter_smap_files(smap_dir, start_date=start, end_date=end)
    if not smap_files:
        raise FileNotFoundError("No SMAP rasters matched the requested date range")

    output_path = infer_output_dir(model_artifacts.model_dir, output_dir)
    resolved_rosetta = rosetta_level
    if resolved_rosetta is None:
        resolved_rosetta = depth_to_rosetta_level(depth_cm)
    if resolved_rosetta is None:
        raise ValueError(f"Could not infer rosetta_level from depth_cm={depth_cm}")

    print(f"Model dir: {model_artifacts.model_dir}")
    print(f"Static dir: {Path(static_dir).expanduser().resolve()}")
    print(f"Output dir: {output_path}")
    print(
        f"Feature contract: {len(model_artifacts.feature_names)} total features, "
        f"{len(static_features)} raster-derived, depth_cm={depth_cm}, "
        f"rosetta_level={resolved_rosetta}"
    )
    print(f"Dates matched: {len(smap_files)}")

    for date, smap_path in smap_files:
        out_name = f"suction_{date.strftime('%Y%m%d')}.tif"
        out_path = output_path / out_name
        if out_path.exists() and not overwrite:
            print(f"SKIP {out_name} (exists)")
            continue

        theta_flat, profile = read_theta(smap_path, static_stack.grid)
        valid_mask = np.isfinite(theta_flat)
        valid_idx = np.flatnonzero(valid_mask)

        if valid_idx.size == 0:
            cube = build_output_cube(
                predictions=np.empty(0, dtype=np.float32),
                valid_idx=valid_idx,
                shape=(static_stack.grid.height, static_stack.grid.width),
                write_linear=write_linear,
            )
        else:
            feature_matrix = build_feature_matrix(
                feature_names=model_artifacts.feature_names,
                static_stack=static_stack,
                theta_flat=theta_flat,
                valid_idx=valid_idx,
                depth_cm=depth_cm,
                rosetta_level=resolved_rosetta,
            )
            predictions = predict_in_batches(
                model_artifacts=model_artifacts,
                feature_matrix=feature_matrix,
                batch_size=batch_size,
            )
            cube = build_output_cube(
                predictions=predictions,
                valid_idx=valid_idx,
                shape=(static_stack.grid.height, static_stack.grid.width),
                write_linear=write_linear,
            )

        write_prediction_raster(
            out_path=out_path,
            profile=profile,
            data=cube,
            write_linear=write_linear,
            depth_cm=depth_cm,
            rosetta_level=resolved_rosetta,
            model_dir=model_artifacts.model_dir,
            source_path=smap_path,
        )
        print(
            f"WROTE {out_name} "
            f"({valid_idx.size}/{theta_flat.size} valid theta pixels)"
        )


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run the direct RF model over CONUS 9 km rasters",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help=f"Directory containing saved direct RF artifacts (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--static-dir",
        default=DEFAULT_STATIC_DIR,
        help=f"Directory containing *_ease2.tif static rasters (default: {DEFAULT_STATIC_DIR})",
    )
    parser.add_argument(
        "--smap-dir",
        default=DEFAULT_SMAP_DIR,
        help=f"Directory containing daily SMAP GeoTIFFs (default: {DEFAULT_SMAP_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for prediction rasters (default: inferred from model dir)",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Inclusive start date in YYYYMMDD",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Inclusive end date in YYYYMMDD",
    )
    parser.add_argument(
        "--depth-cm",
        type=float,
        default=5.0,
        help="Depth to inject as a model feature (default: 5.0)",
    )
    parser.add_argument(
        "--rosetta-level",
        type=int,
        default=None,
        help="Optional explicit rosetta level; otherwise inferred from --depth-cm",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Prediction batch size in valid pixels (default: 50000)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output rasters",
    )
    parser.add_argument(
        "--write-linear",
        action="store_true",
        help="Write a second band with suction in cm H2O",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()
    run_prediction(
        model_dir=args.model_dir,
        static_dir=args.static_dir,
        smap_dir=args.smap_dir,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        depth_cm=args.depth_cm,
        rosetta_level=args.rosetta_level,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        write_linear=args.write_linear,
    )


if __name__ == "__main__":
    main()
