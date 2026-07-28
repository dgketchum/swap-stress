"""
Run the trained direct model over aligned EASE-Grid2 rasters.

This module loads a saved Random Forest model, its fitted imputer, and the
ordered feature list produced by ``train_direct.py``. It then assembles the
matching feature matrix from:

- daily SMAP L3 soil moisture GeoTIFFs (``theta``)
- pre-aligned static covariate rasters (band descriptions define feature names)
- fixed metadata features required by the model (``depth_cm``,
  ``rosetta_level``)

Usage:
    uv run python -m swapstress.inference.predict \
        --config /home/dgketchum/code/swap-stress/configs/predict_9km_global_pruned.toml
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

from swapstress.sources.depth import depth_to_rosetta_level

SMAP_FILENAME_RE = re.compile(r"^smap_(?:sm|l4(?:_sm)?)_(\d{8})\.tif$")
DEFAULT_MODEL_DIR = "/nas/soils/swapstress/models/direct_rf_9km_global_pruned"
DEFAULT_STATIC_DIR = "/nas/soils/swapstress/inference/global_features/rasters_ease2"
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
                missing_desc = [
                    i + 1 for i, desc in enumerate(descriptions) if not desc
                ]
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

    for path in sorted(smap_path.glob("smap_*.tif")):
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


def report_imputer_fill(feature_names: list[str], feature_matrix: np.ndarray) -> None:
    """Print how many cells the imputer will fill, per feature.

    The imputer substitutes a training-set value wherever a covariate is
    missing, which is silent at the raster level: a feature absent over a whole
    region still yields a prediction. Printing the per-feature counts makes a
    coverage gap in the static stack visible before it reaches the product.
    """
    nan_counts = np.isnan(feature_matrix).sum(axis=0)
    has_nans = np.flatnonzero(nan_counts)
    total_cells = feature_matrix.shape[0]
    print(
        f"Imputer fill report ({len(has_nans)} of {len(feature_names)} "
        f"features have NaNs):"
    )
    if not has_nans.size:
        print("  (none)")
        return
    for col_idx in has_nans[np.argsort(nan_counts[has_nans])[::-1]]:
        n = nan_counts[col_idx]
        print(
            f"  {feature_names[col_idx]:<45s}  {n:>9,} / {total_cells:,}  "
            f"({100 * n / total_cells:.1f}%)"
        )


def predict_in_batches(
    model_artifacts: ModelArtifacts,
    feature_matrix: np.ndarray,
    batch_size: int,
    quantiles: tuple[float, ...] | None = None,
) -> np.ndarray:
    """Apply the fitted imputer and RF model in chunks.

    Parameters
    ----------
    quantiles : tuple of float, optional
        If the model is a QRF, predict at these quantiles and return an
        (n_rows, len(quantiles)) array.  None → standard mean prediction.
    """
    n_rows = feature_matrix.shape[0]
    is_qrf = quantiles is not None and hasattr(model_artifacts.model, "predict")

    if is_qrf:
        predictions = np.empty((n_rows, len(quantiles)), dtype=np.float32)
    else:
        predictions = np.empty(n_rows, dtype=np.float32)

    for start in range(0, n_rows, batch_size):
        stop = min(start + batch_size, n_rows)
        chunk = feature_matrix[start:stop]
        transformed = model_artifacts.imputer.transform(chunk)

        if is_qrf:
            pred = model_artifacts.model.predict(
                transformed,
                quantiles=list(quantiles),
            )
            predictions[start:stop] = pred.astype(np.float32, copy=False)
        else:
            predictions[start:stop] = model_artifacts.model.predict(
                transformed,
            ).astype(np.float32, copy=False)

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


def validate_feature_contract(
    model_artifacts: ModelArtifacts,
) -> tuple[set[str], set[str]]:
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


def _predict_one_day(
    date,
    smap_path: Path,
    output_path: Path,
    model_artifacts: ModelArtifacts,
    static_stack: StaticRasterStack,
    depth_cm: float,
    rosetta_level: int,
    batch_size: int,
    overwrite: bool,
    write_linear: bool,
    quantiles: tuple[float, ...] | None,
    imputer_fill_report: bool = False,
) -> bool:
    """Predict suction for a single day. Returns True if a raster was written."""
    out_name = f"suction_{date.strftime('%Y%m%d')}.tif"
    out_path = output_path / out_name
    if out_path.exists() and not overwrite:
        return False

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
            rosetta_level=rosetta_level,
        )
        if imputer_fill_report:
            print(f"\n{date.date()}  valid theta px: {valid_idx.size:,}")
            report_imputer_fill(model_artifacts.feature_names, feature_matrix)
        predictions = predict_in_batches(
            model_artifacts=model_artifacts,
            feature_matrix=feature_matrix,
            batch_size=batch_size,
            quantiles=quantiles,
        )
        if quantiles is not None and predictions.ndim == 2:
            median_idx = list(quantiles).index(0.5) if 0.5 in quantiles else 0
            cube = build_output_cube(
                predictions=predictions[:, median_idx],
                valid_idx=valid_idx,
                shape=(static_stack.grid.height, static_stack.grid.width),
                write_linear=write_linear,
            )
            if len(quantiles) >= 2:
                iqr = predictions[:, -1] - predictions[:, 0]
                height, width = static_stack.grid.height, static_stack.grid.width
                iqr_grid = np.full(height * width, NODATA_VALUE, dtype=np.float32)
                iqr_grid[valid_idx] = iqr
                cube = np.concatenate(
                    [cube, iqr_grid.reshape(1, height, width)],
                    axis=0,
                )
        else:
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
        rosetta_level=rosetta_level,
        model_dir=model_artifacts.model_dir,
        source_path=smap_path,
    )
    print(f"WROTE {out_name} ({valid_idx.size}/{theta_flat.size} valid theta pixels)")
    return True


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
    config_dict: dict | None = None,
    quantiles: tuple[float, ...] | None = None,
    n_jobs: int = 1,
    imputer_fill_report: bool = False,
) -> None:
    """Run daily predictions over the requested SMAP date range."""
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
    print(f"Dates matched: {len(smap_files)} (n_jobs={n_jobs})")

    if n_jobs > 1:
        # Avoid oversubscription: limit RF's internal threading when
        # we parallelize across days using threads.
        if hasattr(model_artifacts.model, "n_jobs"):
            model_artifacts.model.n_jobs = 1

    common_kwargs = dict(
        output_path=output_path,
        model_artifacts=model_artifacts,
        static_stack=static_stack,
        depth_cm=depth_cm,
        rosetta_level=resolved_rosetta,
        batch_size=batch_size,
        overwrite=overwrite,
        write_linear=write_linear,
        quantiles=quantiles,
        imputer_fill_report=imputer_fill_report,
    )

    if n_jobs == 1:
        results = [
            _predict_one_day(date, smap_path, **common_kwargs)
            for date, smap_path in smap_files
        ]
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=10)(
            delayed(_predict_one_day)(date, smap_path, **common_kwargs)
            for date, smap_path in smap_files
        )

    n_written = sum(1 for r in results if r)

    # Write provenance artifact
    if config_dict is not None:
        from swapstress.config import write_provenance

        upstream_prov = model_artifacts.model_dir / "provenance.json"
        prov_path = write_provenance(
            output_dir=str(output_path),
            config=config_dict,
            run_type="predict",
            extras={
                "inputs": {
                    "n_smap_files": len(smap_files),
                    "n_static_rasters": len(list(Path(static_dir).glob("*_ease2.tif"))),
                    "n_model_features": len(model_artifacts.feature_names),
                },
                "outputs": {
                    "n_rasters_written": n_written,
                },
                "upstream": {
                    "model_provenance": str(upstream_prov)
                    if upstream_prov.exists()
                    else None,
                },
            },
        )
        print(f"Saved provenance to {prov_path}")


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    from swapstress.cli import add_common_args

    parser = argparse.ArgumentParser(
        prog="swapstress-predict",
        description="Stage 05: run the direct model over aligned EASE-Grid2 rasters",
    )
    add_common_args(parser)
    parser.add_argument(
        "--model-dir",
        default=None,
        help=f"Directory containing saved direct RF artifacts (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--static-dir",
        default=None,
        help=f"Directory containing *_ease2.tif static rasters (default: {DEFAULT_STATIC_DIR})",
    )
    parser.add_argument(
        "--smap-dir",
        default=None,
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
        default=None,
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
        default=None,
        help="Prediction batch size in valid pixels (default: 50000)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
        help="Overwrite existing output rasters",
    )
    parser.add_argument(
        "--write-linear",
        action="store_true",
        default=None,
        help="Write a second band with suction in cm H2O",
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="*",
        default=None,
        help="Quantiles for QRF prediction (e.g., 0.1 0.5 0.9). Adds IQR uncertainty band.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel workers for daily prediction (default: 1).",
    )
    parser.add_argument(
        "--imputer-fill-report",
        action="store_true",
        help="Print per-feature counts of cells the imputer fills, per day.",
    )
    return parser


def main(argv=None) -> None:
    """CLI entry point."""
    from swapstress.cli import report_paths, resolve

    config = resolve(build_parser(), argv)

    if config["dry_run"]:
        report_paths(
            "05 predict",
            {
                "model dir": config.get("model_dir", DEFAULT_MODEL_DIR),
                "static rasters": config.get("static_dir", DEFAULT_STATIC_DIR),
                "smap rasters": config.get("smap_dir", DEFAULT_SMAP_DIR),
            },
            {"predictions": config.get("output_dir") or "(inferred from model dir)"},
        )
        return

    run_prediction(
        model_dir=config.get("model_dir", DEFAULT_MODEL_DIR),
        static_dir=config.get("static_dir", DEFAULT_STATIC_DIR),
        smap_dir=config.get("smap_dir", DEFAULT_SMAP_DIR),
        output_dir=config.get("output_dir"),
        start_date=config.get("start_date"),
        end_date=config.get("end_date"),
        depth_cm=config.get("depth_cm", 5.0),
        rosetta_level=config.get("rosetta_level"),
        batch_size=config.get("batch_size", 50000),
        overwrite=config.get("overwrite", False),
        write_linear=config.get("write_linear", False),
        config_dict=config,
        quantiles=tuple(config["quantiles"]) if config.get("quantiles") else None,
        n_jobs=config.get("n_jobs", 1),
        imputer_fill_report=config.get("imputer_fill_report", False),
    )


if __name__ == "__main__":
    main()
