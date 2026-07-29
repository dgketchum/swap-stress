"""
Window global EASE-Grid2 M09 daily rasters to a named subset.

Stage 05 predicts on the full global grid; the released product is CONUS
(global training, CONUS application). This utility crops each daily Level 1
raster to the same CONUS window the SMAP download uses
(``features.smap_download.resolve_ease2_grid``), so gap-fill and packaging
run on a small fraction of the global pixel count and the deposit grid
matches the windowed SMAP record. All bands and their descriptions are
preserved.

Usage:
    swapstress-window --config configs/window_9km_conus.toml
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import rasterio
from rasterio.windows import Window

from swapstress.config import write_provenance
from swapstress.features.smap_download import resolve_ease2_grid
from swapstress.inference.gapfill import discover_source_rasters

DEFAULT_PREFIX = "suction"
DEFAULT_GRID_SCOPE = "conus"


def window_raster(
    src_path: Path,
    out_path: Path,
    row_slice: slice,
    col_slice: slice,
    transform,
) -> None:
    """Crop one raster to the window, preserving profile and band names."""
    window = Window.from_slices(row_slice, col_slice)
    with rasterio.open(src_path) as src:
        data = src.read(window=window)
        profile = src.profile.copy()
        descriptions = src.descriptions
    profile.update(height=data.shape[1], width=data.shape[2], transform=transform)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)
        for i, desc in enumerate(descriptions, start=1):
            if desc:
                dst.set_band_description(i, desc)


def run_window(
    source_dir: str,
    output_dir: str,
    start_date: str | None = None,
    end_date: str | None = None,
    overwrite: bool = False,
    prefix: str = DEFAULT_PREFIX,
    grid_scope: str = DEFAULT_GRID_SCOPE,
    config_dict: dict | None = None,
) -> None:
    """Window every daily raster in *source_dir* into *output_dir*."""
    start = datetime.strptime(start_date, "%Y%m%d").date() if start_date else None
    end = datetime.strptime(end_date, "%Y%m%d").date() if end_date else None

    row_slice, col_slice, transform = resolve_ease2_grid(grid_scope)
    rasters = discover_source_rasters(source_dir, prefix, start, end)
    if not rasters:
        raise FileNotFoundError(f"No {prefix}_YYYYMMDD.tif rasters under {source_dir}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_rows = row_slice.stop - row_slice.start
    n_cols = col_slice.stop - col_slice.start
    print(f"Windowing {len(rasters)} rasters to {grid_scope} ({n_rows} x {n_cols})")

    written = skipped = 0
    for d, src_path in rasters.items():
        out_path = out_dir / src_path.name
        if out_path.exists() and not overwrite:
            skipped += 1
            continue
        window_raster(src_path, out_path, row_slice, col_slice, transform)
        written += 1
        if written % 500 == 0:
            print(f"  {written} written (through {d.isoformat()})")

    print(f"Done: {written} written, {skipped} skipped existing")
    write_provenance(
        output_dir,
        config_dict or {},
        run_type="window",
        extras={
            "inputs": {"source rasters": str(source_dir)},
            "outputs": {"windowed rasters": str(output_dir)},
        },
        filename="provenance_window.json",
    )


def build_parser() -> argparse.ArgumentParser:
    from swapstress.cli import add_common_args

    parser = argparse.ArgumentParser(
        prog="swapstress-window",
        description="Window global EASE-Grid2 daily rasters to the CONUS subset",
    )
    add_common_args(parser)
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Directory of global <prefix>_YYYYMMDD.tif rasters",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for windowed rasters",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help=f"Filename prefix before _YYYYMMDD.tif (default: {DEFAULT_PREFIX})",
    )
    parser.add_argument(
        "--grid-scope",
        default=None,
        help=f"Named window from resolve_ease2_grid (default: {DEFAULT_GRID_SCOPE})",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="First date to window (YYYYMMDD)",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Last date to window (YYYYMMDD)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
        help="Rewrite windowed rasters that already exist",
    )
    return parser


def main(argv=None) -> None:
    from swapstress.cli import report_paths, resolve

    config = resolve(build_parser(), argv, required=["source_dir", "output_dir"])

    if config.get("dry_run"):
        report_paths(
            "window",
            {"global rasters": config["source_dir"]},
            {"windowed rasters": config["output_dir"]},
        )
        return

    run_window(
        source_dir=config["source_dir"],
        output_dir=config["output_dir"],
        start_date=config.get("start_date"),
        end_date=config.get("end_date"),
        overwrite=config.get("overwrite", False),
        prefix=config.get("prefix", DEFAULT_PREFIX),
        grid_scope=config.get("grid_scope", DEFAULT_GRID_SCOPE),
        config_dict=config,
    )


if __name__ == "__main__":
    main()
