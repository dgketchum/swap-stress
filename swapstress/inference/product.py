"""Stage 07: swapstress-package -- write the released product.

The model predicts one thing, ``log10_suction_cm``. This stage turns a directory
of those daily rasters into the released files: the same quantity in the three
representations Table 2 documents, plus the uncertainty band when the model was
run with quantiles and, at Level 2, a per-pixel flag saying which values were
gap-filled.

Two sign conventions coexist by design, each matching its own community's norm.
Suction head is positive and increases as soil dries; matric potential is
negative and approaches zero as soil wets. Every band carries its convention in
its own metadata rather than leaving a reuser to infer it from the values.

The conversion is exact, not approximate. Because the target is a base-10
logarithm, going to MPa is an additive shift in log space, so no reported metric
moves; :mod:`swapstress.units` holds the constant and the argument.

Levels
------
**Level 1** is the direct model output: valid wherever a SMAP retrieval was
available that day, gaps left as gaps. **Level 2** is the gap-filled series, and
carries the extra ``gapfill_flag`` band marking which pixels were interpolated
along the time axis rather than retrieved. Deriving that flag needs both levels,
which is why ``--level 2`` also wants ``--level1-dir``.

Containers
----------
:func:`build_bands` computes the band stack and returns it with its metadata,
without touching disk. :func:`write_geotiff` is one container for that stack;
the NetCDF time-stacked alternative is still an open release decision, and can
be added against the same band list without changing what the bands mean.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
import rasterio

from swapstress.units import MPA_TO_CM

# Matches predict's and gapfill's output naming, <prefix>_YYYYMMDD.tif.
DATE_PATTERN = re.compile(r"_(\d{8})\.tif$")

NODATA_VALUE = -9999.0
CONVENTIONS = "CF-1.10"

# EASE-Grid2 global M09. Named here so the released metadata states the grid
# rather than leaving it implicit in the affine transform.
GRID_NAME = "EASE-Grid 2.0 Global (M09)"
GRID_EPSG = 6933

# CF standard names are pinned at release time, not guessed here: several
# plausible candidates exist for soil water potential and picking one in code
# would bake a decision that belongs to the release. Fill this in (band name ->
# standard name) via --standard-name, or by editing it when the choice is made;
# bands with no entry simply carry no standard_name attribute.
STANDARD_NAMES: dict[str, str] = {}


@dataclass(frozen=True)
class BandSpec:
    """One released band: what it holds, in what units, with which sign."""

    name: str
    units: str
    long_name: str
    sign: str  # 'positive', 'negative', or 'none'
    comment: str = ""
    flag_values: str = ""  # set for flag bands, per CF

    def attributes(self, standard_names: Optional[dict] = None) -> dict:
        """CF-style attributes for this band."""
        attrs = {
            "long_name": self.long_name,
            "units": self.units,
            "_FillValue": str(NODATA_VALUE),
            "grid_mapping": "crs",
        }
        if self.sign != "none":
            attrs["sign_convention"] = self.sign
        if self.comment:
            attrs["comment"] = self.comment
        if self.flag_values:
            attrs["flag_values"] = self.flag_values
        name = (standard_names or STANDARD_NAMES).get(self.name)
        if name:
            attrs["standard_name"] = name
        return attrs


LOG10_SUCTION = BandSpec(
    name="log10_suction_cm",
    units="log10(cm)",
    long_name="Base-10 logarithm of soil water suction head",
    sign="positive",
    comment=(
        "The model's native target. Roughly 0-6 over the range of soils; "
        "field capacity is about 2.5 and the permanent wilting point about 4.18."
    ),
)

MATRIC_POTENTIAL = BandSpec(
    name="matric_potential_MPa",
    units="MPa",
    long_name="Soil matric potential",
    sign="negative",
    comment=(
        "Negative by convention: wet soil is near -0.01 MPa and the permanent "
        "wilting point is -1.5 MPa. Exactly -10**log10_suction_cm / 10197.16, "
        "an additive shift in log space, so no error metric changes with the "
        "choice of unit."
    ),
)

SUCTION = BandSpec(
    name="suction_cm",
    units="cm",
    long_name="Soil water suction head",
    sign="positive",
    comment="Linear convenience band; exactly 10**log10_suction_cm.",
)

UNCERTAINTY = BandSpec(
    name="uncertainty",
    units="log10(cm)",
    long_name="Quantile-regression-forest prediction interval width",
    sign="none",
    comment=(
        "Width of the QRF quantile interval in log10 units, identical whether "
        "read as log10(cm) or log10(|MPa|). Present only when the model was run "
        "with quantiles."
    ),
)

GAPFILL_FLAG = BandSpec(
    name="gapfill_flag",
    units="1",
    long_name="Gap-fill indicator",
    sign="none",
    comment=(
        "1 where the value was interpolated along the time axis because no "
        "retrieval was available, 0 where it comes from a same-day retrieval. "
        "Level 2 only."
    ),
    flag_values="0: retrieved, 1: gap-filled",
)


@dataclass
class ProductBands:
    """A computed band stack and the metadata describing it."""

    specs: List[BandSpec]
    data: np.ndarray  # (band, row, col), float32, NODATA_VALUE where invalid
    valid: np.ndarray = field(repr=False)  # (row, col) bool

    def index(self, name: str) -> int:
        for i, spec in enumerate(self.specs):
            if spec.name == name:
                return i
        raise KeyError(f"No band named '{name}'. Have: {[s.name for s in self.specs]}")

    def band(self, name: str) -> np.ndarray:
        return self.data[self.index(name)]


def build_bands(
    log10_suction_cm: np.ndarray,
    uncertainty: Optional[np.ndarray] = None,
    gapfill_flag: Optional[np.ndarray] = None,
) -> ProductBands:
    """Derive the released bands from one day of model output.

    Pure computation -- no I/O, no container assumptions. Invalid pixels are
    carried through as NODATA rather than being filled: a gap in the retrieval
    is information, and the ``gapfill_flag`` band is how a filled value is
    distinguished from a retrieved one.
    """
    log10 = np.asarray(log10_suction_cm, dtype=np.float32)
    valid = np.isfinite(log10) & (log10 != NODATA_VALUE)

    def empty_like():
        return np.full(log10.shape, NODATA_VALUE, dtype=np.float32)

    # Exponentiate only where valid: 10 ** -9999 would silently become 0.0 and
    # a NODATA pixel would read as perfectly wet soil.
    suction = empty_like()
    suction[valid] = np.power(10.0, log10[valid].astype(np.float64)).astype(np.float32)

    mpa = empty_like()
    mpa[valid] = (-suction[valid].astype(np.float64) / MPA_TO_CM).astype(np.float32)

    specs = [LOG10_SUCTION, MATRIC_POTENTIAL, SUCTION]
    bands = [np.where(valid, log10, NODATA_VALUE).astype(np.float32), mpa, suction]

    if uncertainty is not None:
        unc = np.asarray(uncertainty, dtype=np.float32)
        specs.append(UNCERTAINTY)
        bands.append(np.where(valid, unc, NODATA_VALUE).astype(np.float32))

    if gapfill_flag is not None:
        flag = np.asarray(gapfill_flag, dtype=np.float32)
        specs.append(GAPFILL_FLAG)
        bands.append(np.where(valid, flag, NODATA_VALUE).astype(np.float32))

    return ProductBands(specs=specs, data=np.stack(bands, axis=0), valid=valid)


def file_attributes(
    level: int,
    date_str: str,
    provenance: Optional[str],
    source: Optional[str],
) -> dict:
    """File-level CF attributes, including the link back to the run record."""
    attrs = {
        "Conventions": CONVENTIONS,
        "title": f"SWAP-Stress soil water potential, Level {level}",
        "product_level": str(level),
        "date": date_str,
        "grid": GRID_NAME,
        "grid_epsg": str(GRID_EPSG),
        "history": (
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} "
            "swapstress-package"
        ),
    }
    if source:
        attrs["source"] = source
    if provenance:
        attrs["provenance"] = provenance
    return attrs


def write_geotiff(
    out_path: Path,
    profile: dict,
    bands: ProductBands,
    level: int,
    date_str: str,
    provenance: Optional[str] = None,
    source: Optional[str] = None,
    standard_names: Optional[dict] = None,
) -> Path:
    """Write *bands* as a multi-band GeoTIFF.

    A GeoTIFF carries one dtype for every band, so the flag band rides along as
    float32 0.0/1.0 rather than as the uint8 a NetCDF container would use. The
    band spec is unchanged either way; only the storage differs.
    """
    out_profile = dict(profile)
    out_profile.update(
        driver="GTiff",
        dtype="float32",
        count=bands.data.shape[0],
        nodata=NODATA_VALUE,
        compress="zstd",
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(bands.data)
        dst.update_tags(**file_attributes(level, date_str, provenance, source))
        for i, spec in enumerate(bands.specs, start=1):
            dst.set_band_description(i, spec.name)
            dst.update_tags(i, **spec.attributes(standard_names))
    return out_path


def read_source_raster(path: Path) -> tuple[np.ndarray, Optional[np.ndarray], dict]:
    """Read a predict/gapfill raster into (log10, uncertainty, profile).

    Band 1 is always ``log10_suction_cm``. A band literally named
    ``uncertainty`` is carried through; the linear ``suction_cm`` band that
    predict can also write is ignored, because this stage recomputes it.
    """
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        descriptions = list(src.descriptions)
        log10 = src.read(1).astype(np.float32)
        if src.nodata is not None and not np.isnan(src.nodata):
            log10[log10 == src.nodata] = NODATA_VALUE

        uncertainty = None
        if "uncertainty" in descriptions:
            uncertainty = src.read(descriptions.index("uncertainty") + 1).astype(
                np.float32
            )
    return log10, uncertainty, profile


def derive_gapfill_flag(level2: np.ndarray, level1_path: Optional[Path]) -> np.ndarray:
    """Mark pixels that Level 2 has and Level 1 does not.

    The gapfill stage records ``is_gap_filled`` per file, which says whether a
    whole day was interpolated, not which pixels were. Comparing the two levels
    is what recovers the per-pixel answer.
    """
    filled = np.isfinite(level2) & (level2 != NODATA_VALUE)
    if level1_path is None or not level1_path.exists():
        # Without the Level 1 day there is no way to tell a retrieved pixel from
        # an interpolated one, and guessing would put a wrong flag in a released
        # file. Say so instead.
        raise FileNotFoundError(
            f"Level 2 packaging needs the matching Level 1 raster to derive "
            f"gapfill_flag; {level1_path} is missing. Pass --level1-dir, or "
            f"package at --level 1."
        )
    with rasterio.open(level1_path) as src:
        retrieved = src.read(1).astype(np.float32)
        nodata = src.nodata
    if nodata is not None and not np.isnan(nodata):
        retrieved_valid = np.isfinite(retrieved) & (retrieved != nodata)
    else:
        retrieved_valid = np.isfinite(retrieved)
    return (filled & ~retrieved_valid).astype(np.float32)


def find_rasters(source_dir: Path, prefix: str) -> List[Path]:
    """Daily rasters in date order."""
    return sorted(source_dir.glob(f"{prefix}_*.tif"))


def date_from_path(path: Path) -> str:
    match = DATE_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Cannot read a YYYYMMDD date from {path.name}")
    return match.group(1)


def package_release(
    source_dir: str,
    output_dir: str,
    level: int,
    level1_dir: Optional[str] = None,
    prefix: str = "suction",
    out_prefix: str = "swapstress_psi",
    provenance: Optional[str] = None,
    standard_names: Optional[dict] = None,
    overwrite: bool = False,
    writer: Callable[..., Path] = write_geotiff,
) -> List[Path]:
    """Convert a directory of daily model rasters into released files."""
    source = Path(source_dir)
    out_root = Path(output_dir)
    level1 = Path(level1_dir) if level1_dir else None

    rasters = find_rasters(source, prefix)
    if not rasters:
        raise FileNotFoundError(f"No {prefix}_*.tif rasters under {source}")
    print(f"{len(rasters)} daily rasters in {source}")

    written: List[Path] = []
    for path in rasters:
        date_str = date_from_path(path)
        out_path = out_root / f"{out_prefix}_L{level}_{date_str}.tif"
        if out_path.exists() and not overwrite:
            print(f"  exists, skipping: {out_path.name}")
            continue

        log10, uncertainty, profile = read_source_raster(path)
        flag = None
        if level == 2:
            flag = derive_gapfill_flag(log10, (level1 / path.name) if level1 else None)

        bands = build_bands(log10, uncertainty=uncertainty, gapfill_flag=flag)
        writer(
            out_path=out_path,
            profile=profile,
            bands=bands,
            level=level,
            date_str=date_str,
            provenance=provenance,
            source=str(path),
            standard_names=standard_names,
        )
        written.append(out_path)
        summary = f"  wrote {out_path.name}  {len(bands.specs)} bands"
        if flag is not None:
            summary += f"  {int(flag.sum()):,} gap-filled px"
        print(summary)

    return written


def build_parser() -> argparse.ArgumentParser:
    from swapstress.cli import add_common_args

    parser = argparse.ArgumentParser(
        prog="swapstress-package",
        description="Stage 07: write the released product from daily model rasters.",
    )
    add_common_args(parser)
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Directory of <prefix>_YYYYMMDD.tif model rasters to package.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the released files.",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=None,
        choices=[1, 2],
        help="1 = direct model output, 2 = gap-filled (adds gapfill_flag).",
    )
    parser.add_argument(
        "--level1-dir",
        default=None,
        help="Level 1 rasters, needed at --level 2 to derive gapfill_flag.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Input filename prefix before _YYYYMMDD.tif (default: suction).",
    )
    parser.add_argument(
        "--out-prefix",
        default=None,
        help="Released filename prefix (default: swapstress_psi).",
    )
    parser.add_argument(
        "--standard-name",
        action="append",
        default=None,
        metavar="BAND=NAME",
        help="Pin a CF standard_name for a band, e.g. "
        "matric_potential_MPa=soil_water_potential. Repeatable.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite files that already exist.",
    )
    return parser


def parse_standard_names(entries: Optional[Sequence[str]]) -> dict:
    names = dict(STANDARD_NAMES)
    for entry in entries or []:
        if "=" not in entry:
            raise SystemExit(f"--standard-name expects BAND=NAME, got '{entry}'")
        band, name = entry.split("=", 1)
        names[band.strip()] = name.strip()
    return names


def main(argv=None) -> None:
    from swapstress.cli import report_paths, resolve, stage_provenance

    config = resolve(build_parser(), argv, required=["source_dir", "output_dir"])
    config.setdefault("level", 1)
    config.setdefault("prefix", "suction")
    config.setdefault("out_prefix", "swapstress_psi")

    if config["dry_run"]:
        inputs = {"model rasters": config["source_dir"]}
        if config["level"] == 2:
            inputs["level 1 rasters"] = config.get("level1_dir")
        report_paths("07 package", inputs, {"released files": config["output_dir"]})
        return

    written = package_release(
        source_dir=config["source_dir"],
        output_dir=config["output_dir"],
        level=config["level"],
        level1_dir=config.get("level1_dir"),
        prefix=config["prefix"],
        out_prefix=config["out_prefix"],
        provenance=os.path.join(config["output_dir"], "provenance_package.json"),
        standard_names=parse_standard_names(config.get("standard_name")),
        overwrite=config.get("overwrite", False),
    )
    print(f"\n{len(written)} files written to {config['output_dir']}")
    stage_provenance(
        config["output_dir"],
        config,
        run_type="package",
        extras={"n_files": len(written), "upstream": config["source_dir"]},
    )


if __name__ == "__main__":
    main()

# ========================= EOF ====================================================================
