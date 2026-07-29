"""Stage 07: swapstress-package -- write the released product.

The model predicts one thing, ``log10_suction_cm``. This stage turns a directory
of those daily rasters into the released files: the same quantity in the three
representations Table 2 documents, plus the QRF quantile pair when the model was
run with quantiles and, at Level 2, a per-pixel flag saying which values were
gap-filled.

Two sign conventions coexist by design, each matching its own community's norm.
Suction head is positive and increases as soil dries; matric potential is
negative and approaches zero as soil wets. Every band carries its convention in
its own metadata rather than leaving a reuser to infer it from the values.

The conversion is exact, not approximate. Because the target is a base-10
logarithm, going to MPa is an additive shift in log space, so no reported metric
moves; :mod:`swapstress.units` holds the constant and the argument.

Uncertainty
-----------
The released representation is the **quantile pair** ``log10_suction_cm_q025``
and ``log10_suction_cm_q975``, not a single interval width. Two reasons, and
both point the same way. The width is derivable from the pair and the pair is
not derivable from the width, and the product ships only what a reuser cannot
reconstruct. And a QRF interval in log space is asymmetric about the median, so
a width alone discards where the mass sits -- which for a stress product is the
question, since the dry side of the interval is the side that matters.

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
without touching disk. Two containers consume that same band list, and neither
changes what a band means:

**NetCDF-4/CF, time-stacked by year** (``--container netcdf``) is the archive of
record for a static, versioned, citable deposit. CF is native there rather than
riding in TIFF tags, time is a real coordinate instead of a filename to parse,
each variable carries its own dtype, and a point time series is one chunked read
instead of thousands of file opens. It uses deflate rather than zstd: worse
ratio, but a deposit should optimize for opening everywhere in ten years.

**Per-day GeoTIFF** (``--container geotiff``) is the derived convenience form for
GIS users, and what the pipeline writes internally.

A year of this grid across the released bands is tens of gigabytes, far past
what fits in memory, so the NetCDF path creates the file with its full time
dimension up front and writes each day into its slice as it is read.
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
# The flag band is uint8 where the container allows it, and -9999 does not fit.
FLAG_FILL_VALUE = 255
CONVENTIONS = "CF-1.10"

# The released prediction interval, as QRF quantile levels: a central 95% band.
# Named here rather than in the inference stage because the band names encode
# these numbers, so the release has exactly one place that fixes them.
QUANTILE_LEVELS = (0.025, 0.975)
# Its nominal coverage, derived rather than restated -- the number the coverage
# analysis compares against and the figure labels.
RELEASE_NOMINAL = round(QUANTILE_LEVELS[1] - QUANTILE_LEVELS[0], 6)

# NetCDF deposit settings. The chunk shape is the one decision that is expensive
# to get wrong, because it is fixed at write time: (time=1, y=all, x=all) makes
# maps fast and point time series pathological, and the reverse also fails. This
# is the balance -- about 2 MB uncompressed per chunk, both access patterns
# served. Deflate rather than zstd: worse ratio, but every reader has it.
CHUNK_TIME, CHUNK_Y, CHUNK_X = 30, 128, 128
DEFLATE_LEVEL = 4
TIME_UNITS = "days since 1970-01-01 00:00:00"
TIME_CALENDAR = "standard"
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

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
    # Storage type and sentinel for containers that allow one per variable.
    # GeoTIFF ignores both -- it carries a single dtype for every band.
    dtype: str = "float32"
    fill_value: float = NODATA_VALUE

    @property
    def typed_fill(self):
        """The fill sentinel in this band's own dtype."""
        return np.dtype(self.dtype).type(self.fill_value)

    def attributes(
        self,
        standard_names: Optional[dict] = None,
        fill_value: Optional[float] = None,
    ) -> dict:
        """CF-style attributes for this band.

        *fill_value* lets a container state the sentinel it actually wrote:
        NetCDF stores the flag band as uint8/255, GeoTIFF as float32/-9999, and
        the metadata has to match the bytes rather than the intent.
        """
        fill = self.fill_value if fill_value is None else fill_value
        attrs = {
            "long_name": self.long_name,
            "units": self.units,
            "_FillValue": str(fill),
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


def quantile_band_name(level: float) -> str:
    """The band name for a quantile *level*, e.g. 0.025 -> ``..._q025``.

    The suffix is the level in per mille, three digits, so 0.025 and 0.975 read
    as ``q025``/``q975`` and nothing rounds two distinct levels together.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"A quantile level lies strictly in (0, 1); got {level}")
    return f"log10_suction_cm_q{level * 1000:03.0f}"


Q025_BAND_NAME = quantile_band_name(QUANTILE_LEVELS[0])
Q975_BAND_NAME = quantile_band_name(QUANTILE_LEVELS[1])

# The uncertainty representation, decided at release: the pair, not the width.
# See the module docstring -- width is derivable from the pair, the pair is not
# derivable from the width, and the interval is asymmetric in log space.
_QUANTILE_COMMENT = (
    "{ordinal} percentile of the QRF predictive distribution, in the model's "
    "own units. Together with {other} it is the released 95% prediction "
    "interval; the interval width q975 - q025 is deliberately not shipped as "
    "its own band because it is exactly derivable from the pair, while the "
    "pair is not derivable from a width. The interval is asymmetric about the "
    "median. A width read off this pair is identical whether the values are "
    "read as log10(cm) or log10(|MPa|), since the unit change is an additive "
    "shift. Present only when the model was run with quantiles."
)

LOG10_SUCTION_Q025 = BandSpec(
    name=Q025_BAND_NAME,
    units="log10(cm)",
    long_name="Lower bound of the QRF 95% prediction interval, log10 suction head",
    sign="positive",
    comment=_QUANTILE_COMMENT.format(ordinal="2.5th", other=Q975_BAND_NAME),
)

LOG10_SUCTION_Q975 = BandSpec(
    name=Q975_BAND_NAME,
    units="log10(cm)",
    long_name="Upper bound of the QRF 95% prediction interval, log10 suction head",
    sign="positive",
    comment=_QUANTILE_COMMENT.format(ordinal="97.5th", other=Q025_BAND_NAME),
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
    # A flag needs one byte, not four, and the float NODATA sentinel does not
    # fit in it. Containers that allow per-variable types get uint8/255.
    dtype="uint8",
    fill_value=FLAG_FILL_VALUE,
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


def check_interval(
    log10: np.ndarray,
    q025: np.ndarray,
    q975: np.ndarray,
    valid: np.ndarray,
) -> None:
    """Refuse a quantile pair that does not bracket the median it came with.

    A QRF returns its quantiles as order statistics of one predictive
    distribution, so ``q025 <= median <= q975`` holds by construction. A
    violation therefore does not mean an unlucky pixel -- it means the bands
    were paired up wrongly, or a quantile raster is masked where the median is
    not. Either way the released interval would be a lie, so it stops here.
    """
    low_high = int(np.count_nonzero(q025[valid] > log10[valid]))
    high_low = int(np.count_nonzero(q975[valid] < log10[valid]))
    if not (low_high or high_low):
        return
    raise ValueError(
        f"The quantile pair does not bracket the median: {Q025_BAND_NAME} "
        f"exceeds it at {low_high:,} pixels and {Q975_BAND_NAME} falls below "
        f"it at {high_low:,} of {int(valid.sum()):,} valid pixels. The bands "
        "are swapped, misaligned, or one of them carries NODATA where the "
        "median does not."
    )


def build_bands(
    log10_suction_cm: np.ndarray,
    q025: Optional[np.ndarray] = None,
    q975: Optional[np.ndarray] = None,
    gapfill_flag: Optional[np.ndarray] = None,
    include_linear_suction: bool = True,
) -> ProductBands:
    """Derive the released bands from one day of model output.

    Pure computation -- no I/O, no container assumptions. Invalid pixels are
    carried through as NODATA rather than being filled: a gap in the retrieval
    is information, and the ``gapfill_flag`` band is how a filled value is
    distinguished from a retrieved one.

    The three representations are exact transforms of one another, so storing
    all of them costs 3x for no added information. ``include_linear_suction``
    drops ``suction_cm``, which is the one worth dropping first: it spans about
    0 to 1e6 linearly, so it is high-entropy and compresses worst, while
    ``log10_suction_cm`` is smooth. The log band has to stay regardless -- it is
    the model's actual output, and the units the quantile pair is expressed in.

    ``q025``/``q975`` are the released uncertainty representation. A run without
    quantiles has neither and its files simply carry no interval, the same way
    Level 1 carries no gap-fill flag. Half a pair is not a lesser case of that:
    it is a broken run, and it raises.
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

    specs = [LOG10_SUCTION, MATRIC_POTENTIAL]
    bands = [np.where(valid, log10, NODATA_VALUE).astype(np.float32), mpa]

    if include_linear_suction:
        specs.append(SUCTION)
        bands.append(suction)

    if (q025 is None) != (q975 is None):
        missing = Q975_BAND_NAME if q025 is not None else Q025_BAND_NAME
        raise ValueError(
            f"The released uncertainty representation is the {Q025_BAND_NAME} / "
            f"{Q975_BAND_NAME} pair, and {missing} is missing. Package a run "
            "with both quantile rasters or with neither."
        )

    if q025 is not None:
        low = np.asarray(q025, dtype=np.float32)
        high = np.asarray(q975, dtype=np.float32)
        check_interval(log10, low, high, valid)
        specs.extend([LOG10_SUCTION_Q025, LOG10_SUCTION_Q975])
        bands.append(np.where(valid, low, NODATA_VALUE).astype(np.float32))
        bands.append(np.where(valid, high, NODATA_VALUE).astype(np.float32))

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
    time_coverage: Optional[tuple] = None,
) -> dict:
    """File-level CF attributes, including the link back to the run record.

    *date_str* is the single day for a per-day file and the group key (a year)
    for a stack; *time_coverage* carries the span the stack actually holds.
    """
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
    if time_coverage:
        attrs["time_coverage_start"], attrs["time_coverage_end"] = time_coverage
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
            # Every band really is float32/-9999 here, whatever the spec would
            # prefer, so the reported sentinel has to say so.
            dst.update_tags(i, **spec.attributes(standard_names, NODATA_VALUE))
    return out_path


def grid_coordinates(profile: dict) -> tuple:
    """Cell-centre x and y coordinates from the affine transform."""
    transform = profile["transform"]
    if transform.b or transform.d:
        raise ValueError(
            "Rotated grids have no 1-D x/y coordinate variables; this product "
            "is on a north-up EASE-Grid2 raster."
        )
    x = transform.c + (np.arange(profile["width"]) + 0.5) * transform.a
    y = transform.f + (np.arange(profile["height"]) + 0.5) * transform.e
    return x, y


def grid_mapping_attributes(profile: dict) -> dict:
    """CF grid-mapping attributes for the raster's CRS.

    The projection parameters are only asserted when the raster really is on the
    expected grid; on anything else the WKT is emitted alone rather than
    claiming parameters that were never checked.
    """
    attrs = {"grid": GRID_NAME}
    crs = profile.get("crs")
    epsg = crs.to_epsg() if crs is not None else None

    if epsg == GRID_EPSG:
        attrs.update(
            grid_mapping_name="lambert_cylindrical_equal_area",
            standard_parallel=30.0,
            longitude_of_central_meridian=0.0,
            false_easting=0.0,
            false_northing=0.0,
            semi_major_axis=6378137.0,
            inverse_flattening=298.257223563,
        )
    if epsg is not None:
        attrs["epsg_code"] = f"EPSG:{epsg}"
    if crs is not None:
        attrs["crs_wkt"] = crs.to_wkt()
    return attrs


def time_value(date_str: str) -> float:
    """A YYYYMMDD date as a CF time coordinate value."""
    day = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
    return float((day - EPOCH).days)


class NetCDFStack:
    """A time-stacked NetCDF-4/CF file, written one day at a time.

    A year of the global M09 grid across the released bands runs to tens of
    gigabytes, so the file is created with its full time dimension up front and
    each day is written into its slice as it is read. Nothing accumulates in
    memory beyond a single day.
    """

    def __init__(
        self,
        out_path: Path,
        profile: dict,
        specs: List[BandSpec],
        dates: Sequence[str],
        level: int,
        provenance: Optional[str] = None,
        source: Optional[str] = None,
        standard_names: Optional[dict] = None,
        group_key: str = "",
    ):
        import netCDF4

        self.specs = list(specs)
        self.dates = list(dates)
        self.path = out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        x, y = grid_coordinates(profile)
        self.ds = netCDF4.Dataset(out_path, "w", format="NETCDF4")
        self.ds.createDimension("time", len(self.dates))
        self.ds.createDimension("y", len(y))
        self.ds.createDimension("x", len(x))

        self.ds.setncatts(
            file_attributes(
                level,
                group_key or self.dates[0],
                provenance,
                source,
                time_coverage=(self.dates[0], self.dates[-1]),
            )
        )

        time_var = self.ds.createVariable("time", "f8", ("time",))
        time_var.setncatts(
            {
                "standard_name": "time",
                "long_name": "time",
                "units": TIME_UNITS,
                "calendar": TIME_CALENDAR,
                "axis": "T",
            }
        )
        time_var[:] = [time_value(d) for d in self.dates]

        for name, values, axis, long_name in (
            ("y", y, "Y", "y coordinate of projection"),
            ("x", x, "X", "x coordinate of projection"),
        ):
            var = self.ds.createVariable(name, "f8", (name,))
            var.setncatts(
                {
                    "standard_name": f"projection_{name}_coordinate",
                    "long_name": long_name,
                    "units": "m",
                    "axis": axis,
                }
            )
            var[:] = values

        crs_var = self.ds.createVariable("crs", "i4")
        crs_var.setncatts(grid_mapping_attributes(profile))

        chunks = (
            min(CHUNK_TIME, len(self.dates)),
            min(CHUNK_Y, len(y)),
            min(CHUNK_X, len(x)),
        )
        for spec in self.specs:
            var = self.ds.createVariable(
                spec.name,
                spec.dtype,
                ("time", "y", "x"),
                zlib=True,
                complevel=DEFLATE_LEVEL,
                chunksizes=chunks,
                fill_value=spec.typed_fill,
            )
            attrs = spec.attributes(standard_names, spec.fill_value)
            # createVariable already wrote _FillValue, typed to match the
            # variable. Setting it again -- and as a string -- is rejected.
            attrs.pop("_FillValue", None)
            var.setncatts(attrs)

    def write_day(self, index: int, bands: ProductBands) -> None:
        """Write one day into its time slice."""
        names = [s.name for s in bands.specs]
        if names != [s.name for s in self.specs]:
            # A day that gained or lost a band mid-stack means the inputs are
            # not one homogeneous run; stacking them would silently misalign
            # what each variable holds.
            raise ValueError(
                f"{self.path.name}: day {self.dates[index]} has bands {names}, "
                f"but the stack was opened with {[s.name for s in self.specs]}"
            )

        for spec in self.specs:
            data = bands.band(spec.name)
            if spec.dtype == "float32":
                self.ds.variables[spec.name][index, :, :] = data
                continue
            # Narrower types cannot hold the float sentinel, so invalid pixels
            # take the band's own fill rather than a cast of -9999.
            out = np.full(data.shape, spec.fill_value, dtype=spec.dtype)
            out[bands.valid] = data[bands.valid].astype(spec.dtype)
            self.ds.variables[spec.name][index, :, :] = out

    def close(self) -> None:
        self.ds.close()


def read_source_raster(path: Path) -> tuple:
    """Read a predict/gapfill raster into (log10, q025, q975, profile).

    Band 1 is always ``log10_suction_cm``. The quantile bands are found by name,
    so a run without them reads back as ``(log10, None, None, profile)``; the
    linear ``suction_cm`` band that predict can also write is ignored, because
    this stage recomputes it.

    A raster carrying one half of the pair is refused here rather than in
    :func:`build_bands`, so the message can name the file that is wrong.
    """
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        descriptions = list(src.descriptions)
        log10 = src.read(1).astype(np.float32)
        if src.nodata is not None and not np.isnan(src.nodata):
            log10[log10 == src.nodata] = NODATA_VALUE

        present = [n for n in (Q025_BAND_NAME, Q975_BAND_NAME) if n in descriptions]
        if len(present) == 1:
            raise ValueError(
                f"{path} carries {present[0]} but not the other half of the "
                f"released quantile pair ({Q025_BAND_NAME} / {Q975_BAND_NAME}). "
                "Re-run inference with --release-quantiles."
            )

        quantiles = [
            src.read(descriptions.index(name) + 1).astype(np.float32)
            for name in present
        ] or [None, None]
    return log10, quantiles[0], quantiles[1], profile


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


def _day_bands(
    path: Path,
    level: int,
    level1: Optional[Path],
    include_linear_suction: bool,
    require_quantiles: bool = False,
) -> tuple:
    """Read one source raster and derive its released bands."""
    log10, q025, q975, profile = read_source_raster(path)
    if require_quantiles and q025 is None:
        # The stacked container would catch this by refusing a day whose bands
        # changed mid-stack, but the per-day GeoTIFF path would happily write a
        # file with no interval in it. A release run says what it expects.
        raise ValueError(
            f"{path} carries no quantile bands, and --require-quantiles was "
            f"asked for. The released interval is {Q025_BAND_NAME} / "
            f"{Q975_BAND_NAME}; re-run inference for this day with "
            "--release-quantiles, or package without --require-quantiles."
        )
    flag = None
    if level == 2:
        flag = derive_gapfill_flag(log10, (level1 / path.name) if level1 else None)
    bands = build_bands(
        log10,
        q025=q025,
        q975=q975,
        gapfill_flag=flag,
        include_linear_suction=include_linear_suction,
    )
    return bands, profile, flag


def group_rasters(rasters: Sequence[Path], group_by: str = "year") -> dict:
    """Group daily rasters into the files a stacked container will hold.

    Yearly keeps each file individually downloadable and resumable, isolates a
    corrupt file to one year, and gives a natural subsetting unit; ``all`` puts
    the whole record in one file.
    """
    groups: dict = {}
    for path in rasters:
        date_str = date_from_path(path)
        if group_by == "year":
            key = date_str[:4]
        elif group_by == "all":
            key = "all"
        else:
            raise ValueError(f"--group-by expects 'year' or 'all', got '{group_by}'")
        groups.setdefault(key, []).append(path)
    return groups


def _package_geotiff(
    rasters: Sequence[Path],
    out_root: Path,
    level: int,
    level1: Optional[Path],
    out_prefix: str,
    provenance: Optional[str],
    standard_names: Optional[dict],
    overwrite: bool,
    include_linear_suction: bool,
    require_quantiles: bool,
    writer: Callable[..., Path],
) -> List[Path]:
    """One file per day."""
    written: List[Path] = []
    for path in rasters:
        date_str = date_from_path(path)
        out_path = out_root / f"{out_prefix}_L{level}_{date_str}.tif"
        if out_path.exists() and not overwrite:
            print(f"  exists, skipping: {out_path.name}")
            continue

        bands, profile, flag = _day_bands(
            path, level, level1, include_linear_suction, require_quantiles
        )
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


def _package_netcdf(
    rasters: Sequence[Path],
    out_root: Path,
    level: int,
    level1: Optional[Path],
    out_prefix: str,
    provenance: Optional[str],
    standard_names: Optional[dict],
    overwrite: bool,
    include_linear_suction: bool,
    require_quantiles: bool,
    source_dir: Path,
    group_by: str,
) -> List[Path]:
    """One time-stacked file per group."""
    written: List[Path] = []
    for key, paths in group_rasters(rasters, group_by).items():
        out_path = out_root / f"{out_prefix}_L{level}_{key}.nc"
        if out_path.exists() and not overwrite:
            print(f"  exists, skipping: {out_path.name}")
            continue

        dates = [date_from_path(p) for p in paths]
        stack: Optional[NetCDFStack] = None
        filled = 0
        try:
            for i, path in enumerate(paths):
                bands, profile, flag = _day_bands(
                    path, level, level1, include_linear_suction, require_quantiles
                )
                if stack is None:
                    stack = NetCDFStack(
                        out_path,
                        profile,
                        bands.specs,
                        dates,
                        level,
                        provenance=provenance,
                        source=str(source_dir),
                        standard_names=standard_names,
                        group_key=key,
                    )
                stack.write_day(i, bands)
                if flag is not None:
                    filled += int(flag.sum())
        except BaseException:
            # A stack that failed part way through is a file with silent holes
            # in the middle of its time axis. Remove it rather than leave
            # something depositable-looking behind.
            if stack is not None:
                stack.close()
                out_path.unlink(missing_ok=True)
            raise
        else:
            stack.close()

        written.append(out_path)
        summary = f"  wrote {out_path.name}  {len(dates)} days"
        if level == 2:
            summary += f"  {filled:,} gap-filled px"
        print(summary)
    return written


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
    container: str = "geotiff",
    include_linear_suction: bool = True,
    require_quantiles: bool = False,
    group_by: str = "year",
) -> List[Path]:
    """Convert a directory of daily model rasters into released files.

    ``netcdf`` stacks the days into one CF file per group and is the archive of
    record for a deposit; ``geotiff`` writes one file per day.

    ``require_quantiles`` refuses a source raster that carries no
    ``log10_suction_cm_q025`` / ``log10_suction_cm_q975`` pair. Off by default,
    because packaging a median-only experiment is legitimate; on for a release
    run, where a day quietly missing its interval is the failure to catch.
    """
    source = Path(source_dir)
    out_root = Path(output_dir)
    level1 = Path(level1_dir) if level1_dir else None

    rasters = find_rasters(source, prefix)
    if not rasters:
        raise FileNotFoundError(f"No {prefix}_*.tif rasters under {source}")
    print(f"{len(rasters)} daily rasters in {source}")

    if container == "geotiff":
        return _package_geotiff(
            rasters,
            out_root,
            level,
            level1,
            out_prefix,
            provenance,
            standard_names,
            overwrite,
            include_linear_suction,
            require_quantiles,
            writer,
        )
    if container == "netcdf":
        return _package_netcdf(
            rasters,
            out_root,
            level,
            level1,
            out_prefix,
            provenance,
            standard_names,
            overwrite,
            include_linear_suction,
            require_quantiles,
            source,
            group_by,
        )
    raise ValueError(f"--container expects 'geotiff' or 'netcdf', got '{container}'")


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
        "--container",
        default=None,
        choices=["netcdf", "geotiff"],
        help="netcdf = time-stacked CF files, the deposit archive of record; "
        "geotiff = one file per day (default: geotiff).",
    )
    parser.add_argument(
        "--group-by",
        default=None,
        choices=["year", "all"],
        help="Days per stacked NetCDF file (default: year).",
    )
    parser.add_argument(
        "--drop-linear-suction",
        action="store_true",
        default=None,
        help="Omit the suction_cm band. It is an exact transform of "
        "log10_suction_cm and compresses worst; recommended for the deposit.",
    )
    parser.add_argument(
        "--require-quantiles",
        action="store_true",
        default=None,
        help=f"Refuse a source raster without the {Q025_BAND_NAME} / "
        f"{Q975_BAND_NAME} pair. Use for a release run, where a day silently "
        "missing its prediction interval is the failure to catch.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
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
    config.setdefault("container", "geotiff")
    config.setdefault("group_by", "year")

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
        container=config["container"],
        include_linear_suction=not config.get("drop_linear_suction", False),
        require_quantiles=config.get("require_quantiles", False),
        group_by=config["group_by"],
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
