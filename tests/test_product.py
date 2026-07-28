"""Tests for swapstress.inference.product — the released band stack.

The gate for this stage is the round trip: write the product, read it back, and
confirm the MPa band is exactly the documented function of the model's own
output and is negative everywhere it is valid.
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from swapstress.inference.product import (
    NODATA_VALUE,
    build_bands,
    derive_gapfill_flag,
    date_from_path,
    package_release,
    parse_standard_names,
    read_source_raster,
    write_geotiff,
)
from swapstress.units import MPA_TO_CM

# A spread of realistic log10 suction values plus a NODATA pixel: dry surface,
# wilting point, field capacity, wet, and a gap.
SAMPLE = np.array(
    [
        [5.0, 4.18, 2.5],
        [1.0, 0.0, NODATA_VALUE],
    ],
    dtype=np.float32,
)


def _profile(shape):
    height, width = shape
    return {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": rasterio.crs.CRS.from_epsg(6933),
        "transform": from_origin(-17367530.45, 7314540.83, 9008.05, 9008.05),
        "nodata": NODATA_VALUE,
    }


def _write_source(path, data, descriptions=("log10_suction_cm",)):
    profile = _profile(data.shape[-2:])
    profile["count"] = data.shape[0] if data.ndim == 3 else 1
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data if data.ndim == 3 else data[np.newaxis])
        for i, name in enumerate(descriptions, start=1):
            dst.set_band_description(i, name)
    return path


class TestBuildBands:
    def test_default_band_order(self):
        bands = build_bands(SAMPLE)
        assert [s.name for s in bands.specs] == [
            "log10_suction_cm",
            "matric_potential_MPa",
            "suction_cm",
        ]

    def test_mpa_is_the_documented_conversion(self):
        bands = build_bands(SAMPLE)
        valid = bands.valid
        expected = -np.power(10.0, SAMPLE[valid].astype(np.float64)) / MPA_TO_CM
        assert bands.band("matric_potential_MPa")[valid] == pytest.approx(
            expected, rel=1e-6
        )

    def test_mpa_is_negative_everywhere_valid(self):
        bands = build_bands(SAMPLE)
        assert np.all(bands.band("matric_potential_MPa")[bands.valid] < 0)

    def test_suction_is_positive_everywhere_valid(self):
        bands = build_bands(SAMPLE)
        assert np.all(bands.band("suction_cm")[bands.valid] > 0)

    def test_nodata_is_not_exponentiated(self):
        """10 ** -9999 is 0.0, which would read as perfectly wet soil."""
        bands = build_bands(SAMPLE)
        gap = ~bands.valid
        assert gap.sum() == 1
        for name in ("log10_suction_cm", "matric_potential_MPa", "suction_cm"):
            assert np.all(bands.band(name)[gap] == NODATA_VALUE)

    def test_nan_counts_as_invalid(self):
        data = SAMPLE.copy()
        data[0, 0] = np.nan
        bands = build_bands(data)
        assert not bands.valid[0, 0]
        assert bands.band("suction_cm")[0, 0] == NODATA_VALUE

    def test_wilting_point_is_minus_1p5_mpa(self):
        """4.18 is the rounded figure quoted in the docs; this is the exact one."""
        log10_pwp = np.log10(1.5 * MPA_TO_CM)
        bands = build_bands(np.array([[log10_pwp]], dtype=np.float32))
        assert bands.band("matric_potential_MPa")[0, 0] == pytest.approx(-1.5, rel=1e-5)

    def test_rounded_wilting_point_lands_close(self):
        bands = build_bands(np.array([[4.18]], dtype=np.float32))
        assert bands.band("matric_potential_MPa")[0, 0] == pytest.approx(-1.5, rel=2e-2)

    def test_uncertainty_band_added_when_present(self):
        unc = np.full_like(SAMPLE, 0.3)
        bands = build_bands(SAMPLE, uncertainty=unc)
        assert "uncertainty" in [s.name for s in bands.specs]
        assert np.all(bands.band("uncertainty")[bands.valid] == pytest.approx(0.3))

    def test_gapfill_band_added_when_present(self):
        flag = np.zeros_like(SAMPLE)
        flag[1, 0] = 1.0
        bands = build_bands(SAMPLE, gapfill_flag=flag)
        assert bands.band("gapfill_flag")[1, 0] == 1.0
        assert bands.band("gapfill_flag")[0, 0] == 0.0

    def test_band_lookup_rejects_unknown_name(self):
        with pytest.raises(KeyError, match="No band named"):
            build_bands(SAMPLE).band("nope")


class TestRoundTrip:
    """The Phase 5 gate: write, reread, confirm the identity holds on disk."""

    def test_mpa_survives_the_write(self, tmp_path):
        bands = build_bands(SAMPLE)
        out = write_geotiff(
            out_path=tmp_path / "product.tif",
            profile=_profile(SAMPLE.shape),
            bands=bands,
            level=1,
            date_str="20240101",
        )

        with rasterio.open(out) as src:
            names = list(src.descriptions)
            log10 = src.read(names.index("log10_suction_cm") + 1)
            mpa = src.read(names.index("matric_potential_MPa") + 1)
            suction = src.read(names.index("suction_cm") + 1)
            nodata = src.nodata

        valid = log10 != nodata
        expected = -np.power(10.0, log10[valid].astype(np.float64)) / MPA_TO_CM
        assert mpa[valid] == pytest.approx(expected, rel=1e-6)
        assert np.all(mpa[valid] < 0)
        assert suction[valid] == pytest.approx(
            np.power(10.0, log10[valid].astype(np.float64)), rel=1e-6
        )

    def test_grid_and_nodata_are_preserved(self, tmp_path):
        out = write_geotiff(
            out_path=tmp_path / "product.tif",
            profile=_profile(SAMPLE.shape),
            bands=build_bands(SAMPLE),
            level=1,
            date_str="20240101",
        )
        with rasterio.open(out) as src:
            assert src.crs.to_epsg() == 6933
            assert src.nodata == NODATA_VALUE
            assert src.count == 3

    def test_band_metadata_is_written(self, tmp_path):
        out = write_geotiff(
            out_path=tmp_path / "product.tif",
            profile=_profile(SAMPLE.shape),
            bands=build_bands(SAMPLE),
            level=1,
            date_str="20240101",
            provenance="/release/provenance_package.json",
        )
        with rasterio.open(out) as src:
            file_tags = src.tags()
            mpa_tags = src.tags(2)

        assert file_tags["Conventions"].startswith("CF-")
        assert file_tags["product_level"] == "1"
        assert file_tags["grid_epsg"] == "6933"
        assert file_tags["provenance"] == "/release/provenance_package.json"
        assert mpa_tags["units"] == "MPa"
        assert mpa_tags["sign_convention"] == "negative"

    def test_standard_name_omitted_until_pinned(self, tmp_path):
        """Guessing a CF standard name is a release decision, not a code one."""
        out = write_geotiff(
            out_path=tmp_path / "product.tif",
            profile=_profile(SAMPLE.shape),
            bands=build_bands(SAMPLE),
            level=1,
            date_str="20240101",
        )
        with rasterio.open(out) as src:
            assert "standard_name" not in src.tags(2)

    def test_standard_name_written_when_pinned(self, tmp_path):
        out = write_geotiff(
            out_path=tmp_path / "product.tif",
            profile=_profile(SAMPLE.shape),
            bands=build_bands(SAMPLE),
            level=1,
            date_str="20240101",
            standard_names={"matric_potential_MPa": "soil_water_potential"},
        )
        with rasterio.open(out) as src:
            assert src.tags(2)["standard_name"] == "soil_water_potential"


class TestGapfillFlag:
    def test_flags_pixels_level1_lacks(self, tmp_path):
        level1 = SAMPLE.copy()
        level1[1, 0] = NODATA_VALUE  # a gap that gapfill later filled
        _write_source(tmp_path / "suction_20240101.tif", level1)

        flag = derive_gapfill_flag(SAMPLE, tmp_path / "suction_20240101.tif")
        assert flag[1, 0] == 1.0
        assert flag[0, 0] == 0.0
        # The pixel that is NODATA in both levels was never filled.
        assert flag[1, 2] == 0.0

    def test_missing_level1_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="gapfill_flag"):
            derive_gapfill_flag(SAMPLE, tmp_path / "absent.tif")

    def test_no_level1_dir_raises(self):
        with pytest.raises(FileNotFoundError, match="gapfill_flag"):
            derive_gapfill_flag(SAMPLE, None)


class TestReadSource:
    def test_reads_log10_and_ignores_linear_band(self, tmp_path):
        stack = np.stack([SAMPLE, np.power(10.0, SAMPLE)]).astype(np.float32)
        path = _write_source(
            tmp_path / "suction_20240101.tif",
            stack,
            descriptions=("log10_suction_cm", "suction_cm"),
        )
        log10, uncertainty, _ = read_source_raster(path)
        assert uncertainty is None
        assert log10[0, 0] == pytest.approx(5.0)

    def test_carries_uncertainty_band_through(self, tmp_path):
        stack = np.stack([SAMPLE, np.full_like(SAMPLE, 0.25)]).astype(np.float32)
        path = _write_source(
            tmp_path / "suction_20240101.tif",
            stack,
            descriptions=("log10_suction_cm", "uncertainty"),
        )
        _, uncertainty, _ = read_source_raster(path)
        assert uncertainty is not None
        assert uncertainty[0, 0] == pytest.approx(0.25)


class TestPackageRelease:
    def test_writes_one_file_per_day(self, tmp_path):
        src = tmp_path / "inference"
        src.mkdir()
        for day in ("20240101", "20240102"):
            _write_source(src / f"suction_{day}.tif", SAMPLE)

        written = package_release(
            source_dir=str(src), output_dir=str(tmp_path / "release"), level=1
        )
        assert [p.name for p in written] == [
            "swapstress_psi_L1_20240101.tif",
            "swapstress_psi_L1_20240102.tif",
        ]

    def test_level2_adds_the_flag_band(self, tmp_path):
        level1 = tmp_path / "inference"
        level2 = tmp_path / "gapfill"
        level1.mkdir()
        level2.mkdir()
        gapped = SAMPLE.copy()
        gapped[1, 0] = NODATA_VALUE
        _write_source(level1 / "suction_20240101.tif", gapped)
        _write_source(level2 / "suction_20240101.tif", SAMPLE)

        written = package_release(
            source_dir=str(level2),
            output_dir=str(tmp_path / "release"),
            level=2,
            level1_dir=str(level1),
        )
        with rasterio.open(written[0]) as dst:
            names = list(dst.descriptions)
            assert "gapfill_flag" in names
            flag = dst.read(names.index("gapfill_flag") + 1)
        assert flag[1, 0] == 1.0

    def test_skips_existing_without_overwrite(self, tmp_path):
        src = tmp_path / "inference"
        src.mkdir()
        _write_source(src / "suction_20240101.tif", SAMPLE)
        out = tmp_path / "release"

        assert len(package_release(str(src), str(out), level=1)) == 1
        assert len(package_release(str(src), str(out), level=1)) == 0
        assert len(package_release(str(src), str(out), level=1, overwrite=True)) == 1

    def test_empty_source_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No suction_"):
            package_release(str(tmp_path), str(tmp_path / "out"), level=1)


class TestHelpers:
    def test_date_from_path(self):
        from pathlib import Path

        assert date_from_path(Path("suction_20240101.tif")) == "20240101"

    def test_date_from_path_rejects_undated(self):
        from pathlib import Path

        with pytest.raises(ValueError, match="YYYYMMDD"):
            date_from_path(Path("suction.tif"))

    def test_parse_standard_names(self):
        names = parse_standard_names(["matric_potential_MPa=soil_water_potential"])
        assert names["matric_potential_MPa"] == "soil_water_potential"

    def test_parse_standard_names_rejects_malformed(self):
        with pytest.raises(SystemExit, match="BAND=NAME"):
            parse_standard_names(["nope"])
