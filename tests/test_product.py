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
    CHUNK_X,
    CHUNK_Y,
    FLAG_FILL_VALUE,
    GRID_EPSG,
    NODATA_VALUE,
    build_bands,
    derive_gapfill_flag,
    date_from_path,
    grid_coordinates,
    grid_mapping_attributes,
    group_rasters,
    package_release,
    parse_standard_names,
    read_source_raster,
    time_value,
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


class TestLinearSuctionBand:
    """suction_cm is an exact transform and the first band worth dropping."""

    def test_included_by_default(self):
        assert "suction_cm" in [s.name for s in build_bands(SAMPLE).specs]

    def test_can_be_dropped(self):
        bands = build_bands(SAMPLE, include_linear_suction=False)
        names = [s.name for s in bands.specs]
        assert names == ["log10_suction_cm", "matric_potential_MPa"]

    def test_dropping_it_leaves_the_others_untouched(self):
        full = build_bands(SAMPLE)
        lean = build_bands(SAMPLE, include_linear_suction=False)
        for name in ("log10_suction_cm", "matric_potential_MPa"):
            np.testing.assert_array_equal(full.band(name), lean.band(name))


class TestGridMetadata:
    def test_coordinates_are_cell_centres(self):
        profile = _profile((2, 3))
        x, y = grid_coordinates(profile)
        transform = profile["transform"]
        assert x[0] == pytest.approx(transform.c + 0.5 * transform.a)
        assert y[0] == pytest.approx(transform.f + 0.5 * transform.e)
        assert len(x) == 3 and len(y) == 2

    def test_rotated_grid_is_refused(self):
        profile = _profile((2, 3))
        profile["transform"] = rasterio.Affine(9008.0, 1.0, 0.0, 1.0, -9008.0, 0.0)
        with pytest.raises(ValueError, match="Rotated"):
            grid_coordinates(profile)

    def test_projection_parameters_stated_for_the_expected_grid(self):
        attrs = grid_mapping_attributes(_profile((2, 3)))
        assert attrs["grid_mapping_name"] == "lambert_cylindrical_equal_area"
        assert attrs["standard_parallel"] == 30.0
        assert attrs["epsg_code"] == f"EPSG:{GRID_EPSG}"

    def test_projection_parameters_withheld_on_another_crs(self):
        """Claiming unchecked parameters would be worse than omitting them."""
        profile = _profile((2, 3))
        profile["crs"] = rasterio.crs.CRS.from_epsg(4326)
        attrs = grid_mapping_attributes(profile)
        assert "grid_mapping_name" not in attrs
        assert "standard_parallel" not in attrs
        assert "crs_wkt" in attrs

    def test_time_value_is_days_since_epoch(self):
        assert time_value("19700101") == 0.0
        assert time_value("19700102") == 1.0
        assert time_value("20240101") == 19723.0


class TestGrouping:
    def _paths(self, *dates):
        from pathlib import Path

        return [Path(f"suction_{d}.tif") for d in dates]

    def test_groups_by_year(self):
        groups = group_rasters(self._paths("20231231", "20240101", "20240102"))
        assert sorted(groups) == ["2023", "2024"]
        assert len(groups["2024"]) == 2

    def test_group_all_is_one_bucket(self):
        groups = group_rasters(self._paths("20231231", "20240101"), group_by="all")
        assert list(groups) == ["all"]
        assert len(groups["all"]) == 2

    def test_rejects_unknown_grouping(self):
        with pytest.raises(ValueError, match="year"):
            group_rasters(self._paths("20240101"), group_by="decade")


class TestNetCDFRelease:
    """The deposit container: same numbers, CF-native metadata."""

    def _release(self, tmp_path, dates=("20240101",), **kwargs):
        src = tmp_path / "src"
        src.mkdir(exist_ok=True)
        for date in dates:
            _write_source(src / f"suction_{date}.tif", SAMPLE)
        out = tmp_path / "out"
        return package_release(
            str(src), str(out), level=1, container="netcdf", **kwargs
        )

    def test_round_trip_matches_the_documented_conversion(self, tmp_path):
        netCDF4 = pytest.importorskip("netCDF4")
        written = self._release(tmp_path)
        with netCDF4.Dataset(written[0]) as ds:
            ds.set_auto_mask(False)
            log10 = ds["log10_suction_cm"][0]
            mpa = ds["matric_potential_MPa"][0]
        valid = log10 != NODATA_VALUE
        expected = -np.power(10.0, log10[valid].astype(np.float64)) / MPA_TO_CM
        np.testing.assert_allclose(mpa[valid], expected, rtol=1e-6)
        assert (mpa[valid] < 0).all()
        assert (mpa[~valid] == NODATA_VALUE).all()

    def test_time_coordinate_decodes_to_the_source_dates(self, tmp_path):
        netCDF4 = pytest.importorskip("netCDF4")
        written = self._release(tmp_path, dates=("20240101", "20240102"))
        with netCDF4.Dataset(written[0]) as ds:
            times = netCDF4.num2date(
                ds["time"][:], ds["time"].units, ds["time"].calendar
            )
        assert [t.strftime("%Y%m%d") for t in times] == ["20240101", "20240102"]

    def test_one_file_per_year(self, tmp_path):
        pytest.importorskip("netCDF4")
        written = self._release(tmp_path, dates=("20231231", "20240101", "20240102"))
        assert sorted(p.name for p in written) == [
            "swapstress_psi_L1_2023.nc",
            "swapstress_psi_L1_2024.nc",
        ]

    def test_carries_cf_file_attributes(self, tmp_path):
        netCDF4 = pytest.importorskip("netCDF4")
        written = self._release(tmp_path, dates=("20240101", "20240102"))
        with netCDF4.Dataset(written[0]) as ds:
            assert ds.Conventions == "CF-1.10"
            assert ds.time_coverage_start == "20240101"
            assert ds.time_coverage_end == "20240102"
            assert ds["log10_suction_cm"].grid_mapping == "crs"
            assert ds["crs"].epsg_code == f"EPSG:{GRID_EPSG}"

    def test_chunking_clamps_to_the_array(self, tmp_path):
        netCDF4 = pytest.importorskip("netCDF4")
        written = self._release(tmp_path)
        with netCDF4.Dataset(written[0]) as ds:
            time_chunk, y_chunk, x_chunk = ds["log10_suction_cm"].chunking()
        assert time_chunk == 1  # one day in the file
        assert y_chunk == min(CHUNK_Y, SAMPLE.shape[0])
        assert x_chunk == min(CHUNK_X, SAMPLE.shape[1])

    def test_skips_existing_unless_overwritten(self, tmp_path):
        pytest.importorskip("netCDF4")
        assert len(self._release(tmp_path)) == 1
        assert len(self._release(tmp_path)) == 0
        assert len(self._release(tmp_path, overwrite=True)) == 1

    def test_rejects_unknown_container(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        _write_source(src / "suction_20240101.tif", SAMPLE)
        with pytest.raises(ValueError, match="geotiff"):
            package_release(str(src), str(tmp_path / "out"), level=1, container="zarr")

    def test_a_day_with_different_bands_is_refused(self, tmp_path):
        """Stacking a heterogeneous run would misalign what a variable holds."""
        pytest.importorskip("netCDF4")
        src = tmp_path / "src"
        src.mkdir()
        # Day one carries an uncertainty band, day two does not.
        stacked = np.stack([SAMPLE, np.full_like(SAMPLE, 0.4)])
        _write_source(
            src / "suction_20240101.tif",
            stacked,
            descriptions=("log10_suction_cm", "uncertainty"),
        )
        _write_source(src / "suction_20240102.tif", SAMPLE)
        with pytest.raises(ValueError, match="but the stack was opened with"):
            package_release(
                str(src), str(tmp_path / "out"), level=1, container="netcdf"
            )
        # A half-written stack has silent holes in its time axis; it must not
        # survive to look like a finished file.
        assert not (tmp_path / "out" / "swapstress_psi_L1_2024.nc").exists()

    def test_opens_as_a_time_series_in_xarray(self, tmp_path):
        """Point time series in one read is why the deposit is stacked."""
        xr = pytest.importorskip("xarray")
        written = self._release(tmp_path, dates=("20240101", "20240102"))
        with xr.open_dataset(written[0], decode_coords="all") as ds:
            series = ds["log10_suction_cm"].isel(y=0, x=0)
            assert series.sizes == {"time": 2}
            assert series.values == pytest.approx([5.0, 5.0])
            assert str(ds["time"].values[0])[:10] == "2024-01-01"


class TestNetCDFFlagBand:
    """The flag is one byte with its own sentinel, not a float -9999."""

    def _level2(self, tmp_path):
        level1 = tmp_path / "l1"
        level2 = tmp_path / "l2"
        level1.mkdir()
        level2.mkdir()
        # Level 1 has a gap in the last pixel; Level 2 fills it.
        retrieved = np.array([[5.0, 4.18, 2.5], [1.0, 0.0, NODATA_VALUE]], np.float32)
        filled = np.array([[5.0, 4.18, 2.5], [1.0, 0.0, 3.0]], np.float32)
        _write_source(level1 / "suction_20240101.tif", retrieved)
        _write_source(level2 / "suction_20240101.tif", filled)
        return package_release(
            str(level2),
            str(tmp_path / "out"),
            level=2,
            level1_dir=str(level1),
            container="netcdf",
        )

    def test_flag_is_uint8_with_its_own_fill(self, tmp_path):
        netCDF4 = pytest.importorskip("netCDF4")
        written = self._level2(tmp_path)
        with netCDF4.Dataset(written[0]) as ds:
            var = ds["gapfill_flag"]
            assert var.dtype == np.uint8
            assert var._FillValue == FLAG_FILL_VALUE

    def test_flag_marks_only_the_filled_pixel(self, tmp_path):
        netCDF4 = pytest.importorskip("netCDF4")
        written = self._level2(tmp_path)
        with netCDF4.Dataset(written[0]) as ds:
            ds.set_auto_mask(False)
            flag = ds["gapfill_flag"][0]
        assert flag[1, 2] == 1  # the pixel Level 1 was missing
        assert flag[0, 0] == 0  # retrieved
        assert set(np.unique(flag).tolist()) <= {0, 1}

    def test_geotiff_still_reports_its_own_float_sentinel(self, tmp_path):
        """The GeoTIFF carries one dtype, so its metadata must say -9999."""
        out = tmp_path / "psi.tif"
        bands = build_bands(SAMPLE, gapfill_flag=np.zeros_like(SAMPLE))
        write_geotiff(out, _profile(SAMPLE.shape), bands, level=2, date_str="20240101")
        with rasterio.open(out) as src:
            tags = src.tags(bands.index("gapfill_flag") + 1)
        assert tags["_FillValue"] == str(NODATA_VALUE)
