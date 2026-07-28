"""Tests for the published GSHP parameter loader.

These pin the decisions that made it safe to retire our own GSHP fitting: that
alpha is converted out of the published 1/m, that the quality filter and the
SWCC restriction do what the callers assume, and that an invalid parameter set
surviving the quality filter is an error rather than something to drop.
"""

import numpy as np
import pandas as pd
import pytest

from swapstress.sources.gshp import (
    ALPHA_PER_M_TO_PER_CM,
    FREE_THETA_CLASS,
    GOOD_QUALITY_FLAG,
    load_published_params,
    sanitize_profile_id,
    to_swrc_arrays,
)
from swapstress.swrc import theta_from_psi


def _row(
    layer_id,
    profile_id="p1",
    alpha=2.0,
    n=1.6,
    thetar=0.05,
    thetas=0.45,
    flag=GOOD_QUALITY_FLAG,
    swcc=FREE_THETA_CLASS,
    top=0.0,
    bot=20.0,
):
    return dict(
        layer_id=layer_id,
        profile_id=profile_id,
        alpha=alpha,
        n=n,
        thetar=thetar,
        thetas=thetas,
        data_flag=flag,
        SWCC_classes=swcc,
        hzn_top=top,
        hzn_bot=bot,
        latitude_decimal_degrees=45.0,
        longitude_decimal_degrees=-110.0,
        se_alpha=0.2,
        se_n=0.03,
        lab_head_m=1.0,
        lab_wrc=0.4,
    )


def _csv(tmp_path, rows, name="wrc.csv"):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


class TestUnitConversion:
    def test_alpha_converted_from_per_metre(self, tmp_path):
        got = load_published_params(_csv(tmp_path, [_row("a", alpha=2.5)]))
        assert got.loc[0, "alpha"] == pytest.approx(0.025)

    def test_conversion_constant_is_a_hundredth(self):
        assert ALPHA_PER_M_TO_PER_CM == pytest.approx(0.01)

    def test_alpha_uncertainty_converted_too(self, tmp_path):
        got = load_published_params(_csv(tmp_path, [_row("a")]))
        assert got.loc[0, "se_alpha"] == pytest.approx(0.002)

    def test_n_is_not_rescaled(self, tmp_path):
        got = load_published_params(_csv(tmp_path, [_row("a", n=1.6)]))
        assert got.loc[0, "n"] == pytest.approx(1.6)
        assert got.loc[0, "se_n"] == pytest.approx(0.03)


class TestQualityFilter:
    def test_drops_flagged_layers_by_default(self, tmp_path):
        rows = [_row("a"), _row("b", flag="upper limit for alpha")]
        got = load_published_params(_csv(tmp_path, rows))
        assert list(got["layer_id"]) == ["a"]

    def test_keeps_them_when_disabled(self, tmp_path):
        rows = [_row("a"), _row("b", flag="upper limit for alpha")]
        got = load_published_params(_csv(tmp_path, rows), good_quality_only=False)
        assert sorted(got["layer_id"]) == ["a", "b"]

    def test_flags_parameters_resting_on_the_published_bounds(self, tmp_path):
        rows = [_row("a", alpha=100.0), _row("b", n=7.0), _row("c")]
        got = load_published_params(_csv(tmp_path, rows), good_quality_only=False)
        bound = dict(zip(got["layer_id"], got["at_param_bound"]))
        assert bound["a"] and bound["b"] and not bound["c"]


class TestSwccRestriction:
    def test_restricts_to_requested_classes(self, tmp_path):
        rows = [_row("a", swcc="YWYD"), _row("b", swcc="NWYD")]
        got = load_published_params(
            _csv(tmp_path, rows), swcc_classes=(FREE_THETA_CLASS,)
        )
        assert list(got["layer_id"]) == ["a"]

    def test_free_theta_class_is_ywyd(self):
        """NWYD's theta_s carries a texture PTF prior; only YWYD is free."""
        assert FREE_THETA_CLASS == "YWYD"


class TestInvalidParametersSurface:
    def test_raises_when_an_invalid_layer_survives_the_quality_filter(self, tmp_path):
        rows = [_row("a"), _row("bad", thetar=0.45, thetas=0.45)]
        with pytest.raises(ValueError, match="fail valid_params"):
            load_published_params(_csv(tmp_path, rows))

    def test_reports_but_returns_when_reading_unfiltered(self, tmp_path, capsys):
        rows = [
            _row("a"),
            _row("bad", thetar=0.45, thetas=0.45, flag="flat upper profile for n"),
        ]
        got = load_published_params(_csv(tmp_path, rows), good_quality_only=False)
        assert len(got) == 2
        assert "fail valid_params" in capsys.readouterr().out

    def test_missing_parameter_columns_is_an_error(self, tmp_path):
        path = tmp_path / "bad.csv"
        pd.DataFrame({"layer_id": ["a"], "alpha": [1.0]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="missing GSHP parameter columns"):
            load_published_params(str(path))


class TestLayerCollapse:
    def test_one_row_per_layer_not_per_observation(self, tmp_path):
        rows = [_row("a"), _row("a"), _row("a"), _row("b")]
        got = load_published_params(_csv(tmp_path, rows))
        assert sorted(got["layer_id"]) == ["a", "b"]

    def test_depth_is_the_horizon_midpoint(self, tmp_path):
        got = load_published_params(_csv(tmp_path, [_row("a", top=10.0, bot=30.0)]))
        assert got.loc[0, "depth_cm"] == pytest.approx(20.0)


class TestProfileIdSanitization:
    def test_path_delimiters_are_replaced(self):
        assert sanitize_profile_id("a/b\\c") == "a_b_c"

    def test_applied_on_load(self, tmp_path):
        got = load_published_params(_csv(tmp_path, [_row("a", profile_id="x/y")]))
        assert got.loc[0, "profile_id"] == "x_y"


class TestFeedsSwrcDirectly:
    def test_arrays_evaluate_through_the_canonical_curve(self, tmp_path):
        got = load_published_params(_csv(tmp_path, [_row("a"), _row("b", n=2.1)]))
        tr, ts, al, n = to_swrc_arrays(got)
        theta = theta_from_psi(np.full(len(got), 100.0), tr, ts, al, n)
        assert np.all(np.isfinite(theta))
        assert np.all((theta > tr) & (theta < ts))
