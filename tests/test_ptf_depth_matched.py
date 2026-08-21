"""Tests for the depth-matched POLARIS PTF rerun.

Per ``notes/polaris_depth_matched_handoff.md``: the layer-selection rules,
strict inverse-domain status, independent per-layer alpha conversion, and
reproduction of the legacy fixed-0-5 baseline are all pure logic that needs
no live Earth Engine session to verify.
"""

import numpy as np
import pandas as pd
import pytest

from swapstress.units import KPA_TO_CM
from swapstress.validation.ptf_depth_matched import (
    _assemble_depth_matched,
    _parse_polaris_layer_results,
    classify_domain_status,
    select_layer_nearest_midpoint,
    select_layer_primary,
    select_layer_upper_at_boundary,
)


class TestLayerSelectionRules:
    """Reproduces the exact counts verified in the handoff against the
    5,244-row candidate set's depth_cm distribution."""

    DEPTH_COUNTS = {
        2.5: 449,
        3.0: 55,
        3.5: 16,
        4.0: 151,
        4.5: 23,
        5.0: 2122,
        5.5: 20,
        6.0: 47,
        6.5: 503,
        7.0: 71,
        7.5: 1276,
        8.0: 60,
        8.5: 23,
        9.0: 378,
        9.5: 50,
    }

    @pytest.fixture
    def depth_cm(self):
        depths = []
        for d, n in self.DEPTH_COUNTS.items():
            depths.extend([d] * n)
        return np.array(depths)

    def test_primary_rule_counts(self, depth_cm):
        layer = select_layer_primary(depth_cm)
        assert (layer == "0_5").sum() == 2816
        assert (layer == "5_15").sum() == 2428

    def test_primary_rule_boundary(self):
        layer = select_layer_primary(np.array([4.9, 5.0, 5.1]))
        assert list(layer) == ["0_5", "0_5", "5_15"]

    def test_upper_at_boundary_counts(self, depth_cm):
        layer = select_layer_upper_at_boundary(depth_cm)
        assert (layer == "0_5").sum() == 694
        assert (layer == "5_15").sum() == 4550

    def test_upper_at_boundary_boundary(self):
        layer = select_layer_upper_at_boundary(np.array([4.9, 5.0, 5.1]))
        assert list(layer) == ["0_5", "5_15", "5_15"]

    def test_nearest_midpoint_counts(self, depth_cm):
        layer = select_layer_nearest_midpoint(depth_cm)
        assert (layer == "0_5").sum() == 2883
        assert (layer == "5_15").sum() == 2361

    def test_nearest_midpoint_boundary(self):
        layer = select_layer_nearest_midpoint(np.array([6.0, 6.25, 6.5]))
        assert list(layer) == ["0_5", "5_15", "5_15"]

    def test_every_row_gets_exactly_one_layer(self, depth_cm):
        for fn in (
            select_layer_primary,
            select_layer_upper_at_boundary,
            select_layer_nearest_midpoint,
        ):
            layer = fn(depth_cm)
            assert set(np.unique(layer)) <= {"0_5", "5_15"}
            assert len(layer) == len(depth_cm)


class TestClassifyDomainStatus:
    def test_missing_parameters(self):
        status = classify_domain_status([0.2], [np.nan], [0.4], [0.02], [1.5])
        assert status[0] == "missing_parameters"

    def test_invalid_parameters_theta_s_le_theta_r(self):
        status = classify_domain_status([0.2], [0.4], [0.4], [0.02], [1.5])
        assert status[0] == "invalid_parameters"

    def test_invalid_parameters_n_le_one(self):
        status = classify_domain_status([0.2], [0.05], [0.4], [0.02], [1.0])
        assert status[0] == "invalid_parameters"

    def test_invalid_parameters_alpha_le_zero(self):
        status = classify_domain_status([0.2], [0.05], [0.4], [0.0], [1.5])
        assert status[0] == "invalid_parameters"

    def test_below_or_equal_theta_r(self):
        status = classify_domain_status([0.05], [0.05], [0.4], [0.02], [1.5])
        assert status[0] == "below_or_equal_theta_r"

    def test_above_or_equal_theta_s(self):
        status = classify_domain_status([0.4], [0.05], [0.4], [0.02], [1.5])
        assert status[0] == "above_or_equal_theta_s"

    def test_in_domain(self):
        status = classify_domain_status([0.2], [0.05], [0.4], [0.02], [1.5])
        assert status[0] == "in_domain"

    def test_every_row_gets_exactly_one_status(self):
        rng = np.random.default_rng(0)
        n = 200
        theta = rng.uniform(0.0, 0.6, n)
        theta_r = rng.uniform(0.0, 0.3, n)
        theta_s = rng.uniform(0.0, 0.6, n)
        alpha = rng.uniform(-0.05, 0.2, n)
        vgn = rng.uniform(0.5, 3.0, n)
        # Sprinkle in some missing parameters.
        theta_r[:10] = np.nan
        status = classify_domain_status(theta, theta_r, theta_s, alpha, vgn)
        assert (status == "").sum() == 0


class TestParsePolarisLayerResults:
    """Independent alpha conversion and unit handling for both layers."""

    def test_independent_alpha_conversion(self):
        results = [
            {
                "sample_id": "a",
                "theta_r_0_5": 0.05,
                "theta_s_0_5": 0.4,
                "alpha_0_5": -1.0,
                "n_0_5": 1.4,
                "theta_r_5_15": 0.06,
                "theta_s_5_15": 0.38,
                "alpha_5_15": -0.5,
                "n_5_15": 1.6,
            }
        ]
        out = _parse_polaris_layer_results(results, ["a"])
        assert out["pol_0_5_alpha"][0] == pytest.approx(10.0**-1.0 / KPA_TO_CM)
        assert out["pol_5_15_alpha"][0] == pytest.approx(10.0**-0.5 / KPA_TO_CM)
        # The two layers' conversions do not leak into each other.
        assert out["pol_0_5_alpha"][0] != out["pol_5_15_alpha"][0]
        assert out["pol_0_5_theta_r"][0] == 0.05
        assert out["pol_5_15_theta_r"][0] == 0.06
        assert out["pol_0_5_n"][0] == 1.4
        assert out["pol_5_15_n"][0] == 1.6

    def test_missing_sample_stays_nan(self):
        out = _parse_polaris_layer_results([], ["a", "b"])
        for layer in ("0_5", "5_15"):
            for p in ("theta_r", "theta_s", "alpha", "n"):
                assert np.isnan(out[f"pol_{layer}_{p}"]).all()

    def test_partial_layer_missing(self):
        """A site inside the 0-5 raster but off the 5-15 raster edge."""
        results = [
            {
                "sample_id": "a",
                "theta_r_0_5": 0.05,
                "theta_s_0_5": 0.4,
                "alpha_0_5": -1.0,
                "n_0_5": 1.4,
            }
        ]
        out = _parse_polaris_layer_results(results, ["a"])
        assert out["pol_0_5_theta_r"][0] == 0.05
        assert np.isnan(out["pol_5_15_theta_r"][0])


class TestAssembleDepthMatched:
    @pytest.fixture
    def candidates(self):
        return pd.DataFrame(
            {
                "sample_id": ["s1", "s2", "s3", "s4"],
                "source": ["gshp"] * 4,
                "theta": [0.20, 0.20, 0.20, 0.20],
                "log10_suction_cm": [2.0, 2.0, 2.0, 2.0],
                "depth_cm": [2.5, 5.0, 7.5, 9.5],
                "rf_pred": [2.1, 2.1, 2.1, 2.1],
                "ros_theta_r": [0.05, 0.05, 0.05, 0.05],
                "ros_theta_s": [0.4, 0.4, 0.4, 0.4],
                "ros_alpha": [0.02, 0.02, 0.02, 0.02],
                "ros_n": [1.5, 1.5, 1.5, 1.5],
                "ros_log10_suction": [1.9, 1.9, 1.9, 1.9],
            }
        )

    @pytest.fixture
    def site(self):
        # s1..s4 share the same site parameters for simplicity; 0-5 and 5-15
        # are given different values so the join is verifiably layer-specific.
        return pd.DataFrame(
            {
                "sample_id": ["s1", "s2", "s3", "s4"],
                "pol_0_5_theta_r": [0.05] * 4,
                "pol_0_5_theta_s": [0.40] * 4,
                "pol_0_5_alpha": [0.02] * 4,
                "pol_0_5_n": [1.4] * 4,
                "pol_0_5_alpha_units": ["cm^-1"] * 4,
                "pol_5_15_theta_r": [0.08] * 4,
                "pol_5_15_theta_s": [0.35] * 4,
                "pol_5_15_alpha": [0.03] * 4,
                "pol_5_15_n": [1.6] * 4,
                "pol_5_15_alpha_units": ["cm^-1"] * 4,
            }
        )

    def test_primary_layer_assignment(self, candidates, site):
        df = _assemble_depth_matched(candidates, site)
        assert list(df["polaris_layer"]) == ["0_5", "0_5", "5_15", "5_15"]

    def test_selected_params_match_assigned_layer(self, candidates, site):
        df = _assemble_depth_matched(candidates, site)
        # s1 (2.5 cm) selects 0-5; s3 (7.5 cm) selects 5-15.
        assert df.loc[df["sample_id"] == "s1", "pol_selected_theta_r"].iloc[0] == 0.05
        assert df.loc[df["sample_id"] == "s3", "pol_selected_theta_r"].iloc[0] == 0.08

    def test_one_and_only_one_layer_per_row(self, candidates, site):
        df = _assemble_depth_matched(candidates, site)
        for rule in ("primary", "upper_at_boundary", "nearest_midpoint"):
            assert df[f"polaris_layer_{rule}"].isin(["0_5", "5_15"]).all()

    def test_legacy_control_uses_only_0_5(self, candidates, site):
        """The legacy-control columns must never see the 5-15 values, even
        for rows the primary rule routes to 5-15."""
        df = _assemble_depth_matched(candidates, site)
        assert (df["polaris_legacy_0_5_domain_status"] == "in_domain").all()
        # All four rows share identical theta/params at 0-5, so the legacy
        # suction is identical for all rows regardless of depth_cm.
        assert df["polaris_legacy_0_5_log10_suction"].nunique() == 1

    def test_rosetta_untouched(self, candidates, site):
        df = _assemble_depth_matched(candidates, site)
        pd.testing.assert_series_equal(
            df["ros_log10_suction"], candidates["ros_log10_suction"], check_names=False
        )

    def test_domain_status_present_for_every_row(self, candidates, site):
        df = _assemble_depth_matched(candidates, site)
        assert (df["polaris_domain_status"] != "").all()
        assert (df["rosetta_domain_status"] != "").all()

    def test_missing_site_params_yield_missing_parameters_status(
        self, candidates, site
    ):
        site = site.copy()
        site.loc[site["sample_id"] == "s1", "pol_0_5_alpha"] = np.nan
        df = _assemble_depth_matched(candidates, site)
        row = df.loc[df["sample_id"] == "s1"].iloc[0]
        assert row["polaris_domain_status"] == "missing_parameters"
        assert np.isnan(row["pol_depth_matched_log10_suction"])
