"""Tests for soil water potential unit conversions.

The central claim these lock down: converting the model target
``log10(suction_cm)`` to MPa is an exact additive shift in log space, so no
metric expressed in log units changes under the conversion. That is why MPa is
handled as a postprocessing step rather than by retraining.
"""

import math

import numpy as np
import pytest

from swapstress.units import (
    LOG10_MPA_TO_CM,
    MPA_TO_CM,
    log10_abs_mpa_to_log10_suction_cm,
    log10_suction_cm_to_log10_abs_mpa,
    log10_suction_cm_to_mpa,
    mpa_to_suction_cm,
    suction_cm_to_mpa,
)


class TestConstants:
    def test_mpa_to_cm_matches_hydrostatic_definition(self):
        """1 MPa / (rho_w * g) expressed in cm."""
        expected = 1e6 / (1000.0 * 9.80665) * 100.0
        assert MPA_TO_CM == pytest.approx(expected, rel=1e-5)

    def test_log10_constant_is_derived_not_hardcoded(self):
        assert LOG10_MPA_TO_CM == math.log10(MPA_TO_CM)

    def test_log10_constant_value(self):
        assert LOG10_MPA_TO_CM == pytest.approx(4.0084792, abs=1e-7)


class TestSignConvention:
    def test_suction_to_mpa_is_negative(self):
        """Matric potential is signed negative; suction head is positive."""
        assert suction_cm_to_mpa(15300.0) < 0

    def test_wilting_point_is_about_minus_1p5_mpa(self):
        # 15300 cm is the conventional permanent wilting point.
        assert suction_cm_to_mpa(15300.0) == pytest.approx(-1.5, abs=0.01)

    def test_field_capacity_is_about_minus_33_kpa(self):
        # 336 cm is the conventional field capacity.
        assert suction_cm_to_mpa(336.0) == pytest.approx(-0.033, abs=0.001)

    def test_mpa_to_suction_returns_positive_magnitude(self):
        assert mpa_to_suction_cm(-1.5) > 0
        assert mpa_to_suction_cm(-1.5) == mpa_to_suction_cm(1.5)

    def test_log10_suction_to_mpa_is_negative(self):
        got = log10_suction_cm_to_mpa(np.array([2.0, 3.0, 4.18]))
        assert np.all(got < 0)


class TestExactLogShift:
    """The identity that makes MPa a free postprocessing step."""

    def test_shift_is_exactly_the_constant(self):
        log10_cm = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        got = log10_suction_cm_to_log10_abs_mpa(log10_cm)
        assert np.allclose(got, log10_cm - LOG10_MPA_TO_CM, rtol=0, atol=0)

    def test_differences_are_preserved(self):
        """Any metric built on differences in log space is invariant."""
        a = np.array([1.0, 2.5, 4.0, 5.5])
        b = np.array([1.3, 2.1, 4.4, 5.2])
        d_cm = a - b
        d_mpa = log10_suction_cm_to_log10_abs_mpa(
            a
        ) - log10_suction_cm_to_log10_abs_mpa(b)
        assert np.allclose(d_cm, d_mpa, rtol=0, atol=1e-12)

    def test_rmse_is_identical_in_either_unit(self):
        rng = np.random.default_rng(0)
        truth = rng.uniform(1.0, 6.0, 500)
        pred = truth + rng.normal(0, 0.65, 500)
        rmse_cm = np.sqrt(np.mean((pred - truth) ** 2))
        rmse_mpa = np.sqrt(
            np.mean(
                (
                    log10_suction_cm_to_log10_abs_mpa(pred)
                    - log10_suction_cm_to_log10_abs_mpa(truth)
                )
                ** 2
            )
        )
        assert rmse_mpa == pytest.approx(rmse_cm, rel=1e-12)

    def test_r2_is_identical_in_either_unit(self):
        rng = np.random.default_rng(1)
        truth = rng.uniform(1.0, 6.0, 500)
        pred = truth + rng.normal(0, 0.65, 500)

        def r2(y, yhat):
            ss_res = np.sum((y - yhat) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - ss_res / ss_tot

        r2_cm = r2(truth, pred)
        r2_mpa = r2(
            log10_suction_cm_to_log10_abs_mpa(truth),
            log10_suction_cm_to_log10_abs_mpa(pred),
        )
        assert r2_mpa == pytest.approx(r2_cm, rel=1e-12)

    def test_interval_width_is_identical(self):
        """QRF prediction-interval widths are differences, so also invariant."""
        q025 = np.array([1.2, 2.4, 3.9])
        q975 = np.array([2.0, 3.1, 4.8])
        w_cm = q975 - q025
        w_mpa = log10_suction_cm_to_log10_abs_mpa(
            q975
        ) - log10_suction_cm_to_log10_abs_mpa(q025)
        assert np.allclose(w_cm, w_mpa, rtol=0, atol=1e-12)


class TestRoundTrips:
    def test_suction_mpa_round_trip(self):
        cm = np.array([1.0, 336.0, 15300.0, 1e6])
        assert np.allclose(mpa_to_suction_cm(suction_cm_to_mpa(cm)), cm, rtol=1e-12)

    def test_log_shift_round_trip(self):
        log10_cm = np.array([0.0, 2.0, 4.18, 6.0])
        back = log10_abs_mpa_to_log10_suction_cm(
            log10_suction_cm_to_log10_abs_mpa(log10_cm)
        )
        assert np.allclose(back, log10_cm, rtol=0, atol=1e-12)

    def test_linear_and_log_paths_agree(self):
        """log10_suction_cm_to_mpa must equal the linear path via 10**x."""
        log10_cm = np.array([1.0, 2.5, 4.0, 5.5])
        via_log = log10_suction_cm_to_mpa(log10_cm)
        via_linear = suction_cm_to_mpa(np.power(10.0, log10_cm))
        assert np.allclose(via_log, via_linear, rtol=1e-12)

    def test_magnitude_consistency_between_representations(self):
        log10_cm = np.array([2.0, 4.18])
        signed = log10_suction_cm_to_mpa(log10_cm)
        log_mag = log10_suction_cm_to_log10_abs_mpa(log10_cm)
        assert np.allclose(np.log10(np.abs(signed)), log_mag, rtol=1e-12)
