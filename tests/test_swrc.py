"""Tests for the canonical van Genuchten retention curve.

The reference values in :func:`test_matches_published_ptf_baseline` were captured
from ``swapstress.validation.ptf_baseline.vg_suction`` before it was replaced. They lock
the numerics that produce the manuscript's Rosetta/POLARIS comparison. If a
change here makes that test fail, the published PTF metrics have moved and the
change is wrong unless the manuscript is being reworked deliberately.
"""

import numpy as np
import pytest

from swapstress.swrc import (
    SE_EPS,
    VanGenuchtenParams,
    log10_psi_from_theta,
    psi_from_theta,
    theta_from_psi,
    valid_params,
)

# A loam-ish reference soil used across the round-trip tests.
TR, TS, AL, N = 0.05, 0.45, 0.01, 1.5


class TestPublishedNumerics:
    """Regression locks against the pre-refactor implementations."""

    def test_matches_published_ptf_baseline(self):
        theta = np.array(
            [0.05, 0.10, 0.20, 0.30, 0.40, 0.4499, 0.45, 0.02, 0.0, 1.0, 0.25]
        )
        theta_r = np.array(
            [0.02, 0.02, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        )
        theta_s = np.full(11, 0.45)
        alpha = np.array(
            [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.145]
        )
        n = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2.68])

        expected = np.array(
            [
                20539.79301836379,
                2876.6460489715837,
                685.886070498941,
                212.42255817098717,
                62.382348857897654,
                0.8257570645768325,
                0.020800865965259197,
                99999999999999.45,
                99999999999999.45,
                0.020800865965259197,
                8.967778198255596,
            ]
        )
        got = psi_from_theta(theta, theta_r, theta_s, alpha, n)
        # Exact equality, not approximate: this is a bit-for-bit lock.
        assert np.array_equal(got, expected)

    def test_flux_variant_reproduced_via_se_eps(self):
        """The flux analyses used se_eps=1e-3 and a 0.01 cm floor."""
        theta = np.array([0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.45, 0.50])
        se = np.clip((theta - TR) / (TS - TR), 1e-3, 1 - 1e-3)
        m = 1.0 - 1.0 / N
        expected = np.log10(
            np.maximum((1.0 / AL) * (se ** (-1.0 / m) - 1.0) ** (1.0 / N), 0.01)
        )
        got = log10_psi_from_theta(theta, TR, TS, AL, N, se_eps=1e-3, psi_floor_cm=0.01)
        assert np.array_equal(got, expected)


class TestRoundTrip:
    def test_psi_theta_psi_across_range(self):
        """psi -> theta -> psi recovers the input across the usable range."""
        psi = np.logspace(0, 4.5, 60)  # 1 cm to ~30 m, spans FC and wilting
        theta = theta_from_psi(psi, TR, TS, AL, N)
        back = psi_from_theta(theta, TR, TS, AL, N)
        assert np.allclose(back, psi, rtol=1e-9)

    def test_theta_psi_theta_interior(self):
        """theta -> psi -> theta recovers theta away from the clipped ends."""
        theta = np.linspace(TR + 0.02, TS - 0.02, 40)
        psi = psi_from_theta(theta, TR, TS, AL, N)
        back = theta_from_psi(psi, TR, TS, AL, N)
        assert np.allclose(back, theta, rtol=1e-9)

    @pytest.mark.parametrize("n", [1.1, 1.5, 2.0, 2.68, 4.0])
    @pytest.mark.parametrize("alpha", [0.001, 0.01, 0.145])
    def test_round_trip_over_param_grid(self, n, alpha):
        """The inverse recovers psi wherever theta stays off the clipped ends.

        The usable range is parameter-dependent: for small alpha or large n,
        theta collapses onto theta_r well before 1e4 cm, and past that point psi
        is not recoverable from theta at all. So the assertion is made on the
        interior region, which is identified from theta rather than assumed.
        """
        psi = np.logspace(0, 4, 25)
        theta = theta_from_psi(psi, TR, TS, alpha, n)
        se = (theta - TR) / (TS - TR)
        interior = (se > 1e-4) & (se < 1.0 - 1e-4)

        assert interior.any(), "no interior points; the test grid says nothing"
        back = psi_from_theta(theta[interior], TR, TS, alpha, n)
        assert np.allclose(back, psi[interior], rtol=1e-7)

    @pytest.mark.parametrize("n", [1.5, 4.0])
    @pytest.mark.parametrize("alpha", [0.001, 0.145])
    def test_beyond_the_interior_the_guard_saturates(self, n, alpha):
        """Past the clip the inverse returns a constant, not the true psi.

        Clipping Se at a *floor* caps psi at a *ceiling*, so beyond the clip the
        inverse silently understates suction, and the shortfall grows without
        bound as the soil dries. That ceiling is parameter-dependent -- roughly
        690 cm at alpha=0.145, n=4, but 1e14 cm at alpha=0.01, n=1.5 -- so it
        cannot be documented as a single number.

        Pinned so this stays a visible saturation rather than drifting into an
        inf, a NaN, or a plausible-looking wrong number.
        """
        psi = np.logspace(0, 8, 60)
        theta = theta_from_psi(psi, TR, TS, alpha, n)
        se = (theta - TR) / (TS - TR)
        clipped = se <= SE_EPS
        if not clipped.any():
            pytest.skip("this parameter pair does not reach the clip on this grid")

        back = psi_from_theta(theta[clipped], TR, TS, alpha, n)
        assert np.all(np.isfinite(back))
        # Every clipped point collapses onto the same guard-derived ceiling.
        assert np.allclose(back, back[0], rtol=1e-6)
        # The ceiling never exceeds the true suction: it understates it.
        assert np.all(back <= psi[clipped] * (1.0 + 1e-9))
        # And at the dry end the understatement is severe, not marginal.
        assert psi[clipped][-1] / back[-1] > 100.0


class TestAsymptotes:
    def test_monotonic_decreasing_theta_with_psi(self):
        psi = np.logspace(-1, 6, 200)
        theta = theta_from_psi(psi, TR, TS, AL, N)
        assert np.all(np.diff(theta) < 0)

    def test_theta_bounded_by_residual_and_saturated(self):
        psi = np.logspace(-3, 8, 300)
        theta = theta_from_psi(psi, TR, TS, AL, N)
        assert np.all(theta <= TS + 1e-12)
        assert np.all(theta >= TR - 1e-12)

    def test_wet_end_approaches_saturation(self):
        assert theta_from_psi(0.0, TR, TS, AL, N) == pytest.approx(TS, abs=1e-9)

    def test_dry_end_hits_the_guard_not_infinity(self):
        """At theta <= theta_r the true inverse diverges; the clip makes it finite.

        This is the documented dry-end limitation, pinned so it cannot silently
        change into an inf or a NaN.
        """
        got = psi_from_theta(TR, TR, TS, AL, N)
        assert np.isfinite(got)
        assert got > 1e12  # absurd as a suction; it is the guard showing through

    def test_saturated_end_is_small_but_positive(self):
        got = psi_from_theta(TS, TR, TS, AL, N)
        assert np.isfinite(got)
        assert 0.0 < got < 1.0

    def test_se_eps_controls_the_dry_end_magnitude(self):
        loose = psi_from_theta(TR, TR, TS, AL, N, se_eps=1e-3)
        tight = psi_from_theta(TR, TR, TS, AL, N, se_eps=1e-9)
        assert tight > loose


class TestInvalidParameters:
    """Invalid inputs must yield NaN, never a patched or silently-filled value."""

    @pytest.mark.parametrize(
        "tr,ts,al,n",
        [
            (0.05, 0.45, 0.01, 1.0),  # n must be > 1
            (0.05, 0.45, 0.01, 0.9),  # n < 1
            (0.05, 0.45, 0.0, 1.5),  # alpha must be > 0
            (0.05, 0.45, -0.01, 1.5),  # negative alpha
            (0.45, 0.45, 0.01, 1.5),  # theta_s must exceed theta_r
            (0.50, 0.45, 0.01, 1.5),  # inverted
            (np.nan, 0.45, 0.01, 1.5),  # non-finite
            (0.05, 0.45, 0.01, np.nan),
        ],
    )
    def test_invalid_params_give_nan(self, tr, ts, al, n):
        assert np.isnan(psi_from_theta(0.25, tr, ts, al, n))
        assert np.isnan(theta_from_psi(100.0, tr, ts, al, n))
        assert not valid_params(tr, ts, al, n)

    def test_nan_theta_gives_nan(self):
        assert np.isnan(psi_from_theta(np.nan, TR, TS, AL, N))

    def test_valid_params_accepts_reasonable_soil(self):
        assert valid_params(TR, TS, AL, N)

    def test_mixed_validity_is_elementwise(self):
        n = np.array([1.5, 1.0, 2.0])
        got = psi_from_theta(0.25, TR, TS, AL, n)
        assert np.isfinite(got[0])
        assert np.isnan(got[1])
        assert np.isfinite(got[2])


class TestBroadcasting:
    def test_scalar_theta_array_params(self):
        """The pre-refactor implementation raised IndexError on this shape."""
        n = np.full(5, 1.5)
        got = psi_from_theta(0.25, TR, TS, AL, n)
        assert got.shape == (5,)
        assert np.all(np.isfinite(got))

    def test_array_theta_scalar_params(self):
        theta = np.linspace(0.10, 0.40, 7)
        got = psi_from_theta(theta, TR, TS, AL, N)
        assert got.shape == (7,)

    def test_both_arrays(self):
        theta = np.linspace(0.10, 0.40, 4)
        alpha = np.array([0.01, 0.02, 0.05, 0.145])
        got = psi_from_theta(theta, TR, TS, alpha, N)
        assert got.shape == (4,)


class TestVanGenuchtenParams:
    def test_m_is_mualem_constrained(self):
        p = VanGenuchtenParams(TR, TS, AL, 2.0)
        assert p.m == pytest.approx(0.5)

    def test_methods_match_module_functions(self):
        p = VanGenuchtenParams(TR, TS, AL, N)
        theta = np.linspace(0.10, 0.40, 5)
        assert np.array_equal(p.psi(theta), psi_from_theta(theta, TR, TS, AL, N))
        psi = np.logspace(1, 4, 5)
        assert np.array_equal(p.theta(psi), theta_from_psi(psi, TR, TS, AL, N))

    def test_from_mapping_plain_numbers(self):
        p = VanGenuchtenParams.from_mapping(
            {"theta_r": TR, "theta_s": TS, "alpha": AL, "n": N}
        )
        assert p.n == N
        assert p.depth_cm is None

    def test_from_mapping_lmfit_style_value_dicts(self):
        """Saved curve-fit JSONs store parameters as {"value": x, ...}."""
        p = VanGenuchtenParams.from_mapping(
            {
                "theta_r": {"value": TR, "stderr": 0.01},
                "theta_s": {"value": TS},
                "alpha": {"value": AL},
                "n": {"value": N},
                "depth_cm": 5,
            }
        )
        assert p.theta_r == TR
        assert p.depth_cm == 5.0

    def test_is_valid_flags_bad_params(self):
        assert VanGenuchtenParams(TR, TS, AL, N).is_valid
        assert not VanGenuchtenParams(TR, TS, AL, 1.0).is_valid

    def test_frozen(self):
        p = VanGenuchtenParams(TR, TS, AL, N)
        with pytest.raises(Exception):
            p.n = 2.0


class TestStrictFlag:
    """``strict=False`` exists only for curve-fitting residual models."""

    def test_default_is_strict(self):
        got = theta_from_psi(np.logspace(0, 4, 10), 0.30, 0.25, 0.05, 1.5)
        assert np.all(np.isnan(got))

    def test_permissive_returns_finite_for_inverted_theta(self):
        got = theta_from_psi(np.logspace(0, 4, 10), 0.30, 0.25, 0.05, 1.5, strict=False)
        assert np.all(np.isfinite(got))

    def test_permissive_still_rejects_n_le_1(self):
        """n <= 1 is not a search-space artifact; the equation has no m there."""
        got = theta_from_psi(np.logspace(0, 4, 10), TR, TS, AL, 1.0, strict=False)
        assert np.all(np.isnan(got))

    def test_identical_to_strict_when_params_are_valid(self):
        psi = np.logspace(-1, 6, 200)
        a = theta_from_psi(psi, TR, TS, AL, N)
        b = theta_from_psi(psi, TR, TS, AL, N, strict=False)
        assert np.array_equal(a, b)

    def test_strict_flag_does_not_touch_the_inverse(self):
        """psi_from_theta has no permissive mode; nothing fits on the inverse."""
        import inspect

        assert "strict" not in inspect.signature(psi_from_theta).parameters


class TestDefaults:
    def test_se_eps_default_is_the_published_one(self):
        assert SE_EPS == 1e-6

    def test_no_floor_applied_by_default_in_log10(self):
        """psi_floor_cm=None must not silently clamp."""
        theta = np.array([0.4499])
        unfloored = log10_psi_from_theta(theta, TR, TS, AL, N)
        floored = log10_psi_from_theta(theta, TR, TS, AL, N, psi_floor_cm=10.0)
        assert unfloored[0] < floored[0]
