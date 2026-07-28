"""Tests for the restored van Genuchten fitter.

The fitter was absent from the tree between 39946a5 and its restoration, so
these tests exist to pin the parts that were verified by refitting archived
sites: that the deterministic path recovers known parameters, that the residual
model evaluates permissively (the archived fits depend on it), and that the
pymc model -- the one place the vG equation is deliberately written twice --
agrees with the canonical numpy implementation.
"""

import numpy as np
import pandas as pd
import pytest

from swapstress.fitting import SWRC, VGFitResultLight
from swapstress.swrc import theta_from_psi

TR, TS, AL, N = 0.05, 0.45, 0.02, 1.6


def _synthetic(theta_r=TR, theta_s=TS, alpha=AL, n=N, npts=40, noise=0.0, seed=0):
    psi = np.logspace(0, 4.2, npts)
    theta = theta_from_psi(psi, theta_r, theta_s, alpha, n)
    if noise:
        theta = theta + np.random.default_rng(seed).normal(0, noise, theta.shape)
    return pd.DataFrame({"suction_cm": psi, "theta": theta, "depth_cm": 10.0})


class TestDeterministicFit:
    def test_recovers_known_parameters_from_clean_curve(self, capsys):
        s = SWRC(df=_synthetic())
        s.fit(method="nelder")
        capsys.readouterr()
        r = s.fit_results[10.0]
        assert r is not None and r.success
        assert r.params["theta_s"].value == pytest.approx(TS, abs=5e-3)
        assert r.params["alpha"].value == pytest.approx(AL, rel=0.15)
        assert r.params["n"].value == pytest.approx(N, rel=0.15)

    def test_tolerates_noise(self, capsys):
        s = SWRC(df=_synthetic(noise=0.005, seed=3))
        s.fit(method="nelder")
        capsys.readouterr()
        r = s.fit_results[10.0]
        assert r is not None and r.success
        assert r.params["theta_s"].value == pytest.approx(TS, abs=0.03)

    def test_default_method_is_a_real_lmfit_method(self, capsys):
        """The restored default was 'nelder-meade', which lmfit does not accept.

        Left unfixed, every fit taken with default arguments failed and was
        swallowed by the except branch into a None result.
        """
        s = SWRC(df=_synthetic())
        s.fit()
        capsys.readouterr()
        assert s.fit_results[10.0] is not None
        assert s.fit_results[10.0].success


class TestResidualModelIsPermissive:
    """The archived fits were produced against an unvalidated forward model."""

    def test_returns_finite_when_theta_s_below_theta_r(self):
        psi = np.logspace(0, 5, 20)
        got = SWRC._van_genuchten_model(psi, 0.30, 0.25, 0.05, 1.5)
        assert np.all(np.isfinite(got)), "NaN here aborts the fit outright"

    def test_strict_evaluation_rejects_the_same_parameters(self):
        psi = np.logspace(0, 5, 20)
        assert np.all(np.isnan(theta_from_psi(psi, 0.30, 0.25, 0.05, 1.5)))

    def test_matches_the_archived_permissive_form(self):
        """Bit-for-bit against the implementation the saved fits used."""

        def archived(psi, theta_r, theta_s, alpha, n):
            if n <= 1:
                return np.full_like(psi, np.nan)
            m = 1 - 1 / n
            psi_safe = np.maximum(psi, 1e-9)
            term = 1 + (alpha * psi_safe) ** n
            return theta_r + (theta_s - theta_r) / (term) ** m

        rng = np.random.default_rng(0)
        psi = np.logspace(-2, 6, 120)
        for _ in range(200):
            tr = rng.uniform(0.0, 0.5)
            ts = rng.uniform(0.0, 0.6)  # deliberately allows ts <= tr
            al = rng.uniform(1e-5, 5.0)
            n = rng.uniform(1.0005, 10.0)
            a = archived(psi, tr, ts, al, n)
            b = SWRC._van_genuchten_model(psi, tr, ts, al, n)
            assert np.array_equal(a, b)

    def test_still_nan_for_n_at_or_below_one(self):
        psi = np.logspace(0, 4, 10)
        assert np.all(np.isnan(SWRC._van_genuchten_model(psi, TR, TS, AL, 1.0)))


class TestPymcModelMatchesNumpy:
    """The pymc path rebuilds the equation in pytensor ops; pin the two."""

    def test_pytensor_forward_equals_canonical(self):
        pt = pytest.importorskip("pytensor.tensor")

        psi = np.logspace(-1, 5, 60)
        theta_r, theta_s, alpha, n = TR, TS, AL, N

        # Mirrors fit_bayesian's mu_theta construction exactly.
        psi_safe = pt.maximum(psi, 1e-9)
        m = 1.0 - 1.0 / n
        term = 1.0 + (alpha * psi_safe) ** n
        mu = theta_r + (theta_s - theta_r) / (term**m)
        got = mu.eval()

        expected = theta_from_psi(psi, theta_r, theta_s, alpha, n)
        assert np.allclose(got, expected, rtol=1e-12, atol=0)


class TestVGFitResultLight:
    def test_eval_matches_canonical(self):
        psi = np.logspace(0, 5, 30)
        light = VGFitResultLight(TR, TS, AL, N)
        assert np.allclose(light.eval(psi=psi), theta_from_psi(psi, TR, TS, AL, N))

    def test_exposes_param_values(self):
        light = VGFitResultLight(TR, TS, AL, N)
        assert light.success
        assert light.params["alpha"].value == AL
        assert light.params["n"].value == N


class TestResultsRoundTrip:
    def test_save_then_load_reconstructs_parameters(self, tmp_path, capsys):
        s = SWRC(df=_synthetic())
        s.fit(method="nelder")
        s.save_results(str(tmp_path), output_filename="fit.json")
        capsys.readouterr()

        fitted = {
            k: s.fit_results[10.0].params[k].value
            for k in ("theta_r", "theta_s", "alpha", "n")
        }

        back = SWRC().load_from_results_json(str(tmp_path / "fit.json"))
        r = back.fit_results[10]
        assert r is not None
        for k, v in fitted.items():
            assert r.params[k].value == pytest.approx(v)

    def test_loaded_data_survives(self, tmp_path, capsys):
        s = SWRC(df=_synthetic(npts=25))
        s.fit(method="nelder")
        s.save_results(str(tmp_path), output_filename="fit.json")
        capsys.readouterr()
        back = SWRC().load_from_results_json(str(tmp_path / "fit.json"))
        assert len(back.data_by_depth[10]) == 25
