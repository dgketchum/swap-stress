"""
Unit tests for ``map/evaluation/flux_beta_models.py``.

These synthetic-truth tests are the guarantee that the multiplicative stress-
function comparison (``notes/flux_stress_function_experiment_plan.md`` §5) is not
accidentally rigged toward one predictor:

(a) truth is ψ-driven with *texture-varying* θ→ψ curves → a single global β(ψ)
    transfers to unseen sites better than a global β(θ);
(b) truth is θ-driven (universal in θ) → β(θ) transfers better;
(c) the flux-space NLS recovers a known logistic (x0, k, β_max);
(d) on heteroscedastic ET0, fitting on the flux (this module) recovers β_max
    with less error than fitting on the ratio flux/ET0.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from map.evaluation import flux_beta_models as fbm
from swapstress.swrc import log10_psi_from_theta


# ---------------------------------------------------------------------------
# Synthetic-data helpers
# ---------------------------------------------------------------------------


def _consecutive_dates(n, start=date(2016, 1, 1)):
    return [start + timedelta(days=i) for i in range(n)]


def _vg_invert(theta, theta_r, theta_s, alpha, n_vg):
    """θ → ψ (log10 cm), matching quartile_binned_mlr._vg_invert.

    Only generates synthetic θ→ψ maps for the fixtures below; nothing here is
    asserted against, so delegating keeps the "matching" claim true by
    construction rather than by copy.
    """
    return log10_psi_from_theta(
        theta, theta_r, theta_s, alpha, n_vg, se_eps=1e-3, psi_floor_cm=0.01
    )


def _logistic(x_eff, x0, k, bmax):
    return bmax / (1.0 + np.exp(-k * (x_eff - x0)))


def _make_multisite(rng, truth, n_sites=12, n_days=200):
    """Build a multi-site frame with texture-varying θ→ψ maps.

    ``truth='psi'`` → flux = et0·β(ψ) with a *universal* β in ψ.
    ``truth='theta'`` → flux = et0·β(θ) with a *universal* β in θ.
    In both cases ψ = VG(θ) with a per-site (texture) VG curve, so the two
    predictors carry the same information within a site but map to the universal
    stress response differently across textures.
    """
    theta_r, theta_s = 0.05, 0.45
    alphas = np.linspace(0.006, 0.05, n_sites)
    ns = np.linspace(1.25, 2.4, n_sites)
    rows = []
    for s in range(n_sites):
        theta = rng.uniform(theta_r + 0.02, theta_s - 0.02, n_days)
        psi = _vg_invert(theta, theta_r, theta_s, alphas[s], ns[s])
        et0 = rng.uniform(2.0, 6.0, n_days)
        if truth == "psi":
            beta = _logistic(-psi, x0=-3.0, k=3.0, bmax=1.0)
        else:
            beta = _logistic(theta, x0=0.22, k=30.0, bmax=1.0)
        flux = et0 * beta + rng.normal(0.0, 0.05, n_days)
        for i in range(n_days):
            rows.append(
                {
                    "site_id": f"S{s:02d}",
                    "date": _consecutive_dates(n_days)[i],
                    "theta": theta[i],
                    "suction": psi[i],
                    "et0": et0[i],
                    "flux": flux[i],
                }
            )
    import pandas as pd

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# (a)/(b) transfer discriminates the true driver
# ---------------------------------------------------------------------------


def _median_transfer(frame, predictors):
    skills = fbm.loso_beta_transfer(
        frame, "et0", "flux", predictors, family="logistic", min_site_days=60
    )
    common = set.intersection(*[set(v.keys()) for v in skills.values()])
    out = {}
    for label, d in skills.items():
        vals = np.array([d[s] for s in common])
        out[label] = float(np.median(vals))
    return out, sorted(common)


def test_transfer_favors_psi_when_truth_is_psi():
    rng = np.random.default_rng(0)
    frame = _make_multisite(rng, truth="psi")
    med, common = _median_transfer(frame, {"psi": "suction", "theta": "theta"})
    assert len(common) >= 8
    # ψ is universal across textures; a global β(θ) must average incompatible
    # texture maps → ψ transfers strictly better.
    assert med["psi"] > med["theta"]


def test_transfer_favors_theta_when_truth_is_theta():
    rng = np.random.default_rng(1)
    frame = _make_multisite(rng, truth="theta")
    med, common = _median_transfer(frame, {"psi": "suction", "theta": "theta"})
    assert len(common) >= 8
    assert med["theta"] > med["psi"]


# ---------------------------------------------------------------------------
# (c) NLS recovers a known logistic
# ---------------------------------------------------------------------------


def test_nls_recovers_known_logistic():
    rng = np.random.default_rng(2)
    n = 800
    x = rng.uniform(0.0, 1.0, n)  # polarity +1 predictor
    et0 = rng.uniform(2.0, 6.0, n)
    x0_true, k_true, bmax_true = 0.45, 12.0, 1.1
    flux = et0 * _logistic(x, x0_true, k_true, bmax_true) + rng.normal(0, 0.01, n)
    params = fbm.fit_beta_params(x, et0, flux, "logistic")
    assert params is not None
    x0, k, bmax = params
    assert abs(x0 - x0_true) < 0.05
    assert abs(k - k_true) < 3.0
    assert abs(bmax - bmax_true) < 0.1


# ---------------------------------------------------------------------------
# (d) flux-space fit beats ratio fit under heteroscedastic ET0
# ---------------------------------------------------------------------------


def _ratio_fit_bmax(x, et0, flux):
    """Naive β fit on the ratio flux/et0 (unweighted) — the pathology §4.3 warns
    against. Returns recovered β_max."""
    from scipy import optimize

    ratio = flux / et0

    def resid(p):
        x0, k, bmax = p
        return ratio - fbm._beta_logistic(x, x0, k, bmax)

    res = optimize.least_squares(
        resid,
        [0.45, 8.0, 1.0],
        bounds=([0, 1e-3, 1e-3], [1, 100, fbm.BETA_MAX_CAP]),
        max_nfev=4000,
    )
    return res.x[2]


def test_flux_fit_beats_ratio_fit_under_heteroscedastic_et0():
    rng = np.random.default_rng(3)
    n = 1500
    x = rng.uniform(0.0, 1.0, n)
    # Wide ET0 dynamic range; additive flux noise → the RATIO noise blows up at
    # low ET0, biasing a ratio fit but not a flux-space fit.
    et0 = rng.uniform(0.3, 8.0, n)
    x0_true, k_true, bmax_true = 0.5, 10.0, 1.0
    flux = et0 * _logistic(x, x0_true, k_true, bmax_true) + rng.normal(0, 0.15, n)

    params = fbm.fit_beta_params(x, et0, flux, "logistic")
    assert params is not None
    bmax_flux = params[2]
    bmax_ratio = _ratio_fit_bmax(x, et0, flux)

    err_flux = abs(bmax_flux - bmax_true)
    err_ratio = abs(bmax_ratio - bmax_true)
    assert err_flux < err_ratio


# ---------------------------------------------------------------------------
# GPP potential envelope sanity
# ---------------------------------------------------------------------------


def test_gpp_potential_is_upper_envelope():
    import pandas as pd

    rng = np.random.default_rng(4)
    n = 400
    sw = rng.uniform(100, 400, n)
    # GPP proportional to light but stress-suppressed by a hidden factor in [0,1].
    stress = rng.uniform(0.2, 1.0, n)
    gpp = 0.02 * sw * stress
    sdf = pd.DataFrame({"sw_in": sw, "t_avg": rng.uniform(10, 30, n), "gpp": gpp})
    pot = fbm.gpp_potential(sdf, ["sw_in", "t_avg"], tau=0.90)
    assert pot is not None
    # The τ=0.9 envelope should sit at/above most observed GPP and stay positive.
    assert np.all(pot > 0)
    assert np.mean(gpp <= pot + 1e-9) > 0.8


# ---------------------------------------------------------------------------
# CV skill: signal beats a scrambled predictor
# ---------------------------------------------------------------------------


def test_cv_skill_positive_for_real_signal():
    rng = np.random.default_rng(5)
    n = 220
    dates = _consecutive_dates(n)
    x = rng.uniform(0.0, 1.0, n)
    et0 = rng.uniform(2.0, 6.0, n)
    flux = et0 * _logistic(x, 0.5, 10.0, 1.0) + rng.normal(0, 0.03, n)
    skill = fbm.beta_cv_skill(x, et0, flux, dates, polarity=1, family="logistic")
    assert np.isfinite(skill)
    assert skill > 0.5
    # A shuffled predictor carries no phase information → near-zero/negative skill.
    x_shuf = rng.permutation(x)
    skill_shuf = fbm.beta_cv_skill(
        x_shuf, et0, flux, dates, polarity=1, family="logistic"
    )
    assert (not np.isfinite(skill_shuf)) or skill_shuf < skill
