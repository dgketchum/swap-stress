"""
Unit tests for ``map/evaluation/flux_plant_potential.py``.

Synthetic-truth guards that the product-of-sigmoids multiplicative β (the
supply×demand / fusion form of ``notes/flux_plant_potential_plan.md`` §4) is
correctly specified and not accidentally rigged:

(a) the flux-space NLS recovers a known single-factor logistic β;
(b) a two-factor product β recovers a known supply×demand truth and the fusion
    (adding a third informative factor) raises out-of-sample CV skill, while a
    pure-noise third factor does not;
(c) CV skill is positive for real signal and collapses for a scrambled predictor;
(d) a wrong-polarity monotone β cannot manufacture skill (the fairness property
    the ΔVOD sign choice relies on);
(e) the seasonal-harmonic anomaly removes an injected annual cycle.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from research.flux import flux_plant_potential as fpp


def _dates(n, start=date(2016, 1, 1)):
    return [start + timedelta(days=i) for i in range(n)]


def _logistic(x_eff, x0, k, bmax=1.0):
    return bmax / (1.0 + np.exp(-k * (x_eff - x0)))


# ---------------------------------------------------------------------------
# (a) single-factor NLS recovers a known logistic
# ---------------------------------------------------------------------------


def test_product_beta_recovers_single_logistic():
    rng = np.random.default_rng(0)
    n = 800
    x = rng.uniform(0.0, 1.0, n)
    et0 = rng.uniform(2.0, 6.0, n)
    x0_t, k_t, bmax_t = 0.45, 12.0, 1.1
    flux = et0 * _logistic(x, x0_t, k_t, bmax_t) + rng.normal(0, 0.01, n)
    params = fpp.fit_product_beta([x], et0, flux)
    assert params is not None
    x0, k, bmax = params
    assert abs(x0 - x0_t) < 0.05
    assert abs(k - k_t) < 3.0
    assert abs(bmax - bmax_t) < 0.1


# ---------------------------------------------------------------------------
# (b) two-factor supply×demand recovery + fusion value
# ---------------------------------------------------------------------------


def _make_supply_demand(rng, n, with_plant=False):
    supply = rng.uniform(0.0, 1.0, n)  # REW-like, polarity +1
    demand = rng.uniform(0.5, 4.0, n)  # VPD-like, polarity −1
    et0 = rng.uniform(2.0, 6.0, n)
    beta = _logistic(supply, 0.4, 10.0) * _logistic(-demand, -2.0, 3.0)
    plant = None
    if with_plant:
        plant = rng.uniform(0.0, 1.0, n)  # extra informative factor, polarity +1
        beta = beta * _logistic(plant, 0.5, 8.0)
    flux = et0 * beta + rng.normal(0, 0.02, n)
    return supply, demand, plant, et0, flux


def test_two_factor_product_beats_single_when_demand_matters():
    rng = np.random.default_rng(1)
    n = 400
    supply, demand, _, et0, flux = _make_supply_demand(rng, n)
    dts = _dates(n)
    sk_supply = fpp.product_beta_cv_skill([supply], [1], et0, flux, dts)
    sk_both = fpp.product_beta_cv_skill([supply, demand], [1, -1], et0, flux, dts)
    assert np.isfinite(sk_both)
    assert sk_both > sk_supply  # the demand axis carries real out-of-sample skill


def test_fusion_adds_for_informative_plant_not_for_noise():
    rng = np.random.default_rng(2)
    n = 500
    supply, demand, plant, et0, flux = _make_supply_demand(rng, n, with_plant=True)
    dts = _dates(n)
    base = fpp.product_beta_cv_skill([supply, demand], [1, -1], et0, flux, dts)
    fused = fpp.product_beta_cv_skill(
        [supply, demand, plant], [1, -1, 1], et0, flux, dts
    )
    noise = rng.uniform(0.0, 1.0, n)
    fused_noise = fpp.product_beta_cv_skill(
        [supply, demand, noise], [1, -1, 1], et0, flux, dts
    )
    assert fused > base  # a truly informative plant factor lifts CV skill
    # A noise third factor must not out-perform the informative fusion.
    assert fused > fused_noise


# ---------------------------------------------------------------------------
# (c) CV skill: signal beats a scrambled predictor
# ---------------------------------------------------------------------------


def test_cv_skill_positive_for_real_signal():
    rng = np.random.default_rng(3)
    n = 220
    dts = _dates(n)
    x = rng.uniform(0.0, 1.0, n)
    et0 = rng.uniform(2.0, 6.0, n)
    flux = et0 * _logistic(x, 0.5, 10.0) + rng.normal(0, 0.03, n)
    skill = fpp.product_beta_cv_skill([x], [1], et0, flux, dts)
    assert np.isfinite(skill) and skill > 0.5
    x_shuf = rng.permutation(x)
    skill_shuf = fpp.product_beta_cv_skill([x_shuf], [1], et0, flux, dts)
    assert (not np.isfinite(skill_shuf)) or skill_shuf < skill


# ---------------------------------------------------------------------------
# (d) wrong polarity cannot manufacture skill (fairness property)
# ---------------------------------------------------------------------------


def test_wrong_polarity_does_not_inflate_skill():
    rng = np.random.default_rng(4)
    n = 260
    dts = _dates(n)
    x = rng.uniform(0.0, 1.0, n)  # true polarity +1
    et0 = rng.uniform(2.0, 6.0, n)
    flux = et0 * _logistic(x, 0.5, 10.0) + rng.normal(0, 0.03, n)
    right = fpp.product_beta_cv_skill([x], [1], et0, flux, dts)
    wrong = fpp.product_beta_cv_skill([x], [-1], et0, flux, dts)
    assert right > 0.4
    # A monotone β fit with the wrong sign yields much lower (≈0 or negative) skill.
    assert (not np.isfinite(wrong)) or wrong < right - 0.2


# ---------------------------------------------------------------------------
# (e) seasonal-harmonic anomaly removes an injected annual cycle
# ---------------------------------------------------------------------------


def test_harmonic_anomaly_removes_seasonal_cycle():
    rng = np.random.default_rng(5)
    doy = np.arange(1, 366, dtype=np.float64)
    ang = 2.0 * np.pi * doy / 365.25
    seasonal = 1.2 + 0.6 * np.sin(ang) + 0.3 * np.cos(ang)
    # A short-lived deficit event superimposed on the seasonal cycle.
    event = np.where((doy > 180) & (doy < 200), -0.4, 0.0)
    values = seasonal + event + rng.normal(0, 0.02, len(doy))
    anom = fpp._harmonic_anomaly(doy, values)
    # The seasonal component is removed → mean anomaly ≈ 0 and its std ≪ raw std.
    assert abs(np.nanmean(anom)) < 0.05
    assert np.nanstd(anom) < np.nanstd(values)
    # The deficit event survives as a clearly negative anomaly.
    assert np.nanmean(anom[(doy > 180) & (doy < 200)]) < -0.2
