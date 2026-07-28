"""
Unit tests for ``map/evaluation/flux_stats.py``.

These tests are the guarantee that the flux re-test machinery is *not* rigged
the way the old in-sample nested Wilcoxon was (see
``notes/flux_robustness_retest_plan.md`` §3). They cover:

(a) a pure-noise predictor → CV Δskill CI contains / goes below 0 and its
    block-permutation p is not significant;
(b) a true-signal predictor → CV Δskill > 0 and its permutation p is small;
(c) the nested partial-F matches an independent t-statistic reference;
(d) the embargo actually removes rows adjacent to each test block.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
from scipy import stats as sp_stats

from research.flux import flux_stats as fs


# ---------------------------------------------------------------------------
# Synthetic-data helpers
# ---------------------------------------------------------------------------


def _ar1(n, rng, phi=0.9, scale=1.0):
    """Autocorrelated (AR(1)) series — mimics daily met/soil-water dynamics."""
    x = np.zeros(n)
    eps = rng.normal(0.0, scale, n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def _consecutive_dates(n, start=date(2016, 1, 1)):
    return [start + timedelta(days=i) for i in range(n)]


def _make_site(n_days, signal_coef, seed):
    """One synthetic site: autocorrelated met + predictor, response driven by
    met plus (optionally) the predictor."""
    rng = np.random.default_rng(seed)
    dates = _consecutive_dates(n_days)
    met = _ar1(n_days, rng, phi=0.9)
    pred = _ar1(n_days, rng, phi=0.9)
    noise = rng.normal(0.0, 1.0, n_days)
    y = 1.0 * met + signal_coef * pred + 1.0 * noise
    return dates, met, pred, y


# ---------------------------------------------------------------------------
# (d) Embargo geometry
# ---------------------------------------------------------------------------


def test_embargo_removes_boundary_rows():
    n = 300
    dates = _consecutive_dates(n)
    ordinals = fs._to_day_ordinals(dates)
    embargo = 15
    folds = fs.blocked_fold_indices(dates, n_blocks=5, embargo_days=embargo)

    assert len(folds) == 5
    for train_idx, test_idx in folds:
        assert len(test_idx) > 0
        t_lo = ordinals[test_idx].min()
        t_hi = ordinals[test_idx].max()
        train_ord = ordinals[train_idx]
        # No training row may lie inside the embargo buffer around the block.
        in_buffer = (train_ord >= t_lo - embargo) & (train_ord <= t_hi + embargo)
        assert not in_buffer.any()
        # Test and train are disjoint.
        assert len(np.intersect1d(train_idx, test_idx)) == 0


def test_embargo_shrinks_training_set():
    n = 300
    dates = _consecutive_dates(n)
    no_embargo = fs.blocked_fold_indices(dates, n_blocks=5, embargo_days=0)
    with_embargo = fs.blocked_fold_indices(dates, n_blocks=5, embargo_days=20)
    # A middle fold must lose rows on both sides when the embargo is applied.
    mid = 2
    assert len(with_embargo[mid][0]) < len(no_embargo[mid][0])


# ---------------------------------------------------------------------------
# (c) Nested partial-F matches an independent reference
# ---------------------------------------------------------------------------


def _ref_partial_f_add_one(X_red, x_new, y):
    """Independent reference: for one added regressor the partial F equals the
    square of that regressor's t-statistic (and shares its p-value)."""
    X = np.column_stack([np.ones(len(y)), X_red, x_new])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    dof = n - k
    sigma2 = (resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(sigma2 * xtx_inv[-1, -1])
    t = beta[-1] / se
    p = 2 * sp_stats.t.sf(abs(t), dof)
    return t**2, p


def test_partial_f_matches_t_squared():
    rng = np.random.default_rng(3)
    n = 200
    X_red = rng.normal(size=(n, 2))
    x_new = rng.normal(size=n)
    y = X_red @ np.array([1.0, -0.5]) + 0.7 * x_new + rng.normal(size=n)

    X_full = np.column_stack([X_red, x_new])
    f_stat, p_val, df1, df2 = fs.partial_f_test(X_red, X_full, y)
    f_ref, p_ref = _ref_partial_f_add_one(X_red, x_new, y)

    assert df1 == 1
    assert df2 == n - 3 - 1
    assert np.isclose(f_stat, f_ref, rtol=1e-6)
    assert np.isclose(p_val, p_ref, rtol=1e-6)


def test_partial_f_multi_regressor_against_scipy_ftest():
    """Two added regressors: verify against the F distribution directly using an
    independently computed SSR ratio."""
    rng = np.random.default_rng(7)
    n = 250
    X_red = rng.normal(size=(n, 2))
    X_extra = rng.normal(size=(n, 2))
    y = (
        X_red @ np.array([0.8, 0.3])
        + X_extra @ np.array([0.4, -0.6])
        + rng.normal(size=n)
    )
    X_full = np.column_stack([X_red, X_extra])

    f_stat, p_val, df1, df2 = fs.partial_f_test(X_red, X_full, y)

    # Independent SSR computation.
    def ssr(X):
        Xd = np.column_stack([np.ones(n), X])
        b, _, _, _ = np.linalg.lstsq(Xd, y, rcond=None)
        r = y - Xd @ b
        return r @ r

    ssr_red, ssr_full = ssr(X_red), ssr(X_full)
    f_ref = ((ssr_red - ssr_full) / 2) / (ssr_full / (n - 4 - 1))
    p_ref = sp_stats.f.sf(f_ref, 2, n - 4 - 1)
    assert df1 == 2
    assert np.isclose(f_stat, f_ref, rtol=1e-6)
    assert np.isclose(p_val, p_ref, rtol=1e-6)


def test_partial_f_degenerate_returns_nan():
    y = np.arange(10.0)
    X_red = np.ones((10, 1))
    # A perfectly collinear added column should yield a non-significant / finite
    # result (adding a duplicate of an existing column adds no information).
    f_stat, p_val, df1, df2 = fs.partial_f_test(
        X_red, np.column_stack([X_red, X_red[:, 0]]), y
    )
    # Adding a duplicate of an existing column adds no information.
    assert np.isnan(f_stat) or f_stat == 0.0


# ---------------------------------------------------------------------------
# Blocked-CV guards
# ---------------------------------------------------------------------------


def test_blocked_cv_short_record_returns_nan():
    dates, met, pred, y = _make_site(120, signal_coef=1.0, seed=1)
    X = np.column_stack([met, pred])
    r2 = fs.blocked_cv_r2(X, y, dates, min_cv_days=150)
    assert np.isnan(r2)


def test_blocked_cv_signal_positive_skill():
    dates, met, pred, y = _make_site(400, signal_coef=1.5, seed=2)
    X = np.column_stack([met, pred])
    r2 = fs.blocked_cv_r2(X, y, dates)
    assert np.isfinite(r2)
    assert r2 > 0.2  # a real linear signal must yield real OOS skill


# ---------------------------------------------------------------------------
# (a)/(b) CV Δskill: noise straddles 0, signal is positive
# ---------------------------------------------------------------------------


def _per_site_cv_deltas(signal_coef, n_sites=40, n_days=300, seed0=100):
    deltas = []
    for s in range(n_sites):
        dates, met, pred, y = _make_site(n_days, signal_coef, seed=seed0 + s)
        res = fs.blocked_cv_delta(
            X_reduced=met.reshape(-1, 1),
            X_full=np.column_stack([met, pred]),
            y=y,
            dates=dates,
        )
        deltas.append(res["delta"])
    return np.array(deltas)


def test_cv_delta_noise_ci_includes_zero():
    deltas = _per_site_cv_deltas(signal_coef=0.0)
    point, lo, hi = fs.site_bootstrap_ci(deltas, n_boot=2000, seed=0)
    assert np.isfinite(point)
    # A useless predictor must not produce a CI that excludes 0 in its favour.
    assert lo < 0.0


def test_cv_delta_signal_ci_excludes_zero():
    deltas = _per_site_cv_deltas(signal_coef=1.5)
    point, lo, hi = fs.site_bootstrap_ci(deltas, n_boot=2000, seed=0)
    assert point > 0.0
    assert lo > 0.0  # a real signal's median gain is significantly positive


def _insample_delta_statistic_fn(met, y):
    """Return a statistic_fn(pred) -> in-sample ΔR² of adding pred beyond met."""
    r2_met = 1.0 - fs._in_sample_ssr(met.reshape(-1, 1), y) / (
        ((y - y.mean()) ** 2).sum()
    )

    def stat(pred):
        X_full = np.column_stack([met, pred])
        r2_full = 1.0 - fs._in_sample_ssr(X_full, y) / (((y - y.mean()) ** 2).sum())
        return r2_full - r2_met

    return stat, r2_met


def test_permutation_signal_is_significant():
    dates, met, pred, y = _make_site(400, signal_coef=1.5, seed=11)
    stat_fn, _ = _insample_delta_statistic_fn(met, y)
    observed = stat_fn(pred)
    null = fs.block_permutation_null(stat_fn, pred, block_len=30, n_perm=300, seed=0)
    p = fs.permutation_p_value(observed, null, alternative="greater")
    assert observed > np.nanmean(null)
    assert p < 0.01


def test_permutation_noise_not_significant():
    dates, met, pred, y = _make_site(400, signal_coef=0.0, seed=12)
    stat_fn, _ = _insample_delta_statistic_fn(met, y)
    observed = stat_fn(pred)
    null = fs.block_permutation_null(stat_fn, pred, block_len=30, n_perm=300, seed=0)
    p = fs.permutation_p_value(observed, null, alternative="greater")
    # Observed in-sample gain should sit inside the autocorrelation noise floor.
    assert p > 0.05


def test_permutation_null_mean_is_positive_floor():
    """The in-sample ΔR² floor is > 0 even for noise — the exact artifact the
    old test mistook for signal."""
    dates, met, pred, y = _make_site(300, signal_coef=0.0, seed=13)
    stat_fn, _ = _insample_delta_statistic_fn(met, y)
    null = fs.block_permutation_null(stat_fn, pred, block_len=30, n_perm=300, seed=1)
    assert np.nanmean(null) > 0.0


# ---------------------------------------------------------------------------
# LOYO
# ---------------------------------------------------------------------------


def test_loyo_needs_three_years():
    # Two years only.
    dates = _consecutive_dates(400, start=date(2016, 1, 1))
    rng = np.random.default_rng(21)
    met = _ar1(400, rng)
    y = met + rng.normal(size=400)
    assert np.isnan(fs.loyo_r2(met.reshape(-1, 1), y, dates, min_years=3))


def test_loyo_signal_positive():
    dates = _consecutive_dates(1200, start=date(2016, 1, 1))  # >3 years
    rng = np.random.default_rng(22)
    met = _ar1(1200, rng)
    pred = _ar1(1200, rng)
    y = met + 1.5 * pred + rng.normal(size=1200)
    r2 = fs.loyo_r2(np.column_stack([met, pred]), y, dates, min_years=3)
    assert np.isfinite(r2)
    assert r2 > 0.2


# ---------------------------------------------------------------------------
# Benjamini–Hochberg FDR
# ---------------------------------------------------------------------------


def test_bh_classic_example():
    # Benjamini & Hochberg (1995) worked example: 15 hypotheses, alpha 0.05,
    # rejects the 4 smallest.
    pvals = np.array(
        [
            0.0001,
            0.0004,
            0.0019,
            0.0095,
            0.0201,
            0.0278,
            0.0298,
            0.0344,
            0.0459,
            0.3240,
            0.4262,
            0.5719,
            0.6528,
            0.7590,
            1.000,
        ]
    )
    rejected, adjusted = fs.benjamini_hochberg(pvals, alpha=0.05)
    assert rejected.sum() == 4
    assert rejected[:4].all()
    assert not rejected[4:].any()
    # Adjusted p-values are monotone non-decreasing in the original p order here.
    assert np.all(np.diff(adjusted) >= -1e-12)


def test_bh_handles_nan():
    pvals = np.array([0.001, np.nan, 0.9, 0.002])
    rejected, adjusted = fs.benjamini_hochberg(pvals, alpha=0.05)
    assert np.isnan(adjusted[1])
    assert not rejected[1]
    assert rejected[0] and rejected[3]


def test_bh_all_null_rejects_none():
    pvals = np.array([0.6, 0.7, 0.8, 0.95])
    rejected, _ = fs.benjamini_hochberg(pvals, alpha=0.05)
    assert not rejected.any()


# ---------------------------------------------------------------------------
# Sign / binomial tests
# ---------------------------------------------------------------------------


def test_sign_test_all_positive():
    res = fs.sign_test(np.array([0.1, 0.2, 0.05, 0.3, 0.15]), alternative="greater")
    assert res["frac_pos"] == 1.0
    assert res["p"] < 0.05


def test_binomial_rate_test_above_chance():
    # 20/50 sites significant, vs a 5% chance rate — clearly above chance.
    p = fs.binomial_rate_test(20, 50, rate=0.05, alternative="greater")
    assert p < 1e-6


def test_binomial_rate_test_at_chance():
    p = fs.binomial_rate_test(3, 50, rate=0.05, alternative="greater")
    assert p > 0.05
