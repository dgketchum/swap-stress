"""Tests for the dry-tail / drought-regime stress module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from map.evaluation import flux_drought_stress as fds


def _dates(n):
    return pd.date_range("2016-01-01", periods=n, freq="D").values


def test_const_beta_recovers_scale_and_beats_nothing_when_proportional():
    # flux = 0.7 * potential exactly → constant-β model is perfect → CV R² ≈ 1.
    rng = np.random.default_rng(0)
    n = 400
    pot = rng.uniform(1.0, 5.0, n)
    flux = 0.7 * pot
    skill = fds.const_beta_cv_skill(pot, flux, _dates(n))
    assert skill > 0.99


def test_const_beta_cannot_capture_stress_shape():
    # flux = potential * beta(theta): a soil-water logistic β beats constant-β,
    # because constant-β cannot bend the ratio down when theta is low.
    rng = np.random.default_rng(1)
    n = 500
    pot = rng.uniform(1.0, 5.0, n)
    theta = rng.uniform(0.05, 0.45, n)
    beta = 1.0 / (1.0 + np.exp(-30.0 * (theta - 0.25)))  # stress ramp
    flux = pot * beta + rng.normal(0, 0.02, n)
    dates = _dates(n)
    from map.evaluation import flux_beta_models as fbm

    cv_const = fds.const_beta_cv_skill(pot, flux, dates)
    cv_theta = fbm.beta_cv_skill(
        theta,
        pot,
        flux,
        dates,
        +1,
        n_blocks=fds.DS_N_BLOCKS,
        min_cv_days=fds.DS_MIN_DAYS,
    )
    assert cv_theta - cv_const > 0.1


def test_assign_regime_splits_at_site_median_both_polarities():
    sdf = pd.DataFrame(
        {"ppt_ant30": [1.0, 2.0, 3.0, 4.0], "wdef_ant30": [1.0, 2.0, 3.0, 4.0]}
    )
    # low precip = dry
    reg_p = fds.assign_regime(sdf, "ppt_ant30", higher_is_drier=False)
    assert list(reg_p) == ["dry", "dry", "wet", "wet"]
    # high deficit = dry
    reg_d = fds.assign_regime(sdf, "wdef_ant30", higher_is_drier=True)
    assert list(reg_d) == ["wet", "wet", "dry", "dry"]


def test_regime_skills_matched_rows_and_const_present():
    rng = np.random.default_rng(2)
    n = 300
    df = pd.DataFrame(
        {
            "date": _dates(n),
            "et0": rng.uniform(1.0, 5.0, n),
            "theta_l4_surf": rng.uniform(0.05, 0.45, n),
        }
    )
    df["theta_l4_root"] = df["theta_l4_surf"] + rng.normal(0, 0.01, n)
    df["rew_theta_l4_surf"] = df["theta_l4_surf"]
    df["suction_l4"] = -np.log10(np.clip(df["theta_l4_surf"], 0.01, None))
    beta = 1.0 / (1.0 + np.exp(-25.0 * (df["theta_l4_surf"] - 0.25)))
    df["et_corr"] = df["et0"] * beta + rng.normal(0, 0.02, n)
    preds = {k: v for k, v in fds.PREDICTORS.items()}
    out = fds.regime_skills(df, "et0", "et_corr", preds)
    assert out is not None
    assert "cv_const" in out and np.isfinite(out["cv_const"])
    assert "cv_theta" in out and "rng_theta" in out
    assert out["n_days"] == n
