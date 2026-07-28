"""WS6 / E3 — exploratory threshold (piecewise-linear stress) models.

Water limitation on ET/GPP is classically a *stress ramp*: flux tracks
evaporative demand until soil water falls below a critical point, then declines
roughly linearly to a lower bound. This module asks, out of sample and at
matched complexity, two exploratory questions:

- **E3a (nonlinearity):** does a threshold (piecewise-linear) soil-water term
  out-predict a plain linear soil-water term? i.e. is there a detectable
  breakpoint at all, once we stop scoring in-sample?
- **E3b (θ vs ψ threshold):** does a *suction* (ψ) threshold model out-predict a
  *water-content* (θ) threshold model? If matric potential is the mechanistically
  correct stress variable, a ψ threshold should transfer better across the range
  than a θ threshold.

Model (one soil-water variable ``s``, matched complexity):

    y = a + b · et0 + c · min(s, τ)

``min(s, τ)`` rises with wetness up to the breakpoint τ then saturates — the
canonical stress-ramp shape (``c`` free, so OLS picks the sign). The breakpoint
τ is chosen *within each training fold* by a grid search over training-set
quantiles (so it never sees the held-out block — no leakage). Skill is the
blocked-CV out-of-sample R² vs the *training* mean, on the same folds and the
same embargo as the H3/H5 machinery. ψ is already stored in pF (log10 |ψ|), so a
threshold on it is on a physically sensible scale.

This is **exploratory (M5)** — reported alongside the confirmatory H1–H5, never
used to make a PASS/FAIL claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.flux import flux_cv_analysis as fca
from research.flux import flux_stats as fs

# Match the confirmatory CV configuration exactly.
N_BLOCKS = fca.N_BLOCKS
EMBARGO_DAYS = fca.EMBARGO_DAYS
MIN_CV_DAYS = fca.MIN_CV_DAYS
N_TAU_GRID = 15
TAU_QLO, TAU_QHI = 0.15, 0.85

# (response_label, response_col, theta_col, psi_col, depth_label)
STRESS_CONFIGS = [
    ("ET", "et_corr", "theta_l4_surf", "suction_l4", "L4 surface"),
    ("ET", "et_corr", "theta_l4_root", "suction_l4_root_50", "L4 root50"),
    ("GPP", "gpp", "theta_l4_surf", "suction_l4", "L4 surface"),
    ("GPP", "gpp", "theta_l4_root", "suction_l4_root_50", "L4 root50"),
]


def _ols_fit(X: np.ndarray, y: np.ndarray):
    """OLS with intercept via lstsq. Returns beta or None if degenerate."""
    Xd = np.column_stack([np.ones(len(X)), X])
    if Xd.shape[0] < Xd.shape[1] + 1:
        return None
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return beta


def _predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X]) @ beta


def _threshold_cv_skill(s: np.ndarray, et0: np.ndarray, y: np.ndarray, dates) -> float:
    """Blocked-CV skill of ``y = a + b·et0 + c·min(s, τ)`` with τ selected on
    each training fold. Skill is vs the training mean; NaN unless every
    requested fold is usable (mirrors :func:`flux_stats.blocked_cv_metrics`)."""
    if len(y) < MIN_CV_DAYS:
        return np.nan
    folds = fs.blocked_fold_indices(dates, n_blocks=N_BLOCKS, embargo_days=EMBARGO_DAYS)
    ss_res = 0.0
    ss_tot = 0.0
    n_folds_used = 0
    min_train = 3 + 2  # intercept + et0 + hinge, plus slack
    for tr, te in folds:
        if len(te) == 0 or len(tr) < min_train:
            continue
        s_tr = s[tr]
        taus = np.quantile(s_tr, np.linspace(TAU_QLO, TAU_QHI, N_TAU_GRID))
        taus = np.unique(taus)
        best_ssr = np.inf
        best = None
        for tau in taus:
            X_tr = np.column_stack([et0[tr], np.minimum(s_tr, tau)])
            beta = _ols_fit(X_tr, y[tr])
            if beta is None:
                continue
            resid = y[tr] - _predict(X_tr, beta)
            ssr = float(resid @ resid)
            if ssr < best_ssr:
                best_ssr = ssr
                best = (tau, beta)
        if best is None:
            continue
        tau, beta = best
        X_te = np.column_stack([et0[te], np.minimum(s[te], tau)])
        y_hat = _predict(X_te, beta)
        resid = y[te] - y_hat
        ss_res += float(resid @ resid)
        base = y[te] - float(np.mean(y[tr]))
        ss_tot += float(base @ base)
        n_folds_used += 1
    if n_folds_used < N_BLOCKS or ss_tot < 1e-12:
        return np.nan
    return 1.0 - ss_res / ss_tot


def per_site_stress(
    sdf: pd.DataFrame, theta_col: str, psi_col: str, y_col: str
) -> dict | None:
    """Linear vs threshold CV skill for θ and ψ at one site (identical folds)."""
    clean = sdf.dropna(subset=["et0", theta_col, psi_col, y_col])
    if len(clean) < MIN_CV_DAYS:
        return None
    et0 = clean["et0"].values.astype(np.float64)
    theta = clean[theta_col].values.astype(np.float64)
    psi = clean[psi_col].values.astype(np.float64)
    y = clean[y_col].values.astype(np.float64)
    dates = clean["date"].values

    lin_theta = fca._cv_skill(np.column_stack([et0, theta]), y, dates)
    lin_psi = fca._cv_skill(np.column_stack([et0, psi]), y, dates)
    thr_theta = _threshold_cv_skill(theta, et0, y, dates)
    thr_psi = _threshold_cv_skill(psi, et0, y, dates)
    if not np.isfinite(thr_theta) or not np.isfinite(thr_psi):
        return None
    return {
        "n_days": len(clean),
        "lin_theta": lin_theta,
        "lin_psi": lin_psi,
        "thr_theta": thr_theta,
        "thr_psi": thr_psi,
        # E3a nonlinearity gain (threshold over linear).
        "nonlin_gain_theta": thr_theta - lin_theta,
        "nonlin_gain_psi": thr_psi - lin_psi,
        # E3b θ vs ψ at matched (threshold) complexity.
        "thr_psi_minus_theta": thr_psi - thr_theta,
    }


def run_threshold_stress(daily: pd.DataFrame, configs=STRESS_CONFIGS) -> pd.DataFrame:
    """Aggregate E3 threshold-stress results across the configs."""
    gs = fca.prep_growing_season(daily)
    rows = []
    for resp_name, resp_col, theta_col, psi_col, depth_label in configs:
        if resp_col not in gs.columns:
            continue
        sub = gs.dropna(subset=[resp_col, theta_col, psi_col])
        stats = []
        for _, sdf in sub.groupby("site_id"):
            r = per_site_stress(sdf, theta_col, psi_col, resp_col)
            if r is not None:
                stats.append(r)
        if not stats:
            continue
        S = pd.DataFrame(stats)
        nl_med, nl_lo, nl_hi = fs.site_bootstrap_ci(S["nonlin_gain_theta"].values)
        pt_med, pt_lo, pt_hi = fs.site_bootstrap_ci(S["thr_psi_minus_theta"].values)
        rows.append(
            {
                "response": resp_name,
                "depth": depth_label,
                "n_sites": len(S),
                "med_n_days": int(S["n_days"].median()),
                "skill_lin_theta_med": float(S["lin_theta"].median()),
                "skill_thr_theta_med": float(S["thr_theta"].median()),
                "skill_thr_psi_med": float(S["thr_psi"].median()),
                # E3a: does the threshold beat linear (θ)?
                "nonlin_gain_theta_med": nl_med,
                "nonlin_gain_theta_lo": nl_lo,
                "nonlin_gain_theta_hi": nl_hi,
                "nonlin_theta_sign_p": fs.sign_test(
                    S["nonlin_gain_theta"].values, alternative="greater"
                )["p"],
                # E3b: does the ψ threshold beat the θ threshold?
                "thr_psi_minus_theta_med": pt_med,
                "thr_psi_minus_theta_lo": pt_lo,
                "thr_psi_minus_theta_hi": pt_hi,
                "thr_psi_vs_theta_sign_p": fs.sign_test(
                    S["thr_psi_minus_theta"].values, alternative="greater"
                )["p"],
                "ci_excludes_zero_nonlin": bool(np.isfinite(nl_lo) and nl_lo > 0),
                "ci_excludes_zero_psi_vs_theta": bool(np.isfinite(pt_lo) and pt_lo > 0),
            }
        )
    return pd.DataFrame(rows)
