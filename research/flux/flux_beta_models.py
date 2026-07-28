"""
Multiplicative water-stress functions β(x) for the flux stress-function experiment.

See ``notes/flux_stress_function_experiment_plan.md``. Every prior flux test put
soil water in an *additive* model (``flux ~ met + soil``), which structurally
handicaps matric potential: additive OLS just rescales θ internally to mimic ψ,
and a linear term cannot express a stress threshold. Plant water stress is
physically **multiplicative** — it scales down the flux the atmosphere would
otherwise drive:

    ET  = ET0           · β(x)
    GPP = GPP_potential · β(x)

with β a dimensionless stress factor in ``[0, β_max]``. This module fits β(x) for
any soil-water predictor ``x`` (ψ_RF, ψ_PTF, θ, REW) at **matched complexity** —
every family has the same free-parameter count, so a contrast isolates the
*predictor*, not model flexibility.

Design commitments (plan §4, §10):
- **Fit on the flux, not the ratio.** Minimize ``Σ (flux − potential·β(x))²``, not
  the heteroscedastic ratio ``flux/potential`` (§4.3). The ratio is a plotting
  axis only.
- **Matched families.** ``logistic`` and ``feddes`` each have two shape params
  plus a shared nuisance scale ``β_max`` (lets ET/ET0 exceed 1 from advection /
  envelope slack rather than forcing a hard cap). ``isotonic`` is a shape-free
  monotone robustness check (E3), reported alongside — never for a PASS/FAIL.
- **Polarity.** β rises with plant-available water. θ and REW increase with
  wetness (polarity ``+1``); suction ψ (stored as pF = log10|ψ|) increases with
  *dryness* (polarity ``-1``). Internally we fit on ``x_eff = polarity · x`` so a
  single monotone-increasing β family serves every predictor.
- **Out of sample.** Skill is blocked-CV R² vs the *training* mean on the exact
  fold structure (embargo, min-days, all-folds-required) used by the confirmatory
  H1–H5 machinery in :mod:`map.evaluation.flux_stats` / ``flux_cv_analysis``.

No I/O, no GPU, no Earth Engine — pure CPU statistics over cached arrays.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize

from map.evaluation import flux_cv_analysis as fca
from map.evaluation import flux_stats as fs

# Match the confirmatory CV configuration exactly.
N_BLOCKS = fca.N_BLOCKS
EMBARGO_DAYS = fca.EMBARGO_DAYS
MIN_CV_DAYS = fca.MIN_CV_DAYS

# Fit / guard defaults.
BETA_MAX_CAP = 3.0  # ET/ET0 or GPP/GPP_pot can exceed 1; cap generously.
MIN_RANGE_FRAC = 0.10  # predictor must span ≥ this fraction of its plausible range
MAX_NFEV = 4000


# ---------------------------------------------------------------------------
# β families (each: β_max nuisance scale + 2 shape params, matched complexity)
# ---------------------------------------------------------------------------


def _beta_logistic(
    x_eff: np.ndarray, x0: float, k: float, beta_max: float
) -> np.ndarray:
    """Monotone-increasing logistic in ``x_eff``: onset ``x0``, slope ``k`` > 0.

    ``β(x_eff) = β_max / (1 + exp(-k·(x_eff − x0)))``. Because we fit on
    ``x_eff = polarity·x``, the same increasing form covers ψ (falls with
    suction) and θ/REW (rise with wetness).
    """
    z = -k * (x_eff - x0)
    z = np.clip(z, -60.0, 60.0)  # overflow guard
    return beta_max / (1.0 + np.exp(z))


def _beta_feddes(
    x_eff: np.ndarray, x_wilt: float, width: float, beta_max: float
) -> np.ndarray:
    """FAO-56 / Feddes piecewise-linear stress ramp (two breakpoints).

    β climbs linearly from 0 at the wilting breakpoint ``x_wilt`` to ``β_max`` at
    the critical breakpoint ``x_crit = x_wilt + width`` (``width`` > 0 fit in log
    space by the caller to keep the ordering), then saturates.
    """
    x_crit = x_wilt + width
    frac = (x_eff - x_wilt) / max(x_crit - x_wilt, 1e-9)
    return beta_max * np.clip(frac, 0.0, 1.0)


# family registry: name -> (func, n_shape_params, param_names)
_FAMILY_FUNC = {
    "logistic": _beta_logistic,
    "feddes": _beta_feddes,
}
FAMILIES = ("logistic", "feddes", "isotonic")


# ---------------------------------------------------------------------------
# Polarity
# ---------------------------------------------------------------------------


def predictor_polarity(col: str) -> int:
    """+1 if β increases with the predictor (θ, REW), -1 for suction (ψ, pF).

    Suction columns store log10|ψ| (pF): larger = drier = lower flux, so β
    decreases with the raw value → polarity -1. Everything else (θ, REW) rises
    with plant-available water → +1.
    """
    name = col.lower()
    if name.startswith("suction") or name.startswith("psi") or "_psi" in name:
        return -1
    return 1


# ---------------------------------------------------------------------------
# Parametric fit (weighted, robust-optional NLS on the FLUX)
# ---------------------------------------------------------------------------


def _init_and_bounds(family: str, x_eff: np.ndarray, ratio: np.ndarray):
    """Initial guess + bounds for a family, from the training data statistics."""
    lo_x, hi_x = np.percentile(x_eff, [2, 98])
    span = max(hi_x - lo_x, 1e-6)
    bmax0 = float(np.clip(np.percentile(ratio, 90), 0.1, BETA_MAX_CAP))
    x_min, x_max = float(x_eff.min()), float(x_eff.max())
    if family == "logistic":
        x0_0 = float(np.median(x_eff))
        k0 = 4.0 / span
        p0 = [x0_0, k0, bmax0]
        lb = [x_min - span, 1e-4, 1e-3]
        ub = [x_max + span, 200.0 / span, BETA_MAX_CAP]
        return np.array(p0), (np.array(lb), np.array(ub))
    if family == "feddes":
        # Fit (x_wilt, log_width, beta_max) so width = exp(log_width) > 0.
        x_wilt0 = float(np.percentile(x_eff, 10))
        width0 = 0.5 * span
        p0 = [x_wilt0, np.log(width0), bmax0]
        lb = [x_min - span, np.log(1e-3 * span), 1e-3]
        ub = [x_max, np.log(3.0 * span), BETA_MAX_CAP]
        return np.array(p0), (np.array(lb), np.array(ub))
    raise ValueError(f"unknown parametric family {family!r}")


def _unpack_feddes(params: np.ndarray):
    x_wilt, log_width, beta_max = params
    return x_wilt, float(np.exp(log_width)), beta_max


def _beta_from_params(family: str, x_eff: np.ndarray, params: np.ndarray) -> np.ndarray:
    if family == "logistic":
        return _beta_logistic(x_eff, *params)
    if family == "feddes":
        x_wilt, width, beta_max = _unpack_feddes(params)
        return _beta_feddes(x_eff, x_wilt, width, beta_max)
    raise ValueError(f"unknown parametric family {family!r}")


def fit_beta_params(
    x_eff: np.ndarray,
    potential: np.ndarray,
    flux: np.ndarray,
    family: str,
    weights: np.ndarray | None = None,
    robust: bool = False,
) -> np.ndarray | None:
    """Weighted NLS of ``flux ≈ potential · β(x_eff)`` (fit on the flux, §4.3).

    Returns the raw parameter vector (family-specific packing) or None if the fit
    is degenerate / non-convergent. ``robust`` switches to a soft-L1 loss to blunt
    ET0 outliers.
    """
    x_eff = np.asarray(x_eff, dtype=np.float64)
    potential = np.asarray(potential, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    if weights is None:
        w = np.ones_like(flux)
    else:
        w = np.asarray(weights, dtype=np.float64)
    sw = np.sqrt(np.maximum(w, 0.0))

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(potential > 1e-9, flux / potential, np.nan)
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size < 5 or x_eff.size < 5:
        return None

    p0, (lb, ub) = _init_and_bounds(family, x_eff, ratio)

    def resid(params):
        beta = _beta_from_params(family, x_eff, params)
        return sw * (flux - potential * beta)

    try:
        res = optimize.least_squares(
            resid,
            p0,
            bounds=(lb, ub),
            loss="soft_l1" if robust else "linear",
            max_nfev=MAX_NFEV,
            method="trf",
        )
    except Exception:
        return None
    if not res.success and res.status <= 0:
        return None
    if not np.all(np.isfinite(res.x)):
        return None
    return res.x


def predict_beta(family: str, x_eff: np.ndarray, params) -> np.ndarray:
    """β(x_eff) for a fitted parametric family."""
    return _beta_from_params(family, x_eff, np.asarray(params, dtype=np.float64))


def beta_params_readable(family: str, params, polarity: int) -> dict:
    """Human-readable fitted params (mapped back to native predictor units)."""
    p = np.asarray(params, dtype=np.float64)
    if family == "logistic":
        x0, k, bmax = p
        return {
            "beta_max": float(bmax),
            "x0_native": float(polarity * x0),
            "slope_k": float(k),
        }
    if family == "feddes":
        x_wilt, width, bmax = _unpack_feddes(p)
        x_crit = x_wilt + width
        return {
            "beta_max": float(bmax),
            "x_wilt_native": float(polarity * x_wilt),
            "x_crit_native": float(polarity * x_crit),
        }
    return {}


# ---------------------------------------------------------------------------
# Isotonic (shape-free monotone) β — E3 robustness only
# ---------------------------------------------------------------------------


def _fit_isotonic(x_eff, potential, flux):
    """Weighted isotonic β̂(x_eff) increasing, weights ∝ potential² (matches the
    flux-space NLS objective as closely as a monotone fit allows). Returns a
    fitted ``IsotonicRegression`` or None."""
    from sklearn.isotonic import IsotonicRegression

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(potential > 1e-9, flux / potential, np.nan)
    ok = np.isfinite(ratio) & np.isfinite(x_eff)
    if ok.sum() < 5:
        return None
    w = np.clip(potential[ok] ** 2, 1e-9, None)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.0)
    try:
        iso.fit(x_eff[ok], np.clip(ratio[ok], 0.0, BETA_MAX_CAP), sample_weight=w)
    except Exception:
        return None
    return iso


# ---------------------------------------------------------------------------
# Skill (apparent + blocked CV) for one predictor
# ---------------------------------------------------------------------------


def _flux_skill(flux_test, flux_hat, base_mean) -> tuple[float, float]:
    """Return (ss_res, ss_tot) of a multiplicative-model prediction vs a baseline
    (training mean of the flux). Kept separate so folds can be pooled."""
    resid = flux_test - flux_hat
    ss_res = float(resid @ resid)
    base = flux_test - base_mean
    ss_tot = float(base @ base)
    return ss_res, ss_tot


def _predict_flux_fold(family, x_eff_tr, pot_tr, flux_tr, x_eff_te, pot_te, robust):
    """Fit β on a training fold, return predicted flux on the test fold (or None)."""
    if family == "isotonic":
        iso = _fit_isotonic(x_eff_tr, pot_tr, flux_tr)
        if iso is None:
            return None
        beta_te = np.clip(iso.predict(x_eff_te), 0.0, BETA_MAX_CAP)
        return pot_te * beta_te
    params = fit_beta_params(x_eff_tr, pot_tr, flux_tr, family, robust=robust)
    if params is None:
        return None
    beta_te = predict_beta(family, x_eff_te, params)
    return pot_te * beta_te


def beta_cv_skill(
    x: np.ndarray,
    potential: np.ndarray,
    flux: np.ndarray,
    dates,
    polarity: int,
    family: str = "logistic",
    robust: bool = False,
    n_blocks: int = N_BLOCKS,
    embargo_days: int = EMBARGO_DAYS,
    min_cv_days: int = MIN_CV_DAYS,
) -> float:
    """Blocked-CV out-of-sample skill of ``flux = potential·β(x)`` (§5).

    Skill is ``1 − SS_res/SS_tot`` pooled over folds, with SS_tot vs the *training*
    mean of the flux (so skill can be negative). NaN unless every requested fold is
    usable — mirrors :func:`flux_stats.blocked_cv_metrics` and
    ``flux_stress_models._threshold_cv_skill`` so the numbers are comparable. The
    fold settings are relaxable (the in-situ ceiling case has short records).
    """
    x = np.asarray(x, dtype=np.float64)
    potential = np.asarray(potential, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    if len(flux) < min_cv_days:
        return np.nan
    x_eff = polarity * x
    folds = fs.blocked_fold_indices(dates, n_blocks=n_blocks, embargo_days=embargo_days)
    min_train = 6  # 3 params + slack
    ss_res = 0.0
    ss_tot = 0.0
    n_used = 0
    for tr, te in folds:
        if len(te) == 0 or len(tr) < min_train:
            continue
        flux_hat = _predict_flux_fold(
            family,
            x_eff[tr],
            potential[tr],
            flux[tr],
            x_eff[te],
            potential[te],
            robust,
        )
        if flux_hat is None or not np.all(np.isfinite(flux_hat)):
            continue
        r, t = _flux_skill(flux[te], flux_hat, float(np.mean(flux[tr])))
        ss_res += r
        ss_tot += t
        n_used += 1
    if n_used < n_blocks or ss_tot < 1e-12:
        return np.nan
    return 1.0 - ss_res / ss_tot


def beta_apparent_skill(
    x: np.ndarray,
    potential: np.ndarray,
    flux: np.ndarray,
    polarity: int,
    family: str = "logistic",
    robust: bool = False,
) -> float:
    """In-sample (apparent) R² of the multiplicative model — the overfitting
    reference reported next to the CV skill."""
    x = np.asarray(x, dtype=np.float64)
    potential = np.asarray(potential, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    x_eff = polarity * x
    if family == "isotonic":
        iso = _fit_isotonic(x_eff, potential, flux)
        if iso is None:
            return np.nan
        flux_hat = potential * np.clip(iso.predict(x_eff), 0.0, BETA_MAX_CAP)
    else:
        params = fit_beta_params(x_eff, potential, flux, family, robust=robust)
        if params is None:
            return np.nan
        flux_hat = potential * predict_beta(family, x_eff, params)
    ss_res = float(((flux - flux_hat) ** 2).sum())
    ss_tot = float(((flux - flux.mean()) ** 2).sum())
    if ss_tot < 1e-12:
        return np.nan
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# GPP potential — per-site quantile-regression upper envelope (§2.2)
# ---------------------------------------------------------------------------


def quantile_regression(X: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray | None:
    """τ-quantile linear regression via the standard LP (no statsmodels dep).

    Minimizes the pinball loss ``Σ ρ_τ(y − Xβ)`` with an intercept column. Solved
    as an LP in variables ``[β⁺, β⁻, u, v]`` with ``u,v ≥ 0`` the positive/negative
    residual parts. Returns the coefficient vector (intercept first) or None.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < 10:
        return None
    Xd = np.column_stack([np.ones(n), X])
    p = Xd.shape[1]
    # Variables: beta_pos (p), beta_neg (p), u (n), v (n) with u,v the positive
    # and negative parts of the residual r = y − Xβ (so u − v = r). Pinball cost
    # τ·u + (1−τ)·v: for τ→1 under-prediction (u) is expensive → upper envelope.
    c = np.concatenate([np.zeros(2 * p), tau * np.ones(n), (1.0 - tau) * np.ones(n)])
    # Equality: Xd(bp−bn) + u − v = y.
    A_eq = np.hstack([Xd, -Xd, np.eye(n), -np.eye(n)])
    bounds = [(None, None)] * (2 * p) + [(0, None)] * (2 * n)
    # beta split into non-negative parts needs bounds >=0 too:
    bounds[: 2 * p] = [(0, None)] * (2 * p)
    try:
        res = optimize.linprog(c, A_eq=A_eq, b_eq=y, bounds=bounds, method="highs")
    except Exception:
        return None
    if not res.success:
        return None
    beta = res.x[:p] - res.x[p : 2 * p]
    return beta


def gpp_potential(
    sdf: pd.DataFrame,
    met_cols: list[str],
    tau: float = 0.90,
    floor_q: float = 0.05,
) -> np.ndarray | None:
    """Per-site unstressed GPP envelope: the τ-quantile of GPP given met (§2.2).

    The upper envelope of GPP on ``sw_in`` (+ ``t_avg``, ``vpd`` where present)
    approximates near-unstressed conditions at each met level without pre-labeling
    well-watered days (which would be circular with the REW predictor). Returns
    per-row GPP_pot (floored at a small positive quantile of observed GPP) or None.
    """
    cols = [c for c in met_cols if c in sdf.columns and sdf[c].notna().any()]
    if not cols or "gpp" not in sdf.columns:
        return None
    sub = sdf.dropna(subset=["gpp"] + cols)
    if len(sub) < 30:
        return None
    X = sub[cols].values.astype(np.float64)
    y = sub["gpp"].values.astype(np.float64)
    beta = quantile_regression(X, y, tau)
    if beta is None:
        return None
    Xall = sdf[cols].values.astype(np.float64)
    ok = np.all(np.isfinite(Xall), axis=1)
    pot = np.full(len(sdf), np.nan)
    pot[ok] = beta[0] + Xall[ok] @ beta[1:]
    floor = float(np.nanquantile(y, floor_q))
    floor = max(
        floor, 0.05 * float(np.nanmedian(y)) if np.isfinite(np.nanmedian(y)) else 0.0
    )
    pot = np.where(np.isfinite(pot), np.maximum(pot, floor), np.nan)
    return pot


# ---------------------------------------------------------------------------
# Per-site contrast: CV skill for many predictors on identical folds
# ---------------------------------------------------------------------------


def per_site_beta_skills(
    sdf: pd.DataFrame,
    potential_col: str,
    flux_col: str,
    predictors: dict[str, str],
    family: str = "logistic",
    robust: bool = False,
    n_blocks: int = N_BLOCKS,
    min_cv_days: int = MIN_CV_DAYS,
) -> dict | None:
    """CV + apparent β skill for each predictor at one site, on identical rows.

    ``predictors`` maps a label -> column name. All predictors are evaluated on the
    row set where the response, potential and *every* predictor are present, so a
    contrast Δskill between two predictors is matched row-for-row. Returns a dict
    of ``cv_<label>`` / ``app_<label>`` skills (+ ``n_days``) or None if the site is
    too short after the common-row intersection.
    """
    need = [potential_col, flux_col, *predictors.values()]
    clean = sdf.dropna(subset=need)
    if len(clean) < min_cv_days:
        return None
    pot = clean[potential_col].values.astype(np.float64)
    flux = clean[flux_col].values.astype(np.float64)
    dates = clean["date"].values
    out = {"n_days": len(clean)}
    got_any = False
    for label, col in predictors.items():
        x = clean[col].values.astype(np.float64)
        if x.std() < 1e-12:
            out[f"cv_{label}"] = np.nan
            out[f"app_{label}"] = np.nan
            continue
        pol = predictor_polarity(col)
        out[f"cv_{label}"] = beta_cv_skill(
            x,
            pot,
            flux,
            dates,
            pol,
            family=family,
            robust=robust,
            n_blocks=n_blocks,
            min_cv_days=min_cv_days,
        )
        out[f"app_{label}"] = beta_apparent_skill(
            x, pot, flux, pol, family=family, robust=robust
        )
        if np.isfinite(out[f"cv_{label}"]):
            got_any = True
    if not got_any:
        return None
    return out


# ---------------------------------------------------------------------------
# C5 — leave-one-site-out transfer of a single global β (§5)
# ---------------------------------------------------------------------------


def _global_beta_predict(x_eff_tr, pot_tr, flux_tr, x_eff_te, pot_te, family, robust):
    """Fit one global β on pooled training rows (flux-space NLS), predict the
    held-out site's flux. Returns flux_hat or None."""
    if family == "isotonic":
        iso = _fit_isotonic(x_eff_tr, pot_tr, flux_tr)
        if iso is None:
            return None
        return pot_te * np.clip(iso.predict(x_eff_te), 0.0, BETA_MAX_CAP)
    params = fit_beta_params(x_eff_tr, pot_tr, flux_tr, family, robust=robust)
    if params is None:
        return None
    return pot_te * predict_beta(family, x_eff_te, params)


def loso_beta_transfer(
    frame: pd.DataFrame,
    potential_col: str,
    flux_col: str,
    predictors: dict[str, str],
    family: str = "logistic",
    robust: bool = False,
    min_site_days: int = 60,
) -> dict[str, dict[str, float]]:
    """Per-held-out-site skill of a single **global** β(x) per predictor (C5).

    A global β is fit on the pooled rows of all *other* sites (flux-space weighted
    NLS, no per-site offset — the deployment scenario where the held-out site is
    unseen and site-specific normalization endpoints are unavailable) and used to
    predict the held-out site's flux. Skill is ``1 − SS_res/SS_tot`` with SS_tot vs
    the *held-out site's own mean* flux (an NSE-style per-site baseline; the
    contrast between predictors is evaluated on identical held-out rows).

    Returns ``{label: {site_id: skill}}``; sites shorter than ``min_site_days`` or
    with a degenerate global fit are skipped for that predictor.
    """
    need = [potential_col, flux_col, *predictors.values()]
    work = frame.dropna(subset=need).copy()
    counts = work.groupby("site_id").size()
    keep = counts[counts >= min_site_days].index
    work = work[work["site_id"].isin(keep)]
    if work["site_id"].nunique() < 10:
        return {label: {} for label in predictors}

    sites = work["site_id"].values
    pot = work[potential_col].values.astype(np.float64)
    flux = work[flux_col].values.astype(np.float64)
    uniq = pd.unique(sites)

    skills: dict[str, dict[str, float]] = {label: {} for label in predictors}
    for label, col in predictors.items():
        pol = predictor_polarity(col)
        x_eff = pol * work[col].values.astype(np.float64)
        for test_site in uniq:
            te = sites == test_site
            tr = ~te
            if te.sum() < min_site_days or tr.sum() < 50:
                continue
            flux_hat = _global_beta_predict(
                x_eff[tr], pot[tr], flux[tr], x_eff[te], pot[te], family, robust
            )
            if flux_hat is None or not np.all(np.isfinite(flux_hat)):
                continue
            flux_te = flux[te]
            ss_res = float(((flux_te - flux_hat) ** 2).sum())
            ss_tot = float(((flux_te - flux_te.mean()) ** 2).sum())
            if ss_tot < 1e-12:
                continue
            skills[label][test_site] = 1.0 - ss_res / ss_tot
    return skills
