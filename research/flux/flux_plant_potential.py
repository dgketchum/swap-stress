"""Plant-potential / supply-demand stress-function experiment — orchestrator.

Implements ``notes/flux_plant_potential_plan.md``. The soil-state ψ product carried
no ecosystem-flux information beyond θ (the §3.7 falsification, C1–C5 all fail):
within-site ψ_RF = g(θ) at fixed covariates, so by the data-processing inequality it
cannot add information over VWC. The only way to add information is to change the
*inputs* so the stress variable is no longer a deterministic function of θ. Plant
water potential does this — ψ_leaf ≈ ψ_soil − E·R_plant, and the E·R term
(transpiration demand × plant hydraulic resistance) is **not** a function of θ.

This module builds a plant-state observation (AMSR LPDRv3 X-band VOD, already on
``/nas/soils/swapstress/amsr``) and a supply×demand combination, then tests them
against the same REW/θ baselines and the same out-of-sample machinery as §3.7 so
the results drop in next to the existing falsification test.

Pre-registered contrasts (plan §5), out of sample and FDR-controlled as one family.
The new proxy must clear a **higher** bar than ψ did — beat not just REW but REW×VPD:

- **P1** β(ΔVOD) vs β(REW)              — does an independent plant-water observation
  beat texture-normalized wetness *within* site?
- **P2** β(REW,VPD) vs β(REW)           — honesty guard: how much of any gap is just
  adding the demand axis to the cheap variable (no new sensor)?
- **P3** β(REW,VPD,ΔVOD) vs β(REW,VPD)  — the marginal value of the plant observation
  beyond supply+demand — the number that actually justifies VOD.
- **P4** LOSO transfer β(ΔVOD) vs β(REW)/β(θ) to held-out sites.
- **P5** β(VOD_anom) vs β(REW) for GPP, leakage-safe (SIF/ESI excluded as predictors;
  VOD anomaly detrends slow biomass so it is not GPP_pot leakage).

Machinery reuses :mod:`map.evaluation.flux_stats` (blocked CV, site bootstrap,
BH-FDR, sign test), :mod:`map.evaluation.flux_beta_models` (β_max cap, GPP envelope,
polarity) and :mod:`map.evaluation.flux_features` (ET0, REW). The multiplicative
β for supply×demand and fusion is a **product of monotone sigmoids** (Jarvis/Feddes
form) implemented here — one factor per input, matched-complexity by parameter count.

Data: no new download, no GPU, no Earth Engine. VOD extraction from the cached
NetCDFs is I/O-bound (a few minutes over the study years); the β analysis is CPU
minutes over the cached daily table.

Tables → ``…/flux_validation/plant_potential/``:
  pp_cv.csv · pp_contrasts.csv · pp_transfer_loso.csv · pp_beta_params.csv
Figures A–D (PNG, absolute paths printed on render).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

from map.evaluation import flux_beta_models as fbm
from map.evaluation import flux_cv_analysis as fca
from map.evaluation import flux_features as ff
from map.evaluation import flux_stats as fs

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
AMSR_DIR = "/nas/soils/swapstress/amsr"
BASE_DIR = "/nas/soils/swapstress/evaluation/flux_validation"
DAILY_PARQUET = f"{BASE_DIR}/flux_site_daily.parquet"
META_PARQUET = f"{BASE_DIR}/flux_site_meta.parquet"
OUT_DIR = Path(f"{BASE_DIR}/plant_potential")
VOD_PARQUET = OUT_DIR / "vod_site_daily.parquet"

AMSR_RE = re.compile(r"AMSR-E-2_LPDRv3_Y(\d{4})_([AD])\.nc4$")
AMSR_CRS = "EPSG:3410"  # NSIDC EASE-Grid Global (declared in the NetCDF spatial_ref)

# QA_mask: 1 = high uncertainty, 0 = clean. VOD units Nepers, valid range 0–3.
VOD_VALID_LO, VOD_VALID_HI = 0.0, 3.0

# Weekly composite half-window (days). A centered 7-day mean denoises the single
# AM/PM difference (plan §7) and fills 1–2 day gaps; the 15-day CV embargo exceeds
# the 3-day half-window so the smoothing cannot leak across a fold boundary.
COMPOSITE_WINDOW = 7

# Response configs.
ET_CFG = {"label": "ET", "potential": "et0", "flux": "et_corr"}
GPP_CFG = {"label": "GPP", "potential": "gpp_pot", "flux": "gpp"}

# Short-label → daily-table column, and the β polarity of each predictor.
# Polarity +1: β rises with the raw value (supply / plant-water proxies). −1: β
# falls with the raw value (suction pF and atmospheric demand VPD).
#
# ΔVOD (night−day drawdown) polarity is set +1 on the a-priori reading that a
# larger overnight-recovered swing marks an actively transpiring, non-stressed
# canopy (drought closes stomata and shrinks the swing). This is only an *a
# priori sign*: a monotone β with the wrong polarity yields ~zero/negative CV
# skill, never a false positive, so the sign choice cannot inflate the test.
PRED_COL = {
    "dvod": "dvod",
    "vod_d": "vod_d_w",  # predawn-proxy (overnight-rehydrated)
    "vod_anom": "vod_anom",  # seasonal-detrended deficit anomaly
    "rew": "rew_theta_l4_surf",
    "theta": "theta_l4_surf",
    "vpd": "vpd",
    "psi_rf": "suction_l4",
}
PRED_POLARITY = {
    "dvod": 1,
    "vod_d": 1,
    "vod_anom": 1,
    "rew": 1,
    "theta": 1,
    "vpd": -1,
    "psi_rf": -1,
}

# Single-predictor CV skill summary (one β factor each).
CV_PREDICTORS = ["dvod", "vod_d", "vod_anom", "rew", "theta", "vpd", "psi_rf"]

# Pre-registered contrasts: (name, response, set_a, set_b, note). Positive delta
# favours set_a. CV compares different-complexity sets fairly (out of sample).
CONTRASTS = [
    (
        "P1_dvod_vs_rew",
        "ET",
        ["dvod"],
        ["rew"],
        "plant obs vs texture-normalized wetness",
    ),
    (
        "P2_rewvpd_vs_rew",
        "ET",
        ["rew", "vpd"],
        ["rew"],
        "demand axis on the cheap variable",
    ),
    (
        "P3_fusion_vs_rewvpd",
        "ET",
        ["rew", "vpd", "dvod"],
        ["rew", "vpd"],
        "plant obs beyond supply+demand",
    ),
    (
        "P5_vodanom_vs_rew",
        "GPP",
        ["vod_anom"],
        ["rew"],
        "GPP leakage-safe plant anomaly",
    ),
]

# LOSO transfer (P4): single global β per predictor, contrasts on held-out sites.
TRANSFER_PREDICTORS = ["dvod", "rew", "theta"]
TRANSFER_CONTRASTS = [
    ("P4_transfer_dvod_vs_rew", "dvod", "rew"),
    ("P4_transfer_dvod_vs_theta", "dvod", "theta"),
]

MET_ENV_COLS = ["sw_in", "t_avg", "vpd"]


# ===========================================================================
# VOD extraction
# ===========================================================================


def _study_years(daily: pd.DataFrame) -> list[int]:
    d = pd.to_datetime(pd.Index(daily["date"]))
    return list(range(int(d.year.min()), int(d.year.max()) + 1))


def _project_to_ease2(lats: np.ndarray, lons: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Nearest EASE-Grid (row, col) for each site; clip so a 3×3 window fits."""
    from pyproj import Transformer

    t = Transformer.from_crs("EPSG:4326", AMSR_CRS, always_xy=True)
    xx, yy = t.transform(np.asarray(lons), np.asarray(lats))
    ci = np.array([int(np.argmin(np.abs(x - v))) for v in xx])
    ri = np.array([int(np.argmin(np.abs(y - v))) for v in yy])
    ci = np.clip(ci, 1, len(x) - 2)
    ri = np.clip(ri, 1, len(y) - 2)
    return ri, ci


def _extract_overpass_file(path: str, ri: np.ndarray, ci: np.ndarray):
    """Return (dates, values[n_days, n_sites]) — QA-filtered 3×3-mean VOD.

    Reads the full year array once (fast given the time-chunked HDF5 layout),
    masks QA!=0 and out-of-range VOD, then takes the 3×3 spatial mean at each site.
    """
    import netCDF4 as nc

    m = AMSR_RE.search(os.path.basename(path))
    year = int(m.group(1))
    base = datetime(year, 1, 1)

    ds = nc.Dataset(path)
    try:
        tvals = ds.variables["time"][:]
        vod = np.asarray(ds.variables["VOD"][:], dtype=np.float32)
        qa = np.asarray(ds.variables["QA_mask"][:], dtype=np.int8)
    finally:
        ds.close()

    vod = np.where(qa == 0, vod, np.nan)
    vod = np.where((vod >= VOD_VALID_LO) & (vod <= VOD_VALID_HI), vod, np.nan)

    n_days = vod.shape[0]
    n_sites = len(ri)
    acc = np.zeros((n_days, n_sites), dtype=np.float64)
    cnt = np.zeros((n_days, n_sites), dtype=np.float64)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            vals = vod[:, ri + dr, ci + dc]  # (n_days, n_sites)
            ok = np.isfinite(vals)
            acc = np.where(ok, acc + np.nan_to_num(vals), acc)
            cnt = cnt + ok
    mean3 = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), np.nan)

    dates = [(base + timedelta(days=int(t))).date() for t in tvals]
    return dates, mean3


def extract_vod(sites: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Per-site daily QA-filtered VOD for both overpasses over ``years``.

    Returns a long frame ``(site_id, date, vod_a, vod_d)`` where ``vod_a`` is the
    ~1:30 PM ascending (midday, drawn-down) and ``vod_d`` the ~1:30 AM descending
    (overnight-rehydrated) 3×3-mean VOD.
    """
    import netCDF4 as nc

    ref = sorted(glob.glob(os.path.join(AMSR_DIR, "AMSR-E-2_LPDRv3_Y*_A.nc4")))[0]
    ds = nc.Dataset(ref)
    x = np.asarray(ds.variables["x"][:], dtype=np.float64)
    y = np.asarray(ds.variables["y"][:], dtype=np.float64)
    ds.close()

    site_ids = sites["site_id"].values
    ri, ci = _project_to_ease2(sites["lat"].values, sites["lon"].values, x, y)

    frames = []
    for year in years:
        for overpass, col in (("A", "vod_a"), ("D", "vod_d")):
            path = os.path.join(AMSR_DIR, f"AMSR-E-2_LPDRv3_Y{year}_{overpass}.nc4")
            if not os.path.exists(path):
                print(f"  missing {os.path.basename(path)} — skipped")
                continue
            dates, vals = _extract_overpass_file(path, ri, ci)
            n_days = len(dates)
            df = pd.DataFrame(
                {
                    "site_id": np.tile(site_ids, n_days),
                    "date": np.repeat(dates, len(site_ids)),
                    col: vals.reshape(-1),
                }
            )
            df = df.dropna(subset=[col])
            frames.append(df)
            print(f"  {year} {overpass}: {len(df)} site-days")

    if not frames:
        return pd.DataFrame(columns=["site_id", "date", "vod_a", "vod_d"])

    a = pd.concat([f for f in frames if "vod_a" in f.columns], ignore_index=True)
    d = pd.concat([f for f in frames if "vod_d" in f.columns], ignore_index=True)
    vod = a.merge(d, on=["site_id", "date"], how="outer")
    vod = vod.sort_values(["site_id", "date"]).reset_index(drop=True)
    print(f"VOD table: {len(vod)} rows, {vod['site_id'].nunique()} sites")
    return vod


def run_extract(daily_parquet: str = DAILY_PARQUET, meta_parquet: str = META_PARQUET):
    """Extract VOD at flux-tower coords and cache ``vod_site_daily.parquet``."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    daily = pd.read_parquet(daily_parquet)
    meta = pd.read_parquet(meta_parquet)
    sites = meta.dropna(subset=["lat", "lon"]).drop_duplicates("site_id")
    years = _study_years(daily)
    print(f"Extracting VOD for {len(sites)} sites over {years[0]}–{years[-1]} …")
    vod = extract_vod(sites, years)
    vod.to_parquet(VOD_PARQUET, index=False)
    print(f"Saved {VOD_PARQUET}")
    return vod


# ===========================================================================
# Derived VOD features (weekly composite, ΔVOD, seasonal anomaly)
# ===========================================================================


def _harmonic_anomaly(doy: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Residual of ``values`` against a smooth annual+semiannual DOY climatology.

    Removes the slow seasonal biomass/phenology cycle (plan §2a) with a 2-harmonic
    OLS fit, leaving the short-term water-deficit anomaly. NaN where the fit is
    degenerate (too few points).
    """
    ok = np.isfinite(values)
    if ok.sum() < 20:
        return np.full(len(values), np.nan)
    ang = 2.0 * np.pi * doy / 365.25
    design = np.column_stack(
        [
            np.ones(len(doy)),
            np.sin(ang),
            np.cos(ang),
            np.sin(2 * ang),
            np.cos(2 * ang),
        ]
    )
    beta, *_ = np.linalg.lstsq(design[ok], values[ok], rcond=None)
    clim = design @ beta
    return values - clim


def add_vod_features(vod: pd.DataFrame) -> pd.DataFrame:
    """Weekly-composited ΔVOD, predawn-proxy VOD_D, and seasonal anomaly per site.

    For each site the observed overpass series are reindexed onto a continuous
    daily calendar and centred-rolling-mean composited (``COMPOSITE_WINDOW`` days).
    ``dvod = vod_d_w − vod_a_w`` is the diurnal drawdown; ``vod_anom`` is the
    2-harmonic seasonal residual of the composited night VOD.
    """
    out = []
    for sid, sdf in vod.groupby("site_id"):
        sdf = sdf.sort_values("date")
        idx = pd.date_range(sdf["date"].min(), sdf["date"].max(), freq="D")
        s = sdf.set_index(pd.DatetimeIndex(sdf["date"])).reindex(idx)
        w = COMPOSITE_WINDOW
        vod_a_w = s["vod_a"].rolling(w, center=True, min_periods=1).mean()
        vod_d_w = s["vod_d"].rolling(w, center=True, min_periods=1).mean()
        dvod = vod_d_w - vod_a_w
        doy = idx.dayofyear.to_numpy().astype(np.float64)
        vod_anom = _harmonic_anomaly(doy, vod_d_w.to_numpy())
        rec = pd.DataFrame(
            {
                "site_id": sid,
                "date": idx.date,
                "vod_a_w": vod_a_w.to_numpy(),
                "vod_d_w": vod_d_w.to_numpy(),
                "dvod": dvod.to_numpy(),
                "vod_anom": vod_anom,
            }
        )
        # Keep only days with an actual composite (a real observation nearby).
        rec = rec.dropna(subset=["vod_d_w"], how="all")
        out.append(rec)
    feats = pd.concat(out, ignore_index=True)
    return feats


# ===========================================================================
# Product-of-sigmoids multiplicative β  (Jarvis/Feddes supply×demand form)
# ===========================================================================
#
# β(x_1,…,x_k) = β_max · Π_j σ(k_j·(x_eff_j − x0_j)),  σ = logistic,  x_eff = pol·x.
# One (x0, k) shape pair per factor + a shared β_max nuisance scale → matched
# complexity is controlled by the number of factors. Fit on the FLUX (not the
# heteroscedastic ratio, plan §4 / flux_beta_models §4.3), scored by blocked CV
# on the exact fold structure used by the confirmatory machinery.

N_BLOCKS = fca.N_BLOCKS
EMBARGO_DAYS = fca.EMBARGO_DAYS
MIN_CV_DAYS = fca.MIN_CV_DAYS
BETA_MAX_CAP = fbm.BETA_MAX_CAP
MAX_NFEV = 6000


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip(-z, -60.0, 60.0)))


def product_beta(x_eff: list[np.ndarray], params: np.ndarray) -> np.ndarray:
    """β_max · Π σ(k_j·(x_eff_j − x0_j)). ``params`` = [x0_0,k_0,…,β_max]."""
    beta = np.full(len(x_eff[0]), float(params[-1]), dtype=np.float64)
    for j, xe in enumerate(x_eff):
        x0, k = params[2 * j], params[2 * j + 1]
        beta = beta * _sigmoid(k * (xe - x0))
    return beta


def _init_bounds(x_eff: list[np.ndarray], ratio: np.ndarray):
    """Per-factor logistic init/bounds (mirrors flux_beta_models) + shared β_max."""
    p0, lb, ub = [], [], []
    for xe in x_eff:
        lo_x, hi_x = np.percentile(xe, [2, 98])
        span = max(hi_x - lo_x, 1e-6)
        p0 += [float(np.median(xe)), 4.0 / span]
        lb += [float(xe.min()) - span, 1e-4]
        ub += [float(xe.max()) + span, 200.0 / span]
    bmax0 = float(np.clip(np.percentile(ratio, 90), 0.1, BETA_MAX_CAP))
    p0.append(bmax0)
    lb.append(1e-3)
    ub.append(BETA_MAX_CAP)
    return np.array(p0), (np.array(lb), np.array(ub))


def fit_product_beta(
    x_eff: list[np.ndarray],
    potential: np.ndarray,
    flux: np.ndarray,
    robust: bool = False,
) -> np.ndarray | None:
    """Weighted NLS of ``flux ≈ potential · β(x_eff)`` (fit on the flux)."""
    potential = np.asarray(potential, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(potential > 1e-9, flux / potential, np.nan)
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size < 5 or len(x_eff[0]) < 5:
        return None
    p0, (lb, ub) = _init_bounds(x_eff, ratio)

    def resid(params):
        return flux - potential * product_beta(x_eff, params)

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
    if (not res.success and res.status <= 0) or not np.all(np.isfinite(res.x)):
        return None
    return res.x


def _apply_polarity(cols: list[str], frame_vals: dict, pols: list[int]):
    return [pol * frame_vals[c] for c, pol in zip(cols, pols)]


def product_beta_cv_skill(
    x_cols: list[np.ndarray],
    polarities: list[int],
    potential: np.ndarray,
    flux: np.ndarray,
    dates,
    n_blocks: int = N_BLOCKS,
    embargo_days: int = EMBARGO_DAYS,
    min_cv_days: int = MIN_CV_DAYS,
) -> float:
    """Blocked-CV out-of-sample skill of ``flux = potential·β(x_1…x_k)`` (plan §4).

    Skill is ``1 − SS_res/SS_tot`` pooled over folds with SS_tot vs the *training*
    mean of the flux (so skill can be negative). NaN unless every requested fold is
    usable — mirrors :func:`flux_beta_models.beta_cv_skill`.
    """
    flux = np.asarray(flux, dtype=np.float64)
    potential = np.asarray(potential, dtype=np.float64)
    if len(flux) < min_cv_days:
        return np.nan
    x_eff = [
        pol * np.asarray(x, dtype=np.float64) for x, pol in zip(x_cols, polarities)
    ]
    folds = fs.blocked_fold_indices(dates, n_blocks=n_blocks, embargo_days=embargo_days)
    min_train = 2 * len(x_cols) + 3  # params + slack
    ss_res = ss_tot = 0.0
    n_used = 0
    for tr, te in folds:
        if len(te) == 0 or len(tr) < min_train:
            continue
        params = fit_product_beta([xe[tr] for xe in x_eff], potential[tr], flux[tr])
        if params is None:
            continue
        flux_hat = potential[te] * product_beta([xe[te] for xe in x_eff], params)
        if not np.all(np.isfinite(flux_hat)):
            continue
        resid = flux[te] - flux_hat
        ss_res += float(resid @ resid)
        base = flux[te] - float(np.mean(flux[tr]))
        ss_tot += float(base @ base)
        n_used += 1
    if n_used < n_blocks or ss_tot < 1e-12:
        return np.nan
    return 1.0 - ss_res / ss_tot


def product_beta_apparent_skill(x_cols, polarities, potential, flux) -> float:
    """In-sample R² of the multiplicative model — the overfitting reference."""
    flux = np.asarray(flux, dtype=np.float64)
    potential = np.asarray(potential, dtype=np.float64)
    x_eff = [
        pol * np.asarray(x, dtype=np.float64) for x, pol in zip(x_cols, polarities)
    ]
    params = fit_product_beta(x_eff, potential, flux)
    if params is None:
        return np.nan
    flux_hat = potential * product_beta(x_eff, params)
    ss_res = float(((flux - flux_hat) ** 2).sum())
    ss_tot = float(((flux - flux.mean()) ** 2).sum())
    if ss_tot < 1e-12:
        return np.nan
    return 1.0 - ss_res / ss_tot


# ===========================================================================
# Frame prep
# ===========================================================================


def load_inputs():
    daily = pd.read_parquet(DAILY_PARQUET)
    meta = pd.read_parquet(META_PARQUET)
    cov_df = ff.build_site_covariates(meta[["site_id", "lat", "lon"]])
    return daily, cov_df


def add_gpp_potential(gs: pd.DataFrame) -> pd.DataFrame:
    """Per-site τ=0.90 quantile-regression GPP envelope column ``gpp_pot`` (§1)."""
    gs = gs.copy()
    gs["gpp_pot"] = np.nan
    if "gpp" not in gs.columns:
        return gs
    for _site, idx in gs.groupby("site_id").groups.items():
        sdf = gs.loc[idx]
        cols = ["sw_in", "t_avg"]
        if "vpd" in sdf.columns and sdf["vpd"].notna().mean() > 0.8:
            cols = MET_ENV_COLS
        pot = fbm.gpp_potential(sdf, cols, tau=0.90)
        if pot is not None:
            gs.loc[idx, "gpp_pot"] = pot
    return gs


def prep_frame(daily: pd.DataFrame, vod_feats: pd.DataFrame) -> pd.DataFrame:
    """Growing-season slice with ET0, REW, GPP envelope, and joined VOD features."""
    gs = fca.prep_growing_season(daily)  # adds et0, applies the growing filter
    gs = ff.rew_empirical(gs, "theta_l4_surf")  # -> rew_theta_l4_surf
    gs = add_gpp_potential(gs)
    # Normalize join keys to datetime.date and merge VOD features.
    gs = gs.copy()
    gs["date"] = pd.to_datetime(gs["date"]).dt.date
    vf = vod_feats.copy()
    vf["date"] = pd.to_datetime(vf["date"]).dt.date
    gs = gs.merge(vf, on=["site_id", "date"], how="left")
    return gs


# ===========================================================================
# Per-site contrasts (P1/P2/P3/P5) and LOSO transfer (P4)
# ===========================================================================


def _cols_and_pols(labels: list[str]):
    return [PRED_COL[x] for x in labels], [PRED_POLARITY[x] for x in labels]


def per_site_contrast(
    sdf: pd.DataFrame,
    pot_col: str,
    flux_col: str,
    set_a: list[str],
    set_b: list[str],
    min_cv_days: int = MIN_CV_DAYS,
    n_blocks: int = N_BLOCKS,
) -> dict | None:
    """CV Δskill of β(set_a) − β(set_b) at one site on *identical* rows.

    Rows are the intersection where response, potential and every predictor in
    either set are present, so the delta is matched row-for-row.
    """
    labels = list(dict.fromkeys(set_a + set_b))
    cols = [PRED_COL[x] for x in labels]
    clean = sdf.dropna(subset=[pot_col, flux_col, *cols])
    if len(clean) < min_cv_days:
        return None
    pot = clean[pot_col].values.astype(np.float64)
    flux = clean[flux_col].values.astype(np.float64)
    dates = clean["date"].values

    def skill(labels_set):
        c, p = _cols_and_pols(labels_set)
        xs = [clean[col].values.astype(np.float64) for col in c]
        if any(x.std() < 1e-12 for x in xs):
            return np.nan
        return product_beta_cv_skill(
            xs, p, pot, flux, dates, n_blocks=n_blocks, min_cv_days=min_cv_days
        )

    sk_a = skill(set_a)
    sk_b = skill(set_b)
    if not (np.isfinite(sk_a) and np.isfinite(sk_b)):
        return None
    return {
        "n_days": len(clean),
        "skill_a": sk_a,
        "skill_b": sk_b,
        "delta": sk_a - sk_b,
    }


def run_contrasts(gs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run P1/P2/P3/P5 across sites; return (per_site_long, summary)."""
    per_site_rows = []
    summary_rows = []
    for name, resp, set_a, set_b, note in CONTRASTS:
        cfg = ET_CFG if resp == "ET" else GPP_CFG
        need = [cfg["flux"], cfg["potential"]] + [
            PRED_COL[x] for x in dict.fromkeys(set_a + set_b)
        ]
        sub = gs.dropna(subset=need)
        deltas = []
        for site, sdf in sub.groupby("site_id"):
            r = per_site_contrast(sdf, cfg["potential"], cfg["flux"], set_a, set_b)
            if r is None:
                continue
            r.update({"contrast": name, "response": resp, "site_id": site})
            per_site_rows.append(r)
            deltas.append(r["delta"])
        if not deltas:
            continue
        deltas = np.array(deltas, dtype=np.float64)
        med, lo, hi = fs.site_bootstrap_ci(deltas)
        st = fs.sign_test(deltas, alternative="greater")
        summary_rows.append(
            {
                "contrast": name,
                "response": resp,
                "n_sites": int(len(deltas)),
                "delta_median": med,
                "delta_lo": lo,
                "delta_hi": hi,
                "frac_a_wins": st["frac_pos"],
                "sign_p": st["p"],
                "note": note,
            }
        )
    return pd.DataFrame(per_site_rows), pd.DataFrame(summary_rows)


def product_loso_transfer(
    frame: pd.DataFrame,
    pot_col: str,
    flux_col: str,
    predictors: list[str],
    min_site_days: int = 60,
) -> dict[str, dict[str, float]]:
    """Per-held-out-site skill of a single **global** β(x) per predictor (P4).

    A global β is fit on pooled rows of all *other* sites (flux-space NLS, no
    per-site offset — the deployment case where the held-out site is unseen and
    per-site REW endpoints are unavailable) and used to predict the held-out
    site's flux. Skill is NSE vs the held-out site's own mean; the contrast between
    predictors is on identical held-out rows.
    """
    cols = [PRED_COL[x] for x in predictors]
    work = frame.dropna(subset=[pot_col, flux_col, *cols]).copy()
    counts = work.groupby("site_id").size()
    keep = counts[counts >= min_site_days].index
    work = work[work["site_id"].isin(keep)]
    if work["site_id"].nunique() < 10:
        return {label: {} for label in predictors}

    sites = work["site_id"].values
    pot = work[pot_col].values.astype(np.float64)
    flux = work[flux_col].values.astype(np.float64)
    uniq = pd.unique(sites)

    skills: dict[str, dict[str, float]] = {label: {} for label in predictors}
    for label in predictors:
        pol = PRED_POLARITY[label]
        x_eff = pol * work[PRED_COL[label]].values.astype(np.float64)
        for test_site in uniq:
            te = sites == test_site
            tr = ~te
            if te.sum() < min_site_days or tr.sum() < 50:
                continue
            params = fit_product_beta([x_eff[tr]], pot[tr], flux[tr])
            if params is None:
                continue
            flux_hat = pot[te] * product_beta([x_eff[te]], params)
            if not np.all(np.isfinite(flux_hat)):
                continue
            flux_te = flux[te]
            ss_res = float(((flux_te - flux_hat) ** 2).sum())
            ss_tot = float(((flux_te - flux_te.mean()) ** 2).sum())
            if ss_tot < 1e-12:
                continue
            skills[label][test_site] = 1.0 - ss_res / ss_tot
    return skills


def run_transfer(gs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """P4 LOSO transfer (ET) for {ΔVOD, REW, θ}; return (per_site, contrasts)."""
    skills = product_loso_transfer(
        gs, ET_CFG["potential"], ET_CFG["flux"], TRANSFER_PREDICTORS
    )
    common = (
        sorted(set.intersection(*[set(v.keys()) for v in skills.values()]))
        if all(skills.values())
        else []
    )
    per_site_rows = []
    for site in common:
        row = {"response": "ET", "site_id": site}
        for label in TRANSFER_PREDICTORS:
            row[f"transfer_{label}"] = skills[label][site]
        per_site_rows.append(row)
    per_site = pd.DataFrame(per_site_rows)

    contrast_rows = []
    for name, a, b in TRANSFER_CONTRASTS:
        if per_site.empty:
            continue
        deltas = (per_site[f"transfer_{a}"] - per_site[f"transfer_{b}"]).values
        deltas = deltas[np.isfinite(deltas)]
        if len(deltas) < 3:
            continue
        med, lo, hi = fs.site_bootstrap_ci(deltas)
        st = fs.sign_test(deltas, alternative="greater")
        contrast_rows.append(
            {
                "contrast": name,
                "response": "ET",
                "n_sites": int(len(deltas)),
                "delta_median": med,
                "delta_lo": lo,
                "delta_hi": hi,
                "frac_a_wins": st["frac_pos"],
                "sign_p": st["p"],
                "note": f"LOSO transfer Δskill β({a})−β({b})",
            }
        )
    return per_site, pd.DataFrame(contrast_rows)


# ===========================================================================
# Single-predictor CV summary
# ===========================================================================


def run_single_predictor_cv(gs: pd.DataFrame) -> pd.DataFrame:
    """Per-site single-factor β CV skill for each predictor, both responses."""
    rows = []
    for cfg in (ET_CFG, GPP_CFG):
        sub = gs.dropna(subset=[cfg["flux"], cfg["potential"]])
        skills = {p: [] for p in CV_PREDICTORS}
        for _site, sdf in sub.groupby("site_id"):
            for p in CV_PREDICTORS:
                col = PRED_COL[p]
                clean = sdf.dropna(subset=[cfg["flux"], cfg["potential"], col])
                if len(clean) < MIN_CV_DAYS:
                    skills[p].append(np.nan)
                    continue
                x = clean[col].values.astype(np.float64)
                if x.std() < 1e-12:
                    skills[p].append(np.nan)
                    continue
                s = product_beta_cv_skill(
                    [x],
                    [PRED_POLARITY[p]],
                    clean[cfg["potential"]].values.astype(np.float64),
                    clean[cfg["flux"]].values.astype(np.float64),
                    clean["date"].values,
                )
                skills[p].append(s)
        for p in CV_PREDICTORS:
            v = np.array(skills[p], dtype=np.float64)
            med, lo, hi = fs.site_bootstrap_ci(v)
            rows.append(
                {
                    "response": cfg["label"],
                    "predictor": p,
                    "n_sites": int(np.isfinite(v).sum()),
                    "cv_skill_med": med,
                    "cv_skill_lo": lo,
                    "cv_skill_hi": hi,
                }
            )
    return pd.DataFrame(rows)


# ===========================================================================
# β parameters (ΔVOD stress-onset per site)
# ===========================================================================


def fit_beta_param_table(gs: pd.DataFrame, cov_df: pd.DataFrame) -> pd.DataFrame:
    """Fitted single-factor β(ΔVOD) onset/slope per site (ET) + covariates."""
    col = PRED_COL["dvod"]
    pol = PRED_POLARITY["dvod"]
    sub = gs.dropna(subset=["et_corr", "et0", col])
    rows = []
    for site, sdf in sub.groupby("site_id"):
        if len(sdf) < MIN_CV_DAYS:
            continue
        x_eff = pol * sdf[col].values.astype(np.float64)
        params = fit_product_beta([x_eff], sdf["et0"].values, sdf["et_corr"].values)
        if params is None:
            continue
        rows.append(
            {
                "site_id": site,
                "x0_native": float(pol * params[0]),
                "slope_k": float(params[1]),
                "beta_max": float(params[2]),
                "n_days": len(sdf),
            }
        )
    tbl = pd.DataFrame(rows)
    if tbl.empty:
        return tbl
    keep = [
        c for c in ["site_id", "sand", "clay", "aridity_index"] if c in cov_df.columns
    ]
    return tbl.merge(cov_df[keep], on="site_id", how="left")


# ===========================================================================
# Contrast synthesis (BH-FDR, PASS/FAIL, interpretation grid)
# ===========================================================================


def synthesize_contrasts(
    p_summary: pd.DataFrame, transfer_contrasts: pd.DataFrame
) -> pd.DataFrame:
    """Combine P1/P2/P3/P5 + P4 into one family; BH-FDR; PASS/FAIL; reading."""
    conf = pd.concat([p_summary, transfer_contrasts], ignore_index=True)
    if conf.empty:
        return conf
    conf["ci_excludes_zero_positive"] = np.isfinite(conf["delta_lo"]) & (
        conf["delta_lo"] > 0
    )
    rej, adj = fs.benjamini_hochberg(conf["sign_p"].values, alpha=0.05)
    conf["p_bh"] = adj
    conf["bh_reject"] = rej
    conf["verdict"] = np.where(
        conf["ci_excludes_zero_positive"] & conf["bh_reject"], "PASS", "FAIL"
    )
    return conf


def interpretation(conf: pd.DataFrame) -> str:
    """Pre-committed reading of the P1/P2/P3 outcome grid (plan §6)."""
    if conf.empty:
        return "No contrasts computed."

    def passed(name):
        row = conf[conf["contrast"] == name]
        return bool(row["verdict"].eq("PASS").any()) if not row.empty else False

    p1, p2, p3 = (
        passed("P1_dvod_vs_rew"),
        passed("P2_rewvpd_vs_rew"),
        passed("P3_fusion_vs_rewvpd"),
    )
    p4 = any(passed(n) for n, _, _ in TRANSFER_CONTRASTS)
    if p1 and p3:
        return (
            "P1 & P3 PASS → plant observation adds real information beyond "
            "supply+demand. Build the product around plant potential (headline)."
        )
    if (not p1) and p2:
        return (
            "VOD ≈ REW but REW×VPD > REW → the value is the *demand axis*, not VOD. "
            "A cheap SPAC index (REW×VPD) suffices; no new sensor needed."
        )
    if p4 and not (p1 or p3):
        return (
            "Transfer-only benefit (P4) → plant potential helps map-scale transfer "
            "but not within-site; a scoped claim, not a headline."
        )
    if not (p1 or p2 or p3 or p4):
        return (
            "Null persists up the ladder → soil water is genuinely 2nd-order to met "
            "at this scale; report and stop."
        )
    return "Mixed outcome — see the contrast table; no single grid cell dominates."


# ===========================================================================
# Figures
# ===========================================================================


def _save(fig, path: Path) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def figA_beta_curves(gs: pd.DataFrame, out: Path) -> Path:
    """Fitted β(ΔVOD) vs β(REW) vs β(θ) on pooled ET data with empirical clouds."""
    import matplotlib.pyplot as plt

    sub = gs.dropna(
        subset=["et_corr", "et0", "dvod", "rew_theta_l4_surf", "theta_l4_surf"]
    )
    if len(sub) > 40000:
        sub = sub.sample(40000, random_state=0)
    pot = sub["et0"].values.astype(np.float64)
    flux = sub["et_corr"].values.astype(np.float64)
    emp = np.clip(flux / np.maximum(pot, 1e-6), 0, BETA_MAX_CAP)
    specs = [
        ("ΔVOD", "dvod"),
        ("θ (L4 surf)", "theta_l4_surf"),
        ("REW", "rew_theta_l4_surf"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (title, col) in zip(axes, specs):
        x = sub[col].values.astype(np.float64)
        pol = PRED_POLARITY["dvod"] if col == "dvod" else 1
        ax.scatter(x, emp, s=3, alpha=0.05, color="#999999")
        params = fit_product_beta([pol * x], pot, flux)
        if params is not None:
            grid = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 200)
            ax.plot(grid, product_beta([pol * grid], params), color="#ee6677", lw=2)
        ax.set_xlabel(title)
        ax.set_ylim(0, min(BETA_MAX_CAP, 1.6))
        ax.axhline(1.0, color="k", lw=0.4, ls=":")
    axes[0].set_ylabel("β = ET / ET0")
    fig.suptitle("Fig A — Fitted multiplicative stress functions β(x) (pooled ET)")
    return _save(fig, out / "figA_beta_curves.png")


def figB_contrasts(conf: pd.DataFrame, out: Path) -> Path:
    """P1–P5 decision bars: Δskill with bootstrap CI, coloured by verdict."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.4))
    if not conf.empty:
        c = conf.reset_index(drop=True)
        x = np.arange(len(c))
        meds = c["delta_median"].values
        err = np.vstack([meds - c["delta_lo"].values, c["delta_hi"].values - meds])
        colors = ["#228833" if v == "PASS" else "#bbbbbb" for v in c["verdict"]]
        ax.bar(x, meds, color=colors)
        ax.errorbar(x, meds, yerr=err, fmt="none", ecolor="k", capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(c["contrast"], rotation=30, ha="right", fontsize=7)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("Δ CV skill (a − b)")
    ax.set_title("Fig B — Plant-potential contrasts (PASS = green, CI>0 & BH-FDR)")
    return _save(fig, out / "figB_contrasts.png")


def figC_transfer(transfer_ps: pd.DataFrame, out: Path) -> Path:
    """LOSO transfer skill by predictor (ET) — the map-scale decision figure."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.2))
    if transfer_ps is not None and not transfer_ps.empty:
        labels = TRANSFER_PREDICTORS
        meds, los, his = [], [], []
        for label in labels:
            v = transfer_ps[f"transfer_{label}"].values
            m, lo, hi = fs.site_bootstrap_ci(v)
            meds.append(m)
            los.append(lo)
            his.append(hi)
        x = np.arange(len(labels))
        meds = np.array(meds)
        err = np.vstack([meds - np.array(los), np.array(his) - meds])
        ax.bar(x, meds, color=["#66ccee", "#ccbb44", "#4477aa"])
        ax.errorbar(x, meds, yerr=err, fmt="none", ecolor="k", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels([f"β({p})" for p in labels])
        ax.set_ylabel("median LOSO transfer skill (ET)")
        ax.axhline(0, color="k", lw=0.6)
    ax.set_title("Fig C — Global β transfers to unseen sites (P4)")
    return _save(fig, out / "figC_transfer.png")


def figD_onset(bp: pd.DataFrame, out: Path) -> Path:
    """β(ΔVOD) stress-onset distribution / vs aridity across sites."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    if bp is not None and not bp.empty and "x0_native" in bp.columns:
        m = bp["x0_native"].notna()
        if "aridity_index" in bp.columns and bp["aridity_index"].notna().any():
            mm = m & bp["aridity_index"].notna()
            ax.scatter(
                bp.loc[mm, "aridity_index"],
                bp.loc[mm, "x0_native"],
                s=18,
                color="#66ccee",
            )
            ax.set_xlabel("aridity index (P/PET)")
        else:
            ax.hist(bp.loc[m, "x0_native"], bins=25, color="#66ccee")
            ax.set_xlabel("β stress-onset ΔVOD")
        ax.set_ylabel("β stress-onset ΔVOD (x0)")
    ax.set_title("Fig D — ΔVOD stress-onset across sites")
    return _save(fig, out / "figD_onset.png")


# ===========================================================================
# Driver
# ===========================================================================


def run_all(daily: pd.DataFrame, cov_df: pd.DataFrame, vod_feats: pd.DataFrame) -> dict:
    tables: dict[str, pd.DataFrame] = {}
    gs = prep_frame(daily, vod_feats)
    tables["_gs"] = gs

    print("Single-predictor CV skill …")
    tables["pp_cv"] = run_single_predictor_cv(gs)

    print("Per-site contrasts P1/P2/P3/P5 …")
    per_site, p_summary = run_contrasts(gs)
    tables["_pp_per_site"] = per_site

    print("LOSO transfer P4 …")
    transfer_ps, transfer_contrasts = run_transfer(gs)
    tables["_transfer_per_site"] = transfer_ps

    tables["pp_contrasts"] = synthesize_contrasts(p_summary, transfer_contrasts)
    tables["pp_transfer_loso"] = transfer_ps

    print("β(ΔVOD) parameters per site …")
    tables["pp_beta_params"] = fit_beta_param_table(gs, cov_df)
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("extract-vod", help="Extract & cache VOD at flux-tower coords")
    run_p = sub.add_parser("run", help="Run the plant-potential β analysis")
    run_p.add_argument("--out", default=str(OUT_DIR))
    run_p.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    if args.command == "extract-vod":
        run_extract()
        return

    out = Path(getattr(args, "out", str(OUT_DIR)))
    out.mkdir(parents=True, exist_ok=True)

    if not VOD_PARQUET.exists():
        raise SystemExit(
            f"{VOD_PARQUET} not found — run `extract-vod` first (see commands.sh)."
        )

    print("Loading inputs …")
    daily, cov_df = load_inputs()
    vod = pd.read_parquet(VOD_PARQUET)
    print(f"  daily rows={len(daily):,}  sites={daily['site_id'].nunique()}")
    print(f"  vod rows={len(vod):,}  sites={vod['site_id'].nunique()}")

    print("Building VOD features (weekly composite, ΔVOD, anomaly) …")
    vod_feats = add_vod_features(vod)

    print("Running plant-potential analyses …")
    tables = run_all(daily, cov_df, vod_feats)

    written = []
    for name, df in tables.items():
        if name.startswith("_") or df is None or df.empty:
            continue
        path = out / f"{name}.csv"
        df.to_csv(path, index=False)
        written.append(path)

    conf = tables["pp_contrasts"]
    print("\nPlant-potential contrasts (P1–P5, BH-FDR across the family):")
    if not conf.empty:
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(
                conf[
                    [
                        "contrast",
                        "response",
                        "n_sites",
                        "delta_median",
                        "delta_lo",
                        "delta_hi",
                        "sign_p",
                        "bh_reject",
                        "verdict",
                    ]
                ].to_string(index=False)
            )
    print("\nReading:", interpretation(conf))

    print("\nTables written:")
    for p in written:
        print(f"  {p}")

    if not getattr(args, "no_figures", False):
        gs = tables["_gs"]
        print("\nFigures:")
        figs = [
            figA_beta_curves(gs, out),
            figB_contrasts(conf, out),
            figC_transfer(tables["_transfer_per_site"], out),
            figD_onset(tables["pp_beta_params"], out),
        ]
        for p in figs:
            print(f"  {p}")


if __name__ == "__main__":
    main()
