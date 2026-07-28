"""Multiplicative water-stress function experiment — orchestrator.

Implements ``notes/flux_stress_function_experiment_plan.md``. Every prior flux
test put soil water in an *additive* model; this re-tests in the physically
correct *multiplicative* framing ``flux = potential · β(soil)`` where β is a
matched-complexity stress function (:mod:`research.flux.flux_beta_models`). It
reuses the retest's out-of-sample machinery (blocked CV, LOSO, site bootstrap,
BH-FDR from :mod:`research.flux.flux_stats`) and the cached
``flux_site_daily.parquet`` — no re-extraction, CPU-only, minutes.

Pre-registered confirmatory contrasts (plan §1), evaluated out of sample and
FDR-controlled as one family:

- **C1** β(ψ_RF) vs β(REW)      — the decision: does the product beat cheap
  per-site percentile normalization in the correct framing?
- **C2** β(ψ_RF) vs β(θ)        — does the stress form rescue ψ vs raw VWC?
- **C3** β(ψ_obs) vs β(REW)     — ceiling: does *perfect* in-situ ψ beat REW?
- **C4** β(ψ_RF) vs β(ψ_PTF)    — learned mapping vs Rosetta.
- **C5** LOSO transfer          — a single global β(ψ) vs β(θ)/β(REW) on unseen
  sites (the map-scale deployment case where per-site REW endpoints don't exist).

Exploratory: E1 non-leaky depth (antecedent-L3-θ), E2 β-threshold stratification
by texture/aridity, E5 GPP-source robustness.

Tables → ``…/flux_validation/stress_function/``:
  stress_function_cv.csv · stress_transfer_loso.csv · stress_contrasts.csv
  · stress_beta_params.csv · stress_depth_e1.csv
Figures A–F (PNG, absolute paths printed on render).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research.flux import flux_beta_models as fbm  # noqa: E402
from research.flux import flux_cv_analysis as fca  # noqa: E402
from research.flux import flux_features as ff  # noqa: E402
from research.flux import flux_stats as fs  # noqa: E402

DAILY_PARQUET = (
    "/nas/soils/swapstress/evaluation/flux_validation/flux_site_daily.parquet"
)
META_PARQUET = "/nas/soils/swapstress/evaluation/flux_validation/flux_site_meta.parquet"
OUT_DIR = Path("/nas/soils/swapstress/evaluation/flux_validation/stress_function")

# Response configs: (label, potential column, flux column).
ET_CFG = {"label": "ET", "potential": "et0", "flux": "et_corr"}
GPP_CFG = {"label": "GPP", "potential": "gpp_pot", "flux": "gpp"}

# Predictors (all L4 surface + Rosetta L4) — fixed transforms of θ, matched β
# complexity. REW is the empirical per-site percentile normalization.
PREDICTORS_CV = {
    "psi_rf": "suction_l4",
    "theta": "theta_l4_surf",
    "rew": "rew_theta_l4_surf",
    "psi_ptf": "suction_ptf_l4",
}
PREDICTORS_TRANSFER = {
    "psi_rf": "suction_l4",
    "theta": "theta_l4_surf",
    "rew": "rew_theta_l4_surf",
}

# Contrasts within each family (name, spec_a − spec_b; positive favors ψ_RF/ψ_obs).
CV_CONTRASTS = [
    ("C1_psi_vs_rew", "psi_rf", "rew"),
    ("C2_psi_vs_theta", "psi_rf", "theta"),
    ("C4_psi_vs_ptf", "psi_rf", "psi_ptf"),
]
TRANSFER_CONTRASTS = [
    ("C5_transfer_psi_vs_theta", "psi_rf", "theta"),
    ("C5_transfer_psi_vs_rew", "psi_rf", "rew"),
]
INSITU_CONTRASTS = [
    ("C3_ceiling_psiobs_vs_rew", "psi_obs", "rew"),
    ("C3b_ceiling_psiobs_vs_theta", "psi_obs", "theta"),
]

MET_ENV_COLS = ["sw_in", "t_avg", "vpd"]  # GPP envelope covariates (vpd optional)

# ReESH/AmeriFlux sites with lab-fitted VG params (C3 ceiling); see
# quartile_binned_mlr.run_insitu.
REESH_SITES = [
    "US-CPk",
    "US-CdM",
    "US-Cwt",
    "US-GLE",
    "US-Ha1",
    "US-Ha2",
    "US-HB2",
    "US-HB3",
    "US-Ho1",
    "US-Jo2",
    "US-Me2",
    "US-Me6",
    "US-MMS",
    "US-MOz",
    "US-NC2",
    "US-NC4",
    "US-NR1",
    "US-Seg",
    "US-SRM",
    "US-Syv",
    "US-UTM",
    "US-UTW",
    "US-Vcp",
    "US-Vt1",
    "US-Vt2",
    "US-WCr",
    "US-Wjs",
]

# In-situ CV is relaxed (short AmeriFlux records) relative to the satellite CV.
INSITU_N_BLOCKS = 4
INSITU_MIN_DAYS = 100


# ---------------------------------------------------------------------------
# Inputs & frame prep
# ---------------------------------------------------------------------------


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.read_parquet(DAILY_PARQUET)
    meta = pd.read_parquet(META_PARQUET)
    cov_df = ff.build_site_covariates(meta[["site_id", "lat", "lon"]])
    return daily, cov_df


def add_gpp_potential(gs: pd.DataFrame) -> pd.DataFrame:
    """Add a per-site quantile-regression GPP envelope column ``gpp_pot`` (§2.2).

    Uses ``sw_in`` + ``t_avg`` (+ ``vpd`` where a site covers it well) as the met
    predictors of the τ=0.90 upper envelope. Sites without a stable fit get NaN
    ``gpp_pot`` and drop out of the GPP analyses.
    """
    gs = gs.copy()
    gs["gpp_pot"] = np.nan
    if "gpp" not in gs.columns:
        return gs
    for site, idx in gs.groupby("site_id").groups.items():
        sdf = gs.loc[idx]
        cols = ["sw_in", "t_avg"]
        if "vpd" in sdf.columns and sdf["vpd"].notna().mean() > 0.8:
            cols = MET_ENV_COLS
        pot = fbm.gpp_potential(sdf, cols, tau=0.90)
        if pot is not None:
            gs.loc[idx, "gpp_pot"] = pot
    return gs


def prep_frame(daily: pd.DataFrame) -> pd.DataFrame:
    """Growing-season slice with ET0, REW and the per-site GPP envelope."""
    gs = fca.prep_growing_season(daily)  # adds et0, applies the growing filter
    gs = ff.rew_empirical(gs, "theta_l4_surf")  # -> rew_theta_l4_surf
    gs = add_gpp_potential(gs)
    return gs


# ---------------------------------------------------------------------------
# Per-site CV (C1 / C2 / C4)
# ---------------------------------------------------------------------------


def run_per_site_cv(
    gs: pd.DataFrame, cfg: dict, predictors: dict, family: str = "logistic"
) -> pd.DataFrame:
    """Per-site blocked-CV β skill for every predictor (identical rows/site)."""
    sub = gs.dropna(subset=[cfg["flux"], cfg["potential"]])
    rows = []
    for site, sdf in sub.groupby("site_id"):
        r = fbm.per_site_beta_skills(
            sdf, cfg["potential"], cfg["flux"], predictors, family=family
        )
        if r is not None:
            r["site_id"] = site
            rows.append(r)
    return pd.DataFrame(rows)


def summarize_cv(per_site: pd.DataFrame, cfg: dict, predictors: dict) -> pd.DataFrame:
    """Median CV / apparent skill + bootstrap CI per predictor."""
    rows = []
    for label in predictors:
        cv = per_site.get(f"cv_{label}")
        app = per_site.get(f"app_{label}")
        if cv is None:
            continue
        med, lo, hi = fs.site_bootstrap_ci(cv.values)
        rows.append(
            {
                "response": cfg["label"],
                "predictor": label,
                "n_sites": int(cv.notna().sum()),
                "cv_skill_med": med,
                "cv_skill_lo": lo,
                "cv_skill_hi": hi,
                "apparent_skill_med": float(app.median())
                if app is not None
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LOSO transfer (C5)
# ---------------------------------------------------------------------------


def run_transfer(gs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per-held-out-site transfer skill of a single global β per predictor."""
    frame = gs.dropna(subset=[cfg["flux"], cfg["potential"]])
    skills = fbm.loso_beta_transfer(
        frame, cfg["potential"], cfg["flux"], PREDICTORS_TRANSFER, family="logistic"
    )
    common = (
        set.intersection(*[set(v.keys()) for v in skills.values()]) if skills else set()
    )
    common = sorted(common)
    rows = []
    for site in common:
        row = {"response": cfg["label"], "site_id": site}
        for label in PREDICTORS_TRANSFER:
            row[f"transfer_{label}"] = skills[label][site]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# In-situ ceiling (C3)
# ---------------------------------------------------------------------------


def build_insitu_frame() -> pd.DataFrame | None:
    """Assemble in-situ ReESH daily frame with observed-VG ψ (C3 ceiling).

    Reuses the VG-inversion + AmeriFlux BASE loaders from ``quartile_binned_mlr``:
    lab-fitted VG params invert in-situ θ to ψ_obs (no model), paired with tower
    ET/GPP and met. Returns a frame with ``site_id, date, theta, psi_obs,
    rew_theta, et0, et_corr, gpp`` or None if no site assembles.
    """
    from research.flux import quartile_binned_mlr as qbm

    vg = qbm._load_vg_params()
    frames = []
    for site in REESH_SITES:
        norm = site.upper().replace("-", "")
        if norm not in vg:
            continue
        daily = qbm._load_ameriflux_site(site)
        if daily is None or "et" not in daily.columns:
            continue
        if "sw_in" not in daily.columns or "t_avg" not in daily.columns:
            continue
        daily = daily.dropna(subset=["theta", "sw_in", "t_avg"])
        if len(daily) < INSITU_MIN_DAYS:
            continue
        p = vg[norm]
        daily["psi_obs"] = qbm._vg_invert(
            daily["theta"].values, p["theta_r"], p["theta_s"], p["alpha"], p["n_vg"]
        )
        daily["et0"] = ff.priestley_taylor_et0(
            daily["sw_in"].values, daily["t_avg"].values
        )
        daily = daily[(daily["t_avg"] > 5.0) & (daily["sw_in"] > 100.0)]
        daily = daily[daily["et0"] > 0.5]
        if len(daily) < INSITU_MIN_DAYS:
            continue
        daily["site_id"] = site
        daily["et_corr"] = daily["et"]
        frames.append(daily)
    if not frames:
        return None
    frame = pd.concat(frames, ignore_index=True)
    frame = ff.rew_empirical(frame, "theta", out_col="rew_theta")
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def run_insitu_ceiling(insitu: pd.DataFrame) -> pd.DataFrame:
    """Per-site relaxed-CV β skill for {ψ_obs, θ, REW} on the in-situ frame (C3)."""
    predictors = {"psi_obs": "psi_obs", "theta": "theta", "rew": "rew_theta"}
    sub = insitu.dropna(subset=["et_corr", "et0", *predictors.values()])
    rows = []
    for site, sdf in sub.groupby("site_id"):
        r = fbm.per_site_beta_skills(
            sdf,
            "et0",
            "et_corr",
            predictors,
            family="logistic",
            n_blocks=INSITU_N_BLOCKS,
            min_cv_days=INSITU_MIN_DAYS,
        )
        if r is not None:
            r["site_id"] = site
            rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# E1 — non-leaky depth (antecedent-weighted L3 surface θ)
# ---------------------------------------------------------------------------

ANT_TAUS = (7, 15, 30, 60)


def run_depth_e1(daily: pd.DataFrame) -> pd.DataFrame:
    """β CV skill of an L4-free root-zone θ proxy vs instantaneous surface θ (E1).

    The proxy is an antecedent-weighted (causal, gap-aware EWMA) SMAP **L3**
    surface θ — no Catchment-LSM, so no PTF leakage. Reports, per response, the
    best antecedent-τ CV skill against instantaneous L3 surface θ, both in the
    multiplicative β framing. (ψ_RF on the antecedent-θ needs a fresh model
    inference pass and is deferred; the driven-bucket route is left as future
    work — see plan §6.)
    """
    gs = fca.prep_growing_season(daily)
    gs = ff.add_antecedent_weighted(gs, "theta_l3", taus=ANT_TAUS, prefix="theta_l3")
    ant_cols = [f"theta_l3_ant{t}" for t in ANT_TAUS]
    rows = []
    for label, flux_col, pot_col in [("ET", "et_corr", "et0"), ("GPP", "gpp", None)]:
        if flux_col not in gs.columns:
            continue
        if pot_col is None:
            gs_r = add_gpp_potential(gs)
            pot_col = "gpp_pot"
        else:
            gs_r = gs
        sub = gs_r.dropna(subset=[flux_col, pot_col, "theta_l3", *ant_cols])
        inst, best_ant = [], {t: [] for t in ANT_TAUS}
        for _, sdf in sub.groupby("site_id"):
            pot = sdf[pot_col].values.astype(np.float64)
            flux = sdf[flux_col].values.astype(np.float64)
            dates = sdf["date"].values
            if len(sdf) < fbm.MIN_CV_DAYS:
                continue
            s_inst = fbm.beta_cv_skill(
                sdf["theta_l3"].values.astype(np.float64), pot, flux, dates, polarity=1
            )
            if not np.isfinite(s_inst):
                continue
            inst.append(s_inst)
            for t in ANT_TAUS:
                s = fbm.beta_cv_skill(
                    sdf[f"theta_l3_ant{t}"].values.astype(np.float64),
                    pot,
                    flux,
                    dates,
                    polarity=1,
                )
                best_ant[t].append(s if np.isfinite(s) else np.nan)
        if not inst:
            continue
        inst = np.array(inst)
        # Pick the τ with the best median antecedent skill.
        tau_meds = {t: np.nanmedian(best_ant[t]) for t in ANT_TAUS}
        best_tau = max(tau_meds, key=lambda t: tau_meds[t])
        ant = np.array(best_ant[best_tau])
        m = np.isfinite(inst) & np.isfinite(ant)
        delta = ant[m] - inst[m]
        med, lo, hi = fs.site_bootstrap_ci(delta)
        st = fs.sign_test(delta, alternative="greater")
        rows.append(
            {
                "response": label,
                "n_sites": int(m.sum()),
                "best_tau_days": best_tau,
                "skill_instant_l3_med": float(np.nanmedian(inst)),
                "skill_antecedent_med": float(np.nanmedian(ant[m])),
                "ant_minus_instant_med": med,
                "ant_minus_instant_lo": lo,
                "ant_minus_instant_hi": hi,
                "ant_sign_p": st["p"],
                "ci_excludes_zero": bool(np.isfinite(lo) and lo > 0),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# β parameters per site (E2 stratification input)
# ---------------------------------------------------------------------------


def fit_beta_param_table(gs: pd.DataFrame, cov_df: pd.DataFrame) -> pd.DataFrame:
    """Fitted logistic β(ψ_RF) parameters per site (ET) + site covariates (E2)."""
    sub = gs.dropna(subset=["et_corr", "et0", "suction_l4"])
    pol = fbm.predictor_polarity("suction_l4")
    rows = []
    for site, sdf in sub.groupby("site_id"):
        if len(sdf) < fbm.MIN_CV_DAYS:
            continue
        x_eff = pol * sdf["suction_l4"].values.astype(np.float64)
        params = fbm.fit_beta_params(
            x_eff, sdf["et0"].values, sdf["et_corr"].values, "logistic"
        )
        if params is None:
            continue
        readable = fbm.beta_params_readable("logistic", params, pol)
        readable["site_id"] = site
        readable["n_days"] = len(sdf)
        rows.append(readable)
    tbl = pd.DataFrame(rows)
    if tbl.empty:
        return tbl
    keep = ["site_id", "sand", "clay", "texture_usda", "aridity_index"]
    keep = [c for c in keep if c in cov_df.columns]
    return tbl.merge(cov_df[keep], on="site_id", how="left")


# ---------------------------------------------------------------------------
# Contrast synthesis (C1–C5, BH-FDR, PASS/FAIL)
# ---------------------------------------------------------------------------


def _contrast_stats(deltas: np.ndarray) -> dict:
    deltas = np.asarray(deltas, dtype=np.float64)
    deltas = deltas[np.isfinite(deltas)]
    point, lo, hi = fs.site_bootstrap_ci(deltas)
    st = fs.sign_test(deltas, alternative="greater")
    return {
        "n_sites": int(len(deltas)),
        "delta_median": point,
        "delta_lo": lo,
        "delta_hi": hi,
        "frac_a_wins": st["frac_pos"],
        "sign_p": st["p"],
    }


def _deltas_from_per_site(df: pd.DataFrame, col_a: str, col_b: str) -> np.ndarray:
    if col_a not in df.columns or col_b not in df.columns:
        return np.array([])
    return (df[col_a] - df[col_b]).values


def build_stress_contrasts(tables: dict) -> pd.DataFrame:
    """Assemble C1–C5 with effect sizes, BH-FDR across the family, PASS/FAIL."""
    rows = []

    def add(hyp, resp, contrast, deltas, note):
        st = _contrast_stats(deltas)
        rows.append(
            {
                "hypothesis": hyp,
                "response": resp,
                "contrast": contrast,
                **st,
                "ci_excludes_zero_positive": bool(
                    np.isfinite(st["delta_lo"]) and st["delta_lo"] > 0
                ),
                "notes": note,
            }
        )

    # C1/C2/C4 — per-site CV.
    for cfg_label, key in [("ET", "_cv_per_site_ET"), ("GPP", "_cv_per_site_GPP")]:
        ps = tables.get(key)
        if ps is None or ps.empty:
            continue
        for cname, a, b in CV_CONTRASTS:
            d = _deltas_from_per_site(ps, f"cv_{a}", f"cv_{b}")
            add(
                cname.split("_")[0],
                cfg_label.lower(),
                cname,
                d,
                f"per-site blocked-CV Δskill β({a})−β({b})",
            )

    # C5 — LOSO transfer.
    for cfg_label, key in [("ET", "_transfer_ET"), ("GPP", "_transfer_GPP")]:
        ts = tables.get(key)
        if ts is None or ts.empty:
            continue
        for cname, a, b in TRANSFER_CONTRASTS:
            d = _deltas_from_per_site(ts, f"transfer_{a}", f"transfer_{b}")
            add(
                "C5", cfg_label.lower(), cname, d, f"LOSO transfer Δskill β({a})−β({b})"
            )

    # C3 — in-situ ceiling (ET only).
    ps = tables.get("_insitu_per_site")
    if ps is not None and not ps.empty:
        for cname, a, b in INSITU_CONTRASTS:
            d = _deltas_from_per_site(ps, f"cv_{a}", f"cv_{b}")
            add(cname.split("_")[0], "et", cname, d, f"in-situ CV Δskill β({a})−β({b})")

    conf = pd.DataFrame(rows)
    if conf.empty:
        return conf
    rej, adj = fs.benjamini_hochberg(conf["sign_p"].values, alpha=0.05)
    conf["p_bh"] = adj
    conf["bh_reject"] = rej
    conf["verdict"] = np.where(
        conf["ci_excludes_zero_positive"] & conf["bh_reject"], "PASS", "FAIL"
    )
    return conf


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def figA_beta_curves(gs: pd.DataFrame, out: Path) -> Path:
    """Fitted β(ψ) vs β(REW) vs β(θ) on pooled ET data, with empirical clouds."""
    sub = gs.dropna(
        subset=["et_corr", "et0", "suction_l4", "theta_l4_surf", "rew_theta_l4_surf"]
    )
    if len(sub) > 40000:
        sub = sub.sample(40000, random_state=0)
    pot = sub["et0"].values.astype(np.float64)
    flux = sub["et_corr"].values.astype(np.float64)
    emp_beta = np.clip(flux / np.maximum(pot, 1e-6), 0, fbm.BETA_MAX_CAP)
    specs = [
        ("ψ_RF (pF)", "suction_l4"),
        ("θ (L4 surf)", "theta_l4_surf"),
        ("REW", "rew_theta_l4_surf"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (title, col) in zip(axes, specs):
        x = sub[col].values.astype(np.float64)
        pol = fbm.predictor_polarity(col)
        ax.scatter(x, emp_beta, s=3, alpha=0.05, color="#999999")
        params = fbm.fit_beta_params(pol * x, pot, flux, "logistic")
        if params is not None:
            grid = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 200)
            ax.plot(
                grid,
                fbm.predict_beta("logistic", pol * grid, params),
                color="#ee6677",
                lw=2,
            )
        ax.set_xlabel(title)
        ax.set_ylim(0, min(fbm.BETA_MAX_CAP, 1.6))
        ax.axhline(1.0, color="k", lw=0.4, ls=":")
    axes[0].set_ylabel("β = ET / ET0")
    fig.suptitle("Fig A — Fitted multiplicative stress functions β(x) (pooled ET)")
    return _save(fig, out / "figA_beta_curves.png")


def figB_transfer(tables: dict, out: Path) -> Path:
    """LOSO transfer skill by predictor (ET) — the decision figure."""
    ts = tables.get("_transfer_ET")
    fig, ax = plt.subplots(figsize=(6, 4.2))
    if ts is not None and not ts.empty:
        labels = list(PREDICTORS_TRANSFER)
        meds, los, his = [], [], []
        for label in labels:
            v = ts[f"transfer_{label}"].values
            m, lo, hi = fs.site_bootstrap_ci(v)
            meds.append(m)
            los.append(lo)
            his.append(hi)
        x = np.arange(len(labels))
        meds = np.array(meds)
        err = np.vstack([meds - np.array(los), np.array(his) - meds])
        ax.bar(x, meds, color=["#66ccee", "#4477aa", "#ccbb44"])
        ax.errorbar(x, meds, yerr=err, fmt="none", ecolor="k", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(["β(ψ_RF)", "β(θ)", "β(REW)"])
        ax.set_ylabel("median LOSO transfer skill (ET)")
        ax.axhline(0, color="k", lw=0.6)
    ax.set_title("Fig B — Global β transfers to unseen sites (C5)")
    return _save(fig, out / "figB_transfer_skill.png")


def figC_ceiling(tables: dict, out: Path) -> Path:
    """Ceiling: in-situ β(ψ_obs) vs β(REW)/β(θ) beside satellite β(ψ_RF) vs β(REW)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    groups, meds, colors = [], [], []
    ins = tables.get("_insitu_per_site")
    if ins is not None and not ins.empty:
        for label, c in [
            ("psi_obs", "#228833"),
            ("theta", "#4477aa"),
            ("rew", "#ccbb44"),
        ]:
            col = f"cv_{label}"
            if col in ins.columns:
                groups.append(f"in-situ β({label})")
                meds.append(float(ins[col].median()))
                colors.append(c)
    et_cv = tables.get("_cv_per_site_ET")
    if et_cv is not None and not et_cv.empty:
        for label, c in [("psi_rf", "#66ccee"), ("rew", "#eebb44")]:
            col = f"cv_{label}"
            if col in et_cv.columns:
                groups.append(f"sat β({label})")
                meds.append(float(et_cv[col].median()))
                colors.append(c)
    if groups:
        x = np.arange(len(groups))
        ax.bar(x, meds, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("median per-site CV skill")
        ax.axhline(0, color="k", lw=0.6)
    ax.set_title("Fig C — Ceiling: perfect ψ (in-situ) vs RF ψ vs REW")
    return _save(fig, out / "figC_ceiling.png")


def figD_threshold_strat(tables: dict, out: Path) -> Path:
    """β stress-onset (x0 in ψ pF) by aridity / texture (E2)."""
    bp = tables.get("stress_beta_params")
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    if bp is not None and not bp.empty and "x0_native" in bp.columns:
        m = bp["x0_native"].notna()
        if "aridity_index" in bp.columns and bp["aridity_index"].notna().any():
            mm = m & bp["aridity_index"].notna()
            sc = ax.scatter(
                bp.loc[mm, "aridity_index"],
                bp.loc[mm, "x0_native"],
                c=bp.loc[mm, "sand"] if "sand" in bp.columns else None,
                cmap="YlOrBr",
                s=18,
            )
            if "sand" in bp.columns:
                fig.colorbar(sc, ax=ax, label="sand %")
            ax.set_xlabel("aridity index (P/PET)")
        else:
            ax.hist(bp.loc[m, "x0_native"], bins=25, color="#66ccee")
            ax.set_xlabel("β stress-onset ψ (pF)")
        ax.set_ylabel("β stress-onset ψ (pF)")
    ax.set_title("Fig D — Stress-onset threshold varies across sites (E2)")
    return _save(fig, out / "figD_threshold_strat.png")


def figE_example(gs: pd.DataFrame, out: Path) -> Path:
    """Example site: empirical β = ET/ET0 vs ψ with the fitted β curve."""
    sub = gs.dropna(subset=["et_corr", "et0", "suction_l4"])
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    if not sub.empty:
        rng = sub.groupby("site_id")["suction_l4"].agg(lambda v: v.max() - v.min())
        ex = rng.idxmax()
        s = sub[sub["site_id"] == ex]
        pot = s["et0"].values.astype(np.float64)
        flux = s["et_corr"].values.astype(np.float64)
        x = s["suction_l4"].values.astype(np.float64)
        emp = np.clip(flux / np.maximum(pot, 1e-6), 0, fbm.BETA_MAX_CAP)
        ax.scatter(x, emp, s=10, alpha=0.4, color="#4477aa", label="ET/ET0")
        pol = fbm.predictor_polarity("suction_l4")
        params = fbm.fit_beta_params(pol * x, pot, flux, "logistic")
        if params is not None:
            grid = np.linspace(x.min(), x.max(), 200)
            ax.plot(
                grid,
                fbm.predict_beta("logistic", pol * grid, params),
                color="#ee6677",
                lw=2,
                label="β(ψ) fit",
            )
        ax.set_xlabel("ψ_RF (pF)")
        ax.set_ylabel("β = ET / ET0")
        ax.legend(fontsize=8)
        ax.set_title(f"Fig E — Example stress response: {ex}")
    return _save(fig, out / "figE_example.png")


def figF_depth(tables: dict, out: Path) -> Path:
    """Non-leaky depth ladder: instantaneous L3 vs best antecedent-L3 β skill (E1)."""
    dc = tables.get("stress_depth_e1")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    if dc is not None and not dc.empty:
        resp = list(dc["response"])
        x = np.arange(len(resp))
        w = 0.38
        ax.bar(
            x - w / 2,
            dc["skill_instant_l3_med"].values,
            w,
            label="β(θ instant L3)",
            color="#4477aa",
        )
        ax.bar(
            x + w / 2,
            dc["skill_antecedent_med"].values,
            w,
            label="β(θ antecedent L3, L4-FREE)",
            color="#228833",
        )
        for xi, row in zip(x, dc.itertuples()):
            ax.annotate(
                f"Δ={row.ant_minus_instant_med:+.3f}\nτ={row.best_tau_days}d",
                (xi, max(row.skill_instant_l3_med, row.skill_antecedent_med)),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=7,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(resp)
        ax.set_ylabel("median per-site CV skill")
        ax.axhline(0, color="k", lw=0.6)
        ax.legend(fontsize=8)
    ax.set_title("Fig F — Non-leaky depth: antecedent-L3 θ vs instantaneous (E1)")
    return _save(fig, out / "figF_depth_ladder.png")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------


def run_all(daily: pd.DataFrame, cov_df: pd.DataFrame) -> dict:
    tables: dict[str, pd.DataFrame] = {}
    gs = prep_frame(daily)
    tables["_gs"] = gs

    # C1/C2/C4 — per-site CV for ET & GPP.
    cv_summ = []
    for cfg in (ET_CFG, GPP_CFG):
        ps = run_per_site_cv(gs, cfg, PREDICTORS_CV)
        tables[f"_cv_per_site_{cfg['label']}"] = ps
        if not ps.empty:
            cv_summ.append(summarize_cv(ps, cfg, PREDICTORS_CV))
    tables["stress_function_cv"] = (
        pd.concat(cv_summ, ignore_index=True) if cv_summ else pd.DataFrame()
    )

    # C5 — LOSO transfer.
    transfer_all = []
    for cfg in (ET_CFG, GPP_CFG):
        ts = run_transfer(gs, cfg)
        tables[f"_transfer_{cfg['label']}"] = ts
        if not ts.empty:
            transfer_all.append(ts)
    tables["stress_transfer_loso"] = (
        pd.concat(transfer_all, ignore_index=True) if transfer_all else pd.DataFrame()
    )

    # C3 — in-situ ceiling.
    insitu = build_insitu_frame()
    if insitu is not None:
        tables["_insitu_per_site"] = run_insitu_ceiling(insitu)
        tables["_insitu_frame"] = insitu
    else:
        tables["_insitu_per_site"] = pd.DataFrame()

    # E1 — non-leaky depth.
    tables["stress_depth_e1"] = run_depth_e1(daily)

    # E2 — β params per site + covariates.
    tables["stress_beta_params"] = fit_beta_param_table(gs, cov_df)

    # Confirmatory synthesis.
    tables["stress_contrasts"] = build_stress_contrasts(tables)
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading inputs …")
    daily, cov_df = load_inputs()
    print(f"  daily rows={len(daily):,}  sites={daily['site_id'].nunique()}")

    print("Running multiplicative stress-function analyses …")
    tables = run_all(daily, cov_df)

    written = []
    for name, df in tables.items():
        if name.startswith("_") or df is None or df.empty:
            continue
        path = out / f"{name}.csv"
        df.to_csv(path, index=False)
        written.append(path)

    conf = tables["stress_contrasts"]
    print("\nConfirmatory contrasts (C1–C5, BH-FDR across the family):")
    if not conf.empty:
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(
                conf[
                    [
                        "hypothesis",
                        "response",
                        "contrast",
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

    print("\nTables written:")
    for p in written:
        print(f"  {p}")

    if not args.no_figures:
        gs = tables["_gs"]
        print("\nFigures:")
        figs = [
            figA_beta_curves(gs, out),
            figB_transfer(tables, out),
            figC_ceiling(tables, out),
            figD_threshold_strat(tables, out),
            figE_example(gs, out),
            figF_depth(tables, out),
        ]
        for p in figs:
            print(f"  {p}")


if __name__ == "__main__":
    main()
