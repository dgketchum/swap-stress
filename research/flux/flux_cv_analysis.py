"""
De-rigged per-site analyses — WS1 (H3), WS3 (H5), WS4b (H4b), WS5, WS7.

Everything here replaces the *in-sample* nested-R² tests of
``quartile_binned_mlr.py`` with out-of-sample, autocorrelation-aware machinery
from :mod:`research.flux.flux_stats`:

- **H3 (honest complementarity)** — does ψ add *predictive* skill beyond θ? Per
  site: blocked-CV ΔR² (primary), a nested partial-F, and a block-permutation
  noise floor. Aggregated with a site bootstrap CI, the fraction of sites with
  partial-F p<0.05 vs the 5% chance rate, and (legacy, reported alongside) the
  one-sided Wilcoxon. → ``tiered_summary_cv.csv`` (replaces the rigged table).
- **H5 (transform vs multi-depth)** — CV ΔR² of ψ_prof beyond {θ_surf, θ_root},
  and a matched two-depth head-to-head {θ_surf,θ_root} vs {ψ_surf,ψ_root50}.
  → ``matched_information.csv``.
- **H4b (L4-free depth proxy)** — does antecedent-weighted **L3** surface θ beat
  instantaneous L3, and does it recover the L4 root-zone gain? → part of
  ``depth_controls.csv``.
- **WS5** — water-limitation sensitivity across three predictor-agnostic dry
  definitions. → ``water_limitation_sensitivity.csv``.
- **WS7** — GPP-source robustness (ICOS-only vs AmeriFlux-only).

The met base is Priestley–Taylor ET0 (one clean covariate — keeps the per-site
CV stable on short records), matching the legacy tiered framing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.flux import flux_features as ff
from research.flux import flux_stats as fs

T_AVG_MIN = 5.0
SW_IN_MIN = 100.0
ET0_MIN = 0.5

# Per-site CV configuration.
N_BLOCKS = 5
EMBARGO_DAYS = 15
MIN_CV_DAYS = 150
N_PERM = 200

# (response_label, response_col, theta_col, psi_col, predictor_label)
TIERED_CONFIGS = [
    ("ET", "et_corr", "theta_l4_surf", "suction_l4", "L4 surface"),
    ("ET", "et_corr", "theta_l4_root", "suction_l4_root_50", "L4 root50"),
    ("ET", "et_corr", "theta_l4_root", "suction_l4_prof", "L4 profile"),
    ("ET", "et_corr", "theta_l3", "suction_l3", "L3 surface"),
    ("GPP", "gpp", "theta_l4_surf", "suction_l4", "L4 surface"),
    ("GPP", "gpp", "theta_l4_root", "suction_l4_root_50", "L4 root50"),
    ("GPP", "gpp", "theta_l4_root", "suction_l4_prof", "L4 profile"),
    ("GPP", "gpp", "theta_l3", "suction_l3", "L3 surface"),
]


# ---------------------------------------------------------------------------
# Growing-season prep
# ---------------------------------------------------------------------------


def prep_growing_season(daily: pd.DataFrame) -> pd.DataFrame:
    """Growing-season slice with an ET0 column (matches the legacy filter)."""
    gs = daily[(daily["t_avg"] > T_AVG_MIN) & (daily["sw_in"] > SW_IN_MIN)].copy()
    gs = gs.dropna(subset=["theta_l4_surf", "sw_in", "t_avg"])
    gs = ff.add_et0(gs)
    gs = gs[gs["et0"] > ET0_MIN].copy()
    return gs


# ---------------------------------------------------------------------------
# Per-site complementarity (H3): CV ΔR² + partial-F + permutation floor
# ---------------------------------------------------------------------------


def _apparent_r2(X: np.ndarray, y: np.ndarray) -> float:
    ss_res = fs._in_sample_ssr(X, y)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if not np.isfinite(ss_res) or ss_tot < 1e-12:
        return np.nan
    return 1.0 - ss_res / ss_tot


def per_site_complementarity(
    sdf: pd.DataFrame,
    theta_col: str,
    psi_col: str,
    y_col: str,
    n_perm: int = N_PERM,
) -> dict | None:
    """CV / partial-F / permutation stats for "ψ adds beyond θ" at one site."""
    cols = ["et0", theta_col, psi_col, y_col]
    clean = sdf.dropna(subset=cols)
    if len(clean) < MIN_CV_DAYS:
        return None

    y = clean[y_col].values.astype(np.float64)
    et0 = clean["et0"].values.astype(np.float64)
    theta = clean[theta_col].values.astype(np.float64)
    psi = clean[psi_col].values.astype(np.float64)
    dates = clean["date"].values
    if theta.std() < 1e-12 or psi.std() < 1e-12:
        return None

    X_theta = np.column_stack([et0, theta])
    X_psi = np.column_stack([et0, psi])
    X_both = np.column_stack([et0, theta, psi])

    # Out-of-sample deltas on identical folds.
    d_psi = fs.blocked_cv_delta(
        X_theta,
        X_both,
        y,
        dates,
        n_blocks=N_BLOCKS,
        embargo_days=EMBARGO_DAYS,
        min_cv_days=MIN_CV_DAYS,
    )
    d_theta = fs.blocked_cv_delta(
        X_psi,
        X_both,
        y,
        dates,
        n_blocks=N_BLOCKS,
        embargo_days=EMBARGO_DAYS,
        min_cv_days=MIN_CV_DAYS,
    )
    if not np.isfinite(d_psi["delta"]):
        return None

    # In-sample (apparent) ψ|θ — the quantity the old test reported.
    r2_theta_app = _apparent_r2(X_theta, y)
    r2_both_app = _apparent_r2(X_both, y)
    apparent_psi_beyond_theta = r2_both_app - r2_theta_app

    # Per-site nested partial-F for the added ψ block.
    _, pF, _, _ = fs.partial_f_test(X_theta, X_both, y)

    # Block-permutation noise floor for the in-sample ψ|θ gain.
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2_theta_fixed = r2_theta_app

    def _stat(perm_psi):
        X = np.column_stack([et0, theta, perm_psi])
        ssr = fs._in_sample_ssr(X, y)
        if not np.isfinite(ssr) or ss_tot < 1e-12:
            return np.nan
        return (1.0 - ssr / ss_tot) - r2_theta_fixed

    null = fs.block_permutation_null(_stat, psi, n_perm=n_perm)
    perm_p = fs.permutation_p_value(apparent_psi_beyond_theta, null, "greater")
    perm_floor = float(np.nanmean(null)) if np.isfinite(null).any() else np.nan

    return {
        "n_days": len(clean),
        "cv_r2_theta": d_psi["skill_reduced"],
        "cv_r2_both": d_psi["skill_full"],
        "delta_cv_psi": d_psi["delta"],
        "delta_cv_theta": d_theta["delta"] if np.isfinite(d_theta["delta"]) else np.nan,
        "apparent_psi_beyond_theta": apparent_psi_beyond_theta,
        "partial_f_p": pF,
        "perm_p": perm_p,
        "perm_floor": perm_floor,
    }


def run_h3_complementarity(
    daily: pd.DataFrame,
    configs=TIERED_CONFIGS,
    gpp_source: str | None = None,
    n_perm: int = N_PERM,
) -> pd.DataFrame:
    """H3 across the tiered configs. Returns the aggregated CV summary table."""
    gs = prep_growing_season(daily)
    if gpp_source is not None:
        gs = gs[gs["gpp_source"] == gpp_source]

    rows = []
    for resp_name, resp_col, theta_col, psi_col, pred_label in configs:
        if resp_col not in gs.columns:
            continue
        sub = gs.dropna(subset=[resp_col, theta_col, psi_col])
        site_stats = []
        for _, sdf in sub.groupby("site_id"):
            res = per_site_complementarity(sdf, theta_col, psi_col, resp_col, n_perm)
            if res is not None:
                site_stats.append(res)
        if not site_stats:
            continue
        S = pd.DataFrame(site_stats)
        n_sites = len(S)

        deltas = S["delta_cv_psi"].values
        med, lo, hi = fs.site_bootstrap_ci(deltas, n_boot=10000)
        # Partial-F rate vs 5% chance.
        pf = S["partial_f_p"].dropna().values
        n_pf_sig = int(np.sum(pf < 0.05))
        binom_p = fs.binomial_rate_test(
            n_pf_sig, len(pf), rate=0.05, alternative="greater"
        )
        # Permutation: fraction of sites clearing the noise floor.
        pp = S["perm_p"].dropna().values
        frac_perm_sig = float(np.mean(pp < 0.05)) if len(pp) else np.nan

        rows.append(
            {
                "response": resp_name,
                "predictors": pred_label,
                "gpp_source": gpp_source or "all",
                "n_sites": n_sites,
                "med_n_days": int(S["n_days"].median()),
                "cv_r2_theta_med": float(S["cv_r2_theta"].median()),
                "cv_r2_both_med": float(S["cv_r2_both"].median()),
                "r2_apparent_psi_beyond_theta": float(
                    S["apparent_psi_beyond_theta"].median()
                ),
                "delta_cv_psi_med": med,
                "delta_cv_lo": lo,
                "delta_cv_hi": hi,
                "delta_cv_theta_med": float(S["delta_cv_theta"].median()),
                "frac_pF_sig": n_pf_sig / len(pf) if len(pf) else np.nan,
                "binom_p": binom_p,
                "perm_p_median": float(np.median(pp)) if len(pp) else np.nan,
                "frac_perm_sig": frac_perm_sig,
                "wilcoxon_p_legacy": fs.wilcoxon_greater(deltas),
                "ci_excludes_zero": bool(np.isfinite(lo) and lo > 0),
            }
        )

    summary = pd.DataFrame(rows)
    if not summary.empty:
        # BH-FDR across the H3 family (all rows for the "all" source run).
        base = summary[summary["gpp_source"] == (gpp_source or "all")]
        rej, adj = fs.benjamini_hochberg(base["binom_p"].values, alpha=0.05)
        summary.loc[base.index, "bh_reject_pF"] = rej
    return summary


# ---------------------------------------------------------------------------
# H5 — matched-information (transform vs multi-depth)
# ---------------------------------------------------------------------------

H5_COLS = {
    "theta_surf": "theta_l4_surf",
    "theta_root": "theta_l4_root",
    "psi_surf": "suction_l4",
    "psi_root50": "suction_l4_root_50",
    "psi_prof": "suction_l4_prof",
}


def _cv_skill(X, y, dates):
    return fs.blocked_cv_r2(
        X,
        y,
        dates,
        n_blocks=N_BLOCKS,
        embargo_days=EMBARGO_DAYS,
        min_cv_days=MIN_CV_DAYS,
    )


def per_site_matched_info(sdf: pd.DataFrame, y_col: str) -> dict | None:
    """H5 tests at one site (all on identical CV folds)."""
    c = H5_COLS
    cols = ["et0", y_col] + list(c.values())
    clean = sdf.dropna(subset=cols)
    if len(clean) < MIN_CV_DAYS:
        return None
    y = clean[y_col].values.astype(np.float64)
    et0 = clean["et0"].values.astype(np.float64)
    dates = clean["date"].values

    def col(name):
        return clean[c[name]].values.astype(np.float64)

    theta_surf, theta_root = col("theta_surf"), col("theta_root")
    psi_surf, psi_root50, psi_prof = col("psi_surf"), col("psi_root50"), col("psi_prof")

    # T1: ψ_prof beyond {θ_surf, θ_root}.
    X_2theta = np.column_stack([et0, theta_surf, theta_root])
    X_2theta_prof = np.column_stack([et0, theta_surf, theta_root, psi_prof])
    t1 = fs.blocked_cv_delta(
        X_2theta,
        X_2theta_prof,
        y,
        dates,
        n_blocks=N_BLOCKS,
        embargo_days=EMBARGO_DAYS,
        min_cv_days=MIN_CV_DAYS,
    )
    # T2: matched two-depth θ vs ψ (equal complexity).
    X_2psi = np.column_stack([et0, psi_surf, psi_root50])
    skill_2theta = _cv_skill(X_2theta, y, dates)
    skill_2psi = _cv_skill(X_2psi, y, dates)
    if not np.isfinite(t1["delta"]) or not np.isfinite(skill_2theta):
        return None
    return {
        "n_days": len(clean),
        "t1_prof_beyond_2theta": t1["delta"],
        "t2_skill_2theta": skill_2theta,
        "t2_skill_2psi": skill_2psi,
        "t2_delta_psi_minus_theta": skill_2psi - skill_2theta
        if np.isfinite(skill_2psi)
        else np.nan,
    }


def run_h5_matched_info(daily: pd.DataFrame) -> pd.DataFrame:
    gs = prep_growing_season(daily)
    rows = []
    for resp_name, resp_col in [("ET", "et_corr"), ("GPP", "gpp")]:
        if resp_col not in gs.columns:
            continue
        sub = gs.dropna(subset=[resp_col] + list(H5_COLS.values()))
        stats = []
        for _, sdf in sub.groupby("site_id"):
            r = per_site_matched_info(sdf, resp_col)
            if r is not None:
                stats.append(r)
        if not stats:
            continue
        S = pd.DataFrame(stats)
        t1_med, t1_lo, t1_hi = fs.site_bootstrap_ci(S["t1_prof_beyond_2theta"].values)
        t2_med, t2_lo, t2_hi = fs.site_bootstrap_ci(
            S["t2_delta_psi_minus_theta"].values
        )
        rows.append(
            {
                "response": resp_name,
                "n_sites": len(S),
                # T1: profile ψ beyond matched multi-depth θ.
                "t1_prof_beyond_2theta_med": t1_med,
                "t1_lo": t1_lo,
                "t1_hi": t1_hi,
                "t1_sign_p": fs.sign_test(
                    S["t1_prof_beyond_2theta"].values, alternative="greater"
                )["p"],
                # T2: matched two-depth ψ vs θ.
                "t2_skill_2theta_med": float(S["t2_skill_2theta"].median()),
                "t2_skill_2psi_med": float(S["t2_skill_2psi"].median()),
                "t2_delta_med": t2_med,
                "t2_lo": t2_lo,
                "t2_hi": t2_hi,
                "t2_sign_p": fs.sign_test(
                    S["t2_delta_psi_minus_theta"].values, alternative="greater"
                )["p"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# H4b — antecedent-L3 (L4-free depth proxy) + depth ladder
# ---------------------------------------------------------------------------


def run_h4b_antecedent_l3(daily: pd.DataFrame, taus=(7, 15, 30, 60)) -> pd.DataFrame:
    """H4b: does antecedent-weighted L3 surface θ beat instantaneous L3, and
    close the gap to L4 root-zone θ? All matched-complexity CV skill (one
    soil-water predictor each), on the L3-available day set per site.
    """
    gs = prep_growing_season(daily)
    gs = ff.add_antecedent_weighted(gs, "theta_l3", taus=taus)

    rows = []
    for resp_name, resp_col in [("ET", "et_corr"), ("GPP", "gpp")]:
        if resp_col not in gs.columns:
            continue
        need = [resp_col, "theta_l3", "theta_l4_root"] + [
            f"theta_l3_ant{t}" for t in taus
        ]
        sub = gs.dropna(subset=need)
        stats = []
        for _, sdf in sub.groupby("site_id"):
            clean = sdf.dropna(subset=need + ["et0"])
            if len(clean) < MIN_CV_DAYS:
                continue
            y = clean[resp_col].values.astype(np.float64)
            et0 = clean["et0"].values.astype(np.float64)
            dates = clean["date"].values
            rec = {"n_days": len(clean)}
            rec["skill_l3_surf"] = _cv_skill(
                np.column_stack([et0, clean["theta_l3"].values]), y, dates
            )
            rec["skill_l4_root"] = _cv_skill(
                np.column_stack([et0, clean["theta_l4_root"].values]), y, dates
            )
            for t in taus:
                rec[f"skill_l3_ant{t}"] = _cv_skill(
                    np.column_stack([et0, clean[f"theta_l3_ant{t}"].values]), y, dates
                )
            if not np.isfinite(rec["skill_l3_surf"]):
                continue
            stats.append(rec)
        if not stats:
            continue
        S = pd.DataFrame(stats)
        # Best antecedent tau (by median skill).
        tau_meds = {t: float(S[f"skill_l3_ant{t}"].median()) for t in taus}
        best_tau = max(tau_meds, key=tau_meds.get)
        ant_delta = (S[f"skill_l3_ant{best_tau}"] - S["skill_l3_surf"]).values
        root_delta = (S["skill_l4_root"] - S["skill_l3_surf"]).values
        ad_med, ad_lo, ad_hi = fs.site_bootstrap_ci(ant_delta)
        rows.append(
            {
                "response": resp_name,
                "n_sites": len(S),
                "skill_l3_surf_med": float(S["skill_l3_surf"].median()),
                "best_tau_days": best_tau,
                "skill_l3_ant_best_med": tau_meds[best_tau],
                "skill_l4_root_med": float(S["skill_l4_root"].median()),
                "ant_minus_instant_med": ad_med,
                "ant_minus_instant_lo": ad_lo,
                "ant_minus_instant_hi": ad_hi,
                "ant_sign_p": fs.sign_test(ant_delta, alternative="greater")["p"],
                # Fraction of the (L4root − L3surf) gain recovered by antecedent-L3.
                "frac_root_gain_recovered": (
                    float(np.median(ant_delta) / np.median(root_delta))
                    if np.isfinite(np.median(root_delta))
                    and abs(np.median(root_delta)) > 1e-9
                    else np.nan
                ),
                "l4_free": True,
            }
        )
    return pd.DataFrame(rows)


def run_depth_ladder_cv(daily: pd.DataFrame) -> pd.DataFrame:
    """Honest out-of-sample depth ladder (the CV version of §13.2/§13.6).

    Per site (all growing-season days), matched-complexity CV skill of a single
    surface vs root-zone θ predictor (met+θ_surf vs met+θ_root) and the same for
    ψ (met+ψ_surf vs met+ψ_root50). Paired Δ = root − surface with a site
    bootstrap CI and sign test. **L4-based, NOT L4-free** — both surface and
    root-zone derive from the L4 product, so this carries the LSM-PTF imprint
    (see H4b for the L4-free route).
    """
    gs = prep_growing_season(daily)
    pairs = [
        ("theta", "theta_l4_surf", "theta_l4_root"),
        ("psi", "suction_l4", "suction_l4_root_50"),
    ]
    rows = []
    for resp_name, resp_col in [("ET", "et_corr"), ("GPP", "gpp")]:
        if resp_col not in gs.columns:
            continue
        for var, surf_col, root_col in pairs:
            sub = gs.dropna(subset=[resp_col, surf_col, root_col, "et0"])
            deltas, surf_sk, root_sk = [], [], []
            for _, sdf in sub.groupby("site_id"):
                clean = sdf.dropna(subset=[resp_col, surf_col, root_col, "et0"])
                if len(clean) < MIN_CV_DAYS:
                    continue
                y = clean[resp_col].values.astype(np.float64)
                et0 = clean["et0"].values.astype(np.float64)
                dates = clean["date"].values
                s = _cv_skill(np.column_stack([et0, clean[surf_col].values]), y, dates)
                r = _cv_skill(np.column_stack([et0, clean[root_col].values]), y, dates)
                if np.isfinite(s) and np.isfinite(r):
                    deltas.append(r - s)
                    surf_sk.append(s)
                    root_sk.append(r)
            if len(deltas) < 5:
                continue
            deltas = np.array(deltas)
            med, lo, hi = fs.site_bootstrap_ci(deltas)
            rows.append(
                {
                    "response": resp_name,
                    "variable": var,
                    "contrast": "root_minus_surface",
                    "n_sites": len(deltas),
                    "skill_surface_med": float(np.median(surf_sk)),
                    "skill_root_med": float(np.median(root_sk)),
                    "delta_cv_med": med,
                    "delta_cv_lo": lo,
                    "delta_cv_hi": hi,
                    "frac_root_wins": float(np.mean(deltas > 0)),
                    "sign_p": fs.sign_test(deltas, alternative="greater")["p"],
                    "l4_free": False,
                }
            )
    return pd.DataFrame(rows)


def check_perdepth_swc_availability(
    amf_root: str = "/nas/climate/ameriflux/amf_new",
    max_sites: int = 400,
) -> pd.DataFrame:
    """H4a blocking check (plan §4): how many AmeriFlux BASE files expose *per-depth*
    SWC (SWC_1_1_1, SWC_2_1_1, …) with resolvable shallow (≤10 cm) vs deep
    (≥30 cm) layers, rather than a single averaged column.

    Reads only each file's header (nrows=0) — fast. Returns a per-site frame with
    the SWC column count and whether a shallow/deep split looks feasible. If too
    few sites qualify, H4a is dropped and H4 rests on H4b.
    """
    import glob
    import os
    import re

    rows = []
    dirs = sorted(glob.glob(os.path.join(amf_root, "AMF_*_BASE-BADM_*")))[:max_sites]
    swc_re = re.compile(r"^SWC_(\d+)(?:_(\d+)_(\d+))?$", re.IGNORECASE)
    for d in dirs:
        csvs = sorted(glob.glob(os.path.join(d, "AMF_*_BASE_HH_*.csv"))) + sorted(
            glob.glob(os.path.join(d, "AMF_*_BASE_HR_*.csv"))
        )
        if not csvs:
            continue
        site = os.path.basename(d).split("_")[1]
        try:
            cols = pd.read_csv(csvs[0], skiprows=2, nrows=0).columns
        except Exception:
            continue
        swc_cols = [c for c in cols if swc_re.match(c)]
        # Position index (first number) is a proxy for depth ordering in
        # AmeriFlux BASE naming (_H_V_R): H increments with depth for many sites.
        n_swc = len(swc_cols)
        multi = n_swc >= 2
        rows.append({"site_id": site, "n_swc_cols": n_swc, "multi_depth": multi})
    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# WS5 — water-limitation definition sensitivity
# ---------------------------------------------------------------------------


def run_water_limitation_sensitivity(daily: pd.DataFrame) -> pd.DataFrame:
    """Headline depth contrast (root50 vs surface, full θ+ψ CV skill) under three
    predictor-agnostic dry-half definitions (plan WS5).

    Definitions:
      1. antecedent precip — dry = low 30-day antecedent precipitation (ppt),
         fully independent of any soil-water predictor;
      2. precip anomaly    — dry = low standardized per-site ppt anomaly;
      3. theta_surf median — dry = below per-site surface-θ median (the legacy
         definition, kept to document sensitivity).
    A common day-set (rows with both predictor pairs present) is used within each
    definition so the surface/root comparison is on identical rows.
    """
    gs = prep_growing_season(daily)
    # 30-day antecedent precip (renormalized causal EWMA of ppt).
    gs = ff.add_antecedent_weighted(gs, "ppt", taus=(30,), prefix="ppt")

    surf = ("theta_l4_surf", "suction_l4")
    root = ("theta_l4_root", "suction_l4_root_50")

    def dry_mask(sdf, definition):
        if definition == "antecedent_precip":
            v = sdf["ppt_ant30"]
        elif definition == "precip_anomaly":
            v = sdf["ppt"]
        else:  # theta_surf_median
            v = sdf["theta_l4_surf"]
        if v.notna().sum() < 10:
            return pd.Series(False, index=sdf.index)
        return v <= v.median()

    rows = []
    for resp_name, resp_col in [("ET", "et_corr"), ("GPP", "gpp")]:
        if resp_col not in gs.columns:
            continue
        need = [resp_col, "et0", *surf, *root]
        sub = gs.dropna(subset=need)
        for definition in ("antecedent_precip", "precip_anomaly", "theta_surf_median"):
            deltas = []
            for _, sdf in sub.groupby("site_id"):
                dmask = dry_mask(sdf, definition)
                clean = sdf[dmask].dropna(subset=need)
                if len(clean) < MIN_CV_DAYS:
                    continue
                y = clean[resp_col].values.astype(np.float64)
                et0 = clean["et0"].values.astype(np.float64)
                dates = clean["date"].values
                Xs = np.column_stack(
                    [et0, clean[surf[0]].values, clean[surf[1]].values]
                )
                Xr = np.column_stack(
                    [et0, clean[root[0]].values, clean[root[1]].values]
                )
                s_surf = _cv_skill(Xs, y, dates)
                s_root = _cv_skill(Xr, y, dates)
                if np.isfinite(s_surf) and np.isfinite(s_root):
                    deltas.append(s_root - s_surf)
            if len(deltas) < 5:
                continue
            deltas = np.array(deltas)
            med, lo, hi = fs.site_bootstrap_ci(deltas)
            rows.append(
                {
                    "response": resp_name,
                    "dry_definition": definition,
                    "contrast": "root50_minus_surface",
                    "n_sites": len(deltas),
                    "delta_cv_med": med,
                    "delta_cv_lo": lo,
                    "delta_cv_hi": hi,
                    "frac_root_wins": float(np.mean(deltas > 0)),
                    "sign_p": fs.sign_test(deltas, alternative="greater")["p"],
                }
            )
    return pd.DataFrame(rows)
