"""
Cross-site transferability — WS2 / H1 / H2 (the scientific centerpiece).

The per-site OLS null (ψ ≈ θ) is unsurprising: within one site θ→ψ is monotonic,
so OLS just rescales θ to mimic ψ. ψ (matric potential) is physically *comparable
across soil textures* in a way θ is not — ψ = −1.5 MPa means wilting everywhere,
whereas θ = 0.20 is saturated sand or dry clay. That comparability can only pay
off when a **single global stress function** must serve *unseen* sites. This
module runs that test: pooled leave-one-site-out (LOSO) regression, held-out
skill by predictor.

Two framings are reported (plan §2 H1 / WS2):

- **anomaly** (primary, conservative): response de-meaned within each site and
  met standardized within site, so site-level ET differences (vegetation,
  climate) cannot be spuriously attributed to soil water. Tests whether a global
  soil-water *response slope* transfers. Fully out-of-sample.
- **regime** (raw levels): a single global function incl. intercept predicts the
  held-out site's ET. Cross-site regime variance is in play — where ψ's
  texture-invariance can most help — but it is more exposed to non-soil site
  confounds (met covariates absorb the first-order climate differences).

Primary contrasts:
  H1: skill(met+ψ) − skill(met+θ)      — ψ's fair chance vs raw θ
  H2: skill(met+ψ) − skill(met+REW)    — ψ vs any texture normalization

The mechanistic test is the texture-distance stratification: ψ should help most
where the held-out site is texturally *unlike* the training set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.flux import flux_features as ff
from research.flux import flux_stats as fs

# Growing-season thresholds (shared with the rest of the flux validation).
T_AVG_MIN = 5.0
SW_IN_MIN = 100.0
MET_COLS = ["sw_in", "t_avg"]
MIN_SITE_DAYS = 60  # a held-out site needs enough days for a stable skill estimate

# Predictor specs: label -> list of soil-water columns added to the met base.
# All use the L4 surface pair (θ and ψ from the *same* surface retrieval) so the
# only difference between met+θ and met+ψ is the retention-curve transform.
SPECS = {
    "met": [],
    "met+theta": ["theta_l4_surf"],
    "met+rew": ["rew_theta_l4_surf"],
    "met+psi": ["suction_l4"],
    "met+theta+psi": ["theta_l4_surf", "suction_l4"],
    "met+theta_root": ["theta_l4_root"],
    "met+psi_prof": ["suction_l4_prof"],
}

CONTRASTS = [
    ("H1_psi_vs_theta", "met+psi", "met+theta"),
    ("H2_psi_vs_rew", "met+psi", "met+rew"),
    ("psi_prof_vs_theta_root", "met+psi_prof", "met+theta_root"),
    ("combined_vs_theta", "met+theta+psi", "met+theta"),
]


# ---------------------------------------------------------------------------
# Frame preparation
# ---------------------------------------------------------------------------


def prepare_transfer_frame(
    daily: pd.DataFrame,
    response_col: str,
    extra_pred_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Growing-season slice with ET0, REW and all predictor columns present.

    Drops rows missing the response, met covariates, or any soil-water predictor
    used by the specs, so every spec is fit on an identical row set per site.
    """
    gs = daily[(daily["t_avg"] > T_AVG_MIN) & (daily["sw_in"] > SW_IN_MIN)].copy()
    gs = ff.rew_empirical(gs, "theta_l4_surf")

    pred_cols = sorted({c for cols in SPECS.values() for c in cols})
    if extra_pred_cols:
        pred_cols = sorted(set(pred_cols) | set(extra_pred_cols))
    needed = [response_col] + MET_COLS + pred_cols
    gs = gs.dropna(subset=needed)

    # Keep only sites with enough growing-season days for a stable estimate.
    counts = gs.groupby("site_id").size()
    keep = counts[counts >= MIN_SITE_DAYS].index
    gs = gs[gs["site_id"].isin(keep)].copy()
    return gs


# ---------------------------------------------------------------------------
# Within-site transforms (anomaly framing)
# ---------------------------------------------------------------------------


def _within_site_transform(
    df: pd.DataFrame,
    y_col: str,
    met_cols: list[str],
    demean_response: bool,
    standardize_met: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Apply per-site transforms once (each site by its own stats).

    Returns (transformed_df, y) with a ``_y`` column. Because every transform is
    within-site, LOSO can then just leave a site's rows out — the training-site
    transforms are unaffected by which site is held out.
    """
    df = df.copy()
    g = df.groupby("site_id")

    if demean_response:
        df["_y"] = df[y_col] - g[y_col].transform("mean")
    else:
        df["_y"] = df[y_col].astype(np.float64)

    for c in met_cols:
        if standardize_met:
            mean = g[c].transform("mean")
            sd = g[c].transform("std").replace(0, np.nan)
            df[f"_{c}"] = (df[c] - mean) / sd
        else:
            df[f"_{c}"] = df[c].astype(np.float64)

    # Soil-water predictors keep their native cross-site scale (only globally
    # standardized for conditioning) — that cross-site level is exactly where ψ's
    # texture-invariance can help; within-site normalization would erase it.
    df = df.dropna(subset=["_y"] + [f"_{c}" for c in met_cols])
    return df, df["_y"].values.astype(np.float64)


def _global_standardize(df: pd.DataFrame, cols: list[str]) -> dict[str, np.ndarray]:
    out = {}
    for c in cols:
        v = df[c].values.astype(np.float64)
        mu, sd = np.nanmean(v), np.nanstd(v)
        out[c] = (v - mu) / (sd if sd > 1e-12 else 1.0)
    return out


# ---------------------------------------------------------------------------
# LOSO skill for one spec
# ---------------------------------------------------------------------------


def loso_skill_by_site(
    df: pd.DataFrame,
    y_col: str,
    met_cols: list[str],
    pred_cols: list[str],
    demean_response: bool,
    standardize_met: bool,
) -> dict[str, float]:
    """Per-held-out-site out-of-sample skill of a single global model.

    Fits OLS on all *other* sites, predicts the held-out site. Skill baseline is
    0 in the anomaly framing (response de-meaned per site) or the training-y mean
    in the regime framing — so skill can be negative.
    """
    work, y = _within_site_transform(
        df, y_col, met_cols, demean_response, standardize_met
    )
    sites = work["site_id"].values
    met_t = (
        np.column_stack([work[f"_{c}"].values for c in met_cols])
        if met_cols
        else np.empty((len(work), 0))
    )
    pred_std = _global_standardize(work, pred_cols) if pred_cols else {}
    pred_t = (
        np.column_stack([pred_std[c] for c in pred_cols])
        if pred_cols
        else np.empty((len(work), 0))
    )
    X = (
        np.column_stack([met_t, pred_t])
        if (met_cols or pred_cols)
        else np.empty((len(work), 0))
    )
    y = work["_y"].values.astype(np.float64)

    uniq = pd.unique(sites)
    skills: dict[str, float] = {}
    for test_site in uniq:
        test_mask = sites == test_site
        train_mask = ~test_mask
        n_test = int(test_mask.sum())
        if n_test < MIN_SITE_DAYS or train_mask.sum() < X.shape[1] + 5:
            continue

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask], y[test_mask]

        if X.shape[1] == 0:
            # met-only-less baseline: predict training mean.
            y_hat = np.full(n_test, y_tr.mean())
        else:
            y_hat, y_tr_mean = fs._fit_predict(X_tr, y_tr, X_te)
            if y_hat is None:
                continue

        ss_res = float(((y_te - y_hat) ** 2).sum())
        if demean_response:
            ss_tot = float((y_te**2).sum())  # baseline = site mean = 0
        else:
            ss_tot = float(((y_te - y_tr.mean()) ** 2).sum())
        if ss_tot < 1e-12:
            continue
        skills[test_site] = 1.0 - ss_res / ss_tot
    return skills


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------


def run_transferability(
    daily: pd.DataFrame,
    cov_df: pd.DataFrame | None = None,
    responses=(("et", "et_corr"), ("gpp", "gpp")),
    framings=("anomaly", "regime"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run LOSO transferability for all specs, responses and framings.

    Returns ``(summary, per_site)``:
      - ``summary``: one row per (response, framing, contrast) with median Δskill,
        site-bootstrap CI, sign-test p, BH-adjusted p, and the texture-distance
        correlation of the per-site Δ.
      - ``per_site``: per-(response, framing, site) skill for every spec plus the
        contrast Δs and each site's texture distance to the rest of the set.
    """
    tex_z = ff.texture_vector(cov_df) if cov_df is not None else pd.DataFrame()

    per_site_rows = []
    summary_rows = []

    for resp_label, resp_col in responses:
        if resp_col not in daily.columns:
            continue
        frame = prepare_transfer_frame(daily, resp_col)
        n_sites = frame["site_id"].nunique()
        if n_sites < 10:
            continue

        for framing in framings:
            demean = framing == "anomaly"
            standardize_met = framing == "anomaly"

            # Per-site skill for each spec. Every spec includes the met base;
            # they differ only in which soil-water predictors are appended.
            spec_skill: dict[str, dict[str, float]] = {}
            for spec_label, pred_cols in SPECS.items():
                spec_skill[spec_label] = loso_skill_by_site(
                    frame,
                    resp_col,
                    MET_COLS,
                    pred_cols,
                    demean_response=demean,
                    standardize_met=standardize_met,
                )

            # Sites evaluated across all specs (intersection for fair contrasts).
            common = set.intersection(*[set(s.keys()) for s in spec_skill.values()])
            common = sorted(common)

            # Texture distance per held-out site (vs the rest of the common set).
            tex_dist = {}
            if not tex_z.empty:
                for site in common:
                    others = [s for s in common if s != site]
                    tex_dist[site] = ff.texture_distance_to_set(tex_z, site, others)

            for site in common:
                row = {
                    "response": resp_label,
                    "framing": framing,
                    "site_id": site,
                    "texture_dist": tex_dist.get(site, np.nan),
                }
                for spec_label in SPECS:
                    row[f"skill_{spec_label}"] = spec_skill[spec_label][site]
                per_site_rows.append(row)

            # Contrasts.
            for cname, spec_a, spec_b in CONTRASTS:
                deltas = np.array(
                    [spec_skill[spec_a][s] - spec_skill[spec_b][s] for s in common]
                )
                point, lo, hi = fs.site_bootstrap_ci(deltas, n_boot=10000)
                st = fs.sign_test(deltas, alternative="greater")
                # Texture-distance correlation of the per-site delta.
                if not tex_z.empty:
                    dists = np.array([tex_dist.get(s, np.nan) for s in common])
                    m = np.isfinite(dists) & np.isfinite(deltas)
                    if m.sum() >= 5:
                        tex_corr = float(np.corrcoef(dists[m], deltas[m])[0, 1])
                    else:
                        tex_corr = np.nan
                else:
                    tex_corr = np.nan

                summary_rows.append(
                    {
                        "response": resp_label,
                        "framing": framing,
                        "contrast": cname,
                        "spec_a": spec_a,
                        "spec_b": spec_b,
                        "n_sites": len(common),
                        "delta_median": point,
                        "delta_lo": lo,
                        "delta_hi": hi,
                        "frac_a_wins": st["frac_pos"],
                        "sign_p": st["p"],
                        "texture_dist_corr": tex_corr,
                        "ci_excludes_zero": bool(np.isfinite(lo) and lo > 0),
                    }
                )

    summary = pd.DataFrame(summary_rows)
    per_site = pd.DataFrame(per_site_rows)

    # BH-FDR across the confirmatory transfer family (H1 + H2 across responses,
    # anomaly framing only — the pre-registered primary).
    if not summary.empty:
        summary["bh_reject"] = False
        fam_mask = (summary["framing"] == "anomaly") & summary["contrast"].isin(
            ["H1_psi_vs_theta", "H2_psi_vs_rew"]
        )
        fam = summary[fam_mask]
        if not fam.empty:
            rej, adj = fs.benjamini_hochberg(fam["sign_p"].values, alpha=0.05)
            summary.loc[fam.index, "bh_reject"] = rej
            summary.loc[fam.index, "sign_p_bh"] = adj

    return summary, per_site
