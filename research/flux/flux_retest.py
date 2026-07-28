"""Flux-validation retest orchestrator (M6 — synthesis).

Runs every confirmatory + exploratory test from
``notes/flux_robustness_retest_plan.md`` on the assembled daily flux table and
writes the slide-ready artifacts to ``regression/``:

Tables
  - ``tiered_summary_cv.csv``            H3 honest complementarity (WS1) + WS7 GPP splits
  - ``transferability_summary.csv``      H1/H2 LOSO transferability (WS2) — the centerpiece
  - ``transferability_per_site.csv``     per-held-out-site skills (feeds Fig 3/6)
  - ``matched_information.csv``          H5 transform-vs-multi-depth (WS3)
  - ``depth_controls.csv``               H4b antecedent-L3 (L4-free) + OOS depth ladder
  - ``water_limitation_sensitivity.csv`` WS5 dry-regime sensitivity (3 definitions)
  - ``threshold_stress.csv``             E3 exploratory threshold models (WS6)
  - ``confirmatory_results.csv``         pre-registered H1–H5 PASS/FAIL, effect sizes, BH-FDR

Figures (PNG, absolute paths printed on render)
  Fig 1 variance ladder · Fig 2 honest complementarity + noise floor ·
  Fig 3 transferability money figure · Fig 4 depth ladder (L4-free flagged) ·
  Fig 5 transform-vs-multi-depth · Fig 6 where ψ helps + example dry-down.

Nothing here fits models on GPU or hits Earth Engine — it is pure post-hoc
statistics over the cached parquet, safe to run directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from map.evaluation import flux_cv_analysis as fca  # noqa: E402
from map.evaluation import flux_features as ff  # noqa: E402
from map.evaluation import flux_stats as fs  # noqa: E402
from map.evaluation import flux_stress_models as fsm  # noqa: E402
from map.evaluation import flux_transferability as ft  # noqa: E402

DAILY_PARQUET = (
    "/nas/soils/swapstress/evaluation/flux_validation/flux_site_daily.parquet"
)
META_PARQUET = "/nas/soils/swapstress/evaluation/flux_validation/flux_site_meta.parquet"
OUT_DIR = Path("/nas/soils/swapstress/evaluation/flux_validation/regression")

GPP_SOURCES = ["ameriflux_hh", "icos_fullset"]

# The pre-registered primary config for the per-site complementarity figure.
H3_PRIMARY = ("et_corr", "theta_l4_surf", "suction_l4")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.read_parquet(DAILY_PARQUET)
    meta = pd.read_parquet(META_PARQUET)
    cov_df = ff.build_site_covariates(meta[["site_id", "lat", "lon"]])
    return daily, cov_df


# ---------------------------------------------------------------------------
# Run all analyses
# ---------------------------------------------------------------------------


def run_all(daily: pd.DataFrame, cov_df: pd.DataFrame, n_perm: int) -> dict:
    tables: dict[str, pd.DataFrame] = {}

    # WS1 — H3 honest complementarity (all sites) + WS7 GPP-source splits.
    h3_all = fca.run_h3_complementarity(daily, n_perm=n_perm)
    h3_splits = []
    for src in GPP_SOURCES:
        s = fca.run_h3_complementarity(daily, gpp_source=src, n_perm=n_perm)
        s = s[s["response"] == "GPP"]  # only GPP has a source split
        h3_splits.append(s)
    tables["tiered_summary_cv"] = pd.concat([h3_all, *h3_splits], ignore_index=True)

    # WS2 — H1/H2 transferability (the centerpiece).
    transfer_summary, transfer_per_site = ft.run_transferability(daily, cov_df=cov_df)
    tables["transferability_summary"] = transfer_summary
    tables["transferability_per_site"] = transfer_per_site

    # WS3 — H5 transform vs multi-depth.
    tables["matched_information"] = fca.run_h5_matched_info(daily)

    # WS4b + depth ladder — H4 depth, L4-free route + honest OOS depth ladder.
    h4b = fca.run_h4b_antecedent_l3(daily)
    ladder = fca.run_depth_ladder_cv(daily)
    h4b["block"] = "antecedent_l3_l4free"
    ladder["block"] = "depth_ladder_l4based"
    tables["depth_controls"] = pd.concat([h4b, ladder], ignore_index=True)

    # WS5 — water-limitation sensitivity across 3 dry definitions.
    tables["water_limitation_sensitivity"] = fca.run_water_limitation_sensitivity(daily)

    # WS6 / E3 — exploratory threshold stress models.
    tables["threshold_stress"] = fsm.run_threshold_stress(daily)

    # Per-site H3 for the primary config (feeds Fig 2).
    tables["_h3_primary_per_site"] = _h3_primary_per_site(daily, n_perm)

    return tables


def _h3_primary_per_site(daily: pd.DataFrame, n_perm: int) -> pd.DataFrame:
    resp_col, theta_col, psi_col = H3_PRIMARY
    gs = fca.prep_growing_season(daily)
    sub = gs.dropna(subset=[resp_col, theta_col, psi_col])
    rows = []
    for site, sdf in sub.groupby("site_id"):
        r = fca.per_site_complementarity(sdf, theta_col, psi_col, resp_col, n_perm)
        if r is not None:
            r["site_id"] = site
            rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Confirmatory synthesis
# ---------------------------------------------------------------------------


def _row(hyp, statement, response, comparison, n, eff, lo, hi, p, notes):
    return {
        "hypothesis": hyp,
        "statement": statement,
        "response": response,
        "comparison": comparison,
        "n_sites": n,
        "effect_size": eff,
        "ci_lo": lo,
        "ci_hi": hi,
        "p_one_sided": p,
        "ci_excludes_zero_positive": bool(np.isfinite(lo) and lo > 0),
        "notes": notes,
    }


def build_confirmatory(tables: dict) -> pd.DataFrame:
    """Assemble the pre-registered H1–H5 with effect sizes, one-sided p, and
    (BH-corrected across the whole family) PASS/FAIL verdicts."""
    ts = tables["transferability_summary"]
    rows = []

    # H1 — transferability, ψ vs raw θ (anomaly framing, primary).
    for resp in ("et", "gpp"):
        r = ts[
            (ts["response"] == resp)
            & (ts["framing"] == "anomaly")
            & (ts["contrast"] == "H1_psi_vs_theta")
        ]
        if not r.empty:
            r = r.iloc[0]
            rows.append(
                _row(
                    "H1_transferability",
                    "LOSO skill(met+psi) > skill(met+theta)",
                    resp,
                    "met+psi vs met+theta (anomaly)",
                    int(r["n_sites"]),
                    r["delta_median"],
                    r["delta_lo"],
                    r["delta_hi"],
                    r["sign_p"],
                    f"frac_psi_wins={r['frac_a_wins']:.2f}; texture_dist_corr={r['texture_dist_corr']:.3f}",
                )
            )

    # H2 — normalization vs transform: ψ vs REW (anomaly).
    for resp in ("et", "gpp"):
        r = ts[
            (ts["response"] == resp)
            & (ts["framing"] == "anomaly")
            & (ts["contrast"] == "H2_psi_vs_rew")
        ]
        if not r.empty:
            r = r.iloc[0]
            rows.append(
                _row(
                    "H2_normalization",
                    "LOSO skill(met+psi) > skill(met+REW)",
                    resp,
                    "met+psi vs met+rew (anomaly)",
                    int(r["n_sites"]),
                    r["delta_median"],
                    r["delta_lo"],
                    r["delta_hi"],
                    r["sign_p"],
                    "negative delta => REW (cheap texture normalization) matches/beats psi",
                )
            )

    # H3 — honest per-site complementarity: best config per response, CV Δ + perm.
    h3 = tables["tiered_summary_cv"]
    h3 = h3[h3["gpp_source"] == "all"] if "gpp_source" in h3.columns else h3
    for resp in ("ET", "GPP"):
        sub = h3[h3["response"] == resp]
        if sub.empty:
            continue
        best = sub.loc[sub["delta_cv_psi_med"].idxmax()]
        rows.append(
            _row(
                "H3_complementarity",
                "per-site blocked-CV R2(met+theta+psi) > R2(met+theta)",
                resp.lower(),
                f"CV delta psi|theta, best config: {best['predictors']}",
                int(best["n_sites"]),
                best["delta_cv_psi_med"],
                best["delta_cv_lo"],
                best["delta_cv_hi"],
                best["perm_p_median"],
                f"frac_perm_sig={best['frac_perm_sig']:.2f}; "
                f"apparent_in_sample={best['r2_apparent_psi_beyond_theta']:.4f} "
                "(the rigged number)",
            )
        )

    # H4 — depth, L4-free route: antecedent-L3 beats instantaneous L3.
    h4b = tables["depth_controls"]
    h4b = h4b[h4b["block"] == "antecedent_l3_l4free"]
    for _, r in h4b.iterrows():
        rows.append(
            _row(
                "H4_depth_l4free",
                "antecedent-weighted L3 surface theta > instantaneous L3 (no LSM)",
                str(r["response"]).lower(),
                f"L3 antecedent(tau={int(r['best_tau_days'])}d) vs instantaneous L3",
                int(r["n_sites"]),
                r["ant_minus_instant_med"],
                r["ant_minus_instant_lo"],
                r["ant_minus_instant_hi"],
                r["ant_sign_p"],
                f"frac_root_gain_recovered={r['frac_root_gain_recovered']:.2f}",
            )
        )

    # H5 — transform vs multi-depth: ψ_prof beyond {θ_surf, θ_root}.
    h5 = tables["matched_information"]
    for _, r in h5.iterrows():
        rows.append(
            _row(
                "H5_transform_vs_multidepth",
                "CV R2 gain of psi_prof beyond {theta_surf, theta_root} > 0",
                str(r["response"]).lower(),
                "psi_prof beyond matched two-depth theta",
                int(r["n_sites"]),
                r["t1_prof_beyond_2theta_med"],
                r["t1_lo"],
                r["t1_hi"],
                r["t1_sign_p"],
                f"t2_matched_2depth_psi_minus_theta={r['t2_delta_med']:.4f}",
            )
        )

    conf = pd.DataFrame(rows)
    # BH-FDR across the whole confirmatory family (plan §3.5 / line 208).
    p = conf["p_one_sided"].values.astype(float)
    rej, adj = fs.benjamini_hochberg(p, alpha=0.05)
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


def fig1_variance_ladder(tables: dict, out: Path) -> Path:
    """met → +θ → +θ+ψ median CV skill for ET & GPP (H3 configs, L4 surface)."""
    h3 = tables["tiered_summary_cv"]
    h3 = h3[(h3.get("gpp_source", "all") == "all") & (h3["predictors"] == "L4 surface")]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    responses = list(h3["response"])
    x = np.arange(len(responses))
    w = 0.38
    theta = h3["cv_r2_theta_med"].values
    both = h3["cv_r2_both_med"].values
    ax.bar(x - w / 2, theta, w, label="met + θ", color="#4477aa")
    ax.bar(x + w / 2, both, w, label="met + θ + ψ", color="#66ccee")
    for xi, t, b in zip(x, theta, both):
        ax.annotate(
            f"Δψ|θ = {b - t:+.4f}",
            (xi, max(t, b)),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(responses)
    ax.set_ylabel("median per-site blocked-CV skill")
    ax.set_title("Fig 1 — Variance ladder: soil water is a second-order term")
    ax.axhline(0, color="k", lw=0.6)
    ax.legend()
    return _save(fig, out / "fig1_variance_ladder.png")


def fig2_honest_complementarity(tables: dict, out: Path) -> Path:
    """Per-site CV Δψ|θ distribution with the block-permutation noise floor."""
    ps = tables["_h3_primary_per_site"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    delta = ps["delta_cv_psi"].dropna().values
    floor = ps["perm_floor"].dropna().values
    ax.hist(delta, bins=30, color="#66ccee", alpha=0.8, label="observed CV Δψ|θ")
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(
        np.median(delta),
        color="#ee6677",
        lw=2,
        label=f"median = {np.median(delta):+.4f}",
    )
    ax.axvline(
        float(np.median(floor)),
        color="#228833",
        ls="--",
        lw=2,
        label=f"perm. noise floor = {np.median(floor):+.4f}",
    )
    # Clip the view to the bulk (a couple of outlier sites otherwise dominate).
    lo_x, hi_x = np.percentile(delta, [1, 97])
    pad = 0.02
    n_clip = int(np.sum((delta < lo_x - pad) | (delta > hi_x + pad)))
    ax.set_xlim(min(lo_x - pad, -0.03), hi_x + pad)
    ax.set_xlabel("per-site CV ΔR²  (ψ added beyond θ)")
    ax.set_ylabel("sites")
    ax.set_title("Fig 2 — Honest complementarity (ET, L4 surface): the retraction")
    if n_clip:
        ax.annotate(
            f"{n_clip} outlier site(s) beyond axis",
            (0.98, 0.55),
            xycoords="axes fraction",
            ha="right",
            fontsize=7,
            color="#555555",
        )
    ax.legend(fontsize=8)
    return _save(fig, out / "fig2_honest_complementarity.png")


def fig3_transferability(tables: dict, out: Path) -> Path:
    """Money figure: LOSO skill by predictor + Δ(ψ−θ), Δ(ψ−REW) vs texture dist."""
    ps = tables["transferability_per_site"]
    et = ps[(ps["response"] == "et") & (ps["framing"] == "anomaly")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    # Left: median held-out skill by predictor.
    specs = ["skill_met", "skill_met+theta", "skill_met+rew", "skill_met+psi"]
    labels = ["met", "+θ", "+REW", "+ψ"]
    meds = [et[s].median() for s in specs]
    los, his = [], []
    for s in specs:
        _, lo, hi = fs.site_bootstrap_ci(et[s].dropna().values)
        los.append(lo)
        his.append(hi)
    xx = np.arange(len(specs))
    meds = np.array(meds)
    err = np.vstack([meds - np.array(los), np.array(his) - meds])
    axes[0].bar(xx, meds, color=["#bbbbbb", "#4477aa", "#ccbb44", "#66ccee"])
    axes[0].errorbar(xx, meds, yerr=err, fmt="none", ecolor="k", capsize=3)
    axes[0].set_xticks(xx)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("median LOSO held-out skill (ET, anomaly)")
    axes[0].set_title("Held-out skill by predictor")
    axes[0].axhline(0, color="k", lw=0.6)

    # Right: per-site Δ vs texture distance.
    et["d_psi_theta"] = et["skill_met+psi"] - et["skill_met+theta"]
    et["d_psi_rew"] = et["skill_met+psi"] - et["skill_met+rew"]
    m = et["texture_dist"].notna()
    axes[1].scatter(
        et.loc[m, "texture_dist"],
        et.loc[m, "d_psi_theta"],
        s=14,
        color="#66ccee",
        label="ψ − θ",
    )
    axes[1].scatter(
        et.loc[m, "texture_dist"],
        et.loc[m, "d_psi_rew"],
        s=14,
        color="#ccbb44",
        marker="^",
        label="ψ − REW",
    )
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_xlabel("held-out site texture distance to training set")
    axes[1].set_ylabel("Δ held-out skill")
    axes[1].set_title("Where ψ earns its keep")
    axes[1].legend(fontsize=8)
    fig.suptitle("Fig 3 — Transferability (money figure)")
    return _save(fig, out / "fig3_transferability.png")


def fig4_depth_ladder(tables: dict, out: Path) -> Path:
    """Surface vs root vs (L4-free antecedent-L3) depth skill, L4-free flagged."""
    dc = tables["depth_controls"]
    ladder = dc[dc["block"] == "depth_ladder_l4based"]
    h4b = dc[dc["block"] == "antecedent_l3_l4free"]
    fig, ax = plt.subplots(figsize=(7.5, 4.3))

    lad_theta = ladder[ladder["variable"] == "theta"]
    responses = list(lad_theta["response"])
    x = np.arange(len(responses))
    w = 0.26
    ax.bar(
        x - w,
        lad_theta["skill_surface_med"].values,
        w,
        label="θ surface (L4)",
        color="#4477aa",
    )
    ax.bar(
        x,
        lad_theta["skill_root_med"].values,
        w,
        label="θ root-zone (L4)",
        color="#aa3377",
    )
    # L4-free antecedent-L3 best skill.
    ant = h4b.set_index("response")
    ant_vals = [
        ant.loc[r, "skill_l3_ant_best_med"] if r in ant.index else np.nan
        for r in responses
    ]
    ax.bar(x + w, ant_vals, w, label="θ antecedent-L3 (L4-FREE)", color="#228833")
    ax.set_xticks(x)
    ax.set_xticklabels(responses)
    ax.set_ylabel("median per-site CV skill")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(
        "Fig 4 — Depth ladder (green bar is L4-free; root-zone gain does not survive OOS)"
    )
    ax.legend(fontsize=8)
    return _save(fig, out / "fig4_depth_ladder.png")


def fig5_transform_vs_multidepth(tables: dict, out: Path) -> Path:
    """H5 decomposition: profile-ψ beyond matched multi-depth θ."""
    h5 = tables["matched_information"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    responses = list(h5["response"])
    x = np.arange(len(responses))
    w = 0.38
    t1 = h5["t1_prof_beyond_2theta_med"].values
    t2 = h5["t2_delta_med"].values
    b1 = ax.bar(
        x - w / 2, t1, w, label="ψ_prof beyond {θ_surf,θ_root}", color="#66ccee"
    )
    b2 = ax.bar(x + w / 2, t2, w, label="matched 2-depth: ψ − θ", color="#ccbb44")
    ax.errorbar(
        x - w / 2,
        t1,
        yerr=np.vstack([t1 - h5["t1_lo"].values, h5["t1_hi"].values - t1]),
        fmt="none",
        ecolor="k",
        capsize=3,
    )
    ax.errorbar(
        x + w / 2,
        t2,
        yerr=np.vstack([t2 - h5["t2_lo"].values, h5["t2_hi"].values - t2]),
        fmt="none",
        ecolor="k",
        capsize=3,
    )
    ax.bar_label(b1, fmt="%+.4f", padding=3, fontsize=7)
    ax.bar_label(b2, fmt="%+.4f", padding=3, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(responses)
    ax.set_ylabel("CV ΔR²")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(
        "Fig 5 — Transform vs multi-depth: the gain is depth, not the ψ transform"
    )
    ax.legend(fontsize=8)
    return _save(fig, out / "fig5_transform_vs_multidepth.png")


def fig6_where_psi_helps(
    tables: dict, cov_df: pd.DataFrame, daily: pd.DataFrame, out: Path
) -> Path:
    """Stratify transfer Δ(ψ−θ) by aridity + one example dry-down time series."""
    ps = tables["transferability_per_site"]
    et = ps[(ps["response"] == "et") & (ps["framing"] == "anomaly")].copy()
    et["d_psi_theta"] = et["skill_met+psi"] - et["skill_met+theta"]
    merged = et.merge(cov_df[["site_id", "aridity_index"]], on="site_id", how="left")
    merged = merged.dropna(subset=["aridity_index", "d_psi_theta"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    # Left: Δ(ψ−θ) by aridity tercile.
    if len(merged) >= 6:
        merged["arid_bin"] = pd.qcut(
            merged["aridity_index"], 3, labels=["dry", "mid", "humid"]
        )
        grp = merged.groupby("arid_bin", observed=True)["d_psi_theta"]
        meds = grp.median()
        axes[0].bar(range(len(meds)), meds.values, color="#66ccee")
        axes[0].set_xticks(range(len(meds)))
        axes[0].set_xticklabels(list(meds.index))
        axes[0].set_ylabel("median Δ held-out skill (ψ − θ)")
        axes[0].set_xlabel("aridity tercile")
        axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_title("ψ advantage by aridity")

    # Right: example dry-down at the site with the widest θ range.
    gs = fca.prep_growing_season(daily)
    site_ids = et["site_id"].tolist()
    cand = gs[gs["site_id"].isin(site_ids)].dropna(
        subset=["et_corr", "theta_l4_surf", "suction_l4"]
    )
    if not cand.empty:
        rng = cand.groupby("site_id")["theta_l4_surf"].agg(lambda v: v.max() - v.min())
        ex_site = rng.idxmax()
        sdf = cand[cand["site_id"] == ex_site].copy()
        sdf["date"] = pd.to_datetime(sdf["date"])
        sdf = sdf.sort_values("date")
        # Single busiest year for legibility.
        sdf = sdf[sdf["date"].dt.year == sdf["date"].dt.year.mode().iloc[0]]
        ax = axes[1]
        ax.plot(sdf["date"], sdf["theta_l4_surf"], color="#4477aa", label="θ (L4 surf)")
        ax.set_ylabel("θ (m³/m³)", color="#4477aa")
        ax2 = ax.twinx()
        ax2.plot(sdf["date"], sdf["suction_l4"], color="#aa3377", label="ψ (pF)")
        ax2.set_ylabel("ψ (pF)", color="#aa3377")
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("outward", 44))
        ax3.plot(
            sdf["date"], sdf["et_corr"], color="#228833", lw=0.8, alpha=0.7, label="ET"
        )
        ax3.set_ylabel("ET (mm/d)", color="#228833")
        ax.set_title(f"Example dry-down: {ex_site}")
        ax.tick_params(axis="x", rotation=30, labelsize=7)
    fig.suptitle("Fig 6 — Where ψ helps")
    return _save(fig, out / "fig6_where_psi_helps.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--n-perm", type=int, default=fca.N_PERM)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading inputs …")
    daily, cov_df = load_inputs()
    print(f"  daily rows={len(daily):,}  sites={daily['site_id'].nunique()}")

    print("Running analyses …")
    tables = run_all(daily, cov_df, n_perm=args.n_perm)

    # Write tables (skip the private per-site helper frame).
    written = []
    for name, df in tables.items():
        if name.startswith("_") or df is None or df.empty:
            continue
        path = out / f"{name}.csv"
        df.to_csv(path, index=False)
        written.append(path)

    conf = build_confirmatory(tables)
    conf_path = out / "confirmatory_results.csv"
    conf.to_csv(conf_path, index=False)
    written.append(conf_path)

    print("\nConfirmatory results (pre-registered H1–H5):")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(
            conf[
                [
                    "hypothesis",
                    "response",
                    "effect_size",
                    "ci_lo",
                    "ci_hi",
                    "p_one_sided",
                    "bh_reject",
                    "verdict",
                ]
            ].to_string(index=False)
        )

    print("\nTables written:")
    for p in written:
        print(f"  {p}")

    if not args.no_figures:
        print("\nFigures:")
        figs = [
            fig1_variance_ladder(tables, out),
            fig2_honest_complementarity(tables, out),
            fig3_transferability(tables, out),
            fig4_depth_ladder(tables, out),
            fig5_transform_vs_multidepth(tables, out),
            fig6_where_psi_helps(tables, cov_df, daily, out),
        ]
        for p in figs:
            print(f"  {p}")


if __name__ == "__main__":
    main()
