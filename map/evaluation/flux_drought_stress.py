"""Dry-tail / drought-regime stress resolution — is the soil-water flux null
regime-specific?

The confirmatory flux tests (retest H1–H5) and the multiplicative stress-function
retest (:mod:`map.evaluation.flux_stress_retest`) both concluded that soil water
is *second-order to meteorology* for predicting ET/GPP out of sample. But those
tests average over the whole growing season, and on most growing-season days the
canopy is **not** water-limited: β ≈ β_max, the flux equals its potential, and
soil water has no stress variance to explain. If plant water stress is real but
concentrated in the dry tail, pooling over unstressed days dilutes it toward
invisibility. This module asks a sharper, pre-registered question:

    Does soil water resolve stress in the DRY regime that it cannot in the WET
    regime — i.e. is the "2nd-order" verdict a regime-averaged artifact?

Design (mirrors the retest machinery, CPU-only, cached parquet):

- **Regime split is predictor-agnostic.** Each site's growing-season days are
  split at the site median of an antecedent climate index that does *not* use the
  soil-water predictor under test — antecedent 30-day precipitation (primary) and
  an antecedent P−ET0 water-deficit (robustness). Splitting *within* site keeps
  each half balanced and removes cross-site climate confounding.

- **The baseline is constant-β, not the flux mean.** ``flux = c·potential`` (ET is
  a fixed fraction of ET0) already earns large skill because ET tracks demand.
  What we want is the *stress-shape* increment: how much a soil-water β(x) adds
  **over** that scalar. So skill here is ``Δ = cv[β(x)] − cv[const-β]`` on the same
  rows/folds. Δ > 0 means soil water resolves stress the demand scaling cannot.

- **Matched complexity, out of sample.** β(x) is the same logistic family
  (2 shape params + β_max) used everywhere; skill is blocked-CV R² on the retest
  fold structure, relaxed for the shorter per-regime records (fewer blocks, lower
  min-days). Aggregation is the standard site-bootstrap CI + sign test + BH-FDR.

- **ψ vs θ in the dry tail is reported but structurally bounded.** Here ψ
  (``suction_l4``) is a fixed PTF transform of θ (``theta_l4_surf``); within a site
  they carry identical information, so a flexible monotone β cannot separate them
  (data-processing inequality). We report ``cv[ψ] − cv[θ]`` per regime to confirm
  the redundancy, and a dynamic-range diagnostic showing the L4 surface θ does
  **not** collapse onto a residual plateau in the dry tail (unlike a physical
  retention curve near θ_r) — so the "ψ keeps resolving where θ saturates"
  mechanism is absent in the satellite product by construction and can only be
  tested with in-situ co-located ψ (LaCADIAN). See ``notes``.

Tables → ``…/flux_validation/drought_stress/``:
  ds_regime_skills.csv · ds_contrasts.csv · ds_interaction.csv · ds_dynrange.csv
Figures A–C (PNG, absolute paths printed on render).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from map.evaluation import flux_beta_models as fbm  # noqa: E402
from map.evaluation import flux_features as ff  # noqa: E402
from map.evaluation import flux_stats as fs  # noqa: E402
from map.evaluation import flux_stress_retest as frs  # noqa: E402

DAILY_PARQUET = (
    "/nas/soils/swapstress/evaluation/flux_validation/flux_site_daily.parquet"
)
OUT_DIR = Path("/nas/soils/swapstress/evaluation/flux_validation/drought_stress")

# Response configs reused from the stress-function retest.
RESPONSES = [frs.ET_CFG, frs.GPP_CFG]  # ET: et0/et_corr ; GPP: gpp_pot/gpp

# Soil-water predictors (all fixed transforms of L4 surface θ, matched β form).
PREDICTORS = {
    "theta": "theta_l4_surf",
    "theta_root": "theta_l4_root",
    "rew": "rew_theta_l4_surf",
    "psi": "suction_l4",
}
# The pre-registered "does soil water resolve stress" family (β(x) over const-β).
EXISTENCE_PREDICTORS = ["theta", "theta_root", "rew"]

# Predictor-agnostic dryness indices: (label, column, higher_is_drier).
DRY_INDICES = [
    ("precip", "ppt_ant30", False),  # low antecedent precip = dry
    ("deficit", "wdef_ant30", True),  # high antecedent (ET0−ppt) = dry
]
REGIMES = ("dry", "wet")

# Per-regime CV is relaxed vs the full-season retest (each regime is ~half a
# site's growing-season days).
DS_N_BLOCKS = 4
DS_MIN_DAYS = 80


# ---------------------------------------------------------------------------
# Frame prep: growing-season slice + REW + GPP envelope + dryness indices
# ---------------------------------------------------------------------------


def prep_frame(daily: pd.DataFrame) -> pd.DataFrame:
    """Growing-season frame (ET0, REW, GPP envelope) plus antecedent dryness."""
    gs = frs.prep_frame(daily)  # adds et0, rew_theta_l4_surf, gpp_pot
    gs = ff.add_antecedent_weighted(gs, "ppt", taus=(30,), prefix="ppt")
    # Antecedent climatic water deficit: demand (ET0) minus supply (precip).
    gs = gs.assign(_wdef=gs["et0"] - gs["ppt"])
    gs = ff.add_antecedent_weighted(gs, "_wdef", taus=(30,), prefix="wdef")
    return gs.drop(columns=["_wdef"])


def assign_regime(
    sdf: pd.DataFrame, index_col: str, higher_is_drier: bool
) -> pd.Series:
    """Per-site dry/wet label from the site median of a predictor-agnostic index.

    Days at or beyond the site median of the dryness index are ``dry``. Ties at the
    median go to the drier side so a degenerate (constant) index yields all-dry
    rather than crashing (that site simply fails the min-days gate downstream).
    """
    x = sdf[index_col].astype(float)
    med = x.median()
    if higher_is_drier:
        return np.where(x >= med, "dry", "wet")
    return np.where(x <= med, "dry", "wet")


# ---------------------------------------------------------------------------
# Constant-β baseline: flux = c · potential (no soil-water stress shape)
# ---------------------------------------------------------------------------


def const_beta_cv_skill(
    potential: np.ndarray,
    flux: np.ndarray,
    dates,
    n_blocks: int = DS_N_BLOCKS,
    embargo_days: int = fbm.EMBARGO_DAYS,
    min_cv_days: int = DS_MIN_DAYS,
) -> float:
    """Blocked-CV skill of ``flux = c·potential`` (constant β, no soil water).

    ``c`` is the least-squares scale ``Σ(flux·pot)/Σ(pot²)`` fit on each training
    fold. Skill is ``1 − SS_res/SS_tot`` pooled over folds with SS_tot vs the
    *training* mean of the flux — identical fold structure, baseline and
    all-folds-required rule as :func:`flux_beta_models.beta_cv_skill`, so
    ``cv[β(x)] − cv[const-β]`` isolates the stress-shape contribution of ``x``.
    """
    potential = np.asarray(potential, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    if len(flux) < min_cv_days:
        return np.nan
    folds = fs.blocked_fold_indices(dates, n_blocks=n_blocks, embargo_days=embargo_days)
    ss_res = 0.0
    ss_tot = 0.0
    n_used = 0
    for tr, te in folds:
        if len(te) == 0 or len(tr) < 6:
            continue
        pot_tr, flux_tr = potential[tr], flux[tr]
        denom = float(pot_tr @ pot_tr)
        if denom < 1e-12:
            continue
        c = float(pot_tr @ flux_tr) / denom
        flux_hat = c * potential[te]
        resid = flux[te] - flux_hat
        base = flux[te] - float(np.mean(flux_tr))
        ss_res += float(resid @ resid)
        ss_tot += float(base @ base)
        n_used += 1
    if n_used < n_blocks or ss_tot < 1e-12:
        return np.nan
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Per-site, per-regime CV skills (matched rows across predictors + baseline)
# ---------------------------------------------------------------------------


def regime_skills(
    regime_df: pd.DataFrame,
    potential_col: str,
    flux_col: str,
    predictors: dict[str, str],
) -> dict | None:
    """CV skill of const-β and each soil-water β on one site×regime's rows.

    All predictors and the baseline are evaluated on the identical row set (the
    response, potential and every predictor present), so ``cv[a] − cv[b]`` is a
    matched contrast. Returns ``{n_days, cv_const, cv_<label>, dynrange...}`` or
    None if the regime is too short after the common-row intersection.
    """
    need = [potential_col, flux_col, *predictors.values()]
    clean = regime_df.dropna(subset=need)
    if len(clean) < DS_MIN_DAYS:
        return None
    pot = clean[potential_col].values.astype(np.float64)
    flux = clean[flux_col].values.astype(np.float64)
    dates = clean["date"].values
    out = {"n_days": len(clean)}
    out["cv_const"] = const_beta_cv_skill(pot, flux, dates)
    for label, col in predictors.items():
        x = clean[col].values.astype(np.float64)
        if x.std() < 1e-12:
            out[f"cv_{label}"] = np.nan
            continue
        out[f"cv_{label}"] = fbm.beta_cv_skill(
            x,
            pot,
            flux,
            dates,
            fbm.predictor_polarity(col),
            family="logistic",
            n_blocks=DS_N_BLOCKS,
            min_cv_days=DS_MIN_DAYS,
        )
    # Dynamic-range diagnostic (P10–P90) on the matched rows.
    for label, col in predictors.items():
        v = clean[col].values.astype(np.float64)
        out[f"rng_{label}"] = float(np.percentile(v, 90) - np.percentile(v, 10))
    return out


def run_regime_skills(gs: pd.DataFrame) -> pd.DataFrame:
    """Long table of per-site×response×index×regime CV skills + dynamic ranges."""
    rows = []
    for cfg in RESPONSES:
        pot_col, flux_col = cfg["potential"], cfg["flux"]
        if pot_col not in gs.columns or flux_col not in gs.columns:
            continue
        preds = {k: v for k, v in PREDICTORS.items() if v in gs.columns}
        for idx_label, idx_col, hi_dry in DRY_INDICES:
            base = gs.dropna(subset=[idx_col])
            for site, sdf in base.groupby("site_id"):
                reg = assign_regime(sdf, idx_col, hi_dry)
                for regime in REGIMES:
                    rdf = sdf[reg == regime]
                    res = regime_skills(rdf, pot_col, flux_col, preds)
                    if res is None:
                        continue
                    rows.append(
                        {
                            "site_id": site,
                            "response": cfg["label"],
                            "index": idx_label,
                            "regime": regime,
                            **res,
                        }
                    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregation: stress-existence (β over const-β) and ψ-vs-θ redundancy
# ---------------------------------------------------------------------------


def _agg_delta(deltas: np.ndarray) -> dict:
    point, lo, hi = fs.site_bootstrap_ci(deltas)
    st = fs.sign_test(deltas, popmean=0.0, alternative="greater")
    return {
        "n_sites": int(np.sum(np.isfinite(deltas))),
        "delta_median": point,
        "delta_lo": lo,
        "delta_hi": hi,
        "frac_pos": st["frac_pos"],
        "sign_p": st["p"],
    }


def run_contrasts(skills: pd.DataFrame) -> pd.DataFrame:
    """Stress-existence (β over const-β) per response×index×regime×predictor, plus
    the ψ−θ redundancy check, FDR-controlled across the existence family."""
    rows = []
    grp = skills.groupby(["response", "index", "regime"])
    # (a) stress-existence: cv[β(x)] − cv[const-β].
    for (resp, idx, regime), g in grp:
        for pred in EXISTENCE_PREDICTORS:
            col = f"cv_{pred}"
            if col not in g:
                continue
            d = (g[col] - g["cv_const"]).values.astype(np.float64)
            rows.append(
                {
                    "family": "stress_exists",
                    "response": resp,
                    "index": idx,
                    "regime": regime,
                    "contrast": f"{pred}_over_const",
                    **_agg_delta(d),
                }
            )
    # (b) ψ vs θ redundancy (reported, not in the FDR family — expected ≈ 0).
    for (resp, idx, regime), g in grp:
        if "cv_psi" in g and "cv_theta" in g:
            d = (g["cv_psi"] - g["cv_theta"]).values.astype(np.float64)
            rows.append(
                {
                    "family": "psi_vs_theta",
                    "response": resp,
                    "index": idx,
                    "regime": regime,
                    "contrast": "psi_minus_theta",
                    **_agg_delta(d),
                }
            )
    out = pd.DataFrame(rows)
    fam = out["family"] == "stress_exists"
    reject, adj = fs.benjamini_hochberg(out.loc[fam, "sign_p"].values)
    out["bh_reject"] = False
    out["bh_adj_p"] = np.nan
    out.loc[fam, "bh_reject"] = reject
    out.loc[fam, "bh_adj_p"] = adj
    out["verdict"] = np.where(out["bh_reject"], "PASS", "FAIL")
    return out


def run_interaction(skills: pd.DataFrame) -> pd.DataFrame:
    """Paired dry−wet interaction: is a predictor's stress skill dry-concentrated?

    For each site present in *both* regimes, ``Δ_dry − Δ_wet`` where
    ``Δ = cv[β(x)] − cv[const-β]``. A positive median means the soil-water stress
    signal is stronger in the dry tail than the wet regime (the hypothesis)."""
    rows = []
    value_cols = ["cv_const"] + [f"cv_{p}" for p in EXISTENCE_PREDICTORS]
    for (resp, idx), g in skills.groupby(["response", "index"]):
        value_cols_here = [c for c in value_cols if c in g.columns]
        wide = g.pivot_table(index="site_id", columns="regime", values=value_cols_here)
        for pred in EXISTENCE_PREDICTORS:
            col = f"cv_{pred}"
            try:
                d_dry = wide[(col, "dry")] - wide[("cv_const", "dry")]
                d_wet = wide[(col, "wet")] - wide[("cv_const", "wet")]
            except KeyError:
                continue
            inter = (d_dry - d_wet).values.astype(np.float64)
            rows.append(
                {
                    "response": resp,
                    "index": idx,
                    "predictor": pred,
                    **_agg_delta(inter),
                }
            )
    return pd.DataFrame(rows)


def run_dynrange(skills: pd.DataFrame) -> pd.DataFrame:
    """Median per-site dynamic range (P10–P90) of θ vs ψ by regime — evidence on
    whether the L4 surface θ collapses in the dry tail (it does not)."""
    rows = []
    for (resp, idx, regime), g in skills.groupby(["response", "index", "regime"]):
        rows.append(
            {
                "response": resp,
                "index": idx,
                "regime": regime,
                "n_sites": len(g),
                "theta_range_med": float(g["rng_theta"].median()),
                "psi_range_med": float(g["rng_psi"].median()),
            }
        )
    out = pd.DataFrame(rows)
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _index_slice(df: pd.DataFrame, index_label: str) -> pd.DataFrame:
    return df[df["index"] == index_label]


def fig_stress_by_regime(contrasts: pd.DataFrame, index_label: str, path: Path):
    """Bar of Δskill = cv[β(x)] − cv[const-β] by regime for each predictor, ET+GPP."""
    sub = contrasts[
        (contrasts["family"] == "stress_exists") & (contrasts["index"] == index_label)
    ]
    responses = list(dict.fromkeys(sub["response"]))
    fig, axes = plt.subplots(
        1, len(responses), figsize=(5.2 * len(responses), 4.4), squeeze=False
    )
    preds = EXISTENCE_PREDICTORS
    xpos = np.arange(len(preds))
    width = 0.38
    colors = {"dry": "#b2452e", "wet": "#2f6f9f"}
    for ax, resp in zip(axes[0], responses):
        r = sub[sub["response"] == resp]
        for j, regime in enumerate(REGIMES):
            rr = r[r["regime"] == regime].set_index("contrast")
            meds, los, his = [], [], []
            for p in preds:
                key = f"{p}_over_const"
                if key in rr.index:
                    meds.append(rr.loc[key, "delta_median"])
                    los.append(rr.loc[key, "delta_median"] - rr.loc[key, "delta_lo"])
                    his.append(rr.loc[key, "delta_hi"] - rr.loc[key, "delta_median"])
                else:
                    meds.append(np.nan)
                    los.append(np.nan)
                    his.append(np.nan)
            ax.bar(
                xpos + (j - 0.5) * width,
                meds,
                width,
                yerr=[los, his],
                capsize=3,
                color=colors[regime],
                label=regime,
                edgecolor="k",
                linewidth=0.4,
            )
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(xpos)
        ax.set_xticklabels(preds)
        ax.set_title(f"{resp}: soil-water β skill over constant-β")
        ax.set_ylabel("Δ CV R²  (β(x) − const-β)")
        ax.legend(title="regime", frameon=False)
    fig.suptitle(
        f"Dry-tail stress resolution — dryness by {index_label} "
        "(median ± site-bootstrap 95% CI)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_absolute_skill(skills: pd.DataFrame, index_label: str, path: Path):
    """Median absolute CV skill of const-β vs β(θ) by regime (does stress appear)."""
    sub = _index_slice(skills, index_label)
    responses = list(dict.fromkeys(sub["response"]))
    fig, axes = plt.subplots(
        1, len(responses), figsize=(5.2 * len(responses), 4.2), squeeze=False
    )
    for ax, resp in zip(axes[0], responses):
        r = sub[sub["response"] == resp]
        labels = ["cv_const", "cv_theta", "cv_rew"]
        pretty = ["const-β", "β(θ)", "β(REW)"]
        xpos = np.arange(len(labels))
        width = 0.38
        for j, regime in enumerate(REGIMES):
            rr = r[r["regime"] == regime]
            meds = [rr[c].median() if c in rr else np.nan for c in labels]
            ax.bar(
                xpos + (j - 0.5) * width,
                meds,
                width,
                label=regime,
                color={"dry": "#b2452e", "wet": "#2f6f9f"}[regime],
                edgecolor="k",
                linewidth=0.4,
            )
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(xpos)
        ax.set_xticklabels(pretty)
        ax.set_title(f"{resp}: median CV R² by model")
        ax.set_ylabel("CV R² vs training mean")
        ax.legend(title="regime", frameon=False)
    fig.suptitle(
        f"Absolute out-of-sample skill — dryness by {index_label}", fontsize=11
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_dynrange(dynrange: pd.DataFrame, index_label: str, path: Path):
    """θ vs ψ dynamic range (P10–P90) by regime — does θ collapse in the dry tail?"""
    sub = _index_slice(dynrange, index_label)
    responses = list(dict.fromkeys(sub["response"]))
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    resp = responses[0]
    r = sub[sub["response"] == resp].set_index("regime")
    for ax, col, name in zip(
        axes, ["theta_range_med", "psi_range_med"], ["θ (m³/m³)", "ψ (pF, log10|ψ|)"]
    ):
        vals = [r.loc[reg, col] if reg in r.index else np.nan for reg in REGIMES]
        ax.bar(
            REGIMES,
            vals,
            color=["#b2452e", "#2f6f9f"],
            edgecolor="k",
            linewidth=0.4,
        )
        ax.set_title(f"{name} P10–P90 range")
        ax.set_ylabel("dynamic range")
        if vals[0] and vals[1]:
            ax.text(
                0.5,
                0.92,
                f"dry/wet = {vals[0] / vals[1]:.2f}",
                transform=ax.transAxes,
                ha="center",
                fontsize=10,
            )
    fig.suptitle(
        f"Dry-tail dynamic range of L4 θ vs ψ ({resp}, dryness by {index_label})",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------


def interpretation(contrasts: pd.DataFrame, interaction: pd.DataFrame) -> str:
    """Reconcile the two tests. The dry-tail *hypothesis* is the paired
    interaction (Δ_dry − Δ_wet within-site), the more powerful and correct test;
    the per-cell existence family (β over const-β, BH across 24 cells) is the
    blunter absolute-level check reported alongside."""
    exists = contrasts[contrasts["family"] == "stress_exists"]
    dry_pass = exists[(exists["regime"] == "dry") & exists["bh_reject"]]
    # Paired dry-concentration wins: bootstrap CI strictly above 0.
    inter_pos = interaction[
        (interaction["delta_lo"] > 0) & (interaction["delta_median"] > 0)
    ]
    labels = list(inter_pos["response"] + ":" + inter_pos["predictor"])
    if len(inter_pos):
        msg = (
            "Dry-tail hypothesis SUPPORTED (paired interaction): soil-water β adds "
            "more skill in the dry regime than the wet for "
            f"[{', '.join(labels)}] → the growing-season average dilutes a "
            "dry-concentrated stress signal. "
        )
        if len(dry_pass):
            msg += (
                "The absolute dry-regime effect also clears BH-FDR for "
                f"{len(dry_pass)} cell(s)."
            )
        else:
            msg += (
                "The absolute effect is modest — no single dry-regime cell clears "
                "strict BH-FDR — so the signal is real in direction but small."
            )
    else:
        msg = (
            "Dry-tail hypothesis NOT supported: soil-water β does not beat "
            "constant-β more in the dry regime than the wet → the 2nd-order verdict "
            "is not a regime-averaging artifact."
        )
    return msg


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(daily_parquet: str = DAILY_PARQUET, out_dir: Path = OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading inputs …")
    daily = pd.read_parquet(daily_parquet)
    print(f"  daily rows={len(daily):,}  sites={daily['site_id'].nunique()}")

    print("Prepping growing-season frame + antecedent dryness indices …")
    gs = prep_frame(daily)

    print("Per-site × regime CV skills (β(x) vs constant-β) …")
    skills = run_regime_skills(gs)
    skills.to_csv(out_dir / "ds_regime_skills.csv", index=False)

    print("Stress-existence + ψ-vs-θ contrasts …")
    contrasts = run_contrasts(skills)
    contrasts.to_csv(out_dir / "ds_contrasts.csv", index=False)

    print("Dry−wet interaction (dry-concentration) …")
    interaction = run_interaction(skills)
    interaction.to_csv(out_dir / "ds_interaction.csv", index=False)

    print("Dynamic-range diagnostic …")
    dynrange = run_dynrange(skills)
    dynrange.to_csv(out_dir / "ds_dynrange.csv", index=False)

    show = contrasts[contrasts["family"] == "stress_exists"][
        [
            "response",
            "index",
            "regime",
            "contrast",
            "n_sites",
            "delta_median",
            "delta_lo",
            "delta_hi",
            "sign_p",
            "bh_reject",
            "verdict",
        ]
    ]
    print("\nStress-existence: β(x) skill over constant-β (BH-FDR across family):")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(show.to_string(index=False))

    psi = contrasts[contrasts["family"] == "psi_vs_theta"][
        ["response", "index", "regime", "delta_median", "delta_lo", "delta_hi"]
    ]
    print("\nψ − θ redundancy check (expected ≈ 0; ψ_l4 is a PTF transform of θ_l4):")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(psi.to_string(index=False))

    print("\nDry−wet interaction (Δ_dry − Δ_wet; >0 ⇒ stress is dry-concentrated):")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(interaction.to_string(index=False))

    print("\nDynamic range (P10–P90) of L4 θ and ψ by regime:")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(dynrange.to_string(index=False))

    print("\nReading:", interpretation(contrasts, interaction))

    # Figures (primary dryness index = precip).
    figs = []
    for idx_label in ("precip",):
        p1 = out_dir / f"figA_stress_by_regime_{idx_label}.png"
        p2 = out_dir / f"figB_absolute_skill_{idx_label}.png"
        p3 = out_dir / f"figC_dynrange_{idx_label}.png"
        fig_stress_by_regime(contrasts, idx_label, p1)
        fig_absolute_skill(skills, idx_label, p2)
        fig_dynrange(dynrange, idx_label, p3)
        figs += [p1, p2, p3]

    print("\nTables written:")
    for f in ["ds_regime_skills", "ds_contrasts", "ds_interaction", "ds_dynrange"]:
        print(f"  {out_dir / (f + '.csv')}")
    print("\nFigures:")
    for f in figs:
        print(f"  {f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--daily", default=DAILY_PARQUET)
    p.add_argument("--out", default=str(OUT_DIR))
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(daily_parquet=args.daily, out_dir=Path(args.out))
