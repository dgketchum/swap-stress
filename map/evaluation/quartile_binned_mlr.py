"""
Ad-hoc experiment: bin data by VWC quartile, run tiered MLR within each bin.

Tests where psi adds the most complementary information to theta as a function
of wetness regime.

Tiered models per quartile bin:
  ET0 only → ET0 + theta → ET0 + psi → ET0 + theta + psi

For each tier, reports median R² and partial R² across sites, plus the
marginal gain of adding psi beyond theta.

Runs on:
  1. Satellite-scale data (flux_site_daily.parquet, L4 predictors)
  2. In-situ ReESH/AmeriFlux data (8 sites, observed VG psi)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from swapstress.swrc import log10_psi_from_theta

# ---------------------------------------------------------------------------
# Priestley-Taylor ET0
# ---------------------------------------------------------------------------

ALPHA_PT = 1.26
GAMMA = 0.0665  # kPa/C
LAMBDA_V = 2.45  # MJ/kg


def priestley_taylor_et0(sw_in: np.ndarray, t_avg: np.ndarray) -> np.ndarray:
    """Daily reference ET (mm/d) from shortwave radiation and temperature."""
    es = 0.6108 * np.exp(17.27 * t_avg / (t_avg + 237.3))
    delta = 4098.0 * es / (t_avg + 237.3) ** 2
    rn = sw_in * (1.0 - 0.23) * 86400.0 / 1e6  # MJ/m²/d
    et0 = ALPHA_PT * (delta / (delta + GAMMA)) * rn / LAMBDA_V
    return np.maximum(et0, 0.0)


# ---------------------------------------------------------------------------
# OLS helper
# ---------------------------------------------------------------------------


def _ols_r2(X: np.ndarray, y: np.ndarray) -> float:
    n = len(y)
    if n < X.shape[1] + 2:
        return np.nan
    ones = np.ones((n, 1), dtype=np.float64)
    Xa = np.hstack([ones, X])
    try:
        beta, _, rank, _ = np.linalg.lstsq(Xa, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan
    if rank < Xa.shape[1]:
        return np.nan
    y_hat = Xa @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot < 1e-12:
        return np.nan
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Tiered MLR for one site-quartile slice
# ---------------------------------------------------------------------------


def _tiered_mlr(
    sub: pd.DataFrame,
    et0_col: str,
    theta_col: str,
    psi_col: str,
    response_col: str,
    min_n: int = 20,
) -> dict | None:
    """Run tiered MLR on a subset. Returns dict of R² values or None."""
    cols = [et0_col, theta_col, psi_col, response_col]
    clean = sub.dropna(subset=cols)
    if len(clean) < min_n:
        return None

    y = clean[response_col].values.astype(np.float64)
    et0 = clean[et0_col].values.astype(np.float64).reshape(-1, 1)
    theta = clean[theta_col].values.astype(np.float64).reshape(-1, 1)
    psi = clean[psi_col].values.astype(np.float64).reshape(-1, 1)

    # Check for near-constant predictors
    if theta.std() < 1e-12 or psi.std() < 1e-12:
        return None

    r2_et0 = _ols_r2(et0, y)
    r2_theta = _ols_r2(np.hstack([et0, theta]), y)
    r2_psi = _ols_r2(np.hstack([et0, psi]), y)
    r2_both = _ols_r2(np.hstack([et0, theta, psi]), y)

    return {
        "n": len(clean),
        "r2_et0": r2_et0,
        "r2_theta": r2_theta,
        "r2_psi": r2_psi,
        "r2_both": r2_both,
        "partial_theta": r2_theta - r2_et0
        if np.isfinite(r2_theta) and np.isfinite(r2_et0)
        else np.nan,
        "partial_psi": r2_psi - r2_et0
        if np.isfinite(r2_psi) and np.isfinite(r2_et0)
        else np.nan,
        "partial_both": r2_both - r2_et0
        if np.isfinite(r2_both) and np.isfinite(r2_et0)
        else np.nan,
        "psi_beyond_theta": r2_both - r2_theta
        if np.isfinite(r2_both) and np.isfinite(r2_theta)
        else np.nan,
        "theta_beyond_psi": r2_both - r2_psi
        if np.isfinite(r2_both) and np.isfinite(r2_psi)
        else np.nan,
    }


# ---------------------------------------------------------------------------
# Satellite-scale analysis
# ---------------------------------------------------------------------------

Q_MAP = {0: "Q1 (dry)", 1: "Q2", 2: "Q3", 3: "Q4 (wet)"}


def _safe_qcut(x):
    try:
        return pd.qcut(x, 4, labels=False, duplicates="drop")
    except ValueError:
        return pd.Series(np.nan, index=x.index)


def _prep_growing_season(parquet_path: str) -> pd.DataFrame:
    """Load the daily parquet and apply the growing-season + ET0 filter."""
    df = pd.read_parquet(parquet_path)
    gs = df[(df["t_avg"] > 5.0) & (df["sw_in"] > 100.0)].copy()
    gs = gs.dropna(subset=["theta_l4_surf"])
    gs["et0"] = priestley_taylor_et0(gs["sw_in"].values, gs["t_avg"].values)
    gs = gs[gs["et0"] > 0.5]  # drop near-zero ET0 days
    return gs


def run_satellite(parquet_path: str) -> None:
    print("=" * 70)
    print("  SATELLITE-SCALE: Quartile-binned tiered MLR")
    print("=" * 70)

    gs = _prep_growing_season(parquet_path)
    print(f"Growing season: {len(gs)} rows, {gs['site_id'].nunique()} sites")

    q_map = Q_MAP

    # Run for ET and GPP across surface, root-zone, and profile predictors.
    # Quartile bins are computed per-config on that config's theta column, so
    # each predictor pair is stratified by its own wetness regime.
    configs = [
        ("ET", "et_corr", "theta_l4_surf", "suction_l4"),
        ("GPP", "gpp", "theta_l4_surf", "suction_l4"),
        ("ET_root", "et_corr", "theta_l4_root", "suction_l4_root_50"),
        ("GPP_root", "gpp", "theta_l4_root", "suction_l4_root_50"),
        ("ET_prof", "et_corr", "theta_l4_root", "suction_l4_prof"),
        ("GPP_prof", "gpp", "theta_l4_root", "suction_l4_prof"),
    ]

    for resp_name, resp_col, theta_col, psi_col in configs:
        print(f"\n{'─' * 70}")
        print(f"  {resp_name} — {theta_col} / {psi_col}")
        print(f"{'─' * 70}")

        resp_df = gs.dropna(subset=[resp_col, theta_col, psi_col]).copy()
        if resp_df.empty:
            print("  No data")
            continue

        # Per-site quartile bins on this config's theta column
        resp_df["theta_q_num"] = resp_df.groupby("site_id")[theta_col].transform(
            _safe_qcut
        )
        valid_sites = resp_df.groupby("site_id")["theta_q_num"].nunique()
        valid_sites = valid_sites[valid_sites == 4].index
        resp_df = resp_df[resp_df["site_id"].isin(valid_sites)].copy()
        resp_df["theta_quartile"] = resp_df["theta_q_num"].map(q_map)
        if resp_df.empty:
            print("  No sites with 4 quartile bins")
            continue

        quartile_results = []
        for q_label in ["Q1 (dry)", "Q2", "Q3", "Q4 (wet)"]:
            q_df = resp_df[resp_df["theta_quartile"] == q_label]
            site_rows = []
            for site_id, site_df in q_df.groupby("site_id"):
                result = _tiered_mlr(site_df, "et0", theta_col, psi_col, resp_col)
                if result is not None:
                    result["site_id"] = site_id
                    result["quartile"] = q_label
                    site_rows.append(result)

            if not site_rows:
                continue

            site_df_q = pd.DataFrame(site_rows)
            n_sites = len(site_df_q)
            med = site_df_q.median(numeric_only=True)

            # Wilcoxon: does psi add beyond theta?
            psi_gain = site_df_q["psi_beyond_theta"].dropna()
            psi_gain = psi_gain[np.isfinite(psi_gain)]
            if len(psi_gain) >= 5:
                try:
                    _, p_psi = sp_stats.wilcoxon(psi_gain, alternative="greater")
                except ValueError:
                    p_psi = np.nan
                frac_psi_wins = (psi_gain > 0.001).mean()  # >0.1% gain
            else:
                p_psi = np.nan
                frac_psi_wins = np.nan

            quartile_results.append(
                {
                    "quartile": q_label,
                    "n_sites": n_sites,
                    "med_n": med["n"],
                    "r2_et0": med["r2_et0"],
                    "r2_theta": med["r2_theta"],
                    "r2_psi": med["r2_psi"],
                    "r2_both": med["r2_both"],
                    "partial_theta": med["partial_theta"],
                    "partial_psi": med["partial_psi"],
                    "partial_both": med["partial_both"],
                    "psi_beyond_theta": med["psi_beyond_theta"],
                    "theta_beyond_psi": med["theta_beyond_psi"],
                    "frac_psi_adds": frac_psi_wins,
                    "p_psi_adds": p_psi,
                }
            )

        if not quartile_results:
            print("  No results")
            continue

        qdf = pd.DataFrame(quartile_results)
        print(
            f"\n  {'Quartile':<12} {'sites':>5} {'n/site':>6}  "
            f"{'R²(ET0)':>8} {'R²(+θ)':>8} {'R²(+ψ)':>8} {'R²(+θ+ψ)':>9}  "
            f"{'Δψ|θ':>7} {'%ψ adds':>7} {'p':>7}"
        )
        print("  " + "-" * 100)
        for _, r in qdf.iterrows():
            sig = (
                "***"
                if r["p_psi_adds"] < 0.001
                else "**"
                if r["p_psi_adds"] < 0.01
                else "*"
                if r["p_psi_adds"] < 0.05
                else "ns"
            )
            print(
                f"  {r['quartile']:<12} {r['n_sites']:>5.0f} {r['med_n']:>6.0f}  "
                f"{r['r2_et0']:>8.4f} {r['r2_theta']:>8.4f} {r['r2_psi']:>8.4f} {r['r2_both']:>9.4f}  "
                f"{r['psi_beyond_theta']:>+7.4f} {r['frac_psi_adds']:>6.0%} {r['p_psi_adds']:>7.4f} {sig}"
            )

    # Also run with L3 for comparison
    print(f"\n{'─' * 70}")
    print("  ET — L3 predictors (for comparison)")
    print(f"{'─' * 70}")

    # Recompute quartiles using L3 theta (only on L3-available days)
    gs_l3 = gs.dropna(subset=["theta_l3", "suction_l3", "et_corr"])
    gs_l3["theta_q_num"] = gs_l3.groupby("site_id")["theta_l3"].transform(_safe_qcut)
    valid_l3 = gs_l3.groupby("site_id")["theta_q_num"].nunique()
    valid_l3 = valid_l3[valid_l3 == 4].index
    gs_l3 = gs_l3[gs_l3["site_id"].isin(valid_l3)].copy()
    gs_l3["theta_quartile_l3"] = gs_l3["theta_q_num"].map(q_map)

    quartile_results_l3 = []
    for q_label in ["Q1 (dry)", "Q2", "Q3", "Q4 (wet)"]:
        q_df = gs_l3[gs_l3["theta_quartile_l3"] == q_label]
        site_rows = []
        for site_id, site_df in q_df.groupby("site_id"):
            result = _tiered_mlr(site_df, "et0", "theta_l3", "suction_l3", "et_corr")
            if result is not None:
                result["site_id"] = site_id
                result["quartile"] = q_label
                site_rows.append(result)

        if not site_rows:
            continue

        site_df_q = pd.DataFrame(site_rows)
        n_sites = len(site_df_q)
        med = site_df_q.median(numeric_only=True)

        psi_gain = site_df_q["psi_beyond_theta"].dropna()
        psi_gain = psi_gain[np.isfinite(psi_gain)]
        if len(psi_gain) >= 5:
            try:
                _, p_psi = sp_stats.wilcoxon(psi_gain, alternative="greater")
            except ValueError:
                p_psi = np.nan
            frac_psi_wins = (psi_gain > 0.001).mean()
        else:
            p_psi = np.nan
            frac_psi_wins = np.nan

        quartile_results_l3.append(
            {
                "quartile": q_label,
                "n_sites": n_sites,
                "med_n": med["n"],
                "r2_et0": med["r2_et0"],
                "r2_theta": med["r2_theta"],
                "r2_psi": med["r2_psi"],
                "r2_both": med["r2_both"],
                "partial_theta": med["partial_theta"],
                "partial_psi": med["partial_psi"],
                "partial_both": med["partial_both"],
                "psi_beyond_theta": med["psi_beyond_theta"],
                "theta_beyond_psi": med["theta_beyond_psi"],
                "frac_psi_adds": frac_psi_wins,
                "p_psi_adds": p_psi,
            }
        )

    if quartile_results_l3:
        qdf_l3 = pd.DataFrame(quartile_results_l3)
        print(
            f"\n  {'Quartile':<12} {'sites':>5} {'n/site':>6}  "
            f"{'R²(ET0)':>8} {'R²(+θ)':>8} {'R²(+ψ)':>8} {'R²(+θ+ψ)':>9}  "
            f"{'Δψ|θ':>7} {'%ψ adds':>7} {'p':>7}"
        )
        print("  " + "-" * 100)
        for _, r in qdf_l3.iterrows():
            sig = (
                "***"
                if r["p_psi_adds"] < 0.001
                else "**"
                if r["p_psi_adds"] < 0.01
                else "*"
                if r["p_psi_adds"] < 0.05
                else "ns"
            )
            print(
                f"  {r['quartile']:<12} {r['n_sites']:>5.0f} {r['med_n']:>6.0f}  "
                f"{r['r2_et0']:>8.4f} {r['r2_theta']:>8.4f} {r['r2_psi']:>8.4f} {r['r2_both']:>9.4f}  "
                f"{r['psi_beyond_theta']:>+7.4f} {r['frac_psi_adds']:>6.0%} {r['p_psi_adds']:>7.4f} {sig}"
            )


# ---------------------------------------------------------------------------
# Surface vs root-zone head-to-head (the depth hypothesis)
# ---------------------------------------------------------------------------

# Combined-model predictor pairs (theta_col, psi_col), keyed by name.
DEPTH_PAIRS = {
    "surface": ("theta_l4_surf", "suction_l4"),
    "root50": ("theta_l4_root", "suction_l4_root_50"),
    "profile": ("theta_l4_root", "suction_l4_prof"),
}


def _both_model_r2(
    sub: pd.DataFrame, theta_col: str, psi_col: str, response_col: str, min_n: int = 20
) -> float:
    """R² of the full ET0 + theta + psi model for one site."""
    clean = sub.dropna(subset=["et0", theta_col, psi_col, response_col])
    if len(clean) < min_n:
        return np.nan
    y = clean[response_col].values.astype(np.float64)
    et0 = clean["et0"].values.astype(np.float64).reshape(-1, 1)
    theta = clean[theta_col].values.astype(np.float64).reshape(-1, 1)
    psi = clean[psi_col].values.astype(np.float64).reshape(-1, 1)
    if theta.std() < 1e-12 or psi.std() < 1e-12:
        return np.nan
    return _ols_r2(np.hstack([et0, theta, psi]), y)


def run_surface_vs_root(parquet_path: str, output_dir: str) -> None:
    """Does a root-zone (or profile) predictor beat the surface predictor?

    Per site, compares the full ET0 + theta + psi model R² for the surface
    pair vs the root-zone and profile pairs, for ET and GPP, over all
    growing-season days and the dry half. Writes per-site R² to
    surface_vs_root.csv and prints win-rate / median ΔR² / Wilcoxon p.
    """
    print(f"\n\n{'=' * 70}")
    print("  SURFACE vs ROOT-ZONE: does depth help?")
    print(f"{'=' * 70}")

    gs = _prep_growing_season(parquet_path)

    rows = []
    for resp_name, resp_col in [("ET", "et_corr"), ("GPP", "gpp")]:
        base_all = gs.dropna(subset=[resp_col]).copy()
        if base_all.empty:
            continue
        # Dry half by per-site surface-theta median
        med = base_all.groupby("site_id")["theta_l4_surf"].transform("median")
        dry = base_all[base_all["theta_l4_surf"] <= med]

        for filt_name, base in [("all", base_all), ("dry_half", dry)]:
            for site_id, sdf in base.groupby("site_id"):
                rec = {
                    "site_id": site_id,
                    "response": resp_name,
                    "filter": filt_name,
                }
                for pname, (tcol, pcol) in DEPTH_PAIRS.items():
                    rec[f"r2_{pname}"] = _both_model_r2(sdf, tcol, pcol, resp_col)
                rows.append(rec)

    cmp_df = pd.DataFrame(rows)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "surface_vs_root.csv"
    cmp_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nSaved {csv_path}")

    # Summarize root vs surface and profile vs surface
    print(
        f"\n  {'response':<8} {'filter':<9} {'contrast':<18} {'n':>4} "
        f"{'win%':>6} {'ΔR² med':>9} {'p':>8}"
    )
    print("  " + "-" * 70)
    for (resp_name, filt_name), g in cmp_df.groupby(["response", "filter"]):
        for contrast in ("root50", "profile"):
            paired = g.dropna(subset=[f"r2_{contrast}", "r2_surface"])
            diff = (paired[f"r2_{contrast}"] - paired["r2_surface"]).values
            diff = diff[np.isfinite(diff)]
            if len(diff) < 5:
                continue
            try:
                _, pval = sp_stats.wilcoxon(diff, alternative="greater")
            except ValueError:
                pval = np.nan
            sig = (
                "***"
                if pval < 0.001
                else "**"
                if pval < 0.01
                else "*"
                if pval < 0.05
                else "ns"
            )
            print(
                f"  {resp_name:<8} {filt_name:<9} {contrast + ' vs surf':<18} "
                f"{len(diff):>4} {(diff > 0).mean():>6.0%} "
                f"{np.median(diff):>+9.4f} {pval:>8.4f} {sig}"
            )


# ---------------------------------------------------------------------------
# Aggregate tiered summary (all growing-season days, no quartile binning)
# ---------------------------------------------------------------------------


def _sig(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def _wilcoxon_greater(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan
    try:
        _, p = sp_stats.wilcoxon(x, alternative="greater")
    except ValueError:
        return np.nan
    return p


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


def run_tiered_summary(parquet_path: str, output_dir: str) -> None:
    """Aggregate (all growing-season days, no quartile binning) tiered MLR.

    For ET and GPP, reports the median per-site R² of the four nested model
    specs — meteorology only (ET0), met + theta, met + psi, met + theta + psi
    — across the surface, root-zone, profile, and L3 predictor sets. This is
    the headline "flux as a function of meteorology, VWC, and suction" table.
    Writes tiered_summary.csv.
    """
    print(f"\n\n{'=' * 70}")
    print("  AGGREGATE TIERED MLR (all growing-season days)")
    print("  meteorology (ET0) -> +theta -> +psi -> +theta+psi")
    print(f"{'=' * 70}")

    gs = _prep_growing_season(parquet_path)
    print(f"Growing season: {len(gs)} rows, {gs['site_id'].nunique()} sites")

    summary_rows = []
    for resp_name, resp_col, theta_col, psi_col, pred_label in TIERED_CONFIGS:
        if resp_col not in gs.columns:
            continue
        sub = gs.dropna(subset=[resp_col, theta_col, psi_col])
        site_rows = []
        for _, sdf in sub.groupby("site_id"):
            result = _tiered_mlr(sdf, "et0", theta_col, psi_col, resp_col)
            if result is not None:
                site_rows.append(result)
        if not site_rows:
            continue
        site_df = pd.DataFrame(site_rows)
        med = site_df.median(numeric_only=True)
        summary_rows.append(
            {
                "response": resp_name,
                "predictors": pred_label,
                "n_sites": len(site_df),
                "med_n": med["n"],
                "r2_met": med["r2_et0"],
                "r2_met_theta": med["r2_theta"],
                "r2_met_psi": med["r2_psi"],
                "r2_met_theta_psi": med["r2_both"],
                "partial_theta": med["partial_theta"],
                "partial_psi": med["partial_psi"],
                "psi_beyond_theta": med["psi_beyond_theta"],
                "p_psi_beyond_theta": _wilcoxon_greater(
                    site_df["psi_beyond_theta"].values
                ),
                "theta_beyond_psi": med["theta_beyond_psi"],
                "p_theta_beyond_psi": _wilcoxon_greater(
                    site_df["theta_beyond_psi"].values
                ),
            }
        )

    if not summary_rows:
        print("  No results")
        return

    sdf = pd.DataFrame(summary_rows)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "tiered_summary.csv"
    sdf.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nSaved {csv_path}")

    print(
        f"\n  {'resp':<5} {'predictors':<11} {'sites':>5} {'n/site':>6}  "
        f"{'R²(met)':>8} {'+θ':>8} {'+ψ':>8} {'+θ+ψ':>8}  "
        f"{'ψ|θ':>8} {'pψ|θ':>7} {'θ|ψ':>8} {'pθ|ψ':>7}"
    )
    print("  " + "-" * 104)
    for _, r in sdf.iterrows():
        print(
            f"  {r['response']:<5} {r['predictors']:<11} "
            f"{r['n_sites']:>5.0f} {r['med_n']:>6.0f}  "
            f"{r['r2_met']:>8.4f} {r['r2_met_theta']:>8.4f} "
            f"{r['r2_met_psi']:>8.4f} {r['r2_met_theta_psi']:>8.4f}  "
            f"{r['psi_beyond_theta']:>+8.4f} {_sig(r['p_psi_beyond_theta']):>7} "
            f"{r['theta_beyond_psi']:>+8.4f} {_sig(r['p_theta_beyond_psi']):>7}"
        )


# ---------------------------------------------------------------------------
# In-situ ReESH analysis
# ---------------------------------------------------------------------------

VG_DIR = Path("/nas/soils/soil_potential_obs/curve_fits/reesh/bayes")
AMF_DIR = Path("/nas/climate/ameriflux/amf_new")


def _load_vg_params() -> dict[str, dict]:
    """Load Bayesian VG params, keyed by normalized site ID.

    Filenames are like US-CPK_1NB.json — extract site prefix, aggregate
    across replicates by averaging parameters.
    """
    import re

    # Collect per-site replicate params
    site_params: dict[str, list[dict]] = {}
    for jf in sorted(VG_DIR.glob("*.json")):
        # Extract site prefix (e.g., "US-CPK" from "US-CPK_1NB")
        m = re.match(r"^([A-Z]{2}-[A-Za-z0-9]+)", jf.stem)
        if m is None:
            continue
        site_raw = m.group(1)

        with open(jf) as f:
            data = json.load(f)

        # Prefer 10cm depth, fall back to 0cm
        depth_key = None
        for dk in ["10", "0", "50"]:
            if dk in data and data[dk].get("status") in ("converged", "Success"):
                depth_key = dk
                break
        if depth_key is None:
            continue

        p = data[depth_key]["parameters"]

        # Parameters may be dicts with "value" key or plain floats
        def _val(x):
            return x["value"] if isinstance(x, dict) else x

        norm_id = site_raw.upper().replace("-", "")
        if norm_id not in site_params:
            site_params[norm_id] = []
        site_params[norm_id].append(
            {
                "theta_r": _val(p["theta_r"]),
                "theta_s": _val(p["theta_s"]),
                "alpha": _val(p["alpha"]),
                "n_vg": _val(p["n"]),
            }
        )

    # Average across replicates
    params = {}
    for norm_id, reps in site_params.items():
        avg = {}
        for key in ["theta_r", "theta_s", "alpha", "n_vg"]:
            avg[key] = np.mean([r[key] for r in reps])
        params[norm_id] = avg

    return params


def _vg_invert(
    theta: np.ndarray, theta_r: float, theta_s: float, alpha: float, n_vg: float
) -> np.ndarray:
    """Van Genuchten inversion: theta -> psi (log10 cm).

    The guard configuration here is the one the flux analyses were published
    with -- a looser Se clip than the PTF baseline's, plus a 0.01 cm floor -- so
    it is named once rather than repeated at each call site. Reproduces the
    pre-consolidation implementation bit for bit.
    """
    return log10_psi_from_theta(
        theta, theta_r, theta_s, alpha, n_vg, se_eps=1e-3, psi_floor_cm=0.01
    )


def _load_ameriflux_site(site_id: str) -> pd.DataFrame | None:
    """Load daily AmeriFlux BASE data for one site."""
    # Find the directory
    norm = site_id.replace("-", "").upper()
    candidates = list(AMF_DIR.glob(f"AMF_{site_id}_BASE-BADM_*"))
    if not candidates:
        # Try case variations
        candidates = list(AMF_DIR.glob("AMF_*_BASE-BADM_*"))
        candidates = [
            c
            for c in candidates
            if c.name.split("_")[1].upper().replace("-", "") == norm
        ]
    if not candidates:
        return None

    site_dir = candidates[0]
    csvs = sorted(site_dir.glob("AMF_*_BASE_HH_*.csv")) + sorted(
        site_dir.glob("AMF_*_BASE_HR_*.csv")
    )
    if not csvs:
        return None

    try:
        hh = pd.read_csv(csvs[0], na_values=["-9999", "-9999.0"], comment="#")
    except Exception:
        return None
    if "TIMESTAMP_START" not in hh.columns:
        return None

    hh["date"] = pd.to_datetime(
        hh["TIMESTAMP_START"].astype(str).str[:8], format="%Y%m%d"
    )

    # SWC columns (average all depths)
    swc_cols = [c for c in hh.columns if c.startswith("SWC_")]
    if not swc_cols:
        return None

    hh["theta_insitu"] = hh[swc_cols].mean(axis=1) / 100.0  # % -> fraction

    # Met
    sw_candidates = ["SW_IN_F_MDS", "SW_IN_F", "SW_IN"]
    ta_candidates = ["TA_F_MDS", "TA_F", "TA", "T_SONIC"]
    sw_col = next((c for c in sw_candidates if c in hh.columns), None)
    ta_col = next((c for c in ta_candidates if c in hh.columns), None)

    # ET from LE
    le_candidates = ["LE_F_MDS", "LE_CORR", "LE"]
    le_col = next((c for c in le_candidates if c in hh.columns), None)

    # GPP
    gpp_candidates = ["GPP_NT_VUT_REF", "GPP_DT_VUT_REF", "GPP_PI"]
    gpp_col = next((c for c in gpp_candidates if c in hh.columns), None)

    agg = {"theta_insitu": "mean"}
    if sw_col:
        agg[sw_col] = "mean"
    if ta_col:
        agg[ta_col] = "mean"
    if le_col:
        agg[le_col] = "mean"
    if gpp_col:
        agg[gpp_col] = "mean"

    daily = hh.groupby("date").agg(agg).reset_index()

    # Rename
    rename = {"theta_insitu": "theta"}
    if sw_col:
        rename[sw_col] = "sw_in"
    if ta_col:
        rename[ta_col] = "t_avg"
    if le_col:
        rename[le_col] = "le"
    if gpp_col:
        rename[gpp_col] = "gpp"
    daily = daily.rename(columns=rename)

    # LE -> ET (mm/d)
    if "le" in daily.columns:
        daily["et"] = daily["le"] * 86400.0 / 2.45e6  # W/m² -> mm/d
        daily = daily[daily["et"] > 0]

    return daily


def run_insitu() -> None:
    print(f"\n\n{'=' * 70}")
    print("  IN-SITU (ReESH): Quartile-binned tiered MLR")
    print(f"{'=' * 70}")

    vg_params = _load_vg_params()
    print(f"\nLoaded VG params for {len(vg_params)} sites")

    # All ReESH sites that might have AmeriFlux data
    reesh_sites = [
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

    all_site_data = []
    for site_id in reesh_sites:
        norm = site_id.upper().replace("-", "")
        if norm not in vg_params:
            continue
        daily = _load_ameriflux_site(site_id)
        if daily is None or "et" not in daily.columns:
            continue
        if "sw_in" not in daily.columns or "t_avg" not in daily.columns:
            continue

        vg = vg_params[norm]
        daily = daily.dropna(subset=["theta", "sw_in", "t_avg"])
        if len(daily) < 50:
            continue

        daily["psi_obs"] = _vg_invert(
            daily["theta"].values, vg["theta_r"], vg["theta_s"], vg["alpha"], vg["n_vg"]
        )
        daily["et0"] = priestley_taylor_et0(
            daily["sw_in"].values, daily["t_avg"].values
        )

        # Growing season filter
        daily = daily[(daily["t_avg"] > 5.0) & (daily["sw_in"] > 100.0)]
        daily = daily[daily["et0"] > 0.5]

        daily["site_id"] = site_id
        all_site_data.append(daily)

    if not all_site_data:
        print("No in-situ sites with complete data")
        return

    combined = pd.concat(all_site_data, ignore_index=True)
    print(f"Combined: {len(combined)} rows, {combined['site_id'].nunique()} sites")

    # Assign quartile bins per site
    def _safe_qcut(x):
        try:
            return pd.qcut(x, 4, labels=False, duplicates="drop")
        except ValueError:
            return pd.Series(np.nan, index=x.index)

    combined["theta_q_num"] = combined.groupby("site_id")["theta"].transform(_safe_qcut)
    valid_sites = combined.groupby("site_id")["theta_q_num"].nunique()
    valid_sites = valid_sites[valid_sites == 4].index
    combined = combined[combined["site_id"].isin(valid_sites)].copy()
    q_map = {0: "Q1 (dry)", 1: "Q2", 2: "Q3", 3: "Q4 (wet)"}
    combined["theta_quartile"] = combined["theta_q_num"].map(q_map)

    for resp_name, resp_col in [("ET", "et"), ("GPP", "gpp")]:
        if resp_col not in combined.columns:
            print(f"\n  {resp_name}: column not available")
            continue
        resp_df = combined.dropna(subset=[resp_col])
        if resp_df.empty:
            continue

        print(f"\n{'─' * 70}")
        print(f"  {resp_name} — in-situ theta + observed VG psi")
        print(f"{'─' * 70}")

        quartile_results = []
        for q_label in ["Q1 (dry)", "Q2", "Q3", "Q4 (wet)"]:
            q_df = resp_df[resp_df["theta_quartile"] == q_label]
            site_rows = []
            for site_id, site_df in q_df.groupby("site_id"):
                result = _tiered_mlr(
                    site_df, "et0", "theta", "psi_obs", resp_col, min_n=15
                )
                if result is not None:
                    result["site_id"] = site_id
                    result["quartile"] = q_label
                    site_rows.append(result)

            if not site_rows:
                continue

            site_df_q = pd.DataFrame(site_rows)
            n_sites = len(site_df_q)
            med = site_df_q.median(numeric_only=True)

            psi_gain = site_df_q["psi_beyond_theta"].dropna()
            psi_gain = psi_gain[np.isfinite(psi_gain)]
            if len(psi_gain) >= 5:
                try:
                    _, p_psi = sp_stats.wilcoxon(psi_gain, alternative="greater")
                except ValueError:
                    p_psi = np.nan
                frac_psi_wins = (psi_gain > 0.001).mean()
            else:
                p_psi = np.nan
                frac_psi_wins = np.nan

            quartile_results.append(
                {
                    "quartile": q_label,
                    "n_sites": n_sites,
                    "med_n": med["n"],
                    "r2_et0": med["r2_et0"],
                    "r2_theta": med["r2_theta"],
                    "r2_psi": med["r2_psi"],
                    "r2_both": med["r2_both"],
                    "partial_theta": med["partial_theta"],
                    "partial_psi": med["partial_psi"],
                    "partial_both": med["partial_both"],
                    "psi_beyond_theta": med["psi_beyond_theta"],
                    "theta_beyond_psi": med["theta_beyond_psi"],
                    "frac_psi_adds": frac_psi_wins,
                    "p_psi_adds": p_psi,
                }
            )

        if not quartile_results:
            print(f"  No results for {resp_name}")
            continue

        qdf = pd.DataFrame(quartile_results)
        print(
            f"\n  {'Quartile':<12} {'sites':>5} {'n/site':>6}  "
            f"{'R²(ET0)':>8} {'R²(+θ)':>8} {'R²(+ψ)':>8} {'R²(+θ+ψ)':>9}  "
            f"{'Δψ|θ':>7} {'%ψ adds':>7} {'p':>7}"
        )
        print("  " + "-" * 100)
        for _, r in qdf.iterrows():
            p_val = r["p_psi_adds"]
            if np.isnan(p_val):
                sig = "n/a"
                p_str = "   n/a"
            else:
                sig = (
                    "***"
                    if p_val < 0.001
                    else "**"
                    if p_val < 0.01
                    else "*"
                    if p_val < 0.05
                    else "ns"
                )
                p_str = f"{p_val:>7.4f}"
            frac_str = (
                f"{r['frac_psi_adds']:>6.0%}"
                if not np.isnan(r["frac_psi_adds"])
                else "   n/a"
            )
            print(
                f"  {r['quartile']:<12} {r['n_sites']:>5.0f} {r['med_n']:>6.0f}  "
                f"{r['r2_et0']:>8.4f} {r['r2_theta']:>8.4f} {r['r2_psi']:>8.4f} {r['r2_both']:>9.4f}  "
                f"{r['psi_beyond_theta']:>+7.4f} {frac_str} {p_str} {sig}"
            )

        # Per-site detail for the quartile with largest psi gain
        print("\n  Per-site detail by quartile (Δψ|θ = R²(θ+ψ) - R²(θ)):")
        for q_label in ["Q1 (dry)", "Q2", "Q3", "Q4 (wet)"]:
            q_df = resp_df[resp_df["theta_quartile"] == q_label]
            print(f"\n  {q_label}:")
            print(
                f"  {'site':<10} {'n':>5} {'R²(ET0)':>8} {'R²(+θ)':>8} {'R²(+ψ)':>8} {'R²(+θ+ψ)':>9} {'Δψ|θ':>7}"
            )
            print(f"  {'-' * 60}")
            for site_id, site_df in q_df.groupby("site_id"):
                result = _tiered_mlr(
                    site_df, "et0", "theta", "psi_obs", resp_col, min_n=15
                )
                if result is None:
                    continue
                print(
                    f"  {site_id:<10} {result['n']:>5} {result['r2_et0']:>8.4f} "
                    f"{result['r2_theta']:>8.4f} {result['r2_psi']:>8.4f} {result['r2_both']:>9.4f} "
                    f"{result['psi_beyond_theta']:>+7.4f}"
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parquet = "/nas/soils/swapstress/evaluation/flux_validation/flux_site_daily.parquet"
    out_dir = "/nas/soils/swapstress/evaluation/flux_validation/regression"
    run_tiered_summary(parquet, out_dir)
    run_satellite(parquet)
    run_surface_vs_root(parquet, out_dir)
    run_insitu()
