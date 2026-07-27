"""
Regression analysis: soil water predictors vs eddy-covariance ET and GPP.

For each flux tower site during water-limited growing-season days, fits
OLS regressions of the form:

    ET  ~ sw_in + t_avg + [vpd] + predictor
    GPP ~ sw_in + t_avg + [vpd] + predictor

for the soil water predictors (raw theta and predicted suction from three
sensors × two methods, plus L4 root-zone and root-weighted profile suction).
Reports per-site R², ΔR² relative to the θ_L3 baseline, and partial
correlation of the predictor with the flux residual.

Usage:
    python -m map.evaluation.flux_regression \
        --input /nas/soils/swapstress/evaluation/flux_validation/flux_site_daily.parquet \
        --output-dir /nas/soils/swapstress/evaluation/flux_validation/regression
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_INPUT = (
    "/nas/soils/swapstress/evaluation/flux_validation/flux_site_daily.parquet"
)
DEFAULT_OUTPUT_DIR = "/nas/soils/swapstress/evaluation/flux_validation/regression"

# ---------------------------------------------------------------------------
# Predictor definitions
# ---------------------------------------------------------------------------
# (label, column, description)
PREDICTORS = [
    ("theta_l3", "theta_l3", "SMAP L3 surface SM"),
    ("theta_smos", "theta_smos", "SMOS-IC surface SM"),
    ("psi_rf_l3", "suction_l3", "RF prediction, SMAP L3"),
    ("psi_rf_l4", "suction_l4", "RF prediction, L4 surface"),
    ("psi_rf_smos", "suction_smos", "RF prediction, SMOS"),
    ("psi_ptf_l3", "suction_ptf_l3", "Rosetta PTF, SMAP L3"),
    ("psi_ptf_l4", "suction_ptf_l4", "Rosetta PTF, L4 surface"),
    ("psi_ptf_smos", "suction_ptf_smos", "Rosetta PTF, SMOS"),
    # L4 root-zone (0–100 cm integrated theta) at three proxy depths + profile
    ("theta_l4_root", "theta_l4_root", "SMAP L4 root-zone SM"),
    ("psi_rf_l4_root_30", "suction_l4_root_30", "RF, L4 root @30cm"),
    ("psi_rf_l4_root_50", "suction_l4_root_50", "RF, L4 root @50cm"),
    ("psi_rf_l4_root_100", "suction_l4_root_100", "RF, L4 root @100cm"),
    ("psi_rf_l4_prof", "suction_l4_prof", "RF, root-weighted profile"),
]

# Key comparisons (label_a, label_b, description)
COMPARISONS = [
    ("psi_rf_l3", "theta_l3", "A: psi > theta (SMAP)"),
    ("psi_rf_smos", "theta_smos", "A: psi > theta (SMOS)"),
    ("psi_rf_l3", "psi_ptf_l3", "B: learned > PTF"),
    ("psi_rf_smos", "psi_rf_l3", "C: sensor generalization"),
    ("psi_rf_l4", "psi_rf_l3", "D: L4 gap-fill vs L3"),
    ("psi_rf_l4_root_50", "psi_rf_l4", "E: root-zone > surface (psi)"),
    ("psi_rf_l4_prof", "psi_rf_l4", "F: profile > surface (psi)"),
]

# Met covariates
MET_3VAR = ["sw_in", "t_avg"]
MET_4VAR = ["sw_in", "t_avg", "vpd"]

# Water-limited filter thresholds
T_AVG_MIN = 5.0  # °C — growing season
SW_IN_MIN = 100.0  # W/m² — growing season
MIN_SITE_DAYS = 30  # minimum site-days after filtering


# ---------------------------------------------------------------------------
# Water-limited filter
# ---------------------------------------------------------------------------


def apply_water_limited_filter(
    df: pd.DataFrame,
    theta_col: str = "theta_l3",
    t_avg_min: float = T_AVG_MIN,
    sw_in_min: float = SW_IN_MIN,
) -> pd.DataFrame:
    """Filter to water-limited growing-season days.

    Growing season: t_avg > t_avg_min AND sw_in > sw_in_min.
    Dry half: theta below site-specific median (computed on growing-season days).
    """
    gs = df[(df["t_avg"] > t_avg_min) & (df["sw_in"] > sw_in_min)].copy()
    if gs.empty:
        return gs

    # Use theta_l4_surf as fallback theta for the dry-half filter since L3
    # has only 33% coverage.  L4 is gap-free and physically comparable at the
    # surface.
    theta_for_filter = "theta_l4_surf"
    gs = gs.dropna(subset=[theta_for_filter])
    if gs.empty:
        return gs

    medians = gs.groupby("site_id")[theta_for_filter].transform("median")
    dry = gs[gs[theta_for_filter] <= medians].copy()
    return dry


# ---------------------------------------------------------------------------
# OLS helpers
# ---------------------------------------------------------------------------


def _ols_r2(X: np.ndarray, y: np.ndarray) -> float:
    """R² from ordinary least squares (with intercept).

    X: (n, p), y: (n,).  Returns R² or NaN if degenerate.
    """
    n = len(y)
    if n < X.shape[1] + 2:
        return np.nan
    # Add intercept
    ones = np.ones((n, 1), dtype=np.float64)
    Xa = np.hstack([ones, X])
    try:
        beta, residuals, rank, _ = np.linalg.lstsq(Xa, y, rcond=None)
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


def _partial_corr(X_met: np.ndarray, predictor: np.ndarray, y: np.ndarray) -> float:
    """Partial correlation between predictor and y, controlling for X_met.

    Regress both predictor and y on X_met, then correlate the residuals.
    """
    n = len(y)
    if n < X_met.shape[1] + 3:
        return np.nan
    ones = np.ones((n, 1), dtype=np.float64)
    Xa = np.hstack([ones, X_met])
    try:
        beta_y, _, _, _ = np.linalg.lstsq(Xa, y, rcond=None)
        beta_p, _, _, _ = np.linalg.lstsq(Xa, predictor, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan
    res_y = y - Xa @ beta_y
    res_p = predictor - Xa @ beta_p
    if res_p.std() < 1e-12 or res_y.std() < 1e-12:
        return np.nan
    r, _ = sp_stats.pearsonr(res_p, res_y)
    return r


# ---------------------------------------------------------------------------
# Per-site regression
# ---------------------------------------------------------------------------


def _fit_site(
    site_df: pd.DataFrame,
    response_col: str,
    predictor_col: str,
    met_cols: list[str],
) -> dict | None:
    """Fit one predictor at one site. Returns metrics dict or None."""
    cols_needed = met_cols + [predictor_col, response_col]
    sub = site_df.dropna(subset=cols_needed)
    if len(sub) < MIN_SITE_DAYS:
        return None

    y = sub[response_col].values.astype(np.float64)
    X_met = sub[met_cols].values.astype(np.float64)
    x_pred = sub[predictor_col].values.astype(np.float64)

    # Full model: met + predictor
    X_full = np.column_stack([X_met, x_pred])
    r2_full = _ols_r2(X_full, y)

    # Met-only model (for partial R² denominator)
    r2_met = _ols_r2(X_met, y)

    # Partial correlation
    pcorr = _partial_corr(X_met, x_pred, y)

    return {
        "n_days": len(sub),
        "r2": r2_full,
        "r2_met": r2_met,
        "partial_r2": r2_full - r2_met
        if np.isfinite(r2_full) and np.isfinite(r2_met)
        else np.nan,
        "partial_corr": pcorr,
    }


# ---------------------------------------------------------------------------
# Main analysis loop
# ---------------------------------------------------------------------------


def run_regression(
    daily: pd.DataFrame,
    output_dir: Path,
    min_site_days: int = MIN_SITE_DAYS,
) -> pd.DataFrame:
    """Run all regressions and write results.

    Returns the site-level metrics DataFrame.
    """
    global MIN_SITE_DAYS
    MIN_SITE_DAYS = min_site_days

    output_dir.mkdir(parents=True, exist_ok=True)

    # Apply water-limited filter
    print("Applying water-limited filter...")
    wl = apply_water_limited_filter(daily)
    print(f"  {len(wl)} rows, {wl['site_id'].nunique()} sites after filter")

    responses = [
        ("et", "et_corr"),
        ("gpp", "gpp"),
    ]
    model_specs = [
        ("3var", MET_3VAR),
        ("4var", MET_4VAR),
    ]

    all_rows = []

    for resp_label, resp_col in responses:
        resp_df = wl.dropna(subset=[resp_col])
        n_resp_sites = resp_df["site_id"].nunique()
        print(f"\nResponse: {resp_label} ({len(resp_df)} rows, {n_resp_sites} sites)")

        for spec_label, met_cols in model_specs:
            print(f"  Model spec: {spec_label} ({', '.join(met_cols)})")

            for pred_label, pred_col, pred_desc in PREDICTORS:
                n_fitted = 0
                for site_id, site_df in resp_df.groupby("site_id"):
                    result = _fit_site(site_df, resp_col, pred_col, met_cols)
                    if result is None:
                        continue
                    n_fitted += 1
                    row = {
                        "site_id": site_id,
                        "response": resp_label,
                        "model_spec": spec_label,
                        "predictor": pred_label,
                        "predictor_col": pred_col,
                        **result,
                    }
                    # Add site metadata
                    site_meta = site_df.iloc[0]
                    row["lat"] = site_meta["lat"]
                    row["lon"] = site_meta["lon"]
                    row["network"] = site_meta["network"]
                    all_rows.append(row)

                if n_fitted > 0:
                    print(f"    {pred_label}: {n_fitted} sites")

    metrics = pd.DataFrame(all_rows)
    if metrics.empty:
        print("No results — check data coverage.")
        return metrics

    # Compute ΔR² relative to theta_l3 baseline
    metrics = _add_delta_r2(metrics, baseline="theta_l3")

    # Write site-level metrics
    metrics_path = output_dir / "site_metrics.parquet"
    metrics.to_parquet(metrics_path, index=False)
    print(f"\nSaved {metrics_path}")

    # Write CSV for easy inspection
    csv_path = output_dir / "site_metrics.csv"
    metrics.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Saved {csv_path}")

    # Aggregate summary
    _write_summary(metrics, output_dir)

    # Comparison table
    _write_comparisons(metrics, output_dir)

    return metrics


def _add_delta_r2(metrics: pd.DataFrame, baseline: str = "theta_l3") -> pd.DataFrame:
    """Add ΔR² column: R²(predictor) - R²(baseline) for each (site, response, spec)."""
    base = metrics[metrics["predictor"] == baseline][
        ["site_id", "response", "model_spec", "r2"]
    ].rename(columns={"r2": "r2_baseline"})

    merged = metrics.merge(base, on=["site_id", "response", "model_spec"], how="left")
    merged["delta_r2"] = merged["r2"] - merged["r2_baseline"]
    return merged


def _write_summary(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Write aggregate summary tables."""
    agg = (
        metrics.groupby(["response", "model_spec", "predictor"])
        .agg(
            n_sites=("r2", "count"),
            r2_mean=("r2", "mean"),
            r2_median=("r2", "median"),
            delta_r2_mean=("delta_r2", "mean"),
            delta_r2_median=("delta_r2", "median"),
            partial_corr_mean=("partial_corr", "mean"),
            partial_corr_median=("partial_corr", "median"),
            partial_r2_mean=("partial_r2", "mean"),
            partial_r2_median=("partial_r2", "median"),
        )
        .reset_index()
    )

    summary_path = output_dir / "aggregate_summary.csv"
    agg.to_csv(summary_path, index=False, float_format="%.4f")
    print(f"Saved {summary_path}")

    # Print to stdout
    print("\n" + "=" * 80)
    for (resp, spec), g in agg.groupby(["response", "model_spec"]):
        print(f"\n{resp.upper()} — {spec}")
        print(
            f"{'predictor':<16} {'n':>5} {'R² med':>8} {'ΔR² med':>8} {'pcorr med':>10}"
        )
        print("-" * 50)
        for _, row in g.iterrows():
            print(
                f"{row['predictor']:<16} {row['n_sites']:>5.0f} "
                f"{row['r2_median']:>8.4f} {row['delta_r2_median']:>8.4f} "
                f"{row['partial_corr_median']:>10.4f}"
            )
    print("=" * 80)


def _write_comparisons(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Write paired comparison tables for the five key hypotheses."""
    rows = []
    for pred_a, pred_b, desc in COMPARISONS:
        for (resp, spec), g in metrics.groupby(["response", "model_spec"]):
            a = g[g["predictor"] == pred_a].set_index("site_id")
            b = g[g["predictor"] == pred_b].set_index("site_id")
            shared = a.index.intersection(b.index)
            if len(shared) < 5:
                continue

            r2_a = a.loc[shared, "r2"].values
            r2_b = b.loc[shared, "r2"].values
            diff = r2_a - r2_b
            finite = np.isfinite(diff)
            diff = diff[finite]
            if len(diff) < 5:
                continue

            # Wilcoxon signed-rank test
            try:
                _, pval = sp_stats.wilcoxon(diff, alternative="greater")
            except ValueError:
                pval = np.nan

            rows.append(
                {
                    "comparison": desc,
                    "pred_a": pred_a,
                    "pred_b": pred_b,
                    "response": resp,
                    "model_spec": spec,
                    "n_sites": len(diff),
                    "diff_mean": diff.mean(),
                    "diff_median": np.median(diff),
                    "frac_a_wins": (diff > 0).mean(),
                    "wilcoxon_p": pval,
                }
            )

    if not rows:
        return
    comp_df = pd.DataFrame(rows)
    comp_path = output_dir / "comparisons.csv"
    comp_df.to_csv(comp_path, index=False, float_format="%.4f")
    print(f"Saved {comp_path}")

    # Print
    print("\nKey Comparisons (4var model, ET):")
    sub = comp_df[(comp_df["model_spec"] == "4var") & (comp_df["response"] == "et")]
    if sub.empty:
        sub = comp_df[comp_df["response"] == "et"].head(len(COMPARISONS))
    for _, row in sub.iterrows():
        sig = (
            "***"
            if row["wilcoxon_p"] < 0.001
            else "**"
            if row["wilcoxon_p"] < 0.01
            else "*"
            if row["wilcoxon_p"] < 0.05
            else "ns"
        )
        print(
            f"  {row['comparison']:<35} n={row['n_sites']:>3.0f}  "
            f"ΔR²={row['diff_median']:+.4f}  "
            f"A wins {row['frac_a_wins']:.0%}  "
            f"p={row['wilcoxon_p']:.4f} {sig}"
        )


# ---------------------------------------------------------------------------
# Stratified analysis
# ---------------------------------------------------------------------------


def stratify_by_network(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by network (AmeriFlux/ICOS/OzFlux)."""
    return (
        metrics.groupby(["response", "model_spec", "predictor", "network"])
        .agg(
            n_sites=("r2", "count"),
            r2_median=("r2", "median"),
            delta_r2_median=("delta_r2", "median"),
            partial_corr_median=("partial_corr", "median"),
        )
        .reset_index()
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description="Flux tower regression: soil water predictors vs ET/GPP"
    )
    p.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to flux_site_daily.parquet",
    )
    p.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for results",
    )
    p.add_argument(
        "--min-site-days",
        type=int,
        default=MIN_SITE_DAYS,
        help="Minimum site-days after filtering (default: 30)",
    )
    args = p.parse_args()

    print(f"Loading {args.input}...")
    daily = pd.read_parquet(args.input)
    print(f"  {len(daily)} rows, {daily['site_id'].nunique()} sites")

    metrics = run_regression(
        daily,
        output_dir=Path(args.output_dir),
        min_site_days=args.min_site_days,
    )

    if not metrics.empty:
        # Network stratification
        by_net = stratify_by_network(metrics)
        net_path = Path(args.output_dir) / "by_network.csv"
        by_net.to_csv(net_path, index=False, float_format="%.4f")
        print(f"Saved {net_path}")


if __name__ == "__main__":
    main()
