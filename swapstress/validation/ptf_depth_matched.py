"""
Depth-matched POLARIS rerun for the PTF comparison.

Per ``notes/polaris_depth_matched_handoff.md``: the published PTF comparison
applies POLARIS 0-5 cm van Genuchten (VG) parameters to every held-out
observation, although 46.3% of the candidate rows are measured deeper than
5 cm. This module reruns the comparison with the POLARIS layer matched to
each observation's depth, alongside the prespecified sensitivity rules and a
strict inverse-domain status, without touching the release artifacts, the
QRF predictions, the Rosetta estimates, or the candidate row construction.

Subcommands
-----------
prep      Sample POLARIS 0-5 and 5-15 cm vG parameters from EE, in one call,
          at the sites already present in the corrected site table.
build     Join the depth-matched POLARIS parameters onto the held-out
          candidate rows, select a layer per the primary and sensitivity
          depth rules, and classify strict inverse-domain status.
metrics   Compute the paired accuracy metrics and write the summary outputs.

Usage:
    python -m swapstress.validation.ptf_depth_matched prep \
        --site-params /nas/soils/swapstress/releases/v03_20260729/evaluation/site_vg_params_cm.parquet \
        --output /nas/soils/swapstress/evaluation/ptf_depth_matched/site_vg_params_depth_matched.parquet

    python -m swapstress.validation.ptf_depth_matched build \
        --site-params-depth-matched /nas/soils/swapstress/evaluation/ptf_depth_matched/site_vg_params_depth_matched.parquet \
        --output /nas/soils/swapstress/evaluation/ptf_depth_matched/ptf_depth_matched_observations.parquet

    python -m swapstress.validation.ptf_depth_matched metrics \
        --observations /nas/soils/swapstress/evaluation/ptf_depth_matched/ptf_depth_matched_observations.parquet \
        --output-dir /nas/soils/swapstress/evaluation/ptf_depth_matched
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from swapstress.swrc import psi_from_theta, theta_from_psi, valid_params
from swapstress.units import KPA_TO_CM

DEFAULT_SITE_PARAMS_CM = (
    "/nas/soils/swapstress/releases/v03_20260729/evaluation/site_vg_params_cm.parquet"
)
DEFAULT_MODEL_DIR = "/nas/soils/swapstress/models/direct_qrf_9km_global_pruned"
# Read-only reference: the release evaluation directory. Never written to by
# this module.
DEFAULT_PTF_DIR = "/nas/soils/swapstress/releases/v03_20260729/evaluation"

DEFAULT_DEPTH_MATCHED_DIR = "/nas/soils/swapstress/evaluation/ptf_depth_matched"
DEFAULT_SITE_PARAMS_DEPTH_MATCHED = os.path.join(
    DEFAULT_DEPTH_MATCHED_DIR, "site_vg_params_depth_matched.parquet"
)
DEFAULT_OBSERVATIONS = os.path.join(
    DEFAULT_DEPTH_MATCHED_DIR, "ptf_depth_matched_observations.parquet"
)

POLARIS_LAYERS = ("0_5", "5_15")
POLARIS_ASSETS = {
    "0_5": {
        "theta_r": "projects/sat-io/open-datasets/polaris/theta_r_mean/theta_r_0_5",
        "theta_s": "projects/sat-io/open-datasets/polaris/theta_s_mean/theta_s_0_5",
        "alpha": "projects/sat-io/open-datasets/polaris/alpha_mean/alpha_0_5",
        "n": "projects/sat-io/open-datasets/polaris/n_mean/n_0_5",
    },
    "5_15": {
        "theta_r": "projects/sat-io/open-datasets/polaris/theta_r_mean/theta_r_5_15",
        "theta_s": "projects/sat-io/open-datasets/polaris/theta_s_mean/theta_s_5_15",
        "alpha": "projects/sat-io/open-datasets/polaris/alpha_mean/alpha_5_15",
        "n": "projects/sat-io/open-datasets/polaris/n_mean/n_5_15",
    },
}
POLARIS_SCALE = 250

# The columns that identify one observation across the candidate/holdout/PTF
# tables. Matches swapstress.figures.fig05_ptf_comparison.JOIN_KEYS; kept as
# a separate constant here so this module has no import-time dependency on
# the (matplotlib-backed) figures package.
JOIN_KEYS = ["sample_id", "source", "theta", "log10_suction_cm"]

DOMAIN_STATUSES = (
    "missing_parameters",
    "invalid_parameters",
    "below_or_equal_theta_r",
    "above_or_equal_theta_s",
    "in_domain",
)

# ---------------------------------------------------------------------------
# Prep: sample both POLARIS layers from EE in one call
# ---------------------------------------------------------------------------


def _sample_polaris_layers_at_sites(
    lats, lons, sample_ids, *, scale=POLARIS_SCALE, batch_size=500
):
    """Sample POLARIS 0-5 and 5-15 cm vG parameters from EE, one call per batch.

    Returns a dict with ``pol_0_5_theta_r``, ``pol_0_5_theta_s``,
    ``pol_0_5_alpha``, ``pol_0_5_n``, and the same four keys under
    ``pol_5_15_*``, each an ndarray aligned to ``sample_ids``.

    Alpha is converted independently for each layer, from POLARIS's
    log10(kPa^-1) to the pipeline's natural cm^-1 (see
    ``swapstress.units.KPA_TO_CM``); n is distributed in natural scale and
    taken as-is. Both layers are sampled in the same ``sampleRegions`` call
    per batch, so asset, projection, and pixel-snap are identical between
    layers and between this run and the original 0-5 cm extraction.
    """
    import ee

    ee.Initialize(project="ee-dgketchum")

    bands = []
    for layer in POLARIS_LAYERS:
        assets = POLARIS_ASSETS[layer]
        bands.extend(
            [
                ee.Image(assets["theta_r"]).rename(f"theta_r_{layer}"),
                ee.Image(assets["theta_s"]).rename(f"theta_s_{layer}"),
                ee.Image(assets["alpha"]).rename(f"alpha_{layer}"),
                ee.Image(assets["n"]).rename(f"n_{layer}"),
            ]
        )
    stack = ee.Image.cat(bands)

    features = []
    for sid, lat, lon in zip(sample_ids, lats, lons):
        pt = ee.Geometry.Point([float(lon), float(lat)])
        features.append(ee.Feature(pt, {"sample_id": str(sid)}))

    all_results = []
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        batch_fc = ee.FeatureCollection(features[start:end])
        sampled = stack.sampleRegions(
            collection=batch_fc,
            properties=["sample_id"],
            scale=scale,
        )
        results = sampled.getInfo()
        for feat in results.get("features", []):
            all_results.append(feat["properties"])
        print(
            f"  POLARIS EE batch {start}-{end}: {len(results.get('features', []))} sampled"
        )

    return _parse_polaris_layer_results(all_results, sample_ids)


def _parse_polaris_layer_results(all_results, sample_ids):
    """Turn raw EE ``sampleRegions`` feature properties into per-layer arrays.

    Pure (no EE calls), so the alpha conversion and unit handling for both
    layers is testable without a live Earth Engine session. ``all_results``
    is a list of per-feature property dicts keyed like
    ``theta_r_0_5``/``alpha_5_15``/etc, each carrying ``sample_id``.
    """
    result_map = {}
    for r in all_results:
        sid = r.get("sample_id")
        if sid is not None:
            result_map[sid] = r

    n_sites = len(sample_ids)
    out = {}
    for layer in POLARIS_LAYERS:
        for p in ("theta_r", "theta_s", "alpha", "n"):
            out[f"pol_{layer}_{p}"] = np.full(n_sites, np.nan)

    for i, sid in enumerate(sample_ids):
        r = result_map.get(str(sid))
        if r is None:
            continue
        for layer in POLARIS_LAYERS:
            tr = r.get(f"theta_r_{layer}")
            ts = r.get(f"theta_s_{layer}")
            a = r.get(f"alpha_{layer}")
            nv = r.get(f"n_{layer}")
            if tr is not None:
                out[f"pol_{layer}_theta_r"][i] = tr
            if ts is not None:
                out[f"pol_{layer}_theta_s"][i] = ts
            if a is not None:
                out[f"pol_{layer}_alpha"][i] = (
                    10.0**a / KPA_TO_CM
                )  # log10(1/kPa) -> 1/cm
            if nv is not None:
                out[f"pol_{layer}_n"][i] = nv  # already natural scale

    return out


def run_prep(args):
    """Sample both POLARIS layers at the corrected site table's CONUS sites."""
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        print(f"Output exists: {output_path} (use --overwrite)")
        return

    sites = pd.read_parquet(args.site_params)
    print(f"Loaded {len(sites)} sites from {args.site_params}")

    conus = (
        (sites["lon"] >= -125)
        & (sites["lon"] <= -66)
        & (sites["lat"] >= 24)
        & (sites["lat"] <= 50)
    )
    conus_sites = sites[conus]
    print(
        f"Sampling POLARIS 0-5 and 5-15 cm from EE at {len(conus_sites)} CONUS sites ..."
    )
    pol = _sample_polaris_layers_at_sites(
        conus_sites["lat"].values,
        conus_sites["lon"].values,
        conus_sites["sample_id"].values,
    )

    out = sites.copy()
    for k, v in pol.items():
        out.loc[conus, k] = v
    out["pol_0_5_alpha_units"] = "cm^-1"
    out["pol_5_15_alpha_units"] = "cm^-1"

    for layer in POLARIS_LAYERS:
        n_valid = int(np.isfinite(out[f"pol_{layer}_alpha"]).sum())
        print(
            f"  {n_valid}/{len(conus_sites)} sites with valid {layer.replace('_', '-')} cm params"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(str(output_path), index=False)
    print(f"\nWrote {output_path}")


# ---------------------------------------------------------------------------
# Layer selection rules
# ---------------------------------------------------------------------------


def select_layer_primary(depth_cm):
    """``depth_cm <= 5.0 -> '0_5'``, ``depth_cm > 5.0 -> '5_15'``."""
    depth_cm = np.asarray(depth_cm, dtype=np.float64)
    return np.where(depth_cm <= 5.0, "0_5", "5_15")


def select_layer_upper_at_boundary(depth_cm):
    """Sensitivity: ``depth_cm < 5.0 -> '0_5'``, ``depth_cm >= 5.0 -> '5_15'``."""
    depth_cm = np.asarray(depth_cm, dtype=np.float64)
    return np.where(depth_cm < 5.0, "0_5", "5_15")


def select_layer_nearest_midpoint(depth_cm):
    """Sensitivity: nearest of representative midpoints 2.5 cm / 10 cm, switching at 6.25 cm."""
    depth_cm = np.asarray(depth_cm, dtype=np.float64)
    return np.where(depth_cm < 6.25, "0_5", "5_15")


DEPTH_RULES = {
    "primary": select_layer_primary,
    "upper_at_boundary": select_layer_upper_at_boundary,
    "nearest_midpoint": select_layer_nearest_midpoint,
}


# ---------------------------------------------------------------------------
# Strict inverse-domain status
# ---------------------------------------------------------------------------


def classify_domain_status(theta, theta_r, theta_s, alpha, n):
    """One of :data:`DOMAIN_STATUSES` per row.

    ``missing_parameters``: any of theta_r/theta_s/alpha/n is non-finite.
    ``invalid_parameters``: all four are finite but fail
        :func:`swapstress.swrc.valid_params` (alpha<=0, n<=1, theta_s<=theta_r).
    ``below_or_equal_theta_r`` / ``above_or_equal_theta_s``: valid parameters,
        but the observed theta is outside the retention curve's domain.
    ``in_domain``: valid parameters and ``theta_r < theta < theta_s``.
    """
    theta = np.asarray(theta, dtype=np.float64)
    theta_r = np.asarray(theta_r, dtype=np.float64)
    theta_s = np.asarray(theta_s, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)

    has_params = (
        np.isfinite(theta_r)
        & np.isfinite(theta_s)
        & np.isfinite(alpha)
        & np.isfinite(n)
    )
    valid = has_params & valid_params(theta_r, theta_s, alpha, n)

    status = np.full(theta.shape, "", dtype=object)
    status[~has_params] = "missing_parameters"
    status[has_params & ~valid] = "invalid_parameters"
    status[valid & (theta <= theta_r)] = "below_or_equal_theta_r"
    status[valid & (theta >= theta_s)] = "above_or_equal_theta_s"
    status[valid & (theta > theta_r) & (theta < theta_s)] = "in_domain"
    return status


# ---------------------------------------------------------------------------
# Candidate construction (depth-aware variant of fig05's load_holdout)
# ---------------------------------------------------------------------------


def load_holdout_with_depth(model_dir=DEFAULT_MODEL_DIR, ptf_dir=DEFAULT_PTF_DIR):
    """PTF comparison rows restricted to the model's spatial holdout, with depth_cm.

    Reproduces ``swapstress.figures.fig05_ptf_comparison.load_holdout`` exactly
    -- same alignment check, same join keys, same dedup -- and additionally
    carries ``depth_cm`` through from ``test_set_full.parquet``, which the
    released ``ptf_comparison_observations.parquet`` does not retain.
    """
    model = Path(model_dir)
    test = pd.read_parquet(model / "test_set_full.parquet")
    preds = pd.read_parquet(model / "predictions.parquet")
    if len(test) != len(preds):
        raise ValueError(
            f"{model.name}: test_set_full has {len(test):,} rows and "
            f"predictions has {len(preds):,}; they must be row-aligned."
        )
    if not np.allclose(
        test["log10_suction_cm"].values, preds["observed"].values, equal_nan=True
    ):
        raise ValueError(
            f"{model.name}: predictions.parquet is not row-aligned with "
            "test_set_full.parquet -- 'observed' does not match "
            "'log10_suction_cm'."
        )

    holdout = test[JOIN_KEYS + ["depth_cm"]].copy()
    holdout["rf_pred"] = preds["predicted"].values
    holdout = holdout.drop_duplicates(subset=JOIN_KEYS)

    ptf = pd.read_parquet(Path(ptf_dir) / "ptf_comparison_observations.parquet")
    ptf = ptf.drop_duplicates(subset=JOIN_KEYS)

    merged = ptf.merge(holdout, on=JOIN_KEYS, how="inner")
    print(
        f"{len(ptf):,} PTF observations, {len(holdout):,} held-out rows, "
        f"{len(merged):,} in both"
    )
    if merged.empty:
        raise ValueError(
            "No PTF observations fall in the model's spatial holdout; there "
            "is nothing out-of-sample to score."
        )
    if merged["depth_cm"].isna().any():
        raise ValueError(
            f"{merged['depth_cm'].isna().sum()} candidate rows have no depth_cm "
            "after the join; depth-layer selection cannot proceed for them."
        )
    return merged


# ---------------------------------------------------------------------------
# Build: depth-aware observation table
# ---------------------------------------------------------------------------


def build_observation_table(
    model_dir=DEFAULT_MODEL_DIR,
    ptf_dir=DEFAULT_PTF_DIR,
    site_params_depth_matched=DEFAULT_SITE_PARAMS_DEPTH_MATCHED,
):
    """Load the candidate rows and depth-matched site params, then assemble.

    Thin disk-reading wrapper around :func:`_assemble_depth_matched`, which
    holds the actual join/selection/scoring logic and takes plain DataFrames
    so it is testable without the release and model-holdout files on disk.
    """
    candidates = load_holdout_with_depth(model_dir, ptf_dir)
    site = pd.read_parquet(site_params_depth_matched)
    return _assemble_depth_matched(candidates, site)


def _assemble_depth_matched(candidates, site):
    """Join depth-matched POLARIS params onto the candidate rows and score them.

    Retains, per row: depth_cm, the layer picked by each of the three
    depth-boundary rules, both layers' raw parameter sets (for auditability),
    the primary-rule selected parameters, strict domain status for both
    POLARIS (selected layer) and Rosetta, the strict and capped log10
    suction, and a same-cache 0-5 cm legacy control.
    """
    site_cols = ["sample_id"]
    for layer in POLARIS_LAYERS:
        site_cols += [
            f"pol_{layer}_theta_r",
            f"pol_{layer}_theta_s",
            f"pol_{layer}_alpha",
            f"pol_{layer}_n",
            f"pol_{layer}_alpha_units",
        ]
    site = site[site_cols].drop_duplicates(subset=["sample_id"])

    df = candidates.merge(site, on="sample_id", how="left")
    if len(df) != len(candidates):
        raise ValueError(
            f"site join changed row count: {len(candidates)} candidates -> "
            f"{len(df)} after merge; the sample_id join became one-to-many."
        )

    depth_cm = df["depth_cm"].values
    for rule_name, rule_fn in DEPTH_RULES.items():
        layer_label = rule_fn(depth_cm)
        df[f"polaris_layer_{rule_name}"] = layer_label
        for p in ("theta_r", "theta_s", "alpha", "n"):
            df[f"pol_selected_{rule_name}_{p}"] = np.where(
                layer_label == "0_5",
                df[f"pol_0_5_{p}"].values,
                df[f"pol_5_15_{p}"].values,
            )

    # Primary-rule convenience columns (unsuffixed), used for the headline metrics.
    df["polaris_layer"] = df["polaris_layer_primary"]
    for p in ("theta_r", "theta_s", "alpha", "n"):
        df[f"pol_selected_{p}"] = df[f"pol_selected_primary_{p}"]

    df["polaris_domain_status"] = classify_domain_status(
        df["theta"].values,
        df["pol_selected_theta_r"].values,
        df["pol_selected_theta_s"].values,
        df["pol_selected_alpha"].values,
        df["pol_selected_n"].values,
    )
    pol_h_capped = psi_from_theta(
        df["theta"].values,
        df["pol_selected_theta_r"].values,
        df["pol_selected_theta_s"].values,
        df["pol_selected_alpha"].values,
        df["pol_selected_n"].values,
    )
    df["pol_depth_matched_log10_suction_capped"] = np.where(
        pol_h_capped > 0, np.log10(pol_h_capped), np.nan
    )
    df["pol_depth_matched_log10_suction"] = np.where(
        df["polaris_domain_status"].values == "in_domain",
        df["pol_depth_matched_log10_suction_capped"].values,
        np.nan,
    )

    # Forward diagnostic: theta_predicted(psi_observed), over rows with valid
    # (not necessarily in-domain) selected-layer parameters.
    psi_observed_cm = 10.0 ** df["log10_suction_cm"].values
    df["theta_forward_pred_depth_matched"] = theta_from_psi(
        psi_observed_cm,
        df["pol_selected_theta_r"].values,
        df["pol_selected_theta_s"].values,
        df["pol_selected_alpha"].values,
        df["pol_selected_n"].values,
        strict=True,
    )

    # Same-cache 0-5 cm legacy control, from the newly sampled 0-5 columns.
    df["polaris_legacy_0_5_domain_status"] = classify_domain_status(
        df["theta"].values,
        df["pol_0_5_theta_r"].values,
        df["pol_0_5_theta_s"].values,
        df["pol_0_5_alpha"].values,
        df["pol_0_5_n"].values,
    )
    legacy_h = psi_from_theta(
        df["theta"].values,
        df["pol_0_5_theta_r"].values,
        df["pol_0_5_theta_s"].values,
        df["pol_0_5_alpha"].values,
        df["pol_0_5_n"].values,
    )
    df["polaris_legacy_0_5_log10_suction_capped"] = np.where(
        legacy_h > 0, np.log10(legacy_h), np.nan
    )
    df["polaris_legacy_0_5_log10_suction"] = np.where(
        df["polaris_legacy_0_5_domain_status"].values == "in_domain",
        df["polaris_legacy_0_5_log10_suction_capped"].values,
        np.nan,
    )

    # Rosetta status, from the release's own ros_* parameters -- classification
    # only, the released ros_log10_suction values are untouched.
    df["rosetta_domain_status"] = classify_domain_status(
        df["theta"].values,
        df["ros_theta_r"].values,
        df["ros_theta_s"].values,
        df["ros_alpha"].values,
        df["ros_n"].values,
    )
    df["ros_log10_suction_strict"] = np.where(
        df["rosetta_domain_status"].values == "in_domain",
        df["ros_log10_suction"].values,
        np.nan,
    )

    return df


def run_build(args):
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_observation_table(
        model_dir=args.model_dir,
        ptf_dir=args.ptf_dir,
        site_params_depth_matched=args.site_params_depth_matched,
    )

    n_0_5 = int((df["polaris_layer"] == "0_5").sum())
    n_5_15 = int((df["polaris_layer"] == "5_15").sum())
    print(f"Primary layer assignment: {n_0_5} rows at 0-5 cm, {n_5_15} rows at 5-15 cm")
    print(f"Rows: {len(df)}, sites: {df['sample_id'].nunique()}")

    df.to_parquet(str(output_path), index=False)
    print(f"Wrote {output_path}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _paired_metrics(pred, obs):
    """n, RMSE, MAE, median absolute error, bias, R2 over finite pairs."""
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(obs)
    n = int(mask.sum())
    if n < 2:
        return {
            "n": n,
            "rmse": np.nan,
            "mae": np.nan,
            "medae": np.nan,
            "bias": np.nan,
            "r2": np.nan,
        }
    p, o = pred[mask], obs[mask]
    resid = p - o
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((o - o.mean()) ** 2))
    return {
        "n": n,
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mae": float(np.mean(np.abs(resid))),
        "medae": float(np.median(np.abs(resid))),
        "bias": float(np.mean(resid)),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
    }


def _site_balanced_metrics(pred, obs, site_ids):
    """As :func:`_paired_metrics`, but each site's mean residual counts once.

    Many candidate rows share a single site-level parameter vector (a
    time-series site sampled at multiple theta/suction pairs), so an
    observation-weighted score is implicitly weighted by how often a site was
    sampled. This collapses to one residual per site (its mean) before
    scoring, so a site with 1 observation and a site with 100 contribute
    equally.
    """
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    site_ids = np.asarray(site_ids)
    mask = np.isfinite(pred) & np.isfinite(obs)
    n = int(mask.sum())
    if n < 2:
        return {
            "n_sites": 0,
            "n_obs": n,
            "rmse": np.nan,
            "mae": np.nan,
            "medae": np.nan,
            "bias": np.nan,
            "r2": np.nan,
        }
    resid = pred[mask] - obs[mask]
    site_df = pd.DataFrame({"site": site_ids[mask], "resid": resid, "obs": obs[mask]})
    per_site = site_df.groupby("site").agg(resid=("resid", "mean"), obs=("obs", "mean"))
    r = per_site["resid"].values
    o = per_site["obs"].values
    ss_res = float(np.sum(r**2))
    ss_tot = float(np.sum((o - o.mean()) ** 2))
    return {
        "n_sites": int(len(per_site)),
        "n_obs": n,
        "rmse": float(np.sqrt(np.mean(r**2))),
        "mae": float(np.mean(np.abs(r))),
        "medae": float(np.median(np.abs(r))),
        "bias": float(np.mean(r)),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
    }


def _coverage_row(label, status_col_values):
    counts = pd.Series(status_col_values).value_counts()
    row = {"comparison": label, "n_total": len(status_col_values)}
    for s in DOMAIN_STATUSES:
        row[s] = int(counts.get(s, 0))
    return row


def _exceedance_magnitude(theta, theta_r, theta_s, status):
    """Median/mean magnitude (m3/m3) of dry (theta_r - theta) and wet (theta - theta_s) exceedance."""
    theta = np.asarray(theta, dtype=np.float64)
    theta_r = np.asarray(theta_r, dtype=np.float64)
    theta_s = np.asarray(theta_s, dtype=np.float64)
    status = np.asarray(status)
    dry = status == "below_or_equal_theta_r"
    wet = status == "above_or_equal_theta_s"
    dry_mag = theta_r[dry] - theta[dry]
    wet_mag = theta[wet] - theta_s[wet]
    return {
        "dry_n": int(dry.sum()),
        "dry_median": float(np.median(dry_mag)) if dry.any() else np.nan,
        "dry_mean": float(np.mean(dry_mag)) if dry.any() else np.nan,
        "wet_n": int(wet.sum()),
        "wet_median": float(np.median(wet_mag)) if wet.any() else np.nan,
        "wet_mean": float(np.mean(wet_mag)) if wet.any() else np.nan,
    }


def compute_metrics(df, output_dir):
    """Compute every metric block from the handoff and write the output files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    observed = df["log10_suction_cm"].values
    site_ids = df["sample_id"].values

    rows = []

    # 1/2. Coverage and dry/wet exceedance, primary depth-matched selection and legacy control.
    coverage_rows = [
        _coverage_row("polaris_depth_matched", df["polaris_domain_status"].values),
        _coverage_row(
            "polaris_legacy_0_5_control", df["polaris_legacy_0_5_domain_status"].values
        ),
        _coverage_row("rosetta", df["rosetta_domain_status"].values),
    ]
    exceed_rows = [
        {
            "comparison": "polaris_depth_matched",
            **_exceedance_magnitude(
                df["theta"].values,
                df["pol_selected_theta_r"].values,
                df["pol_selected_theta_s"].values,
                df["polaris_domain_status"].values,
            ),
        },
        {
            "comparison": "polaris_legacy_0_5_control",
            **_exceedance_magnitude(
                df["theta"].values,
                df["pol_0_5_theta_r"].values,
                df["pol_0_5_theta_s"].values,
                df["polaris_legacy_0_5_domain_status"].values,
            ),
        },
        {
            "comparison": "rosetta",
            **_exceedance_magnitude(
                df["theta"].values,
                df["ros_theta_r"].values,
                df["ros_theta_s"].values,
                df["rosetta_domain_status"].values,
            ),
        },
    ]

    def add(label, pred, obs=observed, weighted=True, site_balanced=True):
        m = _paired_metrics(pred, obs)
        rows.append({"comparison": label, "weighting": "observation", **m})
        if site_balanced:
            sb = _site_balanced_metrics(pred, obs, site_ids)
            rows.append({"comparison": label, "weighting": "site_balanced", **sb})

    # 3. POLARIS depth-matched vs QRF on the same POLARIS-valid rows.
    pol_valid = df["polaris_domain_status"].values == "in_domain"
    add("polaris_depth_matched_strict", df["pol_depth_matched_log10_suction"].values)
    add("qrf_on_polaris_valid_rows", np.where(pol_valid, df["rf_pred"].values, np.nan))
    add(
        "polaris_depth_matched_capped_se_eps_sensitivity",
        df["pol_depth_matched_log10_suction_capped"].values,
    )

    # Legacy 0-5 control, reproduced from this same cache.
    add(
        "polaris_legacy_0_5_control_strict",
        df["polaris_legacy_0_5_log10_suction"].values,
    )
    add(
        "polaris_legacy_0_5_control_capped_se_eps_sensitivity",
        df["polaris_legacy_0_5_log10_suction_capped"].values,
    )

    # Rosetta, QRF full coverage, for reference (unchanged from the release).
    add("rosetta_strict", df["ros_log10_suction_strict"].values)
    # ros_log10_suction is the release's own Se-clipped inversion, scored on
    # every parameter-available row regardless of domain status -- the same
    # capped regime as the POLARIS *_capped_se_eps_sensitivity rows above,
    # kept here so all three methods' capped and strict numbers sit side by
    # side in the same table.
    add("rosetta_capped_se_eps_sensitivity", df["ros_log10_suction"].values)
    add("qrf_full_coverage", df["rf_pred"].values)

    # 4. Common three-way strict-valid subset.
    common = pol_valid & (df["rosetta_domain_status"].values == "in_domain")
    add("three_way_common_subset_qrf", np.where(common, df["rf_pred"].values, np.nan))
    add(
        "three_way_common_subset_rosetta",
        np.where(common, df["ros_log10_suction"].values, np.nan),
    )
    add(
        "three_way_common_subset_polaris_depth_matched",
        np.where(common, df["pol_depth_matched_log10_suction_capped"].values, np.nan),
    )

    # 6. By selected layer.
    for layer in POLARIS_LAYERS:
        mask = df["polaris_layer"].values == layer
        pred = np.where(mask, df["pol_depth_matched_log10_suction"].values, np.nan)
        add(f"polaris_depth_matched_strict_layer_{layer}", pred, site_balanced=False)

    # 8. Forward diagnostic: theta_predicted(psi_observed).
    theta_obs = df["theta"].values
    theta_pred = df["theta_forward_pred_depth_matched"].values
    fmask = np.isfinite(theta_pred) & np.isfinite(theta_obs)
    fresid = theta_pred[fmask] - theta_obs[fmask]
    if fmask.sum() >= 2:
        ss_res = float(np.sum(fresid**2))
        ss_tot = float(np.sum((theta_obs[fmask] - theta_obs[fmask].mean()) ** 2))
        rows.append(
            {
                "comparison": "forward_theta_depth_matched",
                "weighting": "observation",
                "n": int(fmask.sum()),
                "rmse": float(np.sqrt(np.mean(fresid**2))),
                "mae": float(np.mean(np.abs(fresid))),
                "medae": float(np.median(np.abs(fresid))),
                "bias": float(np.mean(fresid)),
                "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            }
        )

    # 9. Depth-boundary sensitivity rules.
    for rule_name in ("upper_at_boundary", "nearest_midpoint"):
        layer_label = df[f"polaris_layer_{rule_name}"].values
        tr = df[f"pol_selected_{rule_name}_theta_r"].values
        ts = df[f"pol_selected_{rule_name}_theta_s"].values
        al = df[f"pol_selected_{rule_name}_alpha"].values
        nn = df[f"pol_selected_{rule_name}_n"].values
        status = classify_domain_status(df["theta"].values, tr, ts, al, nn)
        h = psi_from_theta(df["theta"].values, tr, ts, al, nn)
        log10_capped = np.where(h > 0, np.log10(h), np.nan)
        log10_strict = np.where(status == "in_domain", log10_capped, np.nan)
        add(f"polaris_{rule_name}_strict", log10_strict, site_balanced=False)
        n_0_5 = int((layer_label == "0_5").sum())
        n_5_15 = int((layer_label == "5_15").sum())
        coverage_rows.append(_coverage_row(f"polaris_{rule_name}", status))
        rows.append(
            {
                "comparison": f"polaris_{rule_name}_layer_counts",
                "weighting": "n/a",
                "n": len(layer_label),
                "rmse": np.nan,
                "mae": np.nan,
                "medae": np.nan,
                "bias": np.nan,
                "r2": np.nan,
                "n_0_5": n_0_5,
                "n_5_15": n_5_15,
            }
        )

    metrics_df = pd.DataFrame(rows)
    metrics_path = output_dir / "ptf_depth_matched_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Wrote {metrics_path}")

    coverage_df = pd.DataFrame(coverage_rows)
    coverage_path = output_dir / "ptf_depth_matched_coverage.csv"
    coverage_df.to_csv(coverage_path, index=False)
    print(f"Wrote {coverage_path}")

    exceed_df = pd.DataFrame(exceed_rows)
    exceed_path = output_dir / "ptf_depth_matched_exceedance.csv"
    exceed_df.to_csv(exceed_path, index=False)
    print(f"Wrote {exceed_path}")

    # By source.
    by_source_rows = []
    for src in sorted(df["source"].unique()):
        smask = df["source"].values == src
        sub = df.loc[smask]
        for label, pred_col, status_col in (
            (
                "polaris_depth_matched_strict",
                "pol_depth_matched_log10_suction",
                "polaris_domain_status",
            ),
            ("rosetta_strict", "ros_log10_suction_strict", "rosetta_domain_status"),
            ("qrf_full_coverage", "rf_pred", None),
        ):
            m = _paired_metrics(sub[pred_col].values, sub["log10_suction_cm"].values)
            by_source_rows.append({"source": src, "comparison": label, **m})
    by_source_path = output_dir / "ptf_depth_matched_by_source.csv"
    pd.DataFrame(by_source_rows).to_csv(by_source_path, index=False)
    print(f"Wrote {by_source_path}")

    # By depth (stratified check, requirement 3).
    by_depth_rows = []
    for depth_val in sorted(df["depth_cm"].unique()):
        dmask = df["depth_cm"].values == depth_val
        sub = df.loc[dmask]
        m = _paired_metrics(
            sub["pol_depth_matched_log10_suction"].values,
            sub["log10_suction_cm"].values,
        )
        by_depth_rows.append(
            {"depth_cm": depth_val, "comparison": "polaris_depth_matched_strict", **m}
        )
    for stratum_label, stratum_mask in (
        ("depth_le_5", df["depth_cm"].values <= 5.0),
        ("depth_gt_5", df["depth_cm"].values > 5.0),
    ):
        sub = df.loc[stratum_mask]
        m = _paired_metrics(
            sub["pol_depth_matched_log10_suction"].values,
            sub["log10_suction_cm"].values,
        )
        by_depth_rows.append(
            {
                "depth_cm": stratum_label,
                "comparison": "polaris_depth_matched_strict_stratum",
                **m,
            }
        )
    by_depth_path = output_dir / "ptf_depth_matched_by_depth.csv"
    pd.DataFrame(by_depth_rows).to_csv(by_depth_path, index=False)
    print(f"Wrote {by_depth_path}")

    return {
        "metrics_path": metrics_path,
        "coverage_path": coverage_path,
        "exceedance_path": exceed_path,
        "by_source_path": by_source_path,
        "by_depth_path": by_depth_path,
    }


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_revision():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def write_provenance(df, observations_path, output_paths, output_dir):
    output_dir = Path(output_dir)
    checksums = {"observations": _sha256(observations_path)}
    for name, path in output_paths.items():
        checksums[name] = _sha256(path)

    provenance = {
        "polaris_assets": POLARIS_ASSETS,
        "polaris_scale": POLARIS_SCALE,
        "alpha_conversion": "10**value / KPA_TO_CM  (log10(1/kPa) -> 1/cm)",
        "kpa_to_cm": KPA_TO_CM,
        "layer_rules": {
            "primary": "depth_cm <= 5.0 -> 0-5 cm, depth_cm > 5.0 -> 5-15 cm",
            "upper_at_boundary": "depth_cm < 5.0 -> 0-5 cm, depth_cm >= 5.0 -> 5-15 cm",
            "nearest_midpoint": "midpoints 2.5/10 cm, switch at 6.25 cm",
        },
        "se_eps": 1e-6,
        "candidate_input_paths": {
            "model_dir": DEFAULT_MODEL_DIR,
            "ptf_dir": DEFAULT_PTF_DIR,
        },
        "site_params_depth_matched_path": str(DEFAULT_SITE_PARAMS_DEPTH_MATCHED),
        "n_candidate_rows": int(len(df)),
        "n_sites": int(df["sample_id"].nunique()),
        "git_revision": _git_revision(),
        "output_checksums_sha256": checksums,
    }
    path = output_dir / "provenance.json"
    with open(path, "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"Wrote {path}")
    return path


def run_metrics(args):
    df = pd.read_parquet(args.observations)
    output_paths = compute_metrics(df, args.output_dir)
    write_provenance(df, args.observations, output_paths, args.output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(
        description="Depth-matched POLARIS rerun for the PTF comparison (Fig 5)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser(
        "prep", help="Sample POLARIS 0-5 and 5-15 cm at site coordinates"
    )
    p_prep.add_argument("--site-params", default=DEFAULT_SITE_PARAMS_CM)
    p_prep.add_argument("--output", default=DEFAULT_SITE_PARAMS_DEPTH_MATCHED)
    p_prep.add_argument("--overwrite", action="store_true", default=None)
    p_prep.set_defaults(func=run_prep)

    p_build = sub.add_parser("build", help="Build the depth-aware observation table")
    p_build.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p_build.add_argument("--ptf-dir", default=DEFAULT_PTF_DIR)
    p_build.add_argument(
        "--site-params-depth-matched", default=DEFAULT_SITE_PARAMS_DEPTH_MATCHED
    )
    p_build.add_argument("--output", default=DEFAULT_OBSERVATIONS)
    p_build.set_defaults(func=run_build)

    p_metrics = sub.add_parser(
        "metrics", help="Compute paired metrics from the observation table"
    )
    p_metrics.add_argument("--observations", default=DEFAULT_OBSERVATIONS)
    p_metrics.add_argument("--output-dir", default=DEFAULT_DEPTH_MATCHED_DIR)
    p_metrics.set_defaults(func=run_metrics)

    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
