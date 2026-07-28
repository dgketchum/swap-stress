"""
Compare our direct SWP model against PTF-derived baselines (Rosetta, POLARIS).

For each (theta, suction) observation at rosetta_level=2 in the training table,
computes suction from the van Genuchten equation using Rosetta and POLARIS
surface-layer parameters sampled at each site, then compares those physics-based
estimates against the observed suction and our model's predictions.

Subcommands
-----------
prep      Extract Rosetta L2 and POLARIS 0-5 cm vG parameters at training sites.
evaluate  Compare PTF-derived suction against observed and model-predicted values.

Usage:
    python -m map.evaluation.ptf_baseline prep \
        --training-table /nas/soils/swapstress/training/obs_level_training_9km_global.parquet \
        --output /nas/soils/swapstress/evaluation/ptf_baseline/site_vg_params.parquet

    python -m map.evaluation.ptf_baseline evaluate \
        --training-table /nas/soils/swapstress/training/obs_level_training_9km_global.parquet \
        --vg-params /nas/soils/swapstress/evaluation/ptf_baseline/site_vg_params.parquet \
        --output /nas/soils/swapstress/evaluation/ptf_baseline
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from swapstress.swrc import psi_from_theta

DEFAULT_TRAINING_TABLE = (
    "/nas/soils/swapstress/training/obs_level_training_9km_global.parquet"
)
DEFAULT_ROSETTA_TIF = "/nas/soils/rosetta/geotiff/US_R3H3_L2_VG.tiff"
DEFAULT_OUTPUT_DIR = "/nas/soils/swapstress/evaluation/ptf_baseline"

# ---------------------------------------------------------------------------
# Prep: extract vG parameters at training sites
# ---------------------------------------------------------------------------


def _get_sites(training_table, rosetta_level=2):
    """Load unique site locations at the target rosetta level.

    Returns a DataFrame with columns: sample_id, lat, lon, source
    (one row per unique sample_id).
    """
    df = pd.read_parquet(
        training_table,
        columns=["sample_id", "rosetta_level", "lat", "lon", "source"],
    )
    rl = df[df["rosetta_level"] == rosetta_level].dropna(subset=["lat", "lon"])
    sites = (
        rl.groupby("sample_id")
        .agg({"lat": "first", "lon": "first", "source": "first"})
        .reset_index()
    )
    return sites


def _sample_rosetta_at_sites(rosetta_tif, lats, lons):
    """Sample the Rosetta L2 5-band GeoTIFF at (lat, lon) coordinates.

    Returns dict with keys: ros_theta_r, ros_theta_s, ros_alpha, ros_n.
    Alpha and n are returned in natural (linear) scale.
    """
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    xs, ys = transformer.transform(
        np.asarray(lons, dtype=np.float64),
        np.asarray(lats, dtype=np.float64),
    )

    n = len(lats)
    theta_r = np.full(n, np.nan)
    theta_s = np.full(n, np.nan)
    alpha = np.full(n, np.nan)
    n_param = np.full(n, np.nan)

    with rasterio.open(rosetta_tif) as src:
        nodata = src.nodata
        for i, (x, y) in enumerate(zip(xs, ys)):
            try:
                row, col = src.index(x, y)
            except Exception:
                continue
            if 0 <= row < src.height and 0 <= col < src.width:
                window = rasterio.windows.Window(col, row, 1, 1)
                vals = src.read(window=window).squeeze()
                if nodata is not None and np.any(vals == nodata):
                    continue
                theta_r[i] = vals[0]
                theta_s[i] = vals[1]
                alpha[i] = 10.0 ** vals[2]  # log10(1/cm) -> 1/cm
                n_param[i] = 10.0 ** vals[3]  # log10 -> natural

    return {
        "ros_theta_r": theta_r,
        "ros_theta_s": theta_s,
        "ros_alpha": alpha,
        "ros_n": n_param,
    }


def _sample_polaris_at_sites(lats, lons, sample_ids):
    """Sample POLARIS 0-5 cm vG parameters from EE at (lat, lon) coordinates.

    Returns dict with keys: pol_theta_r, pol_theta_s, pol_alpha, pol_n.
    Alpha is converted from log10(1/cm) to 1/cm; n is in natural scale.
    """
    import ee

    ee.Initialize(project="ee-dgketchum")

    root = "projects/sat-io/open-datasets/polaris"
    stack = ee.Image.cat(
        [
            ee.Image(f"{root}/theta_r_mean/theta_r_0_5").rename("theta_r"),
            ee.Image(f"{root}/theta_s_mean/theta_s_0_5").rename("theta_s"),
            ee.Image(f"{root}/alpha_mean/alpha_0_5").rename("alpha"),
            ee.Image(f"{root}/n_mean/n_0_5").rename("n"),
        ]
    )

    features = []
    for sid, lat, lon in zip(sample_ids, lats, lons):
        pt = ee.Geometry.Point([float(lon), float(lat)])
        features.append(ee.Feature(pt, {"sample_id": str(sid)}))
    # Sample in batches to avoid payload limits
    batch_size = 500
    all_results = []

    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        batch_features = [features[i] for i in range(start, end)]
        batch_fc = ee.FeatureCollection(batch_features)

        sampled = stack.sampleRegions(
            collection=batch_fc,
            properties=["sample_id"],
            scale=250,
        )
        results = sampled.getInfo()
        for feat in results.get("features", []):
            props = feat["properties"]
            all_results.append(props)

        print(
            f"  POLARIS EE batch {start}-{end}: {len(results.get('features', []))} sampled"
        )

    # Build output arrays indexed by sample_id
    result_map = {}
    for r in all_results:
        sid = r.get("sample_id")
        if sid is not None:
            result_map[sid] = r

    n = len(sample_ids)
    theta_r = np.full(n, np.nan)
    theta_s = np.full(n, np.nan)
    alpha = np.full(n, np.nan)
    n_param = np.full(n, np.nan)

    for i, sid in enumerate(sample_ids):
        r = result_map.get(str(sid))
        if r is None:
            continue
        tr = r.get("theta_r")
        ts = r.get("theta_s")
        a = r.get("alpha")
        nv = r.get("n")
        if tr is not None:
            theta_r[i] = tr
        if ts is not None:
            theta_s[i] = ts
        if a is not None:
            alpha[i] = 10.0**a  # log10(1/cm) -> 1/cm
        if nv is not None:
            n_param[i] = nv  # already natural scale

    return {
        "pol_theta_r": theta_r,
        "pol_theta_s": theta_s,
        "pol_alpha": alpha,
        "pol_n": n_param,
    }


def run_prep(args):
    """Extract Rosetta and POLARIS vG params at training table sites."""
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        print(f"Output exists: {output_path} (use --overwrite)")
        return

    print("Loading training table sites at rosetta_level=2 ...")
    sites = _get_sites(args.training_table)
    print(f"  {len(sites)} unique sites")
    print(f"  Sources: {sites['source'].value_counts().to_dict()}")

    # Rosetta: sample local GeoTIFF
    print(f"\nSampling Rosetta L2 at {len(sites)} sites ...")
    ros = _sample_rosetta_at_sites(
        args.rosetta_tif,
        sites["lat"].values,
        sites["lon"].values,
    )
    for k, v in ros.items():
        sites[k] = v
    n_ros = int(np.isfinite(sites["ros_alpha"]).sum())
    print(f"  {n_ros}/{len(sites)} sites with valid Rosetta params")

    # POLARIS: sample from EE (CONUS only)
    conus = (
        (sites["lon"] >= -125)
        & (sites["lon"] <= -66)
        & (sites["lat"] >= 24)
        & (sites["lat"] <= 50)
    )
    conus_sites = sites[conus]
    print(f"\nSampling POLARIS 0-5 cm from EE at {len(conus_sites)} CONUS sites ...")
    pol = _sample_polaris_at_sites(
        conus_sites["lat"].values,
        conus_sites["lon"].values,
        conus_sites["sample_id"].values,
    )
    for k, v in pol.items():
        sites.loc[conus, k] = v
    n_pol = int(np.isfinite(sites.get("pol_alpha", pd.Series(dtype=float))).sum())
    print(f"  {n_pol}/{len(conus_sites)} CONUS sites with valid POLARIS params")

    sites.to_parquet(str(output_path), index=False)
    print(f"\nWrote {output_path}")


# ---------------------------------------------------------------------------
# Evaluate: compare PTF suction against observed and model predictions
# ---------------------------------------------------------------------------


def _metrics(a, b):
    """Compute RMSE, bias, R^2 between two arrays (NaN-safe)."""
    mask = np.isfinite(a) & np.isfinite(b)
    n = mask.sum()
    if n < 2:
        return {"n": int(n), "rmse": np.nan, "bias": np.nan, "r2": np.nan}
    a_v, b_v = a[mask], b[mask]
    diff = a_v - b_v
    rmse = float(np.sqrt(np.mean(diff**2)))
    bias = float(np.mean(diff))
    ss_res = float(np.sum(diff**2))
    ss_tot = float(np.sum((b_v - b_v.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"n": int(n), "rmse": rmse, "bias": bias, "r2": r2}


def run_evaluate(args):
    """Compare PTF-derived suction against observed values."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load full training table for model predictions, then subset
    print("Loading training table (rosetta_level=2) ...")
    full_df = pd.read_parquet(args.training_table)
    rl2 = full_df[full_df["rosetta_level"] == 2].dropna(
        subset=["theta", "log10_suction_cm"],
    )
    print(f"  {len(rl2)} observations across {rl2['sample_id'].nunique()} sites")

    # Run SWAP model: use k-fold out-of-fold predictions if available
    model_log10 = np.full(len(rl2), np.nan)
    if args.model_dir:
        import json

        import joblib

        model_path = Path(args.model_dir)

        # Check for k-fold structure
        kfold_summary = model_path / "kfold_summary.json"
        fold_0_split = model_path / "fold_0" / "spatial_split.json"

        if kfold_summary.exists() and fold_0_split.exists():
            print(f"\nRunning k-fold out-of-fold predictions from {model_path} ...")
            with open(fold_0_split) as f:
                split_info = json.load(f)
            tile_to_fold = split_info["tile_to_fold"]
            n_folds = split_info["n_folds"]

            # Assign each rl2 row to its fold via MGRS_TILE
            rl2_tiles = rl2["MGRS_TILE"].values
            rl2_fold = np.array(
                [
                    tile_to_fold.get(str(t), -1) if pd.notna(t) else -1
                    for t in rl2_tiles
                ],
                dtype=int,
            )

            for fold_i in range(n_folds):
                fold_dir = model_path / f"fold_{fold_i}"
                with open(fold_dir / "direct_rf_features.json") as f:
                    feature_names = json.load(f)
                fold_model = joblib.load(fold_dir / "direct_rf_model.joblib")
                fold_imputer = joblib.load(fold_dir / "direct_rf_imputer.joblib")

                test_mask = rl2_fold == fold_i
                if not test_mask.any():
                    continue
                X = rl2.loc[test_mask, feature_names].values.astype(np.float32)
                X = fold_imputer.transform(X)
                pred = fold_model.predict(X)
                model_log10[test_mask] = pred.astype(np.float64)
                print(f"  fold {fold_i}: {int(test_mask.sum())} test rows")

        else:
            # Single model: predict on all rows (includes training data)
            print(f"\nRunning SWAP model from {model_path} ...")
            with open(model_path / "direct_rf_features.json") as f:
                feature_names = json.load(f)
            model = joblib.load(model_path / "direct_rf_model.joblib")
            imputer = joblib.load(model_path / "direct_rf_imputer.joblib")

            X = rl2[feature_names].values.astype(np.float32)
            X = imputer.transform(X)
            pred = model.predict(X)
            model_log10 = pred.astype(np.float64)

        n_model = int(np.isfinite(model_log10).sum())
        print(f"  {n_model}/{len(rl2)} observations with model predictions")

    # Load site-level vG params and join
    vg = pd.read_parquet(args.vg_params)
    print(f"  {len(vg)} sites with vG params")

    obs = rl2[["sample_id", "source", "theta", "log10_suction_cm"]].copy()
    obs["swap_log10_suction"] = model_log10

    df = obs.merge(
        vg[
            [
                "sample_id",
                "source",
                "ros_theta_r",
                "ros_theta_s",
                "ros_alpha",
                "ros_n",
                "pol_theta_r",
                "pol_theta_s",
                "pol_alpha",
                "pol_n",
            ]
        ],
        on=["sample_id", "source"],
        how="left",
    )

    observed = df["log10_suction_cm"].values
    model_log10 = df["swap_log10_suction"].values

    # Rosetta suction
    ros_h = psi_from_theta(
        df["theta"].values,
        df["ros_theta_r"].values,
        df["ros_theta_s"].values,
        df["ros_alpha"].values,
        df["ros_n"].values,
    )
    ros_log10 = np.where(ros_h > 0, np.log10(ros_h), np.nan)

    # POLARIS suction
    pol_h = psi_from_theta(
        df["theta"].values,
        df["pol_theta_r"].values,
        df["pol_theta_s"].values,
        df["pol_alpha"].values,
        df["pol_n"].values,
    )
    pol_log10 = np.where(pol_h > 0, np.log10(pol_h), np.nan)

    # Compute metrics
    pairs = {
        "rosetta_vs_observed": (ros_log10, observed),
        "polaris_vs_observed": (pol_log10, observed),
        "rosetta_vs_polaris": (ros_log10, pol_log10),
    }
    if args.model_dir:
        pairs["swap_vs_observed"] = (model_log10, observed)
        pairs["swap_vs_rosetta"] = (model_log10, ros_log10)
        pairs["swap_vs_polaris"] = (model_log10, pol_log10)

    print("\n--- Metrics (log10 suction cm) ---")
    rows = []
    for label, (pred, ref) in pairs.items():
        m = _metrics(pred, ref)
        rows.append({"comparison": label, **m})
        print(
            f"  {label}: n={m['n']:,}  RMSE={m['rmse']:.4f}  "
            f"bias={m['bias']:.4f}  R2={m['r2']:.4f}"
        )

    # Per-source breakdown
    print("\n--- By source ---")
    for src in sorted(df["source"].unique()):
        mask = df["source"].values == src
        for label, (pred, ref) in pairs.items():
            m = _metrics(pred[mask], ref[mask])
            tag = f"{label}_{src}"
            rows.append({"comparison": tag, **m})
            print(
                f"  {tag}: n={m['n']:,}  RMSE={m['rmse']:.4f}  "
                f"bias={m['bias']:.4f}  R2={m['r2']:.4f}"
            )

    # Write metrics CSV
    csv_path = output_dir / "ptf_comparison_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["comparison", "n", "rmse", "bias", "r2"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path}")

    # Write merged observation table for downstream analysis
    df["ros_log10_suction"] = ros_log10
    df["pol_log10_suction"] = pol_log10
    obs_path = output_dir / "ptf_comparison_observations.parquet"
    df.to_parquet(str(obs_path), index=False)
    print(f"Wrote {obs_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compare our SWP model against Rosetta/POLARIS PTF baselines",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- prep ---
    p_prep = sub.add_parser(
        "prep",
        help="Extract vG params at training sites",
    )
    p_prep.add_argument(
        "--training-table",
        default=DEFAULT_TRAINING_TABLE,
        help="Path to training parquet",
    )
    p_prep.add_argument(
        "--rosetta-tif",
        default=DEFAULT_ROSETTA_TIF,
        help="Path to Rosetta L2 VG GeoTIFF (100 m, EPSG:5070)",
    )
    p_prep.add_argument(
        "--output",
        default=os.path.join(DEFAULT_OUTPUT_DIR, "site_vg_params.parquet"),
        help="Output parquet for site-level vG params",
    )
    p_prep.add_argument("--overwrite", action="store_true")
    p_prep.set_defaults(func=run_prep)

    # --- evaluate ---
    p_eval = sub.add_parser(
        "evaluate",
        help="Compare PTF suction against observations",
    )
    p_eval.add_argument(
        "--training-table",
        default=DEFAULT_TRAINING_TABLE,
        help="Path to training parquet",
    )
    p_eval.add_argument(
        "--vg-params",
        default=os.path.join(DEFAULT_OUTPUT_DIR, "site_vg_params.parquet"),
        help="Site-level vG params from prep step",
    )
    p_eval.add_argument(
        "--model-dir",
        default=None,
        help="Path to saved SWAP model directory (adds model predictions to comparison)",
    )
    p_eval.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for metrics and observation table",
    )
    p_eval.set_defaults(func=run_evaluate)

    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
