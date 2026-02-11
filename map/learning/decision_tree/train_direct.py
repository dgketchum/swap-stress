"""
Train Random Forest to predict log10(suction_cm) directly from EE features + theta.

This is the direct model approach, where the RF learns the mapping:
    (EE features, theta) -> log10(suction_cm)

Evaluation uses site-level holdout to prevent data leakage from spatially
correlated observations.

Usage:
    python -m map.learning.decision_tree.train_direct \\
        --obs-table ~/data/.../obs_level_training_emb_250m.parquet \\
        --output-dir ~/data/.../direct_model_results

    # Exclude feature groups
    python -m map.learning.decision_tree.train_direct \\
        --obs-table ... --output-dir ... \\
        --exclude-groups embeddings polaris smap
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from map.data.features import (
    filter_feature_groups,
    get_feature_columns,
)


def assign_spatial_group(df: pd.DataFrame, resolution_m: float = 250) -> pd.Series:
    """Quantize lat/lon to grid cells for spatial grouping.

    Groups observations that share the same EE pixel (~250 m) so that
    all co-located profiles end up in the same train/test partition.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'lat' and 'lon' columns.
    resolution_m : float
        Grid cell size in metres (default 250 m, matching EE extraction).

    Returns
    -------
    pd.Series
        String labels like ``"45.12300_-112.45600"`` (NaN where coords missing).
    """
    step = resolution_m / 111_320  # degrees per metre at equator
    lat_q = (df["lat"] / step).round() * step
    lon_q = (df["lon"] / step).round() * step
    groups = lat_q.round(5).astype(str) + "_" + lon_q.round(5).astype(str)
    groups[df["lat"].isna() | df["lon"].isna()] = np.nan
    return groups


def create_site_split(
    df: pd.DataFrame,
    group_col: str = "sample_id",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Set[str], Set[str]]:
    """Create train/test split on spatial groups (quantized lat/lon).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (must contain 'lat' and 'lon').
    group_col : str
        Unused, kept for backward-compatible call signatures.
    test_size : float
        Fraction of spatial groups for testing.
    random_state : int
        Random seed.

    Returns
    -------
    tuple of (set, set)
        (train_groups, test_groups)
    """
    groups = assign_spatial_group(df)
    unique_groups = list(groups.dropna().unique())
    train_groups, test_groups = train_test_split(
        unique_groups,
        test_size=test_size,
        random_state=random_state,
    )
    return set(train_groups), set(test_groups)


def apply_site_split(
    df: pd.DataFrame,
    train_sites: Set[str],
    test_sites: Set[str],
    group_col: str = "sample_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply pre-computed spatial-group split to a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (must contain 'lat' and 'lon').
    train_sites : set
        Spatial group labels for training.
    test_sites : set
        Spatial group labels for testing.
    group_col : str
        Unused, kept for backward-compatible call signatures.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (train_df, test_df)
    """
    groups = assign_spatial_group(df)
    train_mask = groups.isin(train_sites)
    test_mask = groups.isin(test_sites)
    return df[train_mask.values].copy(), df[test_mask.values].copy()


def filter_complete_samples(
    df: pd.DataFrame,
    required_feature_count: int = 200,
) -> pd.DataFrame:
    """Filter to samples with lat/lon (indicator of EE feature availability)."""
    initial_count = len(df)
    if "lat" in df.columns and "lon" in df.columns:
        has_coords = df["lat"].notna() & df["lon"].notna()
        df = df[has_coords].copy()
        print(
            f"  Filtered to samples with lat/lon: {len(df)}/{initial_count} "
            f"({len(df) / initial_count * 100:.1f}%)"
        )
    return df


def audit_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Audit dataset for feature missingness by source.

    Identifies blocking features (100% missing for any source).

    Returns
    -------
    dict
        Audit results with counts_by_source, blocking_features.
    """
    if "source" not in df.columns:
        return {"counts_by_source": {}, "blocking_features": []}

    sources = df["source"].unique()
    counts_by_source = df["source"].value_counts().to_dict()

    blocking_features = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        for source in sources:
            source_df = df[df["source"] == source]
            if len(source_df) > 0 and source_df[col].isna().mean() >= 0.9999:
                blocking_features.append(col)
                break

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        counts_path = os.path.join(output_dir, "dataset_counts.json")
        with open(counts_path, "w") as f:
            json.dump(
                {
                    "counts_by_source": counts_by_source,
                    "blocking_features": blocking_features,
                    "total_rows": len(df),
                },
                f,
                indent=2,
            )

    return {
        "counts_by_source": counts_by_source,
        "blocking_features": blocking_features,
    }


def filter_blocking_features(
    feature_cols: List[str],
    blocking_features: List[str],
) -> List[str]:
    """Remove features that are 100% missing for any included source."""
    blocking_set = set(blocking_features)
    filtered = [f for f in feature_cols if f not in blocking_set]
    n_removed = len(feature_cols) - len(filtered)
    if n_removed > 0:
        print(f"  Removed {n_removed} blocking features")
    return filtered


def build_preprocessor(add_indicator: bool = True) -> SimpleImputer:
    """Create median imputer for handling NaN values in features."""
    return SimpleImputer(strategy="median", add_indicator=add_indicator)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    linear: bool = False,
) -> Dict[str, float]:
    """
    Compute prediction metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True values (log10 scale).
    y_pred : np.ndarray
        Predicted values (log10 scale).
    linear : bool
        If True, transform to linear scale (cm H2O) before computing.

    Returns
    -------
    dict
        Dictionary with rmse, mae, r2, bias, n.
    """
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isinf(y_pred)
    if valid.sum() == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "bias": np.nan, "n": 0}

    yt, yp = y_true[valid], y_pred[valid]
    if linear:
        yt, yp = 10**yt, 10**yp

    return {
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "r2": float(r2_score(yt, yp)),
        "bias": float(np.mean(yp - yt)),
        "n": int(valid.sum()),
    }


def compute_metrics_by_source(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sources: np.ndarray,
    linear: bool = False,
) -> pd.DataFrame:
    """Compute metrics stratified by data source."""
    results = []
    for source in np.unique(sources[~pd.isna(sources)]):
        mask = sources == source
        if mask.sum() < 10:
            continue
        metrics = compute_metrics(y_true[mask], y_pred[mask], linear=linear)
        metrics["source"] = source
        results.append(metrics)
    return pd.DataFrame(results)


def compute_metrics_by_site(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    site_ids: np.ndarray,
    linear: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute per-site metrics and site-weighted aggregates."""
    results = []
    for site in np.unique(site_ids[~pd.isna(site_ids)]):
        mask = site_ids == site
        if mask.sum() < 3:
            continue
        metrics = compute_metrics(y_true[mask], y_pred[mask], linear=linear)
        metrics["site_id"] = site
        results.append(metrics)

    per_site_df = pd.DataFrame(results)
    if len(per_site_df) > 0:
        summary = {
            "mean_rmse": per_site_df["rmse"].mean(),
            "median_rmse": per_site_df["rmse"].median(),
            "mean_r2": per_site_df["r2"].mean(),
            "median_r2": per_site_df["r2"].median(),
            "mean_mae": per_site_df["mae"].mean(),
            "median_mae": per_site_df["mae"].median(),
            "n_sites": len(per_site_df),
        }
    else:
        summary = {
            k: np.nan
            for k in [
                "mean_rmse",
                "median_rmse",
                "mean_r2",
                "median_r2",
                "mean_mae",
                "median_mae",
            ]
        }
        summary["n_sites"] = 0

    return per_site_df, summary


def train_and_evaluate(
    obs_table_path: str,
    output_dir: str,
    exclude_groups: Optional[List[str]] = None,
    n_estimators: int = 250,
    test_size: float = 0.2,
    random_state: int = 42,
    drop_blocking_features: bool = True,
) -> Dict:
    """
    Train direct RF model and evaluate with site-level holdout.

    Parameters
    ----------
    obs_table_path : str
        Path to observation-level training parquet.
    output_dir : str
        Directory for output files.
    exclude_groups : list of str, optional
        Feature groups to exclude.
    n_estimators : int
        Number of trees.
    test_size : float
        Fraction of sites for testing.
    random_state : int
        Random seed.
    drop_blocking_features : bool
        If True, remove features 100% missing for any source.

    Returns
    -------
    dict
        Results including metrics, config, feature list.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading observation table...")
    df = pd.read_parquet(obs_table_path)
    if df.index.name:
        df = df.reset_index()
    print(f"  {len(df)} observations, {df.shape[1]} columns")

    # Filter to complete samples
    df = filter_complete_samples(df)

    # Get feature columns
    feature_cols = get_feature_columns(df, include_depth=True, include_embeddings=False)

    # Audit and remove blocking features
    if drop_blocking_features:
        audit = audit_dataset(df, feature_cols, output_dir)
        if audit["blocking_features"]:
            feature_cols = filter_blocking_features(
                feature_cols, audit["blocking_features"]
            )

    # Apply group exclusions
    if exclude_groups:
        feature_cols = filter_feature_groups(feature_cols, exclude_groups)
        print(f"  After excluding {exclude_groups}: {len(feature_cols)} features")

    print(f"  Using {len(feature_cols)} features + theta")

    # Site-level split
    print("Creating spatial-group split...")
    train_sites, test_sites = create_site_split(
        df, "sample_id", test_size, random_state
    )
    train_df, test_df = apply_site_split(df, train_sites, test_sites, "sample_id")

    # Clean: require theta and target
    train_df = train_df.dropna(subset=["theta", "log10_suction_cm"])
    test_df = test_df.dropna(subset=["theta", "log10_suction_cm"])
    print(f"  Train: {len(train_df)} obs from {len(train_sites)} spatial groups")
    print(f"  Test:  {len(test_df)} obs from {len(test_sites)} spatial groups")

    if "source" in train_df.columns:
        print("  Train by source:", train_df["source"].value_counts().to_dict())
        print("  Test by source:", test_df["source"].value_counts().to_dict())

    # Build feature matrix (features + theta)
    all_features = feature_cols + ["theta"]

    # Impute missing features
    imputer = build_preprocessor(add_indicator=False)
    X_train = imputer.fit_transform(train_df[all_features].values)
    X_test = imputer.transform(test_df[all_features].values)

    y_train = train_df["log10_suction_cm"].values
    y_test = test_df["log10_suction_cm"].values

    # Train
    print(f"Training RF with {n_estimators} trees...")
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Overall metrics
    metrics = compute_metrics(y_test, y_pred)
    print(
        f"\nOverall: R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, "
        f"MAE={metrics['mae']:.4f}"
    )

    # Source-level metrics
    source_metrics = None
    if "source" in test_df.columns:
        source_metrics = compute_metrics_by_source(
            y_test,
            y_pred,
            test_df["source"].values,
        )
        print("\nBy source:")
        for _, row in source_metrics.iterrows():
            print(
                f"  {row['source']}: R2={row['r2']:.4f}, "
                f"RMSE={row['rmse']:.4f} (n={row['n']:.0f})"
            )

    # Site-level metrics (grouped by spatial cell)
    spatial_groups = assign_spatial_group(test_df).values
    site_metrics, site_summary = compute_metrics_by_site(y_test, y_pred, spatial_groups)
    print(
        f"\nSite-weighted: mean R2={site_summary['mean_r2']:.4f}, "
        f"median R2={site_summary['median_r2']:.4f} "
        f"({site_summary['n_sites']} spatial groups)"
    )

    # Feature importance (MDI)
    importances = model.feature_importances_
    importance_dict = {
        all_features[i]: float(importances[i]) for i in np.argsort(importances)[::-1]
    }

    # Save results
    results = {
        "overall_metrics": metrics,
        "site_weighted_metrics": site_summary,
        "source_metrics": source_metrics.to_dict("records")
        if source_metrics is not None
        else [],
        "feature_importance": importance_dict,
        "config": {
            "obs_table": obs_table_path,
            "exclude_groups": exclude_groups,
            "n_estimators": n_estimators,
            "test_size": test_size,
            "random_state": random_state,
            "n_features": len(all_features),
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_train_sites": len(train_sites),
            "n_test_sites": len(test_sites),
            "drop_blocking_features": drop_blocking_features,
        },
    }

    results_path = os.path.join(output_dir, "direct_model_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to {results_path}")

    # Save model
    import joblib

    model_path = os.path.join(output_dir, "direct_rf_model.joblib")
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")

    # Save feature list
    features_path = os.path.join(output_dir, "direct_rf_features.json")
    with open(features_path, "w") as f:
        json.dump(all_features, f, indent=2)

    # Save site-level metrics
    if len(site_metrics) > 0:
        site_metrics.to_csv(
            os.path.join(output_dir, "metrics_by_site.csv"), index=False
        )

    if source_metrics is not None:
        source_metrics.to_csv(
            os.path.join(output_dir, "metrics_by_source.csv"), index=False
        )

    # Save predictions and scatter plot
    pred_df = pd.DataFrame({"observed": y_test, "predicted": y_pred})
    if "source" in test_df.columns:
        pred_df["source"] = test_df["source"].values
    pred_df.to_parquet(os.path.join(output_dir, "predictions.parquet"), index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    if "source" in pred_df.columns:
        for src in sorted(pred_df["source"].unique()):
            mask = pred_df["source"] == src
            ax.scatter(
                pred_df.loc[mask, "observed"],
                pred_df.loc[mask, "predicted"],
                s=2,
                alpha=0.3,
                label=src,
            )
        ax.legend(markerscale=4, fontsize=8)
    else:
        ax.scatter(pred_df["observed"], pred_df["predicted"], s=2, alpha=0.3)

    lo = min(pred_df["observed"].min(), pred_df["predicted"].min())
    hi = max(pred_df["observed"].max(), pred_df["predicted"].max())
    ax.plot([lo, hi], [lo, hi], "k-", lw=0.8)
    ax.set_xlabel("Observed log$_{10}$(suction) [cm]")
    ax.set_ylabel("Predicted log$_{10}$(suction) [cm]")
    ax.set_title("Direct Model — Spatial-Group Holdout")
    ax.text(
        0.05,
        0.95,
        f"R² = {metrics['r2']:.3f}\n"
        f"RMSE = {metrics['rmse']:.3f}\n"
        f"MAE = {metrics['mae']:.3f}\n"
        f"n = {metrics['n']}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        family="monospace",
    )
    fig.tight_layout()
    scatter_path = os.path.join(output_dir, "scatter_direct.png")
    fig.savefig(scatter_path, dpi=200)
    plt.close(fig)
    print(f"Saved scatter plot to {scatter_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train direct RF model: EE features + theta -> log10(suction_cm)",
    )
    parser.add_argument(
        "--obs-table",
        type=str,
        required=True,
        help="Path to observation-level training parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for output files.",
    )
    parser.add_argument(
        "--exclude-groups",
        type=str,
        nargs="*",
        default=None,
        help="Feature groups to exclude.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=250,
        help="Number of RF trees (default: 250).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of sites for testing (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    args = parser.parse_args()

    train_and_evaluate(
        obs_table_path=args.obs_table,
        output_dir=args.output_dir,
        exclude_groups=args.exclude_groups,
        n_estimators=args.n_estimators,
        test_size=args.test_size,
        random_state=args.random_state,
    )
