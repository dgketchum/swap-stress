"""
Feature importance analysis for the direct suction prediction model.

Two complementary analyses:

1. **Permutation importance** — sklearn's permutation_importance on held-out test set.
   Ranks individual features, then aggregates by group via classify_feature().

2. **Group-level ablation** — retrain RF with each feature group excluded, measure R² drop.
   Key comparisons include landsat_bands vs landsat_indices vs full landsat.

Usage:
    python -m map.learning.decision_tree.feature_importance \\
        --obs-table ~/data/.../obs_level_training_emb_250m.parquet \\
        --output-dir ~/data/.../feature_importance
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

from map.data.features import (
    aggregate_importance_by_group,
    classify_feature,
    filter_feature_groups,
    get_feature_columns,
)
from map.learning.decision_tree.train_direct import (
    apply_site_split,
    build_preprocessor,
    compute_metrics,
    create_site_split,
    filter_complete_samples,
)


def run_permutation_importance(
    model: RandomForestRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute permutation importance on held-out test set.

    Parameters
    ----------
    model : RandomForestRegressor
        Trained model.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        Test targets.
    feature_names : list of str
        Feature names corresponding to X_test columns.
    n_repeats : int
        Number of permutation repeats.
    random_state : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: feature, importance_mean, importance_std, group.
        Sorted by importance_mean descending.
    """
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    rows = []
    for i, name in enumerate(feature_names):
        rows.append(
            {
                "feature": name,
                "importance_mean": float(result.importances_mean[i]),
                "importance_std": float(result.importances_std[i]),
                "group": classify_feature(name),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return df


def run_group_ablation(
    obs_table_path: str,
    base_feature_cols: List[str],
    train_sites,
    test_sites,
    groups_to_ablate: Optional[List[str]] = None,
    n_estimators: int = 250,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Retrain model with each feature group excluded, measure R² drop.

    Parameters
    ----------
    obs_table_path : str
        Path to observation-level training data.
    base_feature_cols : list of str
        Full feature column list (before any exclusion).
    train_sites : set
        Training site IDs.
    test_sites : set
        Test site IDs.
    groups_to_ablate : list of str, optional
        Groups to test. If None, uses all groups from FEATURE_GROUPS
        plus 'landsat_bands' and 'landsat_indices'.
    n_estimators : int
        Number of RF trees.
    random_state : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: excluded_group, r2, r2_drop, n_features_removed.
        Sorted by r2_drop descending (most impactful group first).
    """
    if groups_to_ablate is None:
        groups_to_ablate = [
            "landsat",
            "landsat_bands",
            "landsat_indices",
            "sentinel1",
            "smap",
            "gridmet",
            "soilgrids",
            "fao",
            "polaris",
            "terrain",
            "coords",
            "embeddings",
        ]

    # Load data
    df = pd.read_parquet(obs_table_path)
    if df.index.name:
        df = df.reset_index()

    train_df, test_df = apply_site_split(df, train_sites, test_sites, "sample_id")
    train_df = train_df.dropna(subset=["theta", "log10_suction_cm"])
    test_df = test_df.dropna(subset=["theta", "log10_suction_cm"])

    y_test = test_df["log10_suction_cm"].values

    # Baseline model (all features)
    all_features = base_feature_cols + ["theta"]
    imputer = build_preprocessor(add_indicator=False)
    X_train = imputer.fit_transform(train_df[all_features].values)
    X_test = imputer.transform(test_df[all_features].values)

    baseline_model = RandomForestRegressor(
        n_estimators=n_estimators,
        n_jobs=-1,
        random_state=random_state,
    )
    baseline_model.fit(X_train, train_df["log10_suction_cm"].values)
    baseline_r2 = float(r2_score(y_test, baseline_model.predict(X_test)))
    print(f"Baseline R²: {baseline_r2:.4f} ({len(all_features)} features)")

    results = [
        {
            "excluded_group": "none (baseline)",
            "r2": baseline_r2,
            "r2_drop": 0.0,
            "n_features_removed": 0,
        }
    ]

    for group in groups_to_ablate:
        ablated_features = filter_feature_groups(base_feature_cols, [group])
        n_removed = len(base_feature_cols) - len(ablated_features)

        if n_removed == 0:
            print(f"  {group}: no features to remove, skipping")
            continue

        ablated_all = ablated_features + ["theta"]

        abl_imputer = build_preprocessor(add_indicator=False)
        X_train_abl = abl_imputer.fit_transform(train_df[ablated_all].values)
        X_test_abl = abl_imputer.transform(test_df[ablated_all].values)

        abl_model = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=-1,
            random_state=random_state,
        )
        abl_model.fit(X_train_abl, train_df["log10_suction_cm"].values)
        abl_r2 = float(r2_score(y_test, abl_model.predict(X_test_abl)))
        r2_drop = baseline_r2 - abl_r2

        print(
            f"  -{group}: R²={abl_r2:.4f} (drop={r2_drop:+.4f}, "
            f"removed {n_removed} features)"
        )

        results.append(
            {
                "excluded_group": group,
                "r2": abl_r2,
                "r2_drop": r2_drop,
                "n_features_removed": n_removed,
            }
        )

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("r2_drop", ascending=False).reset_index(drop=True)
    return result_df


def run_analysis(
    obs_table_path: str,
    output_dir: str,
    n_estimators: int = 250,
    n_repeats: int = 10,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict:
    """
    Run full feature importance analysis.

    Parameters
    ----------
    obs_table_path : str
        Path to observation-level training parquet.
    output_dir : str
        Directory for output files.
    n_estimators : int
        Number of RF trees.
    n_repeats : int
        Number of permutation repeats.
    test_size : float
        Fraction of sites for testing.
    random_state : int
        Random seed.

    Returns
    -------
    dict
        Combined results.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    df = pd.read_parquet(obs_table_path)
    if df.index.name:
        df = df.reset_index()
    df = filter_complete_samples(df)

    # Get features
    feature_cols = get_feature_columns(df, include_depth=True, include_embeddings=False)
    all_features = feature_cols + ["theta"]
    print(f"  {len(feature_cols)} EE features + theta")

    # Site-level split
    train_sites, test_sites = create_site_split(
        df, "sample_id", test_size, random_state
    )
    train_df, test_df = apply_site_split(df, train_sites, test_sites, "sample_id")
    train_df = train_df.dropna(subset=["theta", "log10_suction_cm"])
    test_df = test_df.dropna(subset=["theta", "log10_suction_cm"])

    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")

    # Train baseline model
    print("\nTraining baseline model...")
    imputer = build_preprocessor(add_indicator=False)
    X_train = imputer.fit_transform(train_df[all_features].values)
    X_test = imputer.transform(test_df[all_features].values)
    y_train = train_df["log10_suction_cm"].values
    y_test = test_df["log10_suction_cm"].values

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    baseline_metrics = compute_metrics(y_test, model.predict(X_test))
    print(f"  Baseline R²={baseline_metrics['r2']:.4f}")

    # 1. Permutation importance
    print("\n1. Permutation importance...")
    perm_df = run_permutation_importance(
        model,
        X_test,
        y_test,
        all_features,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    perm_path = os.path.join(output_dir, "permutation_importance.csv")
    perm_df.to_csv(perm_path, index=False)
    print(f"  Saved to {perm_path}")

    # Top features
    print("\n  Top 20 features:")
    for _, row in perm_df.head(20).iterrows():
        print(
            f"    {row['feature']:<40s} {row['importance_mean']:.4f} ({row['group']})"
        )

    # Aggregate by group
    perm_importance_dict = dict(zip(perm_df["feature"], perm_df["importance_mean"]))
    group_importance = aggregate_importance_by_group(
        perm_importance_dict, normalize=True
    )

    print("\n  Group importance (permutation):")
    for group, imp in group_importance.items():
        print(f"    {group:<25s} {imp:.4f}")

    group_imp_path = os.path.join(output_dir, "group_importance_permutation.json")
    with open(group_imp_path, "w") as f:
        json.dump(group_importance, f, indent=2)

    # 2. Group ablation
    print("\n2. Group ablation...")
    ablation_df = run_group_ablation(
        obs_table_path,
        feature_cols,
        train_sites,
        test_sites,
        n_estimators=n_estimators,
        random_state=random_state,
    )
    ablation_path = os.path.join(output_dir, "group_ablation.csv")
    ablation_df.to_csv(ablation_path, index=False)
    print(f"  Saved to {ablation_path}")

    # MDI importance (from trained model)
    mdi_dict = {
        all_features[i]: float(model.feature_importances_[i])
        for i in np.argsort(model.feature_importances_)[::-1]
    }
    group_mdi = aggregate_importance_by_group(mdi_dict, normalize=True)

    mdi_path = os.path.join(output_dir, "group_importance_mdi.json")
    with open(mdi_path, "w") as f:
        json.dump(group_mdi, f, indent=2)

    # Combined results
    results = {
        "baseline_metrics": baseline_metrics,
        "group_importance_permutation": group_importance,
        "group_importance_mdi": group_mdi,
        "config": {
            "obs_table": obs_table_path,
            "n_estimators": n_estimators,
            "n_repeats": n_repeats,
            "test_size": test_size,
            "random_state": random_state,
            "n_features": len(all_features),
        },
    }

    results_path = os.path.join(output_dir, "feature_importance_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to {results_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature importance analysis for direct suction model.",
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
        "--n-estimators",
        type=int,
        default=250,
        help="Number of RF trees (default: 250).",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=10,
        help="Number of permutation repeats (default: 10).",
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

    run_analysis(
        obs_table_path=args.obs_table,
        output_dir=args.output_dir,
        n_estimators=args.n_estimators,
        n_repeats=args.n_repeats,
        test_size=args.test_size,
        random_state=args.random_state,
    )
