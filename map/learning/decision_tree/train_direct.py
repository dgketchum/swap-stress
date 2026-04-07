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
from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from map.learning.direct.data import (
    prepare_direct_data,
)
from map.learning.direct.preprocessing import prepare_rf_arrays
from map.learning.direct.reporting import evaluate_and_report


def train_and_evaluate(
    obs_table_path: str,
    output_dir: str,
    exclude_groups: Optional[List[str]] = None,
    n_estimators: int = 250,
    test_size: float = 0.2,
    random_state: int = 42,
    drop_blocking_features: bool = True,
    resolution_m: float = 250,
    config_dict: Optional[Dict] = None,
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
    # Shared data loading and spatial split
    data = prepare_direct_data(
        obs_table_path=obs_table_path,
        output_dir=output_dir,
        exclude_groups=exclude_groups,
        drop_blocking_features=drop_blocking_features,
        resolution_m=resolution_m,
        test_size=test_size,
        random_state=random_state,
    )

    df = data["df"]
    all_features = data["all_features"]
    train_df = data["train_df"]
    test_df = data["test_df"]
    train_sites = data["train_sites"]
    test_sites = data["test_sites"]

    # Impute and build arrays
    X_train, X_test, y_train, y_test, imputer = prepare_rf_arrays(
        train_df,
        test_df,
        all_features,
    )

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

    # Feature importance (MDI)
    importances = model.feature_importances_
    importance_dict = {
        all_features[i]: float(importances[i]) for i in np.argsort(importances)[::-1]
    }

    # Standard evaluation and reporting
    results = evaluate_and_report(
        y_test=y_test,
        y_pred=y_pred,
        test_df=test_df,
        output_dir=output_dir,
        all_features=all_features,
        train_df=train_df,
        train_sites=train_sites,
        test_sites=test_sites,
        resolution_m=resolution_m,
        feature_importance=importance_dict,
        extra_config={
            "obs_table": obs_table_path,
            "exclude_groups": exclude_groups,
            "n_estimators": n_estimators,
            "test_size": test_size,
            "random_state": random_state,
            "drop_blocking_features": drop_blocking_features,
        },
        model_family="rf",
    )

    # Save RF-specific artifacts
    import joblib

    model_path = os.path.join(output_dir, "direct_rf_model.joblib")
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")

    imputer_path = os.path.join(output_dir, "direct_rf_imputer.joblib")
    joblib.dump(imputer, imputer_path)

    features_path = os.path.join(output_dir, "direct_rf_features.json")
    with open(features_path, "w") as f:
        json.dump(all_features, f, indent=2)

    # Write provenance artifact
    if config_dict is not None:
        from map.config import input_checksum, write_provenance

        prov_path = write_provenance(
            output_dir=output_dir,
            config=config_dict,
            run_type="train",
            extras={
                "inputs": {
                    "obs_table_sha256": input_checksum(obs_table_path),
                    "obs_table_n_rows": len(df),
                    "obs_table_n_cols": df.shape[1],
                },
                "outputs": {
                    "n_features": len(all_features),
                    "n_train": len(train_df),
                    "n_test": len(test_df),
                    "n_train_sites": len(train_sites),
                    "n_test_sites": len(test_sites),
                },
                "upstream": None,
            },
        )
        print(f"Saved provenance to {prov_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train direct RF model: EE features + theta -> log10(suction_cm)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to TOML run config.",
    )
    parser.add_argument(
        "--obs-table",
        type=str,
        default=None,
        help="Path to observation-level training parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
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
        default=None,
        help="Number of RF trees (default: 250).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=None,
        help="Fraction of sites for testing (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--resolution-m",
        type=float,
        default=None,
        help="Spatial grouping grid cell size in metres (default: 250).",
    )
    args = parser.parse_args()

    from map.config import feature_groups_to_exclude, load_config

    config = load_config(args.config, vars(args))

    if not config.get("obs_table"):
        parser.error("--obs-table is required (via CLI or TOML config)")
    if not config.get("output_dir"):
        parser.error("--output-dir is required (via CLI or TOML config)")

    # Convert positive feature_groups to exclude_groups
    exclude_groups = config.get("exclude_groups")
    if config.get("feature_groups") is not None:
        exclude_groups = feature_groups_to_exclude(config["feature_groups"])

    train_and_evaluate(
        obs_table_path=config["obs_table"],
        output_dir=config["output_dir"],
        exclude_groups=exclude_groups,
        n_estimators=config.get("n_estimators", 250),
        test_size=config.get("test_size", 0.2),
        random_state=config.get("random_state", 42),
        resolution_m=config.get("resolution_m", 250),
        config_dict=config,
    )
