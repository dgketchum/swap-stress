"""
Train Random Forest to predict log10(suction_cm) directly from EE features + theta.

This is the direct model approach, where the RF learns the mapping:
    (EE features, theta) -> log10(suction_cm)

Evaluation uses site-level holdout to prevent data leakage from spatially
correlated observations.

Usage:
    python -m swapstress.model.train \\
        --obs-table /nas/soils/swapstress/training/obs_level_training_9km_global.parquet \\
        --output-dir /nas/soils/swapstress/models/direct_rf_9km_global_pruned

    # Exclude feature groups
    python -m swapstress.model.train \\
        --obs-table ... --output-dir ... \\
        --exclude-groups embeddings polaris smap
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor

try:
    from quantile_forest import RandomForestQuantileRegressor
except ImportError:
    RandomForestQuantileRegressor = None

from swapstress.model.data import (
    prepare_direct_data,
)
from swapstress.model.preprocessing import prepare_rf_arrays
from swapstress.model.reporting import evaluate_and_report


def train_and_evaluate(
    obs_table_path: str,
    output_dir: str,
    exclude_groups: Optional[List[str]] = None,
    n_estimators: int = 250,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    drop_blocking_features: bool = True,
    resolution_m: float = 250,
    split_manifest: Optional[str] = None,
    config_dict: Optional[Dict] = None,
    holdout_col: Optional[str] = None,
    n_folds: int = 5,
    test_fold: int = 0,
    n_jobs: int = -1,
    quantile: bool = False,
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
    val_size : float
        Fraction of non-test sites for validation.  RF does not use a
        validation set itself, but the split is recorded in the manifest
        so NN trainers reusing the same manifest get an identical holdout.
    random_state : int
        Random seed.
    drop_blocking_features : bool
        If True, remove features 100% missing for any source.
    holdout_col : str or None
        Column for spatial holdout (e.g. 'MGRS_TILE').  None = legacy.
    n_folds : int
        Number of folds for hash-based splitting.
    test_fold : int
        Which fold to hold out as test.

    Returns
    -------
    dict
        Results including metrics, config, feature list.
    """
    import pandas as pd

    from swapstress.model.data import write_split_manifest

    # Shared data loading and spatial split — always request val split
    # so the manifest is consumable by both RF and NN.
    data = prepare_direct_data(
        obs_table_path=obs_table_path,
        output_dir=output_dir,
        exclude_groups=exclude_groups,
        drop_blocking_features=drop_blocking_features,
        resolution_m=resolution_m,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        split_manifest=split_manifest,
        holdout_col=holdout_col,
        n_folds=n_folds,
        test_fold=test_fold,
    )

    df = data["df"]
    all_features = data["all_features"]
    test_df = data["test_df"]
    test_sites = data["test_sites"]

    # RF uses all non-test data (train + val) for fitting.
    # If a three-way manifest was loaded, merge train_df + val_df.
    train_df = data["train_df"]
    train_sites = set(data["train_sites"])
    if data.get("val_df") is not None:
        train_df = pd.concat([train_df, data["val_df"]], ignore_index=True)
        train_sites = train_sites | data["val_sites"]

    # Write three-way split manifest so NN trainers can reuse the holdout
    # (kfold manifests are written by prepare_direct_data when holdout_col is set)
    if holdout_col is None:
        manifest_path = split_manifest or os.path.join(output_dir, "spatial_split.json")
        if not os.path.exists(manifest_path):
            write_split_manifest(
                manifest_path,
                train_groups=data["train_sites"],
                test_groups=data["test_sites"],
                val_groups=data.get("val_sites"),
                random_state=random_state,
                resolution_m=resolution_m,
            )

    # Impute and build arrays
    X_train, X_test, y_train, y_test, imputer = prepare_rf_arrays(
        train_df,
        test_df,
        all_features,
    )

    # Train
    if quantile:
        if RandomForestQuantileRegressor is None:
            raise ImportError("quantile-forest is required: uv add quantile-forest")
        print(f"Training QRF with {n_estimators} trees (n_jobs={n_jobs})...")
        model = RandomForestQuantileRegressor(
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    else:
        print(f"Training RF with {n_estimators} trees (n_jobs={n_jobs})...")
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    model.fit(X_train, y_train)

    # Predict (QRF .predict() returns the mean, same as standard RF)
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
        holdout_col=holdout_col,
        extra_config={
            "obs_table": obs_table_path,
            "exclude_groups": exclude_groups,
            "n_estimators": n_estimators,
            "test_size": test_size,
            "random_state": random_state,
            "drop_blocking_features": drop_blocking_features,
            "resolution_m": resolution_m,
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
        from swapstress.config import input_checksum, write_provenance

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


def build_parser():
    from swapstress.cli import add_common_args

    parser = argparse.ArgumentParser(
        prog="swapstress-train",
        description="Stage 03: train the direct model, "
        "EE features + theta -> log10(suction_cm).",
    )
    add_common_args(parser)
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
    parser.add_argument(
        "--val-size",
        type=float,
        default=None,
        help="Fraction of non-test groups for validation (default: 0.2).",
    )
    parser.add_argument(
        "--split-manifest",
        type=str,
        default=None,
        help="Path to existing spatial_split.json (reuse holdout from prior run).",
    )
    parser.add_argument(
        "--holdout-col",
        type=str,
        default=None,
        help="Column for spatial holdout (e.g. 'MGRS_TILE'). Default: legacy coords.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=None,
        help="Number of folds for hash-based splitting (default: 5).",
    )
    parser.add_argument(
        "--test-fold",
        type=int,
        default=None,
        help="Which fold to hold out as test (default: 0).",
    )
    parser.add_argument(
        "--kfold",
        action="store_true",
        default=None,
        help="Run full k-fold cross-validation instead of single holdout.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel jobs for RF (-1 = all cores, default: -1).",
    )
    parser.add_argument(
        "--quantile",
        action="store_true",
        default=None,
        help="Use RandomForestQuantileRegressor (enables quantile prediction at inference).",
    )
    return parser


def main(argv=None):
    from swapstress.cli import report_paths, resolve
    from swapstress.config import feature_groups_to_exclude

    config = resolve(build_parser(), argv, required=["obs_table", "output_dir"])

    if config["dry_run"]:
        report_paths(
            "03 train",
            {"training table": config["obs_table"]},
            {"model dir": config["output_dir"]},
        )
        return

    # Convert positive feature_groups to exclude_groups
    exclude_groups = config.get("exclude_groups")
    if config.get("feature_groups") is not None:
        exclude_groups = feature_groups_to_exclude(config["feature_groups"])

    holdout_col = config.get("holdout_col")
    n_folds = config.get("n_folds", 5)
    n_jobs = config.get("n_jobs", -1)
    do_kfold = config.get("kfold", False)

    if do_kfold:
        from swapstress.model.crossval import run_kfold_cv

        run_kfold_cv(
            obs_table_path=config["obs_table"],
            output_dir=config["output_dir"],
            n_folds=n_folds,
            holdout_col=holdout_col or "MGRS_TILE",
            trainer_fn=train_and_evaluate,
            trainer_kwargs={
                "exclude_groups": exclude_groups,
                "n_estimators": config.get("n_estimators", 250),
                "test_size": config.get("test_size", 0.2),
                "val_size": config.get("val_size", 0.2),
                "random_state": config.get("random_state", 42),
                "drop_blocking_features": True,
                "resolution_m": config.get("resolution_m", 250),
                "n_jobs": n_jobs,
                "config_dict": config,
            },
        )
    else:
        train_and_evaluate(
            obs_table_path=config["obs_table"],
            output_dir=config["output_dir"],
            exclude_groups=exclude_groups,
            n_estimators=config.get("n_estimators", 250),
            test_size=config.get("test_size", 0.2),
            val_size=config.get("val_size", 0.2),
            random_state=config.get("random_state", 42),
            resolution_m=config.get("resolution_m", 250),
            split_manifest=config.get("split_manifest"),
            config_dict=config,
            holdout_col=holdout_col,
            n_folds=n_folds,
            test_fold=config.get("test_fold", 0),
            n_jobs=n_jobs,
            quantile=config.get("quantile", False),
        )


if __name__ == "__main__":
    main()
