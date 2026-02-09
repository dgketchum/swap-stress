"""
Compare VG Parameter vs Direct Observation approaches for soil water potential prediction.

This script evaluates two modeling paradigms:
1. VG Parameter Approach: EE features → VG params → VG equation → ψ
2. Direct Approach: EE features + θ → log10(ψ)

Both are evaluated on the same held-out test sites to ensure fair comparison.

Usage:
    python -m map.learning.decision_tree.compare_approaches
    python -m map.learning.decision_tree.compare_approaches --systematic
"""

import argparse
import os
import json
import re
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

from map.learning.decision_tree.train_decision_trees import (
    filter_feature_groups,
    _EMBEDDING_PATTERNS,
)


@dataclass
class VGExperimentConfig:
    """Configuration for a VG parameter prediction experiment."""

    name: str
    depth_handling: str  # 'none', 'feature', 'continuous', 'both', 'per_level'
    exclude_feature_groups: Optional[List[str]] = None
    log_transform_params: Optional[List[str]] = (
        None  # VG params to predict on log10 scale
    )
    n_estimators: int = 250

    def __post_init__(self):
        valid_depth = {"none", "feature", "continuous", "both", "per_level"}
        if self.depth_handling not in valid_depth:
            raise ValueError(f"depth_handling must be one of {valid_depth}")
        # Validate log_transform_params
        if self.log_transform_params:
            valid_params = {"theta_r", "theta_s", "alpha", "n"}
            invalid = set(self.log_transform_params) - valid_params
            if invalid:
                raise ValueError(
                    f"Invalid log_transform_params: {invalid}. Must be subset of {valid_params}"
                )


@dataclass
class DirectExperimentConfig:
    """Configuration for a direct suction prediction experiment."""

    name: str
    exclude_feature_groups: Optional[List[str]] = None
    n_estimators: int = 250


# Default experiment configurations - comprehensive set for model selection
# All experiments are scored on suction RMSE using the same site-level holdout
DEFAULT_VG_EXPERIMENTS = [
    # Depth handling variants
    VGExperimentConfig(name="vg_no_depth", depth_handling="none"),
    VGExperimentConfig(name="vg_level_feature", depth_handling="feature"),
    VGExperimentConfig(name="vg_depth_continuous", depth_handling="continuous"),
    VGExperimentConfig(name="vg_both_depth", depth_handling="both"),
    VGExperimentConfig(name="vg_per_level", depth_handling="per_level"),
    # Feature ablation variants (all use depth_handling='feature')
    VGExperimentConfig(
        name="vg_no_soilgrids",
        depth_handling="feature",
        exclude_feature_groups=["soilgrids", "fao"],
    ),
    VGExperimentConfig(
        name="vg_no_embeddings",
        depth_handling="feature",
        exclude_feature_groups=["embeddings"],
    ),
    VGExperimentConfig(
        name="vg_no_polaris",
        depth_handling="feature",
        exclude_feature_groups=["polaris"],
    ),
    VGExperimentConfig(
        name="vg_minimal",
        depth_handling="feature",
        exclude_feature_groups=["soilgrids", "fao", "polaris", "embeddings"],
    ),
    # December 2025 PI feature sets (matches filter_base_data_features / filter_soil_features)
    VGExperimentConfig(
        name="vg_base_data",
        depth_handling="feature",
        exclude_feature_groups=["embeddings", "polaris", "smap"],
    ),
    VGExperimentConfig(
        name="vg_base_data_no_soils",
        depth_handling="feature",
        exclude_feature_groups=["embeddings", "polaris", "smap", "soilgrids", "fao"],
    ),
    # Log-transformed VG parameters (alpha spans 5 orders of magnitude)
    VGExperimentConfig(
        name="vg_log_alpha_n",
        depth_handling="feature",
        log_transform_params=["alpha", "n"],
    ),
    VGExperimentConfig(
        name="vg_log_alpha", depth_handling="feature", log_transform_params=["alpha"]
    ),
]

# Default direct model experiment configurations
# These mirror the VG feature ablation experiments for fair comparison
DEFAULT_DIRECT_EXPERIMENTS = [
    DirectExperimentConfig(name="direct_all_features"),
    DirectExperimentConfig(
        name="direct_base_data",
        exclude_feature_groups=["embeddings", "polaris", "smap"],
    ),
    DirectExperimentConfig(
        name="direct_base_data_no_soils",
        exclude_feature_groups=["embeddings", "polaris", "smap", "soilgrids", "fao"],
    ),
]

# VG parameters
VG_PARAMS = ["theta_r", "theta_s", "alpha", "n"]

# Columns to exclude from features
NON_FEATURE_COLS = {
    # Targets
    "theta_r",
    "theta_s",
    "alpha",
    "n",
    "Ks",
    "log10_alpha",
    "log10_n",
    "log10_Ks",
    "theta",
    "suction_cm",
    "log10_suction_cm",
    # Identifiers
    "sample_id",
    "profile_id",
    "station",
    "site_id",
    "obs_id",
    # Source/metadata
    "source",
    "data_flag",
    "SWCC_class",
    "SWCC_classes",
    "data_ct",
    # Network metadata
    "nwsli_id",
    "network",
    "mesowest_i",
    "obs_ct",
    # Spatial identifiers
    "MGRS_TILE",
    "lat",
    "lon",
    "latitude",
    "longitude",
    # Depth (handle separately)
    "rosetta_level",
    "depth_cm",
    "depth",
}


def filter_complete_samples(
    df: pd.DataFrame, required_feature_count: int = 200
) -> pd.DataFrame:
    """
    Filter dataframe to samples with complete EE features (lat/lon present).

    Samples without lat/lon coordinates have no EE features extracted and should
    be excluded from EE-based model training and evaluation.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with potential incomplete samples.
    required_feature_count : int
        Minimum number of non-NaN feature columns required.
        Default 200 is conservative; full EE stack has ~309 features.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with only complete samples.
    """
    initial_count = len(df)

    # Filter to samples with lat/lon (indicator of EE feature availability)
    if "lat" in df.columns and "lon" in df.columns:
        has_coords = df["lat"].notna() & df["lon"].notna()
        df = df[has_coords].copy()
        print(
            f"  Filtered to samples with lat/lon: {len(df)}/{initial_count} "
            f"({len(df) / initial_count * 100:.1f}%)"
        )
    else:
        print("  Warning: lat/lon columns not found, skipping coordinate filter")

    return df


def audit_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Audit dataset for feature missingness by source.

    Reports row counts and feature missingness by source. Identifies "blocking
    features" that are 100% missing for any source.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with 'source' column.
    feature_cols : list
        List of feature column names to audit.
    output_dir : str, optional
        If provided, saves audit results to CSV/JSON files.

    Returns
    -------
    dict
        Audit results including:
        - counts_by_source: row counts per source
        - missingness_by_source: DataFrame of feature missingness rates
        - blocking_features: list of features 100% missing for any source
    """
    if "source" not in df.columns:
        print("  Warning: 'source' column not found, cannot audit by source")
        return {
            "counts_by_source": {},
            "missingness_by_source": pd.DataFrame(),
            "blocking_features": [],
        }

    sources = df["source"].unique()
    counts_by_source = df["source"].value_counts().to_dict()

    # Compute missingness by source
    missingness_data = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        row = {"feature": col}
        for source in sources:
            source_df = df[df["source"] == source]
            missing_rate = source_df[col].isna().mean() if len(source_df) > 0 else 1.0
            row[f"{source}_missing"] = missing_rate
        missingness_data.append(row)

    missingness_df = pd.DataFrame(missingness_data)

    # Identify blocking features (100% missing for any source)
    blocking_features = []
    for _, row in missingness_df.iterrows():
        for source in sources:
            col_name = f"{source}_missing"
            if col_name in row and row[col_name] >= 0.9999:  # Treat 99.99%+ as blocking
                blocking_features.append(row["feature"])
                break

    # Save outputs if directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Save missingness CSV
        missingness_path = os.path.join(output_dir, "feature_missingness_by_source.csv")
        missingness_df.to_csv(missingness_path, index=False)
        print(f"  Saved feature missingness to {missingness_path}")

        # Save counts JSON
        counts_path = os.path.join(output_dir, "dataset_counts.json")
        with open(counts_path, "w") as f:
            json.dump(
                {
                    "counts_by_source": counts_by_source,
                    "blocking_features": blocking_features,
                    "total_rows": len(df),
                    "n_sources": len(sources),
                },
                f,
                indent=2,
            )
        print(f"  Saved dataset counts to {counts_path}")

    return {
        "counts_by_source": counts_by_source,
        "missingness_by_source": missingness_df,
        "blocking_features": blocking_features,
    }


def filter_blocking_features(
    feature_cols: List[str],
    blocking_features: List[str],
) -> List[str]:
    """
    Remove features that are 100% missing for any included source.

    Parameters
    ----------
    feature_cols : list
        Original list of feature columns.
    blocking_features : list
        Features to remove (100% missing for some source).

    Returns
    -------
    list
        Filtered feature columns.
    """
    blocking_set = set(blocking_features)
    filtered = [f for f in feature_cols if f not in blocking_set]
    n_removed = len(feature_cols) - len(filtered)
    if n_removed > 0:
        print(
            f"  Removed {n_removed} blocking features: {blocking_features[:5]}{'...' if len(blocking_features) > 5 else ''}"
        )
    return filtered


def build_preprocessor(add_indicator: bool = True) -> SimpleImputer:
    """
    Create preprocessor for handling NaN values in features.

    Uses median imputation with optional missing indicator columns.

    Parameters
    ----------
    add_indicator : bool
        Whether to add binary columns indicating which values were imputed.
        Default True helps model learn from missingness patterns.

    Returns
    -------
    SimpleImputer
        Fitted imputer (call fit_transform on training data).
    """
    return SimpleImputer(strategy="median", add_indicator=add_indicator)


def van_genuchten(
    theta: np.ndarray,
    theta_r: np.ndarray,
    theta_s: np.ndarray,
    alpha: np.ndarray,
    n: np.ndarray,
) -> np.ndarray:
    """
    Compute soil water potential (suction) from volumetric water content using VG equation.

    Parameters
    ----------
    theta : array-like
        Volumetric water content (0-1).
    theta_r : array-like
        Residual water content.
    theta_s : array-like
        Saturated water content.
    alpha : array-like
        VG alpha parameter (1/cm).
    n : array-like
        VG n parameter (dimensionless).

    Returns
    -------
    np.ndarray
        Soil water potential (suction) in cm H2O.
    """
    # Effective saturation
    Se = (theta - theta_r) / (theta_s - theta_r)
    Se = np.clip(Se, 1e-10, 1 - 1e-10)  # Avoid division issues

    m = 1 - 1 / n

    # Inverse VG: h = (1/alpha) * (Se^(-1/m) - 1)^(1/n)
    h = (1.0 / alpha) * (Se ** (-1.0 / m) - 1) ** (1.0 / n)

    return h


def get_feature_columns(
    df: pd.DataFrame,
    include_depth: bool = True,
    include_embeddings: bool = False,
    include_rosetta: bool = False,
) -> List[str]:
    """
    Get feature columns from dataframe, excluding targets and identifiers.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    include_depth : bool
        Whether to include depth columns (rosetta_level, depth_cm).
    include_embeddings : bool
        Whether to include embedding columns (e00-e63, A00-A63).
        Default False because they have partial availability.
    include_rosetta : bool
        Whether to include Rosetta prediction columns (US_R3H3_*).
        Default False because they have partial availability.
    """
    # Pattern for Rosetta target columns
    rosetta_pattern = re.compile(r"^US_R3H3_L\d+_VG_")

    feature_cols = []
    for c in df.columns:
        # Skip non-feature columns
        if c in NON_FEATURE_COLS:
            continue

        # Skip Rosetta columns unless explicitly included
        if not include_rosetta and rosetta_pattern.match(c):
            continue

        # Skip embedding columns unless explicitly included
        if not include_embeddings and any(pat.match(c) for pat in _EMBEDDING_PATTERNS):
            continue

        feature_cols.append(c)

    if include_depth:
        if "depth_cm" in df.columns and "depth_cm" not in feature_cols:
            feature_cols.append("depth_cm")
        if "rosetta_level" in df.columns and "rosetta_level" not in feature_cols:
            feature_cols.append("rosetta_level")

    # Remove any columns with all NaN
    feature_cols = [c for c in feature_cols if df[c].notna().any()]

    return sorted(set(feature_cols))


def extract_site_id(sample_id) -> str:
    """Extract site identifier from sample_id for grouping."""
    # Handle NaN or non-string values
    if pd.isna(sample_id) or not isinstance(sample_id, str):
        return str(sample_id)
    # sample_id format: {source}_{profile}_{depth}
    # We want {source}_{profile}
    parts = sample_id.rsplit("_", 1)
    return parts[0] if len(parts) > 1 else sample_id


def create_site_split(
    df: pd.DataFrame,
    group_col: str = "sample_id",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Set[str], Set[str]]:
    """
    Create train/test site split that can be reused across dataframes.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to derive sites from.
    group_col : str
        Column to extract site ID from.
    test_size : float
        Fraction of sites for testing.
    random_state : int
        Random seed.

    Returns
    -------
    train_sites, test_sites : tuple of sets
        Sets of site identifiers for train and test.
    """
    # Extract site IDs
    if group_col in df.columns:
        site_ids = df[group_col].apply(extract_site_id)
    elif df.index.name == group_col or "obs_id" in str(df.index.name):
        site_ids = df.index.to_series().apply(extract_site_id)
    else:
        raise ValueError(f"Cannot find group column: {group_col}")

    unique_sites = list(site_ids.unique())

    train_sites, test_sites = train_test_split(
        unique_sites, test_size=test_size, random_state=random_state
    )

    return set(train_sites), set(test_sites)


def apply_site_split(
    df: pd.DataFrame,
    train_sites: Set[str],
    test_sites: Set[str],
    group_col: str = "sample_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply pre-computed site split to a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    train_sites : set
        Site identifiers for training.
    test_sites : set
        Site identifiers for testing.
    group_col : str
        Column to extract site ID from.

    Returns
    -------
    train_df, test_df : tuple of pd.DataFrame
    """
    # Extract site IDs
    if group_col in df.columns:
        site_ids = df[group_col].apply(extract_site_id)
    elif df.index.name == group_col or "obs_id" in str(df.index.name):
        site_ids = df.index.to_series().apply(extract_site_id)
    else:
        raise ValueError(f"Cannot find group column: {group_col}")

    train_mask = site_ids.isin(train_sites)
    test_mask = site_ids.isin(test_sites)

    return df[train_mask.values].copy(), df[test_mask.values].copy()


def grouped_train_test_split(
    df: pd.DataFrame,
    group_col: str = "sample_id",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data ensuring all observations from a site are in same split.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    group_col : str
        Column to extract group ID from.
    test_size : float
        Fraction of sites for testing.
    random_state : int
        Random seed.

    Returns
    -------
    train_df, test_df : tuple of pd.DataFrame
    """
    # Extract site IDs
    if group_col in df.columns:
        site_ids = df[group_col].apply(extract_site_id)
    elif df.index.name == group_col or "obs_id" in str(df.index.name):
        site_ids = df.index.to_series().apply(extract_site_id)
    else:
        raise ValueError(f"Cannot find group column: {group_col}")

    unique_sites = list(site_ids.unique())

    train_sites, test_sites = train_test_split(
        unique_sites, test_size=test_size, random_state=random_state
    )

    train_mask = site_ids.isin(train_sites)
    test_mask = site_ids.isin(test_sites)

    return df[train_mask.values].copy(), df[test_mask.values].copy()


def load_vg_table(path: str) -> pd.DataFrame:
    """Load VG parameter training table."""
    df = pd.read_parquet(path)
    if df.index.name:
        df = df.reset_index()
    return df


def load_obs_table(path: str) -> pd.DataFrame:
    """Load observation-level training table."""
    df = pd.read_parquet(path)
    if df.index.name:
        df = df.reset_index()
    return df


def train_vg_model(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    n_estimators: int = 200,
    random_state: int = 42,
    log_transform_params: Optional[List[str]] = None,
    preprocessor: Optional[SimpleImputer] = None,
) -> Tuple[Dict[str, RandomForestRegressor], SimpleImputer]:
    """
    Train RF models to predict VG parameters.

    Uses median imputation for missing feature values instead of dropping rows,
    which prevents silent data loss from sources with different feature availability.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data.
    feature_cols : List[str]
        Feature columns.
    n_estimators : int
        Number of trees.
    random_state : int
        Random seed.
    log_transform_params : list, optional
        VG parameters to predict on log10 scale. Predictions will need
        to be transformed back via 10^pred.
    preprocessor : SimpleImputer, optional
        Pre-fitted imputer. If None, a new one is created and fitted.

    Returns
    -------
    tuple
        (models dict mapping param name to trained model, fitted preprocessor)
    """
    models = {}
    log_params = set(log_transform_params or [])

    X_train_raw = train_df[feature_cols].values

    # Create and fit preprocessor if not provided
    if preprocessor is None:
        preprocessor = build_preprocessor(add_indicator=False)
        X_train = preprocessor.fit_transform(X_train_raw)
    else:
        X_train = preprocessor.transform(X_train_raw)

    for param in VG_PARAMS:
        if param not in train_df.columns:
            print(f"  Warning: {param} not in training data")
            continue

        y_train = train_df[param].values

        # Apply log10 transform if requested
        if param in log_params:
            # Ensure positive values for log
            y_train = np.where(y_train > 0, np.log10(y_train), np.nan)

        # Only filter on TARGET NaN - features are imputed
        valid_mask = ~np.isnan(y_train)

        if valid_mask.sum() < 10:
            print(f"  Warning: insufficient data for {param}")
            continue

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=-1,
            random_state=random_state,
        )
        model.fit(X_train[valid_mask], y_train[valid_mask])
        models[param] = model
        log_str = " (log10)" if param in log_params else ""
        print(f"  Trained {param}{log_str} model on {valid_mask.sum()} samples")

    return models, preprocessor


def get_feature_cols_for_config(
    df: pd.DataFrame,
    config: VGExperimentConfig,
) -> List[str]:
    """
    Get feature columns based on experiment config.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to extract features from.
    config : VGExperimentConfig
        Experiment configuration.

    Returns
    -------
    list
        Feature column names.
    """
    # Start with base features (excluding depth, embeddings, and Rosetta columns)
    feature_cols = get_feature_columns(
        df,
        include_depth=False,
        include_embeddings=False,
        include_rosetta=False,
    )

    # Add depth columns based on depth_handling
    if config.depth_handling == "feature":
        if "rosetta_level" in df.columns:
            feature_cols.append("rosetta_level")
    elif config.depth_handling == "continuous":
        if "depth_cm" in df.columns:
            feature_cols.append("depth_cm")
    elif config.depth_handling == "both":
        if "rosetta_level" in df.columns:
            feature_cols.append("rosetta_level")
        if "depth_cm" in df.columns:
            feature_cols.append("depth_cm")
    # 'none' and 'per_level' don't add depth columns to features

    # Apply feature group exclusions
    if config.exclude_feature_groups:
        feature_cols = filter_feature_groups(
            feature_cols, config.exclude_feature_groups
        )

    return sorted(set(feature_cols))


def train_vg_with_depth_handling(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    config: VGExperimentConfig,
    random_state: int = 42,
) -> Dict[str, any]:
    """
    Train VG parameter models with specified depth handling strategy.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with VG parameters and features.
    feature_cols : List[str]
        Feature columns to use.
    config : VGExperimentConfig
        Experiment configuration.
    random_state : int
        Random seed.

    Returns
    -------
    dict
        For per_level: {'level_models': {level: {param: model}}, 'level_preprocessors': {level: imputer},
                        'levels': [1,2,...], 'log_transform_params': [...]}
        Otherwise: {'models': {param: model}, 'preprocessor': imputer, 'log_transform_params': [...]}
    """
    log_params = config.log_transform_params

    if config.depth_handling == "per_level":
        # Train separate models for each Rosetta level
        level_models = {}
        level_preprocessors = {}
        levels = sorted(train_df["rosetta_level"].dropna().unique())
        levels = [int(lvl) for lvl in levels]

        for level in levels:
            level_df = train_df[train_df["rosetta_level"] == level].copy()
            if len(level_df) < 10:
                print(f"  Level {level}: Skipping (only {len(level_df)} samples)")
                continue

            print(f"  Training models for level {level} ({len(level_df)} samples)...")
            models, preprocessor = train_vg_model(
                level_df,
                feature_cols,
                config.n_estimators,
                random_state,
                log_transform_params=log_params,
            )
            level_models[level] = models
            level_preprocessors[level] = preprocessor

        return {
            "level_models": level_models,
            "level_preprocessors": level_preprocessors,
            "levels": levels,
            "log_transform_params": log_params,
        }
    else:
        # Train single set of models across all levels
        models, preprocessor = train_vg_model(
            train_df,
            feature_cols,
            config.n_estimators,
            random_state,
            log_transform_params=log_params,
        )
        return {
            "models": models,
            "preprocessor": preprocessor,
            "log_transform_params": log_params,
        }


def predict_vg_per_level(
    level_models: Dict[int, Dict[str, RandomForestRegressor]],
    test_obs_df: pd.DataFrame,
    feature_cols: List[str],
    log_transform_params: Optional[List[str]] = None,
    level_preprocessors: Optional[Dict[int, SimpleImputer]] = None,
) -> np.ndarray:
    """
    Predict VG parameters using level-specific models.

    Parameters
    ----------
    level_models : dict
        Dict mapping level -> {param: model}.
    test_obs_df : pd.DataFrame
        Test observations with rosetta_level column.
    feature_cols : List[str]
        Feature columns.
    log_transform_params : list, optional
        VG parameters that were trained on log10 scale and need inverse transform.
    level_preprocessors : dict, optional
        Dict mapping level -> fitted SimpleImputer. If provided, features are
        imputed before prediction.

    Returns
    -------
    np.ndarray
        Predicted log10(suction) values.
    """
    n_samples = len(test_obs_df)
    log10_psi = np.full(n_samples, np.nan)
    log_params = set(log_transform_params or [])

    X_test_raw = test_obs_df[feature_cols].values
    theta = test_obs_df["theta"].values
    levels = test_obs_df["rosetta_level"].values

    for level, models in level_models.items():
        mask = levels == level
        if mask.sum() == 0:
            continue

        # Apply imputation if preprocessor available
        if level_preprocessors and level in level_preprocessors:
            X_test_level = level_preprocessors[level].transform(X_test_raw[mask])
        else:
            X_test_level = X_test_raw[mask]

        # Predict VG parameters for this level
        params = {}
        for param in VG_PARAMS:
            if param in models:
                pred = models[param].predict(X_test_level)
                # Inverse transform if trained on log scale
                if param in log_params:
                    pred = 10**pred
                params[param] = pred
            else:
                # Default fallbacks
                defaults = {"theta_r": 0.05, "theta_s": 0.45, "alpha": 0.01, "n": 1.5}
                params[param] = np.full(mask.sum(), defaults.get(param, 0.1))

        # Apply VG equation
        psi = van_genuchten(
            theta[mask],
            params["theta_r"],
            params["theta_s"],
            params["alpha"],
            params["n"],
        )
        psi = np.clip(psi, 1e-6, 1e10)
        log10_psi[mask] = np.log10(psi)

    return log10_psi


def train_direct_model(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    n_estimators: int = 200,
    random_state: int = 42,
    preprocessor: Optional[SimpleImputer] = None,
) -> Tuple[RandomForestRegressor, SimpleImputer]:
    """
    Train RF model to predict log10(suction_cm) directly.

    Uses median imputation for missing feature values instead of dropping rows,
    which prevents silent data loss from sources with different feature availability.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data.
    feature_cols : List[str]
        Feature columns (theta will be added automatically).
    n_estimators : int
        Number of trees.
    random_state : int
        Random seed.
    preprocessor : SimpleImputer, optional
        Pre-fitted imputer. If None, a new one is created and fitted.

    Returns
    -------
    tuple
        (trained RandomForestRegressor model, fitted SimpleImputer preprocessor)
    """
    # Include theta as a feature
    all_features = feature_cols + ["theta"]

    X_train_raw = train_df[all_features].values
    y_train = train_df["log10_suction_cm"].values

    # Create and fit preprocessor if not provided
    if preprocessor is None:
        preprocessor = build_preprocessor(add_indicator=False)
        X_train = preprocessor.fit_transform(X_train_raw)
    else:
        X_train = preprocessor.transform(X_train_raw)

    # Only filter on TARGET NaN - features are imputed
    valid_mask = ~np.isnan(y_train)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_train[valid_mask], y_train[valid_mask])
    print(f"  Trained direct model on {valid_mask.sum()} observations")

    return model, preprocessor


def predict_vg_approach(
    vg_models: Dict[str, RandomForestRegressor],
    test_obs_df: pd.DataFrame,
    feature_cols: List[str],
    log_transform_params: Optional[List[str]] = None,
    preprocessor: Optional[SimpleImputer] = None,
) -> np.ndarray:
    """
    Predict log10(suction) using VG parameter approach.

    1. Predict VG params from EE features
    2. Apply VG equation with observed theta

    Parameters
    ----------
    vg_models : dict
        Dict mapping param name to trained model.
    test_obs_df : pd.DataFrame
        Test observations.
    feature_cols : List[str]
        Feature columns.
    log_transform_params : list, optional
        VG parameters that were trained on log10 scale and need inverse transform.
    preprocessor : SimpleImputer, optional
        Fitted imputer for handling NaN values in test features.
    """
    X_test_raw = test_obs_df[feature_cols].values
    log_params = set(log_transform_params or [])

    # Apply imputation if preprocessor provided
    if preprocessor is not None:
        X_test = preprocessor.transform(X_test_raw)
    else:
        X_test = X_test_raw

    # Predict VG parameters
    params = {}
    for param in VG_PARAMS:
        if param in vg_models:
            pred = vg_models[param].predict(X_test)
            # Inverse transform if trained on log scale
            if param in log_params:
                pred = 10**pred
            params[param] = pred
        else:
            # Use reasonable defaults if model missing
            if param == "theta_r":
                params[param] = np.full(len(X_test), 0.05)
            elif param == "theta_s":
                params[param] = np.full(len(X_test), 0.45)
            elif param == "alpha":
                params[param] = np.full(len(X_test), 0.01)
            elif param == "n":
                params[param] = np.full(len(X_test), 1.5)

    # Get observed theta
    theta = test_obs_df["theta"].values

    # Apply VG equation
    psi = van_genuchten(
        theta,
        params["theta_r"],
        params["theta_s"],
        params["alpha"],
        params["n"],
    )

    # Convert to log10, handling edge cases
    psi = np.clip(psi, 1e-6, 1e10)
    log10_psi = np.log10(psi)

    return log10_psi


def predict_direct_approach(
    direct_model: RandomForestRegressor,
    test_obs_df: pd.DataFrame,
    feature_cols: List[str],
    preprocessor: Optional[SimpleImputer] = None,
) -> np.ndarray:
    """
    Predict log10(suction) directly from EE features + theta.

    Parameters
    ----------
    direct_model : RandomForestRegressor
        Trained model.
    test_obs_df : pd.DataFrame
        Test observations.
    feature_cols : List[str]
        Feature columns (theta will be added automatically).
    preprocessor : SimpleImputer, optional
        Fitted imputer for handling NaN values in test features.
    """
    all_features = feature_cols + ["theta"]
    X_test_raw = test_obs_df[all_features].values

    # Apply imputation if preprocessor provided
    if preprocessor is not None:
        X_test = preprocessor.transform(X_test_raw)
    else:
        X_test = X_test_raw

    return direct_model.predict(X_test)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    linear: bool = False,
) -> Dict[str, float]:
    """
    Compute comparison metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True values (log10 scale).
    y_pred : np.ndarray
        Predicted values (log10 scale).
    linear : bool
        If True, transform to linear scale (cm H₂O) before computing metrics.

    Returns
    -------
    dict
        Dictionary of metrics.
    """
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isinf(y_pred)

    if valid.sum() == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "bias": np.nan, "n": 0}

    y_true_v = y_true[valid]
    y_pred_v = y_pred[valid]

    if linear:
        # Transform from log10 to linear scale (cm H₂O)
        y_true_v = 10**y_true_v
        y_pred_v = 10**y_pred_v

    return {
        "rmse": np.sqrt(mean_squared_error(y_true_v, y_pred_v)),
        "mae": mean_absolute_error(y_true_v, y_pred_v),
        "r2": r2_score(y_true_v, y_pred_v),
        "bias": np.mean(y_pred_v - y_true_v),
        "n": int(valid.sum()),
    }


def compute_metrics_by_source(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sources: np.ndarray,
    linear: bool = False,
) -> pd.DataFrame:
    """
    Compute metrics stratified by data source.

    Parameters
    ----------
    y_true : np.ndarray
        True values (log10 scale).
    y_pred : np.ndarray
        Predicted values (log10 scale).
    sources : np.ndarray
        Source labels for each observation.
    linear : bool
        If True, compute metrics on linear scale.

    Returns
    -------
    pd.DataFrame
        DataFrame with RMSE/MAE/R²/bias per source.
    """
    results = []
    unique_sources = np.unique(sources[~pd.isna(sources)])

    for source in unique_sources:
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
    """
    Compute per-site metrics and site-weighted aggregates.

    Parameters
    ----------
    y_true : np.ndarray
        True values (log10 scale).
    y_pred : np.ndarray
        Predicted values (log10 scale).
    site_ids : np.ndarray
        Site identifiers for each observation.
    linear : bool
        If True, compute metrics on linear scale.

    Returns
    -------
    tuple
        (per_site_df, site_weighted_summary)
        - per_site_df: DataFrame with metrics per site
        - site_weighted_summary: dict with mean/median across sites
    """
    results = []
    unique_sites = np.unique(site_ids[~pd.isna(site_ids)])

    for site in unique_sites:
        mask = site_ids == site
        if mask.sum() < 3:  # Need at least 3 observations per site
            continue

        metrics = compute_metrics(y_true[mask], y_pred[mask], linear=linear)
        metrics["site_id"] = site
        results.append(metrics)

    per_site_df = pd.DataFrame(results)

    # Compute site-weighted aggregates (equal weight to each site)
    if len(per_site_df) > 0:
        site_weighted = {
            "mean_rmse": per_site_df["rmse"].mean(),
            "median_rmse": per_site_df["rmse"].median(),
            "mean_r2": per_site_df["r2"].mean(),
            "median_r2": per_site_df["r2"].median(),
            "mean_mae": per_site_df["mae"].mean(),
            "median_mae": per_site_df["mae"].median(),
            "n_sites": len(per_site_df),
        }
    else:
        site_weighted = {
            "mean_rmse": np.nan,
            "median_rmse": np.nan,
            "mean_r2": np.nan,
            "median_r2": np.nan,
            "mean_mae": np.nan,
            "median_mae": np.nan,
            "n_sites": 0,
        }

    return per_site_df, site_weighted


def compute_stratified_metrics(
    y_true: np.ndarray,
    y_pred_vg: np.ndarray,
    y_pred_direct: np.ndarray,
    theta: np.ndarray,
    depth: np.ndarray,
    linear: bool = False,
) -> pd.DataFrame:
    """Compute metrics stratified by theta range and depth."""
    results = []

    # By theta range
    theta_bins = [(0, 0.15, "dry"), (0.15, 0.30, "mid"), (0.30, 1.0, "wet")]
    for low, high, name in theta_bins:
        mask = (theta >= low) & (theta < high)
        if mask.sum() > 0:
            vg_metrics = compute_metrics(y_true[mask], y_pred_vg[mask], linear=linear)
            direct_metrics = compute_metrics(
                y_true[mask], y_pred_direct[mask], linear=linear
            )
            results.append(
                {
                    "stratification": f"theta_{name}",
                    "n": mask.sum(),
                    "vg_rmse": vg_metrics["rmse"],
                    "vg_r2": vg_metrics["r2"],
                    "direct_rmse": direct_metrics["rmse"],
                    "direct_r2": direct_metrics["r2"],
                }
            )

    # By depth (Rosetta level)
    for level in sorted(np.unique(depth[~np.isnan(depth)])):
        mask = depth == level
        if mask.sum() > 10:
            vg_metrics = compute_metrics(y_true[mask], y_pred_vg[mask], linear=linear)
            direct_metrics = compute_metrics(
                y_true[mask], y_pred_direct[mask], linear=linear
            )
            results.append(
                {
                    "stratification": f"level_{int(level)}",
                    "n": mask.sum(),
                    "vg_rmse": vg_metrics["rmse"],
                    "vg_r2": vg_metrics["r2"],
                    "direct_rmse": direct_metrics["rmse"],
                    "direct_r2": direct_metrics["r2"],
                }
            )

    return pd.DataFrame(results)


def check_monotonicity(
    y_pred: np.ndarray,
    theta: np.ndarray,
    sample_ids: np.ndarray,
) -> Dict[str, float]:
    """
    Check if predictions are monotonic (dψ/dθ < 0) within each sample.

    Returns fraction of samples with monotonicity violations.
    """
    violations = 0
    total = 0

    # Convert to pandas Series to handle mixed types and NaN values
    sample_ids_series = pd.Series(sample_ids)
    unique_ids = sample_ids_series.dropna().unique()

    for sid in unique_ids:
        mask = sample_ids == sid
        if mask.sum() < 3:
            continue

        theta_s = theta[mask]
        psi_s = y_pred[mask]

        # Sort by theta
        order = np.argsort(theta_s)
        theta_sorted = theta_s[order]
        psi_sorted = psi_s[order]

        # Check monotonicity: as theta increases, psi should decrease
        dtheta = np.diff(theta_sorted)
        dpsi = np.diff(psi_sorted)

        # Violation: theta increases but psi also increases
        violation_mask = (dtheta > 0.001) & (dpsi > 0.1)

        if violation_mask.any():
            violations += 1
        total += 1

    return {
        "violation_rate": violations / total if total > 0 else 0,
        "violations": violations,
        "total_samples": total,
    }


def plot_comparison(
    y_true: np.ndarray,
    y_pred_vg: np.ndarray,
    y_pred_direct: np.ndarray,
    theta: np.ndarray,
    output_path: str,
):
    """Generate comparison plots.

    Style convention:
        - Observed data: solid black 1:1 line
        - VG model: steelblue, open circles (dashed outline style)
        - Direct model: grey filled scatter
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    valid_vg = ~np.isnan(y_pred_vg) & ~np.isinf(y_pred_vg)
    valid_direct = ~np.isnan(y_pred_direct)

    vg_color = "steelblue"
    direct_color = "0.55"

    # 1. Predicted vs Observed - VG approach
    ax = axes[0, 0]
    ax.scatter(
        y_true[valid_vg],
        y_pred_vg[valid_vg],
        alpha=0.15,
        s=3,
        facecolors="none",
        edgecolors=vg_color,
        linewidths=0.4,
    )
    ax.plot([0, 7], [0, 7], "k-", lw=1.5)
    ax.set_xlabel("Observed log10(ψ)")
    ax.set_ylabel("Predicted log10(ψ)")
    ax.set_title("VG Parameter Approach")
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 7)

    # 2. Predicted vs Observed - Direct approach
    ax = axes[0, 1]
    ax.scatter(
        y_true[valid_direct],
        y_pred_direct[valid_direct],
        alpha=0.15,
        s=3,
        c=direct_color,
        edgecolors="none",
    )
    ax.plot([0, 7], [0, 7], "k-", lw=1.5)
    ax.set_xlabel("Observed log10(ψ)")
    ax.set_ylabel("Predicted log10(ψ)")
    ax.set_title("Direct Approach")
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 7)

    # 3. Residuals comparison
    ax = axes[0, 2]
    residuals_vg = y_pred_vg[valid_vg] - y_true[valid_vg]
    residuals_direct = y_pred_direct[valid_direct] - y_true[valid_direct]
    ax.hist(
        residuals_vg,
        bins=50,
        alpha=0.5,
        color=vg_color,
        linestyle="--",
        edgecolor=vg_color,
        linewidth=0.8,
        label=f"VG (σ={np.std(residuals_vg):.2f})",
        density=True,
    )
    ax.hist(
        residuals_direct,
        bins=50,
        alpha=0.4,
        color=direct_color,
        edgecolor="0.35",
        linewidth=0.5,
        label=f"Direct (σ={np.std(residuals_direct):.2f})",
        density=True,
    )
    ax.set_xlabel("Residual (pred - obs)")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distribution")
    ax.legend()
    ax.set_xlim(-5, 5)

    # 4. Error vs theta - VG
    ax = axes[1, 0]
    error_vg = np.abs(y_pred_vg - y_true)
    ax.scatter(
        theta[valid_vg],
        error_vg[valid_vg],
        alpha=0.15,
        s=3,
        facecolors="none",
        edgecolors=vg_color,
        linewidths=0.4,
    )
    ax.set_xlabel("θ (VWC)")
    ax.set_ylabel("|Error| log10(ψ)")
    ax.set_title("VG Error vs θ")
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 5)

    # 5. Error vs theta - Direct
    ax = axes[1, 1]
    error_direct = np.abs(y_pred_direct - y_true)
    ax.scatter(
        theta[valid_direct],
        error_direct[valid_direct],
        alpha=0.15,
        s=3,
        c=direct_color,
        edgecolors="none",
    )
    ax.set_xlabel("θ (VWC)")
    ax.set_ylabel("|Error| log10(ψ)")
    ax.set_title("Direct Error vs θ")
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 5)

    # 6. Direct vs VG predictions
    ax = axes[1, 2]
    both_valid = valid_vg & valid_direct
    sc = ax.scatter(
        y_pred_vg[both_valid],
        y_pred_direct[both_valid],
        alpha=0.15,
        s=3,
        c=theta[both_valid],
        cmap="viridis",
        edgecolors="none",
    )
    ax.plot([0, 7], [0, 7], "k-", lw=1.5)
    ax.set_xlabel("VG Prediction")
    ax.set_ylabel("Direct Prediction")
    ax.set_title("VG vs Direct (color=θ)")
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 7)
    fig.colorbar(sc, ax=ax, label="θ", shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to {output_path}")


def plot_retention_curves(
    test_obs_df: pd.DataFrame,
    y_pred_vg: np.ndarray,
    y_pred_direct: np.ndarray,
    output_path: str,
    n_samples: int = 9,
):
    """Plot example retention curves for selected test sites."""
    # Get sample_ids with enough observations
    if "sample_id" in test_obs_df.columns:
        sample_col = "sample_id"
    else:
        test_obs_df = test_obs_df.reset_index()
        sample_col = "obs_id"
        test_obs_df["sample_id"] = test_obs_df[sample_col].apply(extract_site_id)
        sample_col = "sample_id"

    sample_counts = test_obs_df[sample_col].value_counts()
    good_samples = sample_counts[sample_counts >= 5].index[:n_samples]

    if len(good_samples) == 0:
        print("No samples with enough observations for curve plotting")
        return

    n_cols = 3
    n_rows = (len(good_samples) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = (
        axes.flatten()
        if n_rows > 1
        else [axes]
        if n_rows == 1 and n_cols == 1
        else axes
    )

    for idx, sample_id in enumerate(good_samples):
        if idx >= len(axes):
            break

        ax = axes[idx]
        mask = test_obs_df[sample_col] == sample_id

        theta = test_obs_df.loc[mask, "theta"].values
        y_true = test_obs_df.loc[mask, "log10_suction_cm"].values
        y_vg = y_pred_vg[mask.values] if hasattr(mask, "values") else y_pred_vg[mask]
        y_direct = (
            y_pred_direct[mask.values]
            if hasattr(mask, "values")
            else y_pred_direct[mask]
        )

        # Sort by theta for line plot
        order = np.argsort(theta)

        ax.scatter(theta, y_true, c="black", s=20, label="Observed", zorder=3)
        ax.plot(theta[order], y_vg[order], "b-", lw=2, label="VG", alpha=0.7)
        ax.plot(theta[order], y_direct[order], "r--", lw=2, label="Direct", alpha=0.7)

        ax.set_xlabel("θ")
        ax.set_ylabel("log10(ψ)")
        ax.set_title(f"{sample_id[:30]}..." if len(sample_id) > 30 else sample_id)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlim(0, 0.8)
        ax.set_ylim(0, 7)

    # Hide unused axes
    for idx in range(len(good_samples), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved retention curves to {output_path}")


def plot_vg_model_comparison(
    vg_results: Dict[str, Dict],
    direct_r2: float,
    output_path: str,
):
    """
    Generate bar chart comparing all VG model variants by suction R².

    Parameters
    ----------
    vg_results : dict
        Dict mapping model name to results dict containing 'suction_r2'.
    direct_r2 : float
        R² of direct model for reference line.
    output_path : str
        Path to save figure.
    """
    # Sort by R² (descending - higher is better)
    sorted_models = sorted(
        vg_results.items(), key=lambda x: x[1]["suction_r2"], reverse=True
    )
    names = [m[0] for m in sorted_models]
    r2s = [m[1]["suction_r2"] for m in sorted_models]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Bar chart
    bars = ax.bar(range(len(names)), r2s, color="steelblue", alpha=0.8)

    # Highlight best model (first after sorting)
    bars[0].set_color("darkgreen")

    # Add direct model reference line
    ax.axhline(
        y=direct_r2,
        color="red",
        linestyle="--",
        lw=2,
        label=f"Direct Model (R²={direct_r2:.3f})",
    )

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Suction R²")
    ax.set_title("VG Model Comparison: Suction Prediction Performance")
    ax.legend()

    # Add value labels on bars
    for i, (bar, r2) in enumerate(zip(bars, r2s)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{r2:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved VG model comparison to {output_path}")


def run_systematic_comparison(
    vg_table_path: str,
    obs_table_path: str,
    output_dir: str,
    experiments: Optional[List[VGExperimentConfig]] = None,
    direct_experiments: Optional[List[DirectExperimentConfig]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    linear: bool = False,
    drop_blocking_features: bool = True,
) -> Dict:
    """
    Run systematic comparison of multiple VG model configurations against direct prediction.

    Parameters
    ----------
    vg_table_path : str
        Path to VG parameter training table.
    obs_table_path : str
        Path to observation-level training table.
    output_dir : str
        Directory for output files.
    experiments : list of VGExperimentConfig, optional
        List of VG experiments to run. If None, uses DEFAULT_VG_EXPERIMENTS.
    direct_experiments : list of DirectExperimentConfig, optional
        List of direct model experiments. If None, uses DEFAULT_DIRECT_EXPERIMENTS.
    test_size : float
        Fraction of sites for testing.
    random_state : int
        Random seed.
    linear : bool
        If True, compute metrics on linear scale (cm H₂O) instead of log10 scale.
    drop_blocking_features : bool
        If True (default), remove features that are 100% missing for any source.
        This prevents silent data dropping from sources with different feature availability.

    Returns
    -------
    dict
        Full comparison results.
    """
    # Handle None vs empty list: None means use defaults, empty list means skip
    if experiments is None:
        experiments = DEFAULT_VG_EXPERIMENTS
    if direct_experiments is None:
        direct_experiments = DEFAULT_DIRECT_EXPERIMENTS

    # Ensure at least one type of experiment is provided
    if not experiments and not direct_experiments:
        raise ValueError("At least one VG or direct experiment must be provided")

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("SYSTEMATIC VG MODEL COMPARISON")
    print("=" * 70)
    scale_str = "linear (cm H₂O)" if linear else "log10"
    print(f"Metric scale: {scale_str}")

    # Load data
    print("\n1. Loading data...")
    vg_df = load_vg_table(vg_table_path)
    obs_df = load_obs_table(obs_table_path)
    print(f"   VG table (raw): {len(vg_df)} samples")
    print(f"   Obs table (raw): {len(obs_df)} observations")

    # Filter to samples with lat/lon (these have EE features)
    # Samples without lat/lon have no EE features and can't be used
    vg_has_latlon = vg_df["lat"].notna() & vg_df["lon"].notna()
    obs_has_latlon = obs_df["lat"].notna() & obs_df["lon"].notna()

    vg_df = vg_df[vg_has_latlon].copy()
    obs_df = obs_df[obs_has_latlon].copy()

    print(f"   VG table (with lat/lon): {len(vg_df)} samples")
    print(f"   Obs table (with lat/lon): {len(obs_df)} observations")

    # Get base feature columns for audit
    base_features = get_feature_columns(vg_df, include_depth=False)
    obs_base_features = get_feature_columns(obs_df, include_depth=False)
    common_base_features = sorted(set(base_features) & set(obs_base_features))

    # Audit dataset by source
    print("\n1b. Auditing dataset by source...")
    if "source" in obs_df.columns:
        print("   Row counts by source:")
        for source, count in obs_df["source"].value_counts().items():
            print(f"      {source}: {count} observations")

        # Run audit and save results
        audit_result = audit_dataset(obs_df, common_base_features, output_dir)
        blocking_features = audit_result["blocking_features"]

        if blocking_features:
            print(
                f"   Blocking features (100% missing for some source): {blocking_features[:10]}{'...' if len(blocking_features) > 10 else ''}"
            )
            if drop_blocking_features:
                common_base_features = filter_blocking_features(
                    common_base_features, blocking_features
                )
                print(
                    f"   Features after removing blocking features: {len(common_base_features)}"
                )
            else:
                print("   Warning: Keeping blocking features - imputation will be used")
    else:
        print("   Warning: No 'source' column - skipping source audit")
        audit_result = None
        blocking_features = []

    # Create site split ONCE - all experiments use the same holdout
    print("\n2. Creating site split...")
    train_sites, test_sites = create_site_split(
        vg_df, "sample_id", test_size, random_state
    )
    print(f"   Train sites: {len(train_sites)}")
    print(f"   Test sites: {len(test_sites)}")

    # Apply split to both tables
    vg_train, vg_test = apply_site_split(vg_df, train_sites, test_sites, "sample_id")
    obs_train, obs_test = apply_site_split(obs_df, train_sites, test_sites, "sample_id")

    print(f"   VG train: {len(vg_train)}, test: {len(vg_test)} samples")
    print(f"   Obs train: {len(obs_train)}, test: {len(obs_test)} observations")

    # Print row counts by source for train/test
    if "source" in obs_train.columns:
        print("\n   Train observations by source:")
        for source, count in obs_train["source"].value_counts().items():
            print(f"      {source}: {count}")
        print("   Test observations by source:")
        for source, count in obs_test["source"].value_counts().items():
            print(f"      {source}: {count}")

    # Store results for each experiment
    vg_results = {}

    # Run each VG experiment
    print("\n3. Running VG experiments...")
    for config in experiments:
        print(f"\n--- Experiment: {config.name} ---")
        print(f"    Depth handling: {config.depth_handling}")
        if config.exclude_feature_groups:
            print(f"    Excluded groups: {config.exclude_feature_groups}")
        if config.log_transform_params:
            print(f"    Log-transformed params: {config.log_transform_params}")

        # Get features for this config
        feature_cols = get_feature_cols_for_config(vg_df, config)
        # Ensure we only use features in both tables
        feature_cols = [
            f
            for f in feature_cols
            if f in common_base_features or f in ["rosetta_level", "depth_cm"]
        ]
        # Further filter to columns that actually exist
        feature_cols = [
            f for f in feature_cols if f in vg_train.columns and f in obs_test.columns
        ]
        print(f"    Using {len(feature_cols)} features")

        # Prepare training data - only require VG params to be non-NaN
        # (train_vg_model handles NaN in features via valid_mask)
        vg_train_clean = vg_train.dropna(subset=VG_PARAMS)

        # Train VG models
        model_result = train_vg_with_depth_handling(
            vg_train_clean, feature_cols, config, random_state
        )

        # Prepare test data - only require theta and suction to be non-NaN
        required_cols = ["theta", "log10_suction_cm"]
        if config.depth_handling == "per_level" and "rosetta_level" in obs_test.columns:
            required_cols.append("rosetta_level")
        obs_test_clean = obs_test.dropna(subset=required_cols)

        # Generate predictions (using preprocessors for imputation)
        log_params = model_result.get("log_transform_params")
        if config.depth_handling == "per_level":
            y_pred = predict_vg_per_level(
                model_result["level_models"],
                obs_test_clean,
                feature_cols,
                log_transform_params=log_params,
                level_preprocessors=model_result.get("level_preprocessors"),
            )
        else:
            y_pred = predict_vg_approach(
                model_result["models"],
                obs_test_clean,
                feature_cols,
                log_transform_params=log_params,
                preprocessor=model_result.get("preprocessor"),
            )

        y_true = obs_test_clean["log10_suction_cm"].values

        # Compute metrics
        metrics = compute_metrics(y_true, y_pred, linear=linear)
        print(f"    Suction RMSE ({scale_str}): {metrics['rmse']:.4f}")
        print(f"    Suction R²: {metrics['r2']:.4f}")

        # Store VG param metrics if available
        vg_param_metrics = {}
        log_params_set = set(log_params or [])
        preprocessor = model_result.get("preprocessor")
        if "models" in model_result:
            for param, model in model_result["models"].items():
                if param in vg_test.columns:
                    # Only filter by target NaN, not feature NaN (use preprocessor)
                    vg_test_subset = vg_test.dropna(subset=[param])
                    X_vg_test_raw = vg_test_subset[feature_cols].values
                    y_vg_true = vg_test_subset[param].values
                    if len(X_vg_test_raw) > 0:
                        # Apply imputation if preprocessor available
                        if preprocessor is not None:
                            X_vg_test = preprocessor.transform(X_vg_test_raw)
                        else:
                            X_vg_test = X_vg_test_raw
                        y_vg_pred = model.predict(X_vg_test)
                        # If trained on log scale, transform back for comparison
                        if param in log_params_set:
                            y_vg_true_log = np.log10(np.clip(y_vg_true, 1e-10, None))
                            vg_param_metrics[param] = {
                                "r2": float(r2_score(y_vg_true_log, y_vg_pred)),
                                "rmse": float(
                                    np.sqrt(
                                        mean_squared_error(y_vg_true_log, y_vg_pred)
                                    )
                                ),
                                "scale": "log10",
                            }
                        else:
                            vg_param_metrics[param] = {
                                "r2": float(r2_score(y_vg_true, y_vg_pred)),
                                "rmse": float(
                                    np.sqrt(mean_squared_error(y_vg_true, y_vg_pred))
                                ),
                                "scale": "natural",
                            }

        vg_results[config.name] = {
            "config": {
                "name": config.name,
                "depth_handling": config.depth_handling,
                "exclude_feature_groups": config.exclude_feature_groups,
                "log_transform_params": config.log_transform_params,
                "n_estimators": config.n_estimators,
            },
            "n_features": len(feature_cols),
            "suction_rmse": metrics["rmse"],
            "suction_mae": metrics["mae"],
            "suction_r2": metrics["r2"],
            "suction_bias": metrics["bias"],
            "suction_n": metrics["n"],
            "vg_param_metrics": vg_param_metrics,
        }

    # Train direct models
    print("\n4. Training direct models...")

    # Only require theta and suction to be non-NaN (train_direct_model handles feature NaN)
    obs_train_clean = obs_train.dropna(subset=["theta", "log10_suction_cm"])
    obs_test_clean = obs_test.dropna(subset=["theta", "log10_suction_cm"])
    y_true = obs_test_clean["log10_suction_cm"].values

    direct_results = {}
    direct_models = {}  # Store models for later use in plotting

    for config in direct_experiments:
        print(f"\n--- Direct Experiment: {config.name} ---")
        if config.exclude_feature_groups:
            print(f"    Excluded groups: {config.exclude_feature_groups}")

        # Get features for this config
        direct_features = [f for f in common_base_features if f in obs_train.columns]
        if "rosetta_level" in obs_train.columns:
            direct_features.append("rosetta_level")
        if "depth_cm" in obs_train.columns:
            direct_features.append("depth_cm")

        # Apply feature group exclusions
        if config.exclude_feature_groups:
            direct_features = filter_feature_groups(
                direct_features, config.exclude_feature_groups
            )

        print(f"    Using {len(direct_features)} features")

        direct_model, direct_preprocessor = train_direct_model(
            obs_train_clean, direct_features, config.n_estimators, random_state
        )
        direct_models[config.name] = (
            direct_model,
            direct_features,
            direct_preprocessor,
        )

        y_pred_direct = predict_direct_approach(
            direct_model,
            obs_test_clean,
            direct_features,
            preprocessor=direct_preprocessor,
        )
        metrics = compute_metrics(y_true, y_pred_direct, linear=linear)

        print(f"    RMSE ({scale_str}): {metrics['rmse']:.4f}")
        print(f"    R²: {metrics['r2']:.4f}")

        direct_results[config.name] = {
            "config": {
                "name": config.name,
                "exclude_feature_groups": config.exclude_feature_groups,
                "n_estimators": config.n_estimators,
            },
            "n_features": len(direct_features),
            "suction_rmse": metrics["rmse"],
            "suction_mae": metrics["mae"],
            "suction_r2": metrics["r2"],
            "suction_bias": metrics["bias"],
            "suction_n": metrics["n"],
        }

    # Use best direct model for comparison
    best_direct_name = max(
        direct_results.keys(), key=lambda k: direct_results[k]["suction_r2"]
    )
    best_direct = direct_results[best_direct_name]
    direct_model, direct_features, direct_preprocessor = direct_models[best_direct_name]
    # Identify best VG model (by R², higher is better)
    best_vg_name = max(vg_results.keys(), key=lambda k: vg_results[k]["suction_r2"])
    best_vg = vg_results[best_vg_name]

    # Print summary (sorted by R², descending)
    print("\n" + "=" * 70)
    print("SUMMARY - VG MODELS")
    print("=" * 70)
    print(f"\n{'Model':<30} {'R²':<10} {'RMSE':<10}")
    print("-" * 50)
    for name, res in sorted(
        vg_results.items(), key=lambda x: x[1]["suction_r2"], reverse=True
    ):
        marker = " *BEST VG" if name == best_vg_name else ""
        print(
            f"{name:<30} {res['suction_r2']:<10.4f} {res['suction_rmse']:<10.4f}{marker}"
        )

    print("\n" + "=" * 70)
    print("SUMMARY - DIRECT MODELS")
    print("=" * 70)
    print(f"\n{'Model':<30} {'R²':<10} {'RMSE':<10} {'Features':<10}")
    print("-" * 60)
    for name, res in sorted(
        direct_results.items(), key=lambda x: x[1]["suction_r2"], reverse=True
    ):
        marker = " *BEST" if name == best_direct_name else ""
        print(
            f"{name:<30} {res['suction_r2']:<10.4f} {res['suction_rmse']:<10.4f} {res['n_features']:<10}{marker}"
        )

    # Comparison (by R², higher is better)
    print("\n" + "=" * 70)
    print("WINNER ANALYSIS")
    print("=" * 70)
    print(f"Best VG: {best_vg_name} (R²={best_vg['suction_r2']:.4f})")
    print(f"Best Direct: {best_direct_name} (R²={best_direct['suction_r2']:.4f})")
    if best_vg["suction_r2"] > best_direct["suction_r2"]:
        improvement = (
            (best_vg["suction_r2"] - best_direct["suction_r2"])
            / best_direct["suction_r2"]
            * 100
        )
        print(f"\nBest VG model ({best_vg_name}) beats best direct model!")
        print(f"R² improvement: {improvement:.1f}%")
    else:
        improvement = (
            (best_direct["suction_r2"] - best_vg["suction_r2"])
            / abs(best_vg["suction_r2"])
            * 100
        )
        print(
            f"\nBest direct model ({best_direct_name}) beats best VG model ({best_vg_name})"
        )
        print(f"Direct R² advantage: {improvement:.1f}%")

    # Generate plots
    print("\n5. Generating plots...")

    # VG model comparison bar chart
    plot_vg_model_comparison(
        vg_results,
        best_direct["suction_r2"],
        os.path.join(output_dir, "vg_model_comparison.png"),
    )

    # Best VG vs direct scatter comparison (reuse existing plot function)
    # Re-predict with best model for plotting
    best_config = next(c for c in experiments if c.name == best_vg_name)
    best_features = get_feature_cols_for_config(vg_df, best_config)
    best_features = [
        f
        for f in best_features
        if f in common_base_features or f in ["rosetta_level", "depth_cm"]
    ]

    vg_train_clean = vg_train.dropna(
        subset=[f for f in best_features if f in vg_train.columns]
    )
    best_model_result = train_vg_with_depth_handling(
        vg_train_clean, best_features, best_config, random_state
    )

    required_cols = [f for f in best_features if f in obs_test.columns] + [
        "theta",
        "log10_suction_cm",
    ]
    if (
        "rosetta_level" not in required_cols
        and best_config.depth_handling == "per_level"
    ):
        required_cols.append("rosetta_level")
    obs_test_final = obs_test.dropna(
        subset=[c for c in required_cols if c in obs_test.columns]
    )

    best_log_params = best_model_result.get("log_transform_params")
    if best_config.depth_handling == "per_level":
        y_pred_best_vg = predict_vg_per_level(
            best_model_result["level_models"],
            obs_test_final,
            best_features,
            log_transform_params=best_log_params,
            level_preprocessors=best_model_result.get("level_preprocessors"),
        )
    else:
        y_pred_best_vg = predict_vg_approach(
            best_model_result["models"],
            obs_test_final,
            best_features,
            log_transform_params=best_log_params,
            preprocessor=best_model_result.get("preprocessor"),
        )

    # Re-predict direct for same observations
    y_pred_direct_final = predict_direct_approach(
        direct_model, obs_test_final, direct_features, preprocessor=direct_preprocessor
    )
    y_true_final = obs_test_final["log10_suction_cm"].values
    theta_final = obs_test_final["theta"].values

    plot_comparison(
        y_true_final,
        y_pred_best_vg,
        y_pred_direct_final,
        theta_final,
        os.path.join(output_dir, "systematic_comparison.png"),
    )

    # Compute stratified metrics
    print("\n6. Computing stratified metrics...")
    depth_final = (
        obs_test_final["rosetta_level"].values
        if "rosetta_level" in obs_test_final.columns
        else np.zeros(len(obs_test_final))
    )
    stratified = compute_stratified_metrics(
        y_true_final,
        y_pred_best_vg,
        y_pred_direct_final,
        theta_final,
        depth_final,
        linear=linear,
    )
    print(stratified.to_string(index=False))

    # Save stratified results
    stratified_path = os.path.join(output_dir, "stratified_metrics.csv")
    stratified.to_csv(stratified_path, index=False)
    print(f"Saved stratified metrics to {stratified_path}")

    # Compute source-level metrics
    print("\n6b. Computing source-level metrics...")
    source_metrics_vg = None
    source_metrics_direct = None
    if "source" in obs_test_final.columns:
        sources_final = obs_test_final["source"].values

        # VG model metrics by source
        source_metrics_vg = compute_metrics_by_source(
            y_true_final, y_pred_best_vg, sources_final, linear=linear
        )
        source_metrics_vg["model"] = "vg"

        # Direct model metrics by source
        source_metrics_direct = compute_metrics_by_source(
            y_true_final, y_pred_direct_final, sources_final, linear=linear
        )
        source_metrics_direct["model"] = "direct"

        # Combine and save
        source_metrics_all = pd.concat(
            [source_metrics_vg, source_metrics_direct], ignore_index=True
        )
        source_metrics_path = os.path.join(output_dir, "metrics_by_source.csv")
        source_metrics_all.to_csv(source_metrics_path, index=False)
        print(f"   Saved source-level metrics to {source_metrics_path}")

        # Print summary
        print("\n   Source-level R² (VG / Direct):")
        for source in source_metrics_vg["source"].unique():
            vg_r2 = source_metrics_vg[source_metrics_vg["source"] == source][
                "r2"
            ].values[0]
            direct_r2 = source_metrics_direct[
                source_metrics_direct["source"] == source
            ]["r2"].values[0]
            vg_n = source_metrics_vg[source_metrics_vg["source"] == source]["n"].values[
                0
            ]
            print(f"      {source}: {vg_r2:.4f} / {direct_r2:.4f} (n={vg_n})")
    else:
        print("   Warning: No 'source' column - skipping source-level metrics")

    # Compute site-level metrics
    print("\n6c. Computing site-level metrics...")
    if "sample_id" in obs_test_final.columns:
        site_ids_final = obs_test_final["sample_id"].apply(extract_site_id).values
    else:
        site_ids_final = obs_test_final.index.to_series().apply(extract_site_id).values

    # VG model metrics by site
    site_metrics_vg, site_weighted_vg = compute_metrics_by_site(
        y_true_final, y_pred_best_vg, site_ids_final, linear=linear
    )
    site_metrics_vg["model"] = "vg"

    # Direct model metrics by site
    site_metrics_direct, site_weighted_direct = compute_metrics_by_site(
        y_true_final, y_pred_direct_final, site_ids_final, linear=linear
    )
    site_metrics_direct["model"] = "direct"

    # Combine and save
    site_metrics_all = pd.concat(
        [site_metrics_vg, site_metrics_direct], ignore_index=True
    )
    site_metrics_path = os.path.join(output_dir, "metrics_by_site.csv")
    site_metrics_all.to_csv(site_metrics_path, index=False)
    print(f"   Saved site-level metrics to {site_metrics_path}")

    # Print site-weighted summary
    print("\n   Site-weighted metrics (equal weight per site):")
    print(
        f"      VG:     mean R²={site_weighted_vg['mean_r2']:.4f}, median R²={site_weighted_vg['median_r2']:.4f} ({site_weighted_vg['n_sites']} sites)"
    )
    print(
        f"      Direct: mean R²={site_weighted_direct['mean_r2']:.4f}, median R²={site_weighted_direct['median_r2']:.4f} ({site_weighted_direct['n_sites']} sites)"
    )

    # Compute monotonicity checks
    print("\n7. Checking monotonicity...")
    if "sample_id" in obs_test_final.columns:
        sample_ids = obs_test_final["sample_id"].values
    else:
        sample_ids = obs_test_final.index.to_series().apply(extract_site_id).values

    mono_vg = check_monotonicity(y_pred_best_vg, theta_final, sample_ids)
    mono_direct = check_monotonicity(y_pred_direct_final, theta_final, sample_ids)
    print(
        f"   Best VG violation rate: {mono_vg['violation_rate']:.1%} "
        f"({mono_vg['violations']}/{mono_vg['total_samples']} samples)"
    )
    print(
        f"   Direct violation rate: {mono_direct['violation_rate']:.1%} "
        f"({mono_direct['violations']}/{mono_direct['total_samples']} samples)"
    )

    # Save full results
    results = {
        "vg_experiments": vg_results,
        "direct_experiments": direct_results,
        "best_vg_model": best_vg_name,
        "best_direct_model": best_direct_name,
        "comparison": {
            "winner": "vg"
            if best_vg["suction_r2"] > best_direct["suction_r2"]
            else "direct",
            "best_vg_name": best_vg_name,
            "best_vg_r2": best_vg["suction_r2"],
            "best_vg_rmse": best_vg["suction_rmse"],
            "best_direct_name": best_direct_name,
            "best_direct_r2": best_direct["suction_r2"],
            "best_direct_rmse": best_direct["suction_rmse"],
            "r2_difference": abs(best_vg["suction_r2"] - best_direct["suction_r2"]),
        },
        "config": {
            "test_size": test_size,
            "random_state": random_state,
            "n_train_sites": len(train_sites),
            "n_test_sites": len(test_sites),
            "n_vg_samples": len(vg_df),
            "n_obs_samples": len(obs_df),
            "metric_scale": "linear_cm" if linear else "log10",
            "drop_blocking_features": drop_blocking_features,
            "blocking_features_removed": blocking_features
            if drop_blocking_features
            else [],
        },
        "stratified_metrics": stratified.to_dict("records"),
        "source_metrics": {
            "vg": source_metrics_vg.to_dict("records")
            if source_metrics_vg is not None
            else [],
            "direct": source_metrics_direct.to_dict("records")
            if source_metrics_direct is not None
            else [],
        },
        "site_weighted_metrics": {
            "vg": site_weighted_vg,
            "direct": site_weighted_direct,
        },
        "monotonicity": {
            "best_vg": mono_vg,
            "direct": mono_direct,
        },
    }

    results_path = os.path.join(output_dir, "systematic_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to {results_path}")

    # Save predictions for use by composite_swrc.py visualization
    print("\n8. Saving predictions for visualization...")
    pred_df = obs_test_final.copy()
    pred_df["log10_suction_pred_vg"] = y_pred_best_vg
    pred_df["log10_suction_pred_direct"] = y_pred_direct_final
    pred_df["suction_cm_pred_vg"] = 10**y_pred_best_vg
    pred_df["suction_cm_pred_direct"] = 10**y_pred_direct_final

    # Ensure we have identifiers for joining
    if "sample_id" not in pred_df.columns and pred_df.index.name:
        pred_df = pred_df.reset_index()

    vg_pred_path = os.path.join(output_dir, "vg_model_predictions.parquet")
    direct_pred_path = os.path.join(output_dir, "direct_model_predictions.parquet")

    # Save VG predictions
    vg_cols = [
        "sample_id",
        "source",
        "rosetta_level",
        "depth_cm",
        "theta",
        "log10_suction_cm",
        "log10_suction_pred_vg",
        "suction_cm_pred_vg",
    ]
    vg_cols = [c for c in vg_cols if c in pred_df.columns]
    pred_df[vg_cols].to_parquet(vg_pred_path)
    print(f"   Saved VG predictions: {vg_pred_path}")

    # Save direct predictions
    direct_cols = [
        "sample_id",
        "source",
        "rosetta_level",
        "depth_cm",
        "theta",
        "log10_suction_cm",
        "log10_suction_pred_direct",
        "suction_cm_pred_direct",
    ]
    direct_cols = [c for c in direct_cols if c in pred_df.columns]
    pred_df[direct_cols].to_parquet(direct_pred_path)
    print(f"   Saved direct predictions: {direct_pred_path}")

    return results


def run_comparison(
    vg_table_path: str,
    obs_table_path: str,
    output_dir: str,
    test_size: float = 0.2,
    n_estimators: int = 200,
    random_state: int = 42,
    include_depth: bool = True,
) -> Dict:
    """
    Run full comparison between VG parameter and direct observation approaches.

    Parameters
    ----------
    vg_table_path : str
        Path to VG parameter training table.
    obs_table_path : str
        Path to observation-level training table.
    output_dir : str
        Directory for output files.
    test_size : float
        Fraction of sites for testing.
    n_estimators : int
        Number of trees in random forest.
    random_state : int
        Random seed.
    include_depth : bool
        Whether to include depth as a feature.

    Returns
    -------
    dict
        Comparison results.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("VG Parameter vs Direct Observation Approach Comparison")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    vg_df = load_vg_table(vg_table_path)
    obs_df = load_obs_table(obs_table_path)
    print(f"   VG table: {len(vg_df)} samples")
    print(f"   Obs table: {len(obs_df)} observations")

    # Get common feature columns
    vg_features = get_feature_columns(vg_df, include_depth=include_depth)
    obs_features = get_feature_columns(obs_df, include_depth=include_depth)
    common_features = sorted(set(vg_features) & set(obs_features))
    print(f"   Common features: {len(common_features)}")

    # Split data by site
    print("\n2. Splitting data by site...")
    vg_train, vg_test = grouped_train_test_split(
        vg_df, "sample_id", test_size, random_state
    )
    obs_train, obs_test = grouped_train_test_split(
        obs_df, "sample_id", test_size, random_state
    )

    # Get test site IDs from VG split
    if "sample_id" in vg_test.columns:
        test_sites = set(vg_test["sample_id"].apply(extract_site_id))
    else:
        test_sites = set(vg_test.index.to_series().apply(extract_site_id))

    print(f"   VG train: {len(vg_train)}, test: {len(vg_test)} samples")
    print(f"   Obs train: {len(obs_train)}, test: {len(obs_test)} observations")
    print(f"   Test sites: {len(test_sites)}")

    # Ensure obs test uses same sites as VG test
    if "sample_id" in obs_test.columns:
        obs_test_sites = obs_test["sample_id"].apply(extract_site_id)
    else:
        obs_test = obs_test.reset_index()
        obs_test_sites = obs_test["obs_id"].apply(extract_site_id)

    obs_test = obs_test[obs_test_sites.isin(test_sites)]
    print(f"   Aligned obs test: {len(obs_test)} observations")

    # Drop rows with NaN in features
    vg_train = vg_train.dropna(subset=common_features)
    obs_train = obs_train.dropna(subset=common_features + ["theta", "log10_suction_cm"])
    obs_test = obs_test.dropna(subset=common_features + ["theta", "log10_suction_cm"])

    # Train models
    print("\n3. Training VG parameter models...")
    vg_models, vg_preprocessor = train_vg_model(
        vg_train, common_features, n_estimators, random_state
    )

    print("\n4. Training direct model...")
    direct_model, direct_preprocessor = train_direct_model(
        obs_train, common_features, n_estimators, random_state
    )

    # Generate predictions on test observations
    print("\n5. Generating predictions...")
    y_true = obs_test["log10_suction_cm"].values
    theta = obs_test["theta"].values
    depth = (
        obs_test["rosetta_level"].values
        if "rosetta_level" in obs_test.columns
        else np.zeros(len(obs_test))
    )

    y_pred_vg = predict_vg_approach(
        vg_models, obs_test, common_features, preprocessor=vg_preprocessor
    )
    y_pred_direct = predict_direct_approach(
        direct_model, obs_test, common_features, preprocessor=direct_preprocessor
    )

    # Compute overall metrics
    print("\n6. Computing metrics...")
    vg_metrics = compute_metrics(y_true, y_pred_vg)
    direct_metrics = compute_metrics(y_true, y_pred_direct)

    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    print(f"{'Metric':<15} {'VG Approach':<15} {'Direct Approach':<15}")
    print("-" * 45)
    print(f"{'RMSE':<15} {vg_metrics['rmse']:<15.4f} {direct_metrics['rmse']:<15.4f}")
    print(f"{'MAE':<15} {vg_metrics['mae']:<15.4f} {direct_metrics['mae']:<15.4f}")
    print(f"{'R²':<15} {vg_metrics['r2']:<15.4f} {direct_metrics['r2']:<15.4f}")
    print(f"{'Bias':<15} {vg_metrics['bias']:<15.4f} {direct_metrics['bias']:<15.4f}")
    print(f"{'N':<15} {vg_metrics['n']:<15} {direct_metrics['n']:<15}")

    # Stratified metrics
    print("\n" + "=" * 60)
    print("STRATIFIED RESULTS")
    print("=" * 60)
    stratified = compute_stratified_metrics(
        y_true, y_pred_vg, y_pred_direct, theta, depth
    )
    print(stratified.to_string(index=False))

    # Monotonicity check
    print("\n" + "=" * 60)
    print("MONOTONICITY CHECK")
    print("=" * 60)
    if "sample_id" in obs_test.columns:
        sample_ids = obs_test["sample_id"].values
    else:
        sample_ids = obs_test["obs_id"].apply(extract_site_id).values

    mono_vg = check_monotonicity(y_pred_vg, theta, sample_ids)
    mono_direct = check_monotonicity(y_pred_direct, theta, sample_ids)
    print(
        f"VG violation rate: {mono_vg['violation_rate']:.1%} ({mono_vg['violations']}/{mono_vg['total_samples']} samples)"
    )
    print(
        f"Direct violation rate: {mono_direct['violation_rate']:.1%} ({mono_direct['violations']}/{mono_direct['total_samples']} samples)"
    )

    # Generate plots
    print("\n7. Generating plots...")
    plot_comparison(
        y_true,
        y_pred_vg,
        y_pred_direct,
        theta,
        os.path.join(output_dir, "comparison_scatter.png"),
    )

    plot_retention_curves(
        obs_test,
        y_pred_vg,
        y_pred_direct,
        os.path.join(output_dir, "retention_curves.png"),
    )

    # Save results
    results = {
        "vg_metrics": vg_metrics,
        "direct_metrics": direct_metrics,
        "stratified": stratified.to_dict("records"),
        "monotonicity": {
            "vg": mono_vg,
            "direct": mono_direct,
        },
        "config": {
            "test_size": test_size,
            "n_estimators": n_estimators,
            "random_state": random_state,
            "n_features": len(common_features),
            "include_depth": include_depth,
        },
    }

    results_path = os.path.join(output_dir, "comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to {results_path}")

    # Save stratified results
    stratified.to_csv(os.path.join(output_dir, "stratified_metrics.csv"), index=False)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare VG parameter vs direct observation approaches for soil water potential prediction."
    )
    parser.add_argument(
        "--systematic",
        action="store_true",
        help="Run systematic comparison of multiple VG model configurations.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of sites for testing (default: 0.2).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=250,
        help="Number of trees in random forest (default: 250).",
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed (default: 42)."
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Compute metrics on linear scale (cm H₂O) instead of log10 scale.",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        help='Comma-separated list of experiment names to run (e.g., "direct_base_data,vg_base_data_no_soils"). '
        "If not specified, runs all default experiments.",
    )
    parser.add_argument(
        "--drop-source-missing-features",
        action="store_true",
        default=True,
        dest="drop_blocking_features",
        help="Remove features that are 100%% missing for any source (default: ON). "
        "This prevents silent data dropping from sources with different feature availability.",
    )
    parser.add_argument(
        "--no-drop-source-missing-features",
        action="store_false",
        dest="drop_blocking_features",
        help="Keep all features and use imputation only. "
        "Warning: may cause complete-case analysis to drop data from some sources.",
    )
    args = parser.parse_args()

    home = os.path.expanduser("~")
    data_root = os.path.join(
        home, "data", "IrrigationGIS", "soils", "swapstress", "training"
    )

    # Paths to training tables
    vg_table = os.path.join(data_root, "unified_training_emb_250m.parquet")
    obs_table = os.path.join(data_root, "obs_level_training_emb_250m.parquet")

    # Check files exist
    if not os.path.exists(vg_table):
        print(f"VG table not found: {vg_table}")
        print("Run build_training_table.py with build_all_sources=True first")
        exit(1)

    if not os.path.exists(obs_table):
        print(f"Obs table not found: {obs_table}")
        print("Run build_training_table.py with build_observation_level=True first")
        exit(1)

    if args.systematic:
        # Run systematic comparison of multiple VG configurations
        output_dir = os.path.join(data_root, "systematic_comparison")

        # Filter experiments if --experiments specified
        vg_experiments = DEFAULT_VG_EXPERIMENTS
        direct_experiments = DEFAULT_DIRECT_EXPERIMENTS

        if args.experiments:
            requested = [e.strip() for e in args.experiments.split(",")]
            vg_experiments = [e for e in DEFAULT_VG_EXPERIMENTS if e.name in requested]
            direct_experiments = [
                e for e in DEFAULT_DIRECT_EXPERIMENTS if e.name in requested
            ]

            found_names = [e.name for e in vg_experiments] + [
                e.name for e in direct_experiments
            ]
            missing = set(requested) - set(found_names)
            if missing:
                print(f"Warning: experiments not found: {missing}")
                print(
                    f"Available VG experiments: {[e.name for e in DEFAULT_VG_EXPERIMENTS]}"
                )
                print(
                    f"Available direct experiments: {[e.name for e in DEFAULT_DIRECT_EXPERIMENTS]}"
                )

            if not vg_experiments and not direct_experiments:
                print("Error: No valid experiments selected")
                exit(1)

            print(
                f"Running selected experiments: VG={[e.name for e in vg_experiments]}, "
                f"Direct={[e.name for e in direct_experiments]}"
            )

        results = run_systematic_comparison(
            vg_table_path=vg_table,
            obs_table_path=obs_table,
            output_dir=output_dir,
            experiments=vg_experiments if vg_experiments else None,
            direct_experiments=direct_experiments if direct_experiments else None,
            test_size=args.test_size,
            random_state=args.random_state,
            linear=args.linear,
            drop_blocking_features=args.drop_blocking_features,
        )
    else:
        # Run original single comparison
        output_dir = os.path.join(data_root, "comparison_results")
        results = run_comparison(
            vg_table_path=vg_table,
            obs_table_path=obs_table,
            output_dir=output_dir,
            test_size=args.test_size,
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            include_depth=True,
        )

# ========================= EOF ====================================================================
