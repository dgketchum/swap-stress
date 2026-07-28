"""
Data loading, spatial splitting, and filtering for the direct suction task.

Shared between RF and NN trainers so that both model families operate on
identical data with identical holdout partitions.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from map.data.features import (
    filter_feature_groups,
    get_feature_columns,
)


# ---------------------------------------------------------------------------
# Spatial grouping
# ---------------------------------------------------------------------------


def assign_spatial_group(df: pd.DataFrame, resolution_m: float = 250) -> pd.Series:
    """Quantize lat/lon to grid cells for spatial grouping.

    Groups observations that share the same EE pixel so that all co-located
    profiles end up in the same train/test partition.

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
    resolution_m: float = 250,
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
    resolution_m : float
        Grid cell size in metres for spatial grouping.

    Returns
    -------
    tuple of (set, set)
        (train_groups, test_groups)
    """
    groups = assign_spatial_group(df, resolution_m=resolution_m)
    unique_groups = list(groups.dropna().unique())
    train_groups, test_groups = train_test_split(
        unique_groups,
        test_size=test_size,
        random_state=random_state,
    )
    return set(train_groups), set(test_groups)


def create_site_split_with_val(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    resolution_m: float = 250,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Create train/val/test split on spatial groups.

    First splits off the test set, then splits the remainder into train and
    val.  This gives NN models a validation set for early stopping while
    keeping the test set identical to what RF uses with the same
    ``test_size`` and ``random_state``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'lat' and 'lon'.
    test_size : float
        Fraction of spatial groups for final testing.
    val_size : float
        Fraction of the *remaining* groups for validation.
    random_state : int
        Random seed.
    resolution_m : float
        Grid cell size in metres.

    Returns
    -------
    tuple of (set, set, set)
        (train_groups, val_groups, test_groups)
    """
    groups = assign_spatial_group(df, resolution_m=resolution_m)
    unique_groups = list(groups.dropna().unique())

    # First split: train_dev vs test (same as create_site_split)
    train_dev_groups, test_groups = train_test_split(
        unique_groups,
        test_size=test_size,
        random_state=random_state,
    )

    # Second split: train vs val within train_dev
    train_groups, val_groups = train_test_split(
        train_dev_groups,
        test_size=val_size,
        random_state=random_state,
    )

    return set(train_groups), set(val_groups), set(test_groups)


def apply_site_split(
    df: pd.DataFrame,
    train_sites: Set[str],
    test_sites: Set[str],
    group_col: str = "sample_id",
    resolution_m: float = 250,
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
    resolution_m : float
        Grid cell size in metres for spatial grouping.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (train_df, test_df)
    """
    groups = assign_spatial_group(df, resolution_m=resolution_m)
    train_mask = groups.isin(train_sites)
    test_mask = groups.isin(test_sites)
    return df[train_mask.values].copy(), df[test_mask.values].copy()


def apply_site_split_three(
    df: pd.DataFrame,
    train_sites: Set[str],
    val_sites: Set[str],
    test_sites: Set[str],
    resolution_m: float = 250,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply pre-computed three-way spatial split."""
    groups = assign_spatial_group(df, resolution_m=resolution_m)
    return (
        df[groups.isin(train_sites).values].copy(),
        df[groups.isin(val_sites).values].copy(),
        df[groups.isin(test_sites).values].copy(),
    )


# ---------------------------------------------------------------------------
# Split manifest I/O
# ---------------------------------------------------------------------------


def write_split_manifest(
    path: str,
    train_groups: Set[str],
    test_groups: Set[str],
    val_groups: Set[str] | None = None,
    random_state: int = 42,
    resolution_m: float = 250,
) -> str:
    """Save a spatial split manifest to JSON.

    Parameters
    ----------
    path : str
        Output file path.
    train_groups, test_groups, val_groups : set of str
        Spatial group labels.
    random_state : int
        Seed used to create the split.
    resolution_m : float
        Grid cell size.

    Returns
    -------
    str
        Path written.
    """
    doc = {
        "random_state": random_state,
        "resolution_m": resolution_m,
        "n_train": len(train_groups),
        "n_test": len(test_groups),
        "train_groups": sorted(train_groups),
        "test_groups": sorted(test_groups),
    }
    if val_groups is not None:
        doc["n_val"] = len(val_groups)
        doc["val_groups"] = sorted(val_groups)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"Wrote split manifest to {path}")
    return path


def read_split_manifest(
    path: str,
) -> Dict:
    """Load a spatial split manifest.

    Returns
    -------
    dict
        Keys: train_groups (set), test_groups (set), val_groups (set or None),
        random_state, resolution_m.
    """
    with open(path) as f:
        doc = json.load(f)
    result = {
        "train_groups": set(doc["train_groups"]),
        "test_groups": set(doc["test_groups"]),
        "val_groups": set(doc["val_groups"]) if "val_groups" in doc else None,
        "random_state": doc["random_state"],
        "resolution_m": doc["resolution_m"],
    }
    return result


def upgrade_legacy_split_manifest(
    path: str,
    manifest: Dict,
    val_size: float,
    random_state: int,
) -> Dict:
    """Upgrade a legacy two-way manifest to include validation groups.

    Preserves the existing test groups and splits the old train groups into
    train/val using the provided ``val_size`` and ``random_state``.

    Parameters
    ----------
    path : str
        Manifest path to rewrite in place.
    manifest : dict
        Parsed manifest from ``read_split_manifest``.
    val_size : float
        Fraction of non-test groups to allocate to validation.
    random_state : int
        Fallback seed if the manifest does not record one.

    Returns
    -------
    dict
        Upgraded manifest in the same shape returned by
        ``read_split_manifest()``.
    """
    if manifest.get("val_groups") is not None:
        return manifest

    legacy_train_groups = sorted(manifest["train_groups"])
    if len(legacy_train_groups) < 2:
        raise ValueError(
            "Cannot upgrade legacy split manifest with fewer than 2 train groups"
        )

    split_seed = manifest.get("random_state", random_state)
    train_groups, val_groups = train_test_split(
        legacy_train_groups,
        test_size=val_size,
        random_state=split_seed,
    )

    upgraded = {
        "train_groups": set(train_groups),
        "val_groups": set(val_groups),
        "test_groups": set(manifest["test_groups"]),
        "random_state": split_seed,
        "resolution_m": manifest["resolution_m"],
    }

    write_split_manifest(
        path,
        train_groups=upgraded["train_groups"],
        test_groups=upgraded["test_groups"],
        val_groups=upgraded["val_groups"],
        random_state=upgraded["random_state"],
        resolution_m=upgraded["resolution_m"],
    )
    print(f"Upgraded legacy two-way split manifest to three-way: {path}")
    return upgraded


# ---------------------------------------------------------------------------
# MGRS-tile hash-based spatial splitting
# ---------------------------------------------------------------------------


def _tile_to_fold(tile_id: str, n_folds: int) -> int:
    """Map a tile ID to a fold index via MD5 hash."""
    h = hashlib.md5(str(tile_id).encode()).hexdigest()
    return int(h, 16) % n_folds


def assign_mgrs_fold(
    df: pd.DataFrame,
    n_folds: int = 5,
    holdout_col: str = "MGRS_TILE",
) -> pd.Series:
    """Assign each row to a fold based on hash of its spatial-block ID.

    Uses MD5 for uniform, deterministic, session-independent assignment.
    The same tile always maps to the same fold regardless of what other
    data is present.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *holdout_col*.
    n_folds : int
        Number of folds.
    holdout_col : str
        Column containing spatial unit IDs.

    Returns
    -------
    pd.Series
        Integer fold index (0 to n_folds-1).  NaN where holdout_col is missing.
    """
    result = pd.Series(np.nan, index=df.index)
    valid = df[holdout_col].notna()
    result[valid] = df.loc[valid, holdout_col].map(lambda t: _tile_to_fold(t, n_folds))
    return result


def create_mgrs_split(
    df: pd.DataFrame,
    n_folds: int = 5,
    test_fold: int = 0,
    holdout_col: str = "MGRS_TILE",
) -> Tuple[Set[str], Set[str]]:
    """Create train/test split by hashing spatial-block IDs to folds.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *holdout_col*.
    n_folds : int
        Number of folds.
    test_fold : int
        Which fold (0-based) to hold out as test.
    holdout_col : str
        Column with spatial unit IDs.

    Returns
    -------
    tuple of (set, set)
        (train_tiles, test_tiles)
    """
    tiles = df[holdout_col].dropna().unique()
    test_tiles = {t for t in tiles if _tile_to_fold(t, n_folds) == test_fold}
    train_tiles = set(tiles) - test_tiles
    return train_tiles, test_tiles


def create_mgrs_split_with_val(
    df: pd.DataFrame,
    n_folds: int = 5,
    test_fold: int = 0,
    val_fold: int | None = None,
    holdout_col: str = "MGRS_TILE",
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Three-way split from hash-based fold assignment.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *holdout_col*.
    n_folds : int
        Number of folds.
    test_fold : int
        Fold held out for testing.
    val_fold : int or None
        Fold held out for validation.  Defaults to ``(test_fold + 1) % n_folds``.
    holdout_col : str
        Column with spatial unit IDs.

    Returns
    -------
    tuple of (set, set, set)
        (train_tiles, val_tiles, test_tiles)
    """
    if val_fold is None:
        val_fold = (test_fold + 1) % n_folds
    tiles = df[holdout_col].dropna().unique()
    test_tiles: Set[str] = set()
    val_tiles: Set[str] = set()
    train_tiles: Set[str] = set()
    for t in tiles:
        f = _tile_to_fold(t, n_folds)
        if f == test_fold:
            test_tiles.add(t)
        elif f == val_fold:
            val_tiles.add(t)
        else:
            train_tiles.add(t)
    return train_tiles, val_tiles, test_tiles


def apply_mgrs_split(
    df: pd.DataFrame,
    train_tiles: Set[str],
    test_tiles: Set[str],
    holdout_col: str = "MGRS_TILE",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply pre-computed MGRS tile split."""
    return (
        df[df[holdout_col].isin(train_tiles)].copy(),
        df[df[holdout_col].isin(test_tiles)].copy(),
    )


def apply_mgrs_split_three(
    df: pd.DataFrame,
    train_tiles: Set[str],
    val_tiles: Set[str],
    test_tiles: Set[str],
    holdout_col: str = "MGRS_TILE",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply pre-computed three-way MGRS tile split."""
    return (
        df[df[holdout_col].isin(train_tiles)].copy(),
        df[df[holdout_col].isin(val_tiles)].copy(),
        df[df[holdout_col].isin(test_tiles)].copy(),
    )


# ---------------------------------------------------------------------------
# K-fold manifest I/O
# ---------------------------------------------------------------------------


def write_kfold_manifest(
    path: str,
    tile_to_fold: Dict[str, int],
    n_folds: int,
    holdout_col: str = "MGRS_TILE",
) -> str:
    """Save a k-fold spatial split manifest to JSON.

    Parameters
    ----------
    path : str
        Output file path.
    tile_to_fold : dict
        Mapping of tile ID → fold index.
    n_folds : int
        Number of folds.
    holdout_col : str
        Column name used for grouping.

    Returns
    -------
    str
        Path written.
    """
    doc = {
        "split_type": "kfold_mgrs",
        "n_folds": n_folds,
        "holdout_col": holdout_col,
        "n_tiles": len(tile_to_fold),
        "tile_to_fold": dict(sorted(tile_to_fold.items())),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"Wrote k-fold manifest to {path}")
    return path


def read_kfold_manifest(path: str) -> Dict:
    """Load a k-fold spatial split manifest.

    Returns
    -------
    dict
        Keys: tile_to_fold (dict), n_folds (int), holdout_col (str).
    """
    with open(path) as f:
        doc = json.load(f)
    return {
        "tile_to_fold": doc["tile_to_fold"],
        "n_folds": doc["n_folds"],
        "holdout_col": doc.get("holdout_col", "MGRS_TILE"),
    }


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


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
    """Audit dataset for feature missingness by source.

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


def prepare_direct_data(
    obs_table_path: str,
    output_dir: str,
    exclude_groups: Optional[List[str]] = None,
    drop_blocking_features: bool = True,
    resolution_m: float = 250,
    test_size: float = 0.2,
    val_size: float | None = None,
    random_state: int = 42,
    split_manifest: str | None = None,
    holdout_col: str | None = None,
    n_folds: int = 5,
    test_fold: int = 0,
) -> Dict:
    """Load data, discover features, build spatial split — shared by RF and NN.

    Parameters
    ----------
    obs_table_path : str
        Path to observation-level parquet.
    output_dir : str
        For dataset audit artifacts.
    exclude_groups : list of str, optional
        Feature groups to exclude.
    drop_blocking_features : bool
        Remove features 100% missing for any source.
    resolution_m : float
        Spatial grouping grid cell size (legacy coordinate-based split).
    test_size : float
        Fraction of groups for test (legacy coordinate-based split).
    val_size : float or None
        If not None, fraction of non-test groups for validation (NN).
    random_state : int
        Random seed (legacy coordinate-based split).
    split_manifest : str or None
        Path to existing split manifest JSON.  If provided, the split is
        loaded instead of created.
    holdout_col : str or None
        Column for spatial holdout (e.g. ``"MGRS_TILE"``).  When set, uses
        hash-based deterministic fold assignment instead of coordinate
        quantization.  None preserves legacy behavior.
    n_folds : int
        Number of folds for hash-based splitting (default 5).
    test_fold : int
        Which fold to hold out as test (0-based).

    Returns
    -------
    dict with keys:
        df, feature_cols, all_features, train_df, test_df,
        train_sites, test_sites, val_df (if val_size), val_sites (if val_size),
        holdout_col, n_folds, test_fold
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Loading observation table...")
    df = pd.read_parquet(obs_table_path)
    if df.index.name:
        df = df.reset_index()
    print(f"  {len(df)} observations, {df.shape[1]} columns")

    df = filter_complete_samples(df)

    feature_cols = get_feature_columns(df, include_depth=True, include_embeddings=False)

    if drop_blocking_features:
        audit = audit_dataset(df, feature_cols, output_dir)
        if audit["blocking_features"]:
            feature_cols = filter_blocking_features(
                feature_cols, audit["blocking_features"]
            )

    if exclude_groups:
        feature_cols = filter_feature_groups(feature_cols, exclude_groups)
        print(f"  After excluding {exclude_groups}: {len(feature_cols)} features")

    all_features = feature_cols + ["theta"]
    print(f"  Using {len(feature_cols)} features + theta")

    # --- Spatial split ---
    if holdout_col is not None:
        # Hash-based MGRS tile split
        if holdout_col not in df.columns:
            raise ValueError(
                f"holdout_col '{holdout_col}' not found in data. "
                f"Available columns: {sorted(df.columns[:20].tolist())}..."
            )
        n_missing = df[holdout_col].isna().sum()
        if n_missing:
            print(
                f"  Warning: {n_missing} rows missing {holdout_col}, "
                f"will be excluded from train/test"
            )

        print(
            f"Creating hash-based split on {holdout_col} "
            f"(fold {test_fold}/{n_folds})..."
        )
        if val_size is not None:
            train_sites, val_sites, test_sites = create_mgrs_split_with_val(
                df,
                n_folds=n_folds,
                test_fold=test_fold,
                holdout_col=holdout_col,
            )
        else:
            train_sites, test_sites = create_mgrs_split(
                df,
                n_folds=n_folds,
                test_fold=test_fold,
                holdout_col=holdout_col,
            )
            val_sites = None

        # Write kfold manifest
        manifest_path = os.path.join(output_dir, "spatial_split.json")
        tiles = df[holdout_col].dropna().unique()
        tile_to_fold = {t: _tile_to_fold(t, n_folds) for t in tiles}
        write_kfold_manifest(manifest_path, tile_to_fold, n_folds, holdout_col)

        # Apply split
        if val_sites is not None:
            train_df, val_df, test_df = apply_mgrs_split_three(
                df, train_sites, val_sites, test_sites, holdout_col=holdout_col
            )
        else:
            train_df, test_df = apply_mgrs_split(
                df, train_sites, test_sites, holdout_col=holdout_col
            )
            val_df = None

    elif split_manifest and os.path.exists(split_manifest):
        # Load existing legacy manifest
        print(f"Loading split manifest from {split_manifest}")
        manifest = read_split_manifest(split_manifest)
        if val_size is not None and manifest.get("val_groups") is None:
            manifest = upgrade_legacy_split_manifest(
                split_manifest,
                manifest,
                val_size=val_size,
                random_state=random_state,
            )
        train_sites = manifest["train_groups"]
        test_sites = manifest["test_groups"]
        val_sites = manifest.get("val_groups")

        if val_sites is not None:
            train_df, val_df, test_df = apply_site_split_three(
                df, train_sites, val_sites, test_sites, resolution_m=resolution_m
            )
        else:
            train_df, test_df = apply_site_split(
                df, train_sites, test_sites, "sample_id", resolution_m=resolution_m
            )
            val_df = None
    else:
        # Legacy coordinate-based split
        print("Creating spatial-group split...")
        if val_size is not None:
            train_sites, val_sites, test_sites = create_site_split_with_val(
                df,
                test_size=test_size,
                val_size=val_size,
                random_state=random_state,
                resolution_m=resolution_m,
            )
        else:
            train_sites, test_sites = create_site_split(
                df,
                "sample_id",
                test_size,
                random_state,
                resolution_m=resolution_m,
            )
            val_sites = None

        if val_sites is not None:
            train_df, val_df, test_df = apply_site_split_three(
                df, train_sites, val_sites, test_sites, resolution_m=resolution_m
            )
        else:
            train_df, test_df = apply_site_split(
                df, train_sites, test_sites, "sample_id", resolution_m=resolution_m
            )
            val_df = None

    # Drop rows missing theta or target
    train_df = train_df.dropna(subset=["theta", "log10_suction_cm"])
    test_df = test_df.dropna(subset=["theta", "log10_suction_cm"])
    if val_df is not None:
        val_df = val_df.dropna(subset=["theta", "log10_suction_cm"])

    print(f"  Train: {len(train_df)} obs from {len(train_sites)} spatial groups")
    if val_df is not None:
        print(f"  Val:   {len(val_df)} obs from {len(val_sites)} spatial groups")
    print(f"  Test:  {len(test_df)} obs from {len(test_sites)} spatial groups")

    if "source" in train_df.columns:
        print("  Train by source:", train_df["source"].value_counts().to_dict())
        print("  Test by source:", test_df["source"].value_counts().to_dict())

    result = {
        "df": df,
        "feature_cols": feature_cols,
        "all_features": all_features,
        "train_df": train_df,
        "test_df": test_df,
        "train_sites": train_sites,
        "test_sites": test_sites,
        "holdout_col": holdout_col,
        "n_folds": n_folds,
        "test_fold": test_fold,
    }
    if val_df is not None:
        result["val_df"] = val_df
        result["val_sites"] = val_sites

    return result
