"""
Unified training table builder for direct soil water potential prediction.

Combines Earth Engine features with observation-level (theta, suction) data
from multiple sources into a single training table.

Output schema:
    - obs_id: Unique observation ID ({source}_{original_id}_{depth}_{obs_idx})
    - sample_id: Sample identifier for grouping
    - source: Data source name
    - rosetta_level: Depth mapped to Rosetta levels 1-7
    - theta: Volumetric water content (0-1) - INPUT FEATURE
    - log10_suction_cm: log10(suction in cm H2O) - TARGET
    - [EE features]: All extracted geospatial features

Usage:
    from map.data.build_training_table import build_unified_table
    df = build_unified_table(
        ['gshp', 'ncss'],
        output_path='obs_training.parquet',
    )
"""

import argparse
import os
import json
from glob import glob
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from map.data.source_registry import (
    get_source,
    DataSource,
    DataPaths,
    TRAINING_TABLE_DROP_COLS,
    VALID_SCALES,
)
from retention_curve.depth_utils import depth_to_rosetta_level

# Physical limits for data validation
# These are applied during training table construction as a final safety check
SUCTION_CM_MAX = (
    1e6  # cm - max ~10^6 cm = 100 MPa, beyond any realistic soil measurement
)
SUCTION_CM_MIN = 1e-3  # cm - minimum positive value for log transform safety
THETA_MIN = 0.0
THETA_MAX = 1.0


def standardize_observation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names for observation data.

    Handles variations in column naming across different preprocessed sources.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with observation data.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized column names.
    """
    rename_map = {
        "suction": "suction_cm",
        "depth": "depth_cm",
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


def load_observations_from_csv(
    csv_path: str,
    source: DataSource,
) -> pd.DataFrame:
    """
    Load raw (theta, suction_cm) observations from a preprocessed CSV file.

    Parameters
    ----------
    csv_path : str
        Path to preprocessed CSV file.
    source : DataSource
        Source configuration.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [index_col, depth_cm, rosetta_level, theta, suction_cm]
        One row per observation.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # Standardize column names
    df = standardize_observation_columns(df)

    # Required columns
    if "theta" not in df.columns or "suction_cm" not in df.columns:
        return pd.DataFrame()

    # Extract identifier from filename
    identifier = os.path.splitext(os.path.basename(csv_path))[0]

    # Add index column
    df[source.index_col] = str(identifier)

    # Ensure depth_cm exists
    if "depth_cm" not in df.columns:
        df["depth_cm"] = 0.0

    # Add rosetta_level
    df["rosetta_level"] = df["depth_cm"].apply(depth_to_rosetta_level)

    # Filter valid observations with explicit bounds
    df = df[df["theta"].notna() & df["suction_cm"].notna()]
    # Theta must be in [0, 1]
    df = df[(df["theta"] >= THETA_MIN) & (df["theta"] <= THETA_MAX)]
    # Suction must be positive and within physical bounds for log transform
    df = df[(df["suction_cm"] >= SUCTION_CM_MIN) & (df["suction_cm"] <= SUCTION_CM_MAX)]

    # Keep only needed columns
    keep_cols = [source.index_col, "depth_cm", "rosetta_level", "theta", "suction_cm"]
    extra_cols = [
        c
        for c in df.columns
        if c not in keep_cols
        and c in ["sand_tot_psa", "silt_tot_psa", "clay_tot_psa", "db_od"]
    ]
    keep_cols = keep_cols + extra_cols

    return df[[c for c in keep_cols if c in df.columns]].copy()


def load_observations_from_json(
    json_path: str,
    source: DataSource,
) -> pd.DataFrame:
    """
    Load raw (theta, suction_cm) observations from a fitted JSON file.

    Extracts data from res['data']['theta'] and res['data']['suction'] arrays.

    Parameters
    ----------
    json_path : str
        Path to fitted JSON file.
    source : DataSource
        Source configuration.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [index_col, depth_cm, rosetta_level, theta, suction_cm]
        One row per observation.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
            meta = data.pop("metadata", {})
    except Exception:
        return pd.DataFrame()

    rows = []
    for depth_str, res in data.items():
        if not isinstance(res, dict):
            continue
        if res.get("status") != "Success":
            continue

        try:
            depth_cm = float(depth_str)
        except (TypeError, ValueError):
            continue

        # Get raw observation arrays
        obs_data = res.get("data", {})
        theta_arr = obs_data.get("theta", [])
        suction_arr = obs_data.get("suction", obs_data.get("suction_cm", []))

        if not theta_arr or not suction_arr or len(theta_arr) != len(suction_arr):
            continue

        # Get identifier
        depth_meta = meta.get(depth_str, {})
        identifier = (
            depth_meta.get(source.index_col)
            or depth_meta.get("station")
            or depth_meta.get("profile_id")
            or os.path.splitext(os.path.basename(json_path))[0]
        )

        rosetta_level = depth_to_rosetta_level(depth_cm)

        for theta, suction in zip(theta_arr, suction_arr):
            # Filter valid observations with explicit bounds
            if (
                theta is not None
                and suction is not None
                and THETA_MIN <= float(theta) <= THETA_MAX
                and SUCTION_CM_MIN <= float(suction) <= SUCTION_CM_MAX
            ):
                rows.append(
                    {
                        source.index_col: str(identifier),
                        "depth_cm": depth_cm,
                        "rosetta_level": rosetta_level,
                        "theta": float(theta),
                        "suction_cm": float(suction),
                    }
                )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def load_embeddings(
    embeddings_dir: str,
    index_col: str,
) -> pd.DataFrame:
    """
    Load embeddings from per-site parquet files.

    Parameters
    ----------
    embeddings_dir : str
        Directory containing embedding parquet files.
    index_col : str
        Name of index column.

    Returns
    -------
    pd.DataFrame
        DataFrame with embeddings, indexed by site/station ID.
    """
    if not embeddings_dir or not os.path.isdir(embeddings_dir):
        return pd.DataFrame()

    emb_files = glob(os.path.join(embeddings_dir, "*.parquet"))
    if not emb_files:
        return pd.DataFrame()

    rows = {}
    for fp in tqdm(emb_files, desc="Loading embeddings", leave=False):
        identifier = os.path.splitext(os.path.basename(fp))[0]
        try:
            df = pd.read_parquet(fp)
            if len(df) >= 1:
                rows[str(identifier)] = df.iloc[0]
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    emb_df = pd.DataFrame.from_dict(rows, orient="index")
    emb_df.index.name = index_col
    return emb_df


def load_observations_for_source(
    source: DataSource,
    data_root: str,
    fit_method: str = "bayes",
    prefer_preprocessed: bool = True,
) -> pd.DataFrame:
    """
    Load all raw observations for a source from preprocessed CSVs or fitted JSONs.

    Parameters
    ----------
    source : DataSource
        Source configuration.
    data_root : str
        Root data directory.
    fit_method : str
        Fitting method subdirectory for JSON files.
    prefer_preprocessed : bool
        If True, prefer preprocessed CSVs over JSON data arrays.

    Returns
    -------
    pd.DataFrame
        DataFrame with all observations for the source.
        Columns: [index_col, depth_cm, rosetta_level, theta, suction_cm]
    """
    paths = DataPaths(data_root, source)
    frames = []
    loaded_ids = set()

    # Try preprocessed CSVs first if preferred
    if (
        prefer_preprocessed
        and paths.preprocessed_dir
        and os.path.isdir(paths.preprocessed_dir)
    ):
        csv_files = glob(os.path.join(paths.preprocessed_dir, "*.csv"))
        for csv_path in tqdm(
            csv_files, desc=f"Loading {source.name} CSVs", leave=False
        ):
            df = load_observations_from_csv(csv_path, source)
            if not df.empty:
                frames.append(df)
                identifier = os.path.splitext(os.path.basename(csv_path))[0]
                loaded_ids.add(str(identifier))

    # Fall back to JSON for any missing profiles
    if paths.fit_results_dir and os.path.isdir(paths.fit_results_dir):
        json_subdir = os.path.join(paths.fit_results_dir, fit_method)
        if os.path.isdir(json_subdir):
            json_files = glob(os.path.join(json_subdir, "*.json"))
            for json_path in tqdm(
                json_files, desc=f"Loading {source.name} JSONs", leave=False
            ):
                identifier = os.path.splitext(os.path.basename(json_path))[0]
                # Skip if already loaded from CSV
                if str(identifier) in loaded_ids:
                    continue
                df = load_observations_from_json(json_path, source)
                if not df.empty:
                    frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    return combined


def load_source_observations(
    source: DataSource,
    data_root: str,
    fit_method: str = "bayes",
    prefer_preprocessed: bool = True,
    include_embeddings: bool = False,
    scale: str = "9km_global",
) -> pd.DataFrame:
    """
    Load and join EE features with raw observations for a single source.

    This is the observation-level equivalent of load_source_data().

    Parameters
    ----------
    source : DataSource
        Source configuration.
    data_root : str
        Root data directory.
    fit_method : str
        Fitting method for JSON files.
    prefer_preprocessed : bool
        Prefer preprocessed CSVs over JSON data.
    include_embeddings : bool
        Whether to include embeddings.
    scale : str
        Resolution scale passed to DataPaths.

    Returns
    -------
    pd.DataFrame
        Combined features and observations with standardized columns.
        Each row is a single (theta, suction) observation with all EE features.
    """
    if scale != "250m":
        include_embeddings = False
    paths = DataPaths(data_root, source, scale=scale)

    # Load EE features
    ee_table = paths.ee_table
    if not os.path.exists(ee_table):
        raise FileNotFoundError(f"EE features not found: {ee_table}")

    ee_df = pd.read_parquet(ee_table)

    # Reset index if needed
    if ee_df.index.name is not None:
        ee_df = ee_df.reset_index()

    ee_df[source.index_col] = ee_df[source.index_col].astype(str)

    # Normalize station names for station-based sources
    if source.index_col == "station":
        ee_df[source.index_col] = (
            ee_df[source.index_col].str.lower().str.replace("_", "-")
        )

    # Note: ReESH site_id normalization happens below when processing observations
    # EE table already uses underscores (e.g., 'US_CDM'), observations will be normalized to match

    # Drop VG columns from EE data (not needed for observation-level)
    _vg_cols = [
        "theta_r",
        "theta_s",
        "alpha",
        "n",
        "Ks",
        "log10_alpha",
        "log10_n",
        "log10_Ks",
    ]
    vg_cols_to_drop = [c for c in _vg_cols if c in ee_df.columns]
    if vg_cols_to_drop:
        ee_df = ee_df.drop(columns=vg_cols_to_drop)

    # Load observations
    obs_df = load_observations_for_source(
        source,
        data_root,
        fit_method=fit_method,
        prefer_preprocessed=prefer_preprocessed,
    )

    if obs_df.empty:
        raise ValueError(f"No observations found for source {source.name}")

    # Normalize station names in observations
    if source.index_col == "station" and source.index_col in obs_df.columns:
        obs_df[source.index_col] = (
            obs_df[source.index_col].str.lower().str.replace("_", "-")
        )

    # Handle ReESH: extract site portion from full identifier for EE join
    # CSV identifiers are like 'US-CDM_1', 'IN-Martell_ControlDH'
    # EE table has site-only like 'US_CDM', 'IN_Martell'
    if source.name == "reesh" and source.index_col == "site_id":
        # Extract site portion: take first part before underscore, normalize hyphens.
        # Case-fold both sides because the raw CSV Site column (used for preprocessed
        # filenames) and the shapefile site_id (used for EE extraction) sometimes
        # differ in capitalization (e.g. US-HA1 vs US_Ha1).
        def extract_reesh_site(identifier):
            parts = str(identifier).split("_")
            site = parts[0]
            return site.replace("-", "_").upper()

        obs_df[source.index_col] = obs_df[source.index_col].apply(extract_reesh_site)
        ee_df[source.index_col] = ee_df[source.index_col].str.upper()

    # Merge observations with EE features (replicates features for each observation)
    merged = obs_df.merge(
        ee_df.drop_duplicates(subset=source.index_col),
        on=source.index_col,
        how="left",
    )

    # Add log10_suction_cm target
    merged["log10_suction_cm"] = np.log10(merged["suction_cm"].clip(lower=1e-6))

    # Add embeddings if requested
    if include_embeddings:
        emb_dir = paths.embeddings_dir
        if emb_dir and os.path.isdir(emb_dir):
            emb_df = load_embeddings(emb_dir, source.index_col)
            if not emb_df.empty:
                emb_df = emb_df.reset_index()
                merged = merged.merge(emb_df, on=source.index_col, how="left")

    # Add source identifier
    merged["source"] = source.name

    # Create unique sample_id (profile+depth) and obs_id (profile+depth+idx)
    merged["sample_id"] = (
        source.name
        + "_"
        + merged[source.index_col].astype(str)
        + "_"
        + merged["depth_cm"].astype(str)
    )

    # Add observation index within each sample
    merged["obs_idx"] = merged.groupby("sample_id").cumcount()
    merged["obs_id"] = merged["sample_id"] + "_" + merged["obs_idx"].astype(str)
    merged = merged.drop(columns=["obs_idx"])

    return merged


def build_unified_table(
    sources: List[str],
    data_root: str,
    output_path: Optional[str] = None,
    fit_method: str = "bayes",
    include_embeddings: bool = False,
    prefer_preprocessed: bool = True,
    amsr_vod_path: Optional[str] = None,
    scale: str = "9km_global",
) -> pd.DataFrame:
    """
    Build a unified observation-level training table from multiple data sources.

    Each row is a single (theta, suction) observation with all EE features.

    Parameters
    ----------
    sources : list of str
        Source names to include (e.g., ['gshp', 'ncss', 'mt_mesonet']).
    data_root : str
        Root data directory.
    output_path : str, optional
        Path to save output parquet file.
    fit_method : str
        Fitting method for JSON files.
    include_embeddings : bool
        Whether to include embeddings.
    prefer_preprocessed : bool
        Prefer preprocessed CSVs over JSON data arrays.
    amsr_vod_path : str, optional
        Path to AMSR VOD climatology parquet (from amsr_extract.py).
    scale : str
        Resolution scale passed to DataPaths.

    Returns
    -------
    pd.DataFrame
        Unified training table: theta as input, log10_suction_cm as target.
    """
    if scale != "250m":
        include_embeddings = False

    frames = []

    for source_name in sources:
        source = get_source(source_name)
        print(f"Loading {source_name}...")

        try:
            df = load_source_observations(
                source,
                data_root,
                fit_method=fit_method,
                prefer_preprocessed=prefer_preprocessed,
                include_embeddings=include_embeddings,
                scale=scale,
            )
            print(f"  Loaded {len(df)} observations from {source_name}")
            frames.append(df)

        except (FileNotFoundError, ValueError) as e:
            print(f"  Warning: Skipping {source_name} - {e}")
            continue

    if not frames:
        raise ValueError("No data loaded from any source")

    combined = pd.concat(frames, ignore_index=True)

    # Set obs_id as index
    combined = combined.set_index("obs_id")

    # Drop rows with missing observations
    combined = combined.dropna(subset=["theta", "log10_suction_cm"])

    # Validate theta range
    invalid_theta = (combined["theta"] < 0) | (combined["theta"] > 1)
    if invalid_theta.any():
        print(
            f"  Warning: {invalid_theta.sum()} observations with invalid theta (outside 0-1)"
        )
        combined = combined[~invalid_theta]

    # Merge AMSR VOD climatology if provided
    if amsr_vod_path and os.path.exists(amsr_vod_path):
        vod_df = pd.read_parquet(amsr_vod_path)
        n_before = combined.shape[1]
        combined = combined.reset_index().merge(vod_df, on="sample_id", how="left")
        combined = combined.set_index("obs_id")
        n_added = combined.shape[1] - n_before
        print(f"  Merged {n_added} AMSR VOD columns from {amsr_vod_path}")

    # Report statistics
    print(
        f"\nCombined table: {len(combined)} observations, {combined.shape[1]} columns"
    )
    print(f"Sources: {combined['source'].value_counts().to_dict()}")
    print(f"Theta range: {combined['theta'].min():.3f} - {combined['theta'].max():.3f}")
    print(
        f"log10(suction_cm) range: {combined['log10_suction_cm'].min():.2f} - "
        f"{combined['log10_suction_cm'].max():.2f}"
    )
    if "rosetta_level" in combined.columns:
        level_counts = combined["rosetta_level"].value_counts().sort_index().to_dict()
        print(f"Rosetta levels: {level_counts}")

    # Drop metadata columns
    drop_cols = [
        c for c in TRAINING_TABLE_DROP_COLS + ["suction_cm"] if c in combined.columns
    ]
    combined = combined.drop(columns=drop_cols)

    # Reorder columns with important ones first
    priority_cols = [
        "source",
        "sample_id",
        "rosetta_level",
        "depth_cm",
        "theta",
        "log10_suction_cm",
    ]
    priority_cols = [c for c in priority_cols if c in combined.columns]
    other_cols = [c for c in combined.columns if c not in priority_cols]
    combined = combined[priority_cols + other_cols]

    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        combined.to_parquet(output_path)
        print(f"Saved to {output_path}")

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build unified observation-level training table.",
    )
    parser.add_argument(
        "--sources",
        type=str,
        nargs="+",
        default=["gshp", "ncss", "mt_mesonet", "reesh", "lacadian"],
        help="Source names to include.",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default="9km_global",
        choices=VALID_SCALES,
        help="Resolution scale (default: 9km_global). 250m is historical only.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/nas/soils",
        help="Root data directory (default: /nas/soils).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output parquet path (auto-generated if omitted).",
    )
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="Include embeddings (250m only).",
    )
    parser.add_argument(
        "--fit-method",
        type=str,
        default="bayes",
        help="Fitting method for JSON files (default: bayes).",
    )
    args = parser.parse_args()

    output_path_ = args.output
    if output_path_ is None:
        suffix = f"_emb_{args.scale}" if args.embeddings else f"_{args.scale}"
        output_path_ = os.path.join(
            args.data_root,
            "swapstress",
            "training",
            f"obs_level_training{suffix}.parquet",
        )

    build_unified_table(
        sources=args.sources,
        data_root=args.data_root,
        output_path=output_path_,
        fit_method=args.fit_method,
        include_embeddings=args.embeddings,
        prefer_preprocessed=True,
        scale=args.scale,
    )

# ========================= EOF ====================================================================
