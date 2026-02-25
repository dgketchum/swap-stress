"""
Feature group definitions and utilities for soil hydraulic parameter estimation.

Defines named groups of Earth Engine features (Landsat, Sentinel-1, SMAP, etc.)
and provides functions for filtering, classifying, and aggregating feature
importance by group.

Usage:
    from map.data.features import FEATURE_GROUPS, filter_feature_groups, classify_feature

    # Get feature columns excluding certain groups
    filtered = filter_feature_groups(feature_cols, exclude_groups=['embeddings', 'polaris'])

    # Classify a single feature
    group = classify_feature('B5_mean_gs')  # -> 'landsat_bands'

    # Aggregate importance scores by group
    group_importance = aggregate_importance_by_group(importance_dict)
"""

import re
from typing import Dict, List

from map.data import ee_feature_list

# Embedding column patterns
_EMBEDDING_PATTERNS = [
    re.compile(r"^embedding_\d+$"),
    re.compile(r"^e\d+$"),
    re.compile(r"^A\d+$"),
    re.compile(r"^b\d+$"),
    re.compile(r"^US_R3H3_"),
]

# SoilGrids depth-resolved features (e.g., clay_30-60cm_mean)
_SOILGRIDS_DEPTH_RE = re.compile(
    r"^(bdod|cec|cfvo|clay|sand|silt|nitrogen|phh2o|soc|ocd|ocs)_\d+-\d+cm_"
)

_LANDCOVER_FEATURES = {
    "c3s_lccs_class_mode",
    "glc10_lc",
    "gsw",
    "nlcd",
    "cdl_cultivated_mode",
    "cdl_crop_mode",
    "cdl_simple_crop_mode",
    "us_lith",
}

# Meta-groups that expand to sub-groups for filtering
_GROUP_ALIASES = {
    "landsat": {"landsat_bands", "landsat_indices"},
}

# Landsat sub-groups for band vs index importance analysis
LANDSAT_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B10"]
LANDSAT_INDICES = ["nd", "nw", "evi", "gi"]

# Feature group definitions for selective exclusion and importance analysis
FEATURE_GROUPS = {
    "landsat": LANDSAT_BANDS + LANDSAT_INDICES,
    "landsat_bands": LANDSAT_BANDS,
    "landsat_indices": LANDSAT_INDICES,
    "sentinel1": ["VV", "VH", "VH_VV"],
    "smap": list(ee_feature_list._SMAP_L4.keys()) + ["vegetation_water_content"],
    "gridmet": list(ee_feature_list._GRIDMET_VARS.keys()),
    "soilgrids": list(ee_feature_list._SOILGRIDS.keys()),
    "fao": list(ee_feature_list._FAO_SOILS.keys()),
    "polaris": list(ee_feature_list._POLARIS.keys()),
    "amsr_vod": list(ee_feature_list._AMSR_VOD.keys()),
    "terrain": [
        "elevation",
        "slope",
        "aspect",
        "tpi_10000",
        "tpi_22500",
        "topoDiversity",
        "b1",
    ],
    "coords": ["lat", "lon"],
    "embeddings": [],  # Handled by pattern matching
    "worldclim": ["wc"],
    "hihydrosoil": ["hhs"],
    "prism": [
        "ppt",
        "tdmean",
        "tmin",
        "tmax",
        "tmean",
        "vpdmin",
        "vpdmax",
        "solclear",
        "solslope",
        "soltotal",
        "soltrans",
    ],
    "ssurgo": ["ssurgo_awc", "ssurgo_clay", "ssurgo_ksat", "ssurgo_sand"],
    "landcover": list(_LANDCOVER_FEATURES),
}

# Columns that are NOT features
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
    # Depth
    "rosetta_level",
    "depth_cm",
    "depth",
    # EE export artifact
    "constant",
    # NCSS lab measurements (not available at inference time)
    "clay_tot_psa",
    "sand_tot_psa",
    "silt_tot_psa",
    "db_od",
}


def filter_feature_groups(
    feature_cols: List[str],
    exclude_groups: List[str],
) -> List[str]:
    """
    Remove features belonging to specified groups.

    Parameters
    ----------
    feature_cols : list
        List of feature column names.
    exclude_groups : list of str
        Group names to exclude (e.g., ['landsat', 'sentinel1']).

    Returns
    -------
    list
        Filtered feature columns.
    """
    if not exclude_groups:
        return feature_cols

    exclude_set = set()
    for g in exclude_groups:
        exclude_set.add(g)
        exclude_set.update(_GROUP_ALIASES.get(g, set()))

    return [c for c in feature_cols if classify_feature(c) not in exclude_set]


def get_feature_columns(
    df,
    include_depth: bool = True,
    include_embeddings: bool = False,
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
        Whether to include embedding columns.

    Returns
    -------
    list
        Sorted list of feature column names.
    """
    rosetta_pattern = re.compile(r"^US_R3H3_L\d+_VG_")

    feature_cols = []
    for c in df.columns:
        if c in NON_FEATURE_COLS:
            continue
        if rosetta_pattern.match(c):
            continue
        if not include_embeddings and any(pat.match(c) for pat in _EMBEDDING_PATTERNS):
            continue
        feature_cols.append(c)

    if include_depth:
        if "depth_cm" in df.columns and "depth_cm" not in feature_cols:
            feature_cols.append("depth_cm")
        if "rosetta_level" in df.columns and "rosetta_level" not in feature_cols:
            feature_cols.append("rosetta_level")

    feature_cols = [c for c in feature_cols if df[c].notna().any()]

    return sorted(set(feature_cols))


def classify_feature(feature_name: str) -> str:
    """
    Classify a feature column name into its group.

    Parameters
    ----------
    feature_name : str
        Feature column name (e.g., 'B5_mean_gs', 'elevation', 'clay_mean').

    Returns
    -------
    str
        Group name (e.g., 'landsat_bands', 'terrain', 'soilgrids').
        Returns 'other' if no group matches.
    """
    name = str(feature_name)

    # Check embeddings first (pattern-based)
    if any(pat.match(name) for pat in _EMBEDDING_PATTERNS):
        return "embeddings"

    # Check each group via prefix/exact match (skip meta-groups)
    check_order = [
        "landsat_bands",
        "landsat_indices",
        "sentinel1",
        "smap",
        "gridmet",
        "soilgrids",
        "fao",
        "polaris",
        "amsr_vod",
        "terrain",
        "coords",
        "worldclim",
        "hihydrosoil",
        "prism",
        "ssurgo",
        "landcover",
    ]

    for group in check_order:
        group_features = FEATURE_GROUPS[group]
        for f in group_features:
            if name == f or name.startswith(f + "_"):
                return group

    # SoilGrids depth-resolved (e.g., clay_30-60cm_mean)
    if _SOILGRIDS_DEPTH_RE.match(name):
        return "soilgrids"

    # Depth columns
    if name in ("depth_cm", "rosetta_level"):
        return "depth"

    # Theta (input feature in direct model)
    if name == "theta":
        return "theta"

    return "other"


def aggregate_importance_by_group(
    importance_dict: Dict[str, float],
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Aggregate per-feature importance scores into group-level totals.

    Parameters
    ----------
    importance_dict : dict
        Mapping of feature name to importance score.
    normalize : bool
        If True, normalize so group importances sum to 1.

    Returns
    -------
    dict
        Mapping of group name to aggregated importance, sorted descending.
    """
    group_totals = {}
    for feature, importance in importance_dict.items():
        group = classify_feature(feature)
        group_totals[group] = group_totals.get(group, 0.0) + importance

    if normalize:
        total = sum(group_totals.values())
        if total > 0:
            group_totals = {k: v / total for k, v in group_totals.items()}

    return dict(sorted(group_totals.items(), key=lambda x: x[1], reverse=True))


if __name__ == "__main__":
    print("Feature groups:")
    for name, features in FEATURE_GROUPS.items():
        print(f"  {name}: {len(features)} base features")

    print("\nClassification examples:")
    examples = [
        "B5_mean_gs",
        "nd_mean_1",
        "VH_VV_mean",
        "clay_mean",
        "elevation",
        "embedding_42",
        "theta",
        "depth_cm",
    ]
    for ex in examples:
        print(f"  {ex} -> {classify_feature(ex)}")
