"""
Utilities for mapping soil depths to Rosetta vertical levels.

Rosetta uses 7 standard depth layers:
    L1: 0-2.5 cm
    L2: 2.5-10 cm
    L3: 10-22.5 cm
    L4: 22.5-45 cm
    L5: 45-80 cm
    L6: 80-150 cm
    L7: 150-250 cm

This module provides functions to map arbitrary depth measurements
to these standard levels for consistent training data preparation. It also
holds the corresponding POLARIS layer ranges and the measurement-depth to
Rosetta-level table used for the in-situ sources; these were the contents of
``retention_curve/__init__.py`` before the package was assembled.
"""

import re

import numpy as np

ROSETTA_LEVEL_DEPTHS = {
    1: (0, 2.5),
    2: (2.5, 10),
    3: (10, 22.5),
    4: (22.5, 45),
    5: (45, 80),
    6: (80, 150),
    7: (150, 250),
}

ROSETTA_NOMINAL_DEPTHS = {1: 0, 2: 5, 3: 15, 4: 30, 5: 60, 6: 100, 7: 200}

EMPIRICAL_TO_ROSETTA_LEVEL_MAP = {
    5: 2,
    10: 2,
    20: 3,
    45: 4,
    50: 5,
    70: 5,
    91: 6,
    100: 6,
}

# POLARIS depth ranges (cm) and nominal depths (cm)
POLARIS_DEPTH_RANGES = {
    1: (0, 5),
    2: (5, 15),
    3: (15, 30),
    4: (30, 60),
    5: (60, 100),
    6: (100, 200),
}

POLARIS_NOMINAL_DEPTHS = {
    1: 0,
    2: 10,
    3: 22.5,
    4: 45,
    5: 80,
    6: 150,
}


def map_empirical_to_rosetta_level(empirical_depth):
    """
    Maps an empirical measurement depth (cm) to the corresponding Rosetta
    vertical level (1-7).
    """
    return EMPIRICAL_TO_ROSETTA_LEVEL_MAP.get(empirical_depth)


def map_polaris_depth_range_to_rosetta_level(dmin_cm, dmax_cm):
    """
    Maps a POLARIS layer depth range (cm) to the nearest Rosetta level (1-7)
    by midpoint-to-range mapping, with nearest-center fallback on boundaries.
    """
    try:
        mid = 0.5 * (float(dmin_cm) + float(dmax_cm))
    except Exception:
        return None
    for lvl, (lo, hi) in ROSETTA_LEVEL_DEPTHS.items():
        if lo <= mid < hi:
            return int(lvl)
    centers = {
        lvl: (rng[0] + rng[1]) / 2.0 for lvl, rng in ROSETTA_LEVEL_DEPTHS.items()
    }
    levels = list(centers.keys())
    vals = list(centers.values())
    idx = int(min(range(len(vals)), key=lambda i: abs(vals[i] - mid)))
    return int(levels[idx])


def parse_polaris_depth_from_asset(asset_path):
    """
    Parses a POLARIS asset or layer string ending in "_min_max" (cm) and returns (min, max).
    Example: ".../theta_r_15_30" -> (15, 30)
    """
    m = re.search(r"_(\d+)_(\d+)$", str(asset_path))
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def depth_to_rosetta_level(depth_cm):
    """
    Map a depth measurement (cm) to the corresponding Rosetta level (1-7).

    Parameters
    ----------
    depth_cm : float or str
        Depth in centimeters. Can be a string that parses to float.

    Returns
    -------
    int or None
        Rosetta level (1-7), or None if depth cannot be parsed.

    Examples
    --------
    >>> depth_to_rosetta_level(5)
    2
    >>> depth_to_rosetta_level(100)
    6
    >>> depth_to_rosetta_level(300)  # Beyond L7, returns nearest
    7
    """
    try:
        d = float(depth_cm)
    except (TypeError, ValueError):
        return None

    # Direct range match
    for lvl, (lo, hi) in ROSETTA_LEVEL_DEPTHS.items():
        if lo <= d < hi:
            return int(lvl)

    # For depths outside defined ranges, find nearest level center
    centers = {
        lvl: (rng[0] + rng[1]) / 2.0 for lvl, rng in ROSETTA_LEVEL_DEPTHS.items()
    }
    levels = np.array(list(centers.keys()), dtype=int)
    vals = np.array(list(centers.values()), dtype=float)
    idx = int(np.argmin(np.abs(vals - d)))
    return int(levels[idx])


def depth_range_to_rosetta_level(depth_min_cm, depth_max_cm):
    """
    Map a depth range (cm) to the corresponding Rosetta level (1-7).

    Uses the midpoint of the range for mapping.

    Parameters
    ----------
    depth_min_cm : float
        Minimum depth in centimeters.
    depth_max_cm : float
        Maximum depth in centimeters.

    Returns
    -------
    int or None
        Rosetta level (1-7), or None if depths cannot be parsed.

    Examples
    --------
    >>> depth_range_to_rosetta_level(0, 10)
    2
    >>> depth_range_to_rosetta_level(30, 60)
    4
    """
    try:
        midpoint = (float(depth_min_cm) + float(depth_max_cm)) / 2.0
    except (TypeError, ValueError):
        return None
    return depth_to_rosetta_level(midpoint)


def horizon_to_rosetta_level(hzn_top, hzn_bot):
    """
    Map soil horizon boundaries to Rosetta level.

    Convenience wrapper for GSHP-style data where horizon top/bottom
    are provided separately.

    Parameters
    ----------
    hzn_top : float
        Top of horizon in centimeters.
    hzn_bot : float
        Bottom of horizon in centimeters.

    Returns
    -------
    int or None
        Rosetta level (1-7), or None if depths cannot be parsed.
    """
    return depth_range_to_rosetta_level(hzn_top, hzn_bot)


if __name__ == "__main__":
    # Quick validation
    test_depths = [0, 1, 5, 15, 30, 50, 100, 200, 300]
    for d in test_depths:
        print(f"Depth {d:3d} cm -> Level {depth_to_rosetta_level(d)}")
