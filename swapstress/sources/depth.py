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
to these standard levels for consistent training data preparation.
"""

import numpy as np
from retention_curve import ROSETTA_LEVEL_DEPTHS


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
