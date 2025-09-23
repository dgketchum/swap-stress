import os
import re

PARAM_SYMBOLS = {
    'theta_r': r'$\theta_r$',
    'theta_s': r'$\theta_s$',
    'alpha': r'$\alpha$',
    'n': 'n'
}

ROSETTA_LEVEL_DEPTHS = {
    1: (0, 2.5),
    2: (2.5, 10),
    3: (10, 22.5),
    4: (22.5, 45),
    5: (45, 80),
    6: (80, 150),
    7: (150, 250)
}

ROSETTA_NOMINAL_DEPTHS = {
    1: 0,
    2: 5,
    3: 15,
    4: 30,
    5: 60,
    6: 100,
    7: 200
}

EMPIRICAL_TO_ROSETTA_LEVEL_MAP = {
    5: 2,
    10: 2,
    20: 3,
    45: 4,
    50: 5,
    70: 5,
    91: 6,
    100: 6
}


def map_empirical_to_rosetta_level(empirical_depth):
    """
    Maps an empirical measurement depth (cm) to the corresponding Rosetta
    vertical level (1-7).
    """
    return EMPIRICAL_TO_ROSETTA_LEVEL_MAP.get(empirical_depth)


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
    centers = {lvl: (rng[0] + rng[1]) / 2.0 for lvl, rng in ROSETTA_LEVEL_DEPTHS.items()}
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


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
