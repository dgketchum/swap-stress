import os

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


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
