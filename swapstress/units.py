"""Unit conversions for soil water potential.

Internal convention throughout the pipeline is **suction head in cm H2O**,
positive, and its base-10 logarithm ``log10_suction_cm`` -- the model target.

Released products additionally carry **signed matric potential in MPa**
(negative: wet soil is near 0, permanent wilting point is -1.5 MPa). Two sign
conventions coexist by design, each matching its own community's norm; the band
metadata must state this explicitly rather than leave a reuser to infer it.

The key property, and the reason MPa is a postprocessing step rather than a
retraining concern: because the model target is ``log10(suction_cm)``, the
conversion to MPa is an **exact additive shift in log space**::

    log10|psi_MPa| = log10(suction_cm) - LOG10_MPA_TO_CM

R^2, RMSE, and quantile-interval widths expressed in log units are therefore
numerically identical under the conversion. Nothing retrains and no reported
metric changes.
"""

from __future__ import annotations

import math

import numpy as np

# 1 MPa of head = 1e6 Pa / (rho_w * g) = 1e6 / (1000 kg m^-3 * 9.80665 m s^-2)
# = 101.97 m = 10197.16 cm of water.
MPA_TO_CM = 10197.16

# Derived, never hardcoded separately, so the two cannot drift apart.
LOG10_MPA_TO_CM = math.log10(MPA_TO_CM)  # 4.0084792...

# 1 kPa of head under the same hydrostatic definition: 10.19716 cm of water.
# POLARIS distributes its van Genuchten alpha as log10(kPa^-1) -- per the
# readme correction of 2019-06-02, not the cm^-1 its early documentation
# claimed -- so this is the factor that brings it onto the pipeline's cm basis.
KPA_TO_CM = MPA_TO_CM / 1000.0


def suction_cm_to_mpa(suction_cm):
    """Positive suction head (cm H2O) -> signed matric potential (MPa, negative)."""
    return -np.asarray(suction_cm, dtype=np.float64) / MPA_TO_CM


def mpa_to_suction_cm(psi_mpa):
    """Signed matric potential (MPa) -> positive suction head (cm H2O)."""
    return np.abs(np.asarray(psi_mpa, dtype=np.float64)) * MPA_TO_CM


def log10_suction_cm_to_mpa(log10_suction_cm):
    """``log10_suction_cm`` -> signed matric potential (MPa, negative).

    This is the conversion applied when writing the released
    ``matric_potential_MPa`` band.
    """
    suction_cm = np.power(10.0, np.asarray(log10_suction_cm, dtype=np.float64))
    return -suction_cm / MPA_TO_CM


def log10_suction_cm_to_log10_abs_mpa(log10_suction_cm):
    """``log10_suction_cm`` -> ``log10`` of the matric-potential magnitude in MPa.

    An exact additive shift; see the module docstring. Kept as a named function
    so the identity is testable and the constant appears in exactly one place.
    """
    return np.asarray(log10_suction_cm, dtype=np.float64) - LOG10_MPA_TO_CM


def log10_abs_mpa_to_log10_suction_cm(log10_abs_mpa):
    """Inverse of :func:`log10_suction_cm_to_log10_abs_mpa`."""
    return np.asarray(log10_abs_mpa, dtype=np.float64) + LOG10_MPA_TO_CM


# ========================= EOF ====================================================================
