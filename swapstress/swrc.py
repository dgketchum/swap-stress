"""Canonical van Genuchten soil water retention curve.

This module replaces 15 independent copies of the van Genuchten equations that
had accumulated across the repository. The algebra was identical in every copy;
what differed were the numerical guards near saturation and at the dry end --
exactly the instability that motivates predicting suction directly rather than
inverting a fitted retention curve.

Conventions
-----------
- ``psi`` / suction is **positive** head in cm H2O.
- ``alpha`` is in 1/cm, natural scale (not log10).
- ``n > 1``; ``m`` is constrained to ``1 - 1/n`` (the Mualem restriction), which
  is what every call site in this repository assumed.
- Invalid parameter combinations yield ``NaN`` rather than a patched value. Do
  not add fallbacks here: a NaN means the upstream parameters are unusable and
  that should surface, not be silently filled. The one exception is
  ``theta_from_psi(..., strict=False)``, which exists so least-squares fitters
  can evaluate the curve across a search path that passes through invalid
  parameter space; see that function's docstring.

Numerical guards
----------------
:func:`psi_from_theta` clips effective saturation to ``[se_eps, 1 - se_eps]``.
The default ``SE_EPS = 1e-6`` reproduces the behaviour of the PTF baseline that
produces the manuscript's Rosetta/POLARIS comparison, bit for bit. Call sites
that need a different clip (the flux analyses used ``1e-3``) pass ``se_eps``
explicitly so their published numbers are preserved.

Be aware that the clip is what makes the dry end finite at all: at
``theta <= theta_r`` the true inverse diverges. Clipping ``Se`` at a floor caps
``psi`` at a ceiling, so past that point the inverse does not merely lose
precision -- it **understates** suction, by a margin that grows without bound as
the soil dries. The ceiling is strongly parameter-dependent (order 1e14 cm at
``alpha=0.01, n=1.5``, but only ~690 cm at ``alpha=0.145, n=4``), so it cannot be
quoted as a single threshold. This is the reason the dry end is called out as a
fitness-for-use limit, and the reason the released product predicts suction
directly rather than inverting a fitted retention curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Effective-saturation clip. Matches the PTF baseline that generates the
# manuscript's published Rosetta/POLARIS metrics; do not change casually.
SE_EPS = 1e-6

# Forward-direction floor on psi, avoiding 0**n at psi == 0.
PSI_FLOOR_CM = 1e-9

__all__ = [
    "SE_EPS",
    "PSI_FLOOR_CM",
    "VanGenuchtenParams",
    "theta_from_psi",
    "psi_from_theta",
    "log10_psi_from_theta",
    "valid_params",
]


def valid_params(theta_r, theta_s, alpha, n):
    """Boolean mask of parameter combinations the van Genuchten form admits.

    Requires all values finite, ``n > 1``, ``alpha > 0``, and
    ``theta_s > theta_r``.
    """
    theta_r = np.asarray(theta_r, dtype=np.float64)
    theta_s = np.asarray(theta_s, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    return (
        np.isfinite(theta_r)
        & np.isfinite(theta_s)
        & np.isfinite(alpha)
        & np.isfinite(n)
        & (n > 1.0)
        & (alpha > 0.0)
        & (theta_s > theta_r)
    )


@dataclass(frozen=True)
class VanGenuchtenParams:
    """One set of van Genuchten parameters for a single soil layer."""

    theta_r: float
    theta_s: float
    alpha: float  # 1/cm
    n: float  # dimensionless, > 1
    depth_cm: float | None = None

    @property
    def m(self) -> float:
        """Mualem-constrained shape parameter, ``1 - 1/n``."""
        return 1.0 - 1.0 / self.n

    @property
    def is_valid(self) -> bool:
        return bool(valid_params(self.theta_r, self.theta_s, self.alpha, self.n))

    @classmethod
    def from_mapping(cls, mapping, depth_cm=None) -> "VanGenuchtenParams":
        """Build from a dict of parameters.

        Accepts both plain numbers and the ``{"value": x, ...}`` form used by the
        saved curve-fit JSON artifacts.
        """

        def _value(key):
            raw = mapping[key]
            if isinstance(raw, dict) and "value" in raw:
                return float(raw["value"])
            return float(raw)

        return cls(
            theta_r=_value("theta_r"),
            theta_s=_value("theta_s"),
            alpha=_value("alpha"),
            n=_value("n"),
            depth_cm=(
                float(depth_cm)
                if depth_cm is not None
                else (float(mapping["depth_cm"]) if "depth_cm" in mapping else None)
            ),
        )

    def theta(self, psi_cm):
        """Forward: suction (cm) -> volumetric water content."""
        return theta_from_psi(psi_cm, self.theta_r, self.theta_s, self.alpha, self.n)

    def psi(self, theta, *, se_eps: float = SE_EPS):
        """Inverse: volumetric water content -> suction (cm)."""
        return psi_from_theta(
            theta, self.theta_r, self.theta_s, self.alpha, self.n, se_eps=se_eps
        )


def theta_from_psi(
    psi_cm,
    theta_r,
    theta_s,
    alpha,
    n,
    *,
    psi_floor_cm: float = PSI_FLOOR_CM,
    strict: bool = True,
):
    """Forward van Genuchten: theta(psi).

    ``theta = theta_r + (theta_s - theta_r) / (1 + (alpha * psi)**n)**m``

    Parameters
    ----------
    psi_cm : array-like
        Positive suction head in cm H2O.
    theta_r, theta_s : array-like
        Residual and saturated volumetric water content (m3/m3).
    alpha : array-like
        Inverse air-entry parameter, 1/cm.
    n : array-like
        Shape parameter, must be > 1.
    psi_floor_cm : float
        Lower clamp on ``psi`` so that ``psi = 0`` does not raise.
    strict : bool
        When True (the default, and correct for analysis) every condition in
        :func:`valid_params` must hold or the result is ``NaN``.

        When False, only ``n > 1`` is enforced. This exists for **curve-fitting
        residual models only**. A least-squares optimizer walks through
        parameter space that includes ``theta_s <= theta_r``, and returning NaN
        there aborts the fit rather than steering it away; the archived fits
        were produced against this looser evaluation, so reproducing them
        requires it. Never use ``strict=False`` to evaluate a curve for
        analysis -- it will happily return numbers for parameter sets that are
        not retention curves at all.

    Returns
    -------
    np.ndarray
        Volumetric water content; ``NaN`` where parameters are invalid.
    """
    psi_cm = np.asarray(psi_cm, dtype=np.float64)
    theta_r = np.asarray(theta_r, dtype=np.float64)
    theta_s = np.asarray(theta_s, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)

    if strict:
        valid = valid_params(theta_r, theta_s, alpha, n) & np.isfinite(psi_cm)
    else:
        valid = np.isfinite(psi_cm) & np.isfinite(n) & (n > 1.0)
    shape = np.broadcast(psi_cm, theta_r, theta_s, alpha, n).shape
    out = np.full(shape, np.nan, dtype=np.float64)

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        psi_safe = np.maximum(psi_cm, psi_floor_cm)
        m = 1.0 - 1.0 / n
        term = 1.0 + np.power(alpha * psi_safe, n)
        theta = theta_r + (theta_s - theta_r) / np.power(term, m)

    np.copyto(out, np.broadcast_to(theta, shape), where=np.broadcast_to(valid, shape))
    return out


def psi_from_theta(theta, theta_r, theta_s, alpha, n, *, se_eps: float = SE_EPS):
    """Inverse van Genuchten: psi(theta), in cm H2O.

    ``psi = (1/alpha) * (Se**(-1/m) - 1)**(1/n)``, with effective saturation
    ``Se = (theta - theta_r) / (theta_s - theta_r)`` clipped to
    ``[se_eps, 1 - se_eps]``.

    Parameters
    ----------
    theta, theta_r, theta_s : array-like
        Volumetric water content, residual, and saturated (m3/m3).
    alpha : array-like
        Inverse air-entry parameter, 1/cm.
    n : array-like
        Shape parameter, must be > 1.
    se_eps : float
        Effective-saturation clip. The default reproduces the published PTF
        baseline; pass ``1e-3`` to reproduce the flux analyses.

    Returns
    -------
    np.ndarray
        Suction in cm H2O; ``NaN`` where inputs or parameters are invalid.
    """
    theta = np.asarray(theta, dtype=np.float64)
    theta_r = np.asarray(theta_r, dtype=np.float64)
    theta_s = np.asarray(theta_s, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)

    valid = valid_params(theta_r, theta_s, alpha, n) & np.isfinite(theta)
    shape = np.broadcast(theta, theta_r, theta_s, alpha, n).shape
    out = np.full(shape, np.nan, dtype=np.float64)

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        se = np.where(valid, (theta - theta_r) / (theta_s - theta_r), np.nan)
        se = np.clip(se, se_eps, 1.0 - se_eps)
        m = np.where(valid, 1.0 - 1.0 / n, np.nan)
        h = np.where(
            valid,
            (1.0 / alpha) * (se ** (-1.0 / m) - 1.0) ** (1.0 / n),
            np.nan,
        )

    np.copyto(out, np.broadcast_to(h, shape), where=np.broadcast_to(valid, shape))
    return out


def log10_psi_from_theta(
    theta,
    theta_r,
    theta_s,
    alpha,
    n,
    *,
    se_eps: float = SE_EPS,
    psi_floor_cm: float | None = None,
):
    """Inverse van Genuchten returning ``log10`` suction, the model's target unit.

    ``psi_floor_cm`` optionally clamps suction before the log; the flux analyses
    used ``0.01``. Left as ``None`` by default so no floor is imposed silently.
    """
    psi = psi_from_theta(theta, theta_r, theta_s, alpha, n, se_eps=se_eps)
    if psi_floor_cm is not None:
        psi = np.maximum(psi, psi_floor_cm)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log10(psi)


# ========================= EOF ====================================================================
