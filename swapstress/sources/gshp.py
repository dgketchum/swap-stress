"""Published van Genuchten parameters from the GSHP database.

For GSHP layers we use the parameters Gupta et al. (2021) published rather than
refitting the retention curves ourselves. This module is the one place that
reads them, so the unit conversion and the quality filter happen exactly once.

Why not refit
-------------
The two are equivalent in goodness-of-fit and ours are worse in validity. Over
the 10,411 layers where both existed, median RMSE against the same observations
was 0.0074 (ours) versus 0.0082 (published), but 933 of our fits -- 9.0% --
returned ``theta_r >= theta_s``, a parameter set :mod:`swapstress.swrc`
correctly refuses to evaluate. The published set has none. Taking them directly
also raises coverage from 10,411 to 15,259 layers and brings published
uncertainty (``se_alpha``, ``se_n``, and quantiles) that our deterministic fits
never produced.

Compatibility with :mod:`swapstress.swrc`
-----------------------------------------
GSHP fits with ``soilhypfit``, whose ``sat_model`` is
``(1 + (h * alpha)**n)**(-1 + 1/n)``. That is ``(1 + (alpha*h)**n)**-m`` with
``m = 1 - 1/n`` -- the same Mualem-constrained form this package uses. The
parameters are drop-in; only the units differ.

Units
-----
GSHP tabulates head in **metres** (``lab_head_m``), so ``alpha`` and its
uncertainty columns are in **1/m**. Everything here converts to 1/cm on load.
Verified empirically: evaluating the published parameters against the published
observations gives a median RMSE of 0.006 m3/m3 under the metre convention and
0.21 if alpha is taken as 1/cm.

Quality and provenance caveats
------------------------------
``data_flag`` is not decoration. ``alpha`` is capped at exactly 100 1/m and
``n`` at exactly 7 -- those are the optimizer's bounds, and 1,045 layers sit on
them, with a further 2,490 flagged as having a flat likelihood profile. In all,
3,554 of 15,259 layers (23.3%) are not ``"good quality estimate"``, which is why
``good_quality_only`` defaults to True.

``SWCC_classes`` records which end of the curve was measured, and it determines
how freely GSHP could fit ``theta_r``/``theta_s``:

- ``YWYD`` (10,301 layers, 67.5%) -- both ends measured; both parameters fit
  freely on [0, 1]. This is the only fully data-driven subset.
- ``NWYD`` (4,598, 30.1%) -- no wet end; ``theta_s`` is confined to a prediction
  interval from a texture + bulk-density + climate-region regression.
- ``YWND`` (333, 2.2%) -- no dry end; ``theta_r`` is bounded by a specific
  surface area estimate.
- ``NWND`` (27, 0.2%) -- both constrained.

The 30.1% ``NWYD`` subset therefore carries a pedotransfer prior on
``theta_s``. That is harmless for training, which uses observation-level
(theta, psi) pairs and never sees these parameters, but it is circular against
a texture-based PTF. Pass ``swcc_classes=("YWYD",)`` for any comparison with
Rosetta or POLARIS.

Exactly one layer in the full dataset, ``kool_70``, fails
:func:`swapstress.swrc.valid_params`: its six observations are non-monotonic
(theta rises from 0.447 at zero head to 0.495 at 150 m), so the fit collapsed to
``theta_r == theta_s == 0.449333``. GSHP flags it ``"flat upper profile for n"``,
so the default quality filter already removes it. It is the only one.

Note on the ``*_clean.csv`` variant
-----------------------------------
``WRC_dataset_surya_et_al_2021_final_clean.csv`` is a per-profile file built for
Earth Engine point extraction; it collapses each profile with ``first``, so its
parameter columns belong to whichever layer happened to sort first. It is not a
source of per-layer parameters. Point this loader at the full dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from swapstress.swrc import valid_params

# GSHP tabulates head in metres, so its alpha is 1/m.
ALPHA_PER_M_TO_PER_CM = 1.0 / 100.0

GOOD_QUALITY_FLAG = "good quality estimate"

# The only SWCC class with both theta_r and theta_s fit freely from data.
FREE_THETA_CLASS = "YWYD"

# Optimizer bounds in the published fit; parameters resting on these are at the
# edge of the feasible set rather than at an interior optimum.
ALPHA_UPPER_BOUND_PER_M = 100.0
N_UPPER_BOUND = 7.0

_ALPHA_COLS = (
    "alpha",
    "se_alpha",
    "q2.5_alpha",
    "q97.5_alpha",
    "q10_alpha",
    "q90_alpha",
    "q25_alpha",
    "q75_alpha",
)

_RENAME = {
    "thetar": "theta_r",
    "thetas": "theta_s",
    "latitude_decimal_degrees": "latitude",
    "longitude_decimal_degrees": "longitude",
}

__all__ = [
    "ALPHA_PER_M_TO_PER_CM",
    "GOOD_QUALITY_FLAG",
    "FREE_THETA_CLASS",
    "ALPHA_UPPER_BOUND_PER_M",
    "N_UPPER_BOUND",
    "sanitize_profile_id",
    "load_published_params",
    "to_swrc_arrays",
]


def sanitize_profile_id(value) -> str:
    """Strip path delimiters from a GSHP profile id.

    Profile ids are used as filenames elsewhere in the pipeline, so the same
    substitution has to be applied on both sides of any join.
    """
    return str(value).replace("/", "_").replace("\\", "_")


def load_published_params(
    csv_path: str,
    *,
    good_quality_only: bool = True,
    swcc_classes=None,
) -> pd.DataFrame:
    """Read the published van Genuchten parameters, one row per layer.

    Parameters
    ----------
    csv_path : str
        Path to ``WRC_dataset_surya_et_al_2021_final.csv`` (the full dataset,
        not the ``_clean`` per-profile variant).
    good_quality_only : bool
        Keep only ``data_flag == "good quality estimate"``. Leaving this on
        drops 23.3% of layers whose alpha or n is unidentified or resting on an
        optimizer bound. Turn it off only to characterise what is being
        excluded.
    swcc_classes : iterable of str, optional
        Restrict to these ``SWCC_classes``. Pass ``("YWYD",)`` when comparing
        against a texture-based pedotransfer function, so the comparison is not
        contaminated by GSHP's own PTF prior on ``theta_s``.

    Returns
    -------
    pd.DataFrame
        Columns ``profile_id``, ``layer_id``, ``depth_cm``, ``theta_r``,
        ``theta_s``, ``alpha`` (1/cm), ``n``, the available uncertainty columns
        (also converted where they carry alpha's units), ``data_flag``,
        ``SWCC_classes``, ``latitude``, ``longitude``, and ``at_param_bound``.

        Depth is the horizon midpoint, matching ``depth_from_horizon`` in the
        source registry.
    """
    df = pd.read_csv(csv_path, encoding="latin1", low_memory=False)

    missing = [c for c in ("layer_id", "alpha", "n", "thetar", "thetas") if c not in df]
    if missing:
        raise ValueError(f"{csv_path} is missing GSHP parameter columns: {missing}")

    # The parameters are repeated across every observation of a layer; collapse
    # to the layer before any filtering so the counts mean layers, not points.
    layers = df.groupby("layer_id", dropna=False).first().reset_index()

    layers = layers.rename(columns=_RENAME)

    if "profile_id" in layers:
        layers["profile_id"] = layers["profile_id"].apply(sanitize_profile_id)

    if "hzn_top" in layers and "hzn_bot" in layers:
        layers["depth_cm"] = (
            layers["hzn_top"].astype(float) + layers["hzn_bot"].astype(float)
        ) / 2.0

    # Flag bound-resting parameters before the unit conversion, while the
    # published bounds are still expressed in the units they were set in.
    layers["at_param_bound"] = (
        layers["alpha"].astype(float) >= ALPHA_UPPER_BOUND_PER_M
    ) | (layers["n"].astype(float) >= N_UPPER_BOUND)

    for col in _ALPHA_COLS:
        if col in layers:
            layers[col] = layers[col].astype(float) * ALPHA_PER_M_TO_PER_CM

    if good_quality_only:
        if "data_flag" not in layers:
            raise ValueError(
                f"{csv_path} has no 'data_flag' column; cannot apply the quality "
                "filter. Pass good_quality_only=False to read it unfiltered."
            )
        layers = layers[layers["data_flag"] == GOOD_QUALITY_FLAG]

    if swcc_classes is not None:
        if "SWCC_classes" not in layers:
            raise ValueError(f"{csv_path} has no 'SWCC_classes' column")
        layers = layers[layers["SWCC_classes"].isin(list(swcc_classes))]

    keep = [
        "profile_id",
        "layer_id",
        "depth_cm",
        "theta_r",
        "theta_s",
        "alpha",
        "n",
        "se_alpha",
        "se_n",
        "q2.5_alpha",
        "q97.5_alpha",
        "q2.5_n",
        "q97.5_n",
        "data_flag",
        "SWCC_classes",
        "at_param_bound",
        "latitude",
        "longitude",
    ]
    out = layers[[c for c in keep if c in layers]].reset_index(drop=True)

    # These are published fits, not ours. Within the good-quality set every
    # layer should satisfy valid_params, so a failure means the file is not
    # what we think it is and must surface rather than be dropped. Outside it
    # the caller has explicitly asked for the raw set, so report and hand back.
    bad = ~valid_params(out["theta_r"], out["theta_s"], out["alpha"], out["n"])
    if bad.any():
        detail = ", ".join(str(v) for v in out.loc[bad, "layer_id"].head(5))
        if good_quality_only:
            raise ValueError(
                f"{int(bad.sum())} good-quality GSHP layers fail valid_params "
                f"(e.g. {detail}); the published set should have none. "
                f"Check {csv_path}."
            )
        print(
            f"  [GSHP] {int(bad.sum())} of {len(out)} unfiltered layers fail "
            f"valid_params (e.g. {detail}). These are degenerate published fits "
            "that the data_flag filter removes; swrc will return NaN for them."
        )

    return out


def to_swrc_arrays(params: pd.DataFrame):
    """Split a :func:`load_published_params` frame into the four vG arrays.

    Ordered to feed :func:`swapstress.swrc.theta_from_psi` positionally.
    """
    return (
        params["theta_r"].to_numpy(dtype=np.float64),
        params["theta_s"].to_numpy(dtype=np.float64),
        params["alpha"].to_numpy(dtype=np.float64),
        params["n"].to_numpy(dtype=np.float64),
    )


# ========================= EOF ====================================================================
