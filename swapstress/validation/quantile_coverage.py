"""Prediction interval coverage (PICP) for the QRF on the spatial holdout.

The product ships a QRF prediction interval as the ``log10_suction_cm_q025`` /
``log10_suction_cm_q975`` pair, so the descriptor has to say whether that
interval is honest: of the held-out observations, what fraction actually fall
inside it. That fraction is the prediction interval coverage probability, PICP,
and a nominal 95% interval covering ~94% is the number SWSM reports and the one
Fig 6 quotes.

Two views, because they answer different questions:

**Calibration** sweeps the nominal level from 50% to 99% and reports the
empirical coverage of each. A single number at 95% cannot distinguish an
interval that is honest everywhere from one that is too wide in the middle and
too narrow in the tails; the sweep can.

**Coverage by theta decile** holds the nominal level at the released 95% and
asks where the interval is honest. The product's known weak spot is the dry end,
where SMAP floors out and where stress actually matters, so coverage conditional
on theta is the diagnostic that matters for fitness-for-use.

Both need a quantile-capable model. A plain ``RandomForestRegressor`` has no
predictive distribution to take quantiles of, so this analysis is not in
``--analysis all``: it is asked for by name against a model trained with
``swapstress-train --quantile``.

Produces:
    - quantile_coverage.csv           calibration sweep, one row per nominal level
    - quantile_coverage_by_theta.csv  coverage by theta decile at the released level
"""

from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

from swapstress.inference.predict import supports_quantiles
from swapstress.inference.product import RELEASE_NOMINAL
from swapstress.validation.reconstruct_test_set import MODEL_DIR, reconstruct

# Wide enough to show the shape of the calibration curve, with the released
# level on it rather than interpolated to.
NOMINAL_LEVELS = (0.50, 0.80, 0.90, RELEASE_NOMINAL, 0.99)


def interval_levels(nominal: float) -> tuple[float, float]:
    """The two-sided quantile levels bounding a *nominal* central interval."""
    tail = (1.0 - nominal) / 2.0
    return round(tail, 6), round(1.0 - tail, 6)


def predict_quantiles(
    model,
    imputer,
    features: pd.DataFrame,
    levels: list[float],
) -> np.ndarray:
    """Quantile predictions for every row, as (n_rows, len(levels))."""
    if not supports_quantiles(model):
        raise TypeError(
            f"{type(model).__name__} cannot predict quantiles, so it has no "
            "prediction interval to check. Coverage needs a model trained with "
            "swapstress-train --quantile."
        )
    matrix = imputer.transform(features.values.astype(np.float32))
    return np.asarray(model.predict(matrix, quantiles=levels), dtype=np.float64)


def coverage(observed: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    """Fraction of observations inside the closed interval."""
    return float(np.mean((observed >= low) & (observed <= high)))


def calibration_table(
    observed: np.ndarray,
    quantiles: np.ndarray,
    levels: list[float],
    nominals=NOMINAL_LEVELS,
) -> pd.DataFrame:
    """Empirical coverage and interval width at each nominal level."""
    rows = []
    for nominal in nominals:
        lo, hi = interval_levels(nominal)
        low = quantiles[:, levels.index(lo)]
        high = quantiles[:, levels.index(hi)]
        rows.append(
            {
                "nominal": nominal,
                "q_lo": lo,
                "q_hi": hi,
                "picp": coverage(observed, low, high),
                "mean_width": float(np.mean(high - low)),
                "median_width": float(np.median(high - low)),
                "n": int(observed.size),
            }
        )
    return pd.DataFrame(rows)


def coverage_by_theta(
    theta: np.ndarray,
    observed: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    nominal: float,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Coverage of one interval, binned by theta decile.

    Deciles rather than fixed edges, so every bin carries the same number of
    observations and a coverage estimate from one sparse corner of the theta
    range cannot look as certain as one from the bulk.
    """
    bins = pd.qcut(theta, n_bins, duplicates="drop")
    rows = []
    for interval, index in pd.Series(range(theta.size)).groupby(bins, observed=True):
        idx = index.values
        rows.append(
            {
                "theta_lo": interval.left,
                "theta_hi": interval.right,
                "theta_mid": (interval.left + interval.right) / 2,
                "nominal": nominal,
                "picp": coverage(observed[idx], low[idx], high[idx]),
                "mean_width": float(np.mean(high[idx] - low[idx])),
                "n": int(idx.size),
            }
        )
    return pd.DataFrame(rows)


def run(model_dir: str, output_dir: str, n_bins: int = 10) -> tuple:
    """Compute both coverage tables and write them beside the model."""
    test_df = reconstruct(model_dir)
    model = joblib.load(os.path.join(model_dir, "direct_rf_model.joblib"))
    imputer = joblib.load(os.path.join(model_dir, "direct_rf_imputer.joblib"))

    with open(os.path.join(model_dir, "direct_rf_features.json")) as f:
        feature_names = json.load(f)

    levels = sorted({q for n in NOMINAL_LEVELS for q in interval_levels(n)})
    quantiles = predict_quantiles(model, imputer, test_df[feature_names], levels)
    observed = test_df["observed"].values.astype(np.float64)

    calibration = calibration_table(observed, quantiles, levels)

    lo, hi = interval_levels(RELEASE_NOMINAL)
    by_theta = coverage_by_theta(
        theta=test_df["theta"].values.astype(np.float64),
        observed=observed,
        low=quantiles[:, levels.index(lo)],
        high=quantiles[:, levels.index(hi)],
        nominal=RELEASE_NOMINAL,
        n_bins=n_bins,
    )

    os.makedirs(output_dir, exist_ok=True)
    calibration_path = os.path.join(output_dir, "quantile_coverage.csv")
    theta_path = os.path.join(output_dir, "quantile_coverage_by_theta.csv")
    calibration.to_csv(calibration_path, index=False)
    by_theta.to_csv(theta_path, index=False)

    print(f"\nCoverage on {len(test_df):,} held-out observations:")
    for _, row in calibration.iterrows():
        print(
            f"  nominal {row['nominal']:.3f}  PICP {row['picp']:.3f}  "
            f"mean width {row['mean_width']:.3f} log10 cm"
        )
    print(f"Saved {calibration_path}")
    print(f"Saved {theta_path}")
    return calibration, by_theta


def main():
    parser = argparse.ArgumentParser(
        description="QRF prediction interval coverage on the spatial holdout"
    )
    parser.add_argument(
        "--model-dir",
        default=MODEL_DIR,
        help="Path to trained model directory (must be a quantile forest).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <model-dir>/error_analysis/).",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of theta quantile bins (default: 10).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.model_dir, "error_analysis")
    run(args.model_dir, output_dir, n_bins=args.n_bins)


if __name__ == "__main__":
    main()
