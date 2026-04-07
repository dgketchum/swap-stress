"""
Evaluation metrics for the direct suction prediction task.

Shared between RF and NN trainers.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    linear: bool = False,
) -> Dict[str, float]:
    """Compute prediction metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True values (log10 scale).
    y_pred : np.ndarray
        Predicted values (log10 scale).
    linear : bool
        If True, transform to linear scale (cm H2O) before computing.

    Returns
    -------
    dict
        Dictionary with rmse, mae, r2, bias, n.
    """
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isinf(y_pred)
    if valid.sum() == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "bias": np.nan, "n": 0}

    yt, yp = y_true[valid], y_pred[valid]
    if linear:
        yt, yp = 10**yt, 10**yp

    return {
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "r2": float(r2_score(yt, yp)),
        "bias": float(np.mean(yp - yt)),
        "n": int(valid.sum()),
    }


def compute_metrics_by_source(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sources: np.ndarray,
    linear: bool = False,
) -> pd.DataFrame:
    """Compute metrics stratified by data source."""
    results = []
    for source in np.unique(sources[~pd.isna(sources)]):
        mask = sources == source
        if mask.sum() < 10:
            continue
        metrics = compute_metrics(y_true[mask], y_pred[mask], linear=linear)
        metrics["source"] = source
        results.append(metrics)
    return pd.DataFrame(results)


def compute_metrics_by_site(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    site_ids: np.ndarray,
    linear: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute per-site metrics and site-weighted aggregates."""
    results = []
    for site in np.unique(site_ids[~pd.isna(site_ids)]):
        mask = site_ids == site
        if mask.sum() < 3:
            continue
        metrics = compute_metrics(y_true[mask], y_pred[mask], linear=linear)
        metrics["site_id"] = site
        results.append(metrics)

    per_site_df = pd.DataFrame(results)
    if len(per_site_df) > 0:
        summary = {
            "mean_rmse": per_site_df["rmse"].mean(),
            "median_rmse": per_site_df["rmse"].median(),
            "mean_r2": per_site_df["r2"].mean(),
            "median_r2": per_site_df["r2"].median(),
            "mean_mae": per_site_df["mae"].mean(),
            "median_mae": per_site_df["mae"].median(),
            "n_sites": len(per_site_df),
        }
    else:
        summary = {
            k: np.nan
            for k in [
                "mean_rmse",
                "median_rmse",
                "mean_r2",
                "median_r2",
                "mean_mae",
                "median_mae",
            ]
        }
        summary["n_sites"] = 0

    return per_site_df, summary
