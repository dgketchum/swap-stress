"""K-fold spatial cross-validation over MGRS tiles.

Shared by RF and NN trainers — each passes its own ``trainer_fn`` and the
loop calls it once per fold with ``test_fold=k``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List

import numpy as np


def run_kfold_cv(
    obs_table_path: str,
    output_dir: str,
    n_folds: int = 5,
    holdout_col: str = "MGRS_TILE",
    trainer_fn: Callable[..., Dict] | None = None,
    trainer_kwargs: Dict[str, Any] | None = None,
) -> Dict:
    """Run k-fold spatial cross-validation.

    For each fold *k* in ``0 .. n_folds-1`` the trainer is called with
    ``test_fold=k`` and a fold-specific output directory.

    Parameters
    ----------
    obs_table_path : str
        Path to observation-level parquet.
    output_dir : str
        Parent directory; per-fold results go in ``fold_0/`` … ``fold_k/``.
    n_folds : int
        Number of folds.
    holdout_col : str
        Column used for spatial grouping.
    trainer_fn : callable
        Training function (e.g. ``train_and_evaluate``).  Must accept
        ``obs_table_path``, ``output_dir``, ``holdout_col``, ``n_folds``,
        and ``test_fold`` as keyword arguments.
    trainer_kwargs : dict
        Extra keyword arguments forwarded to *trainer_fn*.

    Returns
    -------
    dict
        Aggregated k-fold summary with per-fold and overall metrics.
    """
    if trainer_fn is None:
        raise ValueError("trainer_fn is required")
    if trainer_kwargs is None:
        trainer_kwargs = {}

    os.makedirs(output_dir, exist_ok=True)
    fold_results: List[Dict] = []

    for k in range(n_folds):
        fold_dir = os.path.join(output_dir, f"fold_{k}")
        print(f"\n{'=' * 60}")
        print(f"  K-FOLD {k + 1}/{n_folds}  (test_fold={k})")
        print(f"{'=' * 60}\n")

        result = trainer_fn(
            obs_table_path=obs_table_path,
            output_dir=fold_dir,
            holdout_col=holdout_col,
            n_folds=n_folds,
            test_fold=k,
            **trainer_kwargs,
        )
        fold_results.append(result)

    summary = aggregate_fold_results(fold_results)
    summary_path = os.path.join(output_dir, "kfold_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote k-fold summary to {summary_path}")

    _print_summary(summary)
    return summary


def aggregate_fold_results(fold_results: List[Dict]) -> Dict:
    """Aggregate per-fold metrics into summary statistics.

    Parameters
    ----------
    fold_results : list of dict
        Each dict must have an ``"overall_metrics"`` key with at least
        ``r2``, ``rmse``, ``mae``.

    Returns
    -------
    dict
        ``per_fold``: list of per-fold metric dicts.
        ``aggregated``: mean / std / median of each metric across folds.
        ``per_source_aggregated``: same breakdown by data source (if available).
    """
    per_fold = []
    for i, res in enumerate(fold_results):
        m = res.get("overall_metrics", {})
        per_fold.append(
            {
                "fold": i,
                "r2": m.get("r2"),
                "rmse": m.get("rmse"),
                "mae": m.get("mae"),
                "n_test": m.get("n") or res.get("config", {}).get("n_test"),
            }
        )

    agg: Dict[str, Any] = {}
    for key in ("r2", "rmse", "mae"):
        vals = [f[key] for f in per_fold if f[key] is not None]
        if vals:
            arr = np.array(vals)
            agg[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "median": float(np.median(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

    # Per-source aggregation
    per_source_agg: Dict[str, Dict] = {}
    for res in fold_results:
        source_metrics = res.get("source_metrics", [])
        for sm in source_metrics:
            src = sm.get("source", "unknown")
            if src not in per_source_agg:
                per_source_agg[src] = {"r2": [], "rmse": []}
            if sm.get("r2") is not None:
                per_source_agg[src]["r2"].append(sm["r2"])
            if sm.get("rmse") is not None:
                per_source_agg[src]["rmse"].append(sm["rmse"])

    per_source_summary = {}
    for src, vals in per_source_agg.items():
        entry: Dict[str, Any] = {}
        for key in ("r2", "rmse"):
            if vals[key]:
                arr = np.array(vals[key])
                entry[key] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                }
        per_source_summary[src] = entry

    return {
        "n_folds": len(fold_results),
        "per_fold": per_fold,
        "aggregated": agg,
        "per_source_aggregated": per_source_summary,
    }


def _print_summary(summary: Dict) -> None:
    """Print a human-readable k-fold summary."""
    print(f"\n{'=' * 60}")
    print(f"  K-FOLD SUMMARY  ({summary['n_folds']} folds)")
    print(f"{'=' * 60}")

    agg = summary.get("aggregated", {})
    for key in ("r2", "rmse", "mae"):
        if key in agg:
            s = agg[key]
            print(
                f"  {key.upper():>4s}: {s['mean']:.4f} ± {s['std']:.4f} "
                f"(median {s['median']:.4f}, range [{s['min']:.4f}, {s['max']:.4f}])"
            )

    print("\nPer fold:")
    for f in summary.get("per_fold", []):
        r2 = f.get("r2")
        rmse = f.get("rmse")
        n = f.get("n_test")
        r2_s = f"R2={r2:.4f}" if r2 is not None else "R2=N/A"
        rmse_s = f"RMSE={rmse:.4f}" if rmse is not None else "RMSE=N/A"
        n_s = f"n={n}" if n is not None else ""
        print(f"  fold {f['fold']}: {r2_s}, {rmse_s} {n_s}")

    per_src = summary.get("per_source_aggregated", {})
    if per_src:
        print("\nPer source (mean across folds):")
        for src, vals in sorted(per_src.items()):
            r2 = vals.get("r2", {}).get("mean")
            rmse = vals.get("rmse", {}).get("mean")
            r2_s = f"R2={r2:.4f}" if r2 is not None else "R2=N/A"
            rmse_s = f"RMSE={rmse:.4f}" if rmse is not None else "RMSE=N/A"
            print(f"  {src}: {r2_s}, {rmse_s}")
