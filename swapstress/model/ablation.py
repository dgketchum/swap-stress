"""Feature-group ablation study via 5-fold spatial CV on MGRS tiles.

Runs an all-groups-in baseline, then leave-one-out for each of the 9
feature groups present in the global training table.  10 experiments
total, each a full k-fold CV run.

Usage:
    python -m map.learning.decision_tree.ablation \
        --obs-table /nas/soils/swapstress/training/obs_level_training_9km_global.parquet \
        --output-dir /nas/soils/swapstress/releases/global_pruned_refresh_20260520/feature_importance
"""

from __future__ import annotations

import argparse
import json
import os
import time

from map.learning.decision_tree.train_direct import train_and_evaluate
from map.learning.direct.crossval import run_kfold_cv


# All feature groups present in the 9km global training table
ALL_GROUPS = [
    "worldclim",
    "landsat",
    "soilgrids",
    "fao",
    "global_et0",
    "terrain",
    "landcover",
    "sentinel1",
    "smap",
]


def build_experiments():
    """Return list of (name, exclude_groups) experiment configs.

    First entry is the all-in baseline (no exclusions), followed by
    one leave-one-out experiment per group.
    """
    experiments = [("baseline", [])]
    for group in ALL_GROUPS:
        experiments.append((f"drop_{group}", [group]))
    return experiments


def main(
    obs_table: str,
    output_dir: str,
    n_folds: int = 5,
    n_estimators: int = 250,
    random_state: int = 42,
    n_jobs: int = -1,
):
    os.makedirs(output_dir, exist_ok=True)
    experiments = build_experiments()

    print(f"Ablation study: {len(experiments)} experiments x {n_folds} folds")
    print(f"Groups under test: {ALL_GROUPS}\n")

    all_results = {}
    for i, (name, exclude_groups) in enumerate(experiments):
        exp_dir = os.path.join(output_dir, name)
        print(f"\n{'#' * 70}")
        print(f"  EXPERIMENT {i + 1}/{len(experiments)}: {name}")
        print(f"  Exclude: {exclude_groups}")
        print(f"{'#' * 70}\n")

        t0 = time.time()
        summary = run_kfold_cv(
            obs_table_path=obs_table,
            output_dir=exp_dir,
            n_folds=n_folds,
            holdout_col="MGRS_TILE",
            trainer_fn=train_and_evaluate,
            trainer_kwargs={
                "exclude_groups": exclude_groups,
                "n_estimators": n_estimators,
                "test_size": 0.2,
                "val_size": 0.2,
                "random_state": random_state,
                "drop_blocking_features": True,
                "resolution_m": 9000,
                "n_jobs": n_jobs,
            },
        )
        elapsed = time.time() - t0

        agg = summary["aggregated"]
        all_results[name] = {
            "exclude_groups": exclude_groups,
            "r2_mean": agg["r2"]["mean"],
            "r2_std": agg["r2"]["std"],
            "rmse_mean": agg["rmse"]["mean"],
            "rmse_std": agg["rmse"]["std"],
            "elapsed_s": round(elapsed, 1),
        }
        print(
            f"\n  {name}: R2={agg['r2']['mean']:.4f}±{agg['r2']['std']:.4f}  ({elapsed:.0f}s)"
        )

    summary_path = os.path.join(output_dir, "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote ablation summary to {summary_path}")

    # Print comparison table
    print(f"\n{'=' * 70}")
    print("  ABLATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Experiment':<20s} {'R²':>14s} {'RMSE':>14s} {'ΔR²':>8s}")
    print("-" * 60)

    baseline_r2 = all_results.get("baseline", {}).get("r2_mean")
    for name in ["baseline"] + [e[0] for e in build_experiments()]:
        if name not in all_results:
            continue
        r = all_results[name]
        r2_str = f"{r['r2_mean']:.4f}±{r['r2_std']:.4f}"
        rmse_str = f"{r['rmse_mean']:.4f}±{r['rmse_std']:.4f}"
        delta = ""
        if baseline_r2 is not None and name != "baseline":
            d = r["r2_mean"] - baseline_r2
            delta = f"{d:+.4f}"
        print(f"  {name:<18s} {r2_str:>14s} {rmse_str:>14s} {delta:>8s}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature-group ablation via 5-fold MGRS spatial CV",
    )
    parser.add_argument("--obs-table", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=250)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--n-jobs", type=int, default=-1, help="RF parallel jobs (-1 = all cores)"
    )
    args = parser.parse_args()

    main(
        obs_table=args.obs_table,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
