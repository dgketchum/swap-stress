"""
Compare RF and NN runs by reading their direct_model_results.json files.

Usage:
    python -m research.neural.tabular_nn.compare_runs \\
        --run-dirs /nas/.../direct_rf_9km_conus /nas/.../direct_mlp_9km_conus \\
        --output comparison.csv
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd


def load_run(run_dir: str) -> dict:
    """Load direct_model_results.json from a run directory."""
    path = os.path.join(run_dir, "direct_model_results.json")
    with open(path) as f:
        results = json.load(f)
    results["run_dir"] = run_dir
    results["run_name"] = os.path.basename(run_dir)
    return results


def build_comparison_table(run_dirs: list[str]) -> pd.DataFrame:
    """Build a comparison table across multiple runs.

    Returns a DataFrame with one row per run, columns for overall
    and site-weighted metrics.
    """
    rows = []
    for d in run_dirs:
        r = load_run(d)
        om = r["overall_metrics"]
        sm = r.get("site_weighted_metrics", {})

        row = {
            "run": r["run_name"],
            "run_dir": r["run_dir"],
            "model_family": r.get("model_family", "rf"),
            "model_name": r.get("model_name", "random_forest"),
            "r2": om.get("r2"),
            "rmse": om.get("rmse"),
            "mae": om.get("mae"),
            "bias": om.get("bias"),
            "n": om.get("n"),
            "site_mean_r2": sm.get("mean_r2"),
            "site_median_r2": sm.get("median_r2"),
            "site_mean_rmse": sm.get("mean_rmse"),
            "n_sites": sm.get("n_sites"),
        }

        # NN-specific fields
        if "validation_metrics" in r and r["validation_metrics"]:
            row["val_rmse"] = r["validation_metrics"].get("val_rmse")

        config = r.get("config", {})
        training = r.get("training_summary", {})
        row["n_features"] = config.get("n_features")
        row["n_train"] = config.get("n_train")
        row["n_test"] = config.get("n_test")
        row["lambda_mono"] = config.get("lambda_mono", 0.0)
        row["lambda_bound"] = config.get("lambda_bound", 0.0)
        row["bound_lo"] = config.get("bound_lo")
        row["bound_hi"] = config.get("bound_hi")
        row["best_val_rmse"] = training.get("best_val_rmse")
        row["best_epoch"] = training.get("best_epoch")
        row["stopped_epoch"] = training.get("stopped_epoch")

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RF and NN runs.")
    parser.add_argument(
        "--run-dirs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to run output directories.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write comparison CSV.",
    )
    args = parser.parse_args()

    df = build_comparison_table(args.run_dirs)
    print(df.to_string(index=False))

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
