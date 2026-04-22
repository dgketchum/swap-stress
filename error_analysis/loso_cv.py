"""
Phase 1B: Leave-one-source-out cross-validation (transportability test).

Trains RF on 4 of 5 sources, tests on the held-out 5th. Repeats for all 5.
Reports per-fold R2, RMSE, MAE, bias, and conditional bias by theta-decile.

NOTE: LOSO is a compound domain-shift test, not a clean method-bias experiment.
Holding out a source removes its geography, climate, texture, and measurement
physics simultaneously. Results characterize model transportability across domains,
not measurement-method bias in isolation.

Produces:
    - loso_results.csv        (per-fold overall metrics)
    - loso_conditional.csv    (per-fold conditional bias by theta-decile)
    - loso_summary.png        (bar chart comparing folds)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from map.learning.direct.metrics import compute_metrics

from error_analysis.conditional_bias import compute_conditional_bias

MODEL_DIR = "/nas/soils/swapstress/models/direct_rf_9km_global_pruned"


def run_loso(
    obs_table: str,
    all_features: list[str],
    n_estimators: int = 250,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run leave-one-source-out CV.

    Uses the saved feature list from the model directory so that feature
    selection is not re-derived from the full table (which would leak
    held-out information into preprocessing).

    Returns
    -------
    results_df : pd.DataFrame
        One row per held-out source with overall metrics.
    conditional_df : pd.DataFrame
        Conditional bias by theta-decile per fold.
    """
    full_df = pd.read_parquet(obs_table)
    full_df = full_df.dropna(subset=["theta", "log10_suction_cm", "lat", "lon"])
    sources = sorted(full_df["source"].unique())
    print(f"Sources: {sources}")

    result_rows = []
    conditional_rows = []

    for held_out in sources:
        print(f"\n{'=' * 60}")
        print(f"LOSO fold: holding out {held_out}")

        train_df = full_df[full_df["source"] != held_out]
        test_df = full_df[full_df["source"] == held_out]

        if len(test_df) < 10:
            print(f"  Skipping {held_out}: only {len(test_df)} test samples")
            continue

        print(
            f"  Train: {len(train_df)} samples from {train_df['source'].nunique()} sources"
        )
        print(f"  Test: {len(test_df)} samples ({held_out})")

        # Impute and train
        X_train = train_df[all_features].values.astype(np.float32)
        X_test = test_df[all_features].values.astype(np.float32)
        y_train = train_df["log10_suction_cm"].values
        y_test = test_df["log10_suction_cm"].values

        from sklearn.impute import SimpleImputer

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        rf = RandomForestRegressor(
            n_estimators=n_estimators, n_jobs=-1, random_state=random_state
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        metrics = compute_metrics(y_test, y_pred)
        print(
            f"  R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, bias={metrics['bias']:.4f}"
        )

        result_rows.append(
            {
                "held_out_source": held_out,
                "n_train": len(train_df),
                "n_test": len(test_df),
                **metrics,
            }
        )

        # Conditional bias for this fold
        fold_df = test_df.copy()
        fold_df["observed"] = y_test
        fold_df["predicted"] = y_pred
        cond = compute_conditional_bias(fold_df, n_bins=10)
        cond["held_out_source"] = held_out
        conditional_rows.append(cond)

    results_df = pd.DataFrame(result_rows)
    conditional_df = pd.concat(conditional_rows, ignore_index=True)
    return results_df, conditional_df


def plot_loso_summary(
    results_df: pd.DataFrame,
    output_dir: str,
    baseline_metrics: dict | None = None,
    filename: str = "loso_summary.png",
) -> str:
    """Bar chart of per-fold R2 and RMSE with baseline reference."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sources = results_df["held_out_source"].values
    x = np.arange(len(sources))

    # R2
    ax = axes[0]
    ax.bar(x, results_df["r2"], color="C0", alpha=0.8)
    if baseline_metrics:
        ax.axhline(
            baseline_metrics["r2"],
            color="k",
            linestyle="--",
            linewidth=1,
            label="baseline",
        )
        ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(sources, rotation=30, ha="right")
    ax.set_ylabel("R$^2$")
    ax.set_title("LOSO: R$^2$ on held-out source")

    # RMSE
    ax = axes[1]
    ax.bar(x, results_df["rmse"], color="C1", alpha=0.8)
    if baseline_metrics:
        ax.axhline(
            baseline_metrics["rmse"],
            color="k",
            linestyle="--",
            linewidth=1,
            label="baseline",
        )
        ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(sources, rotation=30, ha="right")
    ax.set_ylabel("RMSE (log$_{10}$ cm)")
    ax.set_title("LOSO: RMSE on held-out source")

    fig.tight_layout()
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Phase 1B: leave-one-source-out CV")
    parser.add_argument(
        "--model-dir",
        default=MODEL_DIR,
        help="Path to trained model directory (for config and baseline).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <model-dir>/error_analysis/).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=250,
        help="Number of RF trees per fold (default: 250).",
    )
    args = parser.parse_args()

    model_path = Path(args.model_dir)
    output_dir = args.output_dir or os.path.join(args.model_dir, "error_analysis")
    os.makedirs(output_dir, exist_ok=True)

    with open(model_path / "direct_model_results.json") as f:
        model_results = json.load(f)

    config = model_results["config"]

    with open(model_path / "direct_rf_features.json") as f:
        all_features = json.load(f)

    results_df, conditional_df = run_loso(
        obs_table=config["obs_table"],
        all_features=all_features,
        n_estimators=args.n_estimators,
        random_state=config.get("random_state", 42),
    )

    results_df.to_csv(os.path.join(output_dir, "loso_results.csv"), index=False)
    conditional_df.to_csv(os.path.join(output_dir, "loso_conditional.csv"), index=False)

    print(f"\n{'=' * 60}")
    print("LOSO Results:")
    print(results_df.to_string(index=False))

    baseline = model_results["overall_metrics"]
    print(
        f"\nBaseline (all sources): R2={baseline['r2']:.4f}, RMSE={baseline['rmse']:.4f}"
    )
    for _, row in results_df.iterrows():
        ratio = row["rmse"] / baseline["rmse"]
        print(f"  {row['held_out_source']}: RMSE ratio = {ratio:.2f}x baseline")

    plot_loso_summary(results_df, output_dir, baseline_metrics=baseline)


if __name__ == "__main__":
    main()
