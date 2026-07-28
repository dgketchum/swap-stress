"""
Section 5.3: Climate-region holdout cross-validation.

Uses Beck et al. (2018) 30-class Koppen-Geiger classification sampled at
each training observation's lat/lon. Runs two levels of holdout:
  1. Major-zone (A/B/C/D/E) leave-one-out
  2. Sub-class (Cfa, BSk, Dfa, ...) leave-one-out for classes with >min_samples

Produces:
    - regional_cv_results.csv     (per-fold overall metrics)
    - regional_cv_summary.png     (bar chart by region)
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
from sklearn.impute import SimpleImputer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swapstress.validation.reconstruct_test_set import sample_beck_koppen
from swapstress.model.metrics import compute_metrics

MODEL_DIR = "/nas/soils/swapstress/models/direct_rf_9km_global_pruned"


def run_regional_cv(
    obs_table: str,
    all_features: list[str],
    n_estimators: int = 250,
    random_state: int = 42,
    min_samples: int = 100,
    level: str = "major",
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Run leave-one-climate-region-out CV using Beck et al. (2018) Koppen.

    Uses the saved feature list from the model directory so that feature
    selection is not re-derived from the full table (which would leak
    held-out information into preprocessing).

    Parameters
    ----------
    level : str
        "major" for 5 major zones (A-E), "subclass" for 30 Beck sub-classes.

    Returns
    -------
    pd.DataFrame
        One row per held-out region with overall metrics.
    """
    df = pd.read_parquet(obs_table)
    df = df.dropna(subset=["theta", "log10_suction_cm", "lat", "lon"])

    # Must have lat/lon for Beck sampling
    df = df.dropna(subset=["lat", "lon"])

    codes, labels, major = sample_beck_koppen(df["lat"].values, df["lon"].values)

    if level == "major":
        df["region"] = major
    else:
        df["region"] = labels

    # Drop unknowns (ocean/nodata)
    df = df[df["region"] != "unknown"]

    regions = sorted(df["region"].unique())
    region_counts = df["region"].value_counts()
    print(f"Level: {level}")
    print(f"Regions ({len(regions)}): {dict(region_counts)}")

    result_rows = []
    for held_out in regions:
        n_test = region_counts.get(held_out, 0)
        if n_test < min_samples:
            print(f"  Skipping {held_out}: only {n_test} samples (< {min_samples})")
            continue

        print(f"\nRegion holdout: {held_out} ({n_test} samples)")

        train_df = df[df["region"] != held_out]
        test_df = df[df["region"] == held_out]

        X_train = train_df[all_features].values.astype(np.float32)
        X_test = test_df[all_features].values.astype(np.float32)
        y_train = train_df["log10_suction_cm"].values
        y_test = test_df["log10_suction_cm"].values

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        rf = RandomForestRegressor(
            n_estimators=n_estimators, n_jobs=n_jobs, random_state=random_state
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        metrics = compute_metrics(y_test, y_pred)
        print(
            f"  R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, "
            f"bias={metrics['bias']:.4f}"
        )

        result_rows.append(
            {
                "held_out_region": held_out,
                "n_train": len(train_df),
                "n_test": len(test_df),
                **metrics,
            }
        )

    return pd.DataFrame(result_rows)


def plot_regional_summary(
    results_df: pd.DataFrame,
    output_dir: str,
    baseline_metrics: dict | None = None,
    filename: str = "regional_cv_summary.png",
) -> str:
    """Bar chart of per-region R2 and RMSE."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    regions = results_df["held_out_region"].values
    x = np.arange(len(regions))

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
    ax.set_xticklabels(regions, rotation=30, ha="right")
    ax.set_ylabel("R$^2$")
    ax.set_title("Climate-region holdout: R$^2$")

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
    ax.set_xticklabels(regions, rotation=30, ha="right")
    ax.set_ylabel("RMSE (log$_{10}$ cm)")
    ax.set_title("Climate-region holdout: RMSE")

    fig.tight_layout()
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Section 5.3: climate-region holdout CV (Beck et al. 2018 Koppen)"
    )
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
    parser.add_argument(
        "--min-samples",
        type=int,
        default=100,
        help="Minimum samples in a region to include it (default: 100).",
    )
    parser.add_argument(
        "--level",
        choices=["major", "subclass", "both"],
        default="both",
        help="Holdout level: major (A-E), subclass (Cfa, BSk, ...), or both.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel RF workers (default: -1 = all cores).",
    )
    args = parser.parse_args()

    model_path = Path(args.model_dir)
    output_dir = args.output_dir or os.path.join(args.model_dir, "error_analysis")
    os.makedirs(output_dir, exist_ok=True)

    with open(model_path / "direct_model_results.json") as f:
        model_results = json.load(f)

    config = model_results["config"]
    baseline = model_results["overall_metrics"]

    with open(model_path / "direct_rf_features.json") as f:
        all_features = json.load(f)

    levels = ["major", "subclass"] if args.level == "both" else [args.level]

    all_results = []
    for level in levels:
        print(f"\n{'=' * 60}")
        print(f"Running {level}-level regional CV")
        print("=" * 60)

        results_df = run_regional_cv(
            obs_table=config["obs_table"],
            all_features=all_features,
            n_estimators=args.n_estimators,
            random_state=config.get("random_state", 42),
            min_samples=args.min_samples,
            level=level,
            n_jobs=args.n_jobs,
        )
        results_df["level"] = level

        suffix = f"_{level}" if args.level == "both" else ""
        csv_path = os.path.join(output_dir, f"regional_cv_results{suffix}.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")

        print(f"\nBaseline: R2={baseline['r2']:.4f}, RMSE={baseline['rmse']:.4f}")
        for _, row in results_df.iterrows():
            ratio = row["rmse"] / baseline["rmse"]
            print(f"  {row['held_out_region']}: RMSE ratio = {ratio:.2f}x baseline")

        plot_regional_summary(
            results_df,
            output_dir,
            baseline_metrics=baseline,
            filename=f"regional_cv_summary{suffix}.png",
        )
        all_results.append(results_df)

    if len(all_results) > 1:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(
            os.path.join(output_dir, "regional_cv_results.csv"), index=False
        )


if __name__ == "__main__":
    main()
