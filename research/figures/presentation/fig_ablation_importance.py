"""Feature Group Importance bar chart from ablation results.

Reads the existing ablation_summary.json and optionally runs a
theta-ablation experiment to include theta as a bar.

Usage:
    uv run python viz/presentation/fig_ablation_importance.py
    uv run python viz/presentation/fig_ablation_importance.py --run-theta-ablation
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ABLATION_DIR = Path(
    "/nas/soils/swapstress/releases/global_pruned_refresh_20260520/feature_importance"
)
OBS_TABLE = Path("/nas/soils/swapstress/training/obs_level_training_9km_global.parquet")

# Pretty labels for groups
GROUP_LABELS = {
    "theta": r"$\theta$ (SMAP L3)",
    "soilgrids": "SoilGrids",
    "fao": "FAO",
    "worldclim": "WorldClim",
    "terrain": "Terrain",
    "landsat": "Landsat",
    "landcover": "Landcover",
    "global_et0": "Global ET0",
    "smap": "SMAP",
    "sentinel1": "Sentinel-1",
}


def run_theta_ablation(ablation_dir: Path, obs_table: Path) -> dict:
    """Run 5-fold CV without theta and return the result dict."""
    from swapstress.model.crossval import run_kfold_cv
    from swapstress.model.data import prepare_direct_data

    exp_dir = ablation_dir / "drop_theta"

    # We need a custom trainer that excludes theta from the feature list.
    def train_no_theta(obs_table_path, output_dir, **kwargs):
        """Wrapper that strips theta from all_features before training."""
        import os
        import pandas as pd
        from swapstress.model.preprocessing import prepare_rf_arrays
        from swapstress.model.reporting import evaluate_and_report
        from sklearn.ensemble import RandomForestRegressor

        data = prepare_direct_data(
            obs_table_path=obs_table_path,
            output_dir=output_dir,
            exclude_groups=kwargs.get("exclude_groups"),
            drop_blocking_features=kwargs.get("drop_blocking_features", True),
            resolution_m=kwargs.get("resolution_m", 9000),
            test_size=kwargs.get("test_size", 0.2),
            val_size=kwargs.get("val_size", 0.2),
            random_state=kwargs.get("random_state", 42),
            holdout_col=kwargs.get("holdout_col"),
            n_folds=kwargs.get("n_folds", 5),
            test_fold=kwargs.get("test_fold", 0),
        )

        # Remove theta from features
        all_features = [f for f in data["all_features"] if f != "theta"]
        print(f"  Training WITHOUT theta: {len(all_features)} features")

        train_df = data["train_df"]
        if data.get("val_df") is not None:
            train_df = pd.concat([train_df, data["val_df"]], ignore_index=True)

        test_df = data["test_df"]
        train_sites = set(data["train_sites"])
        if data.get("val_sites"):
            train_sites = train_sites | data["val_sites"]

        X_train, X_test, y_train, y_test, imputer = prepare_rf_arrays(
            train_df,
            test_df,
            all_features,
        )

        n_estimators = kwargs.get("n_estimators", 250)
        n_jobs = kwargs.get("n_jobs", -1)
        random_state = kwargs.get("random_state", 42)

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        os.makedirs(output_dir, exist_ok=True)
        results = evaluate_and_report(
            y_test,
            y_pred,
            test_df,
            output_dir,
            all_features=all_features,
            train_df=train_df,
            train_sites=train_sites,
            test_sites=data["test_sites"],
            resolution_m=kwargs.get("resolution_m", 9000),
            holdout_col=kwargs.get("holdout_col"),
        )
        return results

    print("Running theta ablation (5-fold CV without theta)...")
    t0 = time.time()
    summary = run_kfold_cv(
        obs_table_path=str(obs_table),
        output_dir=str(exp_dir),
        n_folds=5,
        holdout_col="MGRS_TILE",
        trainer_fn=train_no_theta,
        trainer_kwargs={
            "n_estimators": 250,
            "test_size": 0.2,
            "val_size": 0.2,
            "random_state": 42,
            "drop_blocking_features": True,
            "resolution_m": 9000,
            "n_jobs": -1,
        },
    )
    elapsed = time.time() - t0

    agg = summary["aggregated"]
    result = {
        "exclude_groups": ["theta"],
        "r2_mean": agg["r2"]["mean"],
        "r2_std": agg["r2"]["std"],
        "rmse_mean": agg["rmse"]["mean"],
        "rmse_std": agg["rmse"]["std"],
        "elapsed_s": round(elapsed, 1),
    }
    print(f"  drop_theta: R2={result['r2_mean']:.4f} ± {result['r2_std']:.4f}")
    return result


def plot_importance(ablation_data: dict, output_path: Path):
    """Two-tier bar chart: theta alone on top, covariate groups below."""
    baseline_r2 = ablation_data["baseline"]["r2_mean"]

    # Separate theta from covariate groups
    theta_delta = None
    cov_groups = []
    cov_deltas = []
    for key, val in ablation_data.items():
        if key == "baseline":
            continue
        group = key.replace("drop_", "")
        delta = baseline_r2 - val["r2_mean"]
        if group == "theta":
            theta_delta = delta
        else:
            cov_groups.append(group)
            cov_deltas.append(delta)

    # Normalize covariates among themselves
    cov_total = sum(abs(d) for d in cov_deltas)
    cov_normed = [d / cov_total for d in cov_deltas]

    # Sort covariates by importance descending
    order = np.argsort(cov_normed)[::-1]
    cov_groups = [cov_groups[i] for i in order]
    cov_normed = [cov_normed[i] for i in order]
    cov_labels = [GROUP_LABELS.get(g, g) for g in cov_groups]
    cov_colors = ["#2980B9" if v >= 0 else "#C0392B" for v in cov_normed]

    n_cov = len(cov_groups)

    # Layout: theta bar on top axis, covariate bars on bottom axis
    fig, (ax_theta, ax_cov) = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        gridspec_kw={"height_ratios": [1, n_cov], "hspace": 0.0},
    )

    fig.suptitle(
        "Feature Group Importance (Leave-One-Out Ablation, 5-Fold CV)", fontsize=14
    )

    # --- Theta bar (top) ---
    theta_r2_drop = f"$\\Delta R^2$ = {theta_delta:.3f}"
    ax_theta.barh(
        [0],
        [1.0],
        color="#2980B9",
        edgecolor="none",
    )
    ax_theta.text(
        0.5,
        0,
        theta_r2_drop,
        va="center",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="white",
    )
    ax_theta.set_yticks([0])
    ax_theta.set_yticklabels([GROUP_LABELS["theta"]], fontsize=11)
    ax_theta.set_xlim(0, 1.05)
    ax_theta.set_xticks([])

    # --- Covariate bars (bottom) ---
    bars = ax_cov.barh(range(n_cov), cov_normed, color=cov_colors, edgecolor="none")
    ax_cov.set_yticks(range(n_cov))
    ax_cov.set_yticklabels(cov_labels, fontsize=11)
    ax_cov.invert_yaxis()
    ax_cov.axvline(0, color="k", lw=0.8)
    ax_cov.set_xlabel("Normalized Importance (among static covariates)", fontsize=11)

    for bar, val in zip(bars, cov_normed):
        x = bar.get_width()
        # Place label inside bar when it's long enough, outside otherwise
        if abs(x) > 0.15:
            offset = -0.008 if x >= 0 else 0.008
            ha = "right" if x >= 0 else "left"
            color = "white"
        else:
            offset = 0.008 if x >= 0 else -0.008
            ha = "left" if x >= 0 else "right"
            color = "k"
        ax_cov.text(
            x + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha=ha,
            fontsize=10,
            color=color,
            fontweight="bold" if color == "white" else "normal",
        )

    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-theta-ablation",
        action="store_true",
        help="Run the theta ablation experiment (slow)",
    )
    parser.add_argument("--output-dir", default="figs/presentation")
    args = parser.parse_args()

    summary_path = ABLATION_DIR / "ablation_summary.json"
    with open(summary_path) as f:
        ablation_data = json.load(f)

    if args.run_theta_ablation:
        theta_result = run_theta_ablation(ABLATION_DIR, OBS_TABLE)
        ablation_data["drop_theta"] = theta_result
        # Save updated summary
        updated_path = ABLATION_DIR / "ablation_summary.json"
        with open(updated_path, "w") as f:
            json.dump(ablation_data, f, indent=2)
        print(f"Updated {updated_path}")
    elif "drop_theta" not in ablation_data:
        print("WARNING: theta ablation not found. Run with --run-theta-ablation")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_importance(ablation_data, out_dir / "fig_ablation_importance.png")


if __name__ == "__main__":
    main()
