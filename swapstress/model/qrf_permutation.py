"""Permutation importance of the released QRF, from its saved artifacts.

Loads the released model, its imputer and feature list, and its spatial
holdout (``test_set_full.parquet``) and permutes one feature at a time --
no retraining. The score is R2 of the *median* (0.5 quantile) prediction,
because the median is the released value; importance is the baseline R2
minus the mean permuted R2 over ``--n-repeats`` shuffles.

This is the run behind descriptor Fig 3's importance panel. The earlier
importance artifacts (``releases/archive/pre_global_pruned_refresh_20260520``)
describe pre-release models and stay quarantined: a descriptor figure has to
describe the model whose predictions are in the released files.

Outputs, in ``--output-dir``:
    permutation_importance.csv          feature, importance_mean/std, group
    group_importance_permutation.json   normalized share by landscape group
    permutation_meta.json               baseline R2, n rows, repeats, timing

A full run is ~650 median predictions over the 41,914-row holdout and takes
hours; progress prints one line per feature so ``tail -f`` shows where it is.

Usage:
    uv run python -m swapstress.model.qrf_permutation \
        --model-dir /nas/soils/swapstress/models/direct_qrf_9km_global_pruned \
        --output-dir /nas/soils/swapstress/releases/v03_20260729/evaluation/feature_importance
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from swapstress.features.features import (
    aggregate_importance_by_group,
    classify_feature,
)

DEFAULT_MODEL_DIR = "/nas/soils/swapstress/models/direct_qrf_9km_global_pruned"
DEFAULT_OUTPUT_DIR = (
    "/nas/soils/swapstress/releases/v03_20260729/evaluation/feature_importance"
)

MEDIAN_QUANTILE = 0.5


def load_artifacts(model_dir: Path):
    """The released model, its imputer, feature order, and holdout table."""
    model = joblib.load(model_dir / "direct_rf_model.joblib")
    imputer = joblib.load(model_dir / "direct_rf_imputer.joblib")
    with open(model_dir / "direct_rf_features.json") as f:
        features = json.load(f)
    test = pd.read_parquet(model_dir / "test_set_full.parquet")
    missing = [c for c in features if c not in test.columns]
    if missing:
        raise ValueError(
            f"{model_dir.name}: test_set_full.parquet lacks feature columns "
            f"{missing}; the artifacts are not from the same training run."
        )
    return model, imputer, features, test


def median_r2(model, X: np.ndarray, y: np.ndarray) -> float:
    """R2 of the released statistic: the forest's median prediction."""
    pred = np.asarray(model.predict(X, quantiles=MEDIAN_QUANTILE)).ravel()
    return float(r2_score(y, pred))


def run(model_dir: str, output_dir: str, n_repeats: int, seed: int) -> None:
    model_path = Path(model_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading artifacts from {model_path} ...", flush=True)
    model, imputer, features, test = load_artifacts(model_path)
    # Prediction parallelism follows the machine this runs on, not whatever
    # n_jobs the model was trained with.
    model.n_jobs = -1

    X = imputer.transform(test[features].values)
    y = test["log10_suction_cm"].values
    print(f"{len(y):,} held-out rows, {len(features)} features", flush=True)

    t0 = time.time()
    baseline = median_r2(model, X, y)
    t_pred = time.time() - t0
    total = len(features) * n_repeats * t_pred
    print(
        f"Baseline median R2 = {baseline:.4f}  ({t_pred:.1f} s per prediction; "
        f"~{total / 3600:.1f} h projected for {len(features)} x {n_repeats})",
        flush=True,
    )

    rng = np.random.default_rng(seed)
    rows = []
    start = time.time()
    for j, name in enumerate(features):
        saved = X[:, j].copy()
        scores = []
        for _ in range(n_repeats):
            X[:, j] = rng.permutation(saved)
            scores.append(median_r2(model, X, y))
        X[:, j] = saved
        drop = baseline - float(np.mean(scores))
        rows.append(
            {
                "feature": name,
                "importance_mean": drop,
                "importance_std": float(np.std(scores)),
                "group": classify_feature(name),
            }
        )
        elapsed = time.time() - start
        print(
            f"[{j + 1:3d}/{len(features)}] {name:<40s} {drop:+.4f}  "
            f"({elapsed / 60:.1f} min elapsed)",
            flush=True,
        )

    df = pd.DataFrame(rows).sort_values("importance_mean", ascending=False)
    csv_path = out / "permutation_importance.csv"
    df.to_csv(csv_path, index=False)

    # Group shares over the landscape covariates alone: theta and the fixed
    # sample descriptors are inputs of a different kind, reported as their own
    # rows in the CSV rather than folded into the normalization.
    landscape = df[~df["group"].isin(["theta", "depth"])]
    group_importance = aggregate_importance_by_group(
        dict(zip(landscape["feature"], landscape["importance_mean"])), normalize=True
    )
    with open(out / "group_importance_permutation.json", "w") as f:
        json.dump(group_importance, f, indent=2)

    meta = {
        "model_dir": str(model_path),
        "n_test": int(len(y)),
        "n_features": len(features),
        "n_repeats": n_repeats,
        "seed": seed,
        "baseline_r2_median": baseline,
        "elapsed_s": round(time.time() - start, 1),
    }
    with open(out / "permutation_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {csv_path}", flush=True)
    print("Top 10:", flush=True)
    for _, row in df.head(10).iterrows():
        print(
            f"  {row['feature']:<40s} {row['importance_mean']:+.4f} ({row['group']})",
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qrf_permutation",
        description="Permutation importance of the released QRF on its holdout.",
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    run(args.model_dir, args.output_dir, args.n_repeats, args.seed)


if __name__ == "__main__":
    main()
