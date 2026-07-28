"""Figure 4: validation scatter -- the direct model against PTF baselines.

Observed vs predicted log10 suction for three estimators on the same
observations: our direct RF, Rosetta, and POLARIS. The PTF columns come from
``swapstress.validation.ptf_baseline``, which pushes each site's published van
Genuchten parameters through the retention equation at the observed theta.

**Held-out rows only.** ``ptf_baseline evaluate`` falls back to predicting every
row with the single fitted model when no k-fold artifacts are present, and that
branch includes the rows the model trained on -- the released
``ptf_comparison_observations.parquet`` was produced that way. Plotting it as-is
would put an in-sample RF beside genuinely out-of-sample PTFs, which is the
first thing a reviewer probes. This module therefore keeps only the
observations that fall in the model's spatial holdout, so all three estimators
are being asked the same out-of-sample question.

Usage:
    uv run swapstress-figures --figure validation-scatter
    uv run python -m swapstress.figures.fig04_validation_scatter --model-dir <dir>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_MODEL_DIR = "/nas/soils/swapstress/models/direct_rf_9km_global_pruned"
DEFAULT_PTF_DIR = "/nas/soils/swapstress/evaluation/ptf_baseline"
DEFAULT_OUTPUT_DIR = "figs/descriptor"

# The columns that identify one observation in both tables.
JOIN_KEYS = ["sample_id", "source", "theta", "log10_suction_cm"]

ESTIMATORS = [
    ("rf_pred", "SWAP direct RF", "#1f4e79"),
    ("ros_log10_suction", "Rosetta", "#b5651d"),
    ("pol_log10_suction", "POLARIS", "#7a4b8f"),
]

AXIS_LIMITS = (0.0, 7.0)


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    """RMSE, bias and R2 over the pairs where both are finite."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    keep = np.isfinite(observed) & np.isfinite(predicted)
    observed, predicted = observed[keep], predicted[keep]
    residual = predicted - observed
    ss_res = float((residual**2).sum())
    ss_tot = float(((observed - observed.mean()) ** 2).sum())
    return {
        "n": int(keep.sum()),
        "rmse": float(np.sqrt((residual**2).mean())),
        "bias": float(residual.mean()),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
    }


def load_holdout(model_dir: str, ptf_dir: str) -> pd.DataFrame:
    """PTF comparison rows restricted to the model's spatial holdout.

    ``predictions.parquet`` is row-aligned with ``test_set_full.parquet``; that
    is checked rather than assumed, because a silent misalignment would produce
    a plausible-looking and entirely wrong scatter.
    """
    model = Path(model_dir)
    test = pd.read_parquet(model / "test_set_full.parquet")
    preds = pd.read_parquet(model / "predictions.parquet")
    if len(test) != len(preds):
        raise ValueError(
            f"{model.name}: test_set_full has {len(test):,} rows and "
            f"predictions has {len(preds):,}; they must be row-aligned."
        )
    if not np.allclose(
        test["log10_suction_cm"].values, preds["observed"].values, equal_nan=True
    ):
        raise ValueError(
            f"{model.name}: predictions.parquet is not row-aligned with "
            "test_set_full.parquet -- 'observed' does not match "
            "'log10_suction_cm'."
        )

    holdout = test[JOIN_KEYS].copy()
    holdout["rf_pred"] = preds["predicted"].values
    holdout = holdout.drop_duplicates(subset=JOIN_KEYS)

    ptf = pd.read_parquet(Path(ptf_dir) / "ptf_comparison_observations.parquet")
    ptf = ptf.drop_duplicates(subset=JOIN_KEYS)

    merged = ptf.merge(holdout, on=JOIN_KEYS, how="inner")
    print(
        f"{len(ptf):,} PTF observations, {len(holdout):,} held-out rows, "
        f"{len(merged):,} in both"
    )
    if merged.empty:
        raise ValueError(
            "No PTF observations fall in the model's spatial holdout; the "
            "figure would have nothing out-of-sample to show."
        )
    return merged


def render(df: pd.DataFrame, output_dir: str) -> Path:
    observed = df["log10_suction_cm"].values
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.6), sharex=True, sharey=True)

    for ax, (column, label, color) in zip(axes, ESTIMATORS):
        predicted = df[column].values
        stats = metrics(observed, predicted)
        ax.scatter(
            observed,
            predicted,
            s=6,
            alpha=0.18,
            color=color,
            edgecolors="none",
            rasterized=True,
        )
        ax.plot(AXIS_LIMITS, AXIS_LIMITS, "k--", linewidth=0.9, zorder=3)
        ax.set_xlim(*AXIS_LIMITS)
        ax.set_ylim(*AXIS_LIMITS)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel(r"Observed $\log_{10}$ suction (cm)", fontsize=9)
        ax.tick_params(labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.text(
            0.04,
            0.96,
            f"n = {stats['n']:,}\n"
            f"RMSE = {stats['rmse']:.2f}\n"
            f"bias = {stats['bias']:+.2f}\n"
            f"R$^2$ = {stats['r2']:+.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3),
        )

    axes[0].set_ylabel(r"Predicted $\log_{10}$ suction (cm)", fontsize=9)
    fig.suptitle(
        "Spatial holdout: the direct model against PTF-derived retention",
        fontsize=12,
        y=1.0,
    )
    fig.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / "fig04_validation_scatter"
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return Path(f"{stem}.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fig04_validation_scatter",
        description="Figure 4: held-out validation scatter against PTF baselines.",
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--ptf-dir", default=DEFAULT_PTF_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    df = load_holdout(args.model_dir, args.ptf_dir)
    observed = df["log10_suction_cm"].values
    for column, label, _ in ESTIMATORS:
        stats = metrics(observed, df[column].values)
        print(
            f"  {label:16} n={stats['n']:5,d}  RMSE={stats['rmse']:.3f}  "
            f"bias={stats['bias']:+.3f}  R2={stats['r2']:+.3f}"
        )
    path = render(df, args.output_dir)
    print(f"Saved to {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
