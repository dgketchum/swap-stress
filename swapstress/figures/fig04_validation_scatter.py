"""Figure 4: validation scatter -- the direct model against PTF baselines.

Observed vs predicted log10 suction for three estimators on the same
observations: our direct quantile RF (its median, which is the released Level 1
value), Rosetta, and POLARIS. The PTF columns come from
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

Drawn to ``swapstress.figures.style``: 183 mm double-column, panel labels at
8 pt bold and everything else between 5 and 7 pt, one sans typeface throughout,
and only the point clouds rasterised.

The defaults point at the 0.3 release. ``ptf_baseline evaluate`` writes
``ptf_comparison_observations.parquet`` straight into the directory given as its
``--output`` -- it makes no subdirectory of its own -- so the release rerun
against ``.../releases/v03_20260729/evaluation`` leaves the table there, beside
the coverage tables the other stage-04 analyses wrote.

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

from swapstress.figures import style

DEFAULT_MODEL_DIR = "/nas/soils/swapstress/models/direct_qrf_9km_global_pruned"
DEFAULT_PTF_DIR = "/nas/soils/swapstress/releases/v03_20260729/evaluation"
DEFAULT_OUTPUT_DIR = "figs/descriptor"

# The columns that identify one observation in both tables.
JOIN_KEYS = ["sample_id", "source", "theta", "log10_suction_cm"]

# Colours are the validated categorical trio, taken in a fixed order so each
# estimator keeps its hue across the descriptor.
ESTIMATORS = [
    ("rf_pred", "SWAP direct QRF", style.CATEGORICAL[0]),
    ("ros_log10_suction", "Rosetta", style.CATEGORICAL[1]),
    ("pol_log10_suction", "POLARIS", style.CATEGORICAL[2]),
]

PANEL_LETTERS = ("a", "b", "c")

AXIS_LIMITS = (0.0, 7.0)

# The two-column width. The height is chosen so three equal-aspect panels fill
# that width exactly: any shorter and the square panels shrink, leaving gaps at
# the sides; any taller and the extra is dead space under the axes.
FIGURE_WIDTH_MM = style.DOUBLE_COLUMN_MM
FIGURE_HEIGHT_MM = 66.0


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


def off_scale(observed: np.ndarray, predicted: np.ndarray) -> int:
    """Pairs the metrics count but the axes cannot show.

    Rosetta and POLARIS put a sixth of their predictions outside 0-7 log10 cm,
    far beyond anything the observations reach. They stay in the RMSE and R2 --
    they are real errors -- but they land off the panel, so the count is printed
    in the corner rather than left to be silently cropped.
    """
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    keep = np.isfinite(observed) & np.isfinite(predicted)
    low, high = AXIS_LIMITS
    inside = (predicted >= low) & (predicted <= high)
    inside &= (observed >= low) & (observed <= high)
    return int((keep & ~inside).sum())


def render(df: pd.DataFrame, output_dir: str) -> Path:
    style.apply()
    observed = df["log10_suction_cm"].values
    fig, axes = plt.subplots(
        1,
        3,
        figsize=style.figsize(FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM),
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    for ax, letter, (column, label, color) in zip(axes, PANEL_LETTERS, ESTIMATORS):
        predicted = df[column].values
        stats = metrics(observed, predicted)
        # Rasterised marks only: the axes, the 1:1 line and every label below
        # stay vector, which is what the artwork guide asks for.
        ax.scatter(
            observed,
            predicted,
            s=2.0,
            alpha=0.22,
            color=color,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )
        ax.plot(
            AXIS_LIMITS,
            AXIS_LIMITS,
            color="black",
            linewidth=0.6,
            dashes=(2.6, 1.8),
            zorder=3,
        )
        ax.set_xlim(*AXIS_LIMITS)
        ax.set_ylim(*AXIS_LIMITS)
        ax.set_aspect("equal")
        ax.set_xticks(range(int(AXIS_LIMITS[0]), int(AXIS_LIMITS[1]) + 1))
        ax.set_yticks(range(int(AXIS_LIMITS[0]), int(AXIS_LIMITS[1]) + 1))
        ax.set_title(label, pad=2.5)
        style.panel_label(ax, letter, dx=-0.10, dy=1.02)
        ax.text(
            0.035,
            0.97,
            f"n = {stats['n']:,}\n"
            f"RMSE = {stats['rmse']:.2f}\n"
            f"bias = {stats['bias']:+.2f}\n"
            f"R$^2$ = {stats['r2']:+.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=style.MAX_TEXT_PT - 1,
            linespacing=1.35,
            zorder=4,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.2),
        )
        dropped = off_scale(observed, predicted)
        if dropped:
            ax.text(
                0.97,
                0.03,
                f"{dropped:,} off scale",
                transform=ax.transAxes,
                va="bottom",
                ha="right",
                fontsize=style.MIN_TEXT_PT,
                color=style.MUTED_INK,
                zorder=4,
            )

    axes[0].set_ylabel(r"Predicted $\log_{10}$ suction (cm)")
    fig.supxlabel(r"Observed $\log_{10}$ suction (cm)", fontsize=style.MAX_TEXT_PT)

    return style.save(fig, Path(output_dir) / "fig04_validation_scatter")


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
