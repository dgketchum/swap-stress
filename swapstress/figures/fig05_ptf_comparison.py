"""Descriptor Fig 5: validation scatter -- the direct model against PTF baselines.

Observed vs predicted matric potential for three estimators, each over its
available cases: our direct quantile RF (its median, which is the released
Level 1 value), Rosetta, and POLARIS. The PTF columns come from
``swapstress.validation.ptf_baseline``, which pushes each site's published van
Genuchten parameters through the retention equation at the observed theta.
The candidate observations are the same held-out rows for all three panels,
but each method's metrics keep only the pairs where that method returns a
finite value, so the displayed n differs by method -- the caption says
"available cases by method" rather than claiming identical samples.

Off-scale honesty: the axes are held to the shared window (expanding them to
include the PTF tails would collapse the QRF structure), so the fraction of
each method's pairs that lands outside the window is stated in the metric
block, and the off-scale predictions are marked as carets on the top/bottom
axis edge at their observed x.

Both axes are ``log10|psi|`` with psi in MPa, the descriptor's presentation
unit, converted from the stored ``log10_suction_cm`` by the exact additive
shift in ``swapstress.units``. Because the shift is common to observed and
predicted, RMSE, bias and R2 are numerically unchanged by it.

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
    uv run python -m swapstress.figures.fig05_ptf_comparison --model-dir <dir>
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from swapstress.figures import style
from swapstress.units import log10_suction_cm_to_log10_abs_mpa

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

# The drawn window, declared in the pipeline's internal units -- 0 to 7 log10
# cm covers everything the observations reach -- and then shifted once into the
# descriptor's presentation unit. Declaring it this way keeps the panels
# showing exactly the region they always showed: the shift is additive, so the
# window moves with the data rather than cropping a different part of it.
AXIS_LIMITS_LOG10_CM = (0.0, 7.0)
AXIS_LIMITS = tuple(
    float(log10_suction_cm_to_log10_abs_mpa(v)) for v in AXIS_LIMITS_LOG10_CM
)
# One tick per decade of potential, as before -- the shift makes the window
# ends non-integer, so the whole decades inside it are taken explicitly rather
# than by truncating the limits (which rounds the wrong way below zero).
DECADE_TICKS = list(range(math.ceil(AXIS_LIMITS[0]), math.floor(AXIS_LIMITS[1]) + 1))

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


def off_scale(observed: np.ndarray, predicted: np.ndarray):
    """Masks for the pairs the metrics count but the axes cannot show.

    Rosetta puts 17% of its predictions outside the drawn window and POLARIS
    7%, far beyond anything the observations reach. They stay in the RMSE and
    R2 -- they are real errors -- but they land off the panel, so their share
    is quoted in the metric block and each one is marked at the axis edge
    rather than left to be silently cropped. A couple of *observations* also
    sit just past the window's low end, so both axes are checked. Returns
    ``(above, below, left, right)`` boolean masks: predicted past the top or
    bottom edge, observed past the left or right edge.
    """
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    keep = np.isfinite(observed) & np.isfinite(predicted)
    low, high = AXIS_LIMITS
    above = keep & (predicted > high)
    below = keep & (predicted < low)
    left = keep & (observed < low)
    right = keep & (observed > high)
    return above, below, left, right


def render(df: pd.DataFrame, output_dir: str) -> Path:
    style.apply()
    # Everything drawn and every metric quoted is in the presentation unit. The
    # shift is common to both axes, so RMSE, bias and R2 are the same numbers
    # they were in log10 cm -- converting here rather than in the caption means
    # the panel and its annotation can never disagree about which unit they are.
    observed = log10_suction_cm_to_log10_abs_mpa(df["log10_suction_cm"].values)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=style.figsize(FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM),
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    for ax, letter, (column, label, color) in zip(axes, PANEL_LETTERS, ESTIMATORS):
        predicted = log10_suction_cm_to_log10_abs_mpa(df[column].values)
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
        ax.set_xticks(DECADE_TICKS)
        ax.set_yticks(DECADE_TICKS)
        ax.set_title(label, pad=2.5)
        style.panel_label(ax, letter, dx=-0.10, dy=1.02)

        # Off-scale predictions: their share goes in the metric block, and each
        # one is a caret on the axis edge at its observed x, so the clipped
        # mass is visible in proportion to its consequence.
        above, below, left, right = off_scale(observed, predicted)
        dropped = int((above | below | left | right).sum())
        low, high = AXIS_LIMITS
        rug_kw = dict(
            s=5.0,
            color=color,
            alpha=0.2,
            linewidths=0.5,
            rasterized=True,
            clip_on=False,
            zorder=2,
        )
        if above.any():
            ax.scatter(observed[above], np.full(above.sum(), high), marker=10, **rug_kw)
        if below.any():
            ax.scatter(observed[below], np.full(below.sum(), low), marker=11, **rug_kw)
        if left.any():
            clipped = np.clip(predicted[left], low, high)
            ax.scatter(np.full(left.sum(), low), clipped, marker=8, **rug_kw)
        if right.any():
            clipped = np.clip(predicted[right], low, high)
            ax.scatter(np.full(right.sum(), high), clipped, marker=9, **rug_kw)

        lines = [
            f"n = {stats['n']:,}",
            f"RMSE = {stats['rmse']:.2f}",
            f"bias = {stats['bias']:+.2f}",
            f"R$^2$ = {stats['r2']:+.2f}",
        ]
        if dropped:
            pct = 100.0 * dropped / stats["n"]
            share = "<0.1%" if pct < 0.1 else f"{pct:.1f}%"
            lines.append(f"{share} outside axes")
        ax.text(
            0.035,
            0.97,
            "\n".join(lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=style.MAX_TEXT_PT - 1,
            linespacing=1.35,
            zorder=4,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.2),
        )

    axes[0].set_ylabel(f"Predicted {style.LOG10_ABS_MPA_AXIS}")
    fig.supxlabel(f"Observed {style.LOG10_ABS_MPA_AXIS}", fontsize=style.MAX_TEXT_PT)

    return style.save(fig, Path(output_dir) / "fig05_ptf_comparison")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fig05_ptf_comparison",
        description="Descriptor Fig 5: held-out validation scatter against PTF baselines.",
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--ptf-dir", default=DEFAULT_PTF_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    df = load_holdout(args.model_dir, args.ptf_dir)
    observed = log10_suction_cm_to_log10_abs_mpa(df["log10_suction_cm"].values)
    for column, label, _ in ESTIMATORS:
        stats = metrics(observed, log10_suction_cm_to_log10_abs_mpa(df[column].values))
        print(
            f"  {label:16} n={stats['n']:5,d}  RMSE={stats['rmse']:.3f}  "
            f"bias={stats['bias']:+.3f}  R2={stats['r2']:+.3f}"
        )
    path = render(df, args.output_dir)
    print(f"Saved to {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
