"""Descriptor Fig 5: PTF applicability and common-subset error.

This figure is deliberately not a three-panel prediction contest. It answers
two narrower questions that are useful to a data-product reader:

1. Over the 5,244 held-out measured theta-potential pairs assembled for the
   mapped-parameter comparison, where can each route return an estimate under
   its own mathematical rules?
2. On the identical rows where both mapped van Genuchten curves are strictly
   invertible, what are the conditional RMSE and MAE of SWAP, Rosetta, and
   depth-matched POLARIS?

Panel a separates a method's applicability from its accuracy. For SWAP,
"evaluable" means that the direct QRF returns a finite estimate. For Rosetta
and POLARIS, it means that mapped parameters are available and the measured
water content satisfies ``theta_r < theta < theta_s``. Water contents outside
that interval are labeled by the curve boundary they exceed; they are not
called invalid observations and epsilon-clipped inversions are never plotted.

Panel b uses the common three-way strict subset for every method. RMSE and MAE
are shown together because they support different conclusions in this sample:
POLARIS has lower observation-weighted squared error, whereas SWAP has lower
absolute error. The panel therefore documents conditional behavior without
claiming universal method superiority. Conventional equal-site-weighted
metrics will be added only after their dedicated reproducible artifact is
persisted.

The input is ``ptf_depth_matched_observations.parquet`` from
``swapstress.validation.ptf_depth_matched``. It contains held-out measured
theta-potential pairs, the released SWAP predictions, mapped Rosetta central
parameters, depth-matched POLARIS mean parameters, and strict domain statuses.
This is an accuracy and applicability audit at the comparison observations;
it is not a SMAP-conditioned accuracy test.

Drawn to ``swapstress.figures.style`` at 183 mm double-column width. Text and
line art remain vector.

Usage:
    uv run swapstress-figures --figure validation-scatter
    uv run python -m swapstress.figures.fig05_ptf_comparison --observations <path>
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from swapstress.figures import style
from swapstress.units import log10_suction_cm_to_log10_abs_mpa
from swapstress.validation.ptf_depth_matched import DEFAULT_OBSERVATIONS

DEFAULT_OUTPUT_DIR = "figs/descriptor"

# Prediction column, strict-status column (None for SWAP), display label, and
# method colour. POLARIS uses the capped column only because it is the persisted
# numeric column; the common strict mask guarantees that no capped boundary
# value enters panel b.
ESTIMATORS = (
    ("rf_pred", None, "SWAP direct", style.CATEGORICAL[0]),
    (
        "ros_log10_suction",
        "rosetta_domain_status",
        "Rosetta",
        style.CATEGORICAL[1],
    ),
    (
        "pol_depth_matched_log10_suction_capped",
        "polaris_domain_status",
        "POLARIS",
        style.CATEGORICAL[2],
    ),
)

DOMAIN_STATUSES = (
    "missing_parameters",
    "invalid_parameters",
    "below_or_equal_theta_r",
    "above_or_equal_theta_s",
    "in_domain",
)

# Panel-a categories are exhaustive and mutually exclusive over the candidate
# set. "Unavailable" includes missing/invalid parameters and any unexpected
# non-finite prediction on a row otherwise marked in-domain.
APPLICABILITY = (
    ("evaluable", "Evaluated / strict inverse", "#1B9E77"),
    ("below", r"θ ≤ θ$_r$", "#D95F02"),
    ("above", r"θ ≥ θ$_s$", "#7570B3"),
    ("unavailable", "Unavailable", style.NO_DATA_GRAY),
)

FIGURE_WIDTH_MM = style.DOUBLE_COLUMN_MM
FIGURE_HEIGHT_MM = 76.0


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    """Return conventional observation-weighted error summaries."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    keep = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[keep]
    predicted = predicted[keep]
    if observed.size == 0:
        raise ValueError("Cannot calculate metrics without finite pairs.")

    residual = predicted - observed
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    return {
        "n": int(observed.size),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
    }


def load_depth_matched(observations_path: str) -> pd.DataFrame:
    """Load and validate the held-out, depth-matched comparison table."""
    df = pd.read_parquet(observations_path)
    required = (
        {"log10_suction_cm", "sample_id", "source"}
        | {column for column, _, _, _ in ESTIMATORS}
        | {status for _, status, _, _ in ESTIMATORS if status is not None}
    )
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{observations_path}: missing required column(s) {sorted(missing)}; "
            "was this built by `swapstress.validation.ptf_depth_matched build`?"
        )

    if not np.isfinite(df["log10_suction_cm"].to_numpy(dtype=float)).all():
        raise ValueError(f"{observations_path}: candidate observations must be finite.")

    allowed = set(DOMAIN_STATUSES)
    for _, status_column, label, _ in ESTIMATORS:
        if status_column is None:
            continue
        observed_statuses = set(df[status_column].dropna().astype(str).unique())
        unknown = observed_statuses - allowed
        if unknown:
            raise ValueError(
                f"{observations_path}: {label} has unknown domain status(es) "
                f"{sorted(unknown)}."
            )

    print(
        f"{len(df):,} depth-matched held-out rows, "
        f"{df['sample_id'].nunique():,} sample-layer identifiers"
    )
    return df


def applicability_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Count mutually exclusive applicability outcomes for each method."""
    rows = []
    n_total = len(df)

    for column, status_column, label, _ in ESTIMATORS:
        finite_prediction = np.isfinite(df[column].to_numpy(dtype=float))
        if status_column is None:
            row = {
                "method": label,
                "evaluable": int(finite_prediction.sum()),
                "below": 0,
                "above": 0,
                "unavailable": int((~finite_prediction).sum()),
            }
        else:
            status = df[status_column].astype(str).to_numpy()
            in_domain = status == "in_domain"
            below = status == "below_or_equal_theta_r"
            above = status == "above_or_equal_theta_s"
            assigned = (in_domain & finite_prediction) | below | above
            row = {
                "method": label,
                "evaluable": int((in_domain & finite_prediction).sum()),
                "below": int(below.sum()),
                "above": int(above.sum()),
                "unavailable": int((~assigned).sum()),
            }

        if sum(row[key] for key, _, _ in APPLICABILITY) != n_total:
            raise ValueError(f"Applicability outcomes do not sum to n={n_total:,}.")
        rows.append(row)

    return pd.DataFrame(rows).set_index("method")


def common_strict_mask(df: pd.DataFrame) -> np.ndarray:
    """Rows with finite observations/predictions and both PTFs in-domain."""
    keep = np.isfinite(df["log10_suction_cm"].to_numpy(dtype=float))
    for column, status_column, _, _ in ESTIMATORS:
        keep &= np.isfinite(df[column].to_numpy(dtype=float))
        if status_column is not None:
            keep &= df[status_column].astype(str).to_numpy() == "in_domain"
    return keep


def common_subset_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Observation-weighted metrics on one identical strict subset."""
    keep = common_strict_mask(df)
    observed = log10_suction_cm_to_log10_abs_mpa(
        df.loc[keep, "log10_suction_cm"].to_numpy(dtype=float)
    )
    rows = []
    for column, _, label, color in ESTIMATORS:
        predicted = log10_suction_cm_to_log10_abs_mpa(
            df.loc[keep, column].to_numpy(dtype=float)
        )
        rows.append({"method": label, "color": color, **metrics(observed, predicted)})
    return pd.DataFrame(rows).set_index("method")


def _plot_applicability(ax, counts: pd.DataFrame) -> None:
    """Panel a: stacked shares of the common candidate set."""
    methods = [label for _, _, label, _ in ESTIMATORS]
    y = np.arange(len(methods))[::-1]
    totals = counts.loc[methods].sum(axis=1).to_numpy(dtype=float)
    left = np.zeros(len(methods), dtype=float)

    for key, legend_label, color in APPLICABILITY:
        values = 100.0 * counts.loc[methods, key].to_numpy(dtype=float) / totals
        ax.barh(
            y,
            values,
            left=left,
            height=0.58,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=legend_label,
        )
        for row_y, start, value in zip(y, left, values):
            if value < 8.0:
                continue
            text_color = style.AXIS_COLOR if key == "unavailable" else "white"
            precision = 0 if np.isclose(value, 100.0) else 1
            ax.text(
                start + value / 2.0,
                row_y,
                f"{value:.{precision}f}%",
                ha="center",
                va="center",
                fontsize=style.MIN_TEXT_PT,
                color=text_color,
                fontweight="bold",
            )
        left += values

    ax.set_yticks(y, methods)
    ax.set_xlim(0.0, 100.0)
    ax.set_xticks((0, 25, 50, 75, 100))
    ax.set_xlabel("Share of candidate pairs (%)")
    ax.set_title("Applicability over held-out measured pairs", pad=18)
    ax.text(
        0.0,
        1.02,
        f"n = {int(totals[0]):,}; SWAP finite estimate, PTF strict inverse",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=style.MIN_TEXT_PT,
        color=style.MUTED_INK,
    )
    ax.grid(axis="x", linewidth=0.35, color=style.GRID_COLOR, zorder=0)
    ax.set_axisbelow(True)
    handles = [
        Patch(facecolor=color, edgecolor="none", label=label)
        for _, label, color in APPLICABILITY
    ]
    # Matplotlib fills a two-column legend column-first. Reorder so the visual
    # reading order is evaluable, below theta_r, above theta_s, unavailable.
    handles = [handles[0], handles[2], handles[1], handles[3]]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.26),
        ncol=2,
        columnspacing=1.2,
        handlelength=1.2,
    )
    style.panel_label(ax, "a", dx=-0.10, dy=1.13)


def _plot_common_error(ax, common: pd.DataFrame) -> None:
    """Panel b: RMSE and MAE on the identical common strict subset."""
    methods = [label for _, _, label, _ in ESTIMATORS]
    y = np.arange(len(methods))[::-1]

    for row_y, method in zip(y, methods):
        row = common.loc[method]
        mae = float(row["mae"])
        rmse = float(row["rmse"])
        color = str(row["color"])
        ax.plot([mae, rmse], [row_y, row_y], color=color, alpha=0.5, linewidth=1.2)
        ax.scatter(
            mae,
            row_y,
            marker="D",
            s=26,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        ax.scatter(
            rmse,
            row_y,
            marker="o",
            s=30,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        ax.annotate(
            f"{mae:.3f}",
            (mae, row_y),
            xytext=(-4, 7),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=style.MIN_TEXT_PT,
            color=color,
        )
        ax.annotate(
            f"{rmse:.3f}",
            (rmse, row_y),
            xytext=(4, 7),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=style.MIN_TEXT_PT,
            color=color,
        )

    n_common = int(common["n"].iloc[0])
    ax.set_yticks(y, methods)
    ax.set_ylim(-0.55, 2.55)
    ax.set_xlim(0.0, 0.86)
    ax.set_xticks(np.arange(0.0, 0.81, 0.2))
    ax.set_xlabel(r"Error (log$_{10}$ |MPa|; lower is better)")
    ax.set_title("Conditional error on the common strict subset", pad=18)
    ax.text(
        0.0,
        1.02,
        f"n = {n_common:,}; identical rows, observation weighted",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=style.MIN_TEXT_PT,
        color=style.MUTED_INK,
    )
    ax.grid(axis="x", linewidth=0.35, color=style.GRID_COLOR, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(
        handles=(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor=style.MUTED_INK,
                markeredgecolor="white",
                label="RMSE",
            ),
            Line2D(
                [],
                [],
                marker="D",
                linestyle="none",
                markerfacecolor=style.MUTED_INK,
                markeredgecolor="white",
                label="MAE",
            ),
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.26),
        ncol=2,
        columnspacing=1.5,
    )
    style.panel_label(ax, "b", dx=-0.10, dy=1.13)


def render(df: pd.DataFrame, output_dir: str) -> Path:
    """Render applicability and common-subset error to the standard outputs."""
    style.apply()
    counts = applicability_counts(df)
    common = common_subset_metrics(df)
    if common["n"].nunique() != 1:
        raise ValueError("Every common-subset metric must use the same denominator.")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=style.figsize(FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM),
        gridspec_kw={"width_ratios": (1.08, 0.92)},
        layout="constrained",
    )
    _plot_applicability(axes[0], counts)
    _plot_common_error(axes[1], common)
    return style.save(fig, Path(output_dir) / "fig05_ptf_comparison")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fig05_ptf_comparison",
        description="Descriptor Fig 5: PTF applicability and common-subset error.",
    )
    parser.add_argument("--observations", default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    df = load_depth_matched(args.observations)
    counts = applicability_counts(df)
    common = common_subset_metrics(df)
    print("Applicability counts (candidate denominator):")
    print(counts.to_string())
    print("Common strict-subset metrics (observation weighted):")
    print(common[["n", "rmse", "mae", "bias", "r2"]].to_string())
    path = render(df, args.output_dir)
    print(f"Saved to {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
