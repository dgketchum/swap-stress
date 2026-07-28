"""Figure 2b: Stacked horizontal histogram of observation count per profile, by source.

A profile is a unique soil pedon or station — the sample_id with its trailing
depth component stripped (e.g., ``gshp_10_15.5`` → ``gshp_10``).  Each bar
segment shows how many profiles from a given source fall into each observation-
count bin.

Usage:
    uv run python -m research.figures.presentation.fig2b_obs_count_bar
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRAINING_TABLE = "/nas/soils/swapstress/training/obs_level_training_9km_global.parquet"

SOURCE_ORDER = ["gshp", "ncss", "lacadian", "mt_mesonet", "reesh"]
SOURCE_LABELS = {
    "gshp": "GSHP",
    "ncss": "NCSS",
    "lacadian": "LaCADIAN",
    "mt_mesonet": "MT Mesonet",
    "reesh": "ReESH",
}

# Match Figure 2 map symbol colors
SOURCE_COLORS = {
    "gshp": "#2ca02c",
    "ncss": "#1f77b4",
    "lacadian": "#9467bd",
    "mt_mesonet": "#ff7f0e",
    "reesh": "#d62728",
}

# Unit bins 2-10, then progressively wider
BIN_EDGES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 20, 30, 50, 100]


def _extract_profile(sample_id: str) -> str:
    """Strip trailing depth from sample_id to get the profile identifier."""
    return re.sub(r"_[\d.]+$", "", str(sample_id))


def _bin_labels(edges):
    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if hi - lo == 1:
            labels.append(str(lo))
        else:
            labels.append(f"{lo}\u2013{hi - 1}")
    labels.append(f"{edges[-1]}+")
    return labels


def main(output_dir):
    df = pd.read_parquet(TRAINING_TABLE, columns=["source", "sample_id"])
    ct = df.groupby(["source", "sample_id"]).size().reset_index(name="data_ct")

    edges_with_inf = BIN_EDGES + [np.inf]
    labels = _bin_labels(BIN_EDGES)

    counts_by_source = {}
    for src in SOURCE_ORDER:
        vals = ct.loc[ct["source"] == src, "data_ct"].values
        hist, _ = np.histogram(vals, bins=edges_with_inf)
        counts_by_source[src] = hist

    fig, ax = plt.subplots(figsize=(4.5, 5.5))

    y = np.arange(len(labels))
    left = np.zeros(len(labels))

    for src in SOURCE_ORDER:
        counts = counts_by_source[src]
        ax.barh(
            y,
            counts,
            left=left,
            color=SOURCE_COLORS[src],
            label=SOURCE_LABELS[src],
            edgecolor="white",
            linewidth=0.5,
            height=0.7,
        )
        left += counts

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_ylabel("Observations per sample", fontsize=9)
    ax.set_xlabel("Samples", fontsize=9)
    ax.tick_params(labelsize=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    ax.legend(
        fontsize=7.5, framealpha=0.9, edgecolor="#ddd", loc="upper right", borderpad=0.5
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig2b_obs_count_bar.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out / 'fig2b_obs_count_bar.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figure 2b: Obs count histogram")
    parser.add_argument("--output-dir", default="figs/presentation")
    args = parser.parse_args()
    main(args.output_dir)
