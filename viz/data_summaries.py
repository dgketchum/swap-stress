"""Stacked histogram of data_ct (observation count per profile) by source."""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SOURCE_LABELS = {
    "gshp": "GSHP",
    "ncss": "NCSS",
    "mt_mesonet": "MT Mesonet",
    "reesh": "ReESH",
}

SOURCE_ORDER = ["gshp", "ncss", "mt_mesonet", "reesh"]

# Unit bins 2–10, then progressively wider bins
BIN_EDGES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 20, 30, 50, 100]


def _bin_labels(edges):
    """Build human-readable labels for each bin."""
    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if hi - lo == 1:
            labels.append(str(lo))
        else:
            labels.append(f"{lo}\u2013{hi - 1}")
    labels.append(f"{edges[-1]}+")
    return labels


def plot_data_count_histogram(training_path, output_path):
    """Create a stacked bar chart of data_ct distribution by source."""
    df = pd.read_parquet(training_path)
    df = df[df["data_ct"].notna() & (df["data_ct"] > 0) & (df["data_ct"] != -9999)]

    colors = sns.color_palette("Set2", n_colors=len(SOURCE_ORDER))

    edges_with_inf = BIN_EDGES + [np.inf]
    labels = _bin_labels(BIN_EDGES)

    counts_by_source = {}
    for src in SOURCE_ORDER:
        vals = df.loc[df["source"] == src, "data_ct"].values
        counts, _ = np.histogram(vals, bins=edges_with_inf)
        counts_by_source[src] = counts

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))

    for src, color in zip(SOURCE_ORDER, colors):
        counts = counts_by_source[src]
        ax.bar(
            x,
            counts,
            bottom=bottom,
            color=color,
            label=SOURCE_LABELS[src],
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += counts

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Observations per profile")
    ax.set_ylabel("Number of profiles")
    ax.set_title("Distribution of observation counts by source")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    default_training = os.path.expanduser(
        "~/data/IrrigationGIS/soils/swapstress/training/unified_training_emb_250m.parquet"
    )
    default_output = os.path.expanduser(
        "~/data/IrrigationGIS/soils/swapstress/training/data_ct_histogram.png"
    )

    import argparse

    parser = argparse.ArgumentParser(description="Plot data_ct histogram by source")
    parser.add_argument(
        "--training", default=default_training, help="Path to training parquet"
    )
    parser.add_argument("--output", default=default_output, help="Output PNG path")
    args = parser.parse_args()

    plot_data_count_histogram(args.training, args.output)
