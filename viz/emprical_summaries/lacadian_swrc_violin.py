import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_lacadian_swrc_violin(
    swrc_csv_path, ts_parquet_path, save_path=None, show=False
):
    df = pd.read_csv(swrc_csv_path)
    df["suction_cm"] = np.abs(df["suction_cm"].astype(float))
    df["theta"] = df["theta"].astype(float)
    df = df.dropna(subset=["suction_cm", "theta", "depth_cm"])

    swrc_depths = sorted(df["depth_cm"].unique())
    n_depths = len(swrc_depths)
    colors = plt.cm.plasma(np.linspace(0, 0.85, n_depths))

    ts = pd.read_parquet(ts_parquet_path)
    ts_depths = sorted(ts["depth_cm"].unique())

    # Extend color palette to cover all timeseries depths not in SWRC
    all_depths = sorted(set(swrc_depths) | set(ts_depths))
    all_colors = plt.cm.plasma(np.linspace(0, 0.85, len(all_depths)))
    full_color = {int(d): c for d, c in zip(all_depths, all_colors)}

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax, axv) = plt.subplots(
        2,
        1,
        figsize=(8, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05},
    )
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    axv.set_facecolor("white")

    # --- SWRC scatter (top panel) ---
    for depth, color in zip(swrc_depths, colors):
        g = df[df["depth_cm"] == depth]
        ax.plot(
            g["theta"],
            g["suction_cm"],
            "o",
            color=color,
            ms=4,
            alpha=0.8,
            label=f"{int(depth)} cm",
        )

    ax.set_yscale("log")
    ax.set_ylabel("Soil Water Potential (cm)", fontsize=12)
    station = os.path.splitext(os.path.basename(swrc_csv_path))[0]
    ax.set_title(
        f"SWRC Observations with VWC Frequency — {station}",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(title="Depth", fontsize=9, title_fontsize=9, frameon=False)
    ax.set_xlim(0.0, 0.65)
    ax.grid(True, which="both", ls="--", c="0.75")

    # --- Violin (bottom panel) — all available VWC depths ---
    violin_data, positions, vcolors, labels = [], [], [], []
    for idx, depth in enumerate(ts_depths, start=1):
        vals = ts.loc[ts["depth_cm"] == depth, "VWC"].values
        vals = vals[np.isfinite(vals)]
        vals = vals[(vals >= 0) & (vals <= 1)]
        if vals.size == 0:
            continue
        violin_data.append(vals)
        positions.append(idx)
        vcolors.append(full_color.get(int(depth), "0.6"))
        labels.append(f"{int(depth)} cm (n={vals.size})")

    if violin_data:
        parts = axv.violinplot(
            violin_data,
            vert=False,
            positions=positions,
            showextrema=False,
            widths=0.85,
        )
        for pc, col in zip(parts["bodies"], vcolors):
            pc.set_facecolor(col)
            pc.set_edgecolor("0.4")
            pc.set_alpha(0.6)
        axv.set_yticks(positions)
        axv.set_yticklabels(labels, fontsize=9)
        axv.set_ylim(0, max(positions) + 1)
        axv.invert_yaxis()
        axv.spines["top"].set_visible(False)
        axv.spines["right"].set_visible(False)
        axv.grid(False)
        axv.set_xlim(ax.get_xlim())
    else:
        axv.axis("off")

    axv.set_xlabel("Volumetric Water Content ($cm^3/cm^3$)", fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(left=0.15)

    if save_path:
        plt.savefig(save_path, dpi=350, bbox_inches="tight")
        print(f"Saved {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_lacadian_dir(swrc_dir, ts_dir, out_dir, show=False):
    os.makedirs(out_dir, exist_ok=True)
    swrc_files = [f for f in os.listdir(swrc_dir) if f.endswith(".csv")]
    ts_set = set(os.listdir(ts_dir))
    for fn in swrc_files:
        station = os.path.splitext(fn)[0]
        ts_fn = f"{station}.parquet"
        if ts_fn not in ts_set:
            print(f"No timeseries for {station}, skipping")
            continue
        plot_lacadian_swrc_violin(
            swrc_csv_path=os.path.join(swrc_dir, fn),
            ts_parquet_path=os.path.join(ts_dir, ts_fn),
            save_path=os.path.join(out_dir, f"{station}_swrc_vwc_violin.png"),
            show=show,
        )


if __name__ == "__main__":
    swrc_dir_ = "/nas/soils/soil_potential_obs/preprocessed/lacadian"
    ts_dir_ = "/nas/soils/soil_potential_obs/lacadian/timeseries"
    out_dir_ = "/nas/soils/soil_potential_obs/lacadian/swrc_vwc_violin"
    plot_lacadian_dir(swrc_dir_, ts_dir_, out_dir_)
