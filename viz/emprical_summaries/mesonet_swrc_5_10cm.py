import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from swapstress.swrc import theta_from_psi


TARGET_DEPTHS = (5, 10)


def _param_value(params, key):
    v = params[key]
    return float(v["value"] if isinstance(v, dict) else v)


def plot_station_swrc_5_10cm(
    parquet_path, fit_json_path, save_path=None, show=False, colormap="plasma"
):
    df = pd.read_parquet(parquet_path)
    df = df[df["Depth [cm]"].isin(TARGET_DEPTHS)].copy()
    if df.empty:
        print(f"No 5/10 cm observations in {parquet_path}")
        return
    df["suction"] = np.abs(df["KPA"].astype(float).values * 10.19716)
    df["theta"] = df["VWC"].astype(float).values

    with open(fit_json_path, "r") as f:
        fits = json.load(f)

    obs_depths = sorted(int(x) for x in df["Depth [cm]"].unique())
    all_depths = [
        d
        for d in obs_depths
        if (fits.get(str(d)) or fits.get(d) or fits.get(float(d)))
        and (fits.get(str(d)) or fits.get(d) or fits.get(float(d))).get("status")
        == "Success"
    ]
    if not all_depths:
        print(f"No successful fits at target depths for {parquet_path}")
        return

    plt.style.use("seaborn-v0_8-darkgrid")
    facecolor = plt.rcParams["figure.facecolor"]
    axescolor = plt.rcParams["axes.facecolor"]

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(axescolor)

    cmap = plt.cm.get_cmap(colormap)
    colors = cmap(np.linspace(0, 0.85, len(all_depths)))

    for color, depth in zip(colors, all_depths):
        g = df[df["Depth [cm]"] == depth]
        entry = fits.get(str(depth)) or fits.get(depth) or fits.get(float(depth))
        params = entry["parameters"]
        tr = _param_value(params, "theta_r")
        ts = _param_value(params, "theta_s")
        al = _param_value(params, "alpha")
        nn = _param_value(params, "n")

        psi_min = max(1e-2, float(np.nanmin(g["suction"].values)))
        psi_max = float(np.nanmax(g["suction"].values))
        if not np.isfinite(psi_min) or not np.isfinite(psi_max) or psi_min >= psi_max:
            continue
        psi_smooth = np.logspace(np.log10(psi_min), np.log10(psi_max), 200)
        theta_fit = theta_from_psi(psi_smooth, tr, ts, al, nn)
        ax.plot(
            theta_fit, psi_smooth, "-", color=color, lw=2, label=f"Depth {depth} cm"
        )

    ax.set_yscale("log")
    ax.set_ylabel("Soil Water Potential (cm) - Log Scale", fontsize=12)
    ax.set_xlabel("Volumetric Water Content ($cm^3/cm^3$)", fontsize=12)

    station_code = os.path.splitext(os.path.basename(parquet_path))[0]
    station_name = None
    if "name" in df.columns and df["name"].notna().any():
        station_name = str(df["name"].dropna().iloc[0])
    subtitle = station_name if station_name else station_code
    ax.set_title(
        f"Soil Water Retention Curves\n({subtitle})", fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.grid(True, which="both", ls="--", c="0.7")
    ax.set_xlim(right=0.65)
    ax.set_ylim(top=10**7)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_mesonet_swrc_5_10cm_dir(preprocessed_dir, fits_dir, out_dir, show=False):
    os.makedirs(out_dir, exist_ok=True)
    parquet_files = sorted(
        f for f in os.listdir(preprocessed_dir) if f.endswith(".parquet")
    )
    for fn in parquet_files:
        station = os.path.splitext(fn)[0]
        fit_path = os.path.join(fits_dir, f"{station}_fit_results.json")
        if not os.path.exists(fit_path):
            print(f"Skipping {station}: no fit results at {fit_path}")
            continue
        parquet_path = os.path.join(preprocessed_dir, fn)
        save_path = os.path.join(out_dir, f"{station}.png")
        plot_station_swrc_5_10cm(parquet_path, fit_path, save_path=save_path, show=show)


if __name__ == "__main__":
    root_ = os.path.join("/nas", "soils", "soil_potential_obs", "mt_mesonet")
    preprocessed_dir_ = os.path.join(root_, "preprocessed_by_station")
    fits_dir_ = os.path.join(root_, "results_by_station")
    out_dir_ = os.path.join(root_, "station_swrc_plots_5_10cm")
    plot_mesonet_swrc_5_10cm_dir(preprocessed_dir_, fits_dir_, out_dir_, show=False)
# ========================= EOF ====================================================================
