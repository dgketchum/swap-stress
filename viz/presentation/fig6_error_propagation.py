"""Figure 6: Error propagation — model error vs SMAP input error.

Two curves vs theta:
  1. Model intrinsic RMSE (conditional bias by theta decile)
  2. Propagated SMAP input uncertainty: median|J(theta)| * sigma_SMAP

Where the SMAP curve exceeds the model curve, better retrievals would
improve the product. Where the model curve dominates, the model is the
bottleneck.

Usage:
    uv run python viz/presentation/fig6_error_propagation.py
"""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ERROR_DIR = Path(
    "/nas/soils/swapstress/releases/global_pruned_refresh_20260520/evaluation"
)
SIGMA_SMAP = 0.04  # SMAP L3 published ubRMSE (m3/m3)
SIGMA_SMAP_OBS = 0.089  # observed RMSE vs ISMN 5 cm in-situ (811 stations, n=595K)
OUT_DIR = Path("figs/presentation")


def main():
    # --- Load conditional bias (model intrinsic error) ---
    cb = pd.read_csv(ERROR_DIR / "conditional_bias_by_decile.csv")
    cb = cb[cb["group"] == "all"].sort_values("theta_mid").reset_index(drop=True)

    # --- Load sensitivity (Jacobian) and bin by the same theta edges ---
    ss = pd.read_csv(ERROR_DIR / "sensitivity_stats.csv")

    # Bin Jacobian using the same theta bin edges as conditional bias
    edges = list(cb["theta_lo"]) + [cb["theta_hi"].iloc[-1]]
    ss["bin"] = pd.cut(ss["theta"], bins=edges, labels=False)
    ss = ss.dropna(subset=["bin"])

    jac_by_bin = (
        ss.groupby("bin")["abs_jacobian"]
        .agg(["median", "mean", "std", "count"])
        .reset_index()
    )

    # Propagated SMAP error per bin
    jac_by_bin["smap_propagated"] = jac_by_bin["median"] * SIGMA_SMAP
    jac_by_bin["smap_propagated_obs"] = jac_by_bin["median"] * SIGMA_SMAP_OBS

    # Align theta midpoints — drop the last bin (0.47–1.0) which extends
    # far beyond the SMAP L3 domain and distorts the x-axis
    n = min(len(jac_by_bin), len(cb)) - 1  # drop last bin
    theta_mid = cb["theta_mid"].values[:n]
    model_rmse = cb["rmse"].values[:n]
    smap_prop = jac_by_bin["smap_propagated"].values[:n]
    smap_prop_obs = jac_by_bin["smap_propagated_obs"].values[:n]

    # --- Find crossover for observed-error curve ---
    diff_hi = smap_prop_obs - model_rmse
    crossover_theta = None
    for i in range(len(diff_hi) - 1):
        if diff_hi[i] * diff_hi[i + 1] < 0:
            frac = diff_hi[i] / (diff_hi[i] - diff_hi[i + 1])
            crossover_theta = theta_mid[i] + frac * (theta_mid[i + 1] - theta_mid[i])
            crossover_val = model_rmse[i] + frac * (model_rmse[i + 1] - model_rmse[i])
            break

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)

    ax.plot(
        theta_mid,
        model_rmse,
        "o-",
        color="#2C3E50",
        lw=2,
        ms=5,
        zorder=4,
        label="Model intrinsic RMSE",
    )
    ax.plot(
        theta_mid,
        smap_prop,
        "s-",
        color="#E74C3C",
        lw=2,
        ms=5,
        zorder=4,
        label=r"$\sigma_{\theta}$ = 0.04 m$^3$m$^{-3}$ (SMAP L3)",
    )
    ax.plot(
        theta_mid,
        smap_prop_obs,
        "^--",
        color="#E74C3C",
        lw=1.5,
        ms=5,
        alpha=0.6,
        zorder=3,
        label=r"$\sigma_{\theta}$ = 0.089 m$^3$m$^{-3}$ (ISMN, 811 sites)",
    )

    # Shade between model and observed-error curve
    ax.fill_between(
        theta_mid,
        model_rmse,
        smap_prop_obs,
        where=smap_prop_obs >= model_rmse,
        alpha=0.10,
        color="#E74C3C",
        label="SMAP-limited",
    )
    ax.fill_between(
        theta_mid,
        model_rmse,
        smap_prop_obs,
        where=smap_prop_obs < model_rmse,
        alpha=0.08,
        color="#2C3E50",
        label="Model-limited",
    )

    # Crossover annotation
    if crossover_theta is not None:
        ax.plot(
            crossover_theta,
            crossover_val,
            "k*",
            ms=12,
            zorder=5,
        )
        ax.annotate(
            f"crossover\n$\\theta$ = {crossover_theta:.2f}",
            xy=(crossover_theta, crossover_val),
            xytext=(crossover_theta + 0.06, crossover_val + 0.15),
            fontsize=9,
            ha="left",
            arrowprops=dict(arrowstyle="->", color="k", lw=0.8),
        )

    ax.set_xlabel(r"$\theta$ (m$^3$ m$^{-3}$)", fontsize=11)
    ax.set_ylabel(
        r"Error in log$_{10}$ suction (cm H$_2$O)",
        fontsize=11,
    )
    ax.set_xlim(0, 0.50)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(
            OUT_DIR / f"fig6_error_propagation.{ext}", dpi=200, bbox_inches="tight"
        )
    plt.close(fig)
    print(f"Saved to {OUT_DIR / 'fig6_error_propagation.png'}")


if __name__ == "__main__":
    main()
