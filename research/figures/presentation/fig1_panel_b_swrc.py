"""Figure 1 Panel B: Soil water retention curves from fitted van Genuchten parameters.

Three ReESH sites spanning the texture triangle, using Bayesian VG fits from
/nas/soils/soil_potential_obs/curve_fits/reesh/bayes/:
  - US-HB2 TR2  depth=0  (sand: 97% sand, 1% silt, 2% clay)
  - US-GLE 1    depth=10 (silt loam: 11% sand, 80% silt, 9% clay)
  - US-UTM TDR3 depth=0  (clay: 1% sand, 39% silt, 60% clay)

Y-axis: suction in cm H₂O (log scale), designed to pixel-align with Panel A's
suction axis. X-axis: volumetric water content (m³/m³), shared with Panel C.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from swapstress.swrc import psi_from_theta

FITS_DIR = Path("/nas/soils/soil_potential_obs/curve_fits/reesh/bayes")

# (json_file, depth_key, label, color, site_note)
SAMPLES = [
    ("US-HB2_TR2.json", "0", "Sand (97%)", "#D4A017", "US-HB2, NH"),
    ("US-GLE_1.json", "10", "Silt loam (80%)", "#2E86C1", "US-GLE, CO"),
    ("US-UTM_TDR3.json", "0", "Clay (60%)", "#A93226", "US-UTM, UT"),
]

# Suction axis limits matching Panel A: 10^0 to 10^6 cm
SUCTION_LO, SUCTION_HI = 1.0, 1e6

# Threshold lines (cm H₂O)
FC_LO, FC_HI = 100.0, 330.0  # field capacity band (-10 to -33 kPa)
PWP_LO, PWP_HI = 10200.0, 20400.0  # wilting point band (-1000 to -2000 kPa)


def _vg_suction(theta, theta_r, theta_s, alpha, n):
    """Inverse van Genuchten for plotting: theta -> suction (cm).

    Uses a looser Se clip and a 1e-3 cm floor than the PTF baseline, matching
    what these figures were drawn with.
    """
    psi = psi_from_theta(theta, theta_r, theta_s, alpha, n, se_eps=1e-9)
    return np.maximum(psi, 1e-3)


def load_vg_params(json_file, depth_key):
    """Load fitted VG parameters from a Bayesian fit JSON."""
    with open(FITS_DIR / json_file) as f:
        data = json.load(f)
    entry = data[depth_key]
    p = entry["parameters"]
    return {
        "theta_r": p["theta_r"]["value"],
        "theta_s": p["theta_s"]["value"],
        "alpha": p["alpha"]["value"],
        "n": p["n"]["value"],
    }


def main(output_dir):
    fig, ax = plt.subplots(figsize=(7.0, 6.0))

    # Per-site label positions: (theta_x, suction_y)
    site_positions = {
        "US-HB2, NH": (0.32, 10**1.0),
        "US-GLE, CO": (0.68, 10**1.8),
        "US-UTM, UT": (0.60, 10**1.0),
    }

    for json_file, depth_key, label, color, site_note in SAMPLES:
        params = load_vg_params(json_file, depth_key)

        theta_grid = np.linspace(
            params["theta_r"] + 0.001, params["theta_s"] - 0.001, 300
        )
        suction = _vg_suction(theta_grid, **params)

        ax.plot(theta_grid, suction, color=color, linewidth=1.8, label=label, zorder=3)

        tx, ty = site_positions[site_note]
        ax.text(
            tx,
            ty,
            site_note,
            fontsize=7.5,
            color=color,
            fontstyle="italic",
            va="center",
            ha="left",
        )

    # SMAP L3 empirical range (1st and 99th percentile from daily CONUS rasters)
    smap_p01 = 0.044
    smap_p99 = 0.582
    ax.axvline(smap_p01, color="#A93226", linewidth=1.0, linestyle="--", zorder=2)
    ax.text(
        smap_p01 + 0.012,
        10**1.2,
        f"SMAP L3 P01\n$\\theta$={smap_p01:.2f}",
        fontsize=7,
        color="#A93226",
        ha="left",
        va="center",
        linespacing=1.2,
    )
    ax.axvline(smap_p99, color="#2E86C1", linewidth=1.0, linestyle="--", zorder=2)
    ax.text(
        smap_p99 - 0.012,
        SUCTION_HI * 0.30,
        f"SMAP L3 P99\n$\\theta$={smap_p99:.2f}",
        fontsize=7,
        color="#2E86C1",
        ha="right",
        va="top",
        linespacing=1.2,
    )

    # Wilting point band
    ax.axhspan(PWP_LO, PWP_HI, color="#791F1F", alpha=0.10, zorder=1)
    ax.axhline(PWP_LO, color="#791F1F", linewidth=0.5, linestyle=":", zorder=2)
    ax.axhline(PWP_HI, color="#791F1F", linewidth=0.5, linestyle=":", zorder=2)
    ax.text(
        0.60,
        (PWP_LO * PWP_HI) ** 0.5,
        "Wilting point",
        fontsize=7.5,
        color="#791F1F",
        va="center",
    )

    ax.axhspan(FC_LO, FC_HI, color="#1D9E75", alpha=0.10, zorder=1)
    ax.axhline(FC_LO, color="#1D9E75", linewidth=0.5, linestyle=":", zorder=2)
    ax.axhline(FC_HI, color="#1D9E75", linewidth=0.5, linestyle=":", zorder=2)
    ax.text(
        0.60,
        (FC_LO * FC_HI) ** 0.5,
        "Field capacity",
        fontsize=7.5,
        color="#0F6E56",
        va="center",
    )

    # Axis config
    ax.set_yscale("log")
    ax.set_ylim(SUCTION_LO, SUCTION_HI)
    ax.set_xlim(0.0, 0.75)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelbottom=False)
    ax.set_ylabel(r"Suction (cm H$_2$O)", fontsize=10)
    ax.tick_params(labelsize=9)

    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig1_panel_b_swrc.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out / 'fig1_panel_b_swrc.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figure 1 Panel B: SWRCs")
    parser.add_argument(
        "--output-dir", default="figs/presentation", help="Output directory for figures"
    )
    args = parser.parse_args()
    main(args.output_dir)
