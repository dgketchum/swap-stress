"""Figure 1 Panels B+C: SWRCs stacked over SMAP theta KDE.

Panel B: Three van Genuchten SWRCs (sand, silt loam, clay) from ReESH.
Panel C: SMAP L3 daily CONUS theta distribution (KDE).
Shared x-axis (VWC). Three vertical lines (P01, median, P99) span both.
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy.stats import gaussian_kde

from swapstress.swrc import psi_from_theta

FITS_DIR = Path("/nas/soils/soil_potential_obs/curve_fits/reesh/bayes")
SMAP_DIR = "/nas/soils/smap/SPL3SMP_E/daily_tif"

SAMPLES = [
    ("US-HB2_TR2.json", "0", "Sand (97%)", "#D4A017", "US-HB2, NH"),
    ("US-GLE_1.json", "10", "Silt loam (80%)", "#2E86C1", "US-GLE, CO"),
    ("US-UTM_TDR3.json", "0", "Clay (60%)", "#A93226", "US-UTM, UT"),
]

SUCTION_LO, SUCTION_HI = 1.0, 1e6
FC_LO, FC_HI = 100.0, 330.0
PWP_LO, PWP_HI = 10200.0, 20400.0

SMAP_P01 = 0.044
SMAP_MEDIAN = 0.197
SMAP_P99 = 0.582
XLIM = (0.0, 0.75)


def _vg_suction(theta, theta_r, theta_s, alpha, n):
    """Inverse van Genuchten for plotting: theta -> suction (cm).

    Uses a looser Se clip and a 1e-3 cm floor than the PTF baseline, matching
    what these figures were drawn with.
    """
    psi = psi_from_theta(theta, theta_r, theta_s, alpha, n, se_eps=1e-9)
    return np.maximum(psi, 1e-3)


def load_vg_params(json_file, depth_key):
    with open(FITS_DIR / json_file) as f:
        data = json.load(f)
    p = data[depth_key]["parameters"]
    return {k: p[k]["value"] for k in ("theta_r", "theta_s", "alpha", "n")}


def load_smap_values(every_nth=10):
    files = sorted(glob.glob(f"{SMAP_DIR}/*.tif"))
    files = files[::every_nth]
    vals = []
    for f in files:
        with rasterio.open(f) as src:
            arr = src.read(1).ravel()
            arr = arr[(arr > 0) & (arr < 1) & np.isfinite(arr)]
            vals.append(arr)
    return np.concatenate(vals)


def draw_smap_lines(ax, label_y=None, label_side="top"):
    """Draw P01 / median / P99 vertical lines on an axes."""
    ax.axvline(SMAP_P01, color="#A93226", lw=0.9, ls="--", zorder=2)
    ax.axvline(SMAP_MEDIAN, color="#444444", lw=0.8, ls=":", zorder=2)
    ax.axvline(SMAP_P99, color="#2E86C1", lw=0.9, ls="--", zorder=2)


def main(output_dir):
    fig, (ax_b, ax_c) = plt.subplots(
        2,
        1,
        figsize=(7.0, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.04},
    )

    # ── Panel B: SWRCs ─────────────────────────────────────────────
    site_positions = {
        "US-HB2, NH": (0.32, 10**1.0),
        "US-GLE, CO": (0.68, 10**1.8),
        "US-UTM, UT": (0.60, 10**1.0),
    }

    for jf, dk, label, color, site in SAMPLES:
        params = load_vg_params(jf, dk)
        tg = np.linspace(params["theta_r"] + 0.001, params["theta_s"] - 0.001, 300)
        psi = _vg_suction(tg, **params)
        ax_b.plot(tg, psi, color=color, lw=1.8, label=label, zorder=3)
        tx, ty = site_positions[site]
        ax_b.text(
            tx,
            ty,
            site,
            fontsize=7.5,
            color=color,
            fontstyle="italic",
            va="center",
            ha="left",
        )

    # SMAP vertical lines on Panel B
    draw_smap_lines(ax_b)

    # SMAP line labels on Panel B
    ax_b.text(
        SMAP_P01 + 0.012,
        10**1.2,
        f"SMAP L3 P01\n$\\theta$={SMAP_P01:.2f}",
        fontsize=7,
        color="#A93226",
        ha="left",
        va="center",
        linespacing=1.2,
    )
    ax_b.text(
        SMAP_MEDIAN + 0.012,
        10**3.8,
        f"SMAP L3 P50\n$\\theta$={SMAP_MEDIAN:.2f}",
        fontsize=7,
        color="#444444",
        ha="left",
        va="center",
        linespacing=1.2,
    )
    ax_b.text(
        SMAP_P99 - 0.012,
        SUCTION_HI * 0.30,
        f"SMAP L3 P99\n$\\theta$={SMAP_P99:.2f}",
        fontsize=7,
        color="#2E86C1",
        ha="right",
        va="top",
        linespacing=1.2,
    )

    # Threshold bands
    ax_b.axhspan(PWP_LO, PWP_HI, color="#791F1F", alpha=0.10, zorder=1)
    ax_b.axhline(PWP_LO, color="#791F1F", lw=0.5, ls=":", zorder=2)
    ax_b.axhline(PWP_HI, color="#791F1F", lw=0.5, ls=":", zorder=2)
    ax_b.text(
        0.60,
        (PWP_LO * PWP_HI) ** 0.5,
        "Wilting point",
        fontsize=7.5,
        color="#791F1F",
        va="center",
    )

    ax_b.axhspan(FC_LO, FC_HI, color="#1D9E75", alpha=0.10, zorder=1)
    ax_b.axhline(FC_LO, color="#1D9E75", lw=0.5, ls=":", zorder=2)
    ax_b.axhline(FC_HI, color="#1D9E75", lw=0.5, ls=":", zorder=2)
    ax_b.text(
        0.60,
        (FC_LO * FC_HI) ** 0.5,
        "Field capacity",
        fontsize=7.5,
        color="#0F6E56",
        va="center",
    )

    ax_b.set_yscale("log")
    ax_b.set_ylim(SUCTION_LO, SUCTION_HI)
    ax_b.set_ylabel(r"Suction (cm H$_2$O)", fontsize=10)
    ax_b.tick_params(labelsize=9)
    ax_b.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # ── Panel C: SMAP KDE ──────────────────────────────────────────
    vals = load_smap_values(every_nth=10)
    rng = np.random.default_rng(42)
    kde = gaussian_kde(rng.choice(vals, 500_000, replace=False), bw_method=0.012)
    x_grid = np.linspace(0.0, 0.75, 500)
    density = kde(x_grid)

    ax_c.plot(x_grid, density, color="#2E86C1", lw=1.0, zorder=3)
    ax_c.fill_between(x_grid, density, color="#2E86C1", alpha=0.08, zorder=2)

    draw_smap_lines(ax_c)

    ax_c.set_xlim(*XLIM)
    ax_c.set_ylim(0, None)
    ax_c.set_xlabel(r"Volumetric Water Content (m$^3$ m$^{-3}$)", fontsize=10)
    ax_c.set_ylabel("Density", fontsize=9)
    ax_c.tick_params(labelsize=9)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # Compact annotation — centered, right of P99
    ax_c.text(
        0.87,
        0.82,
        "SMAP L3 CONUS\n167M Pixel-Days",
        fontsize=7.5,
        color="#5F5E5A",
        ha="center",
        va="top",
        transform=ax_c.transAxes,
        linespacing=1.3,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig1_panel_bc.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out / 'fig1_panel_bc.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figure 1 Panels B+C")
    parser.add_argument(
        "--output-dir", default="figs/presentation", help="Output directory"
    )
    args = parser.parse_args()
    main(args.output_dir)
