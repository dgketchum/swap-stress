"""Figure 11b: Observed vs modeled drought time series at US-MMS.

Shows 2020 growing season at Morgan-Monroe State Forest (Indiana):
observed sensor suction (ReESH multi-sensor mean) vs. RF-predicted
suction extracted from the gap-filled CONUS product. Precip bars on
a secondary axis. Horizontal threshold lines at field capacity and
wilting point.

Usage:
    uv run python viz/presentation/fig11b_drought_timeseries.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

# --- Site ---
SITE = "US-MMS"
LON, LAT = -86.4131, 39.3232

# --- Data paths ---
OBS_PATH = Path("/nas/soils/swapstress/reesh_site_analysis/data/US_MMS.parquet")
PRED_DIR = Path("/nas/soils/swapstress/releases/global_pruned_refresh_20260520/gapfill")

# --- Display range ---
DATE_START = "2020-03-01"
DATE_END = "2020-11-15"

# --- Thresholds (log10 cm) ---
FC_LOG = 2.5  # ~316 cm ≈ -31 kPa
PWP_LOG = 4.18  # ~15,000 cm ≈ -1.5 MPa

NODATA = -9999.0


def _extract_predictions(date_start, date_end):
    """Extract predicted suction at US-MMS pixel for the date range."""
    ref_path = PRED_DIR / "suction_20200701.tif"
    with rasterio.open(ref_path) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x, y = transformer.transform(LON, LAT)
        row, col = src.index(x, y)

    dates, vals = [], []
    d = pd.Timestamp(date_start)
    end = pd.Timestamp(date_end)
    while d <= end:
        path = PRED_DIR / f"suction_{d.strftime('%Y%m%d')}.tif"
        if path.exists():
            with rasterio.open(path) as src:
                v = src.read(1)[row, col]
                if v != NODATA and np.isfinite(v):
                    dates.append(d)
                    vals.append(v)
        d += pd.Timedelta(days=1)
    return pd.Series(vals, index=dates, name="pred_log_psi")


def _load_observations(date_start, date_end):
    """Load observed suction, theta, and precip from the ReESH parquet."""
    df = pd.read_parquet(OBS_PATH)
    df = df[date_start:date_end].copy()
    psi_cols = [c for c in df.columns if c.startswith("psi_cm")]
    df["psi_mean"] = df[psi_cols].mean(axis=1)
    df["log_psi"] = np.log10(df["psi_mean"])
    return df


def main():
    print("Loading observations...", flush=True)
    obs = _load_observations(DATE_START, DATE_END)

    print("Extracting predictions...", flush=True)
    pred = _extract_predictions(DATE_START, DATE_END)

    # --- Figure ---
    fig, (ax_psi, ax_pr) = plt.subplots(
        2,
        1,
        figsize=(10, 5.5),
        dpi=200,
        height_ratios=[4, 1],
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.08, left=0.10, right=0.95, top=0.92, bottom=0.08)

    # --- Panel: suction time series ---
    # Individual sensors (thin, transparent)
    psi_cols = [c for c in obs.columns if c.startswith("psi_cm")]
    for i, col in enumerate(psi_cols):
        log_vals = np.log10(obs[col])
        ax_psi.plot(
            obs.index,
            log_vals,
            color="#999999",
            alpha=0.25,
            lw=0.6,
            zorder=1,
            label="Sensors" if i == 0 else None,
        )

    # Observed mean
    ax_psi.plot(
        obs.index,
        obs["log_psi"],
        color="#2C3E50",
        lw=1.6,
        alpha=0.9,
        zorder=3,
        label="Observed (sensor mean)",
    )

    # Predicted
    ax_psi.plot(
        pred.index,
        pred.values,
        color="#E74C3C",
        lw=1.6,
        ls="--",
        alpha=0.9,
        zorder=3,
        label="Predicted (RF, 9 km)",
    )

    # Thresholds
    ax_psi.axhline(FC_LOG, color="#27AE60", ls=":", lw=1.0, alpha=0.7, zorder=2)
    ax_psi.text(
        obs.index[2],
        FC_LOG - 0.08,
        "Field Capacity",
        fontsize=7.5,
        color="#27AE60",
        va="top",
    )
    ax_psi.axhline(PWP_LOG, color="#C0392B", ls=":", lw=1.0, alpha=0.7, zorder=2)
    ax_psi.text(
        obs.index[2],
        PWP_LOG + 0.08,
        "Permanent Wilting Point",
        fontsize=7.5,
        color="#C0392B",
        va="bottom",
    )

    ax_psi.set_ylabel(r"log$_{10}$ suction (cm H$_2$O)", fontsize=10)
    ax_psi.set_ylim(1.8, 4.5)
    ax_psi.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax_psi.set_title(
        f"{SITE}  ·  Morgan-Monroe State Forest, IN  ·  2020",
        fontsize=11,
        pad=6,
    )
    ax_psi.tick_params(labelsize=8)
    ax_psi.grid(axis="y", alpha=0.15)

    # Secondary y-axis: kPa
    ax_kpa = ax_psi.twinx()
    ax_kpa.set_ylim(1.8, 4.5)
    kpa_ticks = [2.0, 2.5, 3.0, 3.5, 4.0]
    ax_kpa.set_yticks(kpa_ticks)
    ax_kpa.set_yticklabels(
        [f"{10 ** (t - 1):.0f}" for t in kpa_ticks],
        fontsize=7,
    )
    ax_kpa.set_ylabel("kPa", fontsize=9, rotation=270, labelpad=12)
    ax_kpa.tick_params(length=0)

    # --- Panel: precipitation ---
    ax_pr.bar(
        obs.index,
        obs["pr"],
        width=1.0,
        color="#3498DB",
        alpha=0.6,
        edgecolor="none",
    )
    ax_pr.set_ylabel("Precip\n(mm)", fontsize=8)
    ax_pr.set_ylim(0, obs["pr"].max() * 1.15)
    ax_pr.invert_yaxis()
    ax_pr.tick_params(labelsize=7)
    ax_pr.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax_pr.xaxis.set_major_locator(mdates.MonthLocator())
    ax_pr.grid(axis="y", alpha=0.15)

    # --- Save ---
    out_dir = Path("figs/presentation")
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(
            out_dir / f"fig11b_drought_ts_{SITE}.{ext}",
            dpi=200,
            bbox_inches="tight",
        )
    plt.close(fig)
    print(f"Saved to {out_dir / f'fig11b_drought_ts_{SITE}.png'}")


if __name__ == "__main__":
    main()
