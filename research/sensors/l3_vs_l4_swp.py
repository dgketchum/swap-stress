"""
Compare L3 vs L4 suction predictions against VG-derived observed PSI.

At MT Mesonet stations with both Bayesian VG fits and daily VWC time series:
  1. Convert in-situ 5 cm VWC → log10(PSI cm) using fitted VG parameters
  2. Extract predicted log10(suction cm) from L3 and L4 inference rasters
  3. Compare on matching dates

Produces:
  - l3_vs_l4_paired.parquet       Daily paired (obs, L3, L4) at all stations
  - l3_vs_l4_site_metrics.csv     Per-station RMSE, bias, r for each product
  - l3_vs_l4_summary.txt          Headline metrics and texture stratification
  - l3_vs_l4_scatter.png          L3 vs L4 scatter colored by texture
  - l3_vs_l4_bias_by_texture.png  Texture-stratified bias (leakage diagnostic)

Usage:
    python -m research.sensors.l3_vs_l4_swp \
        --l3-dir /path/to/inference_l3 \
        --l4-dir /path/to/inference_l4 \
        --output-dir /path/to/evaluation/l3_vs_l4
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

from research.vg_inversion.site_modeling.prep import select_params_from_bayes_json, theta_to_psi_cm

SUCTION_RE = re.compile(r"^suction_(\d{8})\.tif$")
NODATA = -9999.0

VG_DIR = "/nas/soils/soil_potential_obs/curve_fits/mt_mesonet/bayes"
VWC_DIR = "/nas/soils/vwc_timeseries/mt_mesonet/preprocessed_by_station"
META_CSV = "/nas/soils/vwc_timeseries/mt_mesonet/vwc_observation_summary.csv"
TRAINING_TABLE = "/nas/soils/swapstress/training/obs_level_training_9km_global.parquet"


def _load_stations() -> pd.DataFrame:
    """Load MT Mesonet stations that have both VG fits and VWC time series."""
    vg_ids = {os.path.splitext(f)[0] for f in os.listdir(VG_DIR) if f.endswith(".json")}
    vwc_ids = {
        f.replace("_daily.parquet", "")
        for f in os.listdir(VWC_DIR)
        if f.endswith(".parquet")
    }
    matched = sorted(vg_ids & vwc_ids)

    meta = pd.read_csv(META_CSV)
    meta = meta[meta["station"].isin(matched)].copy()

    # Add texture from training table via nearest-neighbour spatial join
    tt = pd.read_parquet(TRAINING_TABLE)
    tt_mt = tt[tt["source"] == "mt_mesonet"]
    tex_by_site = (
        tt_mt.groupby(["lat", "lon"])
        .agg(
            texture=(
                "TEXTURE_USDA",
                lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown",
            ),
            clay=("clay_tot_psa", "mean"),
            sand=("sand_tot_psa", "mean"),
        )
        .reset_index()
    )
    meta = meta.rename(columns={"latitude": "lat", "longitude": "lon"})
    from scipy.spatial import cKDTree

    tt_coords = tex_by_site[["lat", "lon"]].values
    meta_coords = meta[["lat", "lon"]].values
    tree = cKDTree(tt_coords)
    dists, idxs = tree.query(meta_coords)
    merged = meta.copy()
    merged["texture"] = np.where(
        dists < 0.05,
        tex_by_site.iloc[idxs]["texture"].values,
        "unknown",
    )
    merged["clay"] = np.where(
        dists < 0.05,
        tex_by_site.iloc[idxs]["clay"].values,
        np.nan,
    )
    merged["sand"] = np.where(
        dists < 0.05,
        tex_by_site.iloc[idxs]["sand"].values,
        np.nan,
    )

    print(f"Matched stations: {len(merged)} (VG fits + VWC series + metadata)")
    return merged


def _load_observed_psi(station_id: str, depth_cm: float = 5.0) -> pd.Series:
    """Load VWC time series and convert to log10(PSI cm) using Bayesian VG params."""
    # Load VG parameters
    vg_path = os.path.join(VG_DIR, f"{station_id}.json")
    with open(vg_path) as f:
        vg_data = json.load(f)
    params = select_params_from_bayes_json(vg_data, depth_cm=depth_cm)

    # Load VWC
    vwc_path = os.path.join(VWC_DIR, f"{station_id}_daily.parquet")
    df = pd.read_parquet(vwc_path)
    df["date"] = pd.to_datetime(df["datetime"], utc=True).dt.date
    vwc_col = f"soil_vwc_{int(depth_cm):04d}"
    if vwc_col not in df.columns:
        return pd.Series(dtype=float)

    theta = df.set_index("date")[vwc_col].dropna() / 100.0  # percent → fraction
    theta = theta[theta.between(0.01, 0.99)]

    psi_cm = theta_to_psi_cm(
        theta, params["theta_r"], params["theta_s"], params["alpha"], params["n"]
    )
    log10_psi = np.log10(psi_cm.clip(lower=1e-3))
    log10_psi = log10_psi.replace([np.inf, -np.inf], np.nan).dropna()
    log10_psi.name = "obs_log10_suction"
    return log10_psi


def _extract_raster_at_pixel(inference_dir: Path, row: int, col: int) -> pd.Series:
    """Extract daily suction from inference rasters at one pixel."""
    records = {}
    for fpath in sorted(inference_dir.glob("suction_*.tif")):
        m = SUCTION_RE.match(fpath.name)
        if not m:
            continue
        date_str = m.group(1)
        with rasterio.open(fpath) as src:
            val = src.read(1)[row, col]
        if np.isfinite(val) and val != NODATA:
            records[pd.Timestamp(date_str).date()] = val
    return pd.Series(records, dtype=float)


def _project_stations_to_grid(
    stations: pd.DataFrame, reference_tif: Path
) -> pd.DataFrame:
    """Add raster row/col for each station."""
    wgs_to_ease = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
    xs, ys = wgs_to_ease.transform(stations["lon"].values, stations["lat"].values)
    with rasterio.open(reference_tif) as src:
        inv = ~src.transform
        h, w = src.height, src.width
    cols, rows = inv * (xs, ys)
    stations = stations.copy()
    stations["pixel_row"] = np.round(rows).astype(int)
    stations["pixel_col"] = np.round(cols).astype(int)
    valid = (
        (stations["pixel_row"] >= 0)
        & (stations["pixel_row"] < h)
        & (stations["pixel_col"] >= 0)
        & (stations["pixel_col"] < w)
    )
    return stations[valid].reset_index(drop=True)


def _batch_extract_raster(
    inference_dir: Path, rows: np.ndarray, cols: np.ndarray
) -> dict[str, np.ndarray]:
    """Extract suction at all station pixels for every date. Returns {date_str: vals}."""
    result = {}
    files = sorted(inference_dir.glob("suction_*.tif"))
    for i, fpath in enumerate(files):
        m = SUCTION_RE.match(fpath.name)
        if not m:
            continue
        if i % 50 == 0:
            print(f"  {inference_dir.name}: file {i + 1}/{len(files)}")
        with rasterio.open(fpath) as src:
            data = src.read(1)
        vals = data[rows, cols].astype(np.float32)
        vals[(vals == NODATA) | ~np.isfinite(vals)] = np.nan
        result[m.group(1)] = vals
    return result


def run(l3_dir: str, l4_dir: str, output_dir: str) -> None:
    l3_path = Path(l3_dir)
    l4_path = Path(l4_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    stations = _load_stations()
    ref_tif = sorted(l3_path.glob("suction_*.tif"))[0]
    stations = _project_stations_to_grid(stations, ref_tif)
    print(f"{len(stations)} stations on raster grid")

    rows = stations["pixel_row"].values
    cols = stations["pixel_col"].values
    station_ids = stations["station"].values

    # --- Build observed PSI for each station ---
    print("Computing VG-derived observed PSI...")
    obs_by_station = {}
    for sid in station_ids:
        try:
            obs_by_station[sid] = _load_observed_psi(sid)
        except Exception as e:
            print(f"  SKIP {sid}: {e}")

    # --- Extract L3 and L4 predictions (batch) ---
    print("Extracting L3 predictions...")
    l3_data = _batch_extract_raster(l3_path, rows, cols)
    print("Extracting L4 predictions...")
    l4_data = _batch_extract_raster(l4_path, rows, cols)

    # --- Build paired daily table ---
    records = []
    for i, sid in enumerate(station_ids):
        if sid not in obs_by_station:
            continue
        obs = obs_by_station[sid]
        tex = stations.iloc[i].get("texture", "unknown")
        clay = stations.iloc[i].get("clay", np.nan)

        for date_str, l3_vals in l3_data.items():
            date = pd.Timestamp(date_str).date()
            l3_val = l3_vals[i]
            l4_val = l4_data.get(date_str, np.full(len(rows), np.nan))[i]
            obs_val = obs.get(date, np.nan)
            if np.isfinite(obs_val):
                records.append(
                    {
                        "station": sid,
                        "date": date,
                        "obs": obs_val,
                        "l3": l3_val,
                        "l4": l4_val,
                        "texture": tex,
                        "clay": clay,
                    }
                )
        # Also pick up L4-only dates
        for date_str, l4_vals in l4_data.items():
            if date_str in l3_data:
                continue
            date = pd.Timestamp(date_str).date()
            obs_val = obs.get(date, np.nan)
            if np.isfinite(obs_val):
                records.append(
                    {
                        "station": sid,
                        "date": date,
                        "obs": obs_val,
                        "l3": np.nan,
                        "l4": l4_vals[i],
                        "texture": tex,
                        "clay": clay,
                    }
                )

    paired = pd.DataFrame(records)
    paired.to_parquet(out_path / "l3_vs_l4_paired.parquet", index=False)
    print(f"Paired table: {len(paired)} rows, {paired['station'].nunique()} stations")

    # --- Per-station metrics ---
    def _metrics(grp, pred_col):
        valid = grp.dropna(subset=["obs", pred_col])
        if len(valid) < 10:
            return pd.Series({"rmse": np.nan, "bias": np.nan, "r": np.nan, "n": 0})
        resid = valid[pred_col] - valid["obs"]
        r = (
            np.corrcoef(valid["obs"], valid[pred_col])[0, 1]
            if valid[pred_col].std() > 0
            else np.nan
        )
        return pd.Series(
            {
                "rmse": np.sqrt((resid**2).mean()),
                "bias": resid.mean(),
                "r": r,
                "n": len(valid),
            }
        )

    l3_metrics = paired.groupby("station").apply(_metrics, "l3").add_prefix("l3_")
    l4_metrics = paired.groupby("station").apply(_metrics, "l4").add_prefix("l4_")
    site_metrics = l3_metrics.join(l4_metrics)

    # Add texture
    tex_map = stations.set_index("station")[["texture", "clay"]].to_dict("index")
    site_metrics["texture"] = site_metrics.index.map(
        lambda s: tex_map.get(s, {}).get("texture", "unknown")
    )
    site_metrics["clay"] = site_metrics.index.map(
        lambda s: tex_map.get(s, {}).get("clay", np.nan)
    )

    site_csv = out_path / "l3_vs_l4_site_metrics.csv"
    site_metrics.to_csv(site_csv)
    print(f"Saved {site_csv}")

    # --- Summary ---
    valid_both = site_metrics.dropna(subset=["l3_rmse", "l4_rmse"])
    lines = [
        "L3 vs L4 Suction Comparison — MT Mesonet Stations",
        "=" * 55,
        f"Stations with L3 metrics: {(~site_metrics.l3_rmse.isna()).sum()}",
        f"Stations with L4 metrics: {(~site_metrics.l4_rmse.isna()).sum()}",
        f"Stations with both: {len(valid_both)}",
        "",
        "Overall (station-median):",
        f"  L3 RMSE:  {valid_both.l3_rmse.median():.3f}  bias: {valid_both.l3_bias.median():+.3f}  r: {valid_both.l3_r.median():.3f}",
        f"  L4 RMSE:  {valid_both.l4_rmse.median():.3f}  bias: {valid_both.l4_bias.median():+.3f}  r: {valid_both.l4_r.median():.3f}",
        "",
        f"L4 wins (lower RMSE): {(valid_both.l4_rmse < valid_both.l3_rmse).sum()} / {len(valid_both)}",
        "",
        "By USDA texture:",
        f"{'texture':>20s} {'n':>4s}  {'L3 RMSE':>8s} {'L4 RMSE':>8s} {'L3 bias':>8s} {'L4 bias':>8s} {'L4 wins':>8s}",
    ]
    for tex, grp in sorted(valid_both.groupby("texture"), key=lambda x: -len(x[1])):
        if len(grp) < 3:
            continue
        wins = (grp.l4_rmse < grp.l3_rmse).sum()
        lines.append(
            f"{tex:>20s} {len(grp):4d}  {grp.l3_rmse.median():8.3f} {grp.l4_rmse.median():8.3f} "
            f"{grp.l3_bias.median():+8.3f} {grp.l4_bias.median():+8.3f} {wins:4d}/{len(grp)}"
        )

    summary = "\n".join(lines)
    print(summary)
    (out_path / "l3_vs_l4_summary.txt").write_text(summary)

    # --- Figures ---
    # 1) L3 vs L4 RMSE scatter by texture
    fig, ax = plt.subplots(figsize=(7, 7))
    top_tex = valid_both["texture"].value_counts().head(6).index.tolist()
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_tex)))
    for tex, c in zip(top_tex, colors):
        m = valid_both["texture"] == tex
        ax.scatter(
            valid_both.loc[m, "l3_rmse"],
            valid_both.loc[m, "l4_rmse"],
            s=30,
            alpha=0.7,
            label=f"{tex} (n={m.sum()})",
            color=c,
        )
    other = ~valid_both["texture"].isin(top_tex)
    if other.sum():
        ax.scatter(
            valid_both.loc[other, "l3_rmse"],
            valid_both.loc[other, "l4_rmse"],
            s=20,
            alpha=0.4,
            color="gray",
            label=f"other ({other.sum()})",
        )
    lim = max(valid_both.l3_rmse.max(), valid_both.l4_rmse.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("L3 RMSE (log10 cm)")
    ax.set_ylabel("L4 RMSE (log10 cm)")
    ax.set_title("Per-Station RMSE: L3 vs L4")
    ax.legend(fontsize=7)
    ax.set_aspect("equal")
    scatter_path = out_path / "l3_vs_l4_scatter.png"
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {scatter_path}")

    # 2) Bias by texture
    tex_data = []
    for tex, grp in sorted(valid_both.groupby("texture"), key=lambda x: -len(x[1])):
        if len(grp) < 3:
            continue
        tex_data.append(
            {
                "texture": tex,
                "n": len(grp),
                "l3_bias": grp.l3_bias.median(),
                "l4_bias": grp.l4_bias.median(),
            }
        )
    if tex_data:
        tdf = pd.DataFrame(tex_data).sort_values("n", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        y = np.arange(len(tdf))
        ax.barh(y - 0.15, tdf["l3_bias"], 0.3, label="L3 bias", color="steelblue")
        ax.barh(y + 0.15, tdf["l4_bias"], 0.3, label="L4 bias", color="coral")
        ax.set_yticks(y)
        ax.set_yticklabels(
            [f"{t} (n={n})" for t, n in zip(tdf.texture, tdf.n)], fontsize=8
        )
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("Median bias in log10 suction (cm)")
        ax.set_title("Suction bias by texture — leakage diagnostic")
        ax.legend(fontsize=8)
        bias_path = out_path / "l3_vs_l4_bias_by_texture.png"
        fig.savefig(bias_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {bias_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="L3 vs L4 suction evaluation")
    p.add_argument("--l3-dir", required=True)
    p.add_argument("--l4-dir", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    run(args.l3_dir, args.l4_dir, args.output_dir)
