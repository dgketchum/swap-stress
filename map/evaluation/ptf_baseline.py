"""
Compare our direct SWP model against PTF-derived baselines (Rosetta, POLARIS).

For each day with SMAP L3 theta, inverts the van Genuchten equation using
Rosetta and POLARIS surface-layer parameters to compute suction, then compares
those physics-based estimates against our model's predictions pixel-by-pixel.

Subcommands
-----------
prep      Reproject Rosetta L2 to EASE-Grid2; export POLARIS 0-5 cm from EE.
evaluate  Run daily three-way comparison over a date range.
summarize Aggregate daily results into spatial maps and summary metrics.

Usage:
    python -m map.evaluation.ptf_baseline prep \
        --rosetta-tif /nas/soils/rosetta/geotiff/US_R3H3_L2_VG.tiff \
        --output-dir /nas/soils/swapstress/inference/conus_features

    python -m map.evaluation.ptf_baseline evaluate \
        --pred-dir /nas/soils/swapstress/inference/predictions/direct_qrf_9km_global \
        --smap-dir /nas/soils/smap/SPL3SMP_E/daily_tif \
        --static-dir /nas/soils/swapstress/inference/conus_features \
        --output /nas/soils/swapstress/evaluation/ptf_baseline \
        --start-date 20230701 --end-date 20230731

    python -m map.evaluation.ptf_baseline summarize \
        --eval-dir /nas/soils/swapstress/evaluation/ptf_baseline
"""

import argparse
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio

from map.data.reproject_to_ease2 import reproject_raster
from map.data.smap_download import (
    EASE2_CRS,
    MAP_SCALE,
    _conus_slice,
    _conus_transform,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SMAP_FILENAME_RE = re.compile(r"^smap_sm_(\d{8})\.tif$")
PRED_FILENAME_RE = re.compile(r"^suction_(\d{8})\.tif$")
NODATA = -9999.0

# Both Rosetta and POLARIS rasters use the same 4-band order:
#   1=theta_r  2=theta_s  3=alpha (log10 1/cm)  4=n
# Rosetta stores n as log10; POLARIS stores n in natural scale.

DEFAULT_SMAP_DIR = "/nas/soils/smap/SPL3SMP_E/daily_tif"
DEFAULT_STATIC_DIR = "/nas/soils/swapstress/inference/conus_features"
DEFAULT_PRED_DIR = "/nas/soils/swapstress/inference/predictions/direct_qrf_9km_global"
DEFAULT_OUTPUT_DIR = "/nas/soils/swapstress/evaluation/ptf_baseline"
DEFAULT_ROSETTA_TIF = "/nas/soils/rosetta/geotiff/US_R3H3_L2_VG.tiff"

_SE_EPS = 1e-6  # clamp effective saturation to (eps, 1-eps)


# ---------------------------------------------------------------------------
# Van Genuchten inverse: theta -> suction (cm)
# ---------------------------------------------------------------------------


def vg_suction(theta, theta_r, theta_s, alpha, n):
    """Compute suction (cm H2O) from theta via the inverse van Genuchten eq.

    Parameters
    ----------
    theta, theta_r, theta_s : array-like
        Volumetric water content, residual, and saturated (m3/m3).
    alpha : array-like
        VG alpha in 1/cm (natural scale, not log10).
    n : array-like
        VG shape parameter (natural scale, must be > 1).

    Returns
    -------
    np.ndarray
        Suction in cm H2O.  NaN where inputs are invalid.
    """
    theta = np.asarray(theta, dtype=np.float64)
    theta_r = np.asarray(theta_r, dtype=np.float64)
    theta_s = np.asarray(theta_s, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)

    out = np.full(theta.shape, np.nan, dtype=np.float64)

    valid = (
        np.isfinite(theta)
        & np.isfinite(theta_r)
        & np.isfinite(theta_s)
        & np.isfinite(alpha)
        & np.isfinite(n)
        & (n > 1.0)
        & (alpha > 0.0)
        & (theta_s > theta_r)
    )

    se = np.where(valid, (theta - theta_r) / (theta_s - theta_r), np.nan)
    se = np.clip(se, _SE_EPS, 1.0 - _SE_EPS)

    m = np.where(valid, 1.0 - 1.0 / n, np.nan)
    # h = (1/alpha) * (Se^(-1/m) - 1)^(1/n)
    h = np.where(
        valid,
        (1.0 / alpha) * (se ** (-1.0 / m) - 1.0) ** (1.0 / n),
        np.nan,
    )
    out[valid] = h[valid]
    return out


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------


def _ease2_grid():
    """Return (transform, width, height) for the CONUS EASE-Grid2 subset."""
    row_sl, col_sl = _conus_slice()
    transform = _conus_transform(row_sl, col_sl)
    width = col_sl.stop - col_sl.start
    height = row_sl.stop - row_sl.start
    return transform, width, height


def _read_band(path, band_idx):
    """Read a single band from a GeoTIFF, replacing nodata with NaN."""
    with rasterio.open(path) as src:
        data = src.read(band_idx).astype(np.float64)
        if src.nodata is not None and not np.isnan(src.nodata):
            data[data == src.nodata] = np.nan
    return data


# ---------------------------------------------------------------------------
# Prep subcommand
# ---------------------------------------------------------------------------


def prep_rosetta(rosetta_tif, output_dir, overwrite=False):
    """Reproject Rosetta L2 VG raster to the SMAP EASE-Grid2 CONUS grid."""
    dst_path = os.path.join(output_dir, "rosetta_l2_ease2.tif")
    if os.path.exists(dst_path) and not overwrite:
        print(f"Rosetta EASE2 raster exists: {dst_path} (use --overwrite)")
        return dst_path

    transform, width, height = _ease2_grid()
    print(f"Reprojecting {rosetta_tif} -> {dst_path}")
    print(f"  Target: {width}x{height}, EPSG:6933, {MAP_SCALE:.3f} m")

    # reproject_raster picks bilinear by default (continuous data)
    reproject_raster(rosetta_tif, dst_path, EASE2_CRS, transform, width, height)

    # Add band descriptions
    descs = ["theta_r", "theta_s", "log10_alpha", "log10_n", "log10_Ks"]
    with rasterio.open(dst_path, "r+") as ds:
        for i, d in enumerate(descs[: ds.count]):
            ds.set_band_description(i + 1, d)

    print(f"  Wrote {dst_path}")
    return dst_path


def prep_polaris_ee(output_dir, overwrite=False):
    """Export POLARIS 0-5 cm vG parameters from EE, reproject to EASE-Grid2.

    Exports a 4-band raster (theta_r, theta_s, alpha, n) from the POLARIS
    0-5 cm depth layer on Google Earth Engine.  The EE export writes to a
    local intermediate file in EPSG:5070 at 9 km, then reprojection produces
    the final EASE-Grid2 file.
    """
    import ee

    dst_path = os.path.join(output_dir, "polaris_0_5cm_ease2.tif")
    intermediate = os.path.join(output_dir, "polaris_0_5cm_9km.tif")

    if os.path.exists(dst_path) and not overwrite:
        print(f"POLARIS 0-5 cm EASE2 raster exists: {dst_path} (use --overwrite)")
        return dst_path

    if os.path.exists(intermediate) and not overwrite:
        print(f"Intermediate exists, skipping EE export: {intermediate}")
    else:
        ee.Initialize(project="ee-dgketchum")

        root = "projects/sat-io/open-datasets/polaris"
        params = [
            ("theta_r", f"{root}/theta_r_mean/theta_r_0_5"),
            ("theta_s", f"{root}/theta_s_mean/theta_s_0_5"),
            ("alpha", f"{root}/alpha_mean/alpha_0_5"),
            ("n", f"{root}/n_mean/n_0_5"),
        ]

        bands = []
        for name, asset in params:
            bands.append(ee.Image(asset).rename(name))
        stack = ee.Image.cat(bands)

        # CONUS bounding box
        roi = ee.Geometry.Rectangle([-125, 24, -66, 50])

        desc = "polaris_0_5cm_vg_9km"
        task = ee.batch.Export.image.toDrive(
            image=stack.clip(roi),
            description=desc,
            folder="swap_stress_polaris",
            fileNamePrefix=desc,
            scale=9000,
            crs="EPSG:5070",
            maxPixels=1e9,
        )
        task.start()
        print(f"Started EE export task: {desc} (id: {task.id})")
        print("Download the result from Google Drive and place it at:")
        print(f"  {intermediate}")
        print("Then re-run this command to reproject to EASE-Grid2.")
        return None

    # Reproject intermediate to EASE-Grid2
    transform, width, height = _ease2_grid()
    print(f"Reprojecting {intermediate} -> {dst_path}")
    reproject_raster(intermediate, dst_path, EASE2_CRS, transform, width, height)

    descs = ["theta_r", "theta_s", "alpha", "n"]
    with rasterio.open(dst_path, "r+") as ds:
        for i, d in enumerate(descs[: ds.count]):
            ds.set_band_description(i + 1, d)

    print(f"  Wrote {dst_path}")
    return dst_path


def run_prep(args):
    """Execute the prep subcommand."""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    overwrite = args.overwrite

    if args.rosetta:
        prep_rosetta(args.rosetta_tif, output_dir, overwrite)

    if args.polaris:
        prep_polaris_ee(output_dir, overwrite)


# ---------------------------------------------------------------------------
# Evaluate subcommand
# ---------------------------------------------------------------------------


def _load_vg_params(raster_path, log10_alpha=False, log10_n=False):
    """Load vG parameters from a 4+ band raster.

    Returns (theta_r, theta_s, alpha, n) as 2-D float64 arrays.
    Alpha and n are returned in natural (linear) scale.
    """
    theta_r = _read_band(raster_path, 1)
    theta_s = _read_band(raster_path, 2)
    alpha = _read_band(raster_path, 3)
    n = _read_band(raster_path, 4)

    if log10_alpha:
        alpha = np.power(10.0, alpha)
    if log10_n:
        n = np.power(10.0, n)

    return theta_r, theta_s, alpha, n


def _iter_date_files(smap_dir, pred_dir, start_date, end_date):
    """Yield (date, smap_path, pred_path) for days with both files present."""
    smap_path = Path(smap_dir)
    pred_path = Path(pred_dir)

    smap_dates = {}
    for p in sorted(smap_path.glob("smap_sm_*.tif")):
        m = SMAP_FILENAME_RE.match(p.name)
        if m:
            smap_dates[m.group(1)] = p

    for p in sorted(pred_path.glob("suction_*.tif")):
        m = PRED_FILENAME_RE.match(p.name)
        if not m:
            continue
        datestr = m.group(1)
        if datestr not in smap_dates:
            continue
        dt = datetime.strptime(datestr, "%Y%m%d")
        if start_date and dt < start_date:
            continue
        if end_date and dt > end_date:
            continue
        yield datestr, smap_dates[datestr], p


def _metrics(a, b):
    """Compute RMSE, bias, R^2 between two arrays (NaN-safe)."""
    mask = np.isfinite(a) & np.isfinite(b)
    n = mask.sum()
    if n < 2:
        return {"n": int(n), "rmse": np.nan, "bias": np.nan, "r2": np.nan}
    a_v, b_v = a[mask], b[mask]
    diff = a_v - b_v
    rmse = float(np.sqrt(np.mean(diff**2)))
    bias = float(np.mean(diff))
    ss_res = float(np.sum(diff**2))
    ss_tot = float(np.sum((b_v - b_v.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"n": int(n), "rmse": rmse, "bias": bias, "r2": r2}


def run_evaluate(args):
    """Execute the evaluate subcommand."""
    static_dir = args.static_dir
    smap_dir = args.smap_dir
    pred_dir = args.pred_dir
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    start = datetime.strptime(args.start_date, "%Y%m%d") if args.start_date else None
    end = datetime.strptime(args.end_date, "%Y%m%d") if args.end_date else None

    # Load Rosetta vG params (log10 alpha and n)
    ros_path = os.path.join(static_dir, "rosetta_l2_ease2.tif")
    if not os.path.exists(ros_path):
        raise FileNotFoundError(
            f"Rosetta EASE2 raster not found: {ros_path}\n"
            "Run the 'prep' subcommand first."
        )
    ros_tr, ros_ts, ros_a, ros_n = _load_vg_params(
        ros_path,
        log10_alpha=True,
        log10_n=True,
    )
    grid_shape = ros_tr.shape
    print(f"Rosetta L2 loaded: {grid_shape}")

    # Load POLARIS 0-5 cm vG params
    pol_path = os.path.join(static_dir, "polaris_0_5cm_ease2.tif")
    pol_available = os.path.exists(pol_path)
    if pol_available:
        pol_tr, pol_ts, pol_a, pol_n = _load_vg_params(
            pol_path,
            log10_alpha=True,
            log10_n=False,
        )
        print(f"POLARIS 0-5 cm loaded: {pol_tr.shape}")
    else:
        print(f"POLARIS 0-5 cm raster not found: {pol_path} (skipping POLARIS)")

    # Accumulators for per-pixel statistics
    n_days = np.zeros(grid_shape, dtype=np.int32)
    sum_model = np.zeros(grid_shape, dtype=np.float64)
    sum_ros = np.zeros(grid_shape, dtype=np.float64)
    sum_diff_model_ros = np.zeros(grid_shape, dtype=np.float64)
    sum_sq_diff_model_ros = np.zeros(grid_shape, dtype=np.float64)
    if pol_available:
        sum_pol = np.zeros(grid_shape, dtype=np.float64)
        sum_diff_model_pol = np.zeros(grid_shape, dtype=np.float64)
        sum_sq_diff_model_pol = np.zeros(grid_shape, dtype=np.float64)
        sum_diff_ros_pol = np.zeros(grid_shape, dtype=np.float64)
        sum_sq_diff_ros_pol = np.zeros(grid_shape, dtype=np.float64)

    # All pixel-day values for global metrics
    all_model = []
    all_ros = []
    all_pol = []

    day_count = 0
    for datestr, smap_file, pred_file in _iter_date_files(
        smap_dir,
        pred_dir,
        start,
        end,
    ):
        # Load SMAP theta
        with rasterio.open(smap_file) as src:
            theta = src.read(1).astype(np.float64)
            if src.nodata is not None and not np.isnan(src.nodata):
                theta[theta == src.nodata] = np.nan

        # Load model prediction (log10_suction_cm)
        with rasterio.open(pred_file) as src:
            model_log10 = src.read(1).astype(np.float64)
            if src.nodata is not None and not np.isnan(src.nodata):
                model_log10[model_log10 == src.nodata] = np.nan

        # Compute Rosetta suction
        ros_h = vg_suction(theta, ros_tr, ros_ts, ros_a, ros_n)
        ros_log10 = np.where(ros_h > 0, np.log10(ros_h), np.nan)

        # Compute POLARIS suction
        if pol_available:
            pol_h = vg_suction(theta, pol_tr, pol_ts, pol_a, pol_n)
            pol_log10 = np.where(pol_h > 0, np.log10(pol_h), np.nan)

        # Find pixels valid across model + rosetta (+ polaris if available)
        valid = np.isfinite(model_log10) & np.isfinite(ros_log10)
        if pol_available:
            valid_all = valid & np.isfinite(pol_log10)
        else:
            valid_all = valid

        n_days[valid_all] += 1
        sum_model[valid_all] += model_log10[valid_all]
        sum_ros[valid_all] += ros_log10[valid_all]
        diff_mr = model_log10 - ros_log10
        sum_diff_model_ros[valid_all] += diff_mr[valid_all]
        sum_sq_diff_model_ros[valid_all] += (diff_mr[valid_all]) ** 2

        if pol_available:
            sum_pol[valid_all] += pol_log10[valid_all]
            diff_mp = model_log10 - pol_log10
            sum_diff_model_pol[valid_all] += diff_mp[valid_all]
            sum_sq_diff_model_pol[valid_all] += (diff_mp[valid_all]) ** 2
            diff_rp = ros_log10 - pol_log10
            sum_diff_ros_pol[valid_all] += diff_rp[valid_all]
            sum_sq_diff_ros_pol[valid_all] += (diff_rp[valid_all]) ** 2

        # Collect for global metrics
        all_model.append(model_log10[valid_all])
        all_ros.append(ros_log10[valid_all])
        if pol_available:
            all_pol.append(pol_log10[valid_all])

        day_count += 1
        n_valid = int(valid_all.sum())
        print(f"  {datestr}: {n_valid} valid pixels")

    if day_count == 0:
        print("No matching date pairs found.")
        return

    print(f"\nProcessed {day_count} days")

    # --- Global metrics across all pixel-days ---

    all_model_arr = np.concatenate(all_model)
    all_ros_arr = np.concatenate(all_ros)

    pairs = {"model_vs_rosetta": (all_model_arr, all_ros_arr)}
    if pol_available:
        all_pol_arr = np.concatenate(all_pol)
        pairs["model_vs_polaris"] = (all_model_arr, all_pol_arr)
        pairs["rosetta_vs_polaris"] = (all_ros_arr, all_pol_arr)

    print("\n--- Global metrics (log10 suction cm) ---")
    for label, (a, b) in pairs.items():
        m = _metrics(a, b)
        print(
            f"  {label}: n={m['n']:,}  RMSE={m['rmse']:.4f}  "
            f"bias={m['bias']:.4f}  R2={m['r2']:.4f}"
        )

    # --- Spatial summary maps ---

    has_data = n_days > 0
    mean_model = np.where(has_data, sum_model / n_days, np.nan)
    mean_ros = np.where(has_data, sum_ros / n_days, np.nan)
    mean_diff_mr = np.where(has_data, sum_diff_model_ros / n_days, np.nan)
    rmse_mr = np.where(
        has_data,
        np.sqrt(sum_sq_diff_model_ros / n_days),
        np.nan,
    )

    bands_out = [
        ("mean_model_log10_suction", mean_model),
        ("mean_rosetta_log10_suction", mean_ros),
        ("mean_diff_model_minus_rosetta", mean_diff_mr),
        ("rmse_model_vs_rosetta", rmse_mr),
        ("n_days", n_days.astype(np.float32)),
    ]

    if pol_available:
        mean_pol = np.where(has_data, sum_pol / n_days, np.nan)
        mean_diff_mp = np.where(has_data, sum_diff_model_pol / n_days, np.nan)
        rmse_mp = np.where(
            has_data,
            np.sqrt(sum_sq_diff_model_pol / n_days),
            np.nan,
        )
        mean_diff_rp = np.where(has_data, sum_diff_ros_pol / n_days, np.nan)
        rmse_rp = np.where(
            has_data,
            np.sqrt(sum_sq_diff_ros_pol / n_days),
            np.nan,
        )
        bands_out.extend(
            [
                ("mean_polaris_log10_suction", mean_pol),
                ("mean_diff_model_minus_polaris", mean_diff_mp),
                ("rmse_model_vs_polaris", rmse_mp),
                ("mean_diff_rosetta_minus_polaris", mean_diff_rp),
                ("rmse_rosetta_vs_polaris", rmse_rp),
            ]
        )

    # Write summary raster
    transform, width, height = _ease2_grid()
    out_tif = os.path.join(output_dir, "ptf_comparison_summary.tif")

    with rasterio.open(
        out_tif,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=len(bands_out),
        dtype="float32",
        crs=EASE2_CRS,
        transform=transform,
        nodata=np.nan,
        compress="zstd",
    ) as dst:
        for i, (desc, data) in enumerate(bands_out, 1):
            dst.write(data.astype(np.float32), i)
            dst.set_band_description(i, desc)

    print(f"\nWrote summary raster: {out_tif}")
    print(f"  {len(bands_out)} bands, {width}x{height}, EPSG:6933")

    # Write metrics CSV
    import csv

    csv_path = os.path.join(output_dir, "ptf_comparison_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["comparison", "n", "rmse", "bias", "r2"])
        writer.writeheader()
        for label, (a, b) in pairs.items():
            row = {"comparison": label}
            row.update(_metrics(a, b))
            writer.writerow(row)
    print(f"Wrote metrics: {csv_path}")


# ---------------------------------------------------------------------------
# Summarize subcommand
# ---------------------------------------------------------------------------


def run_summarize(args):
    """Print summary from a completed evaluation directory."""
    import csv

    csv_path = os.path.join(args.eval_dir, "ptf_comparison_metrics.csv")
    tif_path = os.path.join(args.eval_dir, "ptf_comparison_summary.tif")

    if not os.path.exists(csv_path):
        print(f"No metrics file found: {csv_path}")
        return

    print("--- PTF Baseline Comparison Metrics ---\n")
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            print(
                f"  {row['comparison']}: n={int(row['n']):,}  "
                f"RMSE={float(row['rmse']):.4f}  "
                f"bias={float(row['bias']):.4f}  "
                f"R2={float(row['r2']):.4f}"
            )

    if os.path.exists(tif_path):
        with rasterio.open(tif_path) as src:
            print(f"\nSummary raster: {tif_path}")
            print(f"  Bands: {src.count}")
            for i in range(1, src.count + 1):
                desc = src.descriptions[i - 1] or f"band_{i}"
                data = src.read(i)
                valid = data[np.isfinite(data)]
                if len(valid):
                    print(
                        f"    {desc}: min={valid.min():.3f} "
                        f"max={valid.max():.3f} mean={valid.mean():.3f}"
                    )
                else:
                    print(f"    {desc}: no valid data")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compare our SWP model against Rosetta/POLARIS PTF baselines",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- prep ---
    p_prep = sub.add_parser("prep", help="Prepare PTF rasters on EASE-Grid2")
    p_prep.add_argument(
        "--rosetta-tif",
        default=DEFAULT_ROSETTA_TIF,
        help="Path to Rosetta L2 VG GeoTIFF (100 m, EPSG:5070)",
    )
    p_prep.add_argument(
        "--output-dir",
        default=DEFAULT_STATIC_DIR,
        help="Output directory for reprojected rasters",
    )
    p_prep.add_argument("--rosetta", action="store_true", help="Prep Rosetta raster")
    p_prep.add_argument(
        "--polaris",
        action="store_true",
        help="Export POLARIS 0-5 cm from EE",
    )
    p_prep.add_argument("--overwrite", action="store_true")
    p_prep.set_defaults(func=run_prep)

    # --- evaluate ---
    p_eval = sub.add_parser("evaluate", help="Run daily three-way comparison")
    p_eval.add_argument(
        "--pred-dir",
        default=DEFAULT_PRED_DIR,
        help="Model prediction directory",
    )
    p_eval.add_argument(
        "--smap-dir",
        default=DEFAULT_SMAP_DIR,
        help="SMAP daily GeoTIFF directory",
    )
    p_eval.add_argument(
        "--static-dir",
        default=DEFAULT_STATIC_DIR,
        help="Directory with PTF EASE2 rasters",
    )
    p_eval.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for results",
    )
    p_eval.add_argument("--start-date", default=None, help="YYYYMMDD")
    p_eval.add_argument("--end-date", default=None, help="YYYYMMDD")
    p_eval.set_defaults(func=run_evaluate)

    # --- summarize ---
    p_sum = sub.add_parser("summarize", help="Print summary from evaluation results")
    p_sum.add_argument(
        "--eval-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Evaluation output directory",
    )
    p_sum.set_defaults(func=run_summarize)

    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
