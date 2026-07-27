"""
Assemble multi-sensor flux tower validation dataset.

Selects qualifying flux tower sites, extracts daily soil moisture from SMAP L3,
SMAP L4, and SMOS-IC, runs point-level RF suction prediction, and merges with
eddy covariance ET, GPP, and meteorological covariates.

Produces:
  - flux_site_daily.parquet   Daily (site_id, date) rows with all predictors + responses
  - flux_site_meta.parquet    Per-site static metadata and data availability flags

Usage:
    python -m map.evaluation.flux_site_data \
        --output-dir /nas/soils/swapstress/evaluation/flux_validation
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

from map.evaluation.ptf_baseline import _sample_rosetta_at_sites
from swapstress.swrc import psi_from_theta
from map.inference.predict_rasters import (
    ModelArtifacts,
    StaticRasterStack,
    FIXED_FEATURES,
)
from retention_curve.depth_utils import depth_to_rosetta_level
from site_modeling.prep import find_ameriflux_file

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STATION_CSV = "/nas/climate/flux_stations/stations/flux_stations_28DEC2025.csv"
MANIFEST_CSV = "/nas/climate/flux_stations/reports/manifest.csv"
ICOS_ROOT = "/nas/climate/icos"
AMF_ROOT = "/nas/climate/ameriflux/amf_new"
SMAP_L3_DIR = "/nas/soils/smap/SPL3SMP_E/daily_tif_global"
SMAP_L4_DIR = "/nas/soils/smap/SPL4SMGP/daily_tif_global"
SMOS_IC_DIR = "/nas/soils/smos/SMOS_IC/daily_tif"
MODEL_DIR = "/nas/soils/swapstress/models/direct_rf_9km_global_pruned"
STATIC_DIR = "/nas/soils/swapstress/inference/global_features/rasters_ease2"
DEFAULT_OUTPUT_DIR = "/nas/soils/swapstress/evaluation/flux_validation"
ROSETTA_TIF = "/nas/soils/rosetta/geotiff/US_R3H3_L2_VG.tiff"

STUDY_START = "2015-04-01"
STUDY_END = "2021-02-08"

# Filename patterns
L3_RE = re.compile(r"^smap_sm_(\d{8})\.tif$")
L4_RE = re.compile(r"^smap_l4_(\d{8})\.tif$")
SMOS_RE = re.compile(r"^smos_ic_(\d{8})\.tif$")

# GPP conversion: umolCO2 m-2 s-1 (daily mean) → gC m-2 d-1
UMOL_TO_GC_DAY = 12e-6 * 86400  # 1.0368

# ICOS sentinel
ICOS_FILL = -9999.0

# Suction prediction passes: (theta_col, suction_col, depth_cm).
# Surface passes use the 0–5 cm layer (Rosetta L2); root-zone passes feed the
# L4 0–100 cm integrated theta at three proxy depths (Rosetta L4/L5/L6) so the
# regression can compare depth assumptions.
SURFACE_PASSES = [
    ("theta_l3", "suction_l3", 5.0),
    ("theta_l4_surf", "suction_l4", 5.0),
    ("theta_smos", "suction_smos", 5.0),
]
ROOTZONE_PASSES = [
    ("theta_l4_root", "suction_l4_root_30", 30.0),
    ("theta_l4_root", "suction_l4_root_50", 50.0),
    ("theta_l4_root", "suction_l4_root_100", 100.0),
]

# Root-distribution profile weighting (Jackson et al. 1996): the cumulative root
# fraction above depth d (cm) is Y(d) = 1 - beta**d.  Used to combine surface and
# root-zone suction into a single root-weighted profile predictor.
DEFAULT_ROOT_BETA = 0.961  # all-PFT global mean (Jackson et al. 1996)
ROOT_ZONE_DEPTH_CM = 100.0
ROOT_SPLIT_DEPTH_CM = 10.0


# ===================================================================
# Step 1: Site selection
# ===================================================================


def _find_icos_fullset_dd(site_id: str) -> str | None:
    """Find the ICOS FULLSET daily CSV for a site, preferring the latest version."""
    pattern = os.path.join(
        ICOS_ROOT,
        f"FLX_{site_id}_*",
        f"FLX_{site_id}_*FULLSET_DD_*.csv",
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    # Check that GPP column exists
    for path in reversed(matches):
        try:
            cols = pd.read_csv(path, nrows=0).columns
        except Exception:
            continue
        if "GPP_NT_VUT_REF" in cols or "GPP_DT_VUT_REF" in cols:
            return path
    return None


def select_sites(
    study_start: str = STUDY_START,
    study_end: str = STUDY_END,
) -> pd.DataFrame:
    """Select flux sites with ET_corr overlapping the study period.

    Joins the manifest (ET quality metadata) with the station coordinate CSV.
    Probes for ICOS and AmeriFlux GPP availability.
    """
    manifest = pd.read_csv(MANIFEST_CSV)
    manifest["start_date"] = pd.to_datetime(manifest["start_date"])
    manifest["end_date"] = pd.to_datetime(manifest["end_date"])
    start = pd.Timestamp(study_start)
    end = pd.Timestamp(study_end)

    sel = manifest[
        manifest["has_et_corr"]
        & manifest["is_preferred"]
        & (manifest["end_date"] >= start)
        & (manifest["start_date"] <= end)
    ].copy()

    # Join coordinates
    stations = pd.read_csv(STATION_CSV)
    stations = stations.rename(columns={"sid": "site_id"})
    sel = sel.merge(
        stations[["site_id", "lat", "lon"]].drop_duplicates("site_id"),
        on="site_id",
        how="left",
    )
    sel = sel.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    # Probe GPP availability
    icos_paths, amf_paths = [], []
    for sid in sel["site_id"]:
        icos_paths.append(_find_icos_fullset_dd(sid))
        try:
            amf_paths.append(find_ameriflux_file(AMF_ROOT, sid, period="HH"))
        except Exception:
            amf_paths.append(None)

    sel["icos_dd_path"] = icos_paths
    sel["amf_hh_path"] = amf_paths
    sel["has_gpp_icos"] = sel["icos_dd_path"].notna()
    sel["has_gpp_amf"] = sel["amf_hh_path"].notna()

    print(f"Selected {len(sel)} sites ({sel['network'].value_counts().to_dict()})")
    print(f"  GPP (ICOS): {sel['has_gpp_icos'].sum()}")
    print(f"  GPP (AmeriFlux): {sel['has_gpp_amf'].sum()}")
    return sel


# ===================================================================
# Step 2: Grid projection
# ===================================================================


def _project_to_grid(
    lats: np.ndarray,
    lons: np.ndarray,
    reference_tif: str,
    transformer: Transformer,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project WGS84 coords to raster pixel indices. Returns (rows, cols, valid_mask)."""
    xs, ys = transformer.transform(lons, lats)
    with rasterio.open(reference_tif) as src:
        inv = ~src.transform
        h, w = src.height, src.width
    cols, rows = inv * (xs, ys)
    rows = np.round(rows).astype(int)
    cols = np.round(cols).astype(int)
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    return rows, cols, valid


def project_sites_to_grids(
    sites: pd.DataFrame,
    m09_ref: str | None = None,
    m25_ref: str | None = None,
) -> pd.DataFrame:
    """Add M09 and M25 pixel coordinates to site table."""
    if m09_ref is None:
        m09_ref = sorted(glob.glob(os.path.join(SMAP_L3_DIR, "smap_sm_*.tif")))[0]
    if m25_ref is None:
        m25_ref = sorted(glob.glob(os.path.join(SMOS_IC_DIR, "smos_ic_*.tif")))[0]

    t = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
    lats = sites["lat"].values
    lons = sites["lon"].values

    r09, c09, v09 = _project_to_grid(lats, lons, m09_ref, t)
    r25, c25, v25 = _project_to_grid(lats, lons, m25_ref, t)
    valid = v09 & v25

    out = sites[valid].copy()
    out["pixel_row_m09"] = r09[valid]
    out["pixel_col_m09"] = c09[valid]
    out["pixel_row_m25"] = r25[valid]
    out["pixel_col_m25"] = c25[valid]
    dropped = (~valid).sum()
    if dropped:
        print(f"  Dropped {dropped} sites outside grid bounds")
    return out.reset_index(drop=True)


# ===================================================================
# Step 3: Theta extraction
# ===================================================================


def _extract_sensor(
    tif_dir: str,
    filename_re: re.Pattern,
    rows: np.ndarray,
    cols: np.ndarray,
    bands: list[int],
    band_names: list[str],
    start: datetime,
    end: datetime,
    label: str,
) -> dict[str, list]:
    """Extract pixel values at sites from daily TIFs for one sensor.

    Returns dict of lists keyed by column name, plus 'date' list.
    """
    data: dict[str, list] = {"date": []}
    for bn in band_names:
        data[bn] = []

    files = sorted(Path(tif_dir).glob("*.tif"))
    n_files = 0
    for i, fpath in enumerate(files):
        m = filename_re.match(fpath.name)
        if not m:
            continue
        dt = datetime.strptime(m.group(1), "%Y%m%d")
        if dt < start or dt > end:
            continue
        n_files += 1
        if n_files % 200 == 0:
            print(f"  {label}: {n_files} files processed")

        with rasterio.open(fpath) as src:
            for band_idx, bn in zip(bands, band_names):
                arr = src.read(band_idx)
                vals = arr[rows, cols].astype(np.float32)
                vals[(vals < 0) | (vals > 1) | ~np.isfinite(vals)] = np.nan
                data[bn].append(vals)
        data["date"].append(dt.date())

    print(f"  {label}: {n_files} files total")
    return data


def _sensor_dict_to_df(
    data: dict[str, list],
    site_ids: np.ndarray,
    band_names: list[str],
) -> pd.DataFrame:
    """Convert _extract_sensor output to a long DataFrame."""
    if not data["date"]:
        return pd.DataFrame(columns=["site_id", "date"] + band_names)

    n_sites = len(site_ids)
    n_dates = len(data["date"])
    dates = np.repeat(data["date"], n_sites)
    sids = np.tile(site_ids, n_dates)

    df = pd.DataFrame({"site_id": sids, "date": dates})
    for bn in band_names:
        df[bn] = np.concatenate(data[bn])
    return df


def extract_theta_timeseries(
    sites: pd.DataFrame,
    study_start: str = STUDY_START,
    study_end: str = STUDY_END,
) -> pd.DataFrame:
    """Extract daily soil moisture at all sites from three sensors."""
    start = datetime.strptime(study_start, "%Y-%m-%d")
    end = datetime.strptime(study_end, "%Y-%m-%d")
    site_ids = sites["site_id"].values
    r09 = sites["pixel_row_m09"].values
    c09 = sites["pixel_col_m09"].values
    r25 = sites["pixel_row_m25"].values
    c25 = sites["pixel_col_m25"].values

    print("Extracting SMAP L3...")
    l3_data = _extract_sensor(
        SMAP_L3_DIR,
        L3_RE,
        r09,
        c09,
        bands=[1],
        band_names=["theta_l3"],
        start=start,
        end=end,
        label="L3",
    )
    l3_df = _sensor_dict_to_df(l3_data, site_ids, ["theta_l3"])

    print("Extracting SMAP L4...")
    l4_data = _extract_sensor(
        SMAP_L4_DIR,
        L4_RE,
        r09,
        c09,
        bands=[1, 2],
        band_names=["theta_l4_surf", "theta_l4_root"],
        start=start,
        end=end,
        label="L4",
    )
    l4_df = _sensor_dict_to_df(l4_data, site_ids, ["theta_l4_surf", "theta_l4_root"])

    print("Extracting SMOS-IC...")
    smos_data = _extract_sensor(
        SMOS_IC_DIR,
        SMOS_RE,
        r25,
        c25,
        bands=[1],
        band_names=["theta_smos"],
        start=start,
        end=end,
        label="SMOS",
    )
    smos_df = _sensor_dict_to_df(smos_data, site_ids, ["theta_smos"])

    # Merge all sensors
    theta = l3_df.merge(l4_df, on=["site_id", "date"], how="outer")
    theta = theta.merge(smos_df, on=["site_id", "date"], how="outer")

    # Drop rows where all theta values are NaN
    theta_cols = ["theta_l3", "theta_l4_surf", "theta_l4_root", "theta_smos"]
    for c in theta_cols:
        if c not in theta.columns:
            theta[c] = np.nan
    theta = theta.dropna(subset=theta_cols, how="all").reset_index(drop=True)

    print(f"Theta table: {len(theta)} rows, {theta['site_id'].nunique()} sites")
    return theta


# ===================================================================
# Step 4: Point inference
# ===================================================================


def _site_flat_indices(
    theta_df: pd.DataFrame,
    sites: pd.DataFrame,
    grid_w: int,
) -> np.ndarray:
    """Map each row of theta_df to its M09 flat pixel index via site coordinates."""
    site_flat_idx = {}
    for _, row in sites.iterrows():
        sid = row["site_id"]
        flat = int(row["pixel_row_m09"]) * grid_w + int(row["pixel_col_m09"])
        site_flat_idx[sid] = flat
    return theta_df["site_id"].map(site_flat_idx).values.astype(int)


def _run_suction_passes(
    theta_df: pd.DataFrame,
    flat_indices: np.ndarray,
    arts: ModelArtifacts,
    static_stack: StaticRasterStack,
    passes: list[tuple[str, str, float]],
) -> pd.DataFrame:
    """Predict suction for each (theta_col, suction_col, depth_cm) pass.

    Each pass swaps the theta input and the depth_cm / rosetta_level features;
    the static covariates are depth-independent and reused across passes.
    """
    for theta_col, suction_col, depth_cm in passes:
        rosetta_level = depth_to_rosetta_level(depth_cm)
        theta_df[suction_col] = np.nan
        valid_mask = theta_df[theta_col].notna().values
        if valid_mask.sum() == 0:
            print(f"  {suction_col}: 0 valid theta — skipped")
            continue

        n_valid = int(valid_mask.sum())
        valid_idx = flat_indices[valid_mask]
        theta_vals = theta_df.loc[valid_mask, theta_col].values.astype(np.float32)

        # Build feature matrix in model feature order
        n_features = len(arts.feature_names)
        X = np.empty((n_valid, n_features), dtype=np.float32)
        for col_i, feat in enumerate(arts.feature_names):
            if feat == "theta":
                X[:, col_i] = theta_vals
            elif feat == "depth_cm":
                X[:, col_i] = depth_cm
            elif feat == "rosetta_level":
                X[:, col_i] = rosetta_level
            else:
                static_idx = static_stack.feature_to_index[feat]
                X[:, col_i] = static_stack.data[static_idx, valid_idx]

        X_imp = arts.imputer.transform(X)
        preds = arts.model.predict(X_imp).astype(np.float32)
        theta_df.loc[valid_mask, suction_col] = preds
        print(
            f"  {suction_col}: {n_valid} predictions "
            f"(depth_cm={depth_cm:g}, L{rosetta_level})"
        )

    return theta_df


def predict_suction_at_sites(
    theta_df: pd.DataFrame,
    sites: pd.DataFrame,
    model_dir: str = MODEL_DIR,
    static_dir: str = STATIC_DIR,
    passes: list[tuple[str, str, float]] | None = None,
) -> pd.DataFrame:
    """Run the trained RF model at site locations for each prediction pass.

    Batches all valid rows per pass for efficient prediction. By default runs
    the surface passes (0–5 cm for L3/L4/SMOS) plus the L4 root-zone passes
    (30/50/100 cm). Pass an explicit ``passes`` list to run a subset.
    """
    if passes is None:
        passes = SURFACE_PASSES + ROOTZONE_PASSES

    print("Loading model artifacts...")
    arts = ModelArtifacts.load(model_dir)
    feature_set = set(arts.feature_names)
    static_features = feature_set - {"theta"} - FIXED_FEATURES
    static_stack = StaticRasterStack.load(static_dir, static_features)
    grid_w = static_stack.grid.width

    flat_indices = _site_flat_indices(theta_df, sites, grid_w)
    return _run_suction_passes(theta_df, flat_indices, arts, static_stack, passes)


# ===================================================================
# Step 4b: PTF baseline suction
# ===================================================================


def add_ptf_suction(
    theta_df: pd.DataFrame,
    sites: pd.DataFrame,
    rosetta_tif: str = ROSETTA_TIF,
) -> pd.DataFrame:
    """Add Rosetta PTF-derived suction for each sensor's theta.

    Samples Rosetta L2 VG parameters (theta_r, theta_s, alpha, n) at each site,
    then inverts the VG equation at each (site, date, theta) to get log10(suction cm).
    CONUS-only — non-US sites get NaN.
    """
    print(f"  Sampling Rosetta VG params at {len(sites)} sites...")
    ros = _sample_rosetta_at_sites(
        rosetta_tif, sites["lat"].values, sites["lon"].values
    )
    site_ros = pd.DataFrame(
        {
            "site_id": sites["site_id"].values,
            "ros_theta_r": ros["ros_theta_r"],
            "ros_theta_s": ros["ros_theta_s"],
            "ros_alpha": ros["ros_alpha"],
            "ros_n": ros["ros_n"],
        }
    )
    n_valid = site_ros["ros_theta_r"].notna().sum()
    print(f"  Rosetta coverage: {n_valid}/{len(sites)} sites (CONUS only)")

    # Join VG params to daily table
    merged = theta_df.merge(site_ros, on="site_id", how="left")

    # Invert VG for each theta source
    sensor_cols = [
        ("theta_l3", "suction_ptf_l3"),
        ("theta_l4_surf", "suction_ptf_l4"),
        ("theta_smos", "suction_ptf_smos"),
    ]
    for theta_col, suction_col in sensor_cols:
        psi_cm = psi_from_theta(
            merged[theta_col].values,
            merged["ros_theta_r"].values,
            merged["ros_theta_s"].values,
            merged["ros_alpha"].values,
            merged["ros_n"].values,
        )
        merged[suction_col] = np.log10(np.maximum(psi_cm, 1e-3))
        n = np.isfinite(merged[suction_col]).sum()
        print(f"  {suction_col}: {n} valid")

    # Drop VG param columns (site-level, not needed in daily table)
    merged = merged.drop(columns=["ros_theta_r", "ros_theta_s", "ros_alpha", "ros_n"])
    return merged


# ===================================================================
# Step 4c: Root-weighted profile suction
# ===================================================================


def root_fraction_weights(
    beta: float = DEFAULT_ROOT_BETA,
    d_split: float = ROOT_SPLIT_DEPTH_CM,
    d_root: float = ROOT_ZONE_DEPTH_CM,
) -> tuple[float, float]:
    """Two-layer root-fraction weights (surface, deep) from Jackson et al. (1996).

    Y(d) = 1 - beta**d is the cumulative root fraction above depth d (cm). The
    surface layer holds roots in [0, d_split] (represented by surface suction);
    the deep layer holds roots in [d_split, d_root] (represented by the
    root-zone suction). Weights are normalized over the [0, d_root] root zone.
    """
    y_split = 1.0 - beta**d_split
    y_root = 1.0 - beta**d_root
    w_surf = y_split / y_root
    return w_surf, 1.0 - w_surf


def add_rootzone_profile(
    df: pd.DataFrame,
    surf_col: str = "suction_l4",
    root_col: str = "suction_l4_root_50",
    out_col: str = "suction_l4_prof",
    beta: float = DEFAULT_ROOT_BETA,
) -> pd.DataFrame:
    """Add a root-fraction-weighted profile suction from surface + root-zone psi.

    Combines in log10(cm) space: psi_prof = w_surf * psi_surf + w_deep * psi_root.
    The result is a convex combination, so it lies between the surface and
    root-zone suction.
    """
    w_surf, w_deep = root_fraction_weights(beta=beta)
    print(f"  Profile weights (beta={beta}): surface={w_surf:.3f}, deep={w_deep:.3f}")
    df[out_col] = w_surf * df[surf_col] + w_deep * df[root_col]
    n = int(df[out_col].notna().sum())
    print(f"  {out_col}: {n} valid")
    return df


# ===================================================================
# Step 5: Flux data loading
# ===================================================================


def load_et_from_qaqc(
    sites: pd.DataFrame,
    study_start: str = STUDY_START,
    study_end: str = STUDY_END,
) -> pd.DataFrame:
    """Load energy-balance-corrected ET and met covariates from QA/QC daily files."""
    start = pd.Timestamp(study_start)
    end = pd.Timestamp(study_end)
    frames = []

    for _, row in sites.iterrows():
        daily_file = row["daily_file"]
        sid = row["site_id"]
        if not os.path.exists(daily_file):
            continue

        try:
            df = pd.read_csv(daily_file, parse_dates=["date"])
        except Exception:
            continue

        df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
        if df.empty:
            continue

        # ET: prefer ET_corr, fall back to ET
        et_col = "ET_corr" if "ET_corr" in df.columns else "ET"
        if et_col not in df.columns:
            continue

        out = pd.DataFrame(
            {
                "site_id": sid,
                "date": df["date"].dt.date,
                "et_corr": df[et_col].values,
            }
        )

        # Met covariates
        for col in ("sw_in", "t_avg", "ppt"):
            out[col] = df[col].values if col in df.columns else np.nan

        # VPD: direct column, or derive from vp + t_avg
        if "vpd" in df.columns:
            out["vpd"] = df["vpd"].values
        elif "vp" in df.columns and "t_avg" in df.columns:
            t = df["t_avg"].values
            es = 0.6108 * np.exp(17.27 * t / (t + 237.3))
            out["vpd"] = es - df["vp"].values
        else:
            out["vpd"] = np.nan

        frames.append(out)

    if not frames:
        return pd.DataFrame(
            columns=["site_id", "date", "et_corr", "sw_in", "t_avg", "vpd", "ppt"]
        )
    et_df = pd.concat(frames, ignore_index=True)
    print(f"ET data: {len(et_df)} rows, {et_df['site_id'].nunique()} sites")
    return et_df


def _load_gpp_icos(site_id: str, icos_dd_path: str) -> pd.DataFrame | None:
    """Load daily GPP from an ICOS FULLSET DD file."""
    try:
        df = pd.read_csv(icos_dd_path)
    except Exception:
        return None

    # Prefer nighttime partitioning
    gpp_col = None
    for candidate in ("GPP_NT_VUT_REF", "GPP_DT_VUT_REF"):
        if candidate in df.columns:
            gpp_col = candidate
            break
    if gpp_col is None:
        return None

    df[gpp_col] = pd.to_numeric(df[gpp_col], errors="coerce")
    df.loc[df[gpp_col] == ICOS_FILL, gpp_col] = np.nan

    # Parse TIMESTAMP (YYYYMMDD)
    ts_col = "TIMESTAMP" if "TIMESTAMP" in df.columns else df.columns[0]
    df["date"] = pd.to_datetime(df[ts_col].astype(str).str[:8], format="%Y%m%d").dt.date

    gpp_gC = df[gpp_col] * UMOL_TO_GC_DAY
    return pd.DataFrame(
        {
            "site_id": site_id,
            "date": df["date"],
            "gpp": gpp_gC.values,
            "gpp_source": "icos_fullset",
        }
    )


def _load_gpp_ameriflux(site_id: str, amf_hh_path: str) -> pd.DataFrame | None:
    """Load daily GPP from AmeriFlux BASE half-hourly files with proper unit conversion."""
    try:
        df = pd.read_csv(amf_hh_path, skiprows=2)
    except Exception:
        return None

    # Replace sentinels
    df.replace({-9999.0: np.nan, -99.99: np.nan}, inplace=True)

    # Find GPP columns: prefer gap-filled
    gpp_cols = [c for c in df.columns if c.upper().startswith("GPP_PI_F")]
    if not gpp_cols:
        gpp_cols = [c for c in df.columns if c.upper().startswith("GPP_PI")]
    if not gpp_cols:
        gpp_cols = [c for c in df.columns if c.upper().startswith("GPP")]
    if not gpp_cols:
        return None

    gpp_raw = pd.concat(
        [pd.to_numeric(df[c], errors="coerce") for c in gpp_cols], axis=1
    ).mean(axis=1)
    # Note: GPP=0 at night is legitimate — do NOT replace zeros with NaN

    # Parse timestamps
    if "TIMESTAMP_START" not in df.columns:
        return None
    ts = pd.to_datetime(df["TIMESTAMP_START"].astype(str), format="%Y%m%d%H%M")
    df_ts = pd.DataFrame({"gpp_umol": gpp_raw.values}, index=ts)

    # Detect time step
    diffs_td = np.diff(ts.values)
    diffs_sec = diffs_td / np.timedelta64(1, "s")
    dt_seconds = (
        float(np.median(diffs_sec[diffs_sec > 0])) if len(diffs_sec) > 0 else 1800.0
    )

    # Convert umolCO2 m-2 s-1 → gC per timestep, then sum daily
    df_ts["gpp_gC_step"] = df_ts["gpp_umol"] * dt_seconds * 12e-6

    # Daily aggregation: require ≥80% of expected records
    expected_per_day = int(round(86400 / dt_seconds))
    min_records = int(expected_per_day * 0.8)

    daily = df_ts.resample("D").agg(
        gpp=("gpp_gC_step", "sum"),
        n_records=("gpp_gC_step", "count"),
    )
    daily.loc[daily["n_records"] < min_records, "gpp"] = np.nan
    daily = daily[daily["gpp"].notna()]

    if daily.empty:
        return None

    return pd.DataFrame(
        {
            "site_id": site_id,
            "date": daily.index.date,
            "gpp": daily["gpp"].values,
            "gpp_source": "ameriflux_hh",
        }
    )


def load_all_flux_data(
    sites: pd.DataFrame,
    study_start: str = STUDY_START,
    study_end: str = STUDY_END,
) -> pd.DataFrame:
    """Load ET and GPP for all qualifying sites."""
    start = pd.Timestamp(study_start).date()
    end = pd.Timestamp(study_end).date()

    # ET
    et_df = load_et_from_qaqc(sites, study_start, study_end)

    # GPP: ICOS first, then AmeriFlux for sites not covered by ICOS
    gpp_frames = []
    icos_sites = set()

    for _, row in sites[sites["has_gpp_icos"]].iterrows():
        gpp = _load_gpp_icos(row["site_id"], row["icos_dd_path"])
        if gpp is not None and not gpp.empty:
            gpp = gpp[(gpp["date"] >= start) & (gpp["date"] <= end)]
            gpp_frames.append(gpp)
            icos_sites.add(row["site_id"])

    for _, row in sites[sites["has_gpp_amf"]].iterrows():
        if row["site_id"] in icos_sites:
            continue
        gpp = _load_gpp_ameriflux(row["site_id"], row["amf_hh_path"])
        if gpp is not None and not gpp.empty:
            gpp = gpp[(gpp["date"] >= start) & (gpp["date"] <= end)]
            gpp_frames.append(gpp)

    if gpp_frames:
        gpp_df = pd.concat(gpp_frames, ignore_index=True)
        print(f"GPP data: {len(gpp_df)} rows, {gpp_df['site_id'].nunique()} sites")
        flux = et_df.merge(gpp_df, on=["site_id", "date"], how="left")
    else:
        flux = et_df.copy()
        flux["gpp"] = np.nan
        flux["gpp_source"] = None
        print("GPP data: 0 rows")

    return flux


# ===================================================================
# Step 6: Assembly
# ===================================================================


def assemble_daily_table(
    theta_suction: pd.DataFrame,
    flux: pd.DataFrame,
    sites: pd.DataFrame,
) -> pd.DataFrame:
    """Merge theta/suction with flux observations."""
    daily = theta_suction.merge(flux, on=["site_id", "date"], how="inner")

    # Add site metadata
    meta_cols = ["site_id", "network", "lat", "lon"]
    site_meta = sites[meta_cols].drop_duplicates("site_id")
    daily = daily.merge(site_meta, on="site_id", how="left")

    print(f"Daily table: {len(daily)} rows, {daily['site_id'].nunique()} sites")
    return daily


def build_site_meta(
    daily: pd.DataFrame,
    sites: pd.DataFrame,
) -> pd.DataFrame:
    """Build per-site metadata summary."""
    meta = sites[
        [
            "site_id",
            "network",
            "lat",
            "lon",
            "start_date",
            "end_date",
            "has_gpp_icos",
            "has_gpp_amf",
            "pixel_row_m09",
            "pixel_col_m09",
            "pixel_row_m25",
            "pixel_col_m25",
        ]
    ].copy()

    # Count valid days per sensor
    for col in (
        "et_corr",
        "theta_l3",
        "theta_l4_surf",
        "theta_l4_root",
        "theta_smos",
        "gpp",
    ):
        if col in daily.columns:
            counts = daily.dropna(subset=[col]).groupby("site_id").size()
            meta[f"n_days_{col}"] = meta["site_id"].map(counts).fillna(0).astype(int)

    # GPP source
    if "gpp_source" in daily.columns:
        src_map = (
            daily.dropna(subset=["gpp_source"]).groupby("site_id")["gpp_source"].first()
        )
        meta["gpp_source"] = meta["site_id"].map(src_map)

    meta["has_gpp"] = meta.get("n_days_gpp", 0) > 0
    return meta


# ===================================================================
# Step 7: Driver
# ===================================================================


def run(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    study_start: str = STUDY_START,
    study_end: str = STUDY_END,
    model_dir: str = MODEL_DIR,
    static_dir: str = STATIC_DIR,
) -> None:
    """Full pipeline: select → project → extract → predict → load flux → assemble."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Site selection
    print("=" * 60)
    print("Step 1: Site selection")
    sites = select_sites(study_start, study_end)

    # 2. Grid projection
    print("\nStep 2: Grid projection")
    sites = project_sites_to_grids(sites)

    # 3. Theta extraction
    print("\nStep 3: Theta extraction")
    theta = extract_theta_timeseries(sites, study_start, study_end)

    # 4. Point inference
    print("\nStep 4: Point inference")
    theta = predict_suction_at_sites(
        theta, sites, model_dir=model_dir, static_dir=static_dir
    )

    # 4b. PTF baseline
    print("\nStep 4b: PTF baseline suction")
    theta = add_ptf_suction(theta, sites)

    # 4c. Root-weighted profile suction
    print("\nStep 4c: Root-weighted profile suction")
    theta = add_rootzone_profile(theta)

    # 5. Flux data
    print("\nStep 5: Flux data loading")
    flux = load_all_flux_data(sites, study_start, study_end)

    # 6. Assembly
    print("\nStep 6: Assembly")
    daily = assemble_daily_table(theta, flux, sites)
    meta = build_site_meta(daily, sites)

    # Write outputs
    daily_path = out / "flux_site_daily.parquet"
    meta_path = out / "flux_site_meta.parquet"
    daily.to_parquet(daily_path, index=False)
    meta.to_parquet(meta_path, index=False)
    print(f"\nSaved {daily_path}")
    print(f"Saved {meta_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  Sites: {daily['site_id'].nunique()}")
    print(f"  Date range: {daily['date'].min()} to {daily['date'].max()}")
    print(f"  Total rows: {len(daily)}")
    for col in (
        "et_corr",
        "gpp",
        "theta_l3",
        "theta_l4_surf",
        "theta_l4_root",
        "theta_smos",
        "suction_l3",
        "suction_l4",
        "suction_smos",
        "suction_l4_root_30",
        "suction_l4_root_50",
        "suction_l4_root_100",
        "suction_l4_prof",
        "suction_ptf_l3",
        "suction_ptf_l4",
        "suction_ptf_smos",
    ):
        if col in daily.columns:
            n = daily[col].notna().sum()
            print(f"  {col}: {n} valid ({100 * n / len(daily):.1f}%)")


def add_ptf_to_existing(output_dir: str = DEFAULT_OUTPUT_DIR) -> None:
    """Add PTF suction columns to an existing flux_site_daily.parquet."""
    out = Path(output_dir)
    daily_path = out / "flux_site_daily.parquet"
    meta_path = out / "flux_site_meta.parquet"

    daily = pd.read_parquet(daily_path)
    meta = pd.read_parquet(meta_path)
    print(f"Loaded {daily_path}: {len(daily)} rows")

    # Drop existing PTF columns if re-running
    ptf_cols = [c for c in daily.columns if c.startswith("suction_ptf")]
    if ptf_cols:
        daily = daily.drop(columns=ptf_cols)
        print(f"  Dropped existing columns: {ptf_cols}")

    daily = add_ptf_suction(daily, meta)

    daily.to_parquet(daily_path, index=False)
    print(f"Saved {daily_path}")

    n = len(daily)
    for col in [c for c in daily.columns if c.startswith("suction_ptf")]:
        v = daily[col].notna().sum()
        print(f"  {col}: {v} valid ({100 * v / n:.1f}%)")


def _rootzone_cols(columns) -> list[str]:
    """Root-zone + profile suction column names present in a frame."""
    return [
        c for c in columns if c.startswith("suction_l4_root") or c == "suction_l4_prof"
    ]


def add_rootzone_to_existing(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    model_dir: str = MODEL_DIR,
    static_dir: str = STATIC_DIR,
) -> None:
    """Add root-zone + profile suction columns to an existing daily parquet.

    Reuses the already-extracted ``theta_l4_root`` column, so no daily TIFs are
    re-read. Runs RF inference at 30/50/100 cm and computes the root-weighted
    profile from surface + root-zone suction.
    """
    out = Path(output_dir)
    daily_path = out / "flux_site_daily.parquet"
    meta_path = out / "flux_site_meta.parquet"

    daily = pd.read_parquet(daily_path)
    meta = pd.read_parquet(meta_path)
    print(f"Loaded {daily_path}: {len(daily)} rows")

    if "theta_l4_root" not in daily.columns:
        raise ValueError(
            "theta_l4_root not in parquet — run the full pipeline to extract it"
        )
    if "suction_l4" not in daily.columns:
        raise ValueError(
            "suction_l4 (surface) not in parquet — required for the profile predictor"
        )

    # Drop existing root/profile columns if re-running
    existing = _rootzone_cols(daily.columns)
    if existing:
        daily = daily.drop(columns=existing)
        print(f"  Dropped existing columns: {existing}")

    daily = predict_suction_at_sites(
        daily,
        meta,
        model_dir=model_dir,
        static_dir=static_dir,
        passes=ROOTZONE_PASSES,
    )
    daily = add_rootzone_profile(daily)

    daily.to_parquet(daily_path, index=False)
    print(f"Saved {daily_path}")

    n = len(daily)
    for col in _rootzone_cols(daily.columns):
        v = daily[col].notna().sum()
        print(f"  {col}: {v} valid ({100 * v / n:.1f}%)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Assemble flux site validation data")
    sub = p.add_subparsers(dest="command")

    # Full pipeline
    full = sub.add_parser("run", help="Full pipeline")
    full.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    full.add_argument("--study-start", default=STUDY_START)
    full.add_argument("--study-end", default=STUDY_END)
    full.add_argument("--model-dir", default=MODEL_DIR)
    full.add_argument("--static-dir", default=STATIC_DIR)

    # Add PTF to existing parquet
    ptf = sub.add_parser("add-ptf", help="Add PTF suction to existing parquet")
    ptf.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)

    # Add root-zone + profile suction to existing parquet
    rz = sub.add_parser(
        "add-rootzone",
        help="Add L4 root-zone + profile suction to existing parquet",
    )
    rz.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    rz.add_argument("--model-dir", default=MODEL_DIR)
    rz.add_argument("--static-dir", default=STATIC_DIR)

    args = p.parse_args()
    if args.command == "add-ptf":
        add_ptf_to_existing(args.output_dir)
    elif args.command == "add-rootzone":
        add_rootzone_to_existing(
            args.output_dir,
            model_dir=args.model_dir,
            static_dir=args.static_dir,
        )
    else:
        run(
            output_dir=args.output_dir,
            study_start=getattr(args, "study_start", STUDY_START),
            study_end=getattr(args, "study_end", STUDY_END),
            model_dir=getattr(args, "model_dir", MODEL_DIR),
            static_dir=getattr(args, "static_dir", STATIC_DIR),
        )
