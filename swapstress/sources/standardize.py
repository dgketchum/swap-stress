import argparse
import os

import numpy as np
import pandas as pd
from swapstress.sources.gshp import sanitize_profile_id
from swapstress.sources.ncss import ncss_to_standardized, load_ncss_parquet

from tqdm import tqdm

MPA_TO_CM = 10197.16

# Physical limits for data sanity filtering
# Suction: max ~1e6 cm (100 MPa) is beyond any realistic soil measurement
SUCTION_CM_MAX = 1e6
# Theta (VWC): must be in [0, 1] by definition
THETA_MIN = 0.0
THETA_MAX = 1.0
# KPA for MT Mesonet: 200 kPa (~2000 cm) is a reasonable upper bound for field sensors
KPA_MAX = 200.0
# Bulk density bounds for gravimetric->volumetric conversion (g/cm³)
BULK_DENSITY_MIN = 0.5
BULK_DENSITY_MAX = 2.5


def apply_physical_filters(
    df, suction_col="suction_cm", theta_col="theta", source_name=""
):
    """
    Apply physical sanity filters to remove non-physical values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with suction and theta columns.
    suction_col : str
        Name of suction column (in cm).
    theta_col : str
        Name of theta (VWC) column.
    source_name : str
        Name of data source for logging.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with summary printed.
    """
    n_initial = len(df)
    dropped_reasons = []

    # Filter suction: must be positive and <= max
    if suction_col in df.columns:
        mask_suction_neg = df[suction_col] <= 0
        mask_suction_high = df[suction_col] > SUCTION_CM_MAX
        n_neg = mask_suction_neg.sum()
        n_high = mask_suction_high.sum()
        if n_neg > 0:
            dropped_reasons.append(f"suction<=0: {n_neg}")
        if n_high > 0:
            dropped_reasons.append(f"suction>{SUCTION_CM_MAX:.0e}: {n_high}")
        df = df[~mask_suction_neg & ~mask_suction_high]

    # Filter theta: must be in [0, 1]
    if theta_col in df.columns:
        mask_theta_low = df[theta_col] < THETA_MIN
        mask_theta_high = df[theta_col] > THETA_MAX
        n_low = mask_theta_low.sum()
        n_high = mask_theta_high.sum()
        if n_low > 0:
            dropped_reasons.append(f"theta<0: {n_low}")
        if n_high > 0:
            dropped_reasons.append(f"theta>1: {n_high}")
        df = df[~mask_theta_low & ~mask_theta_high]

    n_final = len(df)
    n_dropped = n_initial - n_final
    if n_dropped > 0 and source_name:
        print(
            f"  [{source_name}] Dropped {n_dropped}/{n_initial} rows: {', '.join(dropped_reasons)}"
        )

    return df


def _standardize_depth(d, depth_col=None):
    if depth_col and depth_col in d.columns:
        d = d.rename(columns={depth_col: "depth"})
        return d
    for c in ("depth", "Depth [cm]", "Depth_cm", "stationDepth [cm]"):
        if c in d.columns:
            d = d.rename(columns={c: "depth"}) if c != "depth" else d
            return d
    d["depth"] = 0
    return d


def standardize_reesh(df, depth_col=None):
    d = df.copy()
    if "MPa_Abs" not in d.columns or "Vol_Water" not in d.columns:
        raise ValueError("Expected columns 'MPa_Abs' and 'Vol_Water'")
    # MPa -> cm of water; Vol_Water is percent -> fraction
    d["suction"] = np.abs(d["MPa_Abs"].astype(float).values) * MPA_TO_CM
    d["theta"] = d["Vol_Water"].astype(float).values / 100.0
    d = _standardize_depth(d, depth_col)
    # Prefer Sample_ID as primary identifier if present, otherwise fall back to Site
    if "Sample_ID" in d.columns:
        d["name"] = d["Sample_ID"]
    elif "Site" in d.columns:
        d["name"] = d["Site"]
    keep_extra = [c for c in ("Sample_ID", "Site", "Plot") if c in d.columns]
    d = d.rename(columns={"suction": "suction_cm", "depth": "depth_cm"})
    d = d[["suction_cm", "theta", "depth_cm"] + keep_extra]
    # Apply physical sanity filters
    d = apply_physical_filters(d, source_name="ReESH")
    return d


def standardize_mt_mesonet(df, depth_col=None):
    d = df.copy()
    n_initial = len(d)
    dropped_reasons = []

    if "KPA" in d.columns and "VWC" in d.columns:
        # Filter negative VWC before conversion
        mask_neg_vwc = d["VWC"].astype(float) < 0
        if mask_neg_vwc.sum() > 0:
            dropped_reasons.append(f"VWC<0: {mask_neg_vwc.sum()}")
            d = d[~mask_neg_vwc]

        # Filter extreme KPA values (>200 kPa is beyond field sensor range)
        mask_high_kpa = d["KPA"].astype(float).abs() > KPA_MAX
        if mask_high_kpa.sum() > 0:
            dropped_reasons.append(f"KPA>{KPA_MAX}: {mask_high_kpa.sum()}")
            d = d[~mask_high_kpa]

        d["suction"] = np.abs(d["KPA"].astype(float).values * 10.19716)
        d["theta"] = d["VWC"].astype(float).values
    elif "suction_cm" in d.columns and "theta" in d.columns:
        d["suction"] = np.abs(d["suction_cm"].astype(float).values)
        d["theta"] = d["theta"].astype(float).values
    else:
        raise ValueError("Expected ('KPA','VWC') or ('suction_cm','theta')")

    d = _standardize_depth(d, depth_col)
    if "name" not in d.columns and "station" in d.columns:
        d["name"] = d["station"]
    d = d.rename(columns={"suction": "suction_cm", "depth": "depth_cm"})
    d = d[["suction_cm", "theta", "depth_cm", "name"]]

    # Log source-specific drops (before physical filters)
    n_source_dropped = n_initial - len(d)
    if n_source_dropped > 0 and dropped_reasons:
        print(
            f"  [MT_Mesonet source-filter] Dropped {n_source_dropped}/{n_initial} rows: {', '.join(dropped_reasons)}"
        )

    # Apply physical sanity filters (logs its own drops)
    d = apply_physical_filters(d, source_name="MT_Mesonet")

    return d


def standardize_gshp(df, depth_col=None):
    d = df.copy()
    n_initial = len(d)
    dropped_reasons = []

    # Prefer GSHP high quality data
    if "data_flag" in d.columns:
        n_before = len(d)
        d = d[d["data_flag"] == "good quality estimate"]
        n_quality_filter = n_before - len(d)
        if n_quality_filter > 0:
            dropped_reasons.append(f"quality_filter: {n_quality_filter}")

    d = d.dropna(subset=["lab_head_m", "lab_wrc"])

    # Guard against extreme lab_head_m outliers (e.g., 1e19-1e31 m values in weynants_18)
    # Max realistic: 1e4 m = 1e6 cm = 100 MPa
    LAB_HEAD_M_MAX = 1e4
    mask_extreme_head = d["lab_head_m"].astype(float).abs() > LAB_HEAD_M_MAX
    if mask_extreme_head.sum() > 0:
        dropped_reasons.append(
            f"lab_head_m>{LAB_HEAD_M_MAX:.0e}: {mask_extreme_head.sum()}"
        )
        d = d[~mask_extreme_head]

    d["suction"] = (d["lab_head_m"].astype(float) * 100.0).abs()  # m -> cm
    d["theta"] = d["lab_wrc"].astype(float)
    if "hzn_bot" in d.columns and "hzn_top" in d.columns:
        d["depth"] = (d["hzn_bot"].astype(float) + d["hzn_top"].astype(float)) / 2.0
    else:
        d = _standardize_depth(d, depth_col)
    d = d.rename(columns={"suction": "suction_cm", "depth": "depth_cm"})
    keep = ["suction_cm", "theta", "depth_cm"]
    # Carried through for downstream analysis, not for fitting: we no longer
    # refit GSHP curves. Texture and bulk density are excluded from the feature
    # set (they are lab measurements, unavailable at inference) but are used to
    # bin diagnostics; SWCC_classes identifies the YWYD subset whose theta_r and
    # theta_s were fit freely, which is the only subset safe to compare against
    # a texture-based PTF. See swapstress.sources.gshp.
    keep += [
        c
        for c in (
            "profile_id",
            "SWCC_classes",
            "sand_tot_psa",
            "silt_tot_psa",
            "clay_tot_psa",
            "db_od",
            "climate_classes",
        )
        if c in d.columns
    ]
    d = d[keep]

    # Log source-specific drops (before physical filters)
    n_source_dropped = n_initial - len(d)
    if n_source_dropped > 0 and dropped_reasons:
        print(
            f"  [GSHP source-filter] Dropped {n_source_dropped}/{n_initial} rows: {', '.join(dropped_reasons)}"
        )

    # Apply physical sanity filters (logs its own drops)
    d = apply_physical_filters(d, source_name="GSHP")

    return d


def write_standardized_gshp(soil_csv_path, out_dir, minimum_points):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(soil_csv_path, encoding="latin1")
    if "profile_id" in df.columns:
        df["profile_id"] = df["profile_id"].astype(str).apply(sanitize_profile_id)
    std = standardize_gshp(df)
    stations = 0
    s_min, s_max = np.inf, -np.inf
    t_min, t_max = np.inf, -np.inf
    print(f"writing gshp obs to {out_dir}")
    for pid, r in tqdm(std.groupby("profile_id"), total=len(std.groupby("profile_id"))):
        depth_counts = r.groupby("depth_cm").size()
        keep_depths = depth_counts[depth_counts >= minimum_points].index
        r = r[r["depth_cm"].isin(keep_depths)]
        if r.empty:
            continue
        out_path = os.path.join(out_dir, f"{pid}.csv")
        r[
            [
                "suction_cm",
                "theta",
                "depth_cm",
                "SWCC_classes",
                "sand_tot_psa",
                "silt_tot_psa",
                "clay_tot_psa",
                "db_od",
                "climate_classes",
            ]
        ].to_csv(out_path, index=False)
        stations += 1
        s_min = min(s_min, float(np.nanmin(r["suction_cm"].values)))
        s_max = max(s_max, float(np.nanmax(r["suction_cm"].values)))
        t_min = min(t_min, float(np.nanmin(r["theta"].values)))
        t_max = max(t_max, float(np.nanmax(r["theta"].values)))

    print(
        f"GSHP standardized: stations={stations}, suction_cm=[{s_min:.3g}, {s_max:.3g}], theta=[{t_min:.3f}, {t_max:.3f}]"
    )


def write_standardized_rosetta(curves_wide_csv, out_dir, profile_key):
    os.makedirs(out_dir, exist_ok=True)
    dfw = pd.read_csv(curves_wide_csv)
    if "Index" not in dfw.columns:
        return
    s_min, s_max = np.inf, -np.inf
    t_min, t_max = np.inf, -np.inf
    stations = 0
    for idx, row_df in tqdm(dfw.groupby("Index"), total=dfw["Index"].nunique()):
        cols = row_df.columns[2:]
        h_cols = cols[0::2]
        t_cols = cols[1::2]
        r = row_df.iloc[0]
        recs = []
        for hc, tc in zip(h_cols, t_cols):
            h = r[hc]
            t = r[tc]
            if pd.notna(h) and pd.notna(t):
                recs.append(
                    {
                        "suction_cm": abs(float(h)),
                        "theta": float(t),
                        "depth_cm": 0,
                        "Index": int(idx),
                    }
                )
        d = pd.DataFrame(recs)
        d["profile_id"] = d[profile_key]
        out_path = os.path.join(out_dir, f"{int(idx)}.csv")
        d.to_csv(out_path, index=False)
        if not d.empty:
            stations += 1
            s_min = min(s_min, float(np.nanmin(d["suction_cm"].values)))
            s_max = max(s_max, float(np.nanmax(d["suction_cm"].values)))
            t_min = min(t_min, float(np.nanmin(d["theta"].values)))
            t_max = max(t_max, float(np.nanmax(d["theta"].values)))
    if stations:
        print(
            f"Rosetta standardized: stations={stations}, suction_cm=[{s_min:.3g}, {s_max:.3g}], theta=[{t_min:.3f}, {t_max:.3f}]"
        )


def standardize_lacadian(df, depth_col=None):
    d = df.copy()
    n_initial = len(d)
    dropped_reasons = []

    if "KPA" in d.columns and "VWC" in d.columns:
        mask_neg_vwc = d["VWC"].astype(float) < 0
        if mask_neg_vwc.sum() > 0:
            dropped_reasons.append(f"VWC<0: {mask_neg_vwc.sum()}")
            d = d[~mask_neg_vwc]

        mask_high_kpa = d["KPA"].astype(float).abs() > KPA_MAX
        if mask_high_kpa.sum() > 0:
            dropped_reasons.append(f"KPA>{KPA_MAX}: {mask_high_kpa.sum()}")
            d = d[~mask_high_kpa]

        # -0.1 kPa is the sensor floor (lower detection limit); drop clamped readings
        mask_floor = d["KPA"].astype(float) >= -0.15
        if mask_floor.sum() > 0:
            dropped_reasons.append(f"KPA_floor(-0.1kPa): {mask_floor.sum()}")
            d = d[~mask_floor]

        d["suction"] = np.abs(d["KPA"].astype(float).values * 10.19716)
        d["theta"] = d["VWC"].astype(float).values
    elif "suction_cm" in d.columns and "theta" in d.columns:
        d["suction"] = np.abs(d["suction_cm"].astype(float).values)
        d["theta"] = d["theta"].astype(float).values
    else:
        raise ValueError("Expected ('KPA','VWC') or ('suction_cm','theta')")

    d = _standardize_depth(d, depth_col)
    if "name" not in d.columns and "station" in d.columns:
        d["name"] = d["station"]
    d = d.rename(columns={"suction": "suction_cm", "depth": "depth_cm"})
    d = d[["suction_cm", "theta", "depth_cm", "name"]]

    n_source_dropped = n_initial - len(d)
    if n_source_dropped > 0 and dropped_reasons:
        print(
            f"  [LaCADIAN source-filter] Dropped {n_source_dropped}/{n_initial} rows: {', '.join(dropped_reasons)}"
        )

    d = apply_physical_filters(d, source_name="LaCADIAN")
    return d


def write_standardized_lacadian(swp_csv_path, metadata_csv_path, out_dir, profile_key):
    os.makedirs(out_dir, exist_ok=True)
    for p in [swp_csv_path, metadata_csv_path]:
        if not os.path.exists(p):
            print(f"Error: Source file not found at {p}")
            return
    obs_df = pd.read_csv(swp_csv_path)
    meta_df = pd.read_csv(metadata_csv_path)
    station_col = "station"
    if station_col not in obs_df.columns or station_col not in meta_df.columns:
        print(f"Error: Join column '{station_col}' not found in one or both files.")
        return
    merged = pd.merge(obs_df, meta_df, on=station_col, how="left")
    s_min, s_max = np.inf, -np.inf
    t_min, t_max = np.inf, -np.inf
    stations = 0
    for profile_id, r in tqdm(
        merged.groupby(profile_key), total=merged[station_col].nunique()
    ):
        d = standardize_lacadian(r, depth_col="depth_cm")
        d["profile_id"] = profile_id
        d["station"] = profile_id
        out_path = os.path.join(out_dir, f"{profile_id}.csv")
        d.to_csv(out_path, index=False)
        stations += 1
        s_min = min(s_min, float(np.nanmin(d["suction_cm"].values)))
        s_max = max(s_max, float(np.nanmax(d["suction_cm"].values)))
        t_min = min(t_min, float(np.nanmin(d["theta"].values)))
        t_max = max(t_max, float(np.nanmax(d["theta"].values)))
    if stations:
        print(
            f"LaCADIAN standardized: stations={stations}, suction_cm=[{s_min:.3g}, {s_max:.3g}], theta=[{t_min:.3f}, {t_max:.3f}]"
        )


def write_standardized_mt_mesonet(
    swp_csv_path, metadata_csv_path, out_dir, profile_key
):
    os.makedirs(out_dir, exist_ok=True)
    for p in [swp_csv_path, metadata_csv_path]:
        if not os.path.exists(p):
            print(f"Error: Source file not found at {p}")
            return
    obs_df = pd.read_csv(swp_csv_path)
    meta_df = pd.read_csv(metadata_csv_path)
    station_col = "station"
    if station_col not in obs_df.columns or station_col not in meta_df.columns:
        print(f"Error: Join column '{station_col}' not found in one or both files.")
        return
    merged = pd.merge(obs_df, meta_df, on=station_col, how="left")
    s_min, s_max = np.inf, -np.inf
    t_min, t_max = np.inf, -np.inf
    stations = 0
    for profile_id, r in tqdm(
        merged.groupby(profile_key), total=merged["station"].nunique()
    ):
        d = standardize_mt_mesonet(r, depth_col="Depth [cm]")
        d["profile_id"] = profile_id
        d["station"] = profile_id
        out_path = os.path.join(out_dir, f"{profile_id}.csv")
        d.to_csv(out_path, index=False)
        stations += 1
        s_min = min(s_min, float(np.nanmin(d["suction_cm"].values)))
        s_max = max(s_max, float(np.nanmax(d["suction_cm"].values)))
        t_min = min(t_min, float(np.nanmin(d["theta"].values)))
        t_max = max(t_max, float(np.nanmax(d["theta"].values)))
    if stations:
        print(
            f"MT Mesonet standardized: stations={stations}, suction_cm=[{s_min:.3g}, {s_max:.3g}], theta=[{t_min:.3f}, {t_max:.3f}]"
        )


def write_standardized_reesh(in_dir, out_dir, profile_key):
    os.makedirs(out_dir, exist_ok=True)
    s_min, s_max = np.inf, -np.inf
    t_min, t_max = np.inf, -np.inf
    stations = 0
    files = [
        os.path.join(in_dir, f)
        for f in os.listdir(in_dir)
        if "_SoilWaterRetentionCurves.csv" in f
    ]
    for f in files:
        if not f.endswith(".csv"):
            continue
        p = os.path.join(in_dir, f)
        df = pd.read_csv(p)

        station = df.iloc[0]["Site"]

        if "Sample_ID" not in df.columns:
            continue

        if df["Site"].nunique() > 1:
            raise ValueError

        for profile_id, r in tqdm(df.groupby(profile_key), total=df["Plot"].nunique()):
            d = standardize_reesh(r, depth_col="Depth_cm")
            d["profile_id"] = profile_id
            d["station"] = station
            out_path = os.path.join(out_dir, f"{station}_{profile_id}.csv")
            d.to_csv(out_path, index=False)
            stations += 1
            s_min = min(s_min, float(np.nanmin(d["suction_cm"].values)))
            s_max = max(s_max, float(np.nanmax(d["suction_cm"].values)))
            t_min = min(t_min, float(np.nanmin(d["theta"].values)))
            t_max = max(t_max, float(np.nanmax(d["theta"].values)))

    if stations:
        print(
            f"ReESH standardized: stations={stations}, suction_cm=[{s_min:.3g}, {s_max:.3g}], theta=[{t_min:.3f}, {t_max:.3f}]"
        )


def write_standardized_ncss(parquet_path, out_dir, minimum_points):
    os.makedirs(out_dir, exist_ok=True)
    df = load_ncss_parquet(parquet_path)
    std = ncss_to_standardized(df)
    if "profile_id" in std.columns:
        std["profile_id"] = std["profile_id"].astype(str).apply(sanitize_profile_id)

    s_min, s_max = np.inf, -np.inf
    t_min, t_max = np.inf, -np.inf
    stations = 0
    print(f"writing ncss obs to {out_dir}")
    for pid, r in tqdm(
        std.groupby("profile_id"),
        total=len(std.groupby("profile_id")),
        desc="Processing NCSS data",
    ):
        out_path = os.path.join(out_dir, f"{pid}.csv")
        cols = ["suction_cm", "theta", "depth_cm"]
        extras = [
            c
            for c in (
                "SWCC_classes",
                "sand_tot_psa",
                "silt_tot_psa",
                "clay_tot_psa",
                "db_od",
                "source_db",
            )
            if c in r.columns
        ]
        depth_counts = r.groupby("depth_cm").size()
        keep_depths = depth_counts[depth_counts >= minimum_points].index
        r = r[r["depth_cm"].isin(keep_depths)]
        if r.empty:
            continue
        out_df = r[cols + extras]
        out_df.to_csv(out_path, index=False)
        stations += 1
        s_min = min(s_min, float(np.nanmin(r["suction_cm"].values)))
        s_max = max(s_max, float(np.nanmax(r["suction_cm"].values)))
        t_min = min(t_min, float(np.nanmin(r["theta"].values)))
        t_max = max(t_max, float(np.nanmax(r["theta"].values)))
    if stations:
        print(
            f"NCSS standardized: stations={stations}, suction_cm=[{s_min:.3g}, {s_max:.3g}], theta=[{t_min:.3f}, {t_max:.3f}]"
        )


# Stage 00: swapstress-standardize
#
# Each source arrives in its own shape -- one wide CSV, a station table plus an
# observation table, a directory of per-plot files -- so the writers below do not
# share a signature. _RECIPES names the inputs each one wants; every path comes
# from the registry, so adding a source is a registry edit plus one entry here.

_RECIPES = {
    "gshp": lambda paths, cfg: write_standardized_gshp(
        paths.raw_file("curves"),
        paths.preprocessed_dir,
        minimum_points=cfg["minimum_points"],
    ),
    "ncss": lambda paths, cfg: write_standardized_ncss(
        paths.raw_file("curves"),
        paths.preprocessed_dir,
        minimum_points=cfg["minimum_points"],
    ),
    "mt_mesonet": lambda paths, cfg: write_standardized_mt_mesonet(
        paths.raw_file("swp"),
        paths.raw_file("metadata"),
        paths.preprocessed_dir,
        profile_key="station",
    ),
    "lacadian": lambda paths, cfg: write_standardized_lacadian(
        paths.raw_file("swp"),
        paths.raw_file("metadata"),
        paths.preprocessed_dir,
        profile_key="station",
    ),
    "reesh": lambda paths, cfg: write_standardized_reesh(
        paths.raw_dir,
        paths.preprocessed_dir,
        profile_key="Plot",
    ),
    "rosetta": lambda paths, cfg: write_standardized_rosetta(
        paths.raw_file("curves"),
        paths.preprocessed_dir,
        profile_key="Index",
    ),
}

# What each recipe reads, for the dry run. reesh takes a directory, not files.
_RECIPE_INPUTS = {
    "gshp": ["curves"],
    "ncss": ["curves"],
    "mt_mesonet": ["swp", "metadata"],
    "lacadian": ["swp", "metadata"],
    "reesh": [],
    "rosetta": ["curves"],
}


def build_parser():
    from swapstress.cli import add_common_args
    from swapstress.sources.registry import DEFAULT_SOURCES

    parser = argparse.ArgumentParser(
        prog="swapstress-standardize",
        description="Stage 00: harmonize raw source observations to "
        "(theta, suction_cm, depth_cm).",
    )
    add_common_args(parser)
    parser.add_argument(
        "--sources",
        type=str,
        nargs="+",
        default=None,
        choices=sorted(_RECIPES),
        help=f"Sources to standardize (default: {' '.join(DEFAULT_SOURCES)}).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root data directory (default: /nas/soils).",
    )
    parser.add_argument(
        "--minimum-points",
        type=int,
        default=None,
        help="Minimum retention points per depth to keep a curve (default: 4). "
        "Applies to the lab sources, gshp and ncss.",
    )
    return parser


def main(argv=None):
    from swapstress.cli import report_paths, resolve, stage_provenance
    from swapstress.sources.registry import DEFAULT_SOURCES, DataPaths, get_source

    config = resolve(build_parser(), argv)
    config.setdefault("sources", DEFAULT_SOURCES)
    config.setdefault("data_root", "/nas/soils")
    config.setdefault("minimum_points", 4)

    for name in config["sources"]:
        paths = DataPaths(config["data_root"], get_source(name))
        inputs = {role: paths.raw_file(role) for role in _RECIPE_INPUTS[name]}
        if not inputs:
            inputs = {"directory": paths.raw_dir}

        if config["dry_run"]:
            report_paths(
                f"00 standardize [{name}]",
                inputs,
                {"standardized": paths.preprocessed_dir},
            )
            continue

        print(f"\n=== Standardizing {name} ===")
        _RECIPES[name](paths, config)
        stage_provenance(
            paths.preprocessed_dir,
            config,
            run_type="standardize",
            extras={"source": name, "inputs": inputs},
        )


if __name__ == "__main__":
    main()

# ========================= EOF ====================================================================
