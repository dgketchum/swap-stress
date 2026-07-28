"""
Reshape raw LaCADIAN daily .txt files into SWAP-Stress pipeline format.

LaCADIAN files are tab-separated, no header, with 39 columns.
Column layout (0-indexed):
    0: date (YYYY-MM-DD, UTC)
    1: longitude
    2: latitude
    3: precipitation (mm, daily total)
    4-6: land_surface_temp (mean/min/max)
    7-9: air_temp (mean/min/max)
    10-12: relative_humidity (mean/min/max)
    13: atmospheric_pressure (kPa)
    14: vapor_pressure (kPa)
    15: solar_radiation (W/m2)
    16: wind_speed (m/s)
    17: gust_speed (m/s)
    18: EC (mS/cm)
    19: NDVI
    20-24: soil_moisture at 5/10/20/50/100 cm (m3/m3)
    25-27: matric_potential at 5/20/50 cm (kPa)
    28-32: saturation_extract_EC at 5/10/20/50/100 cm (mS/cm)
    33-37: soil_temperature at 5/10/20/50/100 cm (degC)
    38: NDVI_DAILY

Outputs:
    - Per-station parquet files (long-format timeseries)
    - station_metadata.csv (station, latitude, longitude, date_installed)
    - swp.csv (paired SWP/VWC observations for retention curve fitting)

Usage:
    python -m swapstress.sources.lacadian
"""

import os
from glob import glob

import pandas as pd


# Column indices (0-indexed) for soil variables — single mean value per variable
# VWC columns (m3/m3)
VWC_COLS = {5: 20, 10: 21, 20: 22, 50: 23, 100: 24}
# Matric potential columns (kPa) — only at 5, 20, 50 cm
MP_COLS = {5: 25, 20: 26, 50: 27}
# Coordinate columns: longitude=1, latitude=2
COL_LON = 1
COL_LAT = 2

# Depths where both SWP and VWC sensors are collocated
PAIRED_DEPTHS = [5, 20, 50]


def _station_id_from_path(fp):
    """Extract station ID (e.g., 'LAC001') from filename."""
    basename = os.path.splitext(os.path.basename(fp))[0]
    # Files are like LAC001_Ben_Hur_DAILY_2025
    return basename.split("_")[0]


def parse_daily_txt(txt_path):
    """
    Parse a LaCADIAN daily .txt file into long-format rows.

    Returns
    -------
    pd.DataFrame
        Columns: station, date, latitude, longitude, depth_cm, VWC, KPA
    """
    try:
        df = pd.read_csv(txt_path, sep="\t", header=None, na_values=["NA", ""])
    except Exception as e:
        print(f"Warning: failed to read {txt_path}: {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    station = _station_id_from_path(txt_path)
    lat = df.iloc[0, COL_LAT]
    lon = df.iloc[0, COL_LON]

    records = []
    for _, row in df.iterrows():
        date = row[0]

        # Extract VWC and matric potential at each depth
        for depth in sorted(set(list(VWC_COLS.keys()) + list(MP_COLS.keys()))):
            vwc = None
            kpa = None

            if depth in VWC_COLS:
                col_idx = VWC_COLS[depth]
                if col_idx < len(row):
                    v = row[col_idx]
                    if pd.notna(v):
                        vwc = float(v)

            if depth in MP_COLS:
                col_idx = MP_COLS[depth]
                if col_idx < len(row):
                    v = row[col_idx]
                    if pd.notna(v):
                        kpa = float(v)

            # Only emit row if we have at least one measurement
            if vwc is not None or kpa is not None:
                records.append(
                    {
                        "station": station,
                        "date": date,
                        "latitude": lat,
                        "longitude": lon,
                        "depth_cm": depth,
                        "VWC": vwc,
                        "KPA": kpa,
                    }
                )

    return pd.DataFrame(records)


def process_all_sites(raw_dir, timeseries_dir, metadata_csv, swp_csv):
    """
    Process all raw LaCADIAN daily .txt files.

    Parameters
    ----------
    raw_dir : str
        Directory containing raw .txt files.
    timeseries_dir : str
        Output directory for per-station parquet files.
    metadata_csv : str
        Output path for station_metadata.csv.
    swp_csv : str
        Output path for paired SWP observations (swp.csv).
    """
    os.makedirs(timeseries_dir, exist_ok=True)
    os.makedirs(os.path.dirname(metadata_csv), exist_ok=True)

    txt_files = sorted(glob(os.path.join(raw_dir, "*.txt")))
    if not txt_files:
        print(f"No .txt files found in {raw_dir}")
        return

    # Group files by station (multiple years per station)
    station_files = {}
    for fp in txt_files:
        sid = _station_id_from_path(fp)
        station_files.setdefault(sid, []).append(fp)

    all_long = []
    metadata_rows = []

    for station, files in sorted(station_files.items()):
        frames = []
        for fp in sorted(files):
            df = parse_daily_txt(fp)
            if not df.empty:
                frames.append(df)

        if not frames:
            print(f"  {station}: no data")
            continue

        long_df = pd.concat(frames, ignore_index=True)

        # Deduplicate (same date from overlapping files)
        long_df = long_df.drop_duplicates(subset=["station", "date", "depth_cm"])

        all_long.append(long_df)

        # Write per-station parquet
        out_pq = os.path.join(timeseries_dir, f"{station}.parquet")
        long_df.to_parquet(out_pq, index=False)
        n_vwc = long_df["VWC"].notna().sum()
        n_kpa = long_df["KPA"].notna().sum()
        print(f"  {station}: {len(long_df)} rows, {n_vwc} VWC, {n_kpa} KPA obs")

        # Collect metadata
        earliest = pd.to_datetime(long_df["date"], errors="coerce").min()
        metadata_rows.append(
            {
                "station": station,
                "latitude": long_df["latitude"].iloc[0],
                "longitude": long_df["longitude"].iloc[0],
                "date_installed": earliest.strftime("%Y-%m-%d")
                if pd.notna(earliest)
                else "",
            }
        )

    # Write station metadata
    meta_df = pd.DataFrame(metadata_rows)
    meta_df.to_csv(metadata_csv, index=False)
    print(f"Wrote metadata: {metadata_csv} ({len(meta_df)} stations)")

    # Build paired observations (only at collocated depths)
    if all_long:
        combined = pd.concat(all_long, ignore_index=True)
        paired = combined[combined["depth_cm"].isin(PAIRED_DEPTHS)].copy()
        paired = paired.dropna(subset=["VWC", "KPA"])
        paired = paired[["station", "KPA", "VWC", "depth_cm"]]
        paired.to_csv(swp_csv, index=False)
        n_stations = paired["station"].nunique()
        print(
            f"Wrote paired SWP: {swp_csv} ({len(paired)} observations, "
            f"{n_stations} stations)"
        )

        # Summary stats
        print(f"  VWC range: {paired['VWC'].min():.3f} - {paired['VWC'].max():.3f}")
        print(f"  KPA range: {paired['KPA'].min():.2f} - {paired['KPA'].max():.2f}")
        for d in PAIRED_DEPTHS:
            n = len(paired[paired["depth_cm"] == d])
            print(f"  depth {d}cm: {n} paired obs")


if __name__ == "__main__":
    root_ = os.path.join("/nas", "soils", "soil_potential_obs", "lacadian")
    raw_dir_ = os.path.join(root_, "raw")
    timeseries_dir_ = os.path.join(root_, "timeseries")
    metadata_csv_ = os.path.join(root_, "station_metadata.csv")
    swp_csv_ = os.path.join(root_, "swp.csv")

    process_all_sites(raw_dir_, timeseries_dir_, metadata_csv_, swp_csv_)

# ========================= EOF ====================================================================
