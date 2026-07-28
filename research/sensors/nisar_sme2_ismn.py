"""
Evaluate NISAR L3 SME2 (200 m) soil moisture against ISMN in-situ stations.

Streams HDF5 files remotely via earthaccess (no bulk download), extracts
surface soil moisture at ISMN station locations using EASE-Grid 2.0 200 m
indices, merges with ISMN daily VWC at 5 cm depth, and computes per-station
accuracy metrics for comparison with SMAP L3.

Usage:
    # Full pipeline: search → extract → score → compare
    python -m research.sensors.nisar_sme2_ismn full \
        --l3-csv /nas/soils/vwc_timeseries/ismn/smap_ismn_5cm_comparison.csv \
        --ismn-dir /nas/soils/vwc_timeseries/ismn/processed_time_series/preprocessed_by_station \
        --output /nas/soils/vwc_timeseries/ismn/nisar_sme2_ismn_5cm_comparison.csv

    # Compare only (after extraction)
    python -m research.sensors.nisar_sme2_ismn compare \
        --l3-csv /nas/soils/vwc_timeseries/ismn/smap_ismn_5cm_comparison.csv \
        --nisar-csv /nas/soils/vwc_timeseries/ismn/nisar_sme2_ismn_5cm_comparison.csv
"""

import argparse
import os
import re
import signal

import earthaccess
import h5py
import numpy as np
import pandas as pd
from pyproj import Transformer


class _Timeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _Timeout("Remote read timed out")


# HDF5 paths within NISAR L3 SME2 files
_GRIDS = "science/LSAR/SME2/grids"
_SM_FIELD = f"{_GRIDS}/soilMoisture"
_ROW_IDX = f"{_GRIDS}/EASEGridRowIndex"
_COL_IDX = f"{_GRIDS}/EASEGridColumnIndex"
_QF_FIELD = f"{_GRIDS}/retrievalQualityFlag"
_SM_FILL = -9999.0

# EASE-Grid 2.0 200 m grid parameters
_DX = 200.17900466918945
_X_ORIGIN = -17367430.39
_Y_ORIGIN = 7314440.74

_transformer = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)


def _lonlat_to_ease2_200m(lon: float, lat: float) -> tuple[int, int]:
    """Convert lon/lat to global EASE-Grid 2.0 200 m (row, col)."""
    x, y = _transformer.transform(lon, lat)
    col = round((x - _X_ORIGIN) / _DX)
    row = round((_Y_ORIGIN - y) / _DX)
    return row, col


def precompute_station_coords(l3_csv: str) -> dict:
    """Convert ISMN station locations to EASE-2 200 m grid indices.

    Returns {station_uid: (ease_row_200m, ease_col_200m)}.
    """
    df = pd.read_csv(l3_csv)
    coords = {}
    for _, r in df.iterrows():
        row, col = _lonlat_to_ease2_200m(r["lon"], r["lat"])
        coords[r["station"]] = (row, col)
    return coords


def _parse_date(native_id: str) -> str | None:
    """Extract YYYYMMDD from NISAR granule native ID."""
    m = re.search(r"_(\d{8})T\d{6}_", native_id)
    return m.group(1) if m else None


def extract_from_granule(fobj, station_coords: dict, native_id: str) -> list:
    """Extract soil moisture at ISMN stations from one remote NISAR HDF5.

    Parameters
    ----------
    fobj : file-like
        From earthaccess.open().
    station_coords : dict
        {station_uid: (ease_row_200m, ease_col_200m)}.
    native_id : str
        Granule native ID for date parsing.

    Returns
    -------
    list of dict
        Each dict has keys: station, date, nisar_sm.
    """
    date_str = _parse_date(native_id)
    if date_str is None:
        return []

    try:
        with h5py.File(fobj, "r") as f:
            ease_rows = f[_ROW_IDX][:]
            ease_cols = f[_COL_IDX][:]
            row_min, row_max = int(ease_rows.min()), int(ease_rows.max())
            col_min, col_max = int(ease_cols.min()), int(ease_cols.max())

            # Check which stations fall within this frame
            hits = []
            for station, (sr, sc) in station_coords.items():
                if row_min <= sr <= row_max and col_min <= sc <= col_max:
                    local_r = sr - row_min
                    local_c = sc - col_min
                    hits.append((station, local_r, local_c))

            if not hits:
                return []

            sm = f[_SM_FIELD][:]

            results = []
            for station, lr, lc in hits:
                if 0 <= lr < sm.shape[0] and 0 <= lc < sm.shape[1]:
                    val = float(sm[lr, lc])
                    if val != _SM_FILL and 0 <= val <= 1.0:
                        results.append(
                            {
                                "station": station,
                                "date": date_str,
                                "nisar_sm": val,
                            }
                        )
            return results
    except Exception as e:
        print(f"  Error reading {native_id}: {e}")
        return []


def extract_all_granules(
    station_coords: dict,
    output_path: str,
    bbox: tuple = (-125, 25, -66, 50),
    max_granules: int = 10000,
) -> pd.DataFrame:
    """Extract NISAR SME2 at all ISMN stations via remote streaming.

    Parameters
    ----------
    station_coords : dict
        From precompute_station_coords.
    output_path : str
        Where to write the intermediate parquet.
    bbox : tuple
        (west, south, east, north) for earthaccess search.
    max_granules : int
        Maximum granules to search.

    Returns
    -------
    pd.DataFrame
        Columns: station, date, nisar_sm
    """
    earthaccess.login()
    results = earthaccess.search_data(
        short_name="NISAR_L3_SME2_BETA_V1",
        bounding_box=bbox,
        count=max_granules,
    )
    print(f"Found {len(results)} granules")

    all_records = []
    n_opened = 0
    n_failed = 0
    n_timeout = 0
    read_timeout = 60  # seconds per granule

    # Resume from existing parquet if present
    if os.path.exists(output_path):
        prev = pd.read_parquet(output_path)
        all_records = prev.to_dict("records")
        # Track which (station, date) pairs we already have
        seen = set()
        for rec in all_records:
            d = rec["date"]
            if isinstance(d, pd.Timestamp):
                d = d.strftime("%Y%m%d")
            seen.add((rec["station"], d))
        print(f"Resuming: loaded {len(all_records)} existing extractions")
    else:
        seen = set()

    for i, granule in enumerate(results):
        native_id = granule["meta"]["native-id"]

        # Set alarm for read timeout
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(read_timeout)
        try:
            fobjs = earthaccess.open([granule])
            fobj = fobjs[0]
            n_opened += 1
            recs = extract_from_granule(fobj, station_coords, native_id)
            signal.alarm(0)
        except _Timeout:
            n_timeout += 1
            if n_timeout <= 10:
                print(f"  Timeout ({n_timeout}): {native_id}")
            continue
        except Exception as e:
            signal.alarm(0)
            n_failed += 1
            if n_failed <= 10:
                print(f"  Open failed ({n_failed}): {e}")
            continue

        # Deduplicate against existing records
        for rec in recs:
            key = (rec["station"], rec["date"])
            if key not in seen:
                all_records.append(rec)
                seen.add(key)

        if (i + 1) % 50 == 0:
            print(
                f"  {i + 1}/{len(results)} granules, "
                f"{len(all_records)} extractions, "
                f"{n_failed} failures, {n_timeout} timeouts"
            )

        # Incremental save every 200 granules
        if (i + 1) % 200 == 0 and all_records:
            _save_parquet(all_records, output_path)

    signal.alarm(0)
    print(
        f"\nTotal: {n_opened} granules read, {n_failed} failures, {n_timeout} timeouts"
    )
    print(f"Total extractions: {len(all_records)}")

    if not all_records:
        empty = pd.DataFrame(columns=["station", "date", "nisar_sm"])
        empty.to_parquet(output_path, index=False)
        return empty

    return _save_parquet(all_records, output_path)


def _save_parquet(records: list, output_path: str) -> pd.DataFrame:
    """Aggregate and save extraction records to parquet."""
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")

    agg = (
        df.groupby(["station", "date"])
        .agg(nisar_sm=("nisar_sm", "mean"), n_obs=("nisar_sm", "count"))
        .reset_index()
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    agg.to_parquet(output_path, index=False)
    print(f"  Saved {len(agg)} station-day extractions to {output_path}")
    return agg


def merge_and_score(
    extractions_path: str,
    ismn_station_dir: str,
    output_csv: str,
    l3_csv: str,
    min_pairs: int = 10,
) -> pd.DataFrame:
    """Merge NISAR extractions with ISMN VWC and compute per-station metrics."""
    ext = pd.read_parquet(extractions_path)
    ext["date"] = pd.to_datetime(ext["date"])

    l3_df = pd.read_csv(l3_csv)
    station_meta = l3_df.set_index("station")[["lat", "lon"]].to_dict("index")

    records = []
    n_skip = {"nofile": 0, "nocol": 0, "fewpairs": 0}

    for station in ext["station"].unique():
        meta = station_meta.get(station)
        if meta is None:
            continue

        safe_station = station.replace(":", "_").replace("/", "_").replace(" ", "_")
        pq_path = os.path.join(ismn_station_dir, f"{safe_station}.parquet")
        if not os.path.exists(pq_path):
            n_skip["nofile"] += 1
            continue

        ismn = pd.read_parquet(pq_path)
        if "soil_vwc_5" not in ismn.columns:
            n_skip["nocol"] += 1
            continue

        ismn["datetime"] = pd.to_datetime(ismn["datetime"])
        ismn = ismn[["datetime", "soil_vwc_5"]].dropna().copy()
        ismn["date"] = ismn["datetime"].dt.normalize()
        daily_ismn = ismn.groupby("date")["soil_vwc_5"].mean()

        st_ext = ext[ext["station"] == station].set_index("date")["nisar_sm"]
        paired = pd.DataFrame({"insitu": daily_ismn, "smap": st_ext}).dropna()

        if len(paired) < min_pairs:
            n_skip["fewpairs"] += 1
            continue

        diff = paired["smap"] - paired["insitu"]
        bias = diff.mean()
        rmse = np.sqrt((diff**2).mean())
        ubrmse = np.sqrt(((diff - bias) ** 2).mean())
        r = paired["smap"].corr(paired["insitu"])

        records.append(
            {
                "station": station,
                "lat": meta["lat"],
                "lon": meta["lon"],
                "n_paired": len(paired),
                "bias": bias,
                "rmse": rmse,
                "ubrmse": ubrmse,
                "r": r,
                "insitu_mean": paired["insitu"].mean(),
                "nisar_mean": paired["smap"].mean(),
            }
        )

    print(f"Stations scored: {len(records)}")
    print(f"  Skipped (no parquet): {n_skip['nofile']}")
    print(f"  Skipped (no soil_vwc_5): {n_skip['nocol']}")
    print(f"  Skipped (<{min_pairs} pairs): {n_skip['fewpairs']}")

    result = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    result.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")
    return result


def compare_products(l3_csv: str, nisar_csv: str) -> None:
    """Print head-to-head comparison of SMAP L3 vs NISAR SME2 at ISMN stations."""
    l3 = pd.read_csv(l3_csv)
    nisar = pd.read_csv(nisar_csv)

    print(f"\n{'=' * 70}")
    print("NISAR L3 SME2 (200 m) vs SMAP L3 (9 km) — ISMN 5 cm Comparison")
    print(f"{'=' * 70}\n")

    print(f"{'Metric':<25s} {'SMAP L3':>12s} {'NISAR SME2':>12s}")
    print("-" * 50)

    for metric in ["rmse", "ubrmse", "r", "bias"]:
        l3_med = l3[metric].median()
        n_med = nisar[metric].median()
        print(f"Median {metric:<18s} {l3_med:>12.4f} {n_med:>12.4f}")

    print(f"{'Stations scored':<25s} {len(l3):>12d} {len(nisar):>12d}")
    if "n_paired" in nisar.columns:
        print(
            f"{'Median paired obs':<25s} {'~800':>12s} {nisar['n_paired'].median():>12.0f}"
        )

    # Paired station comparison
    merged = l3.merge(nisar, on="station", suffixes=("_l3", "_nisar"))
    print(f"\nStations in both: {len(merged)}")

    if len(merged) > 0:
        d_ub = merged["ubrmse_l3"] - merged["ubrmse_nisar"]
        d_r = merged["r_nisar"] - merged["r_l3"]
        print(
            f"  ubRMSE: NISAR better at {(d_ub > 0).sum()}/{len(merged)} "
            f"({100 * (d_ub > 0).mean():.0f}%)"
        )
        print(
            f"  R:      NISAR better at {(d_r > 0).sum()}/{len(merged)} "
            f"({100 * (d_r > 0).mean():.0f}%)"
        )
        print(f"  Median ubRMSE improvement: {d_ub.median():.4f}")
        print(f"  Median R improvement: {d_r.median():.4f}")

    # By wetness bin
    print("\n--- By Mean In-Situ VWC ---")
    bins = [0, 0.10, 0.20, 0.30, 0.40, 1.0]
    labels = ["<0.10", "0.10-0.20", "0.20-0.30", "0.30-0.40", ">0.40"]

    if len(merged) > 0:
        merged["vwc_bin"] = pd.cut(
            merged["insitu_mean_l3"], bins=bins, labels=labels, right=False
        )
        print(
            f"{'VWC bin':<12s} {'N':>5s} {'L3 ubRMSE':>12s} "
            f"{'NISAR ubRMSE':>13s} {'Delta':>12s}"
        )
        print("-" * 56)
        for label in labels:
            sub = merged[merged["vwc_bin"] == label]
            if len(sub) < 3:
                continue
            l3_ub = sub["ubrmse_l3"].median()
            n_ub = sub["ubrmse_nisar"].median()
            d = l3_ub - n_ub
            sign = "+" if d >= 0 else ""
            print(
                f"{label:<12s} {len(sub):>5d} {l3_ub:>12.4f} "
                f"{n_ub:>13.4f} {sign}{d:>11.4f}"
            )

    # By network
    print("\n--- By ISMN Network ---")
    if len(merged) > 0:
        merged["network"] = merged["station"].str.split(":").str[0]
        print(
            f"{'Network':<15s} {'N':>5s} {'L3 ubRMSE':>12s} "
            f"{'NISAR ubRMSE':>13s} {'Delta':>12s}"
        )
        print("-" * 59)
        for net in merged["network"].value_counts().index:
            sub = merged[merged["network"] == net]
            if len(sub) < 3:
                continue
            l3_ub = sub["ubrmse_l3"].median()
            n_ub = sub["ubrmse_nisar"].median()
            d = l3_ub - n_ub
            sign = "+" if d >= 0 else ""
            print(
                f"{net:<15s} {len(sub):>5d} {l3_ub:>12.4f} "
                f"{n_ub:>13.4f} {sign}{d:>11.4f}"
            )


def run_full_pipeline(
    l3_csv: str,
    ismn_dir: str,
    output_csv: str,
    min_pairs: int = 10,
) -> None:
    """Run complete NISAR evaluation: extract → score → compare."""
    extractions_path = output_csv.replace(".csv", "_extractions.parquet")

    print("Step 1: Computing station EASE-2 200 m coordinates...")
    station_coords = precompute_station_coords(l3_csv)
    print(f"  {len(station_coords)} stations")

    print("\nStep 2: Extracting NISAR SM at ISMN stations (remote streaming)...")
    extract_all_granules(station_coords, extractions_path)

    print("\nStep 3: Merging with ISMN and scoring...")
    merge_and_score(extractions_path, ismn_dir, output_csv, l3_csv, min_pairs)

    print("\nStep 4: Comparison with SMAP L3...")
    compare_products(l3_csv, output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate NISAR L3 SME2 vs ISMN in-situ soil moisture",
    )
    sub = parser.add_subparsers(dest="command")

    p_full = sub.add_parser("full", help="Run complete pipeline")
    p_full.add_argument("--l3-csv", required=True)
    p_full.add_argument("--ismn-dir", required=True)
    p_full.add_argument("--output", required=True)
    p_full.add_argument("--min-pairs", type=int, default=10)

    p_cmp = sub.add_parser("compare", help="SMAP L3 vs NISAR comparison")
    p_cmp.add_argument("--l3-csv", required=True)
    p_cmp.add_argument("--nisar-csv", required=True)

    args = parser.parse_args()

    if args.command == "full":
        run_full_pipeline(args.l3_csv, args.ismn_dir, args.output, args.min_pairs)
    elif args.command == "compare":
        compare_products(args.l3_csv, args.nisar_csv)
    else:
        parser.print_help()
