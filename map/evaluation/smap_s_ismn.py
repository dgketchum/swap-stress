"""
Evaluate SPL2SMAP_S (3 km) soil moisture against ISMN in-situ stations.

Extracts soil moisture from SPL2SMAP_S HDF5 granules at ISMN station locations
using the embedded EASE-Grid2 M03 row/column indices, merges with ISMN daily
VWC at 5 cm depth, and computes per-station accuracy metrics identical to the
existing L3 comparison.

Usage:
    # Extract SM at ISMN stations from all downloaded granules
    python -m map.evaluation.smap_s_ismn extract \
        --hdf5-dir /nas/soils/smap/SPL2SMAP_S/hdf5 \
        --l3-csv /nas/soils/vwc_timeseries/ismn/smap_ismn_5cm_comparison.csv \
        --output /nas/soils/smap/SPL2SMAP_S/ismn_extractions.parquet

    # Merge with ISMN and score
    python -m map.evaluation.smap_s_ismn score \
        --extractions /nas/soils/smap/SPL2SMAP_S/ismn_extractions.parquet \
        --ismn-dir /nas/soils/vwc_timeseries/ismn/processed_time_series/preprocessed_by_station \
        --output /nas/soils/vwc_timeseries/ismn/smap_s_ismn_5cm_comparison.csv

    # Head-to-head comparison with L3
    python -m map.evaluation.smap_s_ismn compare \
        --l3-csv /nas/soils/vwc_timeseries/ismn/smap_ismn_5cm_comparison.csv \
        --ss-csv /nas/soils/vwc_timeseries/ismn/smap_s_ismn_5cm_comparison.csv
"""

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
import pandas as pd

from map.data.smap_sentinel_download import lonlat_to_m03_colrow

# HDF5 paths within SPL2SMAP_S files
_GRP = "Soil_Moisture_Retrieval_Data_3km"
_SM_FIELD = f"{_GRP}/soil_moisture_3km"
_ROW_FIELD = f"{_GRP}/EASE_row_index_3km"
_COL_FIELD = f"{_GRP}/EASE_column_index_3km"
_FLAG_FIELD = f"{_GRP}/retrieval_qual_flag_3km"
_LAT_FIELD = f"{_GRP}/latitude_3km"
_LON_FIELD = f"{_GRP}/longitude_3km"

_SM_FILL = -9999.0


def precompute_station_m03_coords(l3_csv: str) -> dict:
    """Convert ISMN station locations to M03 grid indices.

    Parameters
    ----------
    l3_csv : str
        Path to the existing L3 ISMN comparison CSV (has station, lat, lon).

    Returns
    -------
    dict
        {station_uid: (ease_row_3km, ease_col_3km)} — 0-based global M03 indices.
    """
    df = pd.read_csv(l3_csv)
    coords = {}
    for _, row in df.iterrows():
        col, r = lonlat_to_m03_colrow(row["lon"], row["lat"])
        coords[row["station"]] = (r, col)
    return coords


def _parse_smap_date(filename: str) -> str | None:
    """Extract SMAP overpass date (YYYYMMDD) from SPL2SMAP_S filename.

    Filename pattern:
    SMAP_L2_SM_SP_..._[SMAP_date]Thhmmss_..._RLVvvv_NNN.h5
    The SMAP date is the first 8-digit group after an underscore.
    """
    m = re.search(r"_(\d{8})T\d{6}_", filename)
    if m:
        return m.group(1)
    return None


def extract_granule(hdf5_path: str, station_coords: dict) -> list:
    """Extract soil moisture at ISMN stations from one SPL2SMAP_S granule.

    Parameters
    ----------
    hdf5_path : str
        Path to one SPL2SMAP_S HDF5 file.
    station_coords : dict
        {station_uid: (ease_row, ease_col)} from precompute_station_m03_coords.

    Returns
    -------
    list of dict
        Each dict has keys: station, date, smap_s_sm, qual_flag.
        Empty list if no stations are covered by this granule.
    """
    fname = os.path.basename(hdf5_path)
    date_str = _parse_smap_date(fname)
    if date_str is None:
        return []

    try:
        with h5py.File(hdf5_path, "r") as f:
            if _SM_FIELD not in f:
                return []

            sm = f[_SM_FIELD][:].ravel().astype(np.float32)
            rows = f[_ROW_FIELD][:].ravel().astype(np.int32)
            cols = f[_COL_FIELD][:].ravel().astype(np.int32)
            flags = f[_FLAG_FIELD][:].ravel().astype(np.uint16)
    except Exception:
        return []

    # Build lookup from (row, col) -> flat index
    # Filter out fill pixels (row/col of 65534 or sm == fill)
    valid = (sm != _SM_FILL) & (sm >= 0) & (sm <= 1.0) & (rows >= 0) & (cols >= 0)
    idx_valid = np.where(valid)[0]

    if len(idx_valid) == 0:
        return []

    pixel_lookup = {}
    for i in idx_valid:
        key = (int(rows[i]), int(cols[i]))
        if key not in pixel_lookup:
            pixel_lookup[key] = i

    results = []
    for station, (sr, sc) in station_coords.items():
        idx = pixel_lookup.get((sr, sc))
        if idx is not None:
            results.append(
                {
                    "station": station,
                    "date": date_str,
                    "smap_s_sm": float(sm[idx]),
                    "qual_flag": int(flags[idx]),
                }
            )

    return results


def _extract_worker(args):
    """Worker for parallel extraction."""
    hdf5_path, station_coords = args
    return extract_granule(hdf5_path, station_coords)


def extract_all_granules(
    hdf5_dir: str,
    station_coords: dict,
    output_path: str,
    n_workers: int = 8,
) -> pd.DataFrame:
    """Extract SPL2SMAP_S SM at all ISMN stations across all granules.

    Aggregates by (station, date): takes mean SM when multiple granules
    provide data for the same station-day.

    Parameters
    ----------
    hdf5_dir : str
        Directory containing SPL2SMAP_S HDF5 files.
    station_coords : dict
        From precompute_station_m03_coords.
    output_path : str
        Where to write the intermediate parquet.
    n_workers : int
        Parallel workers for HDF5 reading.

    Returns
    -------
    pd.DataFrame
        Columns: station, date, smap_s_sm, n_granules, min_qual_flag
    """
    h5_files = sorted(
        os.path.join(hdf5_dir, f) for f in os.listdir(hdf5_dir) if f.endswith(".h5")
    )
    print(f"Processing {len(h5_files)} granules with {n_workers} workers...")

    all_records = []
    done = 0

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {
            ex.submit(extract_granule, fp, station_coords): fp for fp in h5_files
        }
        for fut in as_completed(futures):
            done += 1
            if done % 500 == 0:
                print(
                    f"  {done}/{len(h5_files)} granules processed, "
                    f"{len(all_records)} extractions so far"
                )
            try:
                recs = fut.result()
                all_records.extend(recs)
            except Exception as e:
                print(f"  Error processing {futures[fut]}: {e}")

    print(f"Total raw extractions: {len(all_records)}")

    if not all_records:
        print("WARNING: No extractions found. Check HDF5 directory and station coords.")
        empty = pd.DataFrame(
            columns=["station", "date", "smap_s_sm", "n_granules", "min_qual_flag"]
        )
        empty.to_parquet(output_path, index=False)
        return empty

    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

    # Aggregate by (station, date): mean SM, count, min qual flag
    agg = (
        df.groupby(["station", "date"])
        .agg(
            smap_s_sm=("smap_s_sm", "mean"),
            n_granules=("smap_s_sm", "count"),
            min_qual_flag=("qual_flag", "min"),
        )
        .reset_index()
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    agg.to_parquet(output_path, index=False)
    print(f"Wrote {len(agg)} station-day extractions to {output_path}")

    return agg


def merge_and_score(
    extractions_path: str,
    ismn_station_dir: str,
    output_csv: str,
    l3_csv: str,
    min_pairs: int = 50,
) -> pd.DataFrame:
    """Merge SPL2SMAP_S extractions with ISMN in-situ VWC and compute metrics.

    Parameters
    ----------
    extractions_path : str
        Parquet from extract_all_granules.
    ismn_station_dir : str
        Directory of per-station ISMN parquets.
    output_csv : str
        Where to write the comparison CSV.
    l3_csv : str
        Existing L3 comparison CSV (for station lat/lon and M03 pixel coords).
    min_pairs : int
        Minimum paired observations to include a station.

    Returns
    -------
    pd.DataFrame
        Per-station metrics.
    """
    ext = pd.read_parquet(extractions_path)
    ext["date"] = pd.to_datetime(ext["date"])

    l3_df = pd.read_csv(l3_csv)
    station_meta = l3_df.set_index("station")[["lat", "lon"]].to_dict("index")

    records = []
    n_skip_nofile = 0
    n_skip_nocol = 0
    n_skip_fewpairs = 0

    for station in ext["station"].unique():
        meta = station_meta.get(station)
        if meta is None:
            continue

        # Find ISMN parquet
        safe_station = station.replace(":", "_").replace("/", "_").replace(" ", "_")
        pq_path = os.path.join(ismn_station_dir, f"{safe_station}.parquet")
        if not os.path.exists(pq_path):
            n_skip_nofile += 1
            continue

        ismn = pd.read_parquet(pq_path)
        if "soil_vwc_5" not in ismn.columns:
            n_skip_nocol += 1
            continue

        ismn["datetime"] = pd.to_datetime(ismn["datetime"])
        ismn = ismn[["datetime", "soil_vwc_5"]].dropna().copy()
        ismn["date"] = ismn["datetime"].dt.normalize()

        # Daily mean in-situ VWC
        daily_ismn = ismn.groupby("date")["soil_vwc_5"].mean()

        # Station's SPL2SMAP_S extractions
        st_ext = ext[ext["station"] == station].copy()
        st_ext = st_ext.set_index("date")["smap_s_sm"]

        # Inner join on date
        paired = pd.DataFrame({"insitu": daily_ismn, "smap": st_ext}).dropna()

        if len(paired) < min_pairs:
            n_skip_fewpairs += 1
            continue

        diff = paired["smap"] - paired["insitu"]
        bias = diff.mean()
        rmse = np.sqrt((diff**2).mean())
        ubrmse = np.sqrt(((diff - bias) ** 2).mean())
        r = paired["smap"].corr(paired["insitu"])

        # Compute M03 pixel coords for the CSV
        col, row = lonlat_to_m03_colrow(meta["lon"], meta["lat"])

        records.append(
            {
                "station": station,
                "lat": meta["lat"],
                "lon": meta["lon"],
                "pixel_row": row,
                "pixel_col": col,
                "n_paired": len(paired),
                "bias": bias,
                "rmse": rmse,
                "ubrmse": ubrmse,
                "r": r,
                "insitu_mean": paired["insitu"].mean(),
                "smap_mean": paired["smap"].mean(),
            }
        )

    print(f"Stations scored: {len(records)}")
    print(f"  Skipped (no ISMN parquet): {n_skip_nofile}")
    print(f"  Skipped (no soil_vwc_5): {n_skip_nocol}")
    print(f"  Skipped (<{min_pairs} pairs): {n_skip_fewpairs}")

    result = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    result.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")

    return result


def compare_products(l3_csv: str, ss_csv: str) -> None:
    """Print head-to-head comparison of L3 vs SPL2SMAP_S at ISMN stations."""
    l3 = pd.read_csv(l3_csv)
    ss = pd.read_csv(ss_csv)

    print(f"\n{'=' * 70}")
    print("SPL2SMAP_S (3 km) vs SMAP L3 (9 km) — ISMN 5 cm Comparison")
    print(f"{'=' * 70}\n")

    # Overall summary
    print(f"{'Metric':<25s} {'L3 (9 km)':>12s} {'SS (3 km)':>12s} {'Delta':>12s}")
    print("-" * 62)

    for metric in ["rmse", "ubrmse", "r", "bias"]:
        l3_med = l3[metric].median()
        ss_med = ss[metric].median()
        delta = ss_med - l3_med
        sign = "+" if delta >= 0 else ""
        print(
            f"Median {metric:<18s} {l3_med:>12.4f} {ss_med:>12.4f} {sign}{delta:>11.4f}"
        )

    print(f"{'Stations w/ >=50 pairs':<25s} {len(l3):>12d} {len(ss):>12d}")

    # Paired station comparison (inner join)
    merged = l3.merge(ss, on="station", suffixes=("_l3", "_ss"))
    print(f"\nStations in both products: {len(merged)}")

    if len(merged) > 0:
        delta_ubrmse = merged["ubrmse_l3"] - merged["ubrmse_ss"]
        delta_r = merged["r_ss"] - merged["r_l3"]

        print("\nPer-station improvement (positive = SS better):")
        print(
            f"  ubRMSE reduction:  median={delta_ubrmse.median():.4f}, "
            f"mean={delta_ubrmse.mean():.4f}"
        )
        print(
            f"  Stations where SS wins (ubRMSE): "
            f"{(delta_ubrmse > 0).sum()}/{len(merged)} "
            f"({100 * (delta_ubrmse > 0).mean():.0f}%)"
        )
        print(
            f"  R improvement:     median={delta_r.median():.4f}, "
            f"mean={delta_r.mean():.4f}"
        )
        print(
            f"  Stations where SS wins (R): "
            f"{(delta_r > 0).sum()}/{len(merged)} "
            f"({100 * (delta_r > 0).mean():.0f}%)"
        )

    # Stratified by mean wetness
    print(f"\n{'--- By mean in-situ VWC ---':^62s}")
    bins = [0, 0.10, 0.20, 0.30, 0.40, 1.0]
    labels = ["<0.10", "0.10-0.20", "0.20-0.30", "0.30-0.40", ">0.40"]

    if len(merged) > 0:
        merged["vwc_bin"] = pd.cut(
            merged["insitu_mean_l3"], bins=bins, labels=labels, right=False
        )
        print(
            f"{'VWC bin':<12s} {'N':>5s} {'L3 ubRMSE':>12s} {'SS ubRMSE':>12s} "
            f"{'Delta':>12s}"
        )
        print("-" * 55)
        for label in labels:
            sub = merged[merged["vwc_bin"] == label]
            if len(sub) == 0:
                continue
            l3_ub = sub["ubrmse_l3"].median()
            ss_ub = sub["ubrmse_ss"].median()
            d = l3_ub - ss_ub
            sign = "+" if d >= 0 else ""
            print(
                f"{label:<12s} {len(sub):>5d} {l3_ub:>12.4f} {ss_ub:>12.4f} "
                f"{sign}{d:>11.4f}"
            )

    # Stratified by network
    print(f"\n{'--- By ISMN network ---':^62s}")
    if len(merged) > 0:
        merged["network"] = merged["station"].str.split(":").str[0]
        print(
            f"{'Network':<15s} {'N':>5s} {'L3 ubRMSE':>12s} {'SS ubRMSE':>12s} "
            f"{'Delta':>12s}"
        )
        print("-" * 58)
        net_counts = merged["network"].value_counts()
        for net in net_counts.index:
            sub = merged[merged["network"] == net]
            if len(sub) < 3:
                continue
            l3_ub = sub["ubrmse_l3"].median()
            ss_ub = sub["ubrmse_ss"].median()
            d = l3_ub - ss_ub
            sign = "+" if d >= 0 else ""
            print(
                f"{net:<15s} {len(sub):>5d} {l3_ub:>12.4f} {ss_ub:>12.4f} "
                f"{sign}{d:>11.4f}"
            )

    # Coverage statistics for SPL2SMAP_S
    print(f"\n{'--- SPL2SMAP_S Coverage ---':^62s}")
    print(f"  Median paired obs per station: {ss['n_paired'].median():.0f}")
    print(f"  Mean paired obs per station:   {ss['n_paired'].mean():.0f}")
    print(f"  Min: {ss['n_paired'].min()}, Max: {ss['n_paired'].max()}")
    p25, p75 = ss["n_paired"].quantile([0.25, 0.75])
    print(f"  IQR: [{p25:.0f}, {p75:.0f}]")

    # Decision gate
    print(f"\n{'=' * 70}")
    print("DECISION GATE")
    print(f"{'=' * 70}")

    gate_coverage = len(ss) >= 200
    gate_ubrmse = ss["ubrmse"].median() <= (l3["ubrmse"].median() - 0.005)
    gate_r = ss["r"].median() >= (l3["r"].median() - 0.02)

    print(
        f"  Coverage (>=200 stations):    {'PASS' if gate_coverage else 'FAIL'} "
        f"({len(ss)} stations)"
    )
    print(
        f"  ubRMSE improvement (>=0.005): {'PASS' if gate_ubrmse else 'FAIL'} "
        f"(L3={l3['ubrmse'].median():.4f}, SS={ss['ubrmse'].median():.4f})"
    )
    print(
        f"  R no regression (<=0.02):     {'PASS' if gate_r else 'FAIL'} "
        f"(L3={l3['r'].median():.4f}, SS={ss['r'].median():.4f})"
    )

    if gate_coverage and gate_ubrmse and gate_r:
        print("\n  >>> ALL GATES PASS — proceed to full pipeline evaluation")
    else:
        print("\n  >>> GATE FAILED — L3 pipeline remains. Document and stop.")


def run_full_pipeline(
    hdf5_dir: str,
    l3_csv: str,
    ismn_station_dir: str,
    extractions_path: str,
    output_csv: str,
    n_workers: int = 8,
    min_pairs: int = 50,
) -> None:
    """Run complete evaluation: extract → score → compare."""
    print("Step 1: Precomputing station M03 grid coordinates...")
    station_coords = precompute_station_m03_coords(l3_csv)
    print(f"  {len(station_coords)} stations mapped to M03 grid")

    print("\nStep 2: Extracting SM from SPL2SMAP_S granules...")
    extract_all_granules(hdf5_dir, station_coords, extractions_path, n_workers)

    print("\nStep 3: Merging with ISMN and computing metrics...")
    merge_and_score(extractions_path, ismn_station_dir, output_csv, l3_csv, min_pairs)

    print("\nStep 4: Head-to-head comparison...")
    compare_products(l3_csv, output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate SPL2SMAP_S vs ISMN in-situ soil moisture",
    )
    sub = parser.add_subparsers(dest="command")

    # --- extract ---
    p_ext = sub.add_parser("extract", help="Extract SM at ISMN stations from granules")
    p_ext.add_argument("--hdf5-dir", required=True)
    p_ext.add_argument("--l3-csv", required=True)
    p_ext.add_argument("--output", required=True)
    p_ext.add_argument("--n-workers", type=int, default=8)

    # --- score ---
    p_score = sub.add_parser("score", help="Merge extractions with ISMN and score")
    p_score.add_argument("--extractions", required=True)
    p_score.add_argument("--ismn-dir", required=True)
    p_score.add_argument("--l3-csv", required=True)
    p_score.add_argument("--output", required=True)
    p_score.add_argument("--min-pairs", type=int, default=50)

    # --- compare ---
    p_cmp = sub.add_parser("compare", help="Head-to-head L3 vs SPL2SMAP_S")
    p_cmp.add_argument("--l3-csv", required=True)
    p_cmp.add_argument("--ss-csv", required=True)

    # --- full ---
    p_full = sub.add_parser(
        "full", help="Run complete pipeline: extract → score → compare"
    )
    p_full.add_argument("--hdf5-dir", required=True)
    p_full.add_argument("--l3-csv", required=True)
    p_full.add_argument("--ismn-dir", required=True)
    p_full.add_argument("--extractions", required=True)
    p_full.add_argument("--output", required=True)
    p_full.add_argument("--n-workers", type=int, default=8)
    p_full.add_argument("--min-pairs", type=int, default=50)

    args = parser.parse_args()

    if args.command == "extract":
        coords = precompute_station_m03_coords(args.l3_csv)
        extract_all_granules(args.hdf5_dir, coords, args.output, args.n_workers)

    elif args.command == "score":
        merge_and_score(
            args.extractions, args.ismn_dir, args.output, args.l3_csv, args.min_pairs
        )

    elif args.command == "compare":
        compare_products(args.l3_csv, args.ss_csv)

    elif args.command == "full":
        run_full_pipeline(
            hdf5_dir=args.hdf5_dir,
            l3_csv=args.l3_csv,
            ismn_station_dir=args.ismn_dir,
            extractions_path=args.extractions,
            output_csv=args.output,
            n_workers=args.n_workers,
            min_pairs=args.min_pairs,
        )

    else:
        parser.print_help()
