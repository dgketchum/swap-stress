"""
Evaluate SMAP L4 (SPL4SMGP) soil moisture against ISMN in-situ stations.

Extracts surface soil moisture from L4 daily CONUS GeoTIFFs at ISMN station
locations, merges with ISMN daily VWC at 5 cm depth, computes per-station
accuracy metrics, and produces a three-way comparison with L3 and SPL2SMAP_S.

Usage:
    # Full pipeline: extract → score → compare
    python -m research.sensors.smap_l4_ismn full \
        --tif-dir /nas/soils/smap/SPL4SMGP/daily_tif \
        --l3-csv /nas/soils/vwc_timeseries/ismn/smap_ismn_5cm_comparison.csv \
        --ismn-dir /nas/soils/vwc_timeseries/ismn/processed_time_series/preprocessed_by_station \
        --output /nas/soils/vwc_timeseries/ismn/smap_l4_ismn_5cm_comparison.csv

    # Three-way comparison only
    python -m research.sensors.smap_l4_ismn compare \
        --l3-csv /nas/soils/vwc_timeseries/ismn/smap_ismn_5cm_comparison.csv \
        --ss-csv /nas/soils/vwc_timeseries/ismn/smap_s_ismn_5cm_comparison.csv \
        --l4-csv /nas/soils/vwc_timeseries/ismn/smap_l4_ismn_5cm_comparison.csv
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
import rasterio


def _parse_date(filename: str) -> str | None:
    """Extract YYYYMMDD from smap_l4_sm_YYYYMMDD.tif."""
    m = re.search(r"_(\d{8})\.tif$", filename)
    if m:
        return m.group(1)
    return None


def extract_l4_at_stations(tif_dir: str, l3_csv: str) -> pd.DataFrame:
    """Extract L4 SM at ISMN station locations from daily CONUS GeoTIFFs.

    Uses the M09 pixel_row / pixel_col from the L3 comparison CSV to index
    directly into the L4 rasters (both are on the same 634x295 grid).

    Parameters
    ----------
    tif_dir : str
        Directory containing smap_l4_sm_YYYYMMDD.tif files.
    l3_csv : str
        Existing L3 comparison CSV with station, lat, lon, pixel_row, pixel_col.

    Returns
    -------
    pd.DataFrame
        Columns: station, date, smap_l4_sm
    """
    l3 = pd.read_csv(l3_csv)

    # pixel_row/pixel_col in the L3 CSV are already local CONUS indices
    # (0-based within the 295×634 subset), not global M09 coordinates.
    stations = []
    for _, r in l3.iterrows():
        stations.append(
            {
                "station": r["station"],
                "lat": r["lat"],
                "lon": r["lon"],
                "local_row": int(r["pixel_row"]),
                "local_col": int(r["pixel_col"]),
            }
        )

    tif_files = sorted(f for f in os.listdir(tif_dir) if f.endswith(".tif"))
    print(
        f"Extracting from {len(tif_files)} L4 GeoTIFFs at {len(stations)} stations..."
    )

    records = []
    for i, fname in enumerate(tif_files):
        date_str = _parse_date(fname)
        if date_str is None:
            continue

        fp = os.path.join(tif_dir, fname)
        with rasterio.open(fp) as src:
            sm = src.read(1)

        for st in stations:
            r, c = st["local_row"], st["local_col"]
            if 0 <= r < sm.shape[0] and 0 <= c < sm.shape[1]:
                val = sm[r, c]
                if not np.isnan(val) and 0 <= val <= 1.0:
                    records.append(
                        {
                            "station": st["station"],
                            "date": date_str,
                            "smap_l4_sm": float(val),
                        }
                    )

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(tif_files)} files, {len(records)} extractions")

    print(f"Total extractions: {len(records)}")
    if not records:
        return pd.DataFrame(columns=["station", "date", "smap_l4_sm"])
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df


def merge_and_score(
    extractions: pd.DataFrame,
    ismn_station_dir: str,
    output_csv: str,
    l3_csv: str,
    min_pairs: int = 50,
) -> pd.DataFrame:
    """Merge L4 extractions with ISMN VWC and compute per-station metrics."""
    l3_df = pd.read_csv(l3_csv)
    station_meta = l3_df.set_index("station")[["lat", "lon", "pixel_row", "pixel_col"]]
    station_meta = station_meta.to_dict("index")

    records = []
    n_skip = {"nofile": 0, "nocol": 0, "fewpairs": 0}

    for station in extractions["station"].unique():
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

        st_ext = extractions[extractions["station"] == station].copy()
        st_ext = st_ext.set_index("date")["smap_l4_sm"]

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
                "pixel_row": meta["pixel_row"],
                "pixel_col": meta["pixel_col"],
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
    print(f"  Skipped (no parquet): {n_skip['nofile']}")
    print(f"  Skipped (no soil_vwc_5): {n_skip['nocol']}")
    print(f"  Skipped (<{min_pairs} pairs): {n_skip['fewpairs']}")

    result = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    result.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")
    return result


def compare_three_products(l3_csv: str, ss_csv: str, l4_csv: str) -> None:
    """Print three-way comparison of L3, SPL2SMAP_S, and L4 at ISMN stations."""
    l3 = pd.read_csv(l3_csv)
    l4 = pd.read_csv(l4_csv)

    has_ss = os.path.exists(ss_csv) if ss_csv else False
    ss = pd.read_csv(ss_csv) if has_ss else None

    print(f"\n{'=' * 80}")
    print("SMAP Product Comparison at ISMN 5 cm Stations")
    print(f"{'=' * 80}\n")

    header = f"{'Metric':<25s} {'L3 (9 km)':>12s}"
    if has_ss:
        header += f" {'SS (3 km)':>12s}"
    header += f" {'L4 (9 km)':>12s}"
    print(header)
    print("-" * len(header))

    for metric in ["rmse", "ubrmse", "r", "bias"]:
        row = f"Median {metric:<18s} {l3[metric].median():>12.4f}"
        if has_ss:
            row += f" {ss[metric].median():>12.4f}"
        row += f" {l4[metric].median():>12.4f}"
        print(row)

    row = f"{'Stations (>=50 pairs)':<25s} {len(l3):>12d}"
    if has_ss:
        row += f" {len(ss):>12d}"
    row += f" {len(l4):>12d}"
    print(row)

    # L3 vs L4 paired comparison
    merged = l3.merge(l4, on="station", suffixes=("_l3", "_l4"))
    print(f"\n--- L3 vs L4 Paired Comparison ({len(merged)} stations) ---")
    if len(merged) > 0:
        d_ub = merged["ubrmse_l3"] - merged["ubrmse_l4"]
        d_r = merged["r_l4"] - merged["r_l3"]
        print(
            f"  ubRMSE: L4 better at {(d_ub > 0).sum()}/{len(merged)} stations "
            f"({100 * (d_ub > 0).mean():.0f}%), "
            f"median improvement={d_ub.median():.4f}"
        )
        print(
            f"  R:      L4 better at {(d_r > 0).sum()}/{len(merged)} stations "
            f"({100 * (d_r > 0).mean():.0f}%), "
            f"median improvement={d_r.median():.4f}"
        )

    # Stratified by wetness
    print("\n--- By Mean In-Situ VWC ---")
    bins = [0, 0.10, 0.20, 0.30, 0.40, 1.0]
    labels = ["<0.10", "0.10-0.20", "0.20-0.30", "0.30-0.40", ">0.40"]

    if len(merged) > 0:
        merged["vwc_bin"] = pd.cut(
            merged["insitu_mean_l3"], bins=bins, labels=labels, right=False
        )
        header = f"{'VWC bin':<12s} {'N':>5s} {'L3 ubRMSE':>12s} {'L4 ubRMSE':>12s} {'Delta':>12s}"
        print(header)
        print("-" * len(header))
        for label in labels:
            sub = merged[merged["vwc_bin"] == label]
            if len(sub) == 0:
                continue
            l3_ub = sub["ubrmse_l3"].median()
            l4_ub = sub["ubrmse_l4"].median()
            d = l3_ub - l4_ub
            sign = "+" if d >= 0 else ""
            print(
                f"{label:<12s} {len(sub):>5d} {l3_ub:>12.4f} {l4_ub:>12.4f} {sign}{d:>11.4f}"
            )

    # Stratified by network
    print("\n--- By ISMN Network ---")
    if len(merged) > 0:
        merged["network"] = merged["station"].str.split(":").str[0]
        header = f"{'Network':<15s} {'N':>5s} {'L3 ubRMSE':>12s} {'L4 ubRMSE':>12s} {'Delta':>12s}"
        print(header)
        print("-" * len(header))
        for net in merged["network"].value_counts().index:
            sub = merged[merged["network"] == net]
            if len(sub) < 3:
                continue
            l3_ub = sub["ubrmse_l3"].median()
            l4_ub = sub["ubrmse_l4"].median()
            d = l3_ub - l4_ub
            sign = "+" if d >= 0 else ""
            print(
                f"{net:<15s} {len(sub):>5d} {l3_ub:>12.4f} {l4_ub:>12.4f} {sign}{d:>11.4f}"
            )

    # Coverage stats
    print("\n--- L4 Coverage ---")
    print(f"  Median paired obs per station: {l4['n_paired'].median():.0f}")
    print(
        f"  Mean: {l4['n_paired'].mean():.0f}, "
        f"Min: {l4['n_paired'].min()}, Max: {l4['n_paired'].max()}"
    )


def run_full_pipeline(
    tif_dir: str,
    l3_csv: str,
    ismn_dir: str,
    output_csv: str,
    ss_csv: str = "",
    min_pairs: int = 50,
) -> None:
    """Run complete L4 evaluation: extract → score → compare."""
    print("Step 1: Extracting L4 SM at ISMN stations...")
    extractions = extract_l4_at_stations(tif_dir, l3_csv)

    print("\nStep 2: Merging with ISMN and scoring...")
    merge_and_score(extractions, ismn_dir, output_csv, l3_csv, min_pairs)

    print("\nStep 3: Three-way comparison...")
    compare_three_products(l3_csv, ss_csv, output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate SMAP L4 vs ISMN in-situ soil moisture",
    )
    sub = parser.add_subparsers(dest="command")

    p_full = sub.add_parser("full", help="Run complete pipeline")
    p_full.add_argument("--tif-dir", required=True)
    p_full.add_argument("--l3-csv", required=True)
    p_full.add_argument("--ismn-dir", required=True)
    p_full.add_argument("--output", required=True)
    p_full.add_argument("--ss-csv", default="")
    p_full.add_argument("--min-pairs", type=int, default=50)

    p_cmp = sub.add_parser("compare", help="Three-way comparison")
    p_cmp.add_argument("--l3-csv", required=True)
    p_cmp.add_argument("--ss-csv", default="")
    p_cmp.add_argument("--l4-csv", required=True)

    args = parser.parse_args()

    if args.command == "full":
        run_full_pipeline(
            args.tif_dir,
            args.l3_csv,
            args.ismn_dir,
            args.output,
            args.ss_csv,
            args.min_pairs,
        )
    elif args.command == "compare":
        compare_three_products(args.l3_csv, args.ss_csv, args.l4_csv)
    else:
        parser.print_help()
