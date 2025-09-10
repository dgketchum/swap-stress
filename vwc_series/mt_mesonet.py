import os

import geopandas as gpd
import pandas as pd


def get_all_station_timeseries(metadata_csv_path, output_dir=None, frequency='daily',
                               summary_csv_path=None, summary_geojson_path=None):
    """
    Fetches time series for all stations listed in the metadata and returns a
    dictionary mapping station IDs to pivoted DataFrames.

    The metadata must include a station identifier and the start date for the
    time series. The function constructs a Mesonet API URL per station, pulls
    the long-format CSV, and pivots on the 'element' column to produce a wide
    table with one column per element.

    Args:
        metadata_csv_path (str): Path to the station metadata CSV file.
        output_dir (str, optional): If provided, saves each station's pivoted
            time series to Parquet in `<output_dir>/mesonet_timeseries`.
        frequency (str): One of 'daily' or 'hourly' for the Mesonet API path.

    Returns:
        dict[str, pd.DataFrame]: Mapping of station ID to pivoted DataFrame.
    """
    meta = pd.read_csv(metadata_csv_path)

    station_col = 'station'
    start_col = 'date_installed'

    # Normalize start time to YYYY-MM-DD string
    meta['_start'] = pd.to_datetime(meta[start_col], errors='coerce').dt.date.astype(str)

    results = {}
    for _, row in meta.iterrows():
        station_id = str(row[station_col]).strip()
        start_date = row['_start']
        if not station_id or start_date == 'NaT':
            continue

        url = (
            f"https://mesonet.climate.umt.edu/api/v2/observations/{frequency}/"
            f"?stations={station_id}&type=csv&start_time={start_date}&premade=true&wide=false"
        )

        try:
            df = pd.read_csv(url)
        except Exception as e:
            print(f"Warning: failed to load data for {station_id} from {url}: {e}")
            continue

        if df.empty:
            print(f"No data returned for station {station_id}.")
            continue

        time_col = 'datetime'
        value_col = 'value'
        units_col = 'units'

        if time_col not in df.columns or value_col not in df.columns or units_col not in df.columns:
            print(f"Unexpected schema for station {station_id}: columns={list(df.columns)}")
            continue

        wide_vals = df.pivot_table(index=time_col, columns='element', values=value_col, aggfunc='first')
        wide_units = df.pivot_table(index=time_col, columns='element', values=units_col, aggfunc='first')
        wide_units.columns = [f"{c}_units" for c in wide_units.columns]
        wide_df = wide_vals.join(wide_units, how='left')
        wide_df = wide_df.sort_index().reset_index()
        wide_df.insert(0, 'station', station_id)

        results[station_id] = wide_df

        out_fp = os.path.join(output_dir, f'{station_id}_{frequency}.parquet')
        try:
            wide_df.to_parquet(out_fp, index=False)
            print(out_fp)
        except Exception as e:
            print(f"Warning: failed to save Parquet for {station_id}: {e}")

    # After fetching, optionally write VWC observation summary CSV/GeoJSON
    if summary_csv_path or summary_geojson_path:
        rows = []
        for fn in os.listdir(output_dir or ''):
            if not fn.endswith('.parquet'):
                continue
            fp = os.path.join(output_dir, fn)
            try:
                sdf = pd.read_parquet(fp)
            except Exception as e:
                print(f"Warning: failed reading {fp}: {e}")
                continue

            # Identify VWC element columns (exclude unit columns)
            vwc_cols = [c for c in sdf.columns if isinstance(c, str) and c.startswith('soil_vwc_') and not c.endswith('_units')]
            if not vwc_cols:
                continue

            if 'station' in sdf.columns and len(sdf['station'].dropna()) > 0:
                sid = str(sdf['station'].iloc[0])
            else:
                sid = os.path.basename(fn).split('_')[0]

            counts = sdf[vwc_cols].notna().sum().astype(int).to_dict()
            row = {'station': sid}
            row.update(counts)
            rows.append(row)

        if rows:
            summary_df = pd.DataFrame(rows).fillna(0)
            for c in summary_df.columns:
                if c != 'station':
                    summary_df[c] = summary_df[c].astype(int)

            # Merge station metadata for coordinates
            meta_uniq = meta.drop_duplicates(subset=['station']).copy()
            merged = pd.merge(summary_df, meta_uniq, on='station', how='left')

            if summary_csv_path:
                os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)
                merged.to_csv(summary_csv_path, index=False)
                print(f"Wrote VWC observation summary CSV: {summary_csv_path}")

            if summary_geojson_path:
                lat_col = 'latitude' if 'latitude' in merged.columns else ('lat' if 'lat' in merged.columns else None)
                lon_col = 'longitude' if 'longitude' in merged.columns else ('lon' if 'lon' in merged.columns else None)
                if lat_col and lon_col:
                    gdf = gpd.GeoDataFrame(
                        merged,
                        geometry=gpd.points_from_xy(merged[lon_col], merged[lat_col]),
                        crs='EPSG:4326'
                    )
                    os.makedirs(os.path.dirname(summary_geojson_path), exist_ok=True)
                    gdf.to_file(summary_geojson_path, driver='GeoJSON')
                    print(f"Wrote VWC observation summary GeoJSON: {summary_geojson_path}")
                else:
                    print("Warning: latitude/longitude not found; skipping VWC summary GeoJSON.")


if __name__ == '__main__':
    home_ = os.path.expanduser('~')
    root_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'soil_potential_obs', 'mt_mesonet')
    vwc_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'vwc_timeseries', 'mt_mesonet')

    metadata_csv_ = os.path.join(root_, 'station_metadata.csv')

    vwc_timeseries_ = os.path.join(vwc_, 'preprocessed_by_station')
    vwc_obs_csv_ = os.path.join(vwc_, 'vwc_observation_summary.csv')
    vwc_obs_geojson_ = os.path.join(vwc_, 'vwc_observation_summary.geojson')
    os.makedirs(vwc_, exist_ok=True)
    get_all_station_timeseries(
        metadata_csv_path=metadata_csv_,
        output_dir=vwc_timeseries_,
        frequency='daily',
        summary_csv_path=vwc_obs_csv_,
        summary_geojson_path=vwc_obs_geojson_,
    )

# ========================= EOF ====================================================================
