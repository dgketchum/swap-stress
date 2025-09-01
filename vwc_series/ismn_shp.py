import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import geopandas as gpd
import pandas as pd
from ismn.interface import ISMN_Interface

# Reuse the existing depth helper, do not reimplement
from vwc_series.ismn_vwc import _depth_from_meta

_DS_GLOBAL = None


def _init_worker(ismn_path, networks):
    global _DS_GLOBAL
    _DS_GLOBAL = ISMN_Interface(ismn_path, network=networks, parallel=True)


def _collect_meta_for_dataset(idx, min_depth_m=None, max_depth_m=None):
    """
    Worker function: read metadata for a dataset id and return station-level pieces
    plus a representative depth (mid-depth) for counting.
    """
    global _DS_GLOBAL
    ts, meta = _DS_GLOBAL.read_ts(idx, return_meta=True)

    station = meta['station']['val']
    network = meta['network']['val']
    station_uid = f"{network}:{station}"

    # Mid-depth in meters using shared helper
    dmid_m = _depth_from_meta(meta)

    # Optional filtering by depth when computing over ids
    if min_depth_m is not None and (dmid_m is None or dmid_m < min_depth_m):
        return None
    if max_depth_m is not None and (dmid_m is None or dmid_m > max_depth_m):
        return None

    # Flatten station metadata values; skip instrument details (we only count depths)
    flat_vals = {}
    for key, sub in meta.items():
        try:
            if key == 'instrument':
                continue
            if isinstance(sub, dict) and 'val' in sub:
                flat_vals[key] = sub.get('val')
        except Exception:
            continue

    # Latitude/longitude are essential; attempt to read from meta
    lat = None
    lon = None
    try:
        lat = float(meta['latitude']['val']) if meta['latitude']['val'] is not None else None
        lon = float(meta['longitude']['val']) if meta['longitude']['val'] is not None else None
    except Exception:
        pass

    return {
        'station_uid': station_uid,
        'lat': lat,
        'lon': lon,
        'meta_vals': flat_vals,
        'depth_m': dmid_m,
    }


def build_ismn_station_metadata_shapefile(
        ismn_path,
        out_shp,
        out_csv=None,
        networks=None,
        min_depth_m=None,
        max_depth_m=None,
        num_workers=None,
        show_progress=True,
):
    """
    Build a per-station shapefile aggregating ISMN metadata, with a simplified
    depth summary. Rather than list instruments and depths, includes only
    'depth_ct' = number of unique mid-depths observed for that station.

    - Aggregates over all soil_moisture dataset ids (optionally filtered by depth).
    - Per-station attributes include all available metadata 'val' fields that are
      stable for the station (instrument fields are excluded).
    - Geometry is built from latitude/longitude (EPSG:4326).
    """

    if not networks:
        networks = [d for d in os.listdir(ismn_path)]

    # Discover dataset ids (optionally with depth filtering)
    ds = ISMN_Interface(ismn_path, network=networks, parallel=True)
    ids = ds.get_dataset_ids(variable='soil_moisture', min_depth=min_depth_m, max_depth=max_depth_m)
    workers = num_workers or 1

    # Accumulate station-level records
    station_meta = {}
    station_depths = defaultdict(set)

    iterator = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(ismn_path, networks)) as ex:
        futures = [ex.submit(_collect_meta_for_dataset, idx, min_depth_m, max_depth_m) for idx in ids]
        iterator = as_completed(futures)
        if show_progress:
            try:
                from tqdm import tqdm  # lazy import
                iterator = tqdm(iterator, total=len(futures), desc='Collecting ISMN metadata')
            except Exception:
                pass

        for f in iterator:
            try:
                item = f.result()
            except Exception:
                item = None
            if not item:
                continue

            uid = item['station_uid']
            # Merge meta values (first wins; assume station-level consistency)
            if uid not in station_meta:
                station_meta[uid] = {
                    **item['meta_vals'],
                    'station_uid': uid,
                    'latitude': item['lat'],
                    'longitude': item['lon'],
                }
            else:
                # Backfill missing fields if any
                rec = station_meta[uid]
                for k, v in item['meta_vals'].items():
                    if k not in rec or rec[k] in (None, ''):
                        rec[k] = v
                if rec.get('latitude') is None and item['lat'] is not None:
                    rec['latitude'] = item['lat']
                if rec.get('longitude') is None and item['lon'] is not None:
                    rec['longitude'] = item['lon']

            # Collect depth for counting
            if item['depth_m'] is not None:
                station_depths[uid].add(round(float(item['depth_m']), 4))

    # Normalize to records with depth_ct and geometry
    records = []
    for uid, meta_vals in station_meta.items():
        rec = dict(meta_vals)
        rec['depth_ct'] = int(len(station_depths.get(uid, set())))
        records.append(rec)

    if not records:
        raise RuntimeError('No station metadata collected; check inputs and filters.')

    # Build GeoDataFrame
    df = pd.DataFrame.from_records(records)
    if 'longitude' not in df.columns or 'latitude' not in df.columns:
        raise RuntimeError('Missing latitude/longitude in collected metadata; cannot build geometry.')

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['longitude'], df['latitude']),
        crs='EPSG:4326',
    )

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(out_shp))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Write shapefile
    gdf.to_file(out_shp)

    # Write CSV (drop geometry). Derive default path if not provided.
    if out_csv is None:
        root, _ = os.path.splitext(out_shp)
        out_csv = root + '.csv'
    if os.path.dirname(os.path.abspath(out_csv)):
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    gdf.drop(columns='geometry').to_csv(out_csv, index=False)

    return {
        'n_stations': len(gdf),
        'out_shp': out_shp,
        'out_csv': out_csv,
        'columns': list(gdf.columns),
    }


if __name__ == '__main__':
    # Example usage; adjust paths as needed.
    home = os.path.expanduser('~')
    in_ = os.path.join(home, 'data', 'IrrigationGIS', 'soils', 'vwc_timeseries', 'ismn', 'ismn_db')
    out_shp_ = os.path.join(home, 'data', 'IrrigationGIS', 'soils', 'vwc_timeseries', 'ismn', 'ismn_stations.shp')
    out_csv_ = None  # defaults to same basename with .csv
    res = build_ismn_station_metadata_shapefile(
        ismn_path=in_,
        out_shp=out_shp_,
        out_csv=out_csv_,
        networks=None,
        min_depth_m=0.0,
        max_depth_m=2.0,
        num_workers=1,
    )
    print(res)
