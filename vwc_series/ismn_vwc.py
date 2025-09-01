import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import pandas as pd
from ismn.interface import ISMN_Interface
from tqdm import tqdm


def _depth_from_meta(m):
    try:
        dfrom, dto = m['instrument']['depth_from'], m['instrument']['depth_to']
        if dfrom is None and dto is None:
            return None
        if dfrom is None:
            dfrom = dto
        if dto is None:
            dto = dfrom
        d = float(dfrom) if dfrom is not None else float(dto)
        d2 = float(dto)
        return float(0.5 * (d + d2))
    except Exception:
        return None


def _plot_vwc_with_flags(ts, meta_dict, out_png, vwc_col='soil_moisture', flag_col='soil_moisture_flag'):
    if ts is None or ts.empty or vwc_col not in ts.columns:
        return

    # Prefer soil_moisture_flag; fallback to soil_moisture_orig_flag; else None
    if flag_col not in ts.columns:
        if 'soil_moisture_orig_flag' in ts.columns:
            flag_col = 'soil_moisture_orig_flag'
        else:
            flag_col = None

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 4))

    if flag_col is None:
        ax.plot(ts.index, ts[vwc_col], color='#1f77b4', lw=1)
    else:
        # Color segments by flag: green for 'G', gray for others
        good = ts[vwc_col].where(ts[flag_col] == 'G')
        bad = ts[vwc_col].where(ts[flag_col] != 'G')
        ax.plot(ts.index, good, color='#2ca02c', lw=1, label='Good (G)')
        ax.plot(ts.index, bad, color='#7f7f7f', lw=1, label='Other flags')
        ax.legend(loc='best')

    station = meta_dict['station']['val']
    network = meta_dict['network']['val']
    dmid = _depth_from_meta(meta_dict)
    title = f"{network}:{station} — {dmid:.2f} m" if dmid is not None else f"{network}:{station}"
    ax.set_title(title)
    ax.set_ylabel('VWC (m3/m3)')
    ax.set_xlabel('Date')
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    try:
        plt.savefig(out_png, dpi=200)
    finally:
        plt.close(fig)


def _daily_series(ts, prefer_flagged=True, vwc_col='soil_moisture', flag_col='soil_moisture_flag'):
    df = ts.copy()
    if vwc_col not in df.columns:
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try to coerce index
        if 'date_time' in df.columns:
            df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
            df = df.set_index('date_time')
        else:
            try:
                df.index = pd.to_datetime(df.index, errors='coerce')
            except Exception:
                return None
    df = df.sort_index()

    if prefer_flagged and flag_col in df.columns:
        good = df[df[flag_col] == 'G'][vwc_col].resample('D').mean()
        # Only fill with all-data daily mean where no good obs exist
        daily_all = df[vwc_col].resample('D').mean()
        daily = good.combine_first(daily_all)
    else:
        daily = df[vwc_col].resample('D').mean()
    return daily


_DS_GLOBAL = None


def _init_worker(ismn_path, networks):
    global _DS_GLOBAL
    _DS_GLOBAL = ISMN_Interface(ismn_path, network=networks, parallel=True)


def _acquire_lock(lock_path, retries=200, sleep_s=0.05):
    for _ in range(retries):
        try:
            # Exclusive create. Fails if exists.
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            time.sleep(sleep_s)
    return False


def _release_lock(lock_path):
    try:
        if os.path.exists(lock_path):
            os.remove(lock_path)
    except Exception:
        pass


def _safe_name(s):
    return str(s).replace(':', '_').replace('/', '_').replace(' ', '_')


def _process_dataset_id(idx, out_dir, min_depth_m, max_depth_m, plot):
    global _DS_GLOBAL
    ts, meta = _DS_GLOBAL.read_ts(idx, return_meta=True)

    station = meta['station']['val']
    network = meta['network']['val']
    station_uid = f"{network}:{station}"

    dmid_m = _depth_from_meta(meta)
    if dmid_m is None or dmid_m < min_depth_m or dmid_m > max_depth_m:
        return {'station': station_uid, 'written': False}

    depth_cm = int(dmid_m * 100)
    # Plot per sensor
    if plot:
        plots_dir = os.path.join(out_dir, 'plots')
        safe_station = _safe_name(station)
        safe_network = _safe_name(network)
        plot_fp = os.path.join(plots_dir, safe_network, f"{safe_station}_{depth_cm}cm.png")
        _plot_vwc_with_flags(ts, meta, plot_fp)

    # Daily series
    daily = _daily_series(ts)
    if daily is None or daily.empty:
        return {'station': station_uid, 'written': False}

    col = f"soil_vwc_{depth_cm}"
    df_new = daily.to_frame(col)
    df_new.index.name = 'datetime'

    station_dir = os.path.join(out_dir, 'preprocessed_by_station')
    os.makedirs(station_dir, exist_ok=True)
    safe_station_id = _safe_name(station_uid)
    out_fp = os.path.join(station_dir, f"{safe_station_id}.parquet")
    lock_fp = out_fp + '.lock'

    # Acquire simple file lock to avoid concurrent writes
    if not _acquire_lock(lock_fp):
        # Give up if unable to obtain lock
        return {'station': station_uid, 'written': False}
    try:
        if os.path.exists(out_fp):
            # Merge with existing
            out_df = pd.read_parquet(out_fp)
            if 'datetime' not in out_df.columns:
                return {'station': station_uid, 'written': False}
            out_df['datetime'] = pd.to_datetime(out_df['datetime'], errors='coerce')
            existing = out_df.set_index('datetime').sort_index()

            if col in existing.columns:
                combined = pd.concat([existing[col], df_new[col]], axis=1).mean(axis=1)
                existing[col] = combined
            else:
                existing = existing.join(df_new, how='outer')

            existing = existing.sort_index()
            existing.index.name = 'datetime'
            out_df = existing.reset_index()
            out_df['station'] = station_uid
            # Keep meta consistent (overwrite with current meta to be safe)
            out_df['latitude'] = meta['latitude']['val']
            out_df['longitude'] = meta['longitude']['val']
        else:
            # Create new
            existing = df_new.sort_index()
            existing.index.name = 'datetime'
            out_df = existing.reset_index()
            out_df.insert(1, 'station', station_uid)
            out_df['latitude'] = meta['latitude']['val']
            out_df['longitude'] = meta['longitude']['val']

        # Reorder columns
        vwc_cols = [c for c in out_df.columns if c.startswith('soil_vwc_')]
        out_df = out_df[['datetime', 'station', 'latitude', 'longitude'] + sorted(vwc_cols, key=lambda x: int(x.split('_')[-1]))]
        out_df.to_parquet(out_fp, index=False)
        return {'station': station_uid, 'written': True}
    finally:
        _release_lock(lock_fp)


def build_ismn_vwc_series(
        ismn_path,
        out_dir,
        min_depth_m=0.0,
        max_depth_m=2.0,
        networks=None,
        plot=True,
        max_ids=None,
        num_workers=None,
):
    """
    Convert ISMN soil moisture to per-station daily VWC parquet files and optional PNG plots per sensor.

    - Per-sensor plots show VWC colored by quality flag (green for 'G', gray for others).
    - Per-station parquet has columns: 'datetime', 'station', 'latitude', 'longitude', and soil_vwc_{depth_cm}.
    - Station id is 'network:station' to avoid name collisions.
    """

    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, 'plots')
    station_dir = os.path.join(out_dir, 'preprocessed_by_station')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(station_dir, exist_ok=True)

    if not networks:
        networks = [d for d in os.listdir(ismn_path)]

    ds = ISMN_Interface(ismn_path, network=networks, parallel=True)

    ids = ds.get_dataset_ids(variable='soil_moisture', min_depth=min_depth_m, max_depth=max_depth_m)
    total = len(ids)
    if max_ids is not None:
        ids = ids[:max_ids]

    # Parallel single-pass processing: read meta/ts and write parquet per station incrementally
    workers = num_workers or 1
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(ismn_path, networks)) as ex:
        futures = [ex.submit(_process_dataset_id, idx, out_dir, min_depth_m, max_depth_m, plot) for idx in ids]
        for f in tqdm(as_completed(futures), total=len(futures), desc='Processing ISMN datasets'):
            try:
                results.append(f.result())
            except Exception:
                results.append({'station': None, 'written': False})

    # Derive n_stations from written parquet files
    try:
        n_station_files = len([fn for fn in os.listdir(station_dir) if fn.endswith('.parquet')])
    except Exception:
        n_station_files = 0

    return {
        'n_ids_total': total,
        'n_ids_processed': len(ids),
        'n_stations': n_station_files,
        'out_dir': out_dir,
        'station_dir': station_dir,
        'plots_dir': plots_dir,
    }


if __name__ == '__main__':
    home = os.path.expanduser('~')
    in_ = os.path.join(home, 'data', 'IrrigationGIS', 'soils', 'vwc_timeseries', 'ismn', 'ismn_db')
    out_ = os.path.join(home, 'data', 'IrrigationGIS', 'soils', 'vwc_timeseries', 'ismn', 'processed_time_series')
    try:
        summary = build_ismn_vwc_series(in_, out_dir=out_, min_depth_m=0.0, max_depth_m=2.0,
                                        networks=None, plot=True, num_workers=1)
        print(summary)
    except Exception as e:
        print(f"ISMN processing failed: {e}")
