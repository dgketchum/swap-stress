import os
import pandas as pd
import numpy as np
from utils.gshp import sanitize_profile_id

from tqdm import tqdm

MPA_TO_CM = 10197.16


def _standardize_depth(d, depth_col=None):
    if depth_col and depth_col in d.columns:
        d = d.rename(columns={depth_col: 'depth'})
        return d
    for c in ('depth', 'Depth [cm]', 'Depth_cm', 'stationDepth [cm]'):
        if c in d.columns:
            d = d.rename(columns={c: 'depth'}) if c != 'depth' else d
            return d
    d['depth'] = 0
    return d


def standardize_reesh(df, depth_col=None):
    d = df.copy()
    if 'MPa_Abs' not in d.columns or 'Vol_Water' not in d.columns:
        raise ValueError("Expected columns 'MPa_Abs' and 'Vol_Water'")
    # MPa -> cm of water; Vol_Water is percent -> fraction
    d['suction'] = np.abs(d['MPa_Abs'].astype(float).values) * MPA_TO_CM
    d['theta'] = d['Vol_Water'].astype(float).values / 100.0
    d = _standardize_depth(d, depth_col)
    # Prefer Sample_ID as primary identifier if present, otherwise fall back to Site
    if 'Sample_ID' in d.columns:
        d['name'] = d['Sample_ID']
    elif 'Site' in d.columns:
        d['name'] = d['Site']
    keep_extra = [c for c in ('name', 'Sample_ID', 'Site', 'uid', 'profile_id') if c in d.columns]
    return d[['suction', 'theta', 'depth'] + keep_extra]


def standardize_mt_mesonet(df, depth_col=None):
    d = df.copy()
    if 'KPA' in d.columns and 'VWC' in d.columns:
        d['suction'] = np.abs(d['KPA'].astype(float).values * 10.19716)
        d['theta'] = d['VWC'].astype(float).values
    elif 'suction_cm' in d.columns and 'theta' in d.columns:
        d['suction'] = np.abs(d['suction_cm'].astype(float).values)
        d['theta'] = d['theta'].astype(float).values
    else:
        raise ValueError("Expected ('KPA','VWC') or ('suction_cm','theta')")
    d = _standardize_depth(d, depth_col)
    if 'name' not in d.columns and 'station' in d.columns:
        d['name'] = d['station']
    return d[['suction', 'theta', 'depth'] + [c for c in ('name', 'uid', 'profile_id') if c in d.columns]]


def standardize_gshp(df, depth_col=None):
    d = df.copy()
    # Prefer GSHP high quality data
    if 'data_flag' in d.columns:
        d = d[d['data_flag'] == 'good quality estimate']
    d = d.dropna(subset=['lab_head_m', 'lab_wrc'])
    d['suction'] = (d['lab_head_m'].astype(float) * 100.0).abs()  # m -> cm
    d['theta'] = d['lab_wrc'].astype(float)
    if 'hzn_bot' in d.columns and 'hzn_top' in d.columns:
        d['depth'] = (d['hzn_bot'].astype(float) + d['hzn_top'].astype(float)) / 2.0
    else:
        d = _standardize_depth(d, depth_col)
    keep = ['suction', 'theta', 'depth']
    # GSHP needs these extra data to be fit according to their approach
    keep += [c for c in ('profile_id', 'SWCC_classes',
                         'sand_tot_psa',
                         'silt_tot_psa',
                         'clay_tot_psa',
                         'db_od',
                         'climate_classes') if c in d.columns]
    return d[keep]


def write_standardized_gshp(soil_csv_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(soil_csv_path, encoding='latin1')
    if 'profile_id' in df.columns:
        df['profile_id'] = df['profile_id'].astype(str).apply(sanitize_profile_id)
    std = standardize_gshp(df)
    stations = 0
    s_min, s_max = np.inf, -np.inf
    t_min, t_max = np.inf, -np.inf
    print(f'writing gshp obs to {out_dir}')
    for pid, r in tqdm(std.groupby('profile_id'), total=len(std.groupby('profile_id'))):
        out_path = os.path.join(out_dir, f'{pid}.csv')
        r[['suction', 'theta', 'depth', 'SWCC_classes',
           'sand_tot_psa',
           'silt_tot_psa',
           'clay_tot_psa',
           'db_od',
           'climate_classes']].to_csv(out_path, index=False)
        stations += 1
        s_min = min(s_min, float(np.nanmin(r['suction'].values)))
        s_max = max(s_max, float(np.nanmax(r['suction'].values)))
        t_min = min(t_min, float(np.nanmin(r['theta'].values)))
        t_max = max(t_max, float(np.nanmax(r['theta'].values)))

    print(f"GSHP standardized: stations={stations}, suction_cm=[{s_min:.3g}, {s_max:.3g}], theta=[{t_min:.3f}, {t_max:.3f}]")


def write_standardized_rosetta(curves_wide_csv, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    dfw = pd.read_csv(curves_wide_csv)
    if 'Index' not in dfw.columns:
        return
    s_min, s_max = np.inf, -np.inf
    t_min, t_max = np.inf, -np.inf
    stations = 0
    for idx, row_df in tqdm(dfw.groupby('Index'), total=dfw['Index'].nunique()):
        cols = row_df.columns[2:]
        h_cols = cols[0::2]
        t_cols = cols[1::2]
        r = row_df.iloc[0]
        recs = []
        for hc, tc in zip(h_cols, t_cols):
            h = r[hc]
            t = r[tc]
            if pd.notna(h) and pd.notna(t):
                recs.append({'suction': abs(float(h)), 'theta': float(t), 'depth': 0})
        d = pd.DataFrame(recs)
        out_path = os.path.join(out_dir, f'{int(idx)}.csv')
        d.to_csv(out_path, index=False)
        if not d.empty:
            stations += 1
            s_min = min(s_min, float(np.nanmin(d['suction'].values)))
            s_max = max(s_max, float(np.nanmax(d['suction'].values)))
            t_min = min(t_min, float(np.nanmin(d['theta'].values)))
            t_max = max(t_max, float(np.nanmax(d['theta'].values)))
    if stations:
        print(f"Rosetta standardized: stations={stations}, suction_cm=[{s_min:.3g}, {s_max:.3g}], theta=[{t_min:.3f}, {t_max:.3f}]")


def write_standardized_mt_mesonet(swp_csv_path, metadata_csv_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for p in [swp_csv_path, metadata_csv_path]:
        if not os.path.exists(p):
            print(f"Error: Source file not found at {p}")
            return
    obs_df = pd.read_csv(swp_csv_path)
    meta_df = pd.read_csv(metadata_csv_path)
    station_col = 'station'
    if station_col not in obs_df.columns or station_col not in meta_df.columns:
        print(f"Error: Join column '{station_col}' not found in one or both files.")
        return
    merged = pd.merge(obs_df, meta_df, on=station_col, how='left')
    s_min, s_max = np.inf, -np.inf
    t_min, t_max = np.inf, -np.inf
    stations = 0
    for station, r in tqdm(merged.groupby('station'), total=merged['station'].nunique()):
        d = standardize_mt_mesonet(r, depth_col='Depth [cm]')
        out_path = os.path.join(out_dir, f'{station}.csv')
        d.to_csv(out_path, index=False)
        stations += 1
        s_min = min(s_min, float(np.nanmin(d['suction'].values)))
        s_max = max(s_max, float(np.nanmax(d['suction'].values)))
        t_min = min(t_min, float(np.nanmin(d['theta'].values)))
        t_max = max(t_max, float(np.nanmax(d['theta'].values)))
    if stations:
        print(f"MT Mesonet standardized: stations={stations}, suction_cm=[{s_min:.3g}, {s_max:.3g}], theta=[{t_min:.3f}, {t_max:.3f}]")


def write_standardized_reesh(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    s_min, s_max = np.inf, -np.inf
    t_min, t_max = np.inf, -np.inf
    stations = 0
    files = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if '_SoilWaterRetentionCurves.csv' in f]
    for f in files:
        if not f.endswith('.csv'):
            continue
        p = os.path.join(in_dir, f)
        df = pd.read_csv(p)

        site_id = df.iloc[0]['Site']

        if 'Sample_ID' not in df.columns:
            continue

        if df['Site'].nunique() > 1:
            raise ValueError

        for plot_id, r in tqdm(df.groupby('Plot'), total=df['Plot'].nunique()):
            d = standardize_reesh(r, depth_col='Depth_cm')
            out_path = os.path.join(out_dir, f'{site_id}_{plot_id}.csv')
            d.to_csv(out_path, index=False)
            stations += 1
            s_min = min(s_min, float(np.nanmin(d['suction'].values)))
            s_max = max(s_max, float(np.nanmax(d['suction'].values)))
            t_min = min(t_min, float(np.nanmin(d['theta'].values)))
            t_max = max(t_max, float(np.nanmax(d['theta'].values)))

    if stations:
        print(f"ReESH standardized: stations={stations}, suction_cm=[{s_min:.3g}, {s_max:.3g}], theta=[{t_min:.3f}, {t_max:.3f}]")


if __name__ == '__main__':
    home_ = os.path.expanduser('~')

    run_rosetta = False
    run_mt_mesonet = False
    run_reesh = False

    if run_rosetta:
        root_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'rosetta', 'training_data')
        props_csv_ = os.path.join(root_, 'rosetta_properties.csv')
        curves_wide_csv_ = os.path.join(root_, 'rosetta_curves_wide.csv')
        out_dir_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'soil_potential_obs', 'preprocessed', 'rosetta')
        write_standardized_rosetta(curves_wide_csv_, out_dir_)

    if run_mt_mesonet:
        root_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'soil_potential_obs', 'mt_mesonet')
        swp_csv_ = os.path.join(root_, 'swp.csv')
        metadata_csv_ = os.path.join(root_, 'station_metadata.csv')
        out_dir_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'soil_potential_obs', 'preprocessed', 'mt_mesonet')
        write_standardized_mt_mesonet(swp_csv_, metadata_csv_, out_dir_)

    if run_reesh:
        in_dir_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'soil_potential_obs', 'reesh')
        out_dir_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'soil_potential_obs', 'preprocessed', 'reesh')
        write_standardized_reesh(in_dir_, out_dir_)

# ========================= EOF ====================================================================
