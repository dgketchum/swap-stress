"""
Note: This module can call external R code (rwrap/fit_new_samples.R) to fit
GSHP pedotransfer functions. An R-enabled environment with required R packages
is needed and is not part of this project's Python requirements. Ensure the
Rscript on PATH belongs to that environment, or place local soilhypfit R files
under ~/PycharmProjects/GSHP-database/soilhypfit/R/.
"""

import os
import subprocess
import tempfile
import pandas as pd

BAR_TO_CM = 1019.72
WRC_MAP = {
    'water_retention_6_hundredths': 0.06,
    'water_retention_10th_bar': 0.10,
    'water_retention_third_bar': 0.33,
    'water_retention_1_bar': 1.0,
    'water_retention_2_bar': 2.0,
    'water_retention_5_bar_sieve': 5.0,
    'water_retention_15_bar': 15.0,
}


def ncss_to_standardized(df):
    id_cols = [
        'labsampnum', 'pedon_key', 'hzn_top', 'hzn_bot', 'hzn_mid_cm',
        'sand_total', 'silt_total', 'clay_total',
        'bulk_density_oven_dry',
    ]
    wr_cols = [c for c in WRC_MAP.keys() if c in df.columns]

    d0 = df[id_cols + wr_cols].copy()
    m = d0.melt(id_vars=id_cols, value_vars=wr_cols, var_name='wr_col', value_name='wr_val')
    m = m.dropna(subset=['wr_val'])

    if 'pedon_key' in m.columns:
        m['profile_id'] = m['pedon_key'].astype(str)
    else:
        m['profile_id'] = m['labsampnum'].astype(str)

    if 'hzn_mid_cm' in m.columns:
        m['depth_cm'] = m['hzn_mid_cm']
        if 'hzn_top' in m.columns and 'hzn_bot' in m.columns:
            m['depth_cm'] = m['depth_cm'].fillna((m['hzn_top'].astype(float) + m['hzn_bot'].astype(float)) / 2.0)
    elif 'hzn_top' in m.columns and 'hzn_bot' in m.columns:
        m['depth_cm'] = (m['hzn_top'].astype(float) + m['hzn_bot'].astype(float)) / 2.0
    m['suction_cm'] = m['wr_col'].map(WRC_MAP).astype(float) * BAR_TO_CM

    # NCSS water retention typically reported as gravimetric percent
    grav = m['wr_val'].astype(float) / 100.0
    m['theta'] = grav * m['bulk_density_oven_dry'].astype(float)  # uses oven-dry bulk density

    m['db_od'] = m['bulk_density_oven_dry']
    m['sand_tot_psa'] = m['sand_total']
    m['silt_tot_psa'] = m['silt_total']
    m['clay_tot_psa'] = m['clay_total']
    m['source_db'] = 'NCSS'

    keep = [
        'profile_id', 'depth_cm', 'suction_cm', 'theta',
        'db_od', 'sand_tot_psa', 'silt_tot_psa', 'clay_tot_psa', 'source_db',
    ]
    out = m[keep].copy()

    # Derive SWCC coverage classes by profile
    sm = out.copy()
    sm['suction_m'] = sm['suction_cm'].astype(float) / 100.0
    g = sm.groupby('profile_id')['suction_m']
    has_wet = g.min() <= 0.01
    has_dry = g.max() >= 150.0
    cls = pd.Series('NWND', index=has_wet.index)
    cls.loc[has_wet & has_dry] = 'YWYD'
    cls.loc[has_wet & ~has_dry] = 'YWND'
    cls.loc[~has_wet & has_dry] = 'NWYD'
    cls = cls.rename('SWCC_classes')

    out = out.merge(cls, left_on='profile_id', right_index=True, how='left')
    out = out.sort_values(['profile_id', 'suction_cm']).reset_index(drop=True)
    return out


def load_ncss_parquet(parquet_path):
    df = pd.read_parquet(parquet_path)
    return df


def write_standardized(df, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)


def standardized_to_rfit(df_std, min_obs_per_profile=4):
    """
    Convert standardized NCSS long-form measurements into the column schema expected by
    rwrap/fit_new_samples.R:
      - layer_id: identifier used by the R fitter (use profile_id here)
      - lab_wrc: volumetric water content (theta)
      - lab_head_m: suction in meters (convert from cm)
    Keep auxiliary columns used for bounds/PTF if present:
      db_od, sand_tot_psa, silt_tot_psa, clay_tot_psa, source_db, SWCC_classes
    """
    req = ['profile_id', 'suction_cm', 'theta']
    missing = [c for c in req if c not in df_std.columns]
    if missing:
        raise ValueError(f"standardized frame missing required columns: {missing}")

    d = df_std.copy()

    if min_obs_per_profile and 'profile_id' in d.columns:
        cnt = d.groupby('profile_id').size()
        keep_ids = cnt[cnt >= int(min_obs_per_profile)].index
        d = d[d['profile_id'].isin(keep_ids)]

    out = pd.DataFrame()
    out['layer_id'] = d['profile_id'].astype(str)
    out['lab_wrc'] = pd.to_numeric(d['theta'], errors='coerce')
    out['lab_head_m'] = pd.to_numeric(d['suction_cm'], errors='coerce') / 100.0

    # Pass-through auxiliary columns if available
    for col in ['db_od', 'sand_tot_psa', 'silt_tot_psa', 'clay_tot_psa', 'source_db', 'SWCC_classes']:
        if col in d.columns:
            out[col] = d[col]
    return out


def write_rfit_csv(df_rfit, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_rfit.to_csv(out_csv, index=False)


def run_fit_new_samples(samples_csv, ptf_rds, out_dir,
                        rscript_path=os.path.expanduser(
                            '~/PycharmProjects/GSHP-database/rwrap/fit_new_samples.R'
                        ),
                        min_obs_per_profile=4):
    """
    Call the R script fit_new_samples.R with the prepared CSV.
    Returns the process return code.
    """
    samples_for_run = samples_csv
    if min_obs_per_profile and min_obs_per_profile > 1:
        try:
            df_in = pd.read_csv(samples_csv)
            if 'SWCC_classes' in df_in.columns:
                df_in = df_in[df_in['SWCC_classes'] == 'YWYD']
                if df_in.empty:
                    rc = 0
                    return rc
            if 'layer_id' in df_in.columns:
                counts = df_in.groupby('layer_id').size()
                keep = counts[counts >= int(min_obs_per_profile)].index
                df_f = df_in[df_in['layer_id'].isin(keep)]
                tmp = tempfile.NamedTemporaryFile(prefix='rfit_samples_', suffix='.csv', delete=False)
                tmp.close()
                df_f.to_csv(tmp.name, index=False)
                samples_for_run = tmp.name
        except Exception:
            raise ValueError('Problem fitting the new samples')

    cmd = ['Rscript', rscript_path, ptf_rds, samples_for_run, out_dir]
    env = os.environ.copy()
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    except FileNotFoundError as e:
        raise RuntimeError('Rscript not found on PATH. Please install R and ensure Rscript is available.') from e
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)
    rc = proc.returncode
    if samples_for_run != samples_csv and os.path.exists(samples_for_run):
        try:
            os.remove(samples_for_run)
        except OSError:
            pass
    return rc


if __name__ == '__main__':
    base_dir = os.path.expanduser('~/data/IrrigationGIS/soils/soil_potential_obs/ncss_labdatasqlite')
    in_parquet = os.path.join(base_dir, 'ncss_selection.parquet')
    out_csv = os.path.join(base_dir, 'standardized_ncss.csv')
    out_csv_rfit = os.path.join(base_dir, 'ncss_for_fit_new.csv')

    df_ = load_ncss_parquet(in_parquet)
    std_ = ncss_to_standardized(df_)
    write_standardized(std_, out_csv)

    rfit_df = standardized_to_rfit(std_)
    write_rfit_csv(rfit_df, out_csv_rfit)

    gshp_fit = os.path.expanduser('~/data/IrrigationGIS/soils/soil_potential_obs/curve_fits/gshp/rfit')
    ptf_rds = os.path.join(gshp_fit, 'ptf_model.rds')
    fit_out_dir = os.path.join(base_dir, 'ncss_fit_new_out')
    rscript_path = os.path.expanduser('~/PycharmProjects/GSHP-database/rwrap/fit_new_samples.R')

    if ptf_rds:
        if not os.path.exists(ptf_rds):
            print(f"PTF model RDS not found at {ptf_rds}; skipping fit_new_samples.R call.")
        elif not os.path.exists(rscript_path):
            print(f"fit_new_samples.R not found at {rscript_path}; skipping R call.")
        else:
            os.makedirs(fit_out_dir, exist_ok=True)
            rc = run_fit_new_samples(out_csv_rfit, ptf_rds, fit_out_dir, rscript_path=rscript_path)
            if rc != 0:
                print(f"fit_new_samples.R exited with code {rc}")
    else:
        print('PTF_MODEL_RDS not set; wrote CSVs only. Set env var to run R fitter.')

# ========================= EOF ====================================================================
