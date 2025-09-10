import os

import numpy as np
import pandas as pd

from retention_curve.swrc import SWRC
from retention_curve.gshp_swrc import GshpSWRC


def _csv_files(d):
    return [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.csv')]


def gshp_impute_missing_wet_ptf(csv):
    df = pd.read_csv(csv, encoding='latin1', low_memory=False)
    df = df[df['data_flag'] == 'good quality estimate']
    if 'hzn_bot' in df.columns and 'hzn_top' in df.columns:
        df['depth'] = (df['hzn_bot'].astype(float) + df['hzn_top'].astype(float)) / 2.0
    df['alpha'] = df['alpha'] / 100.0
    df.rename(columns={'thetar': 'theta_r', 'thetas': 'theta_s'}, inplace=True)
    missing_wet_ptf = GshpSWRC.estimate_theta_s_ptf(df)
    return missing_wet_ptf, df


def fit_standardized_dir(in_dir, out_dir, method='slsqp'):
    os.makedirs(out_dir, exist_ok=True)
    files = _csv_files(in_dir)

    for p in files:
        name = os.path.splitext(os.path.basename(p))[0]
        out_name = f"{name}.json"
        df = pd.read_csv(p)

        fitter = SWRC(df=df)
        fitter.fit(report=False, method=method)
        fitter.save_results(output_dir=out_dir, output_filename=out_name)


if __name__ == '__main__':
    home_ = os.path.expanduser('~')
    root = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'soil_potential_obs')

    run_rosetta = True
    run_mt_mesonet = True
    run_reesh = True

    method = 'nelder'

    out_root = os.path.join(root, 'curve_fits')


    if run_rosetta:
        in_dir_ = os.path.join(root, 'preprocessed', 'rosetta')
        out_dir_ = os.path.join(out_root, 'rosetta', method)
        if os.path.exists(in_dir_):
            fit_standardized_dir(in_dir_, out_dir_, method=method)

    if run_mt_mesonet:
        in_dir_ = os.path.join(root, 'preprocessed', 'mt_mesonet')
        out_dir_ = os.path.join(out_root, 'mt_mesonet', method)
        if os.path.exists(in_dir_):
            fit_standardized_dir(in_dir_, out_dir_, method=method)

    if run_reesh:
        in_dir_ = os.path.join(root, 'preprocessed', 'reesh')
        out_dir_ = os.path.join(out_root, 'reesh', method)
        if os.path.exists(in_dir_):
            fit_standardized_dir(in_dir_, out_dir_, method=method)

# ========================= EOF ====================================================================
