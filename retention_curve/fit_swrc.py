import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from retention_curve.swrc import SWRC
from site_modeling.prep import load_vwc_series, find_ameriflux_file, load_ameriflux_halfhourly


def _csv_files(d):
    return [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.csv')]


def _fit_file(args):
    path, out_dir, method, overwrite, ts_source = args
    name = os.path.splitext(os.path.basename(path))[0]
    out_name = f"{name}.json"
    out_path = os.path.join(out_dir, out_name)

    # Skip work if output exists and overwrite is False
    if not overwrite and os.path.exists(out_path):
        print(f'{out_path} already exists, skipping.')
        return out_name

    df = pd.read_csv(path)
    fitter = SWRC(df=df)
    is_bayes = str(method).lower() in {'bayes', 'fit_bayes', 'bayesian'}

    theta_r_cap = None
    theta_s_floor = None
    theta_s_upper = 0.65
    if is_bayes and ts_source is not None:
        src_type, src_path = ts_source
        if src_type == 'mt_mesonet' and src_path:
            s = load_vwc_series(name, src_path, preferred_depth_cm=None)
            theta_r_cap = float(np.nanmin(s.values))
            theta_s_floor = float(np.nanmax(s.values))
        elif src_type == 'reesh' and src_path:
            amf_key = name.split('_')[0]
            amf_fp = find_ameriflux_file(src_path, amf_key, period='HH')
            if amf_fp is not None:
                daily_vwc, _ = load_ameriflux_halfhourly(amf_fp)
                if daily_vwc is not None and not daily_vwc.empty:
                    theta_r_cap = float(np.nanmin(daily_vwc.values))
                    theta_s_floor = float(np.nanmax(daily_vwc.values))
        # Ensure feasible bounds for theta_s
        if theta_s_floor is not None:
            if not np.isfinite(theta_s_floor):
                theta_s_floor = None
            else:
                eps = 1e-3
                theta_s_floor = max(0.0, min(theta_s_floor, 0.99))
                if theta_s_floor >= theta_s_upper - eps:
                    theta_s_upper = min(0.99, theta_s_floor + 0.02)

    if is_bayes:
        eps = 1e-5
        cap = None if theta_r_cap is None else max(1e-6, float(theta_r_cap) - eps)
        fitter.fit_bayesian(draws=2000, tune=1000, theta_r_cap=cap, theta_s_floor=theta_s_floor, theta_s_upper=theta_s_upper)
        fitter.save_bayes_results(output_dir=out_dir, output_filename=out_name)
    else:
        fitter.fit(report=False, method=method)
        fitter.save_results(output_dir=out_dir, output_filename=out_name)
    print(f'fit {name}')
    return out_name


def fit_standardized_dir(in_dir, out_dir, method='slsqp', workers=12, overwrite=False, ts_source=None):
    os.makedirs(out_dir, exist_ok=True)
    files = _csv_files(in_dir)
    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_fit_file, [(p, out_dir, method, overwrite, ts_source) for p in files]))
    else:
        for p in files:
            _fit_file((p, out_dir, method, overwrite, ts_source))


if __name__ == '__main__':
    home_ = os.path.expanduser('~')
    root = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'soil_potential_obs')

    run_rosetta = False
    run_mt_mesonet = False
    run_reesh = True

    method = 'bayes'

    out_root = os.path.join(root, 'curve_fits')

    if run_rosetta:
        in_dir_ = os.path.join(root, 'preprocessed', 'rosetta')
        out_dir_ = os.path.join(out_root, 'rosetta', method)
        if os.path.exists(in_dir_):
            fit_standardized_dir(in_dir_, out_dir_, method=method)

    if run_mt_mesonet:
        in_dir_ = os.path.join(root, 'preprocessed', 'mt_mesonet')
        out_dir_ = os.path.join(out_root, 'mt_mesonet', method)
        vwc_dir_ts_ = os.path.join(root, '..', 'vwc_timeseries', 'mt_mesonet', 'preprocessed_by_station')
        ts_src = ('mt_mesonet', vwc_dir_ts_) if os.path.exists(vwc_dir_ts_) else None
        if os.path.exists(in_dir_):
            fit_standardized_dir(in_dir_, out_dir_, method=method, workers=12, ts_source=ts_src)

    if run_reesh:
        in_dir_ = os.path.join(root, 'preprocessed', 'reesh')
        out_dir_ = os.path.join(out_root, 'reesh', method)
        amf_root_ = os.path.join(os.path.expanduser('~'), 'data', 'IrrigationGIS', 'climate', 'ameriflux', 'amf_new')
        ts_src = ('reesh', amf_root_) if os.path.exists(amf_root_) else None
        if os.path.exists(in_dir_):
            fit_standardized_dir(in_dir_, out_dir_, method=method, ts_source=ts_src, overwrite=True, workers=16)

# ========================= EOF ====================================================================
