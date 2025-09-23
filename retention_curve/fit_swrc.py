import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from retention_curve.swrc import SWRC


def _csv_files(d):
    return [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.csv')]


def _fit_file(args):
    path, out_dir, method = args
    name = os.path.splitext(os.path.basename(path))[0]
    out_name = f"{name}.json"
    df = pd.read_csv(path)
    fitter = SWRC(df=df)
    is_bayes = str(method).lower() in {'bayes', 'fit_bayes', 'bayesian'}
    fitter.fit(report=False, method=method)
    if is_bayes:
        fitter.save_bayes_results(output_dir=out_dir, output_filename=out_name)
    else:
        fitter.save_results(output_dir=out_dir, output_filename=out_name)
    return out_name


def fit_standardized_dir(in_dir, out_dir, method='slsqp', workers=12):
    os.makedirs(out_dir, exist_ok=True)
    files = _csv_files(in_dir)
    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_fit_file, [(p, out_dir, method) for p in files]))
    else:
        for p in files:
            _fit_file((p, out_dir, method))


if __name__ == '__main__':
    home_ = os.path.expanduser('~')
    root = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'soil_potential_obs')

    run_rosetta = False
    run_mt_mesonet = True
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
        if os.path.exists(in_dir_):
            fit_standardized_dir(in_dir_, out_dir_, method=method, workers=1)

    if run_reesh:
        in_dir_ = os.path.join(root, 'preprocessed', 'reesh')
        out_dir_ = os.path.join(out_root, 'reesh', method)
        if os.path.exists(in_dir_):
            fit_standardized_dir(in_dir_, out_dir_, method=method)

# ========================= EOF ====================================================================
