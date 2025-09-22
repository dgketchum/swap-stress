import os
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from retention_curve.swrc import SWRC


def get_station_files(in_dirs):
    files = []
    for d in in_dirs:
        if os.path.exists(d):
            files.extend([p for p in glob(os.path.join(d, '*.csv'))])
            files.extend([p for p in glob(os.path.join(d, '*.parquet'))])
    return files


def _van_genuchten_model(psi, theta_r, theta_s, alpha, n):
    if n <= 1:
        return np.full_like(psi, np.nan)
    m = 1 - 1 / n
    psi_safe = np.maximum(psi, 1e-9)
    term = 1 + (alpha * psi_safe) ** n
    return theta_r + (theta_s - theta_r) / (term) ** m


def _bayes_summary_row(station, depth, trace):
    post = trace.posterior
    tr = float(post['theta_r'].values.mean())
    ts = float(post['theta_s'].values.mean())
    al = float(post['alpha'].values.mean())
    nn = float(post['n'].values.mean())
    tr_se = float(post['theta_r'].values.std())
    ts_se = float(post['theta_s'].values.std())
    al_se = float(post['alpha'].values.std())
    nn_se = float(post['n'].values.std())
    row = pd.DataFrame([{
        'method': 'bayesian', 'success': True, 'AIC': np.nan, 'BIC': np.nan, 'time (s)': np.nan,
        'theta_r': tr, 'theta_r_stderr': tr_se,
        'theta_s': ts, 'theta_s_stderr': ts_se,
        'alpha': al, 'alpha_stderr': al_se,
        'n': nn, 'n_stderr': nn_se,
        'avg_rel_err (%)': np.nan,
        'station': station,
        'depth': depth,
    }])
    return row


def test_fit_methods_across_stations(station_files, results_dir, plots_dir):
    methods_to_test = [
        'least_squares',  # bound-aware local
        'lbfgsb',
        'trust-constr',
        'slsqp',
        'nelder',
        'differential_evolution',
        'bayesian'
    ]

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    all_results = []
    for station_file in station_files:

        station_name = os.path.splitext(os.path.basename(station_file))[0]
        swrc_fitter = SWRC(filepath=station_file, depth_col='depth_cm')
        for depth in swrc_fitter.data_by_depth.keys():
            base_methods = [m for m in methods_to_test if m != 'bayesian']
            res_df = swrc_fitter.test_fit_methods(base_methods, depth=depth)
            if res_df is not None and not res_df.empty:
                res_df['station'] = station_name
                res_df['depth'] = depth
                # append Bayesian summary row using posterior mean/std
                swrc_fitter.fit(method='bayes')
                bt = getattr(swrc_fitter, 'bayes_results', {}).get(depth)
                if bt is not None:
                    bayes_row_ = _bayes_summary_row(station_name, depth, bt)
                    res_df = pd.concat([res_df, bayes_row_], ignore_index=True)
                all_results.append(res_df)

            d = swrc_fitter.data_by_depth[depth]
            d = d.dropna(subset=['suction', 'theta'])
            d = d[np.isfinite(d['suction']) & np.isfinite(d['theta'])]

            fig, ax = plt.subplots(figsize=(7.5, 6.5))
            ax.plot(d['theta'], d['suction'], 'o', ms=5, alpha=0.7, label='Observed')
            psi_min = max(1e-3, float(d['suction'].min()))
            psi_max = float(d['suction'].max())
            psi_smooth = np.logspace(np.log10(psi_min), np.log10(psi_max), 200)

            if res_df is not None and not res_df.empty:
                for _, row in res_df.iterrows():
                    if row.get('method') == 'bayesian':
                        continue  # avoid duplicate line; explicit bayesian plotted below
                    th_r = row.get('theta_r')
                    th_s = row.get('theta_s')
                    al = row.get('alpha')
                    nn = row.get('n')
                    if pd.isna(th_r) or pd.isna(th_s) or pd.isna(al) or pd.isna(nn):
                        continue
                    theta_pred = _van_genuchten_model(psi_smooth, th_r, th_s, al, nn)
                    ax.plot(theta_pred, psi_smooth, lw=2, alpha=0.9, label=row['method'])

            swrc_fitter.fit(method='bayes')
            bt = getattr(swrc_fitter, 'bayes_results', {}).get(depth)
            if bt is not None:
                post = bt.posterior
                tr = float(post['theta_r'].values.mean())
                ts = float(post['theta_s'].values.mean())
                al = float(post['alpha'].values.mean())
                nn = float(post['n'].values.mean())
                theta_pred = _van_genuchten_model(psi_smooth, tr, ts, al, nn)
                ax.plot(theta_pred, psi_smooth, lw=2, alpha=0.9, label='bayesian')

            ax.set_yscale('log')
            ax.set_xlabel('theta')
            ax.set_ylabel('suction (cm)')
            ax.set_title(f'{station_name} depth={depth} cm')
            ax.legend(ncol=2, fontsize=9)
            ax.set_xlim(left=0, right=0.65)
            fig.tight_layout()
            out_png = os.path.join(plots_dir, f'{station_name}_depth{depth}.png')
            fig.savefig(out_png, dpi=200)
            print(f'wrote {out_png}')
            plt.close(fig)

    if not all_results:
        return pd.DataFrame()

    master_df = pd.concat(all_results, ignore_index=True)
    param_cols = ['theta_r', 'theta_s', 'alpha', 'n']
    stderr_cols = [f'{c}_stderr' for c in param_cols]
    dropna_cols = param_cols + stderr_cols
    master_df = master_df.dropna(subset=dropna_cols)
    summary = master_df.groupby('method').agg(
        total_aic=('AIC', 'sum'),
        avg_theta_r_rel_err=('avg_rel_err (%)', 'mean'),
        avg_time=('time (s)', 'mean'),
        num_successes=('success', lambda x: x.sum()),
        num_runs=('success', 'count')
    ).reset_index()
    summary['num_failures'] = summary['num_runs'] - summary['num_successes']
    summary = summary.sort_values(by='total_aic', ascending=True).reset_index(drop=True)
    master_df.to_csv(os.path.join(results_dir, 'all_stations_detailed_fit_results.csv'), index=False)
    summary.to_csv(os.path.join(results_dir, 'overall_method_summary.csv'), index=False)
    return summary


if __name__ == '__main__':
    home_ = os.path.expanduser('~')

    reesh_in_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'soil_potential_obs', 'preprocessed', 'reesh')
    mtm_in_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'soil_potential_obs', 'preprocessed', 'mt_mesonet')

    fits_out_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'soil_potential_obs', 'curve_fits')
    plots_out_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'soil_potential_obs', 'obs_fit_comparison')

    station_files_ = get_station_files([reesh_in_, mtm_in_])
    # station_files_ = [s for s in station_files_ if 'US-MMS_SN' in s]
    test_fit_methods_across_stations(station_files_, fits_out_, plots_out_)

# ========================= EOF ====================================================================
