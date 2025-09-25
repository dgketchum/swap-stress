import os
import json
import re
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

from retention_curve import EMPIRICAL_TO_ROSETTA_LEVEL_MAP


def van_genuchten(suction, theta_r, theta_s, alpha, n):
    """Van Genuchten-Mualem soil water retention curve function."""
    # Ensure parameters are within a valid physical range
    n = np.maximum(n, 1.001)
    alpha = np.maximum(alpha, 1e-9)
    theta_s = np.maximum(theta_r, theta_s)

    m = 1 - 1 / n
    psi_safe = np.maximum(suction, 1e-9)
    return theta_r + (theta_s - theta_r) / (1 + (alpha * psi_safe) ** n) ** m


def plot_station_comparison(station_id, station_df, output_dir):
    """Generate and save a comparison plot for a single station."""
    station_df = station_df.sort_values('depth')
    for depth, row in station_df.iterrows():

        fig, ax = plt.subplots(figsize=(10, 8))
        suction_range = np.logspace(0, 7, 500)

        # Plot raw data points
        raw_data = row.get('raw_data')
        if raw_data and raw_data.get('suction'):
            ax.scatter(raw_data['theta'], raw_data['suction'], c='k', label='Measured', zorder=10, alpha=0.8)

        # Plot fitted curve
        fit_params = row.get('fit_params')
        if fit_params:
            fit_curve = van_genuchten(suction_range, **fit_params)
            ax.plot(fit_curve, suction_range, label='Fitted', color='blue', zorder=5)

        # Plot Rosetta curve
        rosetta_params = row.get('rosetta_params')
        if rosetta_params:
            _ = rosetta_params.pop('Ks', None)
            ros_curve = van_genuchten(suction_range, **rosetta_params)
            ax.plot(ros_curve, suction_range, label='Rosetta', color='red', linestyle='--', zorder=4)

        # Plot inferred curve
        inferred_params = row.get('inferred_params')
        if inferred_params:
            inf_curve = van_genuchten(suction_range, **inferred_params)
            ax.plot(inf_curve, suction_range, label='Inferred (RF)', color='green', linestyle=':', zorder=4)

        ax.set_yscale('log')
        ax.set_ylabel('Suction [cm]')
        ax.set_xlabel('Volumetric Water Content [cm³/cm³]')
        ax.set_title(f'Station: {station_id}, Depth: {depth} cm')
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.5)
        ax.set_xlim(0, max(0.6, ax.get_xlim()[1]))

        plot_filename = os.path.join(output_dir, f'{station_id}_{depth}cm.png')
        plt.savefig(plot_filename)
        plt.close(fig)

    print(f'Saved {len(station_df)} plots for station {station_id}')


def merge_data(fitted_results_dir, rosetta_param_file, inferred_param_file):
    """Merge fitted, rosetta, and inferred parameters into a single DataFrame."""

    # 1. Load empirically fitted data from JSON files
    fit_records = []
    fit_files = [os.path.join(fitted_results_dir, f) for f in os.listdir(fitted_results_dir) if f.endswith('.json')]
    for jf in fit_files:
        with open(jf, 'r') as f:
            data = json.load(f)
            station_id = os.path.basename(jf).split('_')[0].replace('.json', '')
            for depth_str, params in data.items():
                if not depth_str.isdigit():
                    continue
                depth = int(depth_str)
                if params.get('status') == 'Success':
                    p = {k: v['value'] for k, v in params['parameters'].items()}
                    fit_records.append({
                        'station': station_id,
                        'depth': depth,
                        'fit_params': p,
                        'raw_data': params.get('data')
                    })
    if not fit_records:
        return pd.DataFrame()
    fit_df = pd.DataFrame(fit_records)

    # 2. Load Rosetta parameters and unpivot to long format, indexed by station and level
    rosetta_df = pd.read_parquet(rosetta_param_file)
    rosetta_df = rosetta_df.groupby('station').first().reset_index()
    if 'station' in rosetta_df.columns:
        rosetta_df = rosetta_df.set_index('station')

    ros_records = []
    for station_id, row in rosetta_df.iterrows():
        params_by_level = {}
        for col, value in row.items():
            match = re.match(r'US_R3H3_L(\d+)_VG_(.+)', col)
            if match:
                level, param_name = int(match.group(1)), match.group(2)
                if level not in params_by_level:
                    params_by_level[level] = {}
                if 'log10' in param_name:
                    param_name = param_name.replace('log10_', '')
                    value = 10 ** value
                params_by_level[level][param_name] = value

        for level, params in params_by_level.items():
            if len(params) >= 4:
                ros_records.append({
                    'station': station_id,
                    'level': level,
                    'rosetta_params': params
                })
    rosetta_long_df = pd.DataFrame(ros_records)

    # 3. Map empirical depths to Rosetta levels and merge
    fit_df['level'] = fit_df['depth'].map(EMPIRICAL_TO_ROSETTA_LEVEL_MAP)
    merged_df = pd.merge(fit_df, rosetta_long_df, on=['station', 'level'], how='left')

    # 4. Load inferred parameters and merge
    inferred_df = pd.read_parquet(inferred_param_file)
    pred_cols = [c for c in inferred_df.columns if c in ['theta_r', 'theta_s', 'alpha', 'n']]
    if 'station' in inferred_df.columns:
        pred_by_station = inferred_df.groupby('station')[pred_cols].mean()
    elif getattr(inferred_df.index, 'name', None) == 'station':
        pred_by_station = inferred_df[pred_cols]
    else:
        pred_by_station = None  # likely error: predictions missing station identifiers

    if pred_by_station is not None:
        inferred_map = {idx: row.to_dict() for idx, row in pred_by_station.iterrows()}
        merged_df['inferred_params'] = merged_df['station'].map(inferred_map)

    # Clean up and set index
    merged_df.drop(columns=['level'], inplace=True)
    return merged_df.set_index(['station', 'depth'])


def compare_parameter_errors(merged_df):
    """Calculate and print R-squared and RMSE for parameters."""
    print("\n--- Parameter Error Comparison ---")
    df = merged_df.dropna(subset=['fit_params', 'rosetta_params', 'inferred_params']).copy()

    # Unpack dictionaries into columns
    fit_p = pd.json_normalize(df['fit_params']).add_prefix('fit_')
    ros_p = pd.json_normalize(df['rosetta_params']).add_prefix('ros_')
    inf_p = pd.json_normalize(df['inferred_params']).add_prefix('inf_')

    # Reset index to align and concat
    df = pd.concat([
        df.reset_index(),
        fit_p, ros_p, inf_p
    ], axis=1)

    params = ['theta_r', 'theta_s', 'alpha', 'n']
    results = {'Rosetta': {}, 'Inferred': {}}

    for p in params:
        true_col = f'fit_{p}'
        ros_col = f'ros_{p}'
        inf_col = f'inf_{p}'

        # Rosetta vs. Fit
        results['Rosetta'][p] = {
            'r2': r2_score(df[true_col], df[ros_col]),
            'rmse': mean_squared_error(df[true_col], df[ros_col], squared=False)
        }

        # Inferred vs. Fit
        results['Inferred'][p] = {
            'r2': r2_score(df[true_col], df[inf_col]),
            'rmse': mean_squared_error(df[true_col], df[inf_col], squared=False)
        }

    pprint(results)
    return results


if __name__ == '__main__':

    home_ = os.path.expanduser('~')
    root_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils')
    swap_root_ = os.path.join(root_, 'swapstress')

    # Corrected path for fitted results from fit_swrc.py
    fitted_dir_ = os.path.join(root_, 'soil_potential_obs', 'curve_fits', 'mt_mesonet', 'bayes')
    rosetta_pq_ = os.path.join(root_, 'rosetta', 'mt_mesonet', 'extracted_rosetta_points.parquet')
    # inferred_pq_ = os.path.join(swap_root_, 'training', 'predictions', 'stations_predictions_nn.parquet')

    inferred_pq_ = os.path.join(swap_root_, 'training', 'predictions', 'stations_predictions.parquet')

    output_dir_ = os.path.join(swap_root_, 'figures', 'station_curve_comparisons')

    if not os.path.exists(output_dir_):
        os.makedirs(output_dir_)

    # The merge function is now implemented
    merged_df = merge_data(fitted_dir_, rosetta_pq_, inferred_pq_)

    if not merged_df.empty:
        compare_parameter_errors(merged_df)
        # Group by station (the first level of the index)
        for station, station_df in merged_df.groupby(level=0):
            if not all(station_df['inferred_params'].notna()):
                print(f'station {station} has na in inferred parameters')
                continue
            if not all(station_df['fit_params'].notna()):
                print(f'station {station} has na in fit parameters')
                continue
            plot_station_comparison(station, station_df, output_dir_)
    else:
        print("No data to plot. Check input paths and file contents.")

# ========================= EOF ====================================================================
