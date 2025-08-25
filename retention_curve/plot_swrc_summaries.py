import os
import json
from glob import glob

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from retention_curve import PARAM_SYMBOLS


def _load_results_to_df(results_dir):
    """Loads all result JSONs from a directory into a single pandas DataFrame."""
    json_files = glob(os.path.join(results_dir, '**', '*_fit_results.json'), recursive=True)
    if not json_files:
        print(f"No '_fit_results.json' files found in {results_dir}")
        return None

    print(f"Found {len(json_files)} result files to process...")
    all_results = []
    for f in json_files:
        station_name = os.path.basename(f).replace('_fit_results.json', '')
        with open(f, 'r') as jf:
            data = json.load(jf)

        for depth, results in data.items():
            if results.get('status') != 'Success':
                continue

            row = {'station': station_name, 'depth': int(depth), 'n_obs': results.get('n_obs', 0)}

            for param, values in results['parameters'].items():
                row[param] = values['value']
            all_results.append(row)

    if not all_results:
        print("No successful fits found in the result files.")
        return None

    return pd.DataFrame(all_results)


def plot_parameter_summaries(results_dir, output_dir):
    """
    Aggregates SWRC fit results from multiple JSON files and creates
    summary box plots for each parameter by depth.
    """
    df = _load_results_to_df(results_dir)
    if df is None:
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    parameters_to_plot = ['theta_r', 'theta_s', 'alpha', 'n']
    print("Generating summary box plots...")

    for param in parameters_to_plot:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.boxplot(x='depth', y=param, data=df, ax=ax)

        # Add annotations
        unique_depths = sorted(df['depth'].unique())
        for i, depth in enumerate(unique_depths):
            depth_data = df[df['depth'] == depth]
            n_sites = len(depth_data)
            n_obs = depth_data['n_obs'].sum()

            q3 = depth_data[param].quantile(0.75)
            q1 = depth_data[param].quantile(0.25)
            iqr = q3 - q1
            upper_whisker = min(depth_data[param].max(), q3 + 1.5 * iqr)
            y_pos = upper_whisker * 1.05

            text = f"Sites: {n_sites}\nObs: {n_obs}"

            ax.text(i + 0.2, y_pos, text, ha='left', va='bottom', fontsize='small', color='#333333')

        symbol = PARAM_SYMBOLS.get(param, param)
        ax.set_title(f'Fitted {symbol} Distribution by Depth', fontsize=16, fontweight='bold')
        ax.set_xlabel('Depth (cm)', fontsize=12)
        ax.set_ylabel(f'{symbol} Value', fontsize=12)
        plt.tight_layout()

        plot_filename = os.path.join(output_dir, f'{param}_summary_boxplot.png')
        plt.savefig(plot_filename, dpi=300)
        print(f"  - Saved {plot_filename}")
        plt.close(fig)

    print("Summary plots complete.")


def plot_parameter_histograms(results_dir, output_dir):
    """
    Aggregates SWRC fit results and creates a KDE histogram for each parameter,
    using data from all depths and stations.
    """
    df = _load_results_to_df(results_dir)
    if df is None:
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    parameters_to_plot = ['theta_r', 'theta_s', 'alpha', 'n']
    print("\nGenerating parameter histograms...")

    for param in parameters_to_plot:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(data=df, x=param, kde=True, ax=ax)

        symbol = PARAM_SYMBOLS.get(param, param)
        ax.set_title(f'Overall Distribution of Fitted {symbol}', fontsize=16, fontweight='bold')
        ax.set_xlabel(f'{symbol} Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        plt.tight_layout()

        plot_filename = os.path.join(output_dir, f'{param}_overall_histogram.png')
        plt.savefig(plot_filename, dpi=300)
        print(f"  - Saved {plot_filename}")
        plt.close(fig)

    print("Histogram plots complete.")


def _van_genuchten_model_local(psi, theta_r, theta_s, alpha, n):
    if n <= 1:
        return np.full_like(psi, np.nan)
    m = 1 - (1 / n)
    psi_safe = np.maximum(psi, 1e-9)
    term = 1 + (alpha * psi_safe) ** n
    return theta_r + (theta_s - theta_r) / (term) ** m


def plot_parameter_influence(results_dir, output_dir):
    """
    Demonstrates the influence of each fitted van Genuchten parameter on the SWRC shape.
    Plots SWRCs for 10th, 50th, and 90th percentiles of one parameter,
    while holding others at their mean values.
    """
    df = _load_results_to_df(results_dir)
    if df is None:
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    parameters = ['theta_r', 'theta_s', 'alpha', 'n']

    # Calculate global means and percentiles for all parameters
    param_stats = {}
    for p in parameters:
        param_stats[p] = {
            'mean': df[p].mean(),
            'p10': df[p].quantile(0.10),
            'p50': df[p].quantile(0.50),
            'p90': df[p].quantile(0.90)
        }

    print("\nGenerating parameter influence plots...")

    psi_range = np.logspace(0, 7, 100)  # Soil Water Potential range for plotting curves

    for p_influence in parameters:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(8, 7))

        # Base parameters (mean of all others)
        base_params = {p: param_stats[p]['mean'] for p in parameters}

        # Plot curves for 10th, 50th, 90th percentiles
        percentiles = ['p10', 'p50', 'p90']
        labels = ['10th Percentile', 'Median', '90th Percentile']
        colors = ['blue', 'green', 'red']

        for i, percentile_key in enumerate(percentiles):
            current_params = base_params.copy()
            current_params[p_influence] = param_stats[p_influence][percentile_key]

            # Calculate VWC for the current set of parameters
            vwc_curve = _van_genuchten_model_local(psi_range,
                                                   theta_r=current_params['theta_r'],
                                                   theta_s=current_params['theta_s'],
                                                   alpha=current_params['alpha'],
                                                   n=current_params['n'])

            symbol = PARAM_SYMBOLS.get(p_influence, p_influence)

            ax.plot(vwc_curve, psi_range, color=colors[i], label=f'{symbol} - {labels[i]} ({param_stats[p_influence][percentile_key]:.3f})')

        ax.set_yscale('log')
        ax.set_ylabel('Soil Water Potential (cm) - Log Scale', fontsize=12)
        ax.set_xlabel('Volumetric Water Content ($cm^3/cm^3$)', fontsize=12)
        ax.set_xlim(right=0.65)
        ax.set_ylim(top=10 ** 7)

        symbol_influence = PARAM_SYMBOLS.get(p_influence, p_influence)
        ax.set_title(f'Influence of {symbol_influence} on SWRC Shape', fontsize=16, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, which="both", ls="--", c='0.7')
        plt.tight_layout()

        plot_filename = os.path.join(output_dir, f'{p_influence}_influence_plot.png')
        plt.savefig(plot_filename, dpi=300)
        print(f"  - Saved {plot_filename}")
        plt.close(fig)

    print("Parameter influence plots complete.")


if __name__ == '__main__':
    home_ = os.path.expanduser('~')
    root_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'soil_potential_obs', 'mt_mesonet')

    results_dir_ = os.path.join(root_, 'results_by_station')
    plot_output_dir_ = os.path.join(root_, 'summary_plots')

    # plot_parameter_summaries(results_dir=results_dir_,
    #                          output_dir=plot_output_dir_)
    #
    # plot_parameter_histograms(results_dir=results_dir_,
    #                           output_dir=plot_output_dir_)

    plot_parameter_influence(results_dir=results_dir_,
                             output_dir=plot_output_dir_)

# ========================= EOF ====================================================================
