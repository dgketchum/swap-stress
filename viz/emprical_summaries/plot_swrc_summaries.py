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
    json_files = glob(os.path.join(results_dir, '**', '*.json'), recursive=True)
    if not json_files:
        print(f"No '.json' files found in {results_dir}")
        return None

    print(f"Found {len(json_files)} result files to process...")
    all_results = []
    for f in json_files:
        station_name = os.path.basename(f).replace('.json', '')
        with open(f, 'r') as jf:
            data = json.load(jf)

        for depth, results in data.items():
            if results.get('status') != 'Success':
                continue

            row = {'station': station_name, 'depth': int(float(depth)), 'n_obs': results.get('n_obs', 0)}

            for param, values in results['parameters'].items():
                row[param] = values['value']
            all_results.append(row)

    if not all_results:
        print("No successful fits found in the result files.")
        return None

    return pd.DataFrame(all_results)



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

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)

    for idx, p_influence in enumerate(parameters):
        ax = axes.flat[idx]

        base_params = {p: param_stats[p]['mean'] for p in parameters}

        percentiles = ['p10', 'p50', 'p90']
        labels = ['10th Percentile', 'Median', '90th Percentile']
        colors = ['blue', 'green', 'red']

        for i, percentile_key in enumerate(percentiles):
            current_params = base_params.copy()
            current_params[p_influence] = param_stats[p_influence][percentile_key]

            vwc_curve = _van_genuchten_model_local(
                psi_range,
                theta_r=current_params['theta_r'],
                theta_s=current_params['theta_s'],
                alpha=current_params['alpha'],
                n=current_params['n']
            )

            symbol = PARAM_SYMBOLS.get(p_influence, p_influence)
            ax.plot(
                vwc_curve,
                psi_range,
                color=colors[i],
                label=f'{symbol} - {labels[i]} ({param_stats[p_influence][percentile_key]:.3f})'
            )
        ax.set_yscale('log')
        ax.set_xlim(right=0.65)
        ax.set_ylim(top=10 ** 7)
        symbol_influence = PARAM_SYMBOLS.get(p_influence, p_influence)
        ax.set_title(f'{symbol_influence}', fontsize=14, fontweight='bold')
        ax.grid(True, which="both", ls="--", c='0.7')

        if idx % 2 == 0:
            ax.set_ylabel('Soil Water Potential (cm) - Log Scale', fontsize=12)

        if idx >= 2:
            ax.set_xlabel('Volumetric Water Content ($cm^3/cm^3$)', fontsize=12)

        ax.legend(fontsize=10, framealpha=0.6, facecolor='white')

    fig.suptitle(f'van Genuchten Parameter Ranges in GSHP (n={len(df)})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plot_filename = os.path.join(output_dir, 'parameter_influence_panels.png')
    plt.savefig(plot_filename, dpi=300)
    print(f"  - Saved {plot_filename}")
    plt.close(fig)

    print("van Genuchten Parameter influence panel plot complete.")


if __name__ == '__main__':
    home_ = os.path.expanduser('~')
    root_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils')

    fits = os.path.join(root_,'soil_potential_obs', 'curve_fits', 'gshp', 'nelder')
    plot_output_dir_ = os.path.join(root_, 'swapstress', 'figures', 'comparison_plots')

    # plot_parameter_summaries(results_dir=results_dir_,
    #                          output_dir=plot_output_dir_)
    #
    # plot_parameter_histograms(results_dir=results_dir_,
    #                           output_dir=plot_output_dir_)

    plot_parameter_influence(results_dir=fits,
                             output_dir=plot_output_dir_)

# ========================= EOF ====================================================================
