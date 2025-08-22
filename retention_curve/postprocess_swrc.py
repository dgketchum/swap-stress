import os
import json
from glob import glob

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

PARAM_SYMBOLS = {
    'theta_r': r'$\theta_r$',
    'theta_s': r'$\theta_s$',
    'alpha': r'$\alpha$',
    'n': 'n'
}

def plot_parameter_summaries(results_dir, output_dir):
    """
    Aggregates SWRC fit results from multiple JSON files and creates
    summary box plots for each parameter by depth.

    Args:
        results_dir (str): Directory containing the JSON result files.
        output_dir (str): Directory where the output plots will be saved.
    """
    json_files = glob(os.path.join(results_dir, '**', '*_fit_results.json'), recursive=True)
    if not json_files:
        print(f"No '_fit_results.json' files found in {results_dir}")
        return

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
        return

    df = pd.DataFrame(all_results)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    parameters_to_plot = ['theta_r', 'theta_s', 'alpha', 'n']
    print("Generating summary plots...")

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

            # Position text above the top whisker
            q3 = depth_data[param].quantile(0.75)
            q1 = depth_data[param].quantile(0.25)
            iqr = q3 - q1
            upper_whisker = min(depth_data[param].max(), q3 + 1.5 * iqr)
            y_pos = upper_whisker * 1.05  # 5% above the whisker

            text = f"Sites: {n_sites}\nObs: {n_obs}"
            ax.text(i, y_pos, text, ha='center', va='bottom', fontsize='small', color='#333333')

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


if __name__ == '__main__':
    home_ = os.path.expanduser('~')
    root_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'soil_potential_obs', 'mt_mesonet')

    results_dir_ = os.path.join(root_, 'results_by_station')
    plot_output_dir_ = os.path.join(root_, 'summary_plots')

    plot_parameter_summaries(results_dir=results_dir_,
                             output_dir=plot_output_dir_)
# ========================= EOF ====================================================================