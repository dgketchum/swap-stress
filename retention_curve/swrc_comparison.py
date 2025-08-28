import json
import os
import re
from glob import glob

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error

from retention_curve import PARAM_SYMBOLS, EMPIRICAL_TO_ROSETTA_LEVEL_MAP


def _van_genuchten_model(psi, theta_r, theta_s, alpha, n):
    """Helper function to calculate VWC from van Genuchten parameters."""
    if n <= 1:
        return np.full_like(psi, np.nan)
    m = 1 - (1 / n)
    psi_safe = np.maximum(psi, 1e-9)
    term = 1 + (alpha * psi_safe) ** n
    return theta_r + (theta_s - theta_r) / (term) ** m


class SWRCComparison:
    """
    Compares SWRC parameter estimates from various sources against empirical data.
    Sources can include Rosetta, a pre-trained ML model, and a fine-tuned ML model.
    """

    def __init__(self, empirical_results_dir,
                 rosetta_path, pretrained_predictions_path, finetuned_predictions_path,
                 finetuning_split_path, levels=None):
        """
        Initializes the comparator.

        Args:
            empirical_results_dir (str): Path to the directory with empirical fit JSONs.
            rosetta_path (str): Path to the Parquet file with Rosetta parameters.
            pretrained_predictions_path (str): Path to the Parquet file with pre-trained ML predictions.
            finetuned_predictions_path (str): Path to the Parquet file with fine-tuned ML predictions.
            finetuning_split_path (str): Path to the JSON file with fine-tuning train/val split info.
        """
        self.metrics = {}
        self.levels = levels
        self.comparison_df = self._create_comparison_table(
            empirical_results_dir,
            rosetta_path,
            pretrained_predictions_path,
            finetuned_predictions_path,
            finetuning_split_path
        )

    @staticmethod
    def _load_empirical_results(results_dir):
        """Loads all successful empirical fit results from JSON files."""
        json_files = glob(os.path.join(results_dir, '**', '*_fit_results.json'), recursive=True)
        if not json_files:
            raise FileNotFoundError(f"No empirical result files found in {results_dir}")

        all_results = []
        for f in json_files:
            station_name = os.path.basename(f).replace('_fit_results.json', '')
            with open(f, 'r') as jf:
                data = json.load(jf)
            for depth, results in data.items():
                if results.get('status') != 'Success':
                    continue
                row = {'station': station_name, 'depth': int(depth)}
                for param, values in results['parameters'].items():
                    row[f'station_{param}'] = values['value']
                all_results.append(row)
        return pd.DataFrame(all_results)

    def _create_comparison_table(self, empirical_dir, rosetta_path,
                                 pretrained_path, finetuned_path, finetuning_split_path):
        """Merges data from all sources into a single DataFrame for comparison."""

        params = ['theta_r', 'theta_s', 'alpha', 'n']
        base_df = self._load_empirical_results(empirical_dir)
        base_df['rosetta_level'] = base_df['depth'].map(EMPIRICAL_TO_ROSETTA_LEVEL_MAP)
        params_to_melt = [f'station_{p}' for p in params]

        base_df = base_df.melt(
            id_vars=['station', 'rosetta_level'],
            value_vars=params_to_melt,
            var_name='param',
            value_name='value'
        )
        base_df['level_param'] = base_df.apply(lambda r: f'L{r['rosetta_level']}_{r['param']}', axis=1)
        base_df = base_df.pivot_table(
            index='station',
            columns='level_param',
            values='value',
            aggfunc='mean'
        ).reset_index()
        pattern = r'L([1-7])_station_'
        replacement = r'station_L\1_'
        base_df.columns = base_df.columns.str.replace(pattern, replacement, regex=True)

        rosetta_df = pd.read_parquet(rosetta_path)
        rosetta_df = rosetta_df.groupby('station').first()
        rosetta_df['station'] = rosetta_df.index
        rosetta_df.index = list(range(len(rosetta_df)))
        drops = [c for c in rosetta_df.columns if 'US_R3H3' not in c]
        drops.remove('station')
        rosetta_df = rosetta_df.drop(columns=drops)
        pattern = r'US_R3H3_L([1-7])_VG_'
        replacement = r'rosetta_L\1_'
        rosetta_df.columns = rosetta_df.columns.str.replace(pattern, replacement, regex=True)
        base_df = pd.merge(base_df, rosetta_df, on='station', how='left')

        pretrain_df = pd.read_parquet(pretrained_path)
        pretrain_df = pretrain_df.drop(columns=['longitude', 'latitude'])
        pretrain_df['station'] = pretrain_df.index
        pretrain_df.index = list(range(len(pretrain_df)))
        pattern = r'US_R3H3_L([1-7])_VG_'
        replacement = r'pretrain_L\1_'
        pretrain_df.columns = pretrain_df.columns.str.replace(pattern, replacement, regex=True)
        base_df = pd.merge(base_df, pretrain_df, on='station', how='left')

        finetune_df = pd.read_parquet(finetuned_path)
        finetune_df = finetune_df.drop(columns=['longitude', 'latitude'])
        finetune_df['station'] = finetune_df.index
        finetune_df.index = list(range(len(finetune_df)))
        pattern = r'US_R3H3_L([1-7])_VG_'
        replacement = r'finetune_L\1_'
        finetune_df.columns = finetune_df.columns.str.replace(pattern, replacement, regex=True)
        base_df = pd.merge(base_df, finetune_df, on='station', how='left')

        if self.levels:
            filtered_strings = [
                s for s in base_df.columns if (match := re.search(r'L([1-7])', s))
                                              and int(match.group(1)) in self.levels]
            base_df = base_df[['station'] + filtered_strings]

        if os.path.exists(finetuning_split_path):
            with open(finetuning_split_path, 'r') as f:
                split_info = json.load(f)

            train_ids = {d for d in split_info.get('train', [])}
            val_ids = {d for d in split_info.get('validation', [])}

            base_df['unique_id'] = base_df['station'].copy()

            def assign_set(unique_id):
                if unique_id in train_ids:
                    return 'Train'
                elif unique_id in val_ids:
                    return 'Validation'
                return 'Not in Set'

            base_df['finetune_set'] = base_df['unique_id'].apply(assign_set)
            base_df.drop(columns=['unique_id'], inplace=True)
        else:
            print(f"Warning: Finetuning split file not found at {finetuning_split_path}")
            base_df['finetune_set'] = 'Unknown'

        return base_df

    def calculate_metrics(self, sources):
        """Calculates R² and RMSE for each source against the empirical data."""

        params = ['theta_r', 'theta_s', 'alpha', 'n']
        prefix_map = {'_ros': 'rosetta', '_ml_pre': 'pretrain', '_ml_ft': 'finetune'}

        levels_to_process = self.levels
        if not levels_to_process:
            all_levels = set()
            for col in self.comparison_df.columns:
                if match := re.search(r'_L(\d+)_', col):
                    all_levels.add(int(match.group(1)))
            levels_to_process = sorted(list(all_levels))

        for param in params:

            if param not in ['alpha', 'n']:
                continue

            self.metrics[param] = {}
            for level in levels_to_process:
                self.metrics[param][level] = {}
                y_true_col = f'station_L{level}_{param}'
                if y_true_col not in self.comparison_df.columns:
                    continue

                y_true = self.comparison_df[y_true_col]

                for prefix, name in sources.items():
                    pred_prefix = prefix_map.get(prefix)

                    if prefix != '_ros':
                        continue

                    if not pred_prefix:
                        continue

                    y_pred_col = f'{pred_prefix}_L{level}_{param}'
                    if y_pred_col not in self.comparison_df.columns:
                        y_pred_col = f'{pred_prefix}_L{level}_log10_{param}'
                        assert y_pred_col in self.comparison_df.columns
                        exponentiate = True
                    else:
                        exponentiate = False

                    y_pred = self.comparison_df[y_pred_col]
                    df = pd.DataFrame({'true': y_true, 'pred': y_pred}).dropna()
                    if exponentiate:
                        df['true'] = np.log10(df['true'])

                    if df.empty:
                        continue

                    metrics = {
                        'r2': r2_score(df['true'], df['pred']),
                        'mse': mean_squared_error(df['true'], df['pred']),
                        'mae': mean_absolute_error(df['true'], df['pred']),
                        'rmse': root_mean_squared_error(df['true'], df['pred']),
                        'bias': (df['pred'] - df['true']).mean(),
                        'mean_val': df['true'].mean().item(),
                        'std_val': df['true'].std().item(),
                    }

                    self.metrics[param][level][name] = metrics

        print("Metrics calculated.")
        return self.metrics

    def plot_scatter_comparison(self, output_dir, sources):
        """Generates scatter plots comparing each source to empirical data."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        params = ['theta_r', 'theta_s', 'alpha', 'n']
        palette = {'Train': 'blue', 'Validation': 'green', 'Not in Set': '#cccccc', 'Unknown': 'black'}
        markers = {'Train': 'o', 'Validation': 's', 'Not in Set': '.', 'Unknown': 'D'}
        prefix_map = {'_ros': 'rosetta', '_ml_pre': 'pretrain', '_ml_ft': 'finetune'}

        levels_to_process = self.levels
        if not levels_to_process:
            all_levels = set()
            for col in self.comparison_df.columns:
                if match := re.search(r'_L(\d+)_', col):
                    all_levels.add(int(match.group(1)))
            levels_to_process = sorted(list(all_levels))

        for param in params:
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, axes = plt.subplots(1, len(sources), figsize=(6 * len(sources), 5.5), sharey=True, sharex=True)
            if len(sources) == 1:
                axes = [axes]

            all_true_vals = [self.comparison_df[f'station_L{l}_{param}'] for l in levels_to_process if
                             f'station_L{l}_{param}' in self.comparison_df]
            if not all_true_vals:
                plt.close(fig)
                continue

            y_true_all = pd.concat(all_true_vals).dropna()
            min_val, max_val = y_true_all.min() * 0.9, y_true_all.max() * 1.1

            for i, (prefix, name) in enumerate(sources.items()):
                ax = axes[i]
                pred_prefix = prefix_map.get(prefix)
                if not pred_prefix:
                    ax.text(0.5, 0.5, 'Data not found', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{name} vs. Empirical')
                    continue

                true_vals, pred_vals = [], []
                for level in levels_to_process:
                    y_true_col = f'station_L{level}_{param}'
                    if y_true_col not in self.comparison_df:
                        continue

                    y_pred_col = f'{pred_prefix}_L{level}_{param}'
                    exponentiate = False
                    if y_pred_col not in self.comparison_df:
                        y_pred_col = f'{pred_prefix}_L{level}_log10_{param}'
                        if y_pred_col not in self.comparison_df:
                            continue
                        exponentiate = True

                    y_pred = self.comparison_df[y_pred_col]
                    if exponentiate:
                        y_pred = 10 ** y_pred

                    true_vals.append(self.comparison_df[y_true_col])
                    pred_vals.append(y_pred)

                if not pred_vals:
                    ax.text(0.5, 0.5, 'Data not found', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{name} vs. Empirical')
                    continue

                plot_df = pd.DataFrame({'true': pd.concat(true_vals), 'pred': pd.concat(pred_vals)}).dropna()
                plot_df['finetune_set'] = self.comparison_df.loc[plot_df.index, 'finetune_set']

                r2 = r2_score(plot_df['true'], plot_df['pred']) if not plot_df.empty else float('nan')

                sns.scatterplot(data=plot_df, x='true', y='pred', hue='finetune_set', style='finetune_set',
                                palette=palette, markers=markers, ax=ax, alpha=0.8)

                ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
                ax.set_xlabel(f'Empirical {PARAM_SYMBOLS.get(param, param)}')
                if i == 0:
                    ax.set_ylabel(f'Predicted {PARAM_SYMBOLS.get(param, param)}')
                ax.set_title(f'{name} vs. Empirical')
                ax.text(0.05, 0.95, f'R² = {r2:.2f}', transform=ax.transAxes, fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5))

                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys())

                ax.grid(True)
                ax.set_xlim(min_val, max_val)
                ax.set_ylim(min_val, max_val)

            fig.suptitle(f'Comparison for {PARAM_SYMBOLS.get(param, param)}', fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plot_filename = os.path.join(output_dir, f'{param}_scatter_comparison.png')
            plt.savefig(plot_filename, dpi=300)
            plt.close(fig)
            print(f"Saved scatter plot to {plot_filename}")

    def plot_histogram_comparison(self, output_dir, bins=30):
        """
        Plots semi-transparent overlaid histograms comparing empirical (station)
        parameter estimates to Rosetta estimates for each parameter and level.

        - Uses log10 transform for parameters where Rosetta provides log10 values
          (e.g., columns like `rosetta_Lx_log10_alpha`). When this occurs, the
          station values are transformed to log10 for an apples-to-apples
          comparison.
        - If multiple levels are requested/available, creates a subplot per
          level for each parameter.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        params = ['theta_r', 'theta_s', 'alpha', 'n']
        palette = {'Station': '#1f77b4', 'Rosetta': '#ff7f0e'}

        # Determine which levels to plot
        levels_to_process = self.levels
        if not levels_to_process:
            all_levels = set()
            for col in self.comparison_df.columns:
                if match := re.search(r'_L(\d+)_', col):
                    all_levels.add(int(match.group(1)))
            levels_to_process = sorted(list(all_levels))

        for param in params:
            # Collect which levels actually have both station and rosetta data
            available_levels = []
            for level in levels_to_process:
                station_col = f'station_L{level}_{param}'
                ros_col = f'rosetta_L{level}_{param}'
                ros_log_col = f'rosetta_L{level}_log10_{param}'
                if station_col in self.comparison_df.columns and \
                        (ros_col in self.comparison_df.columns or ros_log_col in self.comparison_df.columns):
                    available_levels.append(level)

            if not available_levels:
                continue

            plt.style.use('seaborn-v0_8-whitegrid')
            n_cols = len(available_levels)
            fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), sharey=False)
            if n_cols == 1:
                axes = [axes]

            for ax, level in zip(axes, available_levels):
                station_col = f'station_L{level}_{param}'
                ros_col = f'rosetta_L{level}_{param}'
                ros_log_col = f'rosetta_L{level}_log10_{param}'

                use_log_transform = False
                if ros_col in self.comparison_df.columns:
                    ros_vals = self.comparison_df[ros_col]
                elif ros_log_col in self.comparison_df.columns:
                    ros_vals = self.comparison_df[ros_log_col]
                    use_log_transform = True
                else:
                    # Nothing to plot for this level
                    continue

                sta_vals = self.comparison_df[station_col]

                # Align and clean data
                df_plot = pd.DataFrame({
                    'Station': sta_vals,
                    'Rosetta': ros_vals,
                }).dropna()

                if df_plot.empty:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'L{level} {PARAM_SYMBOLS.get(param, param)}')
                    continue

                # Apply log10 transform if Rosetta provided log10 values
                if use_log_transform:
                    # Remove non-positive station values before log10
                    df_plot = df_plot[df_plot['Station'] > 0].copy()
                    df_plot['Station'] = np.log10(df_plot['Station'])
                    x_label = f'log10({PARAM_SYMBOLS.get(param, param)})'
                else:
                    x_label = f'{PARAM_SYMBOLS.get(param, param)}'

                # Build long-form for seaborn overlay
                long_df = df_plot.melt(var_name='Source', value_name='Value')

                sns.histplot(
                    data=long_df,
                    x='Value',
                    hue='Source',
                    bins=bins,
                    element='step',
                    stat='density',
                    common_bins=True,
                    alpha=0.45,
                    palette=palette,
                    legend=False,
                    ax=ax,
                )

                ax.set_title(f'L{level} {PARAM_SYMBOLS.get(param, param)}', fontsize=13)
                ax.set_xlabel(x_label)
                ax.set_ylabel('Density')
                ax.grid(True, which='major', alpha=0.3)

            for ax in axes:
                handles = [
                    Patch(facecolor=palette['Station'], edgecolor=palette['Station'], alpha=0.45, label='Station'),
                    Patch(facecolor=palette['Rosetta'], edgecolor=palette['Rosetta'], alpha=0.45, label='Rosetta'),
                ]
                ax.legend(handles=handles, loc='best', title=None)

            fig.suptitle(f'Distribution Comparison: Station vs Rosetta for {PARAM_SYMBOLS.get(param, param)}',
                         fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0.03, 1, 0.92])
            out_path = os.path.join(output_dir, f'{param}_hist_comparison.png')
            plt.savefig(out_path, dpi=300)
            plt.close(fig)
            print(f"Saved histogram comparison to {out_path}")


if __name__ == '__main__':
    home_ = os.path.expanduser('~')
    root_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils')
    training_root = os.path.join(root_, 'swapstress', 'training')
    inference_ = os.path.join(root_, 'swapstress', 'inference')

    empirical_dir_ = os.path.join(root_, 'soil_potential_obs', 'mt_mesonet', 'results_by_station')
    meta_path_ = os.path.join(root_, 'soil_potential_obs', 'mt_mesonet', 'station_metadata.csv')
    rosetta_data_path_ = os.path.join(root_, 'rosetta', 'mt_mesonet', 'extracted_rosetta_points.parquet')

    finetuned_data_path_ = os.path.join(inference_, f'finetuned_predictions.parquet')
    pretrained_data_path_ = os.path.join(inference_, f'pretrained_predictions.parquet')

    finetuning_split_path_ = os.path.join(training_root, 'finetuning_split_info.json')

    plots_out_ = os.path.join(root_, 'swapstress', 'comparison_plots')

    comparison = SWRCComparison(
        empirical_results_dir=empirical_dir_,
        rosetta_path=rosetta_data_path_,
        pretrained_predictions_path=pretrained_data_path_,
        finetuned_predictions_path=finetuned_data_path_,
        finetuning_split_path=finetuning_split_path_,
        levels=(2,),
    )

    sources_to_compare = {
        '_ros': 'Rosetta',
        '_ml_pre': 'ML (Pre-trained)',
        '_ml_ft': 'ML (Fine-tuned)'
    }

    metrics_results = comparison.calculate_metrics(sources=sources_to_compare)
    # print("\n--- Calculated Metrics ---")
    # print(json.dumps(metrics_results, indent=2))

    # comparison.plot_scatter_comparison(output_dir=plots_out_, sources=sources_to_compare)

    # Overlay semi-transparent histograms: Station vs Rosetta
    comparison.plot_histogram_comparison(output_dir=plots_out_)

    print("\nComparison complete.")

# ========================= EOF ====================================================================
