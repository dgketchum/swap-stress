import os
import json
from glob import glob

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, root_mean_squared_error

from retention_curve import PARAM_SYMBOLS, ROSETTA_LEVEL_DEPTHS, ROSETTA_NOMINAL_DEPTHS, EMPIRICAL_TO_ROSETTA_LEVEL_MAP, map_empirical_to_rosetta_level

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

    def __init__(self, empirical_results_dir, station_metadata_path,
                 rosetta_path, pretrained_predictions_path, finetuned_predictions_path,
                 finetuning_split_path):
        """
        Initializes the comparator.

        Args:
            empirical_results_dir (str): Path to the directory with empirical fit JSONs.
            station_metadata_path (str): Path to the station metadata CSV.
            rosetta_path (str): Path to the Parquet file with Rosetta parameters.
            pretrained_predictions_path (str): Path to the Parquet file with pre-trained ML predictions.
            finetuned_predictions_path (str): Path to the Parquet file with fine-tuned ML predictions.
            finetuning_split_path (str): Path to the JSON file with fine-tuning train/val split info.
        """
        self.metrics = {}
        self.comparison_df = self._create_comparison_table(
            empirical_results_dir,
            station_metadata_path,
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
                    row[f'{param}_emp'] = values['value']
                all_results.append(row)
        return pd.DataFrame(all_results)

    def _create_comparison_table(self, empirical_dir, metadata_path, rosetta_path,
                                 pretrained_path, finetuned_path, finetuning_split_path):
        """Merges data from all sources into a single DataFrame for comparison."""
        empirical_df = self._load_empirical_results(empirical_dir)
        meta_df = pd.read_csv(metadata_path)
        base_df = pd.merge(empirical_df, meta_df, on='station', how='left')

        params = ['theta_r', 'theta_s', 'alpha', 'n']

        def merge_source(df, source_path, suffix):
            if os.path.exists(source_path):
                source_df = pd.read_parquet(source_path)

                # TODO: still with the sparse output from the extract
                if 'rosetta' in source_path:
                    source_df = source_df.groupby('station').first()

                if 'station' not in source_df.columns or 'depth' not in source_df.columns:
                    print(f"Warning: '{source_path}' is missing 'station' or 'depth' columns.")
                    return df

                renames = {p: f'{p}{suffix}' for p in params if p in source_df.columns}
                source_df.rename(columns=renames, inplace=True)
                merge_cols = ['station', 'depth'] + list(renames.values())
                df = pd.merge(df, source_df[merge_cols], on=['station', 'depth'], how='left')
            else:
                print(f"Warning: Data file not found at {source_path}")
            return df

        base_df = merge_source(base_df, rosetta_path, '_ros')
        base_df = merge_source(base_df, pretrained_path, '_ml_pre')
        base_df = merge_source(base_df, finetuned_path, '_ml_ft')

        # Add fine-tuning set information
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
        for param in params:
            self.metrics[param] = {}
            y_true_col = f'{param}_emp'
            if y_true_col not in self.comparison_df.columns:
                continue
            clean_df = self.comparison_df.dropna(subset=[y_true_col])
            y_true = clean_df[y_true_col]
            for prefix, name in sources.items():
                y_pred_col = f'{param}{prefix}'
                if y_pred_col not in clean_df.columns:
                    continue
                y_pred = clean_df[y_pred_col]
                valid_indices = y_true.index.intersection(y_pred.dropna().index)
                if valid_indices.empty:
                    continue
                y_true_aligned = y_true.loc[valid_indices]
                y_pred_aligned = y_pred.loc[valid_indices]
                self.metrics[param][name] = {
                    'r2': r2_score(y_true_aligned, y_pred_aligned),
                    'rmse': root_mean_squared_error(y_true_aligned, y_pred_aligned),
                }
        print("Metrics calculated.")
        return self.metrics

    def plot_scatter_comparison(self, output_dir, sources):
        """Generates scatter plots comparing each source to empirical data."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        params = ['theta_r', 'theta_s', 'alpha', 'n']
        palette = {'Train': 'blue', 'Validation': 'green', 'Not in Set': '#cccccc', 'Unknown': 'black'}
        markers = {'Train': 'o', 'Validation': 's', 'Not in Set': '.', 'Unknown': 'x'}

        for param in params:
            y_true_col = f'{param}_emp'
            if y_true_col not in self.comparison_df.columns:
                continue

            plt.style.use('seaborn-v0_8-whitegrid')
            fig, axes = plt.subplots(1, len(sources), figsize=(6 * len(sources), 5.5), sharey=True, sharex=True)
            if len(sources) == 1:
                axes = [axes]

            y_true = self.comparison_df[y_true_col].dropna()
            min_val, max_val = y_true.min() * 0.9, y_true.max() * 1.1

            for i, (prefix, name) in enumerate(sources.items()):
                ax = axes[i]
                y_pred_col = f'{param}{prefix}'
                if y_pred_col not in self.comparison_df.columns:
                    ax.text(0.5, 0.5, 'Data not found', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{name} vs. Empirical')
                    continue

                plot_df = self.comparison_df[[y_true_col, y_pred_col, 'finetune_set']].dropna(
                    subset=[y_true_col, y_pred_col])
                r2 = self.metrics.get(param, {}).get(name, {}).get('r2', float('nan'))

                sns.scatterplot(data=plot_df, x=y_true_col, y=y_pred_col,
                                hue='finetune_set', style='finetune_set',
                                palette=palette, markers=markers, ax=ax, alpha=0.8)

                ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
                ax.set_xlabel(f'Empirical {PARAM_SYMBOLS.get(param, param)}')
                if i == 0:
                    ax.set_ylabel(f'Predicted {PARAM_SYMBOLS.get(param, param)}')
                ax.set_title(f'{name} vs. Empirical')
                ax.text(0.05, 0.95, f'R² = {r2:.2f}', transform=ax.transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5))

                handles, labels = ax.get_legend_handles_labels()
                # Keep only unique legend entries
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


if __name__ == '__main__':
    home_ = os.path.expanduser('~')
    root_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils')
    training_root = os.path.join(root_, 'swapstress', 'training')
    inference_ = os.path.join(root_, 'swapstress', 'inference')

    empirical_dir_ = os.path.join(root_, 'soil_potential_obs', 'mt_mesonet', 'results_by_station')
    meta_path_ = os.path.join(root_, 'soil_potential_obs', 'mt_mesonet', 'station_metadata.csv')
    rosetta_data_path_ = os.path.join(root_, 'rosetta', 'mt_mesonet', 'extracted_rosetta_points.parquet')

    pretrained_data_path_ = os.path.join(inference_, f'finetuned_predictions.parquet')
    finetuned_data_path_ = os.path.join(inference_, f'pretrained_predictions.parquet')

    finetuning_split_path_ = os.path.join(training_root, 'finetuning_split_info.json')

    plots_out_ = os.path.join(root_, 'swapstress', 'comparison_plots')

    comparison = SWRCComparison(
        empirical_results_dir=empirical_dir_,
        station_metadata_path=meta_path_,
        rosetta_path=rosetta_data_path_,
        pretrained_predictions_path=pretrained_data_path_,
        finetuned_predictions_path=finetuned_data_path_,
        finetuning_split_path=finetuning_split_path_
    )

    sources_to_compare = {
        '_ros': 'Rosetta',
        '_ml_pre': 'ML (Pre-trained)',
        '_ml_ft': 'ML (Fine-tuned)'
    }

    metrics_results = comparison.calculate_metrics(sources=sources_to_compare)
    print("\n--- Calculated Metrics ---")
    print(json.dumps(metrics_results, indent=2))

    comparison.plot_scatter_comparison(output_dir=plots_out_, sources=sources_to_compare)

    print("\nComparison complete.")

# ========================= EOF ====================================================================
