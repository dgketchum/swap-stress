import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model


class SWRC:
    """
    A class to encapsulate Soil Water Retention Curve (SWRC) data,
    fit the van Genuchten-Mualem model for one or more soil depths,
    and visualize the results.

    This class uses the lmfit library and follows the best practices outlined
    in the comprehensive guide, including using the Trust Region Reflective ('trf')
    algorithm and data-driven initial parameter guesses with physical bounds.
    """

    def __init__(self, filepath=None, depth_col=None):
        """
        Initializes the SWRC_fitter object.

        Args:
            filepath (str, optional): Path to a CSV or Parquet file to load data from.
        """
        self.filepath = filepath
        self.data_by_depth = {}
        self.fit_results = {}
        self._vg_model = Model(self._van_genuchten_model)

        if not depth_col:
            self.depth_col = 'stationDepth [cm]'
        else:
            self.depth_col = depth_col

        if filepath:
            self.load_from_file(filepath)
        else:
            raise ValueError("A filepath is required to initialize the SWRC class.")

    def load_from_file(self, filepath):
        """
        Loads SWRC data from a CSV or Parquet file and groups it by depth.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found at {filepath}")

        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        else:
            raise ValueError("File must be a .csv or .parquet file.")

        if 'KPA' in df.columns and 'VWC' in df.columns:
            df['suction'] = np.abs(df['KPA'].values * 10.19716)
            df['theta'] = df['VWC'].values
        elif 'suction_cm' in df.columns and 'theta' in df.columns:
            df['suction'] = np.abs(df['suction_cm'].values)
            df['theta'] = df['theta'].values
        else:
            raise ValueError("Data must contain either ('suction_cm', 'theta') or ('KPA', 'VWC') columns.")

        if self.depth_col in df.columns:
            for depth, group in df.groupby(self.depth_col):
                self.data_by_depth[depth] = group
            print(f"Data loaded and grouped by {len(self.data_by_depth)} depths.")
        else:
            self.data_by_depth[0] = df
            print("Data loaded as a single dataset (no depth column found).")

    @staticmethod
    def _van_genuchten_model(psi, theta_r, theta_s, alpha, n):
        if n <= 1:
            return np.full_like(psi, np.nan)
        m = 1 - 1 / n
        psi_safe = np.maximum(psi, 1e-9)
        term = 1 + (alpha * psi_safe) ** n
        return theta_r + (theta_s - theta_r) / (term) ** m

    def _generate_initial_params(self, data_for_depth):
        params = self._vg_model.make_params()
        theta_s_init = np.max(data_for_depth['theta'])
        theta_r_init = np.min(data_for_depth['theta'])
        alpha_init = 0.05
        n_init = 1.5
        params['theta_s'].set(value=theta_s_init, min=theta_s_init * 0.9, max=theta_s_init * 1.1)
        params['theta_r'].set(value=theta_r_init, min=0.0, max=theta_r_init * 1.1)
        params['alpha'].set(value=alpha_init, min=1e-5, max=5.0)
        params['n'].set(value=n_init, min=1.01, max=10.0)
        return params

    def fit(self, report=False):
        """
        Fits the van Genuchten model to the SWRC data for each depth.

        Args:
            report (bool): If True, prints the detailed fit report for each depth.

        Returns:
            dict: A dictionary of lmfit.model.ModelResult objects, keyed by depth.
        """
        self.fit_results = {}
        for depth, data_df in self.data_by_depth.items():
            print(f"--- Fitting for Depth: {depth} cm ---")
            initial_params = self._generate_initial_params(data_df)
            try:
                result = self._vg_model.fit(data_df['theta'], initial_params, psi=data_df['suction'], method='trf')
                self.fit_results[depth] = result
                if report:
                    print(result.fit_report())
            except Exception as e:
                print(f"An error occurred during fitting for depth {depth} cm: {e}")
                self.fit_results[depth] = None
        print("\nAll fits complete.")
        return self.fit_results

    def plot(self, save_path=None, show=True, colormap='plasma'):
        """
        Plots the raw SWRC data and the fitted van Genuchten curve for all depths.
        """
        if not self.fit_results:
            print("Fit has not been performed yet. Please run .fit() first.")
            return

        plt.style.use('seaborn-v0_8-darkgrid')

        # Get the correct colors from the style to apply them explicitly
        facecolor = plt.rcParams['figure.facecolor']
        axescolor = plt.rcParams['axes.facecolor']

        fig, ax = plt.subplots(figsize=(8, 7))

        # Force the background colors to override IDE defaults
        fig.patch.set_facecolor(facecolor)
        ax.set_facecolor(axescolor)

        colors = plt.cm.get_cmap(colormap)(np.linspace(0, 0.85, len(self.data_by_depth)))

        for i, (depth, data_df) in enumerate(self.data_by_depth.items()):
            color = colors[i]
            label_prefix = f'Depth {depth} cm' if self.depth_col in data_df.columns else 'Data'

            ax.plot(data_df['theta'], data_df['suction'], 'o', color=color, label=label_prefix)

            fit_result = self.fit_results.get(depth)
            if fit_result and fit_result.success:
                psi_smooth = np.logspace(np.log10(max(1e-2, data_df['suction'].min())),
                                         np.log10(data_df['suction'].max()), 200)
                theta_fit = fit_result.eval(psi=psi_smooth)
                ax.plot(theta_fit, psi_smooth, '-', color=color)

        ax.set_yscale('log')
        ax.set_ylabel('Soil Water Potential (cm) - Log Scale', fontsize=12)
        ax.set_xlabel('Volumetric Water Content ($cm^3/cm^3$)', fontsize=12)

        title = "Soil Water Retention Curves"
        subtitle = ""
        first_depth_key = next(iter(self.data_by_depth))
        data_df = self.data_by_depth[first_depth_key]
        if 'name' in data_df.columns and data_df['name'].iloc[0] is not None:
            station_name = data_df['name'].iloc[0]
            subtitle = f"({station_name})"
        elif self.filepath:
            subtitle = f"({os.path.basename(self.filepath).replace('.parquet', '')})"

        if subtitle:
            title += f"\n{subtitle}"

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, which="both", ls="--", c='0.7')

        ax.set_xlim(right=0.65)
        ax.set_ylim(top=10**7)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())
            print(f"Plot saved to {save_path}")
        if show:
            plt.show()

    @property
    def results_summary(self):
        """
        Returns a dictionary containing the key results of the fit for each depth.
        """
        if not self.fit_results:
            return {"status": "Fit not performed."}

        summary = {}
        for depth, result in self.fit_results.items():
            n_obs = len(self.data_by_depth[depth])
            if result and result.success:
                params = {}
                for name, param in result.params.items():
                    params[name] = {'value': param.value, 'stderr': param.stderr}
                summary[depth] = {
                    'status': 'Success',
                    'n_obs': n_obs,
                    'chi_squared': result.chisqr,
                    'reduced_chi_squared': result.redchi,
                    'aic': result.aic,
                    'bic': result.bic,
                    'parameters': params
                }
            else:
                summary[depth] = {'status': 'Fit Failed or Not Performed', 'n_obs': n_obs}
        return summary

    def save_results(self, output_dir):
        """
        Saves the fit results summary to a JSON file.

        Args:
            output_dir (str): The directory to save the results file in.
        """
        if not self.fit_results:
            print("Fit has not been performed yet. Cannot save results.")
            return

        if not self.filepath:
            print("Cannot determine output filename because data was not loaded from a file.")
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        base_name = os.path.basename(self.filepath)
        file_name_without_ext = os.path.splitext(base_name)[0]
        output_filename = f"{file_name_without_ext}_fit_results.json"
        output_path = os.path.join(output_dir, output_filename)

        summary_data = self.results_summary

        with open(output_path, 'w') as f:
            json.dump(summary_data, f, indent=4)

        print(f"Successfully saved fit results to {output_path}")


if __name__ == '__main__':

    home_ = os.path.expanduser('~')
    root = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'soil_potential_obs', 'mt_mesonet')

    data_ = os.path.join(root, 'preprocessed_by_station')
    results_ = os.path.join(root, 'results_by_station')
    plots_ = os.path.join(root, 'station_swrc_plots')

    station_files = [os.path.join(data_, f) for f in os.listdir(data_)]

    for station_file_ in station_files:

        plt_file = os.path.join(plots_, os.path.basename(station_file_.replace('parquet', 'png')))

        if os.path.exists(station_file_):
            swrc_fitter_ = SWRC(filepath=station_file_, depth_col='Depth [cm]')
            swrc_fitter_.fit(report=True)
            swrc_fitter_.plot(show=False, save_path=plt_file)
            swrc_fitter_.save_results(output_dir=results_)
            print(f'processed {station_file_}')

        else:
            print(f"Error: Data file not found at {station_file_}")

# ========================= EOF ====================================================================
