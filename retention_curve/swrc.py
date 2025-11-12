import os
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
import pymc as pm
try:
    from pymc.sampling.jax import sample_numpyro_nuts
except Exception:
    sample_numpyro_nuts = None
import pytensor.tensor as pt


class VGFitResultLight:
    """
    Lightweight stand-in for lmfit.ModelResult backed by saved parameters.
    Provides a compatible `.eval(psi=...)` for plotting without refitting.
    """
    def __init__(self, theta_r, theta_s, alpha, n):
        self.success = True
        self._theta_r = float(theta_r)
        self._theta_s = float(theta_s)
        self._alpha = float(alpha)
        self._n = float(n)
        # Minimal param objects with `.value`/`.stderr` for downstream consumers
        Param = lambda v: type("Param", (), {"value": float(v), "stderr": None})()
        self.params = {
            'theta_r': Param(theta_r),
            'theta_s': Param(theta_s),
            'alpha': Param(alpha),
            'n': Param(n),
        }

    def eval(self, psi=None, **kwargs):
        psi_arr = np.asarray(psi if psi is not None else kwargs.get('psi'), dtype=float)
        n = max(self._n, 1.0000001)
        m = 1.0 - 1.0 / n
        psi_safe = np.maximum(psi_arr, 1e-9)
        term = 1.0 + (self._alpha * psi_safe) ** n
        return self._theta_r + (self._theta_s - self._theta_r) / (term ** m)


class SWRC:
    """
    A class to encapsulate Soil Water Retention Curve (SWRC) data,
    fit the van Genuchten-Mualem model for one or more depths/groups,
    and visualize the results.

    Input data options:
      - Provide a file path (CSV or Parquet) with either:
          ('KPA','VWC') columns or ('suction_cm','theta') columns, and a depth column
      - Provide a pandas DataFrame via `df` having required columns:
          'suction' (head/psi in cm), 'theta' (VWC), and 'depth' (cm)

    Notes:
      - psi/head and depth units are centimeters (cm)

    Reference:
    Memari, S.S. and Clement, T.P., 2021. PySWR-A Python code for fitting soil water
    retention functions. Computers & Geosciences, 156, p.104897.

    """

    def __init__(self, filepath=None, depth_col=None, df=None):
        """
        Initialize the SWRC fitter from a file or a DataFrame.

        Args:
            filepath (str, optional): Path to a CSV or Parquet file.
            depth_col (str, optional): Column to group by; defaults to 'depth' for DataFrame input
                                       and 'stationDepth [cm]' for file input if present.
            df (pandas.DataFrame, optional): DataFrame with columns 'suction' [cm], 'theta', 'depth' [cm].
        """
        self.filepath = filepath
        self.data_by_depth = {}
        self.fit_results = {}
        self.metadata = {}
        self._vg_model = Model(self._van_genuchten_model)

        if df is not None:
            self.depth_col = depth_col or 'depth_cm'
            self.load_from_dataframe(df)
        elif filepath:
            # prefer standardized depth header
            self.depth_col = depth_col or 'depth_cm'
            self.load_from_file(filepath)

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

        # Ensure a consistent '_cm' column is available for downstream saving
        if 'suction_cm' not in df.columns:
            df['suction_cm'] = df['suction']

        if self.depth_col in df.columns:
            for depth, group in df.groupby(self.depth_col):
                self.data_by_depth[depth] = group
            # print(f"Data loaded and grouped by {len(self.data_by_depth)} depths.")
        else:
            self.data_by_depth[0] = df
            print("Data loaded as a single dataset (no depth column found).")

    def load_from_dataframe(self, df):
        """
        Load SWRC data from a DataFrame. Accepts columns: ('suction'|'suction_cm'), 'theta', ('depth'|'depth_cm').
        Groups by `self.depth_col` if present; otherwise treats as single dataset.
        """
        cols = set(df.columns)
        has_suction = ('suction' in cols) or ('suction_cm' in cols)
        has_depth = ('depth_cm' in cols) or ('depth' in cols)
        if not has_suction or 'theta' not in cols or not has_depth:
            raise ValueError(
                f"DataFrame missing required columns: need ('suction'|'suction_cm'), 'theta', ('depth'|'depth_cm')")

        # Normalize inputs: create internal 'suction' if only 'suction_cm' provided
        if 'suction' not in df.columns and 'suction_cm' in df.columns:
            df['suction'] = np.abs(df['suction_cm'].values)
        # Choose grouping column preferring explicit centimeters
        group_col = 'depth_cm' if 'depth_cm' in df.columns else 'depth'
        self.depth_col = group_col

        meta_cols = [c for c in df.columns if c not in {'suction_cm', 'theta', group_col}]
        meta_series = df[meta_cols + [group_col]].groupby(group_col).first()
        meta_series = meta_series.to_dict(orient='index')
        self.metadata = meta_series

        if 'suction' in df.columns:
            df['suction'] = np.abs(df['suction'].values)

        if group_col in df.columns:
            for key, group in df.groupby(group_col):
                self.data_by_depth[key] = group
            # print(f"DataFrame loaded and grouped by {self.depth_col} ({len(self.data_by_depth)} groups).")
        else:
            self.data_by_depth[0] = df
            print("DataFrame loaded as a single dataset (no grouping column found).")

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
        params['n'].set(value=n_init, min=1.001, max=10.0)
        return params

    def fit(self, report=False, method='nelder-meade'):
        """
        Fits the van Genuchten model to the SWRC data for each depth.

        Args:
            report (bool): If True, prints the detailed fit report for each depth.

        Returns:
            dict: A dictionary of lmfit.model.ModelResult objects, keyed by depth.
            :param report:
            :param method:
        """
        # special Bayesian pathway
        if str(method).lower() in {'bayes', 'fit_bayes', 'bayesian'}:
            return self.fit_bayesian()

        self.fit_results = {}
        for depth, data_df in self.data_by_depth.items():
            print(f"--- Fitting for Depth: {depth} cm ---")
            print(f"--- {len(data_df)} data points ---")
            initial_params = self._generate_initial_params(data_df)
            print(f'--- Initial Parameter Values ---')
            [print(f'{k}: {v.value:.3f}') for k, v in initial_params.items()]
            try:
                result = self._vg_model.fit(
                    data_df['theta'], initial_params, psi=data_df['suction'],
                    method=method, nan_policy='raise'  # likely error: lmfit uses 'nelder' not 'nelder-meade'
                )
                self.fit_results[depth] = result
                if report:
                    print(result.fit_report())
            except Exception as e:
                print(f"An error occurred during fitting for depth {depth} cm: {e}")
                self.fit_results[depth] = None
        print("\nAll fits complete.")
        return self.fit_results

    def fit_bayesian(self, draws=2000, tune=1000, target_accept=0.9, chains=4, cores=4, random_seed=None,
                     theta_r_cap=None, theta_s_floor=None, theta_s_upper=0.65):
        self.bayes_results = {}
        for depth, data_df in self.data_by_depth.items():
            d = data_df.dropna(subset=['suction', 'theta'])
            if len(d) < 4:
                self.bayes_results[depth] = None
                continue
            psi = d['suction'].astype(float).values
            theta = d['theta'].astype(float).values
            with pm.Model() as model:
                # theta_r: truncated Normal near 0.01, optionally capped by time-series min
                cap = None if theta_r_cap is None else max(1e-6, min(float(theta_r_cap), 0.99))
                upper_tr = cap if cap is not None else 0.05
                theta_r = pm.TruncatedNormal('theta_r', mu=0.01, sigma=0.05, lower=0.0, upper=upper_tr)

                # theta_s: truncated Normal near 0.45 with bounds [floor, upper], ensure > theta_r
                floor_val = 0.45 if theta_s_floor is None else float(theta_s_floor)
                upper_val = float(theta_s_upper) if theta_s_upper is not None else 0.65
                eps = 1e-6
                lower_ts = pt.maximum(floor_val, theta_r + eps)
                theta_s = pm.TruncatedNormal('theta_s', mu=0.45, sigma=0.15, lower=lower_ts, upper=upper_val)

                # alpha: recentered to ~0.0098 1/cm (a ≈ 100 1/MPa)
                log_alpha = pm.Normal('log_alpha', mu=np.log(0.0098), sigma=1.0)
                alpha = pm.Deterministic('alpha', pt.exp(log_alpha))

                # n: truncated Normal around 1
                n = pm.TruncatedNormal('n', mu=1.0, sigma=1.0, lower=1.001, upper=8.0)
                m = 1.0 - 1.0 / n
                psi_safe = pt.maximum(psi, 1e-9)
                term = 1.0 + (alpha * psi_safe) ** n
                mu_theta = theta_r + (theta_s - theta_r) / (term ** m)
                sigma = pm.HalfNormal('sigma', sigma=0.05)
                pm.Normal('obs', mu=mu_theta, sigma=sigma, observed=theta)
                trace = pm.sample(
                    draws=draws, tune=tune, chains=chains, cores=cores,
                    target_accept=target_accept, random_seed=random_seed,
                    progressbar=False, return_inferencedata=True
                )
            self.bayes_results[depth] = trace

        return self.bayes_results

    def test_fit_methods(self, methods_to_test, depth=None):
        """
        Tests multiple fitting algorithms and returns a summary DataFrame.

        Args:
            methods_to_test (list): A list of strings with the names of the
                                    fitting methods to test.
            depth (any, optional): The specific depth to test. If None, the first
                                   available depth is used.

        Returns:
            pandas.DataFrame: A DataFrame summarizing the results, sorted by AIC.
        """
        if not self.data_by_depth:
            print("No data loaded.")
            return

        if depth is None:
            depth = next(iter(self.data_by_depth))

        data_df = self.data_by_depth.get(depth)
        if data_df is None:
            print(f"Depth {depth} not found in data.")
            return

        print(f"--- Testing Fit Methods for Depth: {depth} cm ---")

        # Clean data once
        data_df = data_df.dropna(subset=['suction', 'theta'])
        data_df = data_df[np.isfinite(data_df['suction']) & np.isfinite(data_df['theta'])]

        initial_params = self._generate_initial_params(data_df)
        results_list = []

        for method in methods_to_test:
            print(f"\n--- Testing method: {method} ---")
            start_time = time.time()
            try:
                result = self._vg_model.fit(data_df['theta'], initial_params,
                                            psi=data_df['suction'],
                                            method=method, nan_policy='raise')

                duration = time.time() - start_time
                if result.success:
                    summary = {'method': method, 'success': True, 'AIC': result.aic, 'BIC': result.bic,
                               'time (s)': duration}
                    for param_name, param_obj in result.params.items():
                        summary[f'{param_name}'] = param_obj.value
                        summary[f'{param_name}_stderr'] = param_obj.stderr
                    results_list.append(summary)
                else:
                    results_list.append(
                        {'method': method, 'success': False, 'AIC': np.inf, 'BIC': np.inf, 'time (s)': duration})

            except Exception as e:
                duration = time.time() - start_time
                print(f"Method '{method}' failed with an error: {e}")
                results_list.append(
                    {'method': method, 'success': False, 'AIC': np.inf, 'BIC': np.inf, 'time (s)': duration})

        if not results_list:
            print("No methods completed successfully.")
            return pd.DataFrame()

        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)

        param_names = ['theta_r', 'theta_s', 'alpha', 'n']

        rel_err_series = []
        for p in param_names:
            val_col = f'{p}'
            err_col = f'{p}_stderr'
            if val_col in results_df.columns and err_col in results_df.columns:
                rel_err = (results_df[err_col] / np.abs(results_df[val_col])) * 100
                rel_err_series.append(rel_err)

        if rel_err_series:
            avg_rel_err = pd.concat(rel_err_series, axis=1).mean(axis=1)
            results_df['avg_rel_err (%)'] = avg_rel_err

        return results_df

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
        ax.set_ylim(top=10 ** 7)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())
            print(f"Plot saved to {save_path}")
            plt.close()
        if show:
            plt.show()

    @property
    def results_summary(self):
        """
        Returns a dictionary containing the key results of the fit for each depth.
        Also stores raw data points for plotting (suction [cm], theta) and optional metadata.
        """
        if not self.fit_results:
            return {"status": "Fit not performed."}

        summary = {}
        for depth, result in self.fit_results.items():
            n_obs = len(self.data_by_depth[depth])
            # Prepare raw data snapshot for plotting
            df_raw = self.data_by_depth[depth]
            df_raw = df_raw.dropna(subset=['suction', 'theta'])
            df_raw = df_raw[np.isfinite(df_raw['suction']) & np.isfinite(df_raw['theta'])]
            data_blob = {
                'suction_cm': df_raw['suction'].astype(float).tolist(),
                'theta': df_raw['theta'].astype(float).tolist(),
            }
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
                    'parameters': params,
                    'data': data_blob,
                }
            else:
                summary[depth] = {
                    'status': 'Fit Failed or Not Performed',
                    'n_obs': n_obs,
                    'data': data_blob,
                }
        if self.metadata:
            summary['metadata'] = self.metadata.copy()

        return summary

    def save_results(self, output_dir, output_filename=None, add_data=None):
        """
        Saves the fit results summary to a JSON file.

        Args:
            output_dir (str): The directory to save the results file in.
            output_filename (str, optional): Output filename. If None, derives from input file
                                             name when available, else uses 'swrc_fit_results.json'.
        """
        if not self.fit_results:
            print("Fit has not been performed yet. Cannot save results.")
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        if output_filename is None:
            if self.filepath:
                base_name = os.path.basename(self.filepath)
                file_name_without_ext = os.path.splitext(base_name)[0]
                output_filename = f"{file_name_without_ext}_fit_results.json"
            else:
                output_filename = "swrc_fit_results.json"
        output_path = os.path.join(output_dir, output_filename)

        summary_data = self.results_summary

        if add_data:
            [summary_data[k].update(v) for k, v in add_data.items()]

        with open(output_path, 'w') as f:
            json.dump(summary_data, f, indent=4)

        print(f"Successfully saved fit results to {output_path}")


    # moved: test_fit_methods_across_stations now in viz/fitting_comparisons/project_fits.py

    def save_bayes_results(self, output_dir, output_filename=None):
        if not hasattr(self, 'bayes_results') or not self.bayes_results:
            return
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if output_filename is None:
            if self.filepath:
                base_name = os.path.basename(self.filepath)
                file_name_without_ext = os.path.splitext(base_name)[0]
                output_filename = f"{file_name_without_ext}_bayes_results.json"
            else:
                output_filename = "swrc_bayes_results.json"
        output_path = os.path.join(output_dir, output_filename)

        summary = {}
        for depth, trace in self.bayes_results.items():
            if trace is None:
                summary[depth] = {'status': 'Fit Failed or Not Performed'}
                continue
            params = {}
            for name in ['theta_r', 'theta_s', 'alpha', 'n']:
                if hasattr(trace.posterior, name):
                    arr = getattr(trace.posterior, name).values.reshape(-1)
                    mean_ = float(np.mean(arr))
                    params[name] = {
                        'value': mean_,  # align with downstream readers
                        'q025': float(np.quantile(arr, 0.025)),
                        'q975': float(np.quantile(arr, 0.975)),
                    }
            df_raw = self.data_by_depth.get(depth)
            if df_raw is not None:
                d = df_raw.dropna(subset=['suction', 'theta'])
                d = d[np.isfinite(d['suction']) & np.isfinite(d['theta'])]
                data_blob = {
                    'suction_cm': d['suction'].astype(float).tolist(),
                    'theta': d['theta'].astype(float).tolist(),
                }
                n_obs = int(len(d))
            else:
                data_blob = None
                n_obs = 0
            summary[depth] = {
                'status': 'Success',
                'n_obs': n_obs,
                'parameters': params,
                'data': data_blob,
            }

        if self.metadata:
            summary['metadata'] = self.metadata.copy()

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=4)

    # -------- Loading from saved results for plotting without refitting --------
    def load_from_results_json(self, results_path):
        """
        Loads a saved results JSON (from save_results or save_bayes_results) and
        reconstructs minimal data and fit objects sufficient for plotting.

        After calling, `self.data_by_depth` and `self.fit_results` are populated
        and `self.depth_col` is set to 'depth_cm'.
        """
        if not os.path.exists(results_path):
            raise FileNotFoundError(results_path)

        with open(results_path, 'r') as f:
            summary = json.load(f)

        self.data_by_depth = {}
        self.fit_results = {}
        self.depth_col = 'depth_cm'
        self.filepath = results_path

        # Capture optional metadata if present
        if isinstance(summary, dict) and 'metadata' in summary and isinstance(summary['metadata'], dict):
            self.metadata = summary['metadata']

        for depth_key, entry in summary.items():
            if depth_key == 'metadata':
                continue
            entry = entry or {}
            # Convert depth keys to numeric when possible
            try:
                depth_val = int(depth_key)
            except Exception:
                try:
                    depth_val = float(depth_key)
                except Exception:
                    depth_val = depth_key

            # Rebuild raw data DataFrame
            data_blob = entry.get('data') or {}
            suction = data_blob.get('suction_cm') or []
            theta = data_blob.get('theta') or []
            if len(suction) and len(theta) and len(suction) == len(theta):
                df = pd.DataFrame({
                    'suction': np.asarray(suction, dtype=float),
                    'theta': np.asarray(theta, dtype=float),
                    'depth_cm': depth_val,
                })
            else:
                # Create an empty frame to keep structure consistent
                df = pd.DataFrame({'suction': [], 'theta': [], 'depth_cm': []})

            self.data_by_depth[depth_val] = df

            # Rebuild a lightweight fit result object when parameters present
            params = (entry.get('parameters') or {})
            try:
                tr = params.get('theta_r', {}).get('value')
                ts = params.get('theta_s', {}).get('value')
                al = params.get('alpha', {}).get('value')
                n_ = params.get('n', {}).get('value')
            except Exception:
                tr = ts = al = n_ = None

            if all(v is not None for v in (tr, ts, al, n_)):
                self.fit_results[depth_val] = VGFitResultLight(tr, ts, al, n_)
            else:
                self.fit_results[depth_val] = None

        return self


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
