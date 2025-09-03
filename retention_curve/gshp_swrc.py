import numpy as np
import pandas as pd
from .swrc import SWRC


class GshpSWRC(SWRC):
    """
    GSHP-specific subclass of SWRC that applies the GSHP fitting
    procedure (bounds and preprocessing) for YWYD, YWND, NWYD, NWND.

    Keeps the same public API as SWRC.
    """

    def __init__(self, filepath=None, depth_col=None, df=None):
        super().__init__(filepath=filepath, depth_col=depth_col, df=df)
        self._theta_s_ptf = None  # holds {'beta': [...], 'xtx_inv': [[...]], 'sigma': float}

    # --------- PTF training, loading, and setting ---------
    @staticmethod
    def estimate_theta_s_ptf(all_df: pd.DataFrame):
        """
        Train a simple OLS PTF for theta_s using only YWYD rows:
          thetas ~ db_od + clay% + sand% + tropical_flag
        Returns a JSON-serializable dict with beta, xtx_inv, sigma.
        """
        needed = ['thetas', 'db_od', 'sand_tot_psa_percent', 'clay_tot_psa_percent', 'climate_classes', 'SWCC_classes']
        if not all(c in all_df.columns for c in needed):
            return None
        train = all_df.dropna(subset=['thetas', 'db_od', 'sand_tot_psa_percent', 'clay_tot_psa_percent'])
        train = train[train['SWCC_classes'] == 'YWYD']
        if len(train) < 20:
            return None
        reg = (train['climate_classes'].astype(str).str.contains('Tropical', case=False, na=False)).astype(float)
        X = np.column_stack([
            np.ones(len(train)),
            train['db_od'].astype(float).values,
            train['clay_tot_psa_percent'].astype(float).values,
            train['sand_tot_psa_percent'].astype(float).values,
            reg.values,
        ])
        y = train['thetas'].astype(float).values
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        sigma = float(np.sqrt(np.maximum(np.mean(resid ** 2), 1e-12)))
        xtx_inv = np.linalg.pinv(X.T @ X)
        return {
            'beta': beta.tolist(),
            'xtx_inv': xtx_inv.tolist(),
            'sigma': sigma,
        }

    def set_theta_s_ptf(self, model_dict: dict):
        self._theta_s_ptf = model_dict

    def load_theta_s_ptf(self, json_path: str):
        import json
        with open(json_path, 'r') as f:
            self._theta_s_ptf = json.load(f)

    def fit(self, report=False, method='nelder-meade'):
        self.fit_results = {}

        # Expect a preloaded theta_s PTF (set via set_theta_s_ptf or load_theta_s_ptf). If absent, skip θs PI bounds.
        theta_s_model = None
        xtx_inv = None
        sigma = None
        if isinstance(self._theta_s_ptf, dict):
            try:
                theta_s_model = np.array(self._theta_s_ptf.get('beta'), dtype=float)
                xtx_inv = np.array(self._theta_s_ptf.get('xtx_inv'), dtype=float)
                sigma = float(self._theta_s_ptf.get('sigma'))
            except Exception:
                theta_s_model = None

        for depth, data_df in self.data_by_depth.items():
            print(f"--- Fitting for Depth: {depth} cm ---")
            print(f"--- {len(data_df)} data points ---")
            # Prepare per-depth data in GSHP fashion
            df_fit = data_df.dropna(subset=['suction', 'theta']).copy()
            try:
                df_fit.loc[df_fit['suction'] <= 0, 'suction'] = 1.0
            except Exception:
                pass
            if len(df_fit) < 4:
                print("Insufficient observations (<4), skipping.")
                self.fit_results[depth] = None
                continue

            initial_params = self._generate_initial_params(df_fit)
            # Base GSHP bounds
            try:
                initial_params['theta_r'].set(min=0.0, max=1.0)
                initial_params['theta_s'].set(min=0.0, max=1.0)
                initial_params['alpha'].set(min=1.490116e-07, max=100.0)
                initial_params['n'].set(min=1.0, max=7.0)
            except Exception:
                pass

            # Class-aware bounds
            try:
                cls = str(data_df.get('SWCC_classes', pd.Series(['YWYD']))).split('\n')[0]
            except Exception:
                cls = 'YWYD'

            # theta_r PTF bounds (YWND, NWND)
            if cls in ('YWND', 'NWND'):
                try:
                    row0 = data_df.iloc[0]
                    sand = float(row0.get('sand_tot_psa_percent')) if pd.notnull(row0.get('sand_tot_psa_percent')) else np.nan
                    silt = float(row0.get('silt_tot_psa_percent')) if pd.notnull(row0.get('silt_tot_psa_percent')) else np.nan
                    clay = float(row0.get('clay_tot_psa_percent')) if pd.notnull(row0.get('clay_tot_psa_percent')) else np.nan
                    db_od = float(row0.get('db_od')) if pd.notnull(row0.get('db_od')) else np.nan
                    if np.isfinite([sand, silt, clay, db_od]).all():
                        sand_g = sand / 100.0
                        silt_g = silt / 100.0
                        clay_g = clay / 100.0
                        sa_cm2_g = sand_g * 444.0 + silt_g * 11100.0 + clay_g * 7400000.0
                        sa_m2_kg = sa_cm2_g / 10.0
                        thetar_lwr = sa_m2_kg * 997.0 * 3.5e-10 * db_od
                        thetar_upr = sa_m2_kg * 997.0 * 7.0e-10 * db_od
                        lo = max(0.0, float(thetar_lwr))
                        hi = min(1.0, float(thetar_upr))
                        if hi > lo:
                            initial_params['theta_r'].set(min=lo, max=hi)
                except Exception:
                    pass

            # theta_s PTF bounds (NWYD, NWND)
            if cls in ('NWYD', 'NWND') and theta_s_model is not None and xtx_inv is not None and sigma is not None:
                try:
                    row0 = data_df.iloc[0]
                    reg = 1.0 if str(row0.get('climate_classes', '')).lower().find('tropical') >= 0 else 0.0
                    x = np.array([
                        1.0,
                        float(row0.get('db_od', np.nan)),
                        float(row0.get('clay_tot_psa_percent', np.nan)),
                        float(row0.get('sand_tot_psa_percent', np.nan)),
                        reg,
                    ], dtype=float)
                    if np.isfinite(x).all():
                        mu = float(x @ theta_s_model)
                        se_pred = float(np.sqrt(max(0.0, sigma ** 2 * (1.0 + x @ xtx_inv @ x))))
                        lo = max(0.0, mu - 1.96 * se_pred)
                        hi = min(1.0, mu + 1.96 * se_pred)
                        if hi > lo:
                            initial_params['theta_s'].set(min=lo, max=hi)
                except Exception:
                    pass

            # Safety to keep theta_r <= theta_s
            try:
                tr_min = initial_params['theta_r'].min
                ts_min = initial_params['theta_s'].min
                tr_max = initial_params['theta_r'].max
                ts_max = initial_params['theta_s'].max
                if ts_min is not None and tr_min is not None and ts_min == 0:
                    initial_params['theta_s'].set(min=tr_min)
                if tr_max is not None and tr_max == 1 and ts_max is not None:
                    initial_params['theta_r'].set(max=ts_max)
            except Exception:
                pass

            print(f'--- Initial Parameter Values ---')
            [print(f'{k}: {v.value:.3f}') for k, v in initial_params.items()]
            try:
                result = self._vg_model.fit(
                    df_fit['theta'], initial_params, psi=df_fit['suction'],
                    method=method, nan_policy='raise'
                )
                self.fit_results[depth] = result
                if report:
                    print(result.fit_report())
            except Exception as e:
                print(f"An error occurred during fitting for depth {depth} cm: {e}")
                self.fit_results[depth] = None

        print("\nAll fits complete.")
        return self.fit_results

# ========================= EOF ====================================================================
