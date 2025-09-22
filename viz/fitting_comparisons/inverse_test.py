import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lmfit import Model

from retention_curve.swrc import SWRC


class InverseTest(SWRC):
    def __init__(self, filepath=None, depth_col=None, df=None, inverse_log=False, weight_inverse_psi=False):
        super().__init__(filepath=filepath, depth_col=depth_col, df=df)
        self._inverse_log = bool(inverse_log)
        self._weight_inverse_psi = bool(weight_inverse_psi)
        self._vg_model = Model(self._inverse_log10_model) if self._inverse_log else Model(self._inverse_van_genuchten_model)

    @staticmethod
    def _inverse_van_genuchten_model(theta, theta_r, theta_s, alpha, n):
        # inverse of VG: psi(theta) with m = 1 - 1/n
        n = np.maximum(n, 1.001)
        alpha = np.maximum(alpha, 1e-9)
        theta = np.clip(theta, 0.0, 1.0)
        m = 1.0 - 1.0 / n
        num = (theta_s - theta_r)
        den = np.maximum(theta - theta_r, 1e-12)
        ratio = np.maximum(num / den, 1.0)
        inner = np.maximum(ratio ** (1.0 / m) - 1.0, 0.0)
        psi = inner ** (1.0 / n) / alpha
        return psi

    @staticmethod
    def _inverse_log10_model(theta, theta_r, theta_s, alpha, n):
        psi = InverseTest._inverse_van_genuchten_model(theta, theta_r, theta_s, alpha, n)
        return np.log10(np.maximum(psi, 1e-9))

    def fit(self, report=True, method='nelder'):
        self.fit_results = {}
        for depth, data_df in self.data_by_depth.items():
            d = data_df.dropna(subset=['suction', 'theta'])
            d = d[np.isfinite(d['suction']) & np.isfinite(d['theta'])]
            if d.empty:
                self.fit_results[depth] = None
                continue
            params = super()._generate_initial_params(d)
            try:
                y = d['suction'] if not self._inverse_log else np.log10(np.maximum(d['suction'].values, 1e-9))
                w = None
                if self._weight_inverse_psi and not self._inverse_log:
                    w = 1.0 / np.maximum(d['suction'].values, 1.0)
                result = self._vg_model.fit(y, params, theta=d['theta'],
                                            method=method, nan_policy='raise', weights=w)
            except Exception as e:
                print(f'inverse fit failed at depth {depth}: {e}')
                result = None
            self.fit_results[depth] = result
            if report and result is not None:
                print(result.fit_report())
        return self.fit_results


def _van_genuchten_model(psi, theta_r, theta_s, alpha, n):
    if n <= 1:
        return np.full_like(psi, np.nan)
    m = 1 - 1 / n
    psi_safe = np.maximum(psi, 1e-9)
    term = 1 + (alpha * psi_safe) ** n
    return theta_r + (theta_s - theta_r) / (term ** m)


def _load_forward_fit_json(p):
    with open(p, 'r') as f:
        j = json.load(f)
    # pick the first numeric depth
    depths = [k for k in j.keys() if k.isdigit()]
    if not depths:
        return None, None, None
    depths = sorted(depths, key=lambda x: int(x))
    dkey = depths[0]
    params = j[dkey].get('parameters', {})
    pvals = {k: v.get('value') for k, v in params.items()}
    data = j[dkey].get('data', {})
    return int(dkey), pvals, data


def _find_reesh_csv(station_csv_dir, station_key):
    for fn in os.listdir(station_csv_dir):
        if station_key in fn and fn.endswith('.csv'):
            return os.path.join(station_csv_dir, fn)
    return None


def plot_inverse_vs_forward(station_key, reesh_csv_dir, forward_json_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = _find_reesh_csv(reesh_csv_dir, station_key)
    if not csv_path or not os.path.exists(forward_json_path):
        print('required inputs not found')
        return
    df = pd.read_csv(csv_path)
    inv = InverseTest(df=df, depth_col='depth_cm', inverse_log=True)
    inv.fit(report=False, method='nelder')

    depth_fwd, params_fwd, data_fwd = _load_forward_fit_json(forward_json_path)
    depth_keys = list(inv.data_by_depth.keys())
    depth_use = depth_fwd if depth_fwd in depth_keys else depth_keys[0]
    d_obs = inv.data_by_depth[depth_use].dropna(subset=['suction', 'theta'])
    d_obs = d_obs[np.isfinite(d_obs['suction']) & np.isfinite(d_obs['theta'])]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    ax = axes[0]
    ax.plot(d_obs['theta'], d_obs['suction'], 'o', ms=5, alpha=0.75, label='Observed')
    th_min = float(max(0.0, d_obs['theta'].min()))
    th_max = float(min(0.65, d_obs['theta'].max()))
    theta_grid = np.linspace(th_min + 1e-6, th_max, 200)
    inv_res = inv.fit_results.get(depth_use)
    if inv_res and inv_res.success:
        p = inv_res.params
        psi_pred = InverseTest._inverse_van_genuchten_model(theta_grid, p['theta_r'].value, p['theta_s'].value,
                                                            p['alpha'].value, p['n'].value)
        ax.plot(theta_grid, psi_pred, '-', lw=2, label='Inverse (nelder)')
        txt = f"θr={p['theta_r'].value:.3f}, θs={p['theta_s'].value:.3f}\nα={p['alpha'].value:.3g}, n={p['n'].value:.3f}"
        ax.text(0.02, 0.02, txt, transform=ax.transAxes, va='bottom', ha='left')
    ax.set_title(f'{station_key} inverse')
    ax.set_xlabel('theta')
    ax.set_ylabel('suction (cm)')
    ax.set_yscale('log')
    ax.set_xlim(0, 0.65)

    ax = axes[1]
    ax.plot(d_obs['theta'], d_obs['suction'], 'o', ms=5, alpha=0.75, label='Observed')
    if params_fwd and all(k in params_fwd for k in ('theta_r', 'theta_s', 'alpha', 'n')):
        psi_min = max(1e-3, float(d_obs['suction'].min()))
        psi_max = float(d_obs['suction'].max())
        psi_grid = np.logspace(np.log10(psi_min), np.log10(psi_max), 200)
        theta_pred = _van_genuchten_model(psi_grid, params_fwd['theta_r'], params_fwd['theta_s'],
                                          params_fwd['alpha'], params_fwd['n'])
        ax.plot(theta_pred, psi_grid, '-', lw=2, label='Forward (nelder)')
        txt = f"θr={params_fwd['theta_r']:.3f}, θs={params_fwd['theta_s']:.3f}\nα={params_fwd['alpha']:.3g}, n={params_fwd['n']:.3f}"
        ax.text(0.02, 0.02, txt, transform=ax.transAxes, va='bottom', ha='left')
    ax.set_title(f'{station_key} forward')
    ax.set_xlabel('theta')
    ax.set_yscale('log')
    ax.set_xlim(0, 0.65)

    for ax in axes:
        ax.grid(True, which='both', ls='--', alpha=0.5)

    fig.tight_layout()
    out_png = os.path.join(out_dir, f'inverse_vs_forward_{station_key}.png')
    fig.savefig(out_png, dpi=200)
    print(f'wrote {out_png}')
    plt.close(fig)


if __name__ == '__main__':
    home_ = os.path.expanduser('~')
    root_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'soil_potential_obs')

    station_key_ = 'US-MMS_SN'
    reesh_csv_dir_ = os.path.join(root_, 'preprocessed', 'reesh')
    forward_json_ = os.path.join(root_, 'curve_fits', 'reesh', 'nelder', f'{station_key_}.json')
    out_dir_ = os.path.join(root_, 'obs_fit_comparison')
    plot_inverse_vs_forward(station_key_, reesh_csv_dir_, forward_json_, out_dir_)
# ========================= EOF ====================================================================
