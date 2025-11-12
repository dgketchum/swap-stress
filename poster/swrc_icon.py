import os
import json
import numpy as np
import matplotlib.pyplot as plt


def _van_genuchten_model(psi, theta_r, theta_s, alpha, n):
    if n <= 1:
        return np.full_like(psi, np.nan)
    m = 1 - 1 / n
    psi_safe = np.maximum(psi, 1e-9)
    term = 1 + (alpha * psi_safe) ** n
    return theta_r + (theta_s - theta_r) / (term) ** m


def _get_param_value(p):
    if isinstance(p, dict) and 'value' in p:
        return float(p['value'])
    return float(p)


def plot_swrc_icon(results_path, out_path, depth_key=None, show=False, transparent=True):
    with open(results_path, 'r') as f:
        summary = json.load(f)

    keys = [k for k in summary.keys() if k != 'metadata']
    if depth_key is None:
        numeric_keys = [k for k in keys if str(k).isdigit()]
        depth_key = min(numeric_keys, key=lambda x: int(x)) if numeric_keys else keys[0]

    entry = summary[depth_key]
    d = entry.get('data') or {}
    psi_obs = np.asarray(d.get('suction_cm') or [])
    theta_obs = np.asarray(d.get('theta') or [])

    p = entry['parameters']
    theta_r = _get_param_value(p['theta_r'])
    theta_s = _get_param_value(p['theta_s'])
    alpha = _get_param_value(p['alpha'])
    n = _get_param_value(p['n'])

    if psi_obs.size:
        psi_min = max(1e-3, float(np.nanmin(psi_obs)))
        psi_max = float(np.nanmax(psi_obs))
    else:
        psi_min, psi_max = 1e-3, 1e6  # likely error if no data present

    psi_smooth = np.logspace(np.log10(psi_min), np.log10(psi_max), 300)
    theta_fit = _van_genuchten_model(psi_smooth, theta_r, theta_s, alpha, n)

    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=300)

    if psi_obs.size and theta_obs.size:
        ax.plot(theta_obs, psi_obs, 'o', ms=6.0, mec='none', color='#1f77b4', alpha=0.95)
    ax.plot(theta_fit, psi_smooth, '-', color='black', lw=4.0)

    if psi_obs.size and theta_obs.size:
        finite = np.isfinite(psi_obs) & np.isfinite(theta_obs)
        if np.any(finite):
            psi_p90 = float(np.percentile(psi_obs[finite], 90))
            psi_p10 = float(np.percentile(psi_obs[finite], 10))
            theta_p90 = float(_van_genuchten_model(np.array([psi_p90]), theta_r, theta_s, alpha, n)[0])
            theta_p10 = float(_van_genuchten_model(np.array([psi_p10]), theta_r, theta_s, alpha, n)[0])
            ax.plot(theta_p90, psi_p90, 'o', ms=8.0, color='black')
            ax.annotate(r'$\theta_{r}$', xy=(theta_p90, psi_p90), xytext=(10, 0), textcoords='offset points',
                        fontsize=14, ha='left', va='center')
            ax.plot(theta_p10, psi_p10, 'o', ms=8.0, color='black')
            ax.annotate(r'$\theta_{s}$', xy=(theta_p10, psi_p10), xytext=(10, 0), textcoords='offset points',
                        fontsize=14, ha='left', va='center')

    ax.set_yscale('log')
    ax.set_title('van Genuchten Model Fitting', fontsize=14, fontweight='bold')
    ax.set_xlabel('Volumetric Soil Water Content θ', fontsize=16)
    ax.set_ylabel('Soil Water Potential ψ', fontsize=16)

    ax.set_xlim(0.0, 0.65)
    ax.set_ylim(psi_min, psi_max)

    for sp in ax.spines.values():
        sp.set_linewidth(1.6)
    ax.tick_params(axis='both', which='both', labelsize=12, width=1.2, length=4.0)
    ax.grid(False)

    fig.tight_layout(pad=0.6)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=300, transparent=transparent)
    if show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    home_ = os.path.expanduser('~')
    root_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'soil_potential_obs')
    fits_dir_ = os.path.join(root_, 'curve_fits', 'reesh', 'bayes')
    site_base_ = 'IN-Martell'
    candidates_ = [f for f in os.listdir(fits_dir_) if f.startswith(site_base_) and f.endswith('.json')]
    results_path_ = os.path.join(fits_dir_, sorted(candidates_)[0])
    out_path_ = os.path.join('poster', f'swrc_icon_{site_base_.lower().replace('-', '_')}.png')
    plot_swrc_icon(results_path_, out_path_, show=True, transparent=True)
# ========================= EOF ====================================================================
