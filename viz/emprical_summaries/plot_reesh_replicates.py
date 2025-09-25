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


def plot_reesh_site_overlays(site_base, fits_dir, out_dir, show=True, colormap='plasma', max_panels=6):
    files = [f for f in os.listdir(fits_dir) if f.startswith(site_base) and f.endswith('.json')]
    if not files:
        return
    files = sorted(files)
    if max_panels is not None:
        files = files[:int(max_panels)]

    os.makedirs(out_dir, exist_ok=True)

    plt.style.use('seaborn-v0_8-darkgrid')

    # one subplot per replicate file
    n = len(files)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6.5 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    cmap = plt.cm.get_cmap(colormap)

    for i, fn in enumerate(files):
        ax = axes[i]
        path = os.path.join(fits_dir, fn)
        with open(path, 'r') as f:
            data = json.load(f)
        suffix = os.path.splitext(fn)[0]
        if suffix.startswith(site_base + '_'):
            suffix = suffix[len(site_base) + 1:]
        depth_keys = [k for k in data.keys() if str(k).isdigit()]
        dk_sorted = sorted(depth_keys, key=lambda x: int(x))
        colors = cmap(np.linspace(0, 0.85, len(dk_sorted))) if dk_sorted else []
        for c, depth_key in zip(colors, dk_sorted):
            entry = data[depth_key]
            d = entry.get('data') or {}
            psi_obs = np.asarray(d.get('suction_cm') or [])
            theta_obs = np.asarray(d.get('theta') or [])
            if psi_obs.size and theta_obs.size:
                ax.plot(theta_obs, psi_obs, 'o', color=c, ms=4, alpha=0.85, label=f'{depth_key} cm')
            params = entry['parameters']
            tr = params['theta_r']['value'] if isinstance(params['theta_r'], dict) else params['theta_r']
            ts = params['theta_s']['value'] if isinstance(params['theta_s'], dict) else params['theta_s']
            al = params['alpha']['value'] if isinstance(params['alpha'], dict) else params['alpha']
            nn = params['n']['value'] if isinstance(params['n'], dict) else params['n']
            if psi_obs.size:
                psi_min = max(1e-3, float(np.nanmin(psi_obs)))
                psi_max = float(np.nanmax(psi_obs))
            else:
                psi_min, psi_max = 1e-3, 1e7
            psi_smooth = np.logspace(np.log10(psi_min), np.log10(psi_max), 200)
            theta_pred = _van_genuchten_model(psi_smooth, float(tr), float(ts), float(al), float(nn))
            ax.plot(theta_pred, psi_smooth, '-', color=c, lw=2)
        if dk_sorted:
            skey = dk_sorted[0]
            sp = data[skey]['parameters']
            tr_s = sp['theta_r']['value'] if isinstance(sp['theta_r'], dict) else sp['theta_r']
            ts_s = sp['theta_s']['value'] if isinstance(sp['theta_s'], dict) else sp['theta_s']
            al_s = sp['alpha']['value'] if isinstance(sp['alpha'], dict) else sp['alpha']
            nn_s = sp['n']['value'] if isinstance(sp['n'], dict) else sp['n']
            txt = f"theta_r={float(tr_s):.3f}\ntheta_s={float(ts_s):.3f}\nalpha={float(al_s):.3f}\nn={float(nn_s):.3f}"
            ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=9, ha='left', va='bottom')
        ax.set_yscale('log')
        ax.set_title(suffix, fontsize=12)
        ax.grid(True, which='both', ls='--', c='0.7')
        ax.set_xlim(right=0.65)
        ax.set_ylim(top=10 ** 7)
        if i % ncols == 0:
            ax.set_ylabel('Soil Water Potential (cm)')
        else:
            ax.set_ylabel('')
        if i >= (n - ncols):
            ax.set_xlabel('Volumetric Water Content ($cm^3/cm^3$)')
        else:
            ax.set_xlabel('')
        if dk_sorted:
            ax.legend(fontsize=9, frameon=False)

    for j in range(i + 1, nrows * ncols):
        axes[j].axis('off')

    fig.suptitle(f"Soil Water Retention Curves\n({site_base})", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = os.path.join(out_dir, f'{site_base}_reesh_replicates_panels.png')
    fig.savefig(out_png, dpi=300, facecolor=fig.get_facecolor())
    if show:
        plt.show()


if __name__ == '__main__':
    home = os.path.expanduser('~')
    fits_dir_ = os.path.join(home, 'data', 'IrrigationGIS', 'soils', 'soil_potential_obs', 'curve_fits', 'reesh', 'bayes')
    out_dir_ = os.path.join(home, 'data', 'IrrigationGIS', 'soils', 'soil_potential_obs', 'reesh', 'figures')
    site_base_ = 'IN-Martell'
    plot_reesh_site_overlays(site_base_, fits_dir_, out_dir_, show=False)
# ========================= EOF ====================================================================
