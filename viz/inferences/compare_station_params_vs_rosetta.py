import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from retention_curve import parse_polaris_depth_from_asset, map_polaris_depth_range_to_rosetta_level

PARAMS = ['theta_r', 'theta_s', 'alpha', 'n']
LOG10_PARAMS = {'alpha', 'n'}


def _find_rosetta_value(row, level, param):
    pat_plain = re.compile(fr"_L{int(level)}_VG_{param}$", re.IGNORECASE)
    pat_log = re.compile(fr"_L{int(level)}_VG_log10_{param}$", re.IGNORECASE)
    cols = row.index
    col_plain = next((c for c in cols if pat_plain.search(c)), None)
    col_log = next((c for c in cols if pat_log.search(c)), None)
    if col_plain is not None:
        return row[col_plain]
    if col_log is not None:
        v = row[col_log]
        try:
            return 10 ** float(v)
        except Exception:
            return np.nan
    return np.nan


def _prep_series(y_true, y_pred, param):
    s = pd.DataFrame({'true': y_true, 'pred': y_pred})
    s[s['true'] < -9999] = np.nan
    s[s['pred'] < -9999] = np.nan
    s = s.dropna()
    if param in ('theta_r', 'theta_s'):
        s = s[(s['true'] >= 0) & (s['true'] <= 1) & (s['pred'] >= 0) & (s['pred'] <= 1)]
    if param == 'alpha':
        # compare in log10 space for stability
        s['true'] = np.log10(s['true'])
        s['pred'] = np.log10(s['pred'])
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if param == 'n':
        s = s[(s['true'] > 0) & (s['pred'] > 0)]
        s['true'] = np.log10(s['true'])
        s['pred'] = np.log10(s['pred'])
    return s


def _r2(y_true, y_pred):
    if y_true.size == 0:
        return np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / ss_tot if ss_tot != 0 else np.nan)


def _rmse(y_true, y_pred):
    if y_true.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _load_polaris_by_station_level(polaris_parquet):
    df = pd.read_parquet(polaris_parquet)
    if 'station' not in df.columns:
        return pd.DataFrame(columns=['station', 'rosetta_level'])
    df = df.copy()
    df['station'] = df['station'].astype(str).str.lower().str.replace('_', '-', regex=False)
    if 'rosetta_level' not in df.columns:
        if {'depth_min_cm', 'depth_max_cm'}.issubset(df.columns):
            df['rosetta_level'] = [map_polaris_depth_range_to_rosetta_level(a, b) for a, b in
                                   zip(df['depth_min_cm'], df['depth_max_cm'])]
        else:
            if 'layer' in df.columns:
                depths = [parse_polaris_depth_from_asset(v) for v in df['layer']]
                levs = [map_polaris_depth_range_to_rosetta_level(a, b) if a is not None else np.nan for a, b in depths]
                df['rosetta_level'] = levs
            else:
                df['rosetta_level'] = np.nan  # likely error: missing depth info for POLARIS
    keep = ['alpha_mean', 'n_mean', 'theta_r_mean', 'theta_s_mean']
    keep = [c for c in keep if c in df.columns]
    g = df.groupby(['station', 'rosetta_level'])[keep].mean().reset_index()
    ren = {
        'theta_r_mean': 'pol_theta_r',
        'theta_s_mean': 'pol_theta_s',
        'alpha_mean': 'pol_alpha',
        'n_mean': 'pol_n',
    }
    for c in keep:
        g[ren.get(c, c)] = g[c]
    use_cols = ['station', 'rosetta_level'] + [ren[c] for c in keep]
    g = g[use_cols]
    return g


def compare_station_params_vs_rosetta(training_parquet, out_dir, make_scatter=True, polaris_parquet=None):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_parquet(training_parquet)

    keep = [c for c in ['station', 'depth', 'rosetta_level'] if c in df.columns]
    keep += [c for c in PARAMS if c in df.columns]
    # plus all rosetta columns
    rosetta_cols = [c for c in df.columns if re.search(r"_L\d+_VG_", c)]
    use = df[keep + rosetta_cols].copy()

    rows = []
    for idx, row in use.iterrows():
        lvl = row.get('rosetta_level')
        if pd.isna(lvl):
            continue
        rec = {'station': row.get('station'), 'depth': row.get('depth'), 'rosetta_level': int(lvl)}
        ok = True
        for p in PARAMS:
            if p not in row.index:
                ok = False
                break
            rec[f'fit_{p}'] = row[p]
            rec[f'ros_{p}'] = _find_rosetta_value(row, int(lvl), p)
        if ok:
            rows.append(rec)

    if not rows:
        print('no comparable rows found')
        return

    comp = pd.DataFrame(rows)
    metrics = []
    plot_dir = os.path.join(out_dir, 'plots')
    if make_scatter:
        plt.style.use('seaborn-v0_8-whitegrid')
        os.makedirs(plot_dir, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        fig.patch.set_facecolor('white')
        axes = axes.ravel()
    for i, p in enumerate(PARAMS):
        s = _prep_series(comp[f'fit_{p}'], comp[f'ros_{p}'], p)
        if s.empty:
            if make_scatter:
                axes[i].axis('off')
            continue
        r2 = _r2(s['true'].values, s['pred'].values)
        rmse = _rmse(s['true'].values, s['pred'].values)
        bias = float(np.mean(s['pred'].values - s['true'].values))
        metrics.append({'param': p, 'n': len(s), 'r2': r2, 'rmse': rmse, 'bias': bias})

        if make_scatter:
            ax = axes[i]
            vmin = float(min(s['true'].min(), s['pred'].min()))
            vmax = float(max(s['true'].max(), s['pred'].max()))
            rng = vmax - vmin
            pad = 0.05 * rng if rng > 0 else 0.05
            vmin_m, vmax_m = vmin - pad, vmax + pad
            ax.scatter(s['true'], s['pred'], s=12, alpha=0.6, color='#1f77b4', edgecolors='none', rasterized=True)
            ax.plot([vmin_m, vmax_m], [vmin_m, vmax_m], color='0.3', lw=1.2, ls='--')
            ax.set_xlim(vmin_m, vmax_m)
            ax.set_ylim(vmin_m, vmax_m)
            ax.set_aspect('equal', adjustable='box')
            label = f"log10({p})" if p in LOG10_PARAMS else p
            ax.set_xlabel(f"Fit {label}")
            ax.set_ylabel(f"Rosetta {label}")
            ax.set_title(f"{p}")
            ax.text(0.02, 0.98, f"n={len(s)}\nR²={r2:.2f}\nRMSE={rmse:.3f}", transform=ax.transAxes,
                    ha='left', va='top', fontsize=9, bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.8'))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(direction='out', length=4, width=0.8)

    if make_scatter:
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'station_vs_rosetta_all_params.png'), dpi=400)
        plt.close(fig)

    if metrics:
        md = pd.DataFrame(metrics)
        md.to_csv(os.path.join(out_dir, 'station_vs_rosetta_metrics.csv'), index=False)
        comp.to_parquet(os.path.join(out_dir, 'station_vs_rosetta_compact.parquet'), index=False)

    if polaris_parquet:
        pol = _load_polaris_by_station_level(polaris_parquet)
        if not pol.empty:
            m = comp.merge(pol, on=['station', 'rosetta_level'], how='left')
            p_metrics = []
            if make_scatter:
                plt.style.use('seaborn-v0_8-whitegrid')
                p_plot_dir = os.path.join(out_dir, 'plots')
                os.makedirs(p_plot_dir, exist_ok=True)
                p_fig, p_axes = plt.subplots(2, 2, figsize=(10, 10))
                p_fig.patch.set_facecolor('white')
                p_axes = p_axes.ravel()
            for i, p in enumerate(PARAMS):
                col = f'pol_{p}'
                if col not in m.columns:
                    if make_scatter:
                        p_axes[i].axis('off')
                    continue
                s = _prep_series(m[f'fit_{p}'], m[col], p)
                if s.empty:
                    if make_scatter:
                        p_axes[i].axis('off')
                    continue
                r2 = _r2(s['true'].values, s['pred'].values)
                rmse = _rmse(s['true'].values, s['pred'].values)
                bias = float(np.mean(s['pred'].values - s['true'].values))
                p_metrics.append({'param': p, 'n': len(s), 'r2': r2, 'rmse': rmse, 'bias': bias})

                if make_scatter:
                    ax = p_axes[i]
                    vmin = float(min(s['true'].min(), s['pred'].min()))
                    vmax = float(max(s['true'].max(), s['pred'].max()))
                    rng = vmax - vmin
                    pad = 0.05 * rng if rng > 0 else 0.05
                    vmin_m, vmax_m = vmin - pad, vmax + pad
                    ax.scatter(s['true'], s['pred'], s=12, alpha=0.6, color='#2ca02c', edgecolors='none', rasterized=True)
                    ax.plot([vmin_m, vmax_m], [vmin_m, vmax_m], color='0.3', lw=1.2, ls='--')
                    ax.set_xlim(vmin_m, vmax_m)
                    ax.set_ylim(vmin_m, vmax_m)
                    ax.set_aspect('equal', adjustable='box')
                    label = f"log10({p})" if p in LOG10_PARAMS else p
                    ax.set_xlabel(f"Fit {label}")
                    ax.set_ylabel(f"POLARIS {label}")
                    ax.set_title(f"{p}")
                    ax.text(0.02, 0.98, f"n={len(s)}\nR²={r2:.2f}\nRMSE={rmse:.3f}", transform=ax.transAxes,
                            ha='left', va='top', fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.8'))
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.tick_params(direction='out', length=4, width=0.8)

            if p_metrics:
                pd.DataFrame(p_metrics).to_csv(os.path.join(out_dir, 'station_vs_polaris_metrics.csv'), index=False)
            if make_scatter:
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, 'plots', 'station_vs_polaris_all_params.png'), dpi=400)
                plt.close(p_fig)


if __name__ == '__main__':
    home_ = os.path.expanduser('~')
    training_pq_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'swapstress', 'training',
                                 'stations_training_table_250m.parquet')
    out_dir_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'swapstress', 'training', 'station_vs_rosetta')
    polaris_pq_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'polaris', 'polaris_stations.parquet')
    compare_station_params_vs_rosetta(training_pq_, out_dir_, make_scatter=True, polaris_parquet=polaris_pq_)
# ========================= EOF ====================================================================
