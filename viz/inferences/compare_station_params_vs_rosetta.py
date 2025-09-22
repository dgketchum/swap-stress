import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    s = pd.DataFrame({'true': y_true, 'pred': y_pred}).replace([-9999, -9999.0], np.nan).dropna()
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


def compare_station_params_vs_rosetta(training_parquet, out_dir, make_scatter=True):
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
    for p in PARAMS:
        s = _prep_series(comp[f'fit_{p}'], comp[f'ros_{p}'], p)
        if s.empty:
            continue
        r2 = _r2(s['true'].values, s['pred'].values)
        rmse = _rmse(s['true'].values, s['pred'].values)
        bias = float(np.mean(s['pred'].values - s['true'].values))
        metrics.append({'param': p, 'n': len(s), 'r2': r2, 'rmse': rmse, 'bias': bias})

        if make_scatter:
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(s['true'], s['pred'], s=10, alpha=0.7)
            vmin = float(min(s['true'].min(), s['pred'].min()))
            vmax = float(max(s['true'].max(), s['pred'].max()))
            ax.plot([vmin, vmax], [vmin, vmax], 'r--', lw=1)
            label = f"log10({p})" if p in LOG10_PARAMS else p
            ax.set_xlabel(f"Fit {label}")
            ax.set_ylabel(f"Rosetta {label}")
            ax.set_title(f"Station Fit vs Rosetta: {p} (R2={r2:.2f}, RMSE={rmse:.3f})")
            plt.tight_layout()
            plot_dir = os.path.join(out_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, f'station_vs_rosetta_{p}.png'), dpi=300)
            plt.close(fig)

    if metrics:
        md = pd.DataFrame(metrics)
        md.to_csv(os.path.join(out_dir, 'station_vs_rosetta_metrics.csv'), index=False)
        comp.to_parquet(os.path.join(out_dir, 'station_vs_rosetta_compact.parquet'), index=False)


if __name__ == '__main__':
    home_ = os.path.expanduser('~')
    training_pq_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'swapstress', 'training',
                                 'stations_training_table_250m.parquet')
    out_dir_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'swapstress', 'training', 'station_vs_rosetta')
    compare_station_params_vs_rosetta(training_pq_, out_dir_, make_scatter=True)
# ========================= EOF ====================================================================
