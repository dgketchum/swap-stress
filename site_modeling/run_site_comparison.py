import os
import json
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from site_modeling.prep import (
    build_site_dataset_from_ameriflux,
    detect_drydown_mask,
    load_gridmet_series,
    load_ameriflux_halfhourly,
    find_ameriflux_file,
)
from site_modeling.model import (
    evaluate_theta_vs_psi,
    evaluate_multivariate,
    evaluate_lagged,
)


def _ensure_dirs(*dirs: str) -> None:
    [os.makedirs(d, exist_ok=True) for d in dirs]


def list_reesh_site_ids(shapefile: str, id_col: str = 'site_id') -> List[str]:
    gdf = gpd.read_file(shapefile)
    if id_col not in gdf.columns:
        raise KeyError(f'{id_col} not found in shapefile attributes')
    sids = [str(x) for x in gdf[id_col].dropna().unique().tolist()]
    return sids


# selection now handled in site_modeling.prep


def _fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    xx = x[mask].reshape(-1, 1)
    yy = y[mask]
    m = LinearRegression()
    m.fit(xx, yy)
    yhat = m.predict(xx)
    ss_res = float(np.sum((yy - yhat) ** 2))
    ss_tot = float(np.sum((yy - np.mean(yy)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(m.coef_[0]), float(m.intercept_), r2


def plot_scatter_with_fit(
        df: pd.DataFrame,
        x_cols: List[str],
        y_col: str,
        titles: List[str],
        out_path: str,
        point_alpha: float = 0.4,
        dpi: int = 150,
) -> None:
    n = len(x_cols)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()

    for i, (x_col, title) in enumerate(zip(x_cols, titles)):
        ax = axes_arr[i]
        d = df[[x_col, y_col]].dropna()
        x = d[x_col].to_numpy(dtype=float)
        y = d[y_col].to_numpy(dtype=float)
        ax.scatter(x, y, s=12, alpha=point_alpha, color='tab:blue', edgecolor='none')
        if len(d) > 2:
            slope, intercept, r2 = _fit_line(x, y)
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            ys = slope * xs + intercept
            ax.plot(xs, ys, color='tab:red', linewidth=2)
            ax.text(0.02, 0.98, f'R²={r2:.2f}\nslope={slope:.3g}', transform=ax.transAxes,
                    ha='left', va='top', fontsize=9, bbox=dict(boxstyle='round', fc='white', alpha=0.7))
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title)

    for j in range(i + 1, len(axes_arr)):
        fig.delaxes(axes_arr[j])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def evaluate_site(
        df: pd.DataFrame,
        targets: List[str],
        n_splits: int = 5,
        use_drydown: bool = True,
) -> Dict:
    metrics_all = evaluate_theta_vs_psi(df, targets=targets, mask=None, n_splits=n_splits)
    out = {'all': metrics_all}
    if use_drydown and 'theta' in df.columns:
        mask = detect_drydown_mask(df)
        metrics_dd = evaluate_theta_vs_psi(df, targets=targets, mask=mask, n_splits=n_splits)
        out['drydown'] = metrics_dd
    return out


def build_data(
        site_ids: List[str],
        vg_dir: str,
        out_root: str,
        amf_root_or_file: str,
        site_coords: Dict[str, Tuple[float, float]],
        overwrite: bool = False,
) -> Dict[str, pd.DataFrame]:
    data_dir = os.path.join(out_root, 'data')
    summary_dir = os.path.join(out_root, 'summary')
    _ensure_dirs(out_root, data_dir, summary_dir)
    summary_records: List[Dict] = []
    data_by_site: Dict[str, pd.DataFrame] = {}
    for sid in site_ids:

        if 'martell' in sid.lower():
            continue

        if 'cdm' not in sid.lower():
            continue

        data_fp = os.path.join(data_dir, f'{sid}.parquet')
        status = 'ok'
        reason: Optional[str] = None
        df: Optional[pd.DataFrame] = None
        if os.path.exists(data_fp) and not overwrite:
            try:
                df = pd.read_parquet(data_fp)
            except Exception as e:
                status = 'missing'
                reason = f'failed_to_read_existing: {e}'
        else:
            try:
                # Determine AmeriFlux file path if a directory was provided
                amf_fp = None
                if os.path.isdir(amf_root_or_file):
                    amf_fp = find_ameriflux_file(amf_root_or_file, sid, period='HH')
                else:
                    amf_fp = amf_root_or_file if os.path.exists(amf_root_or_file) else None

                if amf_fp is None:
                    status = 'missing'
                    reason = 'ameriflux_file_not_found'
                else:
                    # Pre-check AmeriFlux contents to refine reason codes
                    try:
                        daily_vwc, daily_flux = load_ameriflux_halfhourly(amf_fp)
                        if daily_flux is None:
                            status = 'missing'
                            reason = 'ameriflux_no_targets'
                        elif daily_vwc is None or daily_vwc.empty:
                            status = 'missing'
                            reason = 'ameriflux_no_swc'
                    except FileNotFoundError:
                        status = 'missing'
                        reason = 'ameriflux_file_not_found'
                    except ValueError as ve:
                        msg = str(ve)
                        if 'TIMESTAMP_START' in msg:
                            status = 'missing'
                            reason = 'ameriflux_format_error'
                        else:
                            status = 'missing'
                            reason = f'ameriflux_value_error: {ve}'
                    except Exception as e:
                        status = 'missing'
                        reason = f'ameriflux_read_error: {e}'

                    # Build only if still ok
                    if status == 'ok':
                        df = build_site_dataset_from_ameriflux(
                            site_id=sid,
                            amf_root_or_file=amf_root_or_file,
                            vg_dir=vg_dir,
                        )
                        if df is not None and not df.empty:
                            # Join gridMET if possible
                            if sid in site_coords:
                                lon, lat = site_coords[sid]
                                if isinstance(df.index, pd.DatetimeIndex) and not df.empty:
                                    start_date = str(df.index.min().date())
                                    end_date = str(df.index.max().date())
                                    try:
                                        gm = load_gridmet_series(lon=lon, lat=lat, start_date=start_date,
                                                                 end_date=end_date, variables=('pet', 'pr'))
                                        df = df.join(gm, how='left')
                                    except Exception as ge:
                                        # gridMET failure should not mark dataset missing
                                        print(f'gridMET join failed for {sid}: {ge}')
                            df.to_parquet(data_fp)
                            print(f'wrote {sid}; {df.shape[1]} columns, {len(df)} days to {data_fp}')
                        else:
                            status = 'missing'
                            reason = 'empty_dataset'
            except Exception as e:
                status = 'missing'
                reason = f'build_error: {e}'

        if df is not None and status == 'ok':
            data_by_site[sid] = df

        # Build per-site summary record
        try:
            rec: Dict[str, object] = {'site_id': sid, 'status': status}
            # Coordinates if available
            if sid in site_coords:
                lon, lat = site_coords[sid]
                rec['lon'] = float(lon)
                rec['lat'] = float(lat)
            if df is not None and status == 'ok':
                start_dt = df.index.min()
                end_dt = df.index.max()
                rec['start_date'] = str(getattr(start_dt, 'date', lambda: start_dt)())
                rec['end_date'] = str(getattr(end_dt, 'date', lambda: end_dt)())
                rec['n_ET'] = int(df['ET'].count()) if 'ET' in df.columns else 0
                rec['n_GPP'] = int(df['GPP'].count()) if 'GPP' in df.columns else 0
                rec['has_ET'] = bool(rec['n_ET'] > 0)
                rec['has_GPP'] = bool(rec['n_GPP'] > 0)
                rec['n_psi_cols'] = int(len([c for c in df.columns if isinstance(c, str) and c.startswith('psi_cm_')]))
                counts = df.count()
                for col, n in counts.items():
                    rec[f'n_{col}'] = int(n)
            else:
                rec['reason'] = (reason or 'unknown')[:254]
            summary_records.append(rec)
        except Exception as e:
            print(f'Failed to summarize {sid}: {e}')

    # Write summary CSV and GeoJSON
    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        csv_fp = os.path.join(summary_dir, 'site_data_summary.csv')
        summary_df.to_csv(csv_fp, index=False)
        print(f'wrote flux data summary to {csv_fp}')
        # GeoJSON: require lon/lat
        if 'lon' in summary_df.columns and 'lat' in summary_df.columns:
            try:
                gdf = gpd.GeoDataFrame(
                    summary_df.dropna(subset=['lon', 'lat']).copy(),
                    geometry=gpd.points_from_xy(summary_df.dropna(subset=['lon', 'lat'])['lon'],
                                                summary_df.dropna(subset=['lon', 'lat'])['lat']),
                    crs=4326,
                )
                gj_fp = os.path.join(summary_dir, 'site_data_summary.geojson')
                gdf.to_file(gj_fp, driver='GeoJSON')
                print(f'wrote GeoJSON summary to {gj_fp}')
            except Exception as e:
                print(f'Failed to write GeoJSON summary: {e}')
    return data_by_site


def run_regressions(
        out_root: str,
        targets: Optional[List[str]] = None,
        n_splits: int = 5,
        site_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    fig_dir = os.path.join(out_root, 'figures')
    met_dir = os.path.join(out_root, 'metrics')
    data_dir = os.path.join(out_root, 'data')
    _ensure_dirs(fig_dir, met_dir)

    rows: List[Dict] = []

    files: List[Tuple[str, str]] = []
    if site_ids:
        for sid in site_ids:
            files.append((sid, os.path.join(data_dir, f'{sid}.parquet')))
    else:
        for fp in sorted(os.listdir(data_dir)):
            if fp.endswith('.parquet'):
                sid = os.path.splitext(os.path.basename(fp))[0]
                files.append((sid, os.path.join(data_dir, fp)))

    for sid, fp in files:
        if not os.path.exists(fp):
            continue
        df = pd.read_parquet(fp)
        tgt_list = targets or ['GPP', 'ET', 'LE']
        tgt_list = [t for t in tgt_list if t in df.columns]
        if not tgt_list:
            print(f'No target columns found for {sid}.')
            continue

        d_base = df[['theta'] + tgt_list].copy()
        d_base['psi_cm'] = d_base['theta']
        base_all = evaluate_theta_vs_psi(d_base, targets=tgt_list, mask=None, n_splits=n_splits)
        base_dd = evaluate_theta_vs_psi(d_base, targets=tgt_list, mask=detect_drydown_mask(d_base), n_splits=n_splits)

        metrics_out: Dict[str, Dict] = {'theta': {'all': base_all, 'drydown': base_dd}, 'replicates': {}}
        rep_cols = [c for c in df.columns if isinstance(c, str) and c.startswith('psi_cm_')]
        for col in rep_cols:
            rep = col.split('__', 1)[1] if '__' in col else col
            d_rep = df[['theta'] + tgt_list].copy()
            d_rep['psi_cm'] = df[col]
            rep_all = evaluate_theta_vs_psi(d_rep, targets=tgt_list, mask=None, n_splits=n_splits)
            rep_dd = evaluate_theta_vs_psi(d_rep, targets=tgt_list, mask=detect_drydown_mask(d_rep), n_splits=n_splits)
            metrics_out['replicates'][rep] = {'all': rep_all, 'drydown': rep_dd}

        # Multivariate (control for PET/PR) and lagged analyses
        mv_enabled = all(c in df.columns for c in ('pet', 'pr'))
        if mv_enabled:
            metrics_out['mv'] = {'theta': {}, 'replicates': {}}
            metrics_out['lagged'] = {'theta': {}, 'replicates': {}}
            mask_dd = detect_drydown_mask(df) if 'theta' in df.columns else None
            for t in tgt_list:
                # MV with theta
                mv_theta_all = evaluate_multivariate(df, target=t, predictors=['theta', 'pet', 'pr'], mask=None,
                                                     n_splits=n_splits)
                mv_theta_dd = evaluate_multivariate(df, target=t, predictors=['theta', 'pet', 'pr'], mask=mask_dd,
                                                    n_splits=n_splits)
                metrics_out['mv']['theta'][t] = {'all': mv_theta_all, 'drydown': mv_theta_dd}
                # Lagged with theta (prototype set of lags)
                lag_theta_all = evaluate_lagged(df, target=t, base_predictor='theta', covariates=['pet', 'pr'],
                                                lags=(0, 3, 7, 14), mask=None, n_splits=n_splits)
                lag_theta_dd = evaluate_lagged(df, target=t, base_predictor='theta', covariates=['pet', 'pr'],
                                               lags=(0, 3, 7, 14), mask=mask_dd, n_splits=n_splits)
                metrics_out['lagged']['theta'][t] = {'all': lag_theta_all, 'drydown': lag_theta_dd}

                # MV and lagged for psi replicates
                for col in rep_cols:
                    rep = col.split('__', 1)[1] if '__' in col else col
                    mv_rep_all = evaluate_multivariate(df.rename(columns={col: 'psi_cm'}), target=t,
                                                       predictors=['psi_cm', 'pet', 'pr'], mask=None, n_splits=n_splits)
                    mv_rep_dd = evaluate_multivariate(df.rename(columns={col: 'psi_cm'}), target=t,
                                                      predictors=['psi_cm', 'pet', 'pr'], mask=mask_dd,
                                                      n_splits=n_splits)
                    metrics_out['mv']['replicates'].setdefault(rep, {})[t] = {'all': mv_rep_all, 'drydown': mv_rep_dd}

                    lag_rep_all = evaluate_lagged(df.rename(columns={col: 'psi_cm'}), target=t, base_predictor='psi_cm',
                                                  covariates=['pet', 'pr'], lags=(0, 3, 7, 14), mask=None,
                                                  n_splits=n_splits)
                    lag_rep_dd = evaluate_lagged(df.rename(columns={col: 'psi_cm'}), target=t, base_predictor='psi_cm',
                                                 covariates=['pet', 'pr'], lags=(0, 3, 7, 14), mask=mask_dd,
                                                 n_splits=n_splits)
                    metrics_out['lagged']['replicates'].setdefault(rep, {})[t] = {'all': lag_rep_all,
                                                                                  'drydown': lag_rep_dd}

        with open(os.path.join(met_dir, f'{sid}.json'), 'w') as f:
            json.dump(metrics_out, f, indent=2)

        for t in tgt_list:
            rep_cols = [c for c in df.columns if isinstance(c, str) and c.startswith('psi_cm_')]
            x_cols = ['theta'] + rep_cols
            titles = [f'{sid}: {t} ~ theta'] + [f'{sid}: {t} ~ psi ({c.split("__", 1)[1]})' for c in rep_cols]
            out_png = os.path.join(fig_dir, f'{sid}_{t.lower()}.png')
            plot_scatter_with_fit(df, x_cols=x_cols, y_col=t, titles=titles, out_path=out_png)

        for t in tgt_list:
            # Univariate rows (original)
            row = {'site_id': sid, 'target': t, 'replicate': 'theta', 'predictor': 'theta', 'model': 'uni', 'lag': 0}
            for scope in ['all', 'drydown']:
                row[f'{scope}_r2'] = float(base_all[t]['theta']['r2_mean']) if scope == 'all' else float(
                    base_dd[t]['theta']['r2_mean'])
                row[f'{scope}_rmse'] = float(base_all[t]['theta']['rmse_mean']) if scope == 'all' else float(
                    base_dd[t]['theta']['rmse_mean'])
                row[f'{scope}_folds'] = int(base_all[t]['theta']['folds']) if scope == 'all' else int(
                    base_dd[t]['theta']['folds'])
            rows.append(row)

            for rep, rep_dict in metrics_out['replicates'].items():
                rep_all = rep_dict['all']
                rep_dd = rep_dict['drydown']
                r = {'site_id': sid, 'target': t, 'replicate': rep, 'predictor': 'psi', 'model': 'uni', 'lag': 0}
                for scope in ['all', 'drydown']:
                    src = rep_all if scope == 'all' else rep_dd
                    r[f'{scope}_r2'] = float(src[t]['psi_cm']['r2_mean'])
                    r[f'{scope}_rmse'] = float(src[t]['psi_cm']['rmse_mean'])
                    r[f'{scope}_folds'] = int(src[t]['psi_cm']['folds'])
                rows.append(r)

            # Multivariate rows
            if 'mv' in metrics_out:
                mv_theta = metrics_out['mv']['theta'][t]
                row_mv = {'site_id': sid, 'target': t, 'replicate': 'theta', 'predictor': 'theta+PET+PR', 'model': 'mv',
                          'lag': 0}
                for scope in ['all', 'drydown']:
                    src = mv_theta[scope]
                    row_mv[f'{scope}_r2'] = float(src['r2_mean'])
                    row_mv[f'{scope}_rmse'] = float(src['rmse_mean'])
                    row_mv[f'{scope}_folds'] = int(src['folds'])
                rows.append(row_mv)
                for rep, rep_dict in metrics_out['mv']['replicates'].items():
                    mv_rep = rep_dict[t]
                    r_mv = {'site_id': sid, 'target': t, 'replicate': rep, 'predictor': 'psi+PET+PR', 'model': 'mv',
                            'lag': 0}
                    for scope in ['all', 'drydown']:
                        src = mv_rep[scope]
                        r_mv[f'{scope}_r2'] = float(src['r2_mean'])
                        r_mv[f'{scope}_rmse'] = float(src['rmse_mean'])
                        r_mv[f'{scope}_folds'] = int(src['folds'])
                    rows.append(r_mv)

            # Lagged rows (best lag per scope)
            if 'lagged' in metrics_out:
                lag_theta = metrics_out['lagged']['theta'][t]
                # choose best by r2_mean
                for scope in ['all', 'drydown']:
                    lag_metrics = lag_theta[scope]
                    best_lag = max(lag_metrics.items(),
                                   key=lambda kv: (kv[1].get('r2_mean') if isinstance(kv[1], dict) else -np.inf))[0]
                    src = lag_metrics[best_lag]
                    row_lag = {'site_id': sid, 'target': t, 'replicate': 'theta', 'predictor': 'theta',
                               'model': 'lagged', 'lag': int(best_lag)}
                    row_lag[f'{scope}_r2'] = float(src['r2_mean'])
                    row_lag[f'{scope}_rmse'] = float(src['rmse_mean'])
                    row_lag[f'{scope}_folds'] = int(src['folds'])
                    rows.append(row_lag)
                for rep, rep_dict in metrics_out['lagged']['replicates'].items():
                    lag_rep = rep_dict[t]
                    for scope in ['all', 'drydown']:
                        lag_metrics = lag_rep[scope]
                        best_lag = max(lag_metrics.items(),
                                       key=lambda kv: (kv[1].get('r2_mean') if isinstance(kv[1], dict) else -np.inf))[0]
                        src = lag_metrics[best_lag]
                        r_lag = {'site_id': sid, 'target': t, 'replicate': rep, 'predictor': 'psi', 'model': 'lagged',
                                 'lag': int(best_lag)}
                        r_lag[f'{scope}_r2'] = float(src['r2_mean'])
                        r_lag[f'{scope}_rmse'] = float(src['rmse_mean'])
                        r_lag[f'{scope}_folds'] = int(src['folds'])
                        rows.append(r_lag)

    summary = pd.DataFrame(rows)
    summary_fp = os.path.join(met_dir, 'summary.csv')
    summary.to_csv(summary_fp, index=False)
    return summary


if __name__ == '__main__':
    home_ = os.path.expanduser('~')

    shapefile_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'soil_potential_obs', 'reesh', 'shapefile',
                              'reesh_sites_mgrs_5070.shp')
    vg_dir_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'soil_potential_obs', 'curve_fits', 'reesh',
                           'bayes')
    amf_root_ = os.path.join(home_, 'data', 'IrrigationGIS', 'climate', 'ameriflux', 'amf_new')

    site_ids_ = list_reesh_site_ids(shapefile_)

    out_root_ = '/home/dgketchum/data/IrrigationGIS/soils/swapstress/reesh_site_analysis/'
    gridmet_dir_ = '/home/dgketchum/data/IrrigationGIS/soils/swapstress/vwc/gridmet/reesh'
    overwrite_ = True

    # Build coordinate dict (lon, lat) in EPSG:4326
    gdf_ = gpd.read_file(shapefile_).to_crs(4326)
    coords_by_id_ = {str(r['site_id']): (float(r.geometry.x), float(r.geometry.y)) for _, r in gdf_.iterrows()}

    build_data(
        site_ids=site_ids_,
        vg_dir=vg_dir_,
        out_root=out_root_,
        amf_root_or_file=amf_root_,
        site_coords=coords_by_id_,
        overwrite=overwrite_,
    )

    run_regressions(
        out_root=out_root_,
        targets=None,
        n_splits=5,
        site_ids=site_ids_,
    )

# ========================= EOF ====================================================================
