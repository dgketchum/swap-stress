import json
import os
from glob import glob
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


def _timeseries_cv_indices(n: int, n_splits: int = 5, min_test: int = 30) -> Iterable[tuple]:
    """Yield (train_idx, test_idx) for simple chronological splits.

    Uses sklearn TimeSeriesSplit when possible; falls back to a single 70/30 split.
    Ensures each test fold has at least `min_test` observations where possible.
    """
    if n <= (min_test * 2) or n_splits < 2:
        split = int(n * 0.7)
        yield (np.arange(0, split), np.arange(split, n))
        return

    tscv = TimeSeriesSplit(n_splits=n_splits)
    for tr, te in tscv.split(np.arange(n)):
        if len(te) >= min_test:
            yield (tr, te)


def _fit_eval_single(x: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Dict[str, float]:
    """CV evaluate a single-feature linear regression for y ~ x.

    Returns dict with mean R2, RMSE across folds.
    """
    x = x.reshape(-1, 1)
    n = len(y)
    r2s: List[float] = []
    rmses: List[float] = []

    for tr, te in _timeseries_cv_indices(n, n_splits=n_splits):
        if len(tr) < 2 or len(te) < 1:
            continue
        m = LinearRegression()
        m.fit(x[tr], y[tr])
        yp = m.predict(x[te])
        r2s.append(r2_score(y[te], yp))
        rmses.append(float(np.sqrt(mean_squared_error(y[te], yp))))

    return {
        'r2_mean': float(np.nanmean(r2s)) if r2s else np.nan,
        'rmse_mean': float(np.nanmean(rmses)) if rmses else np.nan,
        'folds': int(len(r2s)),
    }


def _fit_eval_multi(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Dict[str, float]:
    """CV evaluate a multi-feature linear regression for y ~ X.

    Returns dict with mean R2, RMSE across folds.
    """
    n = len(y)
    r2s: List[float] = []
    rmses: List[float] = []

    for tr, te in _timeseries_cv_indices(n, n_splits=n_splits):
        if len(tr) < 2 or len(te) < 1:
            continue
        m = LinearRegression()
        m.fit(X[tr], y[tr])
        yp = m.predict(X[te])
        r2s.append(r2_score(y[te], yp))
        rmses.append(float(np.sqrt(mean_squared_error(y[te], yp))))

    return {
        'r2_mean': float(np.nanmean(r2s)) if r2s else np.nan,
        'rmse_mean': float(np.nanmean(rmses)) if rmses else np.nan,
        'folds': int(len(r2s)),
    }


def evaluate_theta_vs_psi(
    df: pd.DataFrame,
    targets: Iterable[str] = ('GPP', 'ET'),
    mask: Optional[pd.Series] = None,
    n_splits: int = 5,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compare y ~ theta vs y ~ psi_cm for each target.

    Returns nested metrics: {target: {'theta': {...}, 'psi_cm': {...}}}
    """
    # Apply mask if provided
    data = df.copy()
    if mask is not None:
        data = data.loc[mask]

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for tgt in targets:
        if tgt not in data.columns:
            continue
        d = data[['theta', 'psi_cm', tgt]].dropna()
        if d.empty:
            out[tgt] = {'theta': {'r2_mean': np.nan, 'rmse_mean': np.nan, 'folds': 0},
                        'psi_cm': {'r2_mean': np.nan, 'rmse_mean': np.nan, 'folds': 0}}
            continue
        y = d[tgt].to_numpy(dtype=float)
        met_theta = _fit_eval_single(d['theta'].to_numpy(dtype=float), y, n_splits=n_splits)
        met_psi = _fit_eval_single(d['psi_cm'].to_numpy(dtype=float), y, n_splits=n_splits)
        out[tgt] = {'theta': met_theta, 'psi_cm': met_psi}
    return out


def evaluate_multivariate(
    df: pd.DataFrame,
    target: str,
    predictors: Iterable[str],
    mask: Optional[pd.Series] = None,
    n_splits: int = 5,
) -> Dict[str, float]:
    """Evaluate y ~ predictors (linear) with time-series CV.

    Drops rows with NaNs in target/predictors before evaluation.
    """
    data = df.copy()
    if mask is not None:
        data = data.loc[mask]
    cols = [c for c in predictors] + [target]
    d = data[cols].dropna()
    if d.empty:
        return {'r2_mean': np.nan, 'rmse_mean': np.nan, 'folds': 0}
    X = d[list(predictors)].to_numpy(dtype=float)
    y = d[target].to_numpy(dtype=float)
    return _fit_eval_multi(X, y, n_splits=n_splits)


def evaluate_lagged(
    df: pd.DataFrame,
    target: str,
    base_predictor: str,
    covariates: Iterable[str],
    lags: Iterable[int] = (0, 3, 7, 14),
    mask: Optional[pd.Series] = None,
    n_splits: int = 5,
) -> Dict[int, Dict[str, float]]:
    """Evaluate y ~ shifted(base_predictor, lag) + covariates for each lag.

    Returns dict keyed by lag with CV metrics.
    """
    results: Dict[int, Dict[str, float]] = {}
    for lag in lags:
        tmp = df.copy()
        tmp[f'{base_predictor}__lag{lag}'] = tmp[base_predictor].shift(lag)
        preds = [f'{base_predictor}__lag{lag}'] + list(covariates)
        results[int(lag)] = evaluate_multivariate(
            tmp, target=target, predictors=preds, mask=mask, n_splits=n_splits
        )
    return results


def save_metrics(metrics: Dict, out_file: str) -> None:
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(metrics, f, indent=2)


def correlation_matrix(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    method: str = 'pearson',
) -> pd.DataFrame:
    """Compute and return correlation matrix for specified columns, or all numeric columns if cols is None."""
    if cols is None:
        d = df.select_dtypes(include=[np.number])
    else:
        d = df[cols].select_dtypes(include=[np.number])
    return d.corr(method=method)


def calculate_ccf(df: pd.DataFrame, target: str, lags: Iterable[int]) -> pd.DataFrame:
    records = []
    for lag in lags:
        shifted = df[target].shift(lag)
        corr_vwc = shifted.corr(df['theta'])
        corr_psi = shifted.corr(df['psi_cm'])
        records.append({'lag': int(lag), 'corr_vwc': corr_vwc, 'corr_psi': corr_psi})
    out = pd.DataFrame(records, columns=['lag', 'corr_vwc', 'corr_psi'])
    return out

if __name__ == '__main__':
    """Example modeling driver: load prepared datasets and evaluate linear models.

    Edit flags and paths to your environment before running.
    """
    run_eval_for_sites = False

    prep_dir_ = os.path.join('site_modeling', 'outputs', 'prep')
    metrics_dir_ = os.path.join('site_modeling', 'outputs', 'metrics')
    corr_dir_ = os.path.join('site_modeling', 'outputs', 'correlations')
    os.makedirs(metrics_dir_, exist_ok=True)
    os.makedirs(corr_dir_, exist_ok=True)

    # Example site list (filenames in prep_dir_)
    site_files_ = sorted(glob(os.path.join(prep_dir_, '*.parquet')))

    if run_eval_for_sites:
        for fp in site_files_:
            try:
                sid = os.path.splitext(os.path.basename(fp))[0]
                df = pd.read_parquet(fp)
                # Optional: supply a drydown mask if saved alongside the dataset
                metrics = evaluate_theta_vs_psi(df, targets=('GPP', 'ET'), mask=None, n_splits=5)
                save_metrics(metrics, os.path.join(metrics_dir_, f'{sid}.json'))

                # Correlations: if RS proxies present, include; otherwise just core variables
                core_cols = [c for c in ['theta', 'psi_cm', 'GPP', 'ET'] if c in df.columns]
                rs_cols = [c for c in df.columns if c.startswith('landsat_') or c.startswith('ptjpl_')]
                corr = correlation_matrix(df, cols=core_cols + rs_cols)
                corr.to_csv(os.path.join(corr_dir_, f'{sid}.csv'))
                print(f'{sid}: saved metrics and correlations')
            except Exception as e:
                print(f'Failed {fp}: {e}')

# ========================= EOF ====================================================================
