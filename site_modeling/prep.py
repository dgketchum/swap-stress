import json
import os
from glob import glob
from typing import Dict, Optional, Tuple, List, Set

import numpy as np
import pandas as pd


CM_PER_MPA = 10197.16  # cm of water per MPa (approx.)


def _find_site_file(base_dir: str, site_id: str, exts=(".parquet", ".csv")) -> Optional[str]:
    """Locate a per-site file by prefix under a directory.

    Returns the first match among {site_id}_*.{ext} (case-insensitive) or None.
    """
    if not base_dir or not os.path.isdir(base_dir):
        return None
    sid = str(site_id)
    for ext in exts:
        pats = [
            os.path.join(base_dir, f"{sid}{ext}"),
            os.path.join(base_dir, f"{sid}_*{ext}"),
            os.path.join(base_dir, f"*{sid}*{ext}"),
        ]
        for pat in pats:
            hits = sorted(glob(pat))
            if hits:
                return hits[0]
    return None


def load_vg_fit(site_id: str, vg_dir: str, sensor_depth_cm: Optional[float] = None) -> Optional[Dict]:
    """Load empirical vG parameters for a site from JSON fit summary.

    Chooses the depth nearest to `sensor_depth_cm` if provided; otherwise the
    shallowest available depth. Returns dict with keys:
      {'theta_r', 'theta_s', 'alpha', 'n', 'depth_cm'} or None if not found.
    """
    if not vg_dir or not os.path.isdir(vg_dir):
        return None
    fp = _find_site_file(vg_dir, site_id, exts=(".json",))
    if not fp or not os.path.exists(fp):
        return None

    with open(fp, 'r') as f:
        data = json.load(f)

    depths: Dict[float, Dict] = {}
    for k, v in (data or {}).items():
        if k == 'metadata' or not isinstance(v, dict):
            continue
        try:
            d = float(k)
        except Exception:
            continue
        if v.get('status') != 'Success':
            continue
        params = v.get('parameters', {})
        try:
            tr = float(params['theta_r']['value'])
            ts = float(params['theta_s']['value'])
            al = float(params['alpha']['value'])
            n_ = float(params['n']['value'])
        except Exception:
            continue
        depths[d] = {'theta_r': tr, 'theta_s': ts, 'alpha': al, 'n': n_}

    if not depths:
        return None

    # Choose best depth
    dsel = min(depths.keys())
    if sensor_depth_cm is not None and len(depths) > 1:
        dsel = min(depths.keys(), key=lambda x: abs(x - float(sensor_depth_cm)))

    out = depths[dsel].copy()
    out['depth_cm'] = float(dsel)
    return out


def _parse_datetime_index(df: pd.DataFrame, time_cols=("datetime", "date", "time")) -> pd.DataFrame:
    for c in time_cols:
        if c in df.columns:
            d = df.copy()
            d[c] = pd.to_datetime(d[c], errors='coerce')
            d = d.dropna(subset=[c]).set_index(c).sort_index()
            return d
    # Fallback: try to use index if it's datetime-like
    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        raise ValueError("No datetime column found and index is not DatetimeIndex")
    return d.sort_index()


def _find_vwc_column(df: pd.DataFrame, preferred_depth_cm: Optional[float] = None) -> Tuple[str, Optional[float]]:
    """Select a VWC column heuristically, preferring a target depth if available.

    Returns (column_name, parsed_depth_cm or None)
    """
    cols = [c for c in df.columns if isinstance(c, str) and ('vwc' in c.lower()) and not c.lower().endswith('_units')]
    if not cols:
        raise ValueError("No VWC columns found (looking for names containing 'vwc').")

    def _parse_depth(col: str) -> Optional[float]:
        import re
        m = re.search(r"(\d+\.?\d*)\s*cm", col)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
        return None

    if preferred_depth_cm is not None:
        depths_map = {c: _parse_depth(c) for c in cols}
        # Choose the closest depth column with a parsed depth
        with_depth = {c: d for c, d in depths_map.items() if d is not None}
        if with_depth:
            best = min(with_depth.keys(), key=lambda c: abs(with_depth[c] - preferred_depth_cm))
            return best, with_depth[best]

    # Fallback: first VWC column
    cd = cols[0]
    return cd, _parse_depth(cd)


def load_vwc_series(site_id: str, vwc_dir_or_file: str, preferred_depth_cm: Optional[float] = None) -> pd.Series:
    """Load VWC time series and return as daily mean series for one depth.

    Accepts a directory (auto-detect file) or a direct file path (CSV/Parquet).
    """
    fp = vwc_dir_or_file
    if os.path.isdir(vwc_dir_or_file):
        f = _find_site_file(vwc_dir_or_file, site_id)
        if f is None:
            raise FileNotFoundError(f"No VWC file found for site {site_id} under {vwc_dir_or_file}")
        fp = f

    if not os.path.exists(fp):
        raise FileNotFoundError(fp)

    if fp.endswith('.parquet'):
        df = pd.read_parquet(fp)
    else:
        df = pd.read_csv(fp)

    d = _parse_datetime_index(df)
    col, depth_cm = _find_vwc_column(d, preferred_depth_cm)
    s = d[col].astype(float)
    return s.resample('D').mean().rename('theta')


def load_flux_series(
    site_id: str,
    flux_dir_or_file: str,
    gpp_col: str = 'GPP',
    et_col: str = 'ET',
    time_cols=("date", "datetime", "time"),
) -> pd.DataFrame:
    """Load daily fluxes (GPP, ET). If file is a directory, tries to locate a file for site_id.

    The flux file should contain a parseable date/datetime column and the given
    GPP/ET column names (configurable).
    """
    fp = flux_dir_or_file
    if os.path.isdir(flux_dir_or_file):
        f = _find_site_file(flux_dir_or_file, site_id)
        if f is None:
            raise FileNotFoundError(f"No flux file found for site {site_id} under {flux_dir_or_file}")
        fp = f

    if not os.path.exists(fp):
        raise FileNotFoundError(fp)

    if fp.endswith('.parquet'):
        df = pd.read_parquet(fp)
    else:
        df = pd.read_csv(fp)

    d = None
    for c in time_cols:
        if c in df.columns:
            d = df.copy()
            d[c] = pd.to_datetime(d[c], errors='coerce')
            d = d.dropna(subset=[c]).set_index(c).sort_index()
            break
    if d is None:
        # Try index-based
        d = df.copy()
        if not isinstance(d.index, pd.DatetimeIndex):
            raise ValueError("Flux time column not found; expected one of: %s" % (time_cols,))
        d = d.sort_index()

    keep = {}
    if gpp_col in d.columns:
        keep['GPP'] = d[gpp_col].astype(float)
    if et_col in d.columns:
        keep['ET'] = d[et_col].astype(float)
    if not keep:
        raise ValueError(f"Flux columns not found. Expected at least one of: {gpp_col}, {et_col}")

    out = pd.DataFrame(keep).resample('D').mean()
    return out


def theta_to_psi_cm(theta: pd.Series, theta_r: float, theta_s: float, alpha: float, n: float) -> pd.Series:
    """Invert van Genuchten to suction (cm) from volumetric water content.

    psi_cm = ((Se^(-1/m) - 1)^(1/n)) / alpha;  Se = (theta - theta_r) / (theta_s - theta_r)
    """
    n = float(max(n, 1.0000001))
    m = 1.0 - 1.0 / n
    tr = float(theta_r)
    ts = float(theta_s)
    a = float(max(alpha, 1e-12))
    eps = 1e-6

    t = theta.clip(tr + eps, ts - eps).astype(float)
    se = (t - tr) / max(ts - tr, eps)
    inv = (np.power(se, -1.0 / m) - 1.0)
    inv = np.maximum(inv, 0.0)
    psi = np.power(inv, 1.0 / n) / a
    s = pd.Series(psi, index=theta.index, name='psi_cm')
    return s.replace([np.inf, -np.inf], np.nan)


def build_site_dataset(
    site_id: str,
    vg_dir: str,
    vwc_dir_or_file: str,
    flux_dir_or_file: str,
    vwc_depth_cm: Optional[float] = None,
    gpp_col: str = 'GPP',
    et_col: str = 'ET',
) -> pd.DataFrame:
    """Build aligned daily dataset with columns: theta, psi_cm, and available targets (GPP/ET)."""
    vg = load_vg_fit(site_id, vg_dir, sensor_depth_cm=vwc_depth_cm)
    if vg is None:
        raise FileNotFoundError(f"vG fit for {site_id} not found in {vg_dir}")

    theta = load_vwc_series(site_id, vwc_dir_or_file, preferred_depth_cm=vwc_depth_cm)
    psi = theta_to_psi_cm(theta, vg['theta_r'], vg['theta_s'], vg['alpha'], vg['n'])
    df_x = pd.concat([theta, psi], axis=1)

    df_y = load_flux_series(site_id, flux_dir_or_file, gpp_col=gpp_col, et_col=et_col)
    # Align daily and inner join
    df = df_x.join(df_y, how='inner')
    return df


def detect_drydown_mask(
    df: pd.DataFrame,
    theta_col: str = 'theta',
    min_len: int = 5,
    pct_threshold: float = 0.4,
    slope_window: int = 3,
) -> pd.Series:
    """Identify drydown periods using rolling slope < 0 and low-percentile theta.

    Returns a boolean mask indexed like df: True where conditions hold.
    """
    s = df[theta_col].astype(float).copy()
    low = s < s.quantile(pct_threshold)
    slope = s.rolling(slope_window, min_periods=2).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=False)
    neg = slope < 0

    mask = (low & neg).astype(bool)
    # Enforce minimum run length by grouping consecutive True segments
    grp = (mask.ne(mask.shift()).cumsum())
    run_lengths = mask.groupby(grp).transform('sum')
    return (mask & (run_lengths >= min_len)).fillna(False)


# ---------------- AmeriFlux-specific helpers ---------------- #

def _parse_amf_timestamp(ts: pd.Series) -> pd.DatetimeIndex:
    """Parse AmeriFlux TIMESTAMP_START style integers (YYYYMMDDHHMM)."""
    s = ts.astype(str).str.strip().str.zfill(12)
    return pd.to_datetime(s, format='%Y%m%d%H%M', errors='coerce')


def find_reesh_site_ids(vg_reesh_dir: str) -> Set[str]:
    """List site IDs from fitted ReESH JSON filenames.

    Assumes files like '<site_id>*.json'. Returns basenames without extension.
    """
    if not vg_reesh_dir or not os.path.isdir(vg_reesh_dir):
        return set()
    files = glob(os.path.join(vg_reesh_dir, '*.json'))
    ids: Set[str] = set()
    for fp in files:
        base = os.path.basename(fp)
        sid = os.path.splitext(base)[0]
        if sid:
            ids.add(sid)
    return ids


def find_ameriflux_file(amf_root: str, site_id: str, period: str = 'HH') -> Optional[str]:
    """Search recursively for an AmeriFlux BASE CSV for a given site and period.

    Looks for patterns like 'AMF_<site>_BASE_<period>_*.csv' below amf_root.
    """
    if not amf_root or not os.path.isdir(amf_root):
        return None
    pat1 = os.path.join(amf_root, f'**', f'AMF_{site_id}_BASE_{period}_*.csv')
    pat2 = os.path.join(amf_root, f'**', f'AMF_{site_id}_BASE-*', f'AMF_{site_id}_BASE_{period}_*.csv')
    hits = sorted(glob(pat1, recursive=True) + glob(pat2, recursive=True))
    return hits[0] if hits else None


def load_ameriflux_halfhourly(amf_csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load AmeriFlux BASE HH CSV and return (daily_vwc_df, daily_flux_df).

    - Replaces sentinel -9999 with NaN
    - Parses TIMESTAMP_START as DatetimeIndex
    - Aggregates to daily means/sums as appropriate
    - VWC columns: all starting with 'SWC_'
    - Flux: ET derived from LE_* using step duration; GPP included if present
    """
    if not os.path.exists(amf_csv_path):
        raise FileNotFoundError(amf_csv_path)

    df = pd.read_csv(amf_csv_path, skiprows=2)
    # Replace common sentinels
    df = df.replace({-9999: np.nan, -6999: np.nan})

    if 'TIMESTAMP_START' not in df.columns:
        raise ValueError('TIMESTAMP_START missing from AmeriFlux file')

    dti = _parse_amf_timestamp(df['TIMESTAMP_START'])
    df = df.set_index(dti).drop(columns=['TIMESTAMP_START'])
    # Drop TIMESTAMP_END if present
    if 'TIMESTAMP_END' in df.columns:
        df = df.drop(columns=['TIMESTAMP_END'])

    # Identify SWC columns
    swc_cols = [c for c in df.columns if isinstance(c, str) and c.startswith('SWC_')]
    vwc_df = pd.DataFrame(index=df.index)
    for c in swc_cols:
        v = pd.to_numeric(df[c], errors='coerce')
        # Heuristic: if values mostly > 1.2, treat as percent and scale to fraction
        if v.dropna().quantile(0.9) > 1.2:
            v = v / 100.0
        vwc_df[c.lower()] = v
    daily_vwc = vwc_df.resample('D').mean()

    # Flux: derive ET from LE_* if present
    le_cols = [c for c in df.columns if isinstance(c, str) and c.startswith('LE_')]
    flux_df = pd.DataFrame(index=df.index)
    if le_cols:
        le = pd.concat([pd.to_numeric(df[c], errors='coerce') for c in le_cols], axis=1).mean(axis=1)
        # Determine step in seconds by mode of diffs
        diffs = np.diff(df.index.view('i8') // 10 ** 9)
        if len(diffs) == 0:
            dt_seconds = 1800.0
        else:
            # Use the most common diff
            vals, counts = np.unique(diffs, return_counts=True)
            dt_seconds = float(vals[np.argmax(counts)])
        et_mm = le * dt_seconds / 2.45e6  # 1 mm = 1 kg m^-2; lambda ~ 2.45 MJ/kg
        flux_df['ET'] = et_mm
    # GPP if present
    gpp_cols = [c for c in df.columns if isinstance(c, str) and c.upper().startswith('GPP')]
    if gpp_cols:
        gpp = pd.concat([pd.to_numeric(df[c], errors='coerce') for c in gpp_cols], axis=1).mean(axis=1)
        flux_df['GPP'] = gpp
    # Resample daily: ET sum, GPP sum (if half-hourly u-mol CO2 something else; keep sum for now)
    agg = {}
    if 'ET' in flux_df.columns:
        agg['ET'] = 'sum'
    if 'GPP' in flux_df.columns:
        agg['GPP'] = 'sum'
    daily_flux = flux_df.resample('D').agg(agg) if agg else pd.DataFrame(index=daily_vwc.index)
    return daily_vwc, daily_flux


def build_site_dataset_from_ameriflux(
    site_id: str,
    amf_root_or_file: str,
    vg_dir: str,
    vwc_depth_cm: Optional[float] = None,
) -> pd.DataFrame:
    """Build daily dataset using AmeriFlux BASE HH data and empirical vG parameters.

    Returns a wide DataFrame with:
      - All daily SWC columns as theta_<name>
      - Corresponding psi_cm_<name> via vG inversion
      - ET (from LE) and optional GPP (if present)
      - Also includes simple aggregates: theta (mean across SWC) and psi_cm (mean across psi columns)
    """
    vg = load_vg_fit(site_id, vg_dir, sensor_depth_cm=vwc_depth_cm)
    if vg is None:
        raise FileNotFoundError(f"vG fit for {site_id} not found in {vg_dir}")

    amf_fp = amf_root_or_file
    if os.path.isdir(amf_root_or_file):
        f = find_ameriflux_file(amf_root_or_file, site_id, period='HH')
        if f is None:
            raise FileNotFoundError(f"AmeriFlux HH file for {site_id} not found under {amf_root_or_file}")
        amf_fp = f

    daily_vwc, daily_flux = load_ameriflux_halfhourly(amf_fp)
    out = daily_flux.copy()

    psi_cols: List[str] = []
    theta_cols: List[str] = []
    for c in daily_vwc.columns:
        theta_col = f'theta_{c}'
        psi_col = f'psi_cm_{c}'
        out[theta_col] = daily_vwc[c]
        out[psi_col] = theta_to_psi_cm(daily_vwc[c], vg['theta_r'], vg['theta_s'], vg['alpha'], vg['n'])
        psi_cols.append(psi_col)
        theta_cols.append(theta_col)

    # Aggregate simple means across sensors for quick baselines
    if theta_cols:
        out['theta'] = out[theta_cols].mean(axis=1)
    if psi_cols:
        out['psi_cm'] = out[psi_cols].mean(axis=1)

    return out.dropna(how='all')


if __name__ == '__main__':
    """Example prep driver to build per-site aligned datasets for modeling.

    Edit the flags and paths to your environment before running. Outputs are
    written under site_modeling/outputs/prep by default.
    """

    run_build_for_sites = False
    run_build_ameriflux = False

    # Example config
    home_ = os.path.expanduser('~')
    root_ = os.path.join(home_, 'data', 'IrrigationGIS')

    # Inputs
    vg_dir_ = os.path.join(root_, 'soils', 'soil_potential_obs', 'curve_fits', 'mt_mesonet', 'bayes')
    vwc_dir_ = os.path.join(root_, 'soils', 'vwc_timeseries', 'mt_mesonet', 'preprocessed_by_station')
    flux_dir_ = os.path.join(root_, 'soils', 'flux', 'daily')  # user-provided path; adjust
    # AmeriFlux example root (adjust to your setup)
    amf_root_ = os.path.join(home_, 'data', 'IrrigationGIS', 'climate', 'ameriflux', 'amf_new')
    reesh_vg_dir_ = os.path.join(root_, 'soils', 'soil_potential_obs', 'curve_fits', 'reesh', 'bayes')

    # Sites to build
    site_ids_ = [
        # 'mt1234',
    ]

    out_dir_ = os.path.join('site_modeling', 'outputs', 'prep')
    os.makedirs(out_dir_, exist_ok=True)

    if run_build_for_sites:
        for sid in site_ids_:
            try:
                dfi = build_site_dataset(
                    site_id=sid,
                    vg_dir=vg_dir_,
                    vwc_dir_or_file=vwc_dir_,
                    flux_dir_or_file=flux_dir_,
                    vwc_depth_cm=None,
                    gpp_col='GPP',
                    et_col='ET',
                )
                out_fp = os.path.join(out_dir_, f'{sid}.parquet')
                dfi.to_parquet(out_fp)
                print(f'Wrote {out_fp} with shape {dfi.shape}')
            except Exception as e:
                print(f'{sid} failed: {e}')

    if run_build_ameriflux:
        # Option 1: build for a known site code
        site_ids_af = [
            # 'US-MOz',
        ]
        # Option 2: derive from ReESH fits (if present)
        # site_ids_af = sorted(find_reesh_site_ids(reesh_vg_dir_))

        for sid in site_ids_af:
            try:
                dfa = build_site_dataset_from_ameriflux(
                    site_id=sid,
                    amf_root_or_file=amf_root_,
                    vg_dir=reesh_vg_dir_,
                    vwc_depth_cm=None,
                )
                out_fp = os.path.join(out_dir_, f'{sid}_amf.parquet')
                dfa.to_parquet(out_fp)
                print(f'Wrote {out_fp} with shape {dfa.shape}')
            except Exception as e:
                print(f'{sid} failed (AmeriFlux): {e}')

# ========================= EOF ====================================================================
