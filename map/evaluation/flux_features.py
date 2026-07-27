"""
Predictor / feature engineering for the flux robustness re-test.

Builds the derived predictors the re-test needs (plan §4) off the existing
``flux_site_daily.parquet`` — no re-extraction of daily TIFs:

- **REW (relative extractable water)** — the "poor-man's texture normalization"
  baseline for H2. Two variants: an *empirical* per-site percentile normalization
  (works everywhere, no soil data) and a *physical* VG-based one (θ between the
  −33 kPa field-capacity and −1.5 MPa wilting points, CONUS/Rosetta only).
- **Antecedent-weighted L3 surface θ** — the L4-free root-zone proxy for H4b:
  a causal, gap-aware exponential moving average of the SMAP **L3** surface
  retrieval over a decay timescale τ.
- **Priestley–Taylor ET0** — shared reference-ET covariate.
- **Site covariates** (texture, aridity, climate/texture class) sampled once
  from the static EASE2 feature stack → ``flux_site_covariates.parquet`` for the
  E1 stratification and the WS2 texture-distance transfer analysis.

All suction values are log10(cm H2O), consistent with the rest of the pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Priestley–Taylor reference ET (shared)
# ---------------------------------------------------------------------------
ALPHA_PT = 1.26
GAMMA = 0.0665  # kPa/°C
LAMBDA_V = 2.45  # MJ/kg
ALBEDO = 0.23


def priestley_taylor_et0(sw_in: np.ndarray, t_avg: np.ndarray) -> np.ndarray:
    """Daily reference ET (mm/d) from shortwave radiation and mean temperature.

    Identical formulation to ``quartile_binned_mlr.priestley_taylor_et0`` — kept
    here as the single shared source for the re-test modules.
    """
    sw_in = np.asarray(sw_in, dtype=np.float64)
    t_avg = np.asarray(t_avg, dtype=np.float64)
    es = 0.6108 * np.exp(17.27 * t_avg / (t_avg + 237.3))
    delta = 4098.0 * es / (t_avg + 237.3) ** 2
    rn = sw_in * (1.0 - ALBEDO) * 86400.0 / 1e6  # MJ/m²/d
    et0 = ALPHA_PT * (delta / (delta + GAMMA)) * rn / LAMBDA_V
    return np.maximum(et0, 0.0)


def add_et0(df: pd.DataFrame, out_col: str = "et0") -> pd.DataFrame:
    """Add a Priestley–Taylor ET0 column computed from sw_in + t_avg."""
    df = df.copy()
    df[out_col] = priestley_taylor_et0(df["sw_in"].values, df["t_avg"].values)
    return df


# ---------------------------------------------------------------------------
# Relative extractable water (REW) — H2 normalization baseline
# ---------------------------------------------------------------------------


def rew_empirical(
    df: pd.DataFrame,
    theta_col: str,
    group_col: str = "site_id",
    lo_pct: float = 5.0,
    hi_pct: float = 95.0,
    out_col: str | None = None,
) -> pd.DataFrame:
    """Empirical per-site REW: distribution-based texture normalization.

    ``REW = (θ − θ_wilt) / (θ_fc − θ_wilt)`` clipped to [0, 1], with θ_wilt / θ_fc
    taken as the per-site ``lo_pct`` / ``hi_pct`` percentiles of the available θ
    for that column. Needs no soil data, so it works at every site — the
    "poor-man's normalization" baseline that H2 pits ψ against.
    """
    df = df.copy()
    if out_col is None:
        out_col = f"rew_{theta_col}"

    def _norm(s: pd.Series) -> pd.Series:
        vals = s.dropna()
        if len(vals) < 10:
            return pd.Series(np.nan, index=s.index)
        lo = np.percentile(vals, lo_pct)
        hi = np.percentile(vals, hi_pct)
        if hi - lo < 1e-9:
            return pd.Series(np.nan, index=s.index)
        return ((s - lo) / (hi - lo)).clip(0.0, 1.0)

    df[out_col] = df.groupby(group_col)[theta_col].transform(_norm)
    return df


def vg_theta_from_psi(
    psi_log10_cm: np.ndarray,
    theta_r: float,
    theta_s: float,
    alpha: float,
    n: float,
) -> np.ndarray:
    """Forward van Genuchten: θ(ψ). ``psi_log10_cm`` is log10(cm H2O); alpha in
    1/cm, n > 1 (natural scale)."""
    psi_cm = np.power(10.0, np.asarray(psi_log10_cm, dtype=np.float64))
    m = 1.0 - 1.0 / n
    se = 1.0 / np.power(1.0 + np.power(alpha * psi_cm, n), m)
    return theta_r + (theta_s - theta_r) * se


# ψ thresholds in log10(cm H2O): field capacity −33 kPa ≈ 336 cm; permanent
# wilting −1.5 MPa ≈ 15300 cm.
PSI_FC_LOG10 = float(np.log10(336.0))
PSI_WILT_LOG10 = float(np.log10(15300.0))


def rew_physical(
    theta: np.ndarray,
    theta_r: float,
    theta_s: float,
    alpha: float,
    n: float,
) -> np.ndarray:
    """Physical REW using VG-derived field-capacity and wilting θ.

    ``REW = (θ − θ_wilt) / (θ_fc − θ_wilt)`` clipped [0, 1], with θ_fc = θ(−33 kPa)
    and θ_wilt = θ(−1.5 MPa) from the site's VG curve. NaN where params invalid.
    """
    if not (
        np.isfinite([theta_r, theta_s, alpha, n]).all()
        and n > 1.0
        and alpha > 0.0
        and theta_s > theta_r
    ):
        return np.full(np.shape(theta), np.nan)
    theta_fc = vg_theta_from_psi(PSI_FC_LOG10, theta_r, theta_s, alpha, n)
    theta_wilt = vg_theta_from_psi(PSI_WILT_LOG10, theta_r, theta_s, alpha, n)
    if theta_fc - theta_wilt < 1e-6:
        return np.full(np.shape(theta), np.nan)
    rew = (np.asarray(theta, dtype=np.float64) - theta_wilt) / (theta_fc - theta_wilt)
    return np.clip(rew, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Antecedent-weighted surface θ — L4-free root-zone proxy for H4b
# ---------------------------------------------------------------------------


def _causal_ewma_gappy(dates_ordinal: np.ndarray, values: np.ndarray, tau: float):
    """Causal, gap-aware exponentially-weighted mean.

    For each observation i (sorted by date) returns
    ``Σ_{s≤i} exp(−Δt/τ)·θ_s / Σ_{s≤i} exp(−Δt/τ)`` where Δt is the actual day
    gap between successive *available* observations. O(n) recursion; renormalized
    so it is a true weighted mean (not a decaying sum).
    """
    n = len(values)
    out = np.full(n, np.nan)
    s_num = 0.0
    s_den = 0.0
    prev_t = None
    for i in range(n):
        v = values[i]
        if not np.isfinite(v):
            # Missing day: no update, and the running state still decays to the
            # next available day via its Δt. Leave out[i] NaN.
            continue
        if prev_t is None:
            s_num = v
            s_den = 1.0
        else:
            decay = np.exp(-(dates_ordinal[i] - prev_t) / tau)
            s_num = v + decay * s_num
            s_den = 1.0 + decay * s_den
        prev_t = dates_ordinal[i]
        out[i] = s_num / s_den
    return out


def add_antecedent_weighted(
    df: pd.DataFrame,
    value_col: str,
    taus: tuple[int, ...] = (7, 15, 30, 60),
    date_col: str = "date",
    group_col: str = "site_id",
    prefix: str | None = None,
) -> pd.DataFrame:
    """Add antecedent-weighted columns of ``value_col`` for each decay τ (days).

    Computed per site on the (gappy) daily series with a time-aware decay — the
    L4-free root-zone surrogate built only from the surface retrieval (H4b).
    Output columns: ``{prefix}_ant{tau}`` (prefix defaults to ``value_col``).
    """
    df = df.copy()
    if prefix is None:
        prefix = value_col
    ordinal = (
        pd.to_datetime(df[date_col]).values.astype("datetime64[D]").astype(np.int64)
    )
    df = df.assign(_ord=ordinal)

    for tau in taus:
        col = f"{prefix}_ant{tau}"
        df[col] = np.nan

    for _, idx in df.groupby(group_col).groups.items():
        sub = df.loc[idx].sort_values("_ord")
        order = sub.index
        ords = sub["_ord"].values
        vals = sub[value_col].values.astype(np.float64)
        for tau in taus:
            ewma = _causal_ewma_gappy(ords, vals, float(tau))
            df.loc[order, f"{prefix}_ant{tau}"] = ewma

    return df.drop(columns=["_ord"])


# ---------------------------------------------------------------------------
# Site covariates (E1 stratification + WS2 texture distance)
# ---------------------------------------------------------------------------

STATIC_DIR = "/nas/soils/swapstress/inference/global_features/rasters_ease2"
TEXTURE_FEATURES = ["sand_0-5cm_mean", "clay_0-5cm_mean", "silt_0-5cm_mean"]
# Aridity proxy: seasonal WorldClim precip (mm) and annual reference ET (mm).
PRECIP_FEATURES = [
    "wc_prec_spring",
    "wc_prec_summer",
    "wc_prec_autumn",
    "wc_prec_winter",
]
PET_FEATURE = "eto_yearly_mean"
CLASS_FEATURES = ["TEXTURE_USDA", "KOPPEN"]


def _project_to_static_grid(lats, lons, grid):
    """Project WGS84 coords to (row, col) pixel indices on the static EASE2 grid."""
    from pyproj import Transformer

    t = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    xs, ys = t.transform(np.asarray(lons), np.asarray(lats))
    inv = ~grid.transform
    cols, rows = inv * (xs, ys)
    rows = np.round(rows).astype(int)
    cols = np.round(cols).astype(int)
    valid = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
    return rows, cols, valid


def build_site_covariates(
    sites: pd.DataFrame,
    static_dir: str = STATIC_DIR,
) -> pd.DataFrame:
    """Sample static texture / aridity / class covariates at each site pixel.

    ``sites`` needs ``site_id``, ``lat``, ``lon``. Returns one row per site with
    sand/clay/silt (%), an aridity index (annual P / annual PET), and the
    USDA-texture and Köppen class codes. IGBP PFT is not in the station metadata,
    so Köppen climate + USDA texture stand in as the categorical stratifiers.
    """
    from map.inference.predict_rasters import StaticRasterStack

    required = set(TEXTURE_FEATURES + PRECIP_FEATURES + [PET_FEATURE] + CLASS_FEATURES)
    stack = StaticRasterStack.load(static_dir, required)
    grid = stack.grid

    sites = sites.drop_duplicates("site_id").reset_index(drop=True)
    rows, cols, valid = _project_to_static_grid(
        sites["lat"].values, sites["lon"].values, grid
    )
    flat = rows * grid.width + cols

    out = pd.DataFrame({"site_id": sites["site_id"].values})
    for feat in required:
        vals = np.full(len(sites), np.nan, dtype=np.float64)
        idx = stack.feature_to_index[feat]
        vals[valid] = stack.data[idx, flat[valid]]
        out[feat] = vals

    # Aridity index = annual precip / annual PET (both mm). Values >1 humid,
    # <0.65 dry per the UNEP convention (proxy: WorldClim seasonal precip summed).
    annual_p = out[PRECIP_FEATURES].sum(axis=1, min_count=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["aridity_index"] = annual_p / out[PET_FEATURE]
    out["annual_precip_mm"] = annual_p

    # Rename texture columns to short names.
    out = out.rename(
        columns={
            "sand_0-5cm_mean": "sand",
            "clay_0-5cm_mean": "clay",
            "silt_0-5cm_mean": "silt",
            "TEXTURE_USDA": "texture_usda",
            "KOPPEN": "koppen",
        }
    )
    return out


def texture_vector(cov_df: pd.DataFrame) -> pd.DataFrame:
    """Standardized (sand, clay) texture vectors keyed by site_id for distance.

    Silt is 100 − sand − clay (compositional), so sand + clay suffice. Returns a
    site-indexed frame of z-scored sand/clay; sites missing texture are dropped.
    """
    sub = cov_df.dropna(subset=["sand", "clay"]).set_index("site_id")[["sand", "clay"]]
    if sub.empty:
        return sub
    z = (sub - sub.mean()) / sub.std(ddof=0).replace(0, np.nan)
    return z.dropna()


def texture_distance_to_set(tex_z: pd.DataFrame, test_site: str, train_sites) -> float:
    """Mean Euclidean distance (in z-scored texture space) from a held-out site
    to the training set. NaN if the test site or all train sites lack texture."""
    if test_site not in tex_z.index:
        return np.nan
    train = [s for s in train_sites if s in tex_z.index]
    if not train:
        return np.nan
    v = tex_z.loc[test_site].values
    d = np.linalg.norm(tex_z.loc[train].values - v, axis=1)
    return float(np.mean(d))
