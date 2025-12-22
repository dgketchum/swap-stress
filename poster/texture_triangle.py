import os
import sys

import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import use
from matplotlib.lines import Line2D

from retention_curve import PARAM_SYMBOLS

use('Agg')

home_ = os.path.expanduser('~')
soiltriangle_dir_ = os.path.join(home_, 'code', 'SoilTriangle')
if soiltriangle_dir_ not in sys.path:
    sys.path.append(soiltriangle_dir_)

from trianglegraph import SoilTrianglePlot  # noqa: E402


# Generic soil properties mapped to (ReESH_column, GSHP_column)
GENERIC_PROPERTY_MAP = {
    'organic_matter': ('Pct_OM', 'oc'),
    'bulk_density': ('Bulk_Den_g_m3', 'db_33'),
    'saturated_hydraulic_conductivity': ('Med_KsSoil_m_s', 'ksat_lab'),
    'porosity': (None, 'porosity'),
    'ph': (None, 'ph_h2o'),

    'theta_r': ('thetar', 'thetar'),
    'theta_s': ('thetas', 'thetas'),
    'alpha': ('alpha', 'alpha'),
    'n': ('n', 'n'),
}

SWRC_KEYS = {'theta_r', 'theta_s', 'alpha', 'n'}


def _normalize_to_triangle(sand, silt, clay):
    """Convert raw sand/silt/clay values to integer percentages summing to 100.

    Returns (sand, clay, silt) ordered for SoilTrianglePlot, or None if invalid.
    """
    if np.isnan(sand) or np.isnan(silt) or np.isnan(clay):
        return None

    total = sand + silt + clay
    if not np.isfinite(total) or total <= 0:
        return None

    sand_pct = sand * 100.0 / total
    silt_pct = silt * 100.0 / total
    clay_pct = clay * 100.0 / total

    sand_i = int(round(sand_pct))
    silt_i = int(round(silt_pct))
    clay_i = int(round(clay_pct))

    diff = 100 - (sand_i + silt_i + clay_i)
    if diff != 0:
        arr = np.array([sand_i, silt_i, clay_i], dtype=int)
        idx = int(np.argmax(arr))
        arr[idx] += diff
        sand_i, silt_i, clay_i = map(int, arr)

    if not (0 <= sand_i <= 100 and 0 <= silt_i <= 100 and 0 <= clay_i <= 100):
        return None

    if sand_i + silt_i + clay_i != 100:
        return None

    # SoilTriangle expects (bottom, left, right) = (sand, clay, silt)
    return sand_i, clay_i, silt_i


def _load_reesh_swrc_params():
    """Load ReESH SWRC fit parameters (theta_r, theta_s, alpha, n) aggregated by station and depth."""
    try:
        from map.data.station_training_table import extract_station_fit_params
    except Exception:
        return None

    home = os.path.expanduser('~')
    root = os.path.join(home, 'data', 'IrrigationGIS', 'soils')
    results_dir = os.path.join(root, 'soil_potential_obs', 'curve_fits')

    try:
        df = extract_station_fit_params(results_dir, networks=('reesh',), fit_method='bayes')
    except Exception:
        return None

    if df is None or df.empty:
        return None

    df = df.copy()
    df['station'] = df['station'].astype(str)
    df['depth'] = df['depth'].astype(float)
    df['depth_int'] = df['depth'].round(0).astype(int)

    agg_cols = ['theta_r', 'theta_s', 'alpha', 'n']
    present = [c for c in agg_cols if c in df.columns]
    if not present:
        return None

    grouped = (
        df.groupby(['station', 'depth_int'], dropna=False)[present]
        .mean()
        .reset_index()
    )
    return grouped


def _load_reesh_points(reesh_dir, property_column=None, property_key=None, swrc_params=None):
    """Load sand/silt/clay (and optional property) from ReESH *_SoilCharacteristics.csv files."""
    # Build lookup for SWRC-derived properties when requested
    swrc_map = None
    if property_key in SWRC_KEYS and swrc_params is not None:
        dfp = swrc_params.copy()
        if not {'station', 'depth_int'}.issubset(dfp.columns):
            dfp = None
        else:
            param_col = property_key if property_key in dfp.columns else None
            if param_col:
                dfp['station_key'] = dfp['station'].astype(str).str.lower()
                dfp = dfp.dropna(subset=[param_col])
                swrc_map = {
                    (row['station_key'], int(row['depth_int'])): float(row[param_col])
                    for _, row in dfp.iterrows()
                }

    points = []
    props = []
    for fname in sorted(os.listdir(reesh_dir)):
        if not fname.endswith('_SoilCharacteristics.csv'):
            continue
        fpath = os.path.join(reesh_dir, fname)
        df = pd.read_csv(fpath)
        if not {'SAND', 'SILT', 'CLAY'}.issubset(df.columns):
            continue

        depth_col = 'Depth_cm' if 'Depth_cm' in df.columns else 'Depth' if 'Depth' in df.columns else None
        site_col = 'Site' if 'Site' in df.columns else None

        cols = ['SAND', 'SILT', 'CLAY']
        if property_column and property_column in df.columns:
            cols.append(property_column)
        if depth_col and depth_col not in cols:
            cols.append(depth_col)
        if site_col and site_col not in cols:
            cols.append(site_col)

        for _, row in df[cols].iterrows():
            triple = _normalize_to_triangle(
                sand=row['SAND'],
                silt=row['SILT'],
                clay=row['CLAY'],
            )
            if triple is None:
                continue

            points.append(triple)

            # Default: pull directly from soil characteristics column when available
            val = np.nan
            if property_column and property_column in df.columns:
                try:
                    val = float(row[property_column])
                except Exception:
                    val = np.nan

            # Override with SWRC parameter when requested and available
            if property_key in SWRC_KEYS and swrc_map is not None and depth_col and site_col:
                try:
                    depth_val = float(row[depth_col])
                    depth_int = int(round(depth_val))
                    site_val = str(row[site_col]).lower()
                    val = swrc_map.get((site_val, depth_int), np.nan)
                except Exception:
                    pass

            props.append(val)

    return points, np.array(props, dtype=float) if props else np.array([])


def _load_gshp_points(gshp_file, property_column=None, property_key=None):
    """Load one sand/silt/clay (and optional property) per site-depth from the GSHP WRC dataset."""
    df = pd.read_csv(gshp_file, encoding='latin1')

    tex_cols = {'sand_tot_psa', 'silt_tot_psa', 'clay_tot_psa'}
    depth_cols = {'profile_id', 'hzn_top', 'hzn_bot'}
    if not tex_cols.issubset(df.columns):
        return [], np.array([])

    if depth_cols.issubset(df.columns):
        # Group by profile + depth, averaging texture and property
        group_cols = ['profile_id', 'hzn_top', 'hzn_bot']
        cols = list(tex_cols | set(group_cols))
        if property_column and property_column in df.columns and property_column not in cols:
            cols.append(property_column)
        sub = df[cols].copy()
        agg_spec = {c: 'mean' for c in tex_cols}
        if property_column and property_column in sub.columns:
            agg_spec[property_column] = 'mean'
        grouped = (
            sub.groupby(group_cols, dropna=False)
            .agg(agg_spec)
            .reset_index()
        )
        source = grouped
    else:
        # Fallback: use all rows without depth-aware grouping
        cols = list(tex_cols)
        if property_column and property_column in df.columns:
            cols.append(property_column)
        source = df[cols].copy()

    points = []
    props = []
    use_prop = property_column and property_column in source.columns

    for _, row in source.iterrows():
        triple = _normalize_to_triangle(
            sand=row['sand_tot_psa'],
            silt=row['silt_tot_psa'],
            clay=row['clay_tot_psa'],
        )
        if triple is None:
            continue
        points.append(triple)
        if use_prop:
            props.append(row[property_column])
        else:
            props.append(np.nan)

    return points, np.array(props, dtype=float) if props else np.array([])


def plot_texture_triangle(out_path, reesh_dir=None, gshp_file=None, property_key=None):
    """Plot soil texture triangle for ReESH and GSHP datasets.

    Parameters
    ----------
    out_path : str
        Output image filename (passed to SoilTrianglePlot.show).
    reesh_dir : str, optional
        Directory containing ReESH *_SoilCharacteristics.csv files.
    gshp_file : str, optional
        Path to GSHP WRC CSV.
    property_key : str, optional
        Generic soil property name to color by; must be a key in
        GENERIC_PROPERTY_MAP (e.g., 'organic_matter', 'bulk_density',
        'saturated_hydraulic_conductivity', 'porosity', 'ph').
    """
    if reesh_dir is None or gshp_file is None:
        home = os.path.expanduser('~')
        base = os.path.join(home, 'data', 'IrrigationGIS', 'soils', 'soil_potential_obs')
        if reesh_dir is None:
            reesh_dir = os.path.join(base, 'reesh')
        if gshp_file is None:
            gshp_file = os.path.join(base, 'gshp', 'WRC_dataset_surya_et_al_2021_final.csv')

    reesh_prop_col = None
    gshp_prop_col = None
    if property_key is not None:
        if property_key not in GENERIC_PROPERTY_MAP:
            raise ValueError(
                f"Unknown property_key '{property_key}'. "
                f"Available keys: {sorted(GENERIC_PROPERTY_MAP.keys())}"
            )
        reesh_prop_col, gshp_prop_col = GENERIC_PROPERTY_MAP[property_key]

    # GSHP vG params come in the original dataset
    reesh_swrc_params = _load_reesh_swrc_params() if property_key in SWRC_KEYS else None

    reesh_points, reesh_vals = _load_reesh_points(
        reesh_dir,
        property_column=reesh_prop_col,
        property_key=property_key,
        swrc_params=reesh_swrc_params,
    )
    gshp_points, gshp_vals = _load_gshp_points(
        gshp_file,
        property_column=gshp_prop_col,
        property_key=property_key,
    )

    # Optional log10 transform for selected SWRC parameters
    if property_key in {'alpha', 'n'}:
        if reesh_vals.size:
            with np.errstate(divide='ignore', invalid='ignore'):
                reesh_vals = np.where(reesh_vals > 0, np.log10(reesh_vals), np.nan)
        if gshp_vals.size:
            with np.errstate(divide='ignore', invalid='ignore'):
                gshp_vals = np.where(gshp_vals > 0, np.log10(gshp_vals), np.nan)

    # Determine global color scale if a property is requested and present
    use_color = property_key is not None
    vmin = vmax = None
    if use_color:
        arrays = []
        if reesh_vals.size:
            arrays.append(reesh_vals[~np.isnan(reesh_vals)])
        if gshp_vals.size:
            arrays.append(gshp_vals[~np.isnan(gshp_vals)])
        if arrays:
            all_vals = np.concatenate(arrays)
            if all_vals.size:
                vals = all_vals[~np.isnan(all_vals)]
                if vals.size:
                    vmin = float(np.nanmin(vals))
                    mean = float(np.nanmean(vals))
                    std = float(np.nanstd(vals))
                    upper = mean + 2.0 * std
                    vmax_raw = float(np.nanmax(vals))
                    vmax = float(min(vmax_raw, upper))
                else:
                    use_color = False
            else:
                use_color = False
        else:
            use_color = False

    stp = SoilTrianglePlot('Soil Texture')
    stp.soil_categories()

    if gshp_points:
        if use_color and gshp_vals.size:
            stp.scatter(
                gshp_points,
                s=8,
                c=gshp_vals,
                cmap=cm.viridis,
                vmin=vmin,
                vmax=vmax,
                marker='s',
                alpha=0.7,
                label='GSHP',
                zorder=2,
            )
        else:
            stp.scatter(
                gshp_points,
                s=8,
                c='#2ca02c',
                marker='s',
                alpha=0.7,
                label='GSHP',
                zorder=2,
            )

    if reesh_points:
        if use_color and reesh_vals.size:
            stp.scatter(
                reesh_points,
                s=16,
                c=reesh_vals,
                cmap=cm.viridis,
                vmin=vmin,
                vmax=vmax,
                marker='^',
                alpha=0.7,
                label='ReESH',
                zorder=3,
            )
        else:
            stp.scatter(
                reesh_points,
                s=16,
                c='#d62728',
                marker='^',
                alpha=0.7,
                label='ReESH',
                zorder=3,
            )

    if use_color and (reesh_points or gshp_points):
        base_label = PARAM_SYMBOLS.get(property_key, property_key.replace('_', ' ').title())
        if property_key in {'alpha', 'n'}:
            # Indicate that the ramp shows log10-transformed values
            if base_label.startswith('$') and base_label.endswith('$'):
                label = r'$\log_{10}(' + base_label.strip('$') + r')$'
            else:
                label = f'log10({base_label})'
        else:
            label = base_label
        stp.colorbar(label)

    # legend with fixed marker appearance, independent of color ramp
    legend_handles = []
    if gshp_points:
        gshp_color = '0.5' if use_color else '#2ca02c'
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker='s',
                color='none',
                markerfacecolor=gshp_color,
                markeredgecolor=gshp_color,
                markersize=5,
                linestyle='None',
                label='GSHP',
            )
        )
    if reesh_points:
        reesh_color = '0.5' if use_color else '#d62728'
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker='^',
                color='none',
                markerfacecolor=reesh_color,
                markeredgecolor=reesh_color,
                markersize=6,
                linestyle='None',
                label='ReESH',
            )
        )

    if legend_handles:
        plt.legend(handles=legend_handles, loc=1)

    # Match the default framing used in SoilTrianglePlot.show
    plt.axis([-10, 110, -10, 110])
    plt.ylim(-10, 100)
    plt.savefig(out_path)
    plt.show()
    plt.close()


if __name__ == '__main__':

    out_dir_ = '/home/dgketchum/data/IrrigationGIS/soils/swapstress/texture'

    # One figure with no fourth property, matching map markers
    base_out_ = os.path.join(out_dir_, 'texture_triangle.png')
    plot_texture_triangle(base_out_, property_key=None)

# ========================= EOF ====================================================================
