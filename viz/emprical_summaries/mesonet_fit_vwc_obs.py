import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_mesonet_swp_vs_vwc_with_violin(swrc_csv_path, vwc_parquet_path, save_path=None, show=False, station_name=None):
    df = pd.read_csv(swrc_csv_path)
    df = df.rename(columns={'suction_cm': 'suction', 'depth_cm': 'depth'})
    df['suction'] = np.abs(df['suction'].astype(float).values)
    df['theta'] = df['theta'].astype(float).values

    groups = list(df.groupby('depth'))
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(groups)))

    # Load Mesonet VWC time series for this station
    ts = pd.read_parquet(vwc_parquet_path)
    vwc_cols = [c for c in ts.columns if isinstance(c, str) and c.startswith('soil_vwc_') and not c.endswith('_units')]

    # Choose VWC columns that match SWRC depths (e.g., soil_vwc_0005 for 5 cm)
    swrc_depths = {int(round(float(d))) for d, _ in groups if pd.notna(d)}
    expected = {f"soil_vwc_{dd:04d}" for dd in swrc_depths}
    sel_cols = [c for c in vwc_cols if c in expected]
    if not sel_cols:
        sel_cols = vwc_cols

    depth_to_color = {int(round(float(d))): c for (d, _), c in zip(groups, colors) if pd.notna(d)}
    depth_cols = [(dd, f"soil_vwc_{dd:04d}") for dd in swrc_depths if f"soil_vwc_{dd:04d}" in ts.columns]
    depth_cols = sorted(depth_cols, key=lambda x: x[0])

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax, axv) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                  gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.05})
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    axv.set_facecolor('white')

    # Main SWRC scatter
    for (depth, g), color in zip(groups, colors):
        g = g.dropna(subset=['suction', 'theta'])
        ax.plot(g['theta'], g['suction'], 'o', color=color, ms=4, alpha=0.8, label=f'{int(round(depth))} cm')

    ax.set_yscale('log')
    ax.set_ylabel('Soil Water Potential (cm)', fontsize=12)
    ax.set_xlabel('')
    station_code = os.path.splitext(os.path.basename(swrc_csv_path))[0]
    title_base = 'SWRC Observations with VWC Frequency'
    title_full = f"{title_base} — {station_name}" if station_name else f"{title_base} — {station_code}"
    ax.set_title(title_full, fontsize=14, fontweight='bold')
    ax.legend(title='Depth', fontsize=9, title_fontsize=9, frameon=False)
    handles, labels = ax.get_legend_handles_labels()
    pairs = []
    for h, l in zip(handles, labels):
        m = re.search(r'(\d+)', l)
        d = int(m.group(1)) if m else 0  # likely error: legend label missing depth digits
        pairs.append((d, h, l))
    pairs.sort(key=lambda x: x[0])
    handles_s = [h for _, h, _ in pairs]
    labels_s = [l for _, _, l in pairs]
    ax.legend(handles_s, labels_s, title='Depth', fontsize=9, title_fontsize=9, frameon=False)
    ax.grid(True, which='both', ls='--', c='0.75')

    # Limits similar to SWRC.plot
    ax.set_xlim(0.0, 0.65)
    ax.set_ylim(top=10 ** 7)

    # Violin below: per-depth horizontal violins summarizing VWC time series distributions
    if depth_cols:
        violin_data, positions, vcolors, labels = [], [], [], []
        for idx, (dd, col) in enumerate(depth_cols, start=1):
            vals = ts[col].values * 0.01
            vals = vals[np.isfinite(vals)]
            vals = vals[(vals >= 0) & (vals <= 1)]
            if vals.size == 0:
                continue
            violin_data.append(vals)
            positions.append(idx)
            vcolors.append(depth_to_color.get(dd, '0.6'))
            labels.append(f"{dd} cm (n={vals.size})")
        if violin_data:
            parts = axv.violinplot(violin_data, vert=False, positions=positions, showextrema=False, widths=0.85)
            for pc, col in zip(parts['bodies'], vcolors):
                pc.set_facecolor(col)
                pc.set_edgecolor('0.4')
                pc.set_alpha(0.6)
            axv.set_yticks(positions)
            axv.set_yticklabels(labels)
            axv.tick_params(labelsize=9)
            axv.set_ylim(0, max(positions) + 1)
            axv.invert_yaxis()
            axv.set_xlabel('')
            axv.spines['top'].set_visible(False)
            axv.spines['right'].set_visible(False)
            axv.grid(False)
            axv.set_xlim(ax.get_xlim())
        else:
            axv.axis('off')
    else:
        # Fallback: single combined violin across all available VWC columns
        vwc_vals = ts[sel_cols].values.ravel()
        vwc_vals = vwc_vals[np.isfinite(vwc_vals)]
        vwc_vals = vwc_vals[(vwc_vals >= 0) & (vwc_vals <= 1)]
        if vwc_vals.size:
            parts = axv.violinplot(vwc_vals, vert=False, showextrema=False, widths=0.9)
            for pc in parts['bodies']:
                pc.set_facecolor('0.6')
                pc.set_edgecolor('0.4')
                pc.set_alpha(0.6)
            axv.set_yticks([])
            axv.set_ylim(0.5, 1.5)
            axv.set_xlabel('')
            axv.spines['top'].set_visible(False)
            axv.spines['right'].set_visible(False)
            axv.spines['left'].set_visible(False)
            axv.grid(False)
            axv.set_xlim(ax.get_xlim())
        else:
            axv.axis('off')

    # Put the shared x-label on the bottom (applies to violins and scatter x-axis)
    axv.set_xlabel('Volumetric Water Content ($cm^3/cm^3$)', fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(left=0.15)
    if save_path:
        plt.savefig(save_path, dpi=350, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    print(fig)


def plot_mesonet_dirs_swp_vs_vwc_with_violin(swrc_dir, vwc_dir, out_dir, show=False, meta_csv_path=None):
    os.makedirs(out_dir, exist_ok=True)
    swrc_files = [f for f in os.listdir(swrc_dir) if f.endswith('.csv')]
    vwc_set = set(os.listdir(vwc_dir))
    name_map = None
    if meta_csv_path and os.path.exists(meta_csv_path):
        mdf = pd.read_csv(meta_csv_path)
        if 'station' in mdf.columns and 'name' in mdf.columns:
            name_map = dict(zip(mdf['station'].astype(str), mdf['name'].astype(str)))
    for fn in swrc_files:
        station = os.path.splitext(fn)[0]
        vwc_fn = f'{station}_daily.parquet'
        if vwc_fn not in vwc_set:
            continue
        swrc_csv_path = os.path.join(swrc_dir, fn)
        vwc_parquet_path = os.path.join(vwc_dir, vwc_fn)
        out_png = os.path.join(out_dir, f'{station}_swrc_vwc_violin.png')
        st_name = name_map.get(station) if name_map else None
        plot_mesonet_swp_vs_vwc_with_violin(swrc_csv_path, vwc_parquet_path, save_path=out_png, show=show, station_name=st_name)


if __name__ == '__main__':
    home_ = os.path.expanduser('~')
    swrc_dir_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'soil_potential_obs', 'preprocessed', 'mt_mesonet')
    vwc_dir_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'vwc_timeseries', 'mt_mesonet', 'preprocessed_by_station')
    out_dir_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'soil_potential_obs', 'mt_mesonet', 'swrc_vwc_violin')
    meta_csv_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'soil_potential_obs', 'mt_mesonet', 'station_metadata.csv')
    plot_mesonet_dirs_swp_vs_vwc_with_violin(swrc_dir_, vwc_dir_, out_dir_, show=False, meta_csv_path=meta_csv_)
# ========================= EOF ====================================================================
