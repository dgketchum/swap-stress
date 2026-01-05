import os
from glob import glob
import json

import pandas as pd
import geopandas as gpd


def concatenate_and_join(ee_in_dir, out_file, network, rosetta_pqt=None, index_col='site_id',
                         categories=None, categorical_mappings_json=None, dropcols=None):
    """
    Concatenates CSVs from Earth Engine extraction, joins with Rosetta data,
    and saves to a single Parquet file.
    """
    if categorical_mappings_json is not None and categories is None:
        raise ValueError

    csv_files = glob(os.path.join(ee_in_dir, '*.csv'))
    if not csv_files:
        print(f"No CSV files found in {ee_in_dir}")
        return

    print(f"Found {len(csv_files)} CSV files to concatenate.")
    df_list = []

    for f in sorted(csv_files):
        try:
            c = pd.read_csv(f)
            print(os.path.basename(f), len(c))
        except pd.errors.EmptyDataError:
            print(f'Found empty file {os.path.basename(f)}, removing')
            os.remove(f)
            continue
        if c.empty:
            print(f'Found empty file {os.path.basename(f)}, removing')
            os.remove(f)
            continue
        if 'uid' in c.columns:
            print(f'UID in {os.path.basename(f)}, removing')
            os.remove(f)
            continue
        df_list.append(c)

    ee_df = pd.concat(df_list, ignore_index=True)

    if dropcols_:
        drop_from_ee = [c for c in dropcols_ if c in ee_df.columns]
        if dropcols is not None:
            ee_df = ee_df.drop(columns=drop_from_ee)

    ee_df[index_col] = ee_df[index_col].astype(str)
    ee_df.set_index(index_col, inplace=True)

    try:
        ee_df['elevation'] = ee_df['elevation'].astype(float)
    except KeyError:
        pass

    if rosetta_pqt:
        rosetta_df = pd.read_parquet(rosetta_pqt)
        rosetta_df = rosetta_df.groupby(index_col).first()
        if dropcols_:
            drop_from_rose = [c for c in dropcols_ if c in rosetta_df.columns]
            if dropcols is not None:
                rosetta_df = rosetta_df.drop(columns=drop_from_rose)

        try:
            rosetta_df.drop(columns=['elevation'], inplace=True)
        except KeyError:
            pass

        for col in ee_df.columns.to_list():
            if col in rosetta_df:
                ee_df = ee_df.drop(columns=[col])

        final_df = ee_df.join(rosetta_df, how='left')
    else:
        final_df = ee_df

    final_df = final_df[sorted(final_df.columns.to_list())]

    if categorical_mappings_json:
        mappings = {}
        for col in categories:
            final_df.loc[:, col] = final_df[col].values.astype(int)
            mappings[col] = {int(k): int(v) for v, k in enumerate(final_df[col].unique())}

        with open(categorical_mappings_json, 'w') as f:
            json.dump(mappings, f, indent=4)
        print(f"Saved categorical mappings to {categorical_mappings_json}")

    out_dir = os.path.dirname(out_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    final_df.index.name = index_col
    final_df.to_parquet(out_file)
    print(f"Saving final concatenated data to {out_file} {len(final_df)} samples")


if __name__ == '__main__':
    """"""

    home = os.path.expanduser('~')
    root_ = os.path.join(home, 'data', 'IrrigationGIS')

    # placeholders for a single call after blocks
    do_run = False
    ee_in_dir_ = None
    out_file_ = None
    index_col_ = None
    mappings_json_ = None
    rosetta_pqt_ = None
    network_ = None

    categories_ = ['hhs_stc', 'glc10_lc', 'WRB4', 'WRB_PHASES', 'WRB2_CODE',
                   'FAO90', 'KOPPEN', 'TEXTURE_USDA']

    dropcols_ = ['.geo', 'system:index', 'MGRS_TILE', 'name', 'has_swp', 'source', 'network',
                 'HWSD2_ID', 'WISE30s_ID', 'COVERAGE', 'SHARE', 'SWCC_class', 'obs_ct']

    run_mt_mesonet_workflow = False
    run_reesh_workflow = True
    run_gshp_workflow = False
    run_ncss_workflow = False

    if run_gshp_workflow:
        network_ = 'gshp'
        index_col_ = 'profile_id'
        ee_in_dir_ = os.path.join(root_, 'soils', 'swapstress', 'extracts', 'gshp_extracts_250m')
        out_file_ = os.path.join(root_, 'soils', 'swapstress', 'training', 'gshp_ee_data_250m.parquet')
        mappings_json_ = os.path.join(root_, 'soils', 'swapstress', 'training', 'gshp_categorical_mappings_250m.json')
        rosetta_pqt_ = os.path.join(root_, 'soils', 'soil_potential_obs', 'gshp', 'extracted_rosetta_points.parquet')
        do_run = True

    if run_mt_mesonet_workflow:
        network_ = 'mt_mesonet'
        index_col_ = 'station'
        ee_in_dir_ = os.path.join(root_, 'soils', 'swapstress', 'extracts', 'mt_mesonet_extracts_250m')
        out_file_ = os.path.join(root_, 'soils', 'swapstress', 'training', 'mt_ee_data_250m.parquet')
        mappings_json_ = os.path.join(root_, 'soils', 'swapstress', 'training', 'mt_categorical_mappings_250m.json')
        rosetta_pqt_ = os.path.join(root_, 'soils', 'rosetta', 'mt_mesonet', 'extracted_rosetta_points.parquet')
        dropcols_ += ['nwsli_id', 'network', 'mesowest_i', 'gwic_id', 'funded']
        do_run = True

    if run_reesh_workflow:
        network_ = 'reesh'
        index_col_ = 'site_id'
        ee_in_dir_ = os.path.join(root_, 'soils', 'swapstress', 'extracts', 'reesh_extracts_250m')
        out_file_ = os.path.join(root_, 'soils', 'swapstress', 'training', 'reesh_ee_data_250m.parquet')
        mappings_json_ = os.path.join(root_, 'soils', 'swapstress', 'training', 'reesh_categorical_mappings_250m.json')
        rosetta_pqt_ = os.path.join(root_, 'soils', 'soil_potential_obs', 'reesh', 'extracted_rosetta_points.parquet')
        do_run = True

    if run_ncss_workflow:
        network_ = 'ncss'
        index_col_ = 'profile_id'
        ee_in_dir_ = os.path.join(root_, 'soils', 'swapstress', 'extracts', 'ncss_extracts_250m')
        out_file_ = os.path.join(root_, 'soils', 'swapstress', 'training', 'ncss_ee_data_250m.parquet')
        mappings_json_ = os.path.join(root_, 'soils', 'swapstress', 'training', 'ncss_categorical_mappings_250m.json')
        rosetta_pqt_ = None  # NCSS has lab-measured soil properties, no Rosetta needed
        do_run = True

    if do_run:
        concatenate_and_join(
            ee_in_dir=ee_in_dir_,
            out_file=out_file_,
            rosetta_pqt=rosetta_pqt_,
            index_col=index_col_,
            categorical_mappings_json=mappings_json_,
            network=network_,
            categories=categories_,
            dropcols=dropcols_
        )

# ========================= EOF ====================================================================
