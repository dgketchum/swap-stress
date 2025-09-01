import os
from glob import glob
import json

import pandas as pd


def concatenate_and_join(ee_in_dir, out_file, rosetta_pqt=None, index_col='site_id', categorical_mappings_json=None,
                         categories=None, dropcols=None):
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

    for f in csv_files:
        try:
            df_list.append(pd.read_csv(f))
        except pd.errors.EmptyDataError:
            print(f'Found empty file {os.path.basename(f)}, removing')
            os.remove(f)
            continue

    ee_df = pd.concat(df_list, ignore_index=True)

    for c in ['.geo', 'system:index', 'MGRS_TILE']:
        if dropcols is None and c in ee_df.columns:
            dropcols = [c]
        elif dropcols and c not in dropcols and c in ee_df.columns:
            dropcols.append(c)
        else:
            pass

    if dropcols is not None:
        ee_df = ee_df.drop(columns=dropcols)

    ee_df.set_index(index_col, inplace=True)

    if rosetta_pqt:
        rosetta_df = pd.read_parquet(rosetta_pqt)
        rosetta_df = rosetta_df.groupby(index_col).first()

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
            if final_df[col].dtype == 'object':
                final_df[col] = final_df[col].astype('category')
            mappings[col] = {int(k): int(v) for v, k in
                             enumerate(final_df[col].unique())}  # likely error if non-numeric categories

        with open(categorical_mappings_json, 'w') as f:
            json.dump(mappings, f, indent=4)
        print(f"Saved categorical mappings to {categorical_mappings_json}")

    out_dir = os.path.dirname(out_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    final_df.to_parquet(out_file)
    print(f"Saving final concatenated data to {out_file} {len(final_df)} samples")


if __name__ == '__main__':
    """"""
    run_mt_mesonet_workflow = False
    run_general_workflow = False
    use_rosetta = False
    run_gshp_workflow = True

    home = os.path.expanduser('~')
    root_ = os.path.join(home, 'data', 'IrrigationGIS')

    if run_gshp_workflow:
        extracts_dir_ = os.path.join(root_, 'soils', 'swapstress', 'extracts', 'gshp_extracts')
        output_file_ = os.path.join(root_, 'soils', 'swapstress', 'training', 'gshp_training_data.parquet')
        mappings_json_ = os.path.join(root_, 'soils', 'swapstress', 'training', 'gshp_categorical_mappings.json')

        concatenate_and_join(ee_in_dir=extracts_dir_, out_file=output_file_, rosetta_pqt=None, index_col='uid',
                             categorical_mappings_json=mappings_json_,
                             categories=['hhs_stc',
                                         'glc10_lc',
                                         'WRB4',
                                         'WRB_PHASES',
                                         'WRB2_CODE',
                                         'FAO90',
                                         'KOPPEN',
                                         'TEXTURE_USDA'],
                             dropcols=['HWSD2_ID',
                                       'WISE30s_ID',
                                       'COVERAGE',
                                       'SHARE'])

    if run_mt_mesonet_workflow:
        extracts_dir_ = os.path.join(root_, 'soils', 'swapstress', 'mt_mesonet_extracts')
        rosetta_file_ = os.path.join(root_, 'soils', 'rosetta', 'mt_mesonet', 'extracted_rosetta_points.parquet')
        output_file_ = os.path.join(root_, 'soils', 'swapstress', 'training', 'mt_training_data.parquet')
        mappings_json_ = os.path.join(root_, 'soils', 'swapstress', 'training', 'mt_categorical_mappings.json')

        concatenate_and_join(ee_in_dir=extracts_dir_, out_file=output_file_,
                             rosetta_pqt=rosetta_file_ if use_rosetta else None, index_col='station',
                             categorical_mappings_json=mappings_json_, categories=['cdl_mode', 'nlcd'])

    elif run_general_workflow:
        extracts_dir_ = os.path.join(root_, 'soils', 'swapstress', 'extracts')
        rosetta_file_ = os.path.join(root_, 'soils', 'rosetta', 'extracted_rosetta_points.parquet')
        output_file_ = os.path.join(root_, 'soils', 'swapstress', 'training', 'training_data.parquet')
        mappings_json_ = os.path.join(root_, 'soils', 'swapstress', 'training', 'categorical_mappings.json')

        concatenate_and_join(ee_in_dir=extracts_dir_, out_file=output_file_,
                             rosetta_pqt=rosetta_file_ if use_rosetta else None, index_col='site_id',
                             categorical_mappings_json=mappings_json_, categories=['cdl_mode', 'nlcd'])
# ========================= EOF ====================================================================
