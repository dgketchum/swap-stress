import os
from glob import glob
import json

import pandas as pd


def concatenate_and_join(ee_in_dir, out_file, rosetta_pqt=None, index_col='site_id', categorical_mappings_json=None,
                         categories=None, dropcols=None, extra_datasets=None):
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

    ee_df[index_col] = ee_df[index_col].astype(str)
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

    if extra_datasets:
        for ds_name, spec in extra_datasets.items():
            if isinstance(spec, dict):
                fp = spec.get('filepath')
                cols = spec.get('label_columns') or spec.get('columns') or spec.get('labels')
            else:
                continue
            if not fp:
                continue
            if fp.endswith('.parquet'):
                lab = pd.read_parquet(fp)
            elif fp.endswith('.csv'):
                lab = pd.read_csv(fp)
            else:
                lab = pd.read_parquet(fp)  # likely error if not parquet/csv
            if index_col in lab.columns:
                lab[index_col] = lab[index_col].astype(str)
                lab = lab.set_index(index_col)
            if isinstance(cols, list) and len(cols) > 0:
                lab = lab[cols]
            for c in lab.columns.to_list():
                if c in final_df:
                    final_df = final_df.drop(columns=[c])
            final_df = final_df.join(lab, how='left')
            final_df = final_df.dropna()

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
        gshp_directory_ = os.path.join(root_, 'soils', 'vg_paramaer_databases', 'wrc')
        soil_csv_path_ = os.path.join(gshp_directory_, 'WRC_dataset_surya_et_al_2021_final.csv')
        gshp_labels_csv_ = os.path.join(gshp_directory_, 'WRC_dataset_surya_et_al_2021_final_clean.csv')
        output_file_ = os.path.join(root_, 'soils', 'swapstress', 'training', 'gshp_training_data.parquet')
        mappings_json_ = os.path.join(root_, 'soils', 'swapstress', 'training', 'gshp_categorical_mappings.json')

        concatenate_and_join(ee_in_dir=extracts_dir_, out_file=output_file_,
                             rosetta_pqt=None, index_col='profile_id',
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
                                       'SHARE'],
                             extra_datasets={'gshp': {'filepath': gshp_labels_csv_,
                                                      'labels': ['theta_r', 'theta_s', 'alpha', 'n', 'data_flag']}})

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
