import os
import pandas as pd


def build_gshp_training_table(ee_features_pqt, labels_csv, out_file, index_col='profile_id', filter_good_quality=True):
    ee_df = pd.read_parquet(ee_features_pqt)
    if index_col not in ee_df.columns and getattr(ee_df.index, 'name', None) == index_col:
        ee_df = ee_df.reset_index()
    ee_df[index_col] = ee_df[index_col].astype(str)
    ee_df = ee_df.set_index(index_col)

    labels_df = pd.read_csv(labels_csv, encoding='latin1') if labels_csv.endswith('.csv') else pd.read_parquet(labels_csv)
    if index_col in labels_df.columns:
        labels_df[index_col] = labels_df[index_col].astype(str)
        labels_df = labels_df.set_index(index_col)
    if filter_good_quality and 'data_flag' in labels_df.columns:
        labels_df = labels_df[labels_df['data_flag'] == 'good quality estimate']

    keep = [c for c in ['theta_r', 'theta_s', 'alpha', 'n', 'data_flag'] if c in labels_df.columns]
    labels_df = labels_df[keep]

    final_df = ee_df.join(labels_df, how='left').dropna(subset=['theta_r', 'theta_s', 'alpha', 'n'])

    out_dir = os.path.dirname(out_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    final_df.to_parquet(out_file)
    print(f'{len(final_df)} GSHP training samples written to {out_file}')


if __name__ == '__main__':
    run_gshp_workflow = True

    home_ = os.path.expanduser('~')
    root_ = os.path.join(home_, 'data', 'IrrigationGIS')

    if run_gshp_workflow:
        gshp_directory_ = os.path.join(root_, 'soils', 'soil_potential_obs', 'gshp')
        labels_csv_ = os.path.join(gshp_directory_, 'WRC_dataset_surya_et_al_2021_final_clean.csv')
        ee_features_pqt_ = os.path.join(root_, 'soils', 'swapstress', 'training', 'gshp_ee_data_250m.parquet')
        out_file_ = os.path.join(root_, 'soils', 'swapstress', 'training', 'gshp_training_data_250m.parquet')

        build_gshp_training_table(
            ee_features_pqt=ee_features_pqt_,
            labels_csv=labels_csv_,
            out_file=out_file_,
            index_col='profile_id',
            filter_good_quality=True,
        )
# ========================= EOF ====================================================================
