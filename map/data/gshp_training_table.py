import os
from tqdm import tqdm
from glob import glob
import pandas as pd

LABELS_COLS_KEEP = ['alpha', 'data_flag', 'n', 'theta_r', 'theta_s']

def build_gshp_training_table(ee_features_pqt, labels_csv, out_file, index_col='profile_id', filter_good_quality=True,
                              embeddings=None):
    # for "simplicity", all EE extracts concatenated in ee_tables.py will by indexed with 'station'
    ee_df = pd.read_parquet(ee_features_pqt)
    ee_df.index.name = index_col

    labels_df = pd.read_csv(labels_csv, encoding='latin1') if labels_csv.endswith('.csv') else pd.read_parquet(labels_csv)
    if index_col in labels_df.columns:
        labels_df[index_col] = labels_df[index_col].astype(str)
        labels_df = labels_df.set_index(index_col)
    if filter_good_quality and 'data_flag' in labels_df.columns:
        labels_df = labels_df[labels_df['data_flag'] == 'good quality estimate']

    keep = [c for c in LABELS_COLS_KEEP if c in labels_df.columns]
    labels_df = labels_df[keep]

    if any([c for c in LABELS_COLS_KEEP if c in ee_df.columns]):
       for c in  LABELS_COLS_KEEP:
           if c in ee_df.columns:
               ee_df.drop(columns=[c], inplace=True)

    final_df = ee_df.join(labels_df, how='left').dropna(subset=['theta_r', 'theta_s', 'alpha', 'n'])

    if embeddings and os.path.isdir(embeddings):
        emb_files = glob(os.path.join(embeddings, '*.parquet'))
        if emb_files:
            rows = {}
            for fp in tqdm(emb_files, desc='Adding Embedding Features', total=len(emb_files)):
                pid = os.path.splitext(os.path.basename(fp))[0]
                try:
                    df = pd.read_parquet(fp)
                    vec = df.iloc[0] if len(df) >= 1 else None
                except Exception:
                    vec = None  # likely error: malformed embedding file
                if vec is not None:
                    rows[str(pid)] = vec
            if rows:
                emb_df = pd.DataFrame.from_dict(rows, orient='index')
                emb_df.index.name = index_col
                final_df = final_df.join(emb_df, how='left')

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
        out_file_ = os.path.join(root_, 'soils', 'swapstress', 'training', 'gshp_training_data_emb_250m.parquet')

        vwc_root_ = '/data/ssd2/swapstress/vwc'
        embeddings_ = os.path.join(vwc_root_, 'embeddings', 'gshp')

        build_gshp_training_table(
            ee_features_pqt=ee_features_pqt_,
            labels_csv=labels_csv_,
            out_file=out_file_,
            index_col='profile_id',
            filter_good_quality=True,
            embeddings=embeddings_,
        )
# ========================= EOF ====================================================================
