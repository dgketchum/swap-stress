import os
import pandas as pd
import numpy as np

from .swrc import SWRC
from . import ROSETTA_NOMINAL_DEPTHS


class RosettaSWRC(SWRC):
    def __init__(self, filepath=None, depth_col=None, df=None,
                 curves_wide_csv=None, props_csv=None, sample_index=None):

        if curves_wide_csv and props_csv and sample_index is not None:
            props_df = pd.read_csv(props_csv)
            self.properties = props_df[props_df['Index'] == sample_index].to_dict('records')[0]

            curves_wide_df = pd.read_csv(curves_wide_csv)
            sample_wide_df = curves_wide_df[curves_wide_df['Index'] == sample_index]

            all_records = []
            data_columns = sample_wide_df.columns[2:]
            head_cols = data_columns[0::2]
            theta_cols = data_columns[1::2]

            for _, row in sample_wide_df.iterrows():
                for h_col, t_col in zip(head_cols, theta_cols):
                    h = row[h_col]
                    t = row[t_col]
                    if pd.notna(h) and pd.notna(t):
                        all_records.append({'head_cm': h, 'theta': t})
            
            df_long = pd.DataFrame(all_records)
            
            df_std = self._standardize_df(df_long, 'depth')
            super().__init__(filepath=None, depth_col='depth', df=df_std)

        elif df is not None or filepath:
            use_depth = depth_col or 'depth'
            if df is None:
                if filepath.endswith('.csv'):
                    df = pd.read_csv(filepath)
                elif filepath.endswith('.parquet'):
                    df = pd.read_parquet(filepath)
                else:
                    raise ValueError('File must be .csv or .parquet')
            df_std = self._standardize_df(df, use_depth)
            super().__init__(filepath=None, depth_col=use_depth, df=df_std)
        else:
            super().__init__(filepath=None, depth_col=depth_col, df=df)

    @staticmethod
    def _standardize_df(df: pd.DataFrame, depth_col: str) -> pd.DataFrame:
        d = df.copy()

        head_candidates = ['head_cm', 'suction_cm', 'psi_cm', 'head', 'psi']
        theta_candidates = ['theta', 'vwc', 'VWC']

        head_col = next((c for c in head_candidates if c in d.columns), None)
        theta_col = next((c for c in theta_candidates if c in d.columns), None)

        if head_col is None or theta_col is None:
            raise ValueError('Expected head (cm) and theta columns not found')

        d['suction'] = np.abs(d[head_col].astype(float).values)
        d['theta'] = d[theta_col].astype(float).values

        if depth_col in d.columns:
            d = d.rename(columns={depth_col: 'depth'})
        elif 'depth' in d.columns:
            pass
        elif 'level' in d.columns:
            lvl = d['level'].astype(int).values
            d['depth'] = [ROSETTA_NOMINAL_DEPTHS.get(int(v), np.nan) for v in lvl]
        else:
            d['depth'] = 0

        return d


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================