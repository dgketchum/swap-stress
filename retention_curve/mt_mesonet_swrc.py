import os
from typing import List, Optional

import matplotlib.pyplot as plt

from .swrc import SWRC


class MTMesonetSWRC(SWRC):
    """
    MT Mesonet-specific SWRC helper.

    - Defaults depth_col to 'Depth [cm]'.
    - Provides convenience methods for fitting multiple station files
      and saving plots/results.
    """

    def __init__(self, filepath=None, depth_col=None, df=None):
        super().__init__(filepath=filepath, depth_col=depth_col or 'Depth [cm]', df=df)

    @staticmethod
    def fit_station_files(station_files: List[str], results_dir: str, plot_dir: Optional[str] = None,
                          method: str = 'slsqp', report: bool = False) -> None:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        if plot_dir and not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        for station_file in station_files:
            if not os.path.exists(station_file):
                print(f"Missing station file: {station_file}")
                continue

            fname = os.path.basename(station_file).replace('.parquet', '')
            out_img = os.path.join(plot_dir, f"{fname}_{method}.png") if plot_dir else None

            try:
                fitter = MTMesonetSWRC(filepath=station_file)
                fitter.fit(report=report, method=method)
                if out_img:
                    fitter.plot(show=False, save_path=out_img)
                fitter.save_results(output_dir=results_dir)
                print(f"Processed {station_file}")
            except Exception as e:
                print(f"Failed {station_file}: {e}")


if __name__ == '__main__':
    home_ = os.path.expanduser('~')
    root = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'soil_potential_obs', 'mt_mesonet')

    data_ = os.path.join(root, 'preprocessed_by_station')
    results_ = os.path.join(root, 'results_by_station')
    plots_ = os.path.join(root, 'station_swrc_plots')

    station_files_ = [os.path.join(data_, f) for f in os.listdir(data_)]
    MTMesonetSWRC.fit_station_files(station_files_, results_dir=results_, plot_dir=plots_, method='slsqp', report=False)
# ========================= EOF ====================================================================
