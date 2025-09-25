import os
from typing import List

from retention_curve.swrc import SWRC


def _json_files(d: str) -> List[str]:
    if not os.path.exists(d):
        return []
    return [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith('.json')]


def plot_saved_curves_in_dir(fits_dir: str, out_dir: str, show: bool = False) -> int:
    """
    Loads saved SWRC fit JSON files in `fits_dir` and writes plot PNGs to `out_dir`.
    Returns the count of plots generated.
    """
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for fp in _json_files(fits_dir):
        try:
            swrc = SWRC()
            swrc.load_from_results_json(fp)
            name = os.path.splitext(os.path.basename(fp))[0]
            save_path = os.path.join(out_dir, f"{name}.png")
            swrc.plot(save_path=save_path, show=show)
            count += 1
        except Exception as e:
            print(f"Failed to plot {fp}: {e}")
    return count


if __name__ == '__main__':
    # Mirror the project style: set flags and paths here
    home_ = os.path.expanduser('~')
    root = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'soil_potential_obs')

    run_rosetta = False
    run_mt_mesonet = True
    run_reesh = True

    method = 'bayes'  # or e.g., 'slsqp' depending on which fits you saved
    show_plots = False

    out_root = os.path.join(root, 'curve_fits')

    if run_rosetta:
        fits_dir = os.path.join(out_root, 'rosetta', method)
        if os.path.exists(fits_dir):
            plots_dir = os.path.join(fits_dir, 'curve_fits', 'rosetta', 'plots')
            n = plot_saved_curves_in_dir(fits_dir, plots_dir, show=show_plots)
            print(f"rosetta: wrote {n} plot(s) to {plots_dir}")

    if run_mt_mesonet:
        fits_dir = os.path.join(out_root, 'mt_mesonet', method)
        if os.path.exists(fits_dir):
            plots_dir = os.path.join(fits_dir, 'curve_fits', 'mt_mesonet', 'plots')
            n = plot_saved_curves_in_dir(fits_dir, plots_dir, show=show_plots)
            print(f"mt_mesonet: wrote {n} plot(s) to {plots_dir}")

    if run_reesh:
        fits_dir = os.path.join(out_root, 'reesh', method)
        if os.path.exists(fits_dir):
            plots_dir = os.path.join(fits_dir, 'curve_fits', 'reesh', 'plots')
            n = plot_saved_curves_in_dir(fits_dir, plots_dir, show=show_plots)
            print(f"reesh: wrote {n} plot(s) to {plots_dir}")

# ========================= EOF ====================================================================
