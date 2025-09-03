import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

from tqdm import tqdm

import geopandas as gpd
import pandas as pd
import rasterio.sample
from rasterio.windows import bounds
from shapely.geometry import box

from map.data import ROSETTA_CRS

ROSETTA_VG_PARAMS = ['theta_r', 'theta_s', 'log10_alpha', 'log10_n', 'log10_Ks']


def worker_extract(raster_path, window, points_gdf):
    """Worker function to extract points within a given raster window."""
    with rasterio.open(raster_path) as src:
        window_bounds = bounds(window, src.transform)
        points_in_window = points_gdf.cx[window_bounds[0]:window_bounds[2], window_bounds[1]:window_bounds[3]]
        points_in_window = points_in_window[points_in_window.intersects(box(*window_bounds))]

        if points_in_window.empty:
            return None

        coords = [(x, y) for x, y in zip(points_in_window.geometry.x, points_in_window.geometry.y)]

        vals = [v for v in src.sample(coords)]
        band_count = src.count

        assert band_count == 5

        col_names = [f"{os.path.basename(raster_path).split('.')[0]}_{p}" for p in ROSETTA_VG_PARAMS]

        return pd.DataFrame(vals, columns=col_names, index=points_in_window.index)


def extract_rosetta_parameters(points_shp, rosetta_dir, out_parquet, num_workers=4, debug=False):
    """
    Extracts Rosetta soil parameters from GeoTIFFs by chunking the raster.
    """
    print("Reading and reprojecting points...")
    points_gdf = gpd.read_file(points_shp)
    points_reproj = points_gdf.to_crs(ROSETTA_CRS)
    print(f"{len(points_reproj)} points reprojected.")

    rosetta_files = glob(os.path.join(rosetta_dir, '*VG*.tiff'))
    print(f"Found {len(rosetta_files)} Rosetta files to process.")

    all_results = []
    for r_file in rosetta_files:
        print(f"Processing raster: {os.path.basename(r_file)}")
        with rasterio.open(r_file) as src:
            jobs = [window for ij, window in src.block_windows(1)]
            print(f"  - Divided into {len(jobs)} windows.")

            if debug:
                print("Running in debug mode (single process)...")
                for window in jobs:
                    result_chunk = worker_extract(r_file, window, points_reproj)
                    if result_chunk is not None:
                        all_results.append(result_chunk)
            else:
                print(f"Running with {num_workers} workers...")
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = {executor.submit(worker_extract, r_file, window, points_reproj): window for window in
                               jobs}
                    for future in tqdm(as_completed(futures), total=len(jobs)):
                        try:
                            result_chunk = future.result()
                            if result_chunk is not None:
                                all_results.append(result_chunk)
                        except Exception as e:
                            print(f"A window generated an exception: {e}")

    print("Combining results from all chunks...")
    if not all_results:
        print("No data was extracted.")
        return

    extracted_df = pd.concat(all_results)
    final_df = points_reproj.join(extracted_df)

    print(f"Saving results to {out_parquet}...")
    final_df.drop(columns=['geometry']).to_parquet(out_parquet)
    print("Done.")


if __name__ == '__main__':
    home = os.path.expanduser('~')
    root_ = os.path.join(home, 'data', 'IrrigationGIS')

    rosetta_dir_ = os.path.join(root_, 'soils', 'rosetta')
    # points_shp_ = os.path.join(home,  IrrigationGIS', 'soils', 'gis', 'pretraining-roi-10000_mgrs.shp')
    # points_shp_ = os.path.join(home, 'IrrigationGIS', 'soils', 'soil_potential_obs', 'mt_mesonet', 'station_metadata_mgrs.shp')
    points_shp_ = os.path.join(root_, 'soils', 'vg_paramaer_databases', 'wrc', 'wrc_aggregated_mgrs.shp')

    # output_csv_ = os.path.join(home, 'IrrigationGIS', 'soils', 'rosetta', 'mt_mesonet', 'extracted_rosetta_points.parquet')
    output_csv_ = os.path.join(root_, 'soils', 'vg_paramaer_databases', 'wrc', 'extracted_rosetta_points.parquet')

    extract_rosetta_parameters(points_shp=points_shp_,
                               rosetta_dir=os.path.expanduser(rosetta_dir_),
                               out_parquet=output_csv_,
                               num_workers=12,
                               debug=False)
# ========================= EOF ====================================================================
