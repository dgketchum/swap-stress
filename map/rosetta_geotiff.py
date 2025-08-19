import os
from glob import glob

import geopandas as gpd
import pandas as pd
import rasterio

from map import ROSETTA_CRS


def extract_rosetta_parameters(points_shp, rosetta_dir, out_csv):
    """
    Extracts Rosetta soil parameters from GeoTIFFs for a given set of points.
    """

    points_gdf = gpd.read_file(points_shp)
    points_reproj = points_gdf.to_crs(ROSETTA_CRS)

    coords = [(x, y) for x, y in zip(points_reproj.geometry.x, points_reproj.geometry.y)]

    rosetta_files = glob(os.path.join(rosetta_dir, '*L1_VG*.tiff'))
    print(f"Found {len(rosetta_files)} Rosetta files to process.")

    extracted_data = []
    for r_file in rosetta_files:
        print(f"Processing {os.path.basename(r_file)}...")
        with rasterio.open(r_file) as src:
            vals = [v for v in src.sample(coords)]
            band_count = src.count
            col_names = [f"{os.path.basename(r_file).split('.')[0]}_b{i + 1}" for i in range(band_count)]
            df = pd.DataFrame(vals, columns=col_names, index=points_reproj.index)
            extracted_data.append(df)

    if extracted_data:
        points_reproj = points_reproj.join(pd.concat(extracted_data, axis=1))

    print(f"Saving results to {out_csv}...")
    points_reproj.drop(columns=['geometry']).to_csv(out_csv, index=False)
    print("Done.")
    return out_csv


if __name__ == '__main__':
    rosetta_dir_ = '~/data/IrrigationGIS/soils/rosetta/'
    points_shp_ = '/home/dgketchum/data/IrrigationGIS/soils/gis/pretraining-roi-10000_mgrs.shp'
    output_csv_ = '/home/dgketchum/data/IrrigationGIS/soils/rosetta/extracted_rosetta_l1_points.csv'

    extract_rosetta_parameters(points_shp=points_shp_,
                               rosetta_dir=os.path.expanduser(rosetta_dir_),
                               out_csv=output_csv_)

# ========================= EOF ====================================================================
