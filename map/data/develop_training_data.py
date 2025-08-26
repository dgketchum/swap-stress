import os

from map.data.call_ee import is_authorized
from .ee_export import get_bands
from .ee_tables import concatenate_and_join


if __name__ == '__main__':

    run_mt_mesonet_workflow = True
    run_general_workflow = False


    run_gee_extract = True
    run_concatenate = False

    root = '/home/dgketchum/data/IrrigationGIS'
    gcs_bucket = 'wudr'

    if run_mt_mesonet_workflow:
        extracts_dir = os.path.join(root, 'soils', 'swapstress', 'mt_mesonet_extracts')

        if run_gee_extract:
            shapefile = os.path.join(root, 'soils', 'soil_potential_obs', 'mt_mesonet', 'station_metadata_mgrs.shp')
            index = 'station'
            output_prefix = 'swap-stress/mesonet_training_data'
            mgrs_shapefile = os.path.join(root, 'boundaries', 'mgrs', 'mgrs_wgs.shp')

            is_authorized()
            get_bands(shapefile_path=shapefile,
                      mgrs_shp_path=mgrs_shapefile,
                      bucket=gcs_bucket,
                      file_prefix=output_prefix,
                      resolution=4000,
                      index_col=index,
                      split_tiles=True,
                      check_dir=extracts_dir)

        elif run_concatenate:
            rosetta_file_ = os.path.join(root, 'soils', 'rosetta', 'mt_mesonet', 'extracted_rosetta_points.parquet')
            output_file_ = os.path.join(root, 'soils', 'swapstress', 'training', 'mt_training_data.parquet')
            mappings_json_ = os.path.join(root, 'soils', 'swapstress', 'training', 'mt_categorical_mappings.json')

            concatenate_and_join(ee_in_dir=extracts_dir,
                                 rosetta_pqt=rosetta_file_,
                                 out_file=output_file_,
                                 categorical_mappings_json=mappings_json_,
                                 categories=['cdl_mode', 'nlcd'])

    elif run_general_workflow:
        extracts_dir = os.path.join(root, 'soils', 'swapstress', 'extracts')

        if run_gee_extract:
            shapefile = os.path.join(root, 'soils', 'gis', 'pretraining-roi-10000_mgrs.shp')
            index = 'site_id'
            output_prefix = 'swap-stress/training_data'
            mgrs_shapefile = os.path.join(root, 'boundaries', 'mgrs', 'mgrs_wgs.shp')

            is_authorized()
            get_bands(shapefile_path=shapefile,
                      mgrs_shp_path=mgrs_shapefile,
                      bucket=gcs_bucket,
                      file_prefix=output_prefix,
                      resolution=4000,
                      index_col=index,
                      split_tiles=True,
                      check_dir=extracts_dir)

        elif run_concatenate:
            rosetta_file_ = os.path.join(root, 'soils', 'rosetta', 'extracted_rosetta_points.parquet')
            output_file_ = os.path.join(root, 'soils', 'swapstress', 'training', 'training_data.parquet')
            mappings_json_ = os.path.join(root, 'soils', 'swapstress', 'training', 'categorical_mappings.json')

            concatenate_and_join(ee_in_dir=extracts_dir,
                                 rosetta_pqt=rosetta_file_,
                                 out_file=output_file_,
                                 categorical_mappings_json=mappings_json_,
                                 categories=['cdl_mode', 'nlcd'])

# ========================= EOF ====================================================================