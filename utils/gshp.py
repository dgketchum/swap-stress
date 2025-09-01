import os
import re
import pandas as pd
import geopandas as gpd
from pathlib import Path
import os


def process_soil_data(csv_path, shp_path, output_dir):
    """
    Processes soil data by grouping, aggregating, creating a geodataframe,
    performing a spatial join with an MGRS grid, and exporting the results.

    Args:
        csv_path (str or Path): Path to the input soil data CSV file.
        shp_path (str or Path): Path to the MGRS grid shapefile.
        output_dir (str or Path): Directory to save the output files.
    """
    try:
        csv_path = Path(csv_path).expanduser()
        shp_path = Path(shp_path).expanduser()
        output_dir = Path(output_dir).expanduser()

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output will be saved to: {output_dir}")

        print(f"Loading soil data from: {csv_path}")
        df = pd.read_csv(csv_path, encoding='latin1')

        # Prepare a cleaned metadata CSV with a stable UID and normalized coords
        print("Preparing cleaned metadata CSV (uid, classes, coords, flags)...")
        required_cols = ['layer_id', 'SWCC_classes', 'latitude_decimal_degrees', 'longitude_decimal_degrees', 'data_flag',
                         'thetar', 'thetas', 'alpha', 'n']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"Warning: missing expected columns in source CSV: {missing}")

        def _sanitize_uid(val):
            s = str(val) if pd.notnull(val) else ''
            s = re.sub(r"\s+", "", s)
            s = s.replace('.', '_')
            s = re.sub(r"_+", "_", s)
            return s

        # Build a per-site summary using coordinates to avoid duplicate extracts at the same point
        cols_present = [c for c in required_cols if c in df.columns]
        base = df[cols_present].copy()

        lat_col = 'latitude_decimal_degrees' if 'latitude_decimal_degrees' in base.columns else None
        lon_col = 'longitude_decimal_degrees' if 'longitude_decimal_degrees' in base.columns else None

        if lat_col and lon_col:
            group_keys = [lat_col, lon_col]
        else:
            # Fallback to per-layer grouping if coordinates are unavailable
            group_keys = ['layer_id']

        agg_cols = [c for c in cols_present if c not in group_keys]
        agg_spec = {c: 'first' for c in agg_cols}

        grouped_clean = base.groupby(group_keys, dropna=False).agg(agg_spec).reset_index()
        obs_counts = base.groupby(group_keys, dropna=False).size().reset_index(name='obs_ct')
        grouped_clean = grouped_clean.merge(obs_counts, on=group_keys, how='left')

        # Create uid: prefer layer_id if available; otherwise derive from coordinates
        if 'layer_id' in grouped_clean.columns:
            grouped_clean['uid'] = grouped_clean['layer_id'].apply(_sanitize_uid)
        else:
            # Use rounded coordinates to build a stable UID
            grouped_clean['uid'] = grouped_clean.apply(
                lambda r: _sanitize_uid(f"{r['latitude']:.6f}_{r['longitude']:.6f}"), axis=1
            )

        # Drop layer_id if present to keep a single identifier downstream
        clean_df = grouped_clean.drop(columns=['layer_id']) if 'layer_id' in grouped_clean.columns else grouped_clean.copy()

        # Rename coordinates
        rename_map = {}
        if 'latitude_decimal_degrees' in clean_df.columns:
            rename_map['latitude_decimal_degrees'] = 'latitude'
        if 'longitude_decimal_degrees' in clean_df.columns:
            rename_map['longitude_decimal_degrees'] = 'longitude'
        clean_df.rename(columns=rename_map, inplace=True)

        # Enforce unique uid; raise if duplicates found after sanitization (should be unique per coordinate)
        dup_mask = clean_df['uid'].duplicated(keep=False)
        if dup_mask.any():
            dup_vals = sorted(set(clean_df.loc[dup_mask, 'uid']))
            example_vals = ", ".join(dup_vals[:10])
            raise ValueError(f"Duplicate uid values found ({len(dup_vals)} unique duplicates). Examples: {example_vals}")

        rename_map = {'thetar': 'theta_r', 'thetas': 'theta_s'}
        clean_df = clean_df.rename(columns=rename_map)

        clean_name = Path(csv_path).stem + '_clean.csv'
        clean_path = output_dir / clean_name
        clean_df.to_csv(clean_path, index=False)
        print(f"Wrote cleaned metadata CSV: {clean_path}")

        print(f"Loading MGRS grid from: {shp_path}")
        mgrs_gdf = gpd.read_file(shp_path)

        # Use cleaned metadata for spatial processing
        if not {'latitude', 'longitude'}.issubset(clean_df.columns):
            raise ValueError("Cleaned metadata missing 'latitude' and/or 'longitude' columns")

        print("Creating GeoDataFrame from cleaned metadata...")
        geometry = gpd.points_from_xy(clean_df['longitude'], clean_df['latitude'])
        soil_gdf = gpd.GeoDataFrame(clean_df.copy(), geometry=geometry, crs='EPSG:4326')
        print(f"Created GeoDataFrame with {len(soil_gdf)} features.")

        print("Performing spatial join with MGRS grid...")

        if soil_gdf.crs != mgrs_gdf.crs:
            print(f"Reprojecting MGRS grid to {soil_gdf.crs}...")
            mgrs_gdf = mgrs_gdf.to_crs(soil_gdf.crs)

        joined_gdf = gpd.sjoin(
            soil_gdf,
            mgrs_gdf[['MGRS_TILE', 'geometry']],
            how='inner',
            predicate='intersects'
        )

        joined_gdf = joined_gdf.drop(columns=['index_right'])
        print("Spatial join complete.")

        output_csv_path = output_dir / 'wrc_aggregated_mgrs.csv'
        output_shp_path = output_dir / 'wrc_aggregated_mgrs.shp'

        print(f"Exporting CSV to: {output_csv_path}")
        joined_gdf.drop(columns='geometry').to_csv(output_csv_path, index=False)

        print(f"Exporting Shapefile to: {output_shp_path}")
        joined_gdf.to_file(output_shp_path)

        print("\nProcessing finished successfully!")

    except FileNotFoundError as e:
        print(f"Error: File not found. Please check the path: {e.filename}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    home_dir = os.path.expanduser('~')
    root_ = os.path.join(home_dir, 'data', 'IrrigationGIS')

    gshp_directory_ = os.path.join(root_, 'soils', 'vg_paramaer_databases', 'wrc')
    soil_csv_path_ = os.path.join(gshp_directory_, 'WRC_dataset_surya_et_al_2021_final.csv')

    mgrs_shp_path_ = os.path.join(root_, 'boundaries', 'mgrs', 'mgrs_world_attr.shp')

    process_soil_data(soil_csv_path_, mgrs_shp_path_, gshp_directory_)

# ========================= EOF ====================================================================
