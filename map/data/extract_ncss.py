from pathlib import Path
import sqlite3
import pandas as pd
import geopandas as gpd

# The SQL query to select the desired soil data from the NCSS database.
NCSS_SELECT_SQL = """
SELECT
  LPP.labsampnum,
  LL.site_key,
  LL.pedon_key,
  LS.horizontal_datum_name,
  LS.latitude_std_decimal_degrees,
  LS.longitude_std_decimal_degrees,
  LL.layer_sequence,
  LL.layer_type,
  LL.hzn_top,
  LL.hzn_bot,
  LL.hzn_desgn,
  ((LL.hzn_top + LL.hzn_bot)/2.0) AS hzn_mid_cm,
  sand_total,
  silt_total,
  clay_total,
  bulk_density_third_bar,
  bulk_density_oven_dry,
  water_retention_6_hundredths,
  water_retention_10th_bar,
  water_retention_third_bar,
  water_retention_1_bar,
  water_retention_2_bar,
  water_retention_5_bar_sieve,
  water_retention_15_bar
FROM lab_physical_properties AS LPP
JOIN lab_layer  AS LL ON LPP.labsampnum = LL.labsampnum
JOIN lab_site   AS LS ON LL.site_key = LS.site_key
JOIN lab_pedon  AS LP ON LL.pedon_key = LP.pedon_key
WHERE sand_total >= 0
  AND water_retention_third_bar > 0
  AND bulk_density_third_bar BETWEEN 0.5 AND 2.0
  AND LL.hzn_top <= 250
  AND LS.horizontal_datum_name IN ('NAD27','NAD83','WGS84')
"""

def export_ncss_parquet_and_shapefile(in_db: Path, parquet_path: Path, shp_path: Path) -> None:
    """
    Reads data from the source NCSS database and exports the selection
    to both Parquet and Shapefile formats.
    """
    con = sqlite3.connect(f"file:{in_db.as_posix()}?mode=ro", uri=True, timeout=60.0)
    try:
        df = pd.read_sql_query(NCSS_SELECT_SQL, con)
    finally:
        con.close()

    # Count the number of non-null water retention observations for each row
    wr_cols = [
        'water_retention_6_hundredths',
        'water_retention_10th_bar',
        'water_retention_third_bar',
        'water_retention_1_bar',
        'water_retention_2_bar',
        'water_retention_5_bar_sieve',
        'water_retention_15_bar'
    ]
    df['obs_ct'] = df[wr_cols].notna().sum(axis=1)

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path)

    # Create GeoDataFrame for Shapefile export
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude_std_decimal_degrees"], df["latitude_std_decimal_degrees"]),
        crs="EPSG:4326",
    )

    # Define a mapping for renaming columns to be Shapefile-compliant (<= 10 chars)
    # 1 bar = 1019.72 cm H2O
    rename_map = {
        'horizontal_datum_name': 'datum',
        'latitude_std_decimal_degrees': 'lat_dd',
        'longitude_std_decimal_degrees': 'lon_dd',
        'layer_sequence': 'layer_seq',
        'bulk_density_third_bar': 'bd_3bar',
        'bulk_density_oven_dry': 'bd_od',
        'water_retention_6_hundredths': 'wr_61cm',      # 0.06 bar
        'water_retention_10th_bar': 'wr_102cm',    # 0.1 bar
        'water_retention_third_bar': 'wr_340cm',    # 0.33 bar
        'water_retention_1_bar': 'wr_1020cm',   # 1 bar
        'water_retention_2_bar': 'wr_2039cm',   # 2 bar
        'water_retention_5_bar_sieve': 'wr_5099cm',   # 5 bar
        'water_retention_15_bar': 'wr_15296cm',  # 15 bar
    }
    gdf.rename(columns=rename_map, inplace=True)


    # Export to Shapefile with shortened column names
    gdf.to_file(shp_path)

if __name__ == "__main__":
    # Define the input and output paths
    base_dir = Path("~/data/IrrigationGIS/soils/soil_potential_obs/ncss_labdatasqlite").expanduser()
    in_db = base_dir / "ncss_labdata.sqlite"
    parquet_path = base_dir / "ncss_selection.parquet"
    shp_path = base_dir / "ncss_selection.shp"

    # Run the export process
    export_ncss_parquet_and_shapefile(in_db, parquet_path, shp_path)


# ========================= EOF ====================================================================
