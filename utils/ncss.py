"""Standardize NCSS lab water retention data into the pipeline's long form."""

import os
import pandas as pd
import geopandas as gpd

BAR_TO_CM = 1019.72
WRC_MAP = {
    "water_retention_6_hundredths": 0.06,
    "water_retention_10th_bar": 0.10,
    "water_retention_third_bar": 0.33,
    "water_retention_1_bar": 1.0,
    "water_retention_2_bar": 2.0,
    "water_retention_5_bar_sieve": 5.0,
    "water_retention_15_bar": 15.0,
}

# Physical limits for NCSS data sanity filtering
BULK_DENSITY_MIN = 0.5  # g/cm³ - lower bound for soils
BULK_DENSITY_MAX = 2.5  # g/cm³ - upper bound for mineral soils
THETA_MIN = 0.0
THETA_MAX = 1.0
SUCTION_CM_MAX = 1e6  # cm - max realistic suction


def ncss_to_standardized(df):
    id_cols = [
        "labsampnum",
        "pedon_key",
        "hzn_top",
        "hzn_bot",
        "hzn_mid_cm",
        "latitude_std_decimal_degrees",
        "longitude_std_decimal_degrees",
        "sand_total",
        "silt_total",
        "clay_total",
        "bulk_density_oven_dry",
    ]
    wr_cols = [c for c in WRC_MAP.keys() if c in df.columns]

    d0 = df[id_cols + wr_cols].copy()
    m = d0.melt(
        id_vars=id_cols, value_vars=wr_cols, var_name="wr_col", value_name="wr_val"
    )
    m = m.dropna(subset=["wr_val"])
    n_initial = len(m)
    dropped_reasons = []

    if "pedon_key" in m.columns:
        m["profile_id"] = m["pedon_key"].astype(str)
    else:
        m["profile_id"] = m["labsampnum"].astype(str)

    if "hzn_mid_cm" in m.columns:
        m["depth_cm"] = m["hzn_mid_cm"]
        if "hzn_top" in m.columns and "hzn_bot" in m.columns:
            m["depth_cm"] = m["depth_cm"].fillna(
                (m["hzn_top"].astype(float) + m["hzn_bot"].astype(float)) / 2.0
            )
    elif "hzn_top" in m.columns and "hzn_bot" in m.columns:
        m["depth_cm"] = (m["hzn_top"].astype(float) + m["hzn_bot"].astype(float)) / 2.0
    m["suction_cm"] = m["wr_col"].map(WRC_MAP).astype(float) * BAR_TO_CM

    # Filter invalid bulk densities before gravimetric->volumetric conversion
    bd = m["bulk_density_oven_dry"].astype(float)
    mask_bd_invalid = (bd < BULK_DENSITY_MIN) | (bd > BULK_DENSITY_MAX) | bd.isna()
    n_bd_invalid = mask_bd_invalid.sum()
    if n_bd_invalid > 0:
        dropped_reasons.append(f"invalid_bulk_density: {n_bd_invalid}")
        m = m[~mask_bd_invalid]

    # NCSS water retention typically reported as gravimetric percent
    grav = m["wr_val"].astype(float) / 100.0
    m["theta"] = grav * m["bulk_density_oven_dry"].astype(
        float
    )  # uses oven-dry bulk density

    # Filter theta outside physical bounds [0, 1]
    mask_theta_invalid = (m["theta"] < THETA_MIN) | (m["theta"] > THETA_MAX)
    n_theta_invalid = mask_theta_invalid.sum()
    if n_theta_invalid > 0:
        dropped_reasons.append(f"theta_outside_0-1: {n_theta_invalid}")
        m = m[~mask_theta_invalid]

    # Filter extreme suction values
    mask_suction_invalid = (m["suction_cm"] <= 0) | (m["suction_cm"] > SUCTION_CM_MAX)
    n_suction_invalid = mask_suction_invalid.sum()
    if n_suction_invalid > 0:
        dropped_reasons.append(f"suction_invalid: {n_suction_invalid}")
        m = m[~mask_suction_invalid]

    n_final = len(m)
    n_dropped = n_initial - n_final
    if n_dropped > 0:
        print(
            f"  [NCSS] Dropped {n_dropped}/{n_initial} rows: {', '.join(dropped_reasons)}"
        )

    m["db_od"] = m["bulk_density_oven_dry"]
    m["sand_tot_psa"] = m["sand_total"]
    m["silt_tot_psa"] = m["silt_total"]
    m["clay_tot_psa"] = m["clay_total"]
    m["source_db"] = "NCSS"

    keep = [
        "profile_id",
        "depth_cm",
        "suction_cm",
        "theta",
        "latitude_std_decimal_degrees",
        "longitude_std_decimal_degrees",
        "db_od",
        "sand_tot_psa",
        "silt_tot_psa",
        "clay_tot_psa",
        "source_db",
    ]
    out = m[keep].copy()

    # Derive SWCC coverage classes by profile
    sm = out.copy()
    sm["suction_m"] = sm["suction_cm"].astype(float) / 100.0
    g = sm.groupby("profile_id")["suction_cm"]
    has_wet = g.min() <= 150
    has_dry = g.max() >= 14000
    cls = pd.Series("NWND", index=has_wet.index)
    cls.loc[has_wet & has_dry] = "YWYD"
    cls.loc[has_wet & ~has_dry] = "YWND"
    cls.loc[~has_wet & has_dry] = "NWYD"
    cls = cls.rename("SWCC_classes")

    out = out.merge(cls, left_on="profile_id", right_index=True, how="left")
    out = out.sort_values(["profile_id", "suction_cm"]).reset_index(drop=True)
    return out


def load_ncss_parquet(parquet_path):
    df = pd.read_parquet(parquet_path)
    return df


def write_standardized(df, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)


def write_profile_shapefile(out_csv, out_shp):
    df = pd.read_csv(out_csv)
    df["profile_id"] = df["profile_id"].astype(str)
    profiles = df.dropna(
        subset=["latitude_std_decimal_degrees", "longitude_std_decimal_degrees"]
    )
    profiles = profiles.groupby("profile_id", as_index=False).first()
    points = gpd.GeoDataFrame(
        profiles[["profile_id"]].copy(),
        geometry=gpd.points_from_xy(
            profiles["longitude_std_decimal_degrees"].astype(float),
            profiles["latitude_std_decimal_degrees"].astype(float),
        ),
        crs="EPSG:4326",
    )
    os.makedirs(os.path.dirname(out_shp), exist_ok=True)
    points.to_file(out_shp)


def write_profile_mgrs_shapefile(out_csv, out_shp, mgrs_shp):
    df = pd.read_csv(out_csv)
    df["profile_id"] = df["profile_id"].astype(str)
    profiles = df.dropna(
        subset=["latitude_std_decimal_degrees", "longitude_std_decimal_degrees"]
    )
    profiles = profiles.groupby("profile_id", as_index=False).first()

    points = gpd.GeoDataFrame(
        profiles[["profile_id"]].copy(),
        geometry=gpd.points_from_xy(
            profiles["longitude_std_decimal_degrees"].astype(float),
            profiles["latitude_std_decimal_degrees"].astype(float),
        ),
        crs="EPSG:4326",
    )

    mgrs = gpd.read_file(mgrs_shp)[["MGRS_TILE", "geometry"]]
    if mgrs.crs != points.crs:
        mgrs = mgrs.to_crs(points.crs)

    joined = gpd.sjoin(points, mgrs, how="left", predicate="within")
    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])
    joined = joined.dropna(subset=["MGRS_TILE"])
    joined = joined[["profile_id", "MGRS_TILE", "geometry"]].copy()

    os.makedirs(os.path.dirname(out_shp), exist_ok=True)
    joined.to_file(out_shp)
    print(f"wrote {out_shp}")


if __name__ == "__main__":
    base_dir = "/nas/soils/soil_potential_obs/ncss_labdatasqlite"
    in_parquet = os.path.join(base_dir, "ncss_selection.parquet")
    out_csv = os.path.join(base_dir, "standardized_ncss.csv")
    out_shp = os.path.join(base_dir, "ncss_profiles.shp")
    mgrs_shp = "/nas/boundaries/mgrs/mgrs_world_attr.shp"

    df_ = load_ncss_parquet(in_parquet)
    std_ = ncss_to_standardized(df_)
    write_standardized(std_, out_csv)
    write_profile_mgrs_shapefile(out_csv, out_shp, mgrs_shp)

# ========================= EOF ====================================================================
