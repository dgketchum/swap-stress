"""Ad-hoc extraction of FROM-GLC10 land cover class at ISMN CONUS stations.

Asset: projects/sat-io/open-datasets/FROM-GLC10  (10 m, single band)
Class values: 10=Cropland, 20=Forest, 30=Grass, 40=Shrub, 60=Water,
              80=Impervious, 90=Bareland, 100=Snow/Ice
"""

import os

import ee
import geopandas as gpd
import pandas as pd

from map.data.call_ee import is_authorized

CLASS_LABELS = {
    10: "Cropland",
    20: "Forest",
    30: "Grass",
    40: "Shrub",
    60: "Water",
    80: "Impervious",
    90: "Bareland",
    100: "Snow/Ice",
}

CONUS_BOUNDS = (-125.0, 24.0, -66.5, 50.0)


def extract_glc10_at_ismn(shapefile_path, out_csv, batch_size=5000):
    """Sample FROM-GLC10 at each ISMN station within CONUS and write CSV."""

    is_authorized()

    gdf = gpd.read_file(shapefile_path)

    minx, miny, maxx, maxy = CONUS_BOUNDS
    conus = gdf.cx[minx:maxx, miny:maxy].copy()
    print(f"{len(conus)} ISMN stations in CONUS (of {len(gdf)} total)")

    glc = (
        ee.ImageCollection("projects/sat-io/open-datasets/FROM-GLC10")
        .mosaic()
        .rename("glc10")
    )

    results = []

    for start in range(0, len(conus), batch_size):
        batch = conus.iloc[start : start + batch_size]
        fc = ee.FeatureCollection(batch.__geo_interface__)

        sampled = glc.sampleRegions(
            collection=fc,
            properties=["station_ui"],
            scale=10,
            geometries=False,
        )

        rows = sampled.getInfo()["features"]
        for f in rows:
            props = f["properties"]
            code = props.get("glc10")
            results.append(
                {
                    "station_ui": props["station_ui"],
                    "glc10_code": code,
                    "glc10_label": CLASS_LABELS.get(code, "Unknown"),
                }
            )
        print(f"  batch {start}-{start + len(batch)}: {len(rows)} sampled")

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows to {out_csv}")

    print("\nLand cover distribution:")
    print(df["glc10_label"].value_counts().to_string())


if __name__ == "__main__":
    root = "/nas"
    shp = os.path.join(
        root, "soils", "vwc_timeseries", "ismn", "ismn_stations_mgrs.shp"
    )
    out = os.path.join(root, "soils", "vwc_timeseries", "ismn", "ismn_conus_glc10.csv")

    extract_glc10_at_ismn(shp, out)
