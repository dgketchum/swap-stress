"""
Data source registry for soil hydraulic parameter training data.

This module defines a unified interface for all training data sources,
enabling consistent data loading, VG parameter handling, and train-time
source selection.

Sources:
    - GSHP: Global Soil Hydraulic Properties (lab WRC, VG params from CSV)
    - NCSS: National Cooperative Soil Survey (lab WRC, VG params fitted)
    - MT Mesonet: Montana Mesonet stations (field SWP/VWC, VG params fitted)
    - ReESH: Remote sensing ecosystem sites (field WRC, VG params fitted)
    - Rosetta: Gridded prior from pedotransfer (pre-computed VG params)

Usage:
    from swapstress.sources.registry import SOURCES, get_source

    # Get a specific source config
    gshp = get_source('gshp')
    print(gshp.index_col)  # 'profile_id'

    # Iterate over sources that need VG fitting
    for name, src in SOURCES.items():
        if not src.has_vg_params:
            print(f"{name} requires VG fitting")
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, List


# Columns to drop when preparing features (identifiers, metadata)
STANDARD_DROP_COLS = [
    "MGRS_TILE",
    "station",
    "rosetta_level",
    "profile_id",
    "sample_id",
    "nwsli_id",
    "network",
    "mesowest_i",
    "data_flag",
    "obs_ct",
    "SWCC_class",
    "source",
    "depth",
    "depth_cm",
    "lat",
    "lon",
    "latitude",
    "longitude",
]

# Extra metadata columns to drop from training table (shapefile attributes, duplicates)
TRAINING_TABLE_DROP_COLS = [
    "station",
    "site_id",
    "profile_id",  # Index cols (already in sample_id)
    "Latitude",
    "Longitude",
    "latitude",
    "longitude",  # Duplicates of lat/lon
    "date_insta",
    "sub_networ",  # Shapefile metadata
]


@dataclass
class DataSource:
    """Configuration for a training data source."""

    # Identity
    name: str
    description: str

    # Index/identifier configuration
    index_col: str  # Primary key column: 'profile_id', 'station', etc.
    group_col: str  # Column for grouped train/test splits

    # VG parameter configuration
    has_vg_params: bool  # True if source comes with VG params (GSHP, Rosetta)
    vg_param_format: str  # 'natural' (alpha, n) or 'log10' (log10_alpha, log10_n)
    vg_source: str  # 'labels_csv', 'fitted_json', 'rosetta_geotiff', 'rosetta_join'

    # Path components (relative to data root)
    ee_extracts_subdir: str  # e.g., 'gshp_extracts_250m'
    ee_table_filename: str  # e.g., 'gshp_ee_data_250m.parquet'

    # Raw, pre-standardization inputs -- what stage 00 reads. Sources whose
    # standardizer takes a directory rather than named files leave raw_files
    # empty; the roles are whatever that source's write_standardized_* expects.
    raw_subdir: Optional[str] = None  # relative to data_root
    raw_files: Dict[str, str] = field(default_factory=dict)  # role -> filename

    # Earth Engine extraction (stage 01). The MGRS index differs between the
    # CONUS sources and the global ones, and only the Rosetta pretraining ROI is
    # dense enough that a tile has to be split before it will export.
    mgrs_subpath: str = "boundaries/mgrs/mgrs_world_attr.shp"
    ee_split_tiles: bool = False
    ee_region: str = "global"
    ee_prefix_template: Optional[str] = None  # '{resolution}' is substituted

    # For sources with pre-existing VG params
    labels_subpath: Optional[str] = None  # Path to labels CSV/parquet

    # For sources requiring VG fitting
    fit_results_subdir: Optional[str] = None  # Subdir in curve_fits/
    preprocessed_subdir: Optional[str] = None  # Subdir in preprocessed/

    # Depth handling
    depth_col: str = "depth_cm"  # Standard depth column after standardization
    depth_from_horizon: bool = False  # True if depth = (hzn_top + hzn_bot) / 2

    # Optional embeddings directory name
    embeddings_subdir: Optional[str] = None

    # Quality filtering
    quality_filter_col: Optional[str] = None  # e.g., 'data_flag'
    quality_filter_value: Optional[str] = None  # e.g., 'good quality estimate'

    # Shapefile metadata (for 9km single-CSV workflows)
    shapefile_subpath: Optional[str] = None  # relative to data_root
    lat_col: Optional[str] = "latitude"  # coordinate column in shapefile
    lon_col: Optional[str] = "longitude"

    # Extra columns to drop (source-specific)
    extra_drop_cols: List[str] = field(default_factory=list)

    def get_drop_cols(self) -> List[str]:
        """Return all columns to drop when preparing features."""
        return STANDARD_DROP_COLS + self.extra_drop_cols


# =============================================================================
# Source Definitions
# =============================================================================

SOURCES = {
    "gshp": DataSource(
        name="gshp",
        description="Global Soil Hydraulic Properties - lab water retention curves",
        index_col="profile_id",
        group_col="profile_id",
        has_vg_params=True,
        vg_param_format="natural",
        vg_source="labels_csv",
        ee_extracts_subdir="gshp_extracts_250m",
        ee_table_filename="gshp_ee_data_250m.parquet",
        raw_subdir="soil_potential_obs/gshp",
        raw_files={"curves": "WRC_dataset_surya_et_al_2021_final.csv"},
        ee_prefix_template="swapstress/gshp_training_data_{resolution}m",
        labels_subpath="soil_potential_obs/gshp/WRC_dataset_surya_et_al_2021_final_clean.csv",
        preprocessed_subdir="gshp",
        # No fit_results_subdir: GSHP is the one source we do not refit. Its
        # parameters come from the published dataset via swapstress.sources.gshp, so
        # there is no curve_fits/gshp/ directory to fall back to.
        fit_results_subdir=None,
        depth_col="depth_cm",
        depth_from_horizon=True,  # Uses (hzn_top + hzn_bot) / 2
        embeddings_subdir="gshp",
        quality_filter_col="data_flag",
        quality_filter_value="good quality estimate",
        shapefile_subpath="soil_potential_obs/gshp/wrc_aggregated_mgrs.shp",
        extra_drop_cols=["hzn_top", "hzn_bot", "SWCC_classes", "climate_classes"],
    ),
    "ncss": DataSource(
        name="ncss",
        description="National Cooperative Soil Survey - lab water retention data",
        index_col="profile_id",
        group_col="profile_id",
        has_vg_params=False,
        vg_param_format="natural",
        vg_source="fitted_json",
        ee_extracts_subdir="ncss_extracts_250m",
        ee_table_filename="ncss_ee_data_250m.parquet",
        raw_subdir="soil_potential_obs/ncss_labdatasqlite",
        raw_files={"curves": "ncss_selection.parquet"},
        ee_prefix_template="swapstress/ncss_training_data_{resolution}m",
        preprocessed_subdir="ncss",
        fit_results_subdir="ncss",
        depth_col="depth_cm",
        embeddings_subdir="ncss",
        shapefile_subpath="soil_potential_obs/ncss_labdatasqlite/ncss_profiles.shp",
        lat_col=None,
        lon_col=None,
        extra_drop_cols=["SWCC_classes", "source_db"],
    ),
    "mt_mesonet": DataSource(
        name="mt_mesonet",
        description="Montana Mesonet - field soil water potential stations",
        index_col="station",
        group_col="station",
        has_vg_params=False,
        vg_param_format="natural",
        vg_source="fitted_json",
        ee_extracts_subdir="mt_mesonet_extracts_250m",
        ee_table_filename="mt_ee_data_250m.parquet",
        raw_subdir="soil_potential_obs/mt_mesonet",
        raw_files={"swp": "swp.csv", "metadata": "station_metadata.csv"},
        mgrs_subpath="boundaries/mgrs/mgrs_wgs.shp",
        ee_region="conus",
        ee_prefix_template="swapstress/mesonet_training_data_{resolution}m",
        preprocessed_subdir="mt_mesonet",
        fit_results_subdir="mt_mesonet",
        depth_col="depth_cm",
        embeddings_subdir="mt_mesonet",
        # station_metadata_clean_mgrs.shp exists alongside this and carries fewer
        # stray columns, but the extract on disk was sampled at the path below,
        # so the ee_tables join has to keep using it until stage 01 is re-run.
        shapefile_subpath="soil_potential_obs/mt_mesonet/station_metadata_mgrs.shp",
    ),
    "reesh": DataSource(
        name="reesh",
        description="ReESH - Ameriflux ecosystem sites with soil WRC",
        index_col="site_id",
        group_col="site_id",
        has_vg_params=False,
        vg_param_format="natural",
        vg_source="fitted_json",
        ee_extracts_subdir="reesh_extracts_250m",
        ee_table_filename="reesh_ee_data_250m.parquet",
        raw_subdir="soil_potential_obs/reesh",
        ee_prefix_template="swapstress/reesh_training_data_{resolution}m",
        preprocessed_subdir="reesh",
        fit_results_subdir="reesh",
        depth_col="depth_cm",
        embeddings_subdir="reesh",
        shapefile_subpath="soil_potential_obs/reesh/shapefile/reesh_sites_mgrs.shp",
        lat_col="Latitude",
        lon_col="Longitude",
    ),
    "lacadian": DataSource(
        name="lacadian",
        description="LaCADIAN - Louisiana field soil moisture/potential stations",
        index_col="station",
        group_col="station",
        has_vg_params=False,
        vg_param_format="natural",
        vg_source="fitted_json",
        ee_extracts_subdir="lacadian_extracts_250m",
        ee_table_filename="lacadian_ee_data_250m.parquet",
        raw_subdir="soil_potential_obs/lacadian",
        raw_files={"swp": "swp.csv", "metadata": "station_metadata.csv"},
        ee_prefix_template="swapstress/lacadian_training_data_{resolution}m",
        preprocessed_subdir="lacadian",
        fit_results_subdir="lacadian",
        depth_col="depth_cm",
        embeddings_subdir="lacadian",
        shapefile_subpath="soil_potential_obs/lacadian/lacadian_stations_mgrs.shp",
    ),
    "rosetta": DataSource(
        name="rosetta",
        description="Rosetta gridded pedotransfer predictions (7 depth levels)",
        index_col="site_id",
        group_col="site_id",
        has_vg_params=True,
        vg_param_format="log10",
        vg_source="rosetta_join",  # Joined during ee_tables.py processing
        ee_extracts_subdir="rosetta_extracts_250m",
        ee_table_filename="training_data.parquet",
        raw_subdir="rosetta/training_data",
        raw_files={"curves": "rosetta_curves_wide.csv"},
        # The pretraining ROI is a dense CONUS point grid, not a station set, so
        # it uses its own shapefile and is the one source exported tile by tile.
        shapefile_subpath="gis/pretraining-roi-10000_mgrs.shp",
        mgrs_subpath="boundaries/mgrs/mgrs_wgs.shp",
        ee_split_tiles=True,
        ee_region="conus",
        ee_prefix_template="swapstress/training_data",
        depth_col="rosetta_level",  # Uses level (1-7) not depth_cm
        extra_drop_cols=[
            # Rosetta columns are named US_R3H3_L{level}_VG_{param}
            # These get handled specially in training
        ],
    ),
}


# The sources the released model trains on. Rosetta is registered too, but it is
# the 250 m pretraining prior rather than an observation source, so it is opted
# into explicitly rather than picked up by default.
DEFAULT_SOURCES = ["gshp", "ncss", "mt_mesonet", "reesh", "lacadian"]


def get_source(name: str) -> DataSource:
    """
    Get a DataSource configuration by name.

    Parameters
    ----------
    name : str
        Source name (case-insensitive).

    Returns
    -------
    DataSource
        Configuration for the requested source.

    Raises
    ------
    KeyError
        If source name is not found.
    """
    key = name.lower()
    if key not in SOURCES:
        available = ", ".join(SOURCES.keys())
        raise KeyError(f"Unknown source '{name}'. Available: {available}")
    return SOURCES[key]


def list_sources() -> None:
    """Print summary of all registered sources."""
    print(
        f"{'Name':<12} {'Has VG':<8} {'VG Format':<10} {'Index Col':<12} {'Description'}"
    )
    print("-" * 80)
    for name, src in SOURCES.items():
        has_vg = "Yes" if src.has_vg_params else "No"
        print(
            f"{name:<12} {has_vg:<8} {src.vg_param_format:<10} {src.index_col:<12} {src.description}"
        )


# "250m" retained for historical reproducibility; active releases use 9km scales.
VALID_SCALES = ("250m", "9km_conus", "9km_global")


class DataPaths:
    """
    Helper class for resolving data paths for a source.

    Centralizes path construction to avoid hardcoded paths throughout codebase.
    """

    def __init__(
        self,
        data_root: str,
        source: DataSource,
        scale: str = "9km_global",
        boundaries_root: Optional[str] = None,
    ):
        """
        Initialize path resolver.

        Parameters
        ----------
        data_root : str
            Root data directory (e.g., /nas/soils)
        source : DataSource
            Source configuration.
        scale : str
            Resolution scale: "9km_global" (default), "9km_conus", or "250m" (historical).
        boundaries_root : str, optional
            Root holding the MGRS tile index, which sits beside the soils tree
            rather than inside it. Defaults to the parent of *data_root*.
        """
        if scale not in VALID_SCALES:
            raise ValueError(f"Invalid scale '{scale}'. Must be one of {VALID_SCALES}")
        self.data_root = os.path.expanduser(data_root)
        self.source = source
        self.scale = scale
        self.boundaries_root = (
            os.path.expanduser(boundaries_root)
            if boundaries_root
            else os.path.dirname(self.data_root)
        )

    @property
    def is_single_csv(self) -> bool:
        """Whether this scale uses a single CSV (9km) vs per-tile CSVs (250m)."""
        return self.scale != "250m"

    @property
    def ee_extracts_dir(self) -> str:
        """Directory containing raw EE CSV extracts."""
        if self.scale == "9km_conus":
            return os.path.join(
                self.data_root,
                "swapstress",
                "inference",
                "conus_features",
                self.source.name,
            )
        elif self.scale == "9km_global":
            return os.path.join(
                self.data_root,
                "swapstress",
                "inference",
                "global_features",
                self.source.name,
            )
        return os.path.join(
            self.data_root, "swapstress", "extracts", self.source.ee_extracts_subdir
        )

    @property
    def ee_table(self) -> str:
        """Path to concatenated EE features parquet."""
        if self.scale == "9km_conus":
            return os.path.join(
                self.data_root,
                "swapstress",
                "training",
                f"{self.source.name}_ee_data_9km_conus.parquet",
            )
        elif self.scale == "9km_global":
            return os.path.join(
                self.data_root,
                "swapstress",
                "training",
                f"{self.source.name}_ee_data_9km_global.parquet",
            )
        return os.path.join(
            self.data_root, "swapstress", "training", self.source.ee_table_filename
        )

    @property
    def ee_csv_file(self) -> Optional[str]:
        """Path to the single CSV for 9km scales (None for 250m)."""
        if self.is_single_csv:
            return os.path.join(self.ee_extracts_dir, "point_extract_9km.csv")
        return None

    @property
    def raw_dir(self) -> Optional[str]:
        """Directory holding this source's raw, pre-standardization files."""
        if self.source.raw_subdir:
            return os.path.join(self.data_root, self.source.raw_subdir)
        return None

    def raw_file(self, role: str) -> str:
        """Path to a named raw input, e.g. ``raw_file('swp')``.

        Raises rather than returning None: a missing role means the source
        definition and its standardizer disagree, which is a bug, not a
        condition to fall back from.
        """
        if role not in self.source.raw_files:
            known = ", ".join(sorted(self.source.raw_files)) or "(none)"
            raise KeyError(
                f"Source '{self.source.name}' has no raw file role '{role}'. "
                f"Known roles: {known}"
            )
        return os.path.join(self.raw_dir, self.source.raw_files[role])

    @property
    def mgrs_shapefile(self) -> str:
        """MGRS tile index this source's Earth Engine extract is blocked on."""
        return os.path.join(self.boundaries_root, self.source.mgrs_subpath)

    def ee_output_prefix(self, resolution: int) -> Optional[str]:
        """Cloud Storage prefix for this source's exported tiles."""
        if self.source.ee_prefix_template is None:
            return None
        return self.source.ee_prefix_template.format(resolution=resolution)

    @property
    def shapefile(self) -> Optional[str]:
        """Path to source shapefile."""
        if self.source.shapefile_subpath:
            return os.path.join(self.data_root, self.source.shapefile_subpath)
        return None

    @property
    def labels_file(self) -> Optional[str]:
        """Path to labels CSV/parquet (for sources with pre-existing VG params)."""
        if self.source.labels_subpath:
            return os.path.join(self.data_root, self.source.labels_subpath)
        return None

    @property
    def preprocessed_dir(self) -> Optional[str]:
        """Directory containing standardized observation CSVs."""
        if self.source.preprocessed_subdir:
            return os.path.join(
                self.data_root,
                "soil_potential_obs",
                "preprocessed",
                self.source.preprocessed_subdir,
            )
        return None

    @property
    def fit_results_dir(self) -> Optional[str]:
        """Directory containing fitted VG parameter JSONs."""
        if self.source.fit_results_subdir:
            return os.path.join(
                self.data_root,
                "soil_potential_obs",
                "curve_fits",
                self.source.fit_results_subdir,
            )
        return None

    @property
    def embeddings_dir(self) -> Optional[str]:
        """Directory containing embedding parquet files."""
        if self.scale != "250m":
            return None
        if self.source.embeddings_subdir:
            return os.path.join(
                "/data/ssd2/swapstress/vwc/embeddings", self.source.embeddings_subdir
            )
        return None


if __name__ == "__main__":
    list_sources()
    print()

    # Example usage
    gshp = get_source("gshp")
    paths = DataPaths("/nas/soils", gshp)
    print(f"GSHP EE table: {paths.ee_table}")
    print(f"GSHP labels: {paths.labels_file}")
    print(f"GSHP fit results: {paths.fit_results_dir}")
