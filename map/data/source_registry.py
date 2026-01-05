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
    from map.data.source_registry import SOURCES, get_source

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
from typing import Optional, List, Callable


# Standard VG parameter names
VG_PARAMS_NATURAL = ['theta_r', 'theta_s', 'alpha', 'n']
VG_PARAMS_LOG10 = ['theta_r', 'theta_s', 'log10_alpha', 'log10_n', 'log10_Ks']

# Columns to drop when preparing features (identifiers, metadata)
STANDARD_DROP_COLS = [
    'MGRS_TILE', 'station', 'rosetta_level', 'profile_id', 'sample_id',
    'nwsli_id', 'network', 'mesowest_i', 'data_flag', 'obs_ct', 'SWCC_class',
    'source', 'depth', 'depth_cm', 'lat', 'lon', 'latitude', 'longitude',
]

# Extra metadata columns to drop from training table (shapefile attributes, duplicates)
TRAINING_TABLE_DROP_COLS = [
    'station', 'site_id', 'profile_id',  # Index cols (already in sample_id)
    'Latitude', 'Longitude', 'latitude', 'longitude',  # Duplicates of lat/lon
    'date_insta', 'sub_networ',  # Shapefile metadata
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

    # For sources with pre-existing VG params
    labels_subpath: Optional[str] = None  # Path to labels CSV/parquet

    # For sources requiring VG fitting
    fit_results_subdir: Optional[str] = None  # Subdir in curve_fits/
    preprocessed_subdir: Optional[str] = None  # Subdir in preprocessed/

    # Depth handling
    depth_col: str = 'depth_cm'  # Standard depth column after standardization
    depth_from_horizon: bool = False  # True if depth = (hzn_top + hzn_bot) / 2

    # Optional embeddings directory name
    embeddings_subdir: Optional[str] = None

    # Quality filtering
    quality_filter_col: Optional[str] = None  # e.g., 'data_flag'
    quality_filter_value: Optional[str] = None  # e.g., 'good quality estimate'

    # Extra columns to drop (source-specific)
    extra_drop_cols: List[str] = field(default_factory=list)

    def get_vg_param_cols(self) -> List[str]:
        """Return the VG parameter column names for this source."""
        if self.vg_param_format == 'log10':
            return VG_PARAMS_LOG10
        return VG_PARAMS_NATURAL

    def get_drop_cols(self) -> List[str]:
        """Return all columns to drop when preparing features."""
        return STANDARD_DROP_COLS + self.extra_drop_cols + self.get_vg_param_cols()


# =============================================================================
# Source Definitions
# =============================================================================

SOURCES = {
    'gshp': DataSource(
        name='gshp',
        description='Global Soil Hydraulic Properties - lab water retention curves',
        index_col='profile_id',
        group_col='profile_id',
        has_vg_params=True,
        vg_param_format='natural',
        vg_source='labels_csv',
        ee_extracts_subdir='gshp_extracts_250m',
        ee_table_filename='gshp_ee_data_250m.parquet',
        labels_subpath='soil_potential_obs/gshp/WRC_dataset_surya_et_al_2021_final_clean.csv',
        preprocessed_subdir='gshp',
        fit_results_subdir='gshp',
        depth_col='depth_cm',
        depth_from_horizon=True,  # Uses (hzn_top + hzn_bot) / 2
        embeddings_subdir='gshp',
        quality_filter_col='data_flag',
        quality_filter_value='good quality estimate',
        extra_drop_cols=['hzn_top', 'hzn_bot', 'SWCC_classes', 'climate_classes'],
    ),

    'ncss': DataSource(
        name='ncss',
        description='National Cooperative Soil Survey - lab water retention data',
        index_col='profile_id',
        group_col='profile_id',
        has_vg_params=False,
        vg_param_format='natural',
        vg_source='fitted_json',
        ee_extracts_subdir='ncss_extracts_250m',
        ee_table_filename='ncss_ee_data_250m.parquet',
        preprocessed_subdir='ncss',
        fit_results_subdir='ncss',
        depth_col='depth_cm',
        embeddings_subdir='ncss',
        extra_drop_cols=['SWCC_classes', 'source_db'],
    ),

    'mt_mesonet': DataSource(
        name='mt_mesonet',
        description='Montana Mesonet - field soil water potential stations',
        index_col='station',
        group_col='station',
        has_vg_params=False,
        vg_param_format='natural',
        vg_source='fitted_json',
        ee_extracts_subdir='mt_mesonet_extracts_250m',
        ee_table_filename='mt_ee_data_250m.parquet',
        preprocessed_subdir='mt_mesonet',
        fit_results_subdir='mt_mesonet',
        depth_col='depth_cm',
        embeddings_subdir='mt_mesonet',
    ),

    'reesh': DataSource(
        name='reesh',
        description='ReESH - Ameriflux ecosystem sites with soil WRC',
        index_col='site_id',
        group_col='site_id',
        has_vg_params=False,
        vg_param_format='natural',
        vg_source='fitted_json',
        ee_extracts_subdir='reesh_extracts_250m',
        ee_table_filename='reesh_ee_data_250m.parquet',
        preprocessed_subdir='reesh',
        fit_results_subdir='reesh',
        depth_col='depth_cm',
        embeddings_subdir='reesh',
    ),

    'rosetta': DataSource(
        name='rosetta',
        description='Rosetta gridded pedotransfer predictions (7 depth levels)',
        index_col='site_id',
        group_col='site_id',
        has_vg_params=True,
        vg_param_format='log10',
        vg_source='rosetta_join',  # Joined during ee_tables.py processing
        ee_extracts_subdir='rosetta_extracts_250m',
        ee_table_filename='training_data.parquet',
        depth_col='rosetta_level',  # Uses level (1-7) not depth_cm
        extra_drop_cols=[
            # Rosetta columns are named US_R3H3_L{level}_VG_{param}
            # These get handled specially in training
        ],
    ),
}


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
        available = ', '.join(SOURCES.keys())
        raise KeyError(f"Unknown source '{name}'. Available: {available}")
    return SOURCES[key]


def get_sources_requiring_fitting() -> List[DataSource]:
    """Return list of sources that require VG parameter fitting."""
    return [src for src in SOURCES.values() if not src.has_vg_params]


def get_sources_with_params() -> List[DataSource]:
    """Return list of sources that come with VG parameters."""
    return [src for src in SOURCES.values() if src.has_vg_params]


def list_sources() -> None:
    """Print summary of all registered sources."""
    print(f"{'Name':<12} {'Has VG':<8} {'VG Format':<10} {'Index Col':<12} {'Description'}")
    print("-" * 80)
    for name, src in SOURCES.items():
        has_vg = 'Yes' if src.has_vg_params else 'No'
        print(f"{name:<12} {has_vg:<8} {src.vg_param_format:<10} {src.index_col:<12} {src.description}")


class DataPaths:
    """
    Helper class for resolving data paths for a source.

    Centralizes path construction to avoid hardcoded paths throughout codebase.
    """

    def __init__(self, data_root: str, source: DataSource):
        """
        Initialize path resolver.

        Parameters
        ----------
        data_root : str
            Root data directory (e.g., ~/data/IrrigationGIS/soils)
        source : DataSource
            Source configuration.
        """
        self.data_root = os.path.expanduser(data_root)
        self.source = source

    @property
    def ee_extracts_dir(self) -> str:
        """Directory containing raw EE CSV extracts."""
        return os.path.join(self.data_root, 'swapstress', 'extracts', self.source.ee_extracts_subdir)

    @property
    def ee_table(self) -> str:
        """Path to concatenated EE features parquet."""
        return os.path.join(self.data_root, 'swapstress', 'training', self.source.ee_table_filename)

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
            return os.path.join(self.data_root, 'soil_potential_obs', 'preprocessed',
                                self.source.preprocessed_subdir)
        return None

    @property
    def fit_results_dir(self) -> Optional[str]:
        """Directory containing fitted VG parameter JSONs."""
        if self.source.fit_results_subdir:
            return os.path.join(self.data_root, 'soil_potential_obs', 'curve_fits',
                                self.source.fit_results_subdir)
        return None

    @property
    def embeddings_dir(self) -> Optional[str]:
        """Directory containing embedding parquet files."""
        if self.source.embeddings_subdir:
            # Embeddings are on a different mount
            return os.path.join('/data/ssd2/swapstress/vwc/embeddings', self.source.embeddings_subdir)
        return None


if __name__ == '__main__':
    list_sources()
    print()

    # Example usage
    gshp = get_source('gshp')
    paths = DataPaths('~/data/IrrigationGIS/soils', gshp)
    print(f"GSHP EE table: {paths.ee_table}")
    print(f"GSHP labels: {paths.labels_file}")
    print(f"GSHP fit results: {paths.fit_results_dir}")
