import os
import time
import warnings
from datetime import date
from typing import Optional

from et.ptjpl import export_ptjpl_zonal_stats

try:
    # Earth Engine is optional; defer errors to runtime
    import ee  # noqa: F401
except Exception:
    ee = None  # type: ignore

# Reuse the existing, vetted extraction helpers
from map.data.cze_extract import (
    ee_init as _ee_init,
    export_landsat_bands_over_buffers,
)


def ee_init(project: str = 'ee-dgketchum') -> None:
    """Initialize the Earth Engine client using project credentials.

    Wraps map.data.cze_extract.ee_init to keep this module standalone.
    """
    _ee_init(project=project)


def export_landsat_for_sites(
        shapefile: str,
        bucket: str,
        gcs_prefix: str,
        id_col: str,
        start_year: int = 2000,
        end_year: int = date.today().year,
        buffer_m: float = 250.0,
        check_dir: Optional[str] = None,
        debug: bool = False,
) -> None:
    """Export per-scene Landsat (C2 SR) band time series over site buffers.

    Parameters
    - shapefile: Path to point/polygon sites (WGS84 or will be reprojected).
    - bucket: Destination GCS bucket.
    - gcs_prefix: Prefix inside bucket (e.g., 'cze/flux' or 'cze/mesonet').
    - id_col: Site identifier column in shapefile (e.g., 'site_id' or 'station').
    - start_year/end_year: Year bounds (inclusive) for image collections.
    - buffer_m: Buffer radius around point sites (meters). Polygons are used as-is.
    - check_dir: Local folder to skip already-exported CSVs by filename.
    - debug: If True, samples a feature to print band keys.
    """
    export_landsat_bands_over_buffers(
        shapefile=shapefile,
        bucket=bucket,
        gcs_prefix=gcs_prefix,
        id_col=id_col,
        start_yr=start_year,
        end_yr=end_year,
        debug=debug,
        buffer_m=buffer_m,
        check_dir=check_dir,
    )


def export_openet_ptjpl_for_sites(
        shapefile: str,
        id_col: str,
        start_date: str,
        gcs_prefix: str,
        end_date: str,
        chunk: bool,
        chunk_size: int,
        bucket: str = 'wudr',
        check_dir: Optional[str] = None,
        mask_type: str = 'inv_irr',
        **kwargs,
) -> None:
    """Export PT-JPL ET fraction zonal stats to Cloud Storage using local et.ptjpl.

    Parameters
    - shapefile: Sites polygon shapefile (or points; points not supported by zonal stats).
    - id_col: Identifier field present in shapefile (feature_id in exporter).
    - start_date/end_date: ISO dates; year components are used for exports.
    - bucket: GCS bucket name (default 'wudr').
    - check_dir: Local directory for presence checks to skip existing CSV exports.
    - mask_type: One of 'irr', 'inv_irr', 'no_mask'.
    - kwargs: Forwarded to export_ptjpl_zonal_stats (e.g., polygon_asset, select, chunk, chunk_size, state_col, buffer).
    """
    # Parse years
    try:
        start_year = int(str(start_date)[:4])
        end_year = int(str(end_date)[:4])
    except Exception:
        raise ValueError('start_date and end_date must be ISO-like strings (YYYY-MM-DD)')

    # Ensure EE auth
    ee_init()

    export_ptjpl_zonal_stats(
        shapefile=shapefile,
        bucket=bucket,
        feature_id=id_col,
        start_yr=start_year,
        end_yr=end_year,
        gcs_prefix=gcs_prefix,
        check_dir=check_dir,
        mask_type=mask_type,
        chunk=chunk,
        chunk_size=chunk_size,
        **kwargs,
    )


if __name__ == '__main__':
    """Example driver toggles for Landsat and PT-JPL exports.

    Edit paths/flags to your environment before running. Follows the
    project’s pattern of boolean gate flags and user paths under ~/data.
    """

    # Gate flags
    run_reesh_flux_landsat = True
    run_mesonet_landsat = False
    run_ptjpl_flux = True

    # Common config
    home_ = os.path.expanduser('~')
    root_ = os.path.join(home_, 'data', 'IrrigationGIS')
    bucket_ = 'wudr'
    start_year_ = 2000
    end_year_ = date.today().year

    # Initialize Earth Engine once if any EE task is requested
    if run_reesh_flux_landsat or run_mesonet_landsat:
        ee_init()

    # ReESH/Flux sites: Landsat per-scene exports
    if run_reesh_flux_landsat:
        reesh_shp_ = os.path.join(root_, 'soils', 'soil_potential_obs', 'reesh', 'shapefile', 'reesh_sites_mgrs_5070.shp')
        reesh_check_dir_ = os.path.join(root_, 'soils', 'swapstress', 'cze', 'extracts', 'flux', 'landsat')
        os.makedirs(reesh_check_dir_, exist_ok=True)
        # export_landsat_for_sites(
        #     shapefile=reesh_shp_,
        #     bucket=bucket_,
        #     gcs_prefix='cze/flux',
        #     id_col='site_id',
        #     start_year=start_year_,
        #     end_year=end_year_,
        #     buffer_m=250.0,
        #     check_dir=reesh_check_dir_,
        #     debug=False,
        # )

    # OpenET PT-JPL: Landsat ET fraction zonal stats for flux sites
    if run_ptjpl_flux:
        start_ = '2000-01-01'
        end_ = '2024-12-31'
        check_dir_ = os.path.join(root_, 'soils', 'swapstress', 'et', 'flux', 'ptjpl_tables')
        os.makedirs(check_dir_, exist_ok=True)
        flux_shp_ = os.path.join(root_, 'soils', 'soil_potential_obs', 'reesh', 'shapefile',
                                 'reesh_sites_mgrs_5070.shp')
        export_openet_ptjpl_for_sites(
            shapefile=flux_shp_,
            id_col='site_id',
            start_date=start_,
            end_date=end_,
            gcs_prefix='swap/et/ptjpl',
            bucket=bucket_,
            check_dir=check_dir_,
            mask_type='inv_irr',
            buffer=250.0,
            chunk_size=10,
            chunk=True,
        )

# ========================= EOF ====================================================================
