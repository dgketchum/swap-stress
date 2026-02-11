# SWAP-Stress

ML pipeline for estimating soil suction from multi-source soil water data and
remote sensing covariates.

## Overview

SWAP-Stress is currently centered on a **direct suction model**:

- Build a unified **observation-level** table where each row is one measurement:
  - inputs: Earth Engine covariates + `theta` (+ optional embeddings)
  - target: `log10_suction_cm = log10(suction_cm)`
- Train/evaluate a Random Forest with **location-held-out** splitting to avoid
  spatial leakage from shared EE pixels.

The detailed end-to-end workflow is documented in `notes/PIPELINE.md`.

Key leakage/generalization notes:
- `notes/LEAKAGE.md`
- `notes/SPLITTING.md`
- `notes/PIPELINE_REVIEW.md`

## Repository structure (current)

- `map/data/`
  - `call_ee.py`: Earth Engine covariate stack definition.
  - `ee_export.py`: Export EE covariates for point inventories (tile-by-tile).
  - `ee_tables.py`: Concatenate EE exports and write per-source EE parquets.
  - `build_training_table.py`: Build the unified observation-level training table.
  - `features.py`: Feature grouping and selection utilities.
  - `amsr_extract.py`: Optional AMSR VOD climatology extraction and merge support.
- `map/learning/decision_tree/`
  - `train_direct.py`: Direct RF training/evaluation (spatial-group holdout).
  - `feature_importance.py`: Permutation importance + group ablation.
- `retention_curve/standardize_swp.py`: Standardize source observations to `theta`, `suction_cm`, `depth_cm`.
- `utils/`: Source-specific helpers (NCSS conversions, GSHP/ReESH shapefile helpers, etc.).
- `vwc_series/` and `map/learning/mae/`: Optional VWC/gridMET time-series and MAE embeddings.

## Quickstart (direct model)

This assumes you already have:
- per-source EE feature tables (parquets) in the locations expected by `map/data/source_registry.py`, and
- standardized observation CSVs for each source (see `retention_curve/standardize_swp.py`).

1) Build unified observation table:
```bash
uv run python -m map.data.build_training_table
```

2) Train/evaluate the direct RF:
```bash
uv run python -m map.learning.decision_tree.train_direct --obs-table /path/to/obs_level_training_250m.parquet --output-dir /path/to/output_dir
```

3) Run feature importance / ablations:
```bash
uv run python -m map.learning.decision_tree.feature_importance --obs-table /path/to/obs_level_training_250m.parquet --output-dir /path/to/importance_dir
```

See `notes/PIPELINE.md` for the full extraction + build steps and artifact paths.

## Notes

- Earth Engine exports are asynchronous and quota-limited; see `map/data/ee_export.py`.
- The direct model requires `theta` at inference time; evaluation assumes `theta` is available.
- For leakage prevention, use the spatial-group splitting described in `notes/SPLITTING.md`.
