# SWAP-Stress

Code for machine learning to infer soil hydraulic parameters at scale.

## Overview

SWAP-Stress provides code and workflows to estimate soil hydraulic parameters (van Genuchten-Mualem) at scale. The project combines large-scale, gridded priors (Rosetta) with higher-fidelity empirical information (lab/station SWRCs) using a machine-learning pipeline. The pipeline includes:

- Pretraining models on features sampled from Google Earth Engine with Rosetta parameters as labels.
- Deriving empirical labels by fitting van Genuchten parameters to standardized SWRC observations (Mesonet, GSHP, ReESH).
- Optional fine-tuning/transfer learning on empirical data to adapt pretrained models.
- Inference to generate improved parameter estimates for sites or stations of interest.

## Repository Structure

- `map/`: Geospatial data extraction/assembly and ML training/inference
  - `map/data/`
    - `call_ee.py`: Builds the Earth Engine feature stack (Landsat, climate, SAR, soils, terrain, etc.).
    - `ee_export.py`: Samples the EE stack at points (by MGRS tile) and exports CSVs to Cloud Storage.
    - `ee_tables.py`: Concatenates exported CSVs, optionally joins Rosetta labels, writes Parquet, and saves categorical mappings JSON.
    - `rosetta_geotiff.py`: Samples Rosetta GeoTIFFs at point locations to produce VG-parameter tables.
    - `station_training_table.py`: Joins fitted station parameters with EE features to create station training tables.
    - `gshp_training_table.py`: Joins GSHP labels with EE features for GSHP-specific training tables.
  - `map/learning/`
    - `train_tabular_nn.py`: Trains tabular NNs (Vanilla MLP, MLP+embeddings, FT-Transformer via `rtdl`).
    - `train_decision_trees.py`: Random Forest baselines (Rosetta, GSHP, Stations).
    - `transfer_learning.py`: Fine-tuning utilities for adapting pretrained models to empirical labels.
    - `inference_nn.py`: Inference using NN checkpoints (auto-selects best by `val_r2` in filename).
    - `inference_dt.py`: Inference using saved scikit-learn models (RF) and their feature lists.

- `retention_curve/`: SWRC preprocessing, fitting, and summaries
  - `standardize_swp.py`: Standardizes disparate inputs (Mesonet, GSHP, Rosetta, ReESH) to columns: `suction`, `theta`, `depth` (+ optional metadata).
  - `swrc.py`: Core SWRC class to fit van Genuchten parameters per depth with `lmfit` and plot/save results.
  - `gshp_swrc.py`: GSHP-style fitting policy and helpers.
  - `fit_swrc.py`: Batch-fit runner for standardized CSV directories → JSON fit summaries per site.
  - `plot_swrc_summaries.py`: Summary plots from empirical fits.

- `viz/`: Comparison and diagnostic plotting (e.g., empirical vs. Rosetta vs. inferred).
- `vwc_series/`: Time-series helpers (e.g., `mt_mesonet.py` for Mesonet VWC processing).
- `utils/`: Utility helpers (e.g., GSHP preprocessing, identifiers).

R environment note
- Some utilities (e.g., GSHP wrapper via `utils/ncss.py` → `run_fit_new_samples`) call external R scripts. You must use an R-enabled environment with the required R packages. This R setup is not part of the Python project requirements. Ensure `Rscript` on your PATH belongs to that environment, or place local `soilhypfit` R sources under `~/PycharmProjects/GSHP-database/soilhypfit/R/`.

Notes
- The README previously referenced `retention_curve/mt_mesonet.py`; Mesonet-specific time-series code now lives under `vwc_series/mt_mesonet.py`.
- Bayesian fitting in `swrc.py` is scaffolded but currently disabled (imports and `fit_bayesian` are commented out). Use deterministic optimizers supported by `lmfit` (e.g., `nelder`, `least_squares`, `slsqp`).

## Core Workflows

### A) Geospatial Features and Rosetta Labels

- Extract Rosetta parameters at points: `map/data/rosetta_geotiff.py`
  - Samples Rosetta GeoTIFF stacks at point locations; writes a Parquet with columns like `US_R3H3_L{level}_VG_{param}`.
- Build EE feature stack and export per-tile CSVs: `map/data/ee_export.py` (uses `map/data/call_ee.py`)
  - Sets up a multi-sensor, multi-season feature stack and samples at point locations (MGRS tile-by-tile) to Cloud Storage.
- Concatenate CSVs and join Rosetta: `map/data/ee_tables.py`
  - Globs exported CSVs, removes empties, joins Rosetta parquet by index, emits a training Parquet and optional categorical mappings JSON.

### B) Empirical SWRC Labels (Stations/Lab)

- Standardize raw observations to SWRC tables: `retention_curve/standardize_swp.py`
  - Helpers for Mesonet, GSHP, ReESH; outputs standardized CSVs with `suction`, `theta`, `depth` (+ optional metadata).
- Batch-fit van Genuchten parameters per station/depth: `retention_curve/fit_swrc.py`
  - Runs `SWRC`/`GshpSWRC` fits using `lmfit` methods (e.g., `nelder`, `least_squares`, `slsqp`, `lbfgsb`); saves one JSON per station.
- Build training tables joining EE features with empirical labels:
  - Stations (Mesonet/ReESH): `map/data/station_training_table.py`
  - GSHP: `map/data/gshp_training_table.py`

### C) Train Models

- Neural networks (tabular): `map/learning/train_tabular_nn.py`
  - Models: `MLP` (one-hot), `MLPEmbeddings` (categorical embeddings), `FTTransformer` (via `rtdl`).
  - Supports combined multi-target training across `VG_PARAMS` and selecting vertical `levels`.
- Random Forest baselines: `map/learning/train_decision_trees.py`
  - Separate entry points for Rosetta, GSHP, and Stations; writes JSON metrics and optional saved models.
- Optional sequence model from VWC time series: `map/learning/train_sequence_nn.py`.

### D) Transfer Learning

- Fine-tune pretrained Rosetta models on empirical labels: `map/learning/transfer_learning.py`.

### E) Inference

- Neural networks: `map/learning/inference_nn.py`
  - Loads best checkpoints by `val_r2`, aligns scalers/encodings, predicts `US_R3H3_L{level}_VG_{param}` for stations.
- Random Forest: `map/learning/inference_dt.py`
  - Loads saved `joblib` models and feature lists; writes predictions (GSHP params) to Parquet.

### F) Comparison and Visualization

- Compare empirical fits vs. Rosetta vs. ML predictions and plot station curves: `viz/compare_station_curves.py`.
- Summaries and parameter distributions: `retention_curve/plot_swrc_summaries.py`.

## How To Run

### Prerequisites

- Python 3.9+ with the following packages (conda recommended for geospatial deps):
  - Core: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`, `joblib`, `lmfit`
  - Geospatial: `geopandas`, `shapely`, `rasterio`, `pyproj`
  - ML (NNs): `torch`, `pytorch-lightning`, `rtdl`
  - Earth Engine: `earthengine-api`
- GPU optional: training will use `cuda` if available (`map/learning/DEVICE`).
- Google Earth Engine setup:
  - Run `earthengine authenticate` or `ee.Authenticate()` once, then ensure `ee.Initialize(project='<your-project>')` works.
  - Update the hardcoded project in `map/data/call_ee.py:is_authorized()` if needed.
- Data roots: most scripts assume `~/data/IrrigationGIS/...` (see each `__main__` block). Adjust paths and boolean switches before running.

### 1) Extract Rosetta parameters at points

- Use `map/data/rosetta_geotiff.py` to sample Rosetta GeoTIFFs at station or training points.
- In the `__main__` block, set which workflow to run and edit `points_shp_`, `rosetta_dir_`, `out_parquet_`.
- Run: `python map/data/rosetta_geotiff.py`
- Output: Parquet with columns like `US_R3H3_L{level}_VG_{param}` per point.

### 2) Build geospatial feature stack from Earth Engine and export

- `map/data/ee_export.py` calls `stack_bands_climatology(...)` (in `map/data/call_ee.py`) and samples values at point locations, exporting CSVs to Google Cloud Storage.
- Set `bucket`, `file_prefix`, MGRS shapefile, points shapefile, and `index_col`.
- Ensure `is_authorized()` succeeds for your EE project. Run: `python map/data/ee_export.py` after setting the appropriate `run_*_workflow` flag and paths.
- To manage request size, shapefiles with extraction points should have an attribute 'MGRS_TILE' that describes the Military Grid Reference System tile the point falls within.

### 3) Concatenate EE CSVs and join Rosetta to produce training table

- Run `map/data/ee_tables.py` to concatenate exported CSVs and join the Rosetta parquet by the index column.
- Optionally emit categorical mappings JSON for later use by embedding models.
- Run: `python map/data/ee_tables.py` after setting flags/paths.
- Output: Training parquet (e.g., `training_data.parquet`) and optional categorical mappings JSON.

### 4) Train models (NNs, RF)

- Neural nets and FT-Transformer: `map/learning/train_tabular_nn.py`
  - Configure `root` to point to your training data and set output checkpoint/metrics dirs.
  - Choose `mode` (`single` or `combined`) and target vertical `levels`.
  - Models: `MLP` (one-hot), `MLPEmbeddings` (categorical embeddings), `FTTransformer` (requires `rtdl`).
-- Random Forest: `map/learning/train_decision_trees.py`
  - Separate entries for Rosetta, GSHP, or Stations; writes JSON metrics and can save models/features.
  
Optional sequence model from VWC time series:
- 1D CNN on VWC sequences → VG params: `map/learning/train_sequence_nn.py`

### 5) Inference on empirical stations

- Neural nets: `map/learning/inference_nn.py`
  - Finds best checkpoints by `val_r2` in filename, aligns scalers/encodings, predicts `US_R3H3_L{level}_VG_{param}`.
  - Inputs: empirical features parquet, Rosetta training parquet (for stats/columns), categorical mappings JSON, checkpoint dir.
  - Output: Parquet with predicted `US_R3H3_L{level}_VG_{param}` columns.
- Random Forest: `map/learning/inference_dt.py`
  - Loads saved `joblib` model and its feature list; writes predictions for GSHP parameters.

### 6) Empirical SWRC preprocessing and fitting

- Build raw station inputs as needed (e.g., Mesonet time series): `vwc_series/mt_mesonet.py`.
- Standardize to SWRC tables: `retention_curve/standardize_swp.py` helpers produce CSVs with `suction`, `theta`, `depth`.
- Fit per-depth per-station and write JSON summaries: `retention_curve/fit_swrc.py` (choose `method`: `nelder`, `least_squares`, `slsqp`, `lbfgsb`).

### 7) Compare empirical vs Rosetta vs ML predictions

- `retention_curve/swrc_comparison.py` merges empirical fit JSONs with Rosetta parameters and ML predictions, computes R²/RMSE, and saves scatter plots.
- Configure paths to empirical results, Rosetta parquet, pretrained/finetuned prediction parquet(s), and optional finetuning split JSON.

### 8) Summaries and visualization

- `retention_curve/plot_swrc_summaries.py` builds parameter-by-depth boxplots, overall histograms, and parameter-influence plots from empirical fits.
- `map/data/viz.js` is an Earth Engine Code Editor helper to visualize key layers used by the stack function.

### Notes

- VG parameters consistently used across modules: `theta_r`, `theta_s`, `log10_alpha`, `log10_n`, `log10_Ks`.
- Rosetta vertical levels L1–L7 and empirical-to-Rosetta depth mapping are defined in `retention_curve/__init__.py`.
- Many scripts have `__main__` sections with example paths—update them to your environment before running.
- Most scripts are configured via booleans in `__main__` (e.g., `run_*_workflow`) and expect paths under `~/data/IrrigationGIS/...`. Review and edit these before invoking with `python <script>.py`.

## Caveats and Known Limitations

- Earth Engine exports: Requires authentication and quotas apply. Exports are asynchronous; `ee_export.py` supports tile-wise runs and a `check_dir` to skip completed tiles. Use `split_tiles=True` for large MGRS tiles.
- Path assumptions: Many scripts assume a base directory like `~/data/IrrigationGIS/...` and toggle workflows via booleans in `__main__`. Edit paths and flags before running.
- SWRC fitting: Bayesian path in `swrc.py` is scaffolded but disabled. Use `lmfit` optimizers such as `nelder`, `least_squares`, `slsqp`, or `lbfgsb`. Ensure inputs are in centimeters: suction (cm), depth (cm), theta (fraction).
- Checkpoint naming: `inference_nn.py` auto-selects best checkpoints by `val_r2` embedded in filenames. If you change filename patterns, update the selection logic.
- Categorical mappings: For embedding models, keep the `categorical_mappings.json` consistent between training and inference. One-hot paths (`MLP`) must align column sets between train and inference.
- Sentinel handling: Some data-prep code treats values `<= -9999` as missing. Review these filters before large runs to avoid unintended row drops.
- Station IDs: RF station training attempts a group-aware split by `station`; ensure the station identifier is present and consistent if you use that workflow.
- Performance: GeoTIFF extraction and EE exports can be memory/IO intensive. Adjust worker counts and tile scales thoughtfully; GPU is optional but speeds up NN training.
- Reproducibility: Seeds are not globally enforced across all scripts (e.g., Lightning). Set seeds and environment variables if deterministic runs are required.
- CRS alignment: Ensure point data match expected CRS for sampling (e.g., `rosetta_geotiff.py` reprojects to `ROSETTA_CRS` internally) and that MGRS shapefiles align with station coordinates.

## Example: MT Mesonet SWRC Processing

Standardize MT Mesonet observations and fit van Genuchten parameters.

- Prepare MT Mesonet time series as needed: `vwc_series/mt_mesonet.py` (optional helpers).
- Programmatic example (succinct):

```
from retention_curve.standardize_swp import write_standardized_mt_mesonet
from retention_curve.fit_swrc import fit_standardized_dir
import os

home = os.path.expanduser('~')
root = os.path.join(home, 'data', 'IrrigationGIS', 'soils', 'soil_potential_obs')

swp_csv = os.path.join(root, 'mt_mesonet', 'swp.csv')
meta_csv = os.path.join(root, 'mt_mesonet', 'station_metadata.csv')
std_dir = os.path.join(root, 'preprocessed', 'mt_mesonet')

write_standardized_mt_mesonet(swp_csv, meta_csv, std_dir, profile_key='station')

fit_out = os.path.join(root, 'curve_fits', 'mt_mesonet', 'nelder')
fit_standardized_dir(std_dir, fit_out, method='nelder')
```

- Or use the built-in mains:
  - In `retention_curve/standardize_swp.py`, set `run_mt_mesonet = True` and edit the three paths (`swp_csv_`, `metadata_csv_`, `out_dir_`).
  - Run: `python retention_curve/standardize_swp.py`
  - In `retention_curve/fit_swrc.py`, set `run_mt_mesonet = True`, pick `method` (e.g., `nelder`), and set `in_dir_`/`out_dir_`.
  - Run: `python retention_curve/fit_swrc.py`

Outputs
- One CSV per station in `preprocessed/mt_mesonet` with `suction_cm`, `theta`, `depth_cm`, `name`.
- One JSON per station under `curve_fits/mt_mesonet/<method>` containing fitted parameters and a data snapshot.

Example code

```python
import os
from retention_curve.standardize_swp import write_standardized_mt_mesonet
from retention_curve.fit_swrc import fit_standardized_dir

home = os.path.expanduser('~')
root = os.path.join(home, 'data', 'IrrigationGIS', 'soils', 'soil_potential_obs')

swp_csv = os.path.join(root, 'mt_mesonet', 'swp.csv')
meta_csv = os.path.join(root, 'mt_mesonet', 'station_metadata.csv')
std_dir = os.path.join(root, 'preprocessed', 'mt_mesonet')

write_standardized_mt_mesonet(swp_csv, meta_csv, std_dir, profile_key='station')

fit_out = os.path.join(root, 'curve_fits', 'mt_mesonet', 'nelder')
fit_standardized_dir(std_dir, fit_out, method='nelder')
```

Notes
- Ensure units: suction in centimeters, theta as volumetric fraction, depth in centimeters.
