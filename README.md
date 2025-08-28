# SWAP-Stress

Code for machine learning to infer soil hydraulic parameters at scale.

## Overview

This repository contains the Python code used for the SWAP-Stress project. The primary goal is to improve estimates of soil van Genuchten parameters, which are crucial for modeling soil water retention curves (SWRCs). The project implements a machine learning pipeline, including a transfer learning approach, to refine large-scale estimates from the Rosetta model with point-scale laboratory measurements.

## Repository Structure

The repository is organized into two main components:

- `map/`: Contains modules for large-scale data processing and machine learning.
  - `data/`: Scripts for acquiring and processing geospatial data, primarily using Google Earth Engine (GEE). This includes handling the Rosetta soil parameter estimates.
  - `learning/`: The core machine learning pipeline. It includes scripts for training neural networks (PyTorch) and Random Forest models, as well as a dedicated workflow for transfer learning.

- `retention_curve/`: Contains modules related to soil water retention curves and processing of laboratory/station data.
  - `mt_mesonet.py`: Script for fetching and processing soil core data from the Montana Mesonet.
  - `swrc.py`: Core logic for handling SWRC data and calculations.
  - `plot_swrc_summaries.py`: Utilities for visualizing SWRC data.

## Core Workflows

### Data Processing

1.  **Rosetta Data:** Gridded Rosetta v3.1 estimates of van Genuchten parameters (`ALPHA`, `N`, `THETAS`, `THETAR`) are acquired using GEE scripts in `map/data/`.
2.  **Station Data:** Laboratory measurements of soil properties are sourced from various networks, with specific processing implemented for the Montana Mesonet in `retention_curve/mt_mesonet.py`.

### Machine Learning

The primary workflow involves a transfer learning approach:

1.  **Base Model Training:** A base model (either a neural network or Random Forest) is trained on the extensive Rosetta dataset. See `map/learning/train_nn.py` and `map/learning/train_rf.py`.
2.  **Fine-Tuning:** The pre-trained base model is then fine-tuned using the smaller laboratory dataset. The core logic for this is in `map/learning/transfer_learning.py`.
3.  **Inference:** The final model can be used to predict improved soil parameters.

This approach leverages the broad spatial coverage of the Rosetta model while incorporating the high-fidelity information from laboratory measurements in an attempt to produce more accurate soil hydraulic property estimates.

## How To Run

### Prerequisites

- Python environment with: PyTorch (+ Lightning), `rtdl`, scikit-learn, pandas, numpy, geopandas, rasterio, shapely, tqdm, lmfit, seaborn, matplotlib, earthengine-api.
- Google Earth Engine access (`earthengine-api`) and `ee.Initialize(...)` configured for your project.
- Local data directories used by the scripts (update hardcoded paths in `__main__` blocks as needed under `~/data/IrrigationGIS/...`).

### 1) Extract Rosetta parameters at points

- Use `map/data/rosetta_geotiff.py` to sample Rosetta GeoTIFFs at station or training points.
- Adjust `points_shp_`, `rosetta_dir_`, and `output_csv_` in the `__main__` block.
- Output: Parquet with Rosetta VG params joined to points.

### 2) Build geospatial feature stack from Earth Engine and export

- `map/data/ee_export.py` calls `stack_bands_climatology(...)` (in `map/data/call_ee.py`) and samples values at point locations, exporting CSVs to Google Cloud Storage.
- Set `bucket`, `file_prefix`, MGRS shapefile, points shapefile, and `index_col`.
- Ensure `is_authorized()` succeeds for your EE project.

### 3) Concatenate EE CSVs and join Rosetta to produce training table

- Run `map/data/ee_tables.py` to concatenate exported CSVs and join the Rosetta parquet by the index column.
- Optionally emit categorical mappings JSON for later use by embedding models.
- Output: Training parquet (e.g., `training_data.parquet`).

### 4) Train models (NNs and RF)

- Neural nets: `map/learning/train_nn.py`
  - Configure `root` to point to your training data and set output checkpoint/metrics dirs.
  - Choose `mode` (`single` or `combined`) and target vertical levels.
  - Models: `MLP` (one-hot), `MLPEmbeddings` (categorical embeddings), `FTTransformer` (rtdl baseline).
- Random Forest: `map/learning/train_rf.py`
  - Produces JSON metrics for single or combined targets.

### 5) Inference on empirical stations

- `map/learning/inference.py` loads the best checkpoints (by filename `val_r2`), aligns scalers/encodings, and predicts VG parameters for empirical sites.
- Inputs: empirical features parquet, training parquet (for stats/columns), categorical mappings JSON, checkpoints directory.
- Output: Parquet with predicted `US_R3H3_L{level}_VG_{param}` columns.

### 6) Empirical SWRC preprocessing and fitting (Mesonet example)

- Preprocess and split station data: `retention_curve/mt_mesonet.py` (merges SWP + metadata; writes per-station Parquet, CSV + GeoJSON summaries).
- Fit van Genuchten parameters per-depth per-station: `retention_curve/swrc.py` (generates JSON fit results and optional plots).

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
