# SWAP-Stress

Code for the manuscript "Improving regional-scale soil hydraulic property estimates by transfer-learning machine learning models".

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
2.  **Fine-Tuning:** The pre-trained base model is then fine-tuned using the smaller, but more accurate, laboratory dataset. The core logic for this is in `map/learning/transfer_learning.py`.
3.  **Inference:** The final model can be used to predict improved soil parameters.

This approach leverages the broad spatial coverage of the Rosetta model while incorporating the high-fidelity information from laboratory measurements to produce more accurate soil hydraulic property estimates.