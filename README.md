# swap-stress

SWAP-STRESS (Soil Water Hydraulic Parameters from Stressors) is a project to predict soil water retention curve parameters using machine learning.

This repository contains Python scripts to:
- Assemble geospatial training data from Google Earth Engine, including satellite imagery (Landsat, Sentinel-1), climate data (GridMET, PRISM), and soil properties (SSURGO, POLARIS).
- Train machine learning models (Random Forest, PyTorch) to predict van Genuchten soil hydraulic parameters.
- Fit soil water retention curve data to the van Genuchten model.
