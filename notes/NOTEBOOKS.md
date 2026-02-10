# Notebook Plan (SWAP-Stress Onboarding)

Goal: break the repo’s main functionality into a small set of **onboarding notebooks** so collaborators can understand:
- What the major datasets are
- How each source is standardized (especially **units** and **conversions**)
- What the major Earth Engine covariates are and how we sample/visualize them
- How we assemble the unified training table and summarize distributions + spatial coverage

These notebooks should **expose existing code** (imports from this repo) and therefore reflect the current project state.

---

## Repository Anchors (Single Sources of Truth)

Use these modules directly from notebooks:

- Source standardization + unit conversions:
  - `retention_curve/standardize_swp.py` (ReESH, MT Mesonet, GSHP; physical filters)
  - `utils/ncss.py` (NCSS bar→cm conversion + gravimetric→volumetric conversion)
- Data source definitions + canonical path conventions:
  - `map/data/source_registry.py` (`SOURCES`, `DataSource`, `DataPaths`)
- Earth Engine feature stack:
  - `map/data/call_ee.py` (`stack_bands_climatology`, `is_authorized`)
  - `map/data/ee_export.py` (tile export pattern; use carefully due to quotas)
- Feature naming/labeling/grouping:
  - `map/data/ee_feature_list.py` (human-readable labels for feature columns)
  - `map/data/features.py` (feature groups; `get_feature_columns`, `classify_feature`)
- Unified observation-level training table builder (obs + EE features):
  - `map/data/build_training_table.py` (`build_unified_table`)
- Existing mapping utility (global + CONUS):
  - `poster/training_data_map.py` (`plot_training_data_map`)

External notebook example (EE thumbnails into notebooks):
- `/home/dgketchum/code/openet-ptjpl/examples/single_image.ipynb`

Important note on current project state:
- **Sentinel-2 is not currently in the project EE feature stack** (see `map/data/call_ee.py`). If we show Sentinel-2 imagery, it must be clearly labeled as a visualization-only example unless/until it is added to the stack.

---

## Notebook Directory Layout

Add a new directory:
- `notebooks/`

Create three notebooks:
1. `notebooks/source_data.ipynb`
2. `notebooks/earth_engine.ipynb`
3. `notebooks/training_table.ipynb`

All notebooks write figures to:
- `notebooks/_outputs/`

---

## Shared Notebook Conventions

### Configuration (first cell in every notebook)

- Use an overridable data root:
  - `DATA_ROOT = os.environ.get("SWAPSTRESS_DATA_ROOT", os.path.expanduser("~/data/IrrigationGIS/soils"))`
- Use explicit resolution parameter (current defaults are typically 250 m):
  - `RESOLUTION_M = 250`
- Use a consistent output dir:
  - `OUT_DIR = Path("notebooks/_outputs")`

### Guard heavy/quota-limited steps

Default to “load existing artifacts / small demos” and gate expensive operations behind booleans:
- `RUN_STANDARDIZE = False`
- `RUN_EE = False` (Earth Engine API calls)
- `RUN_EE_EXPORT = False` (exports are asynchronous + quota-limited)
- `RUN_BUILD_TRAINING_TABLE = False` (IO-heavy)

### Prefer imports over re-implementation

Notebook code should call repo functions rather than duplicating logic so notebooks track the codebase.

---

## Notebook 1: `source_data.ipynb`

Purpose: introduce the **major input datasets**, their **source units**, our **conversions** to standardized units, and show **distributions** + **basic maps**.

### Sections / cells

1) **Data sources overview (registry-driven)**
- Load `map.data.source_registry.SOURCES` and display a small table:
  - `name`, `description`, `index_col`, `preprocessed_dir`, `fit_results_dir`, `ee_table`

2) **Unit conversions and physical filters**
- Present the standard schema:
  - `suction_cm` (cm H₂O), `theta` (0–1), `depth_cm` (cm)
- Pull conversion logic from:
  - ReESH MPa→cm and Mesonet kPa→cm in `retention_curve/standardize_swp.py`
  - NCSS bar→cm + gravimetric→volumetric conversion in `utils/ncss.py`
- Show a markdown table summarizing:
  - source → original suction units → conversion factor → key filters applied

3) **“Toy” standardization demos (optional; file-exists guarded)**
- For each source, load a small raw sample (if available) and run:
  - `standardize_reesh`, `standardize_mt_mesonet`, `standardize_gshp`
  - `ncss_to_standardized`
- Show output schema + head + summary stats.

4) **Preprocessed inventory + histograms**
- If `DATA_ROOT/soil_potential_obs/preprocessed/<source>` exists:
  - sample N CSVs
  - plot `theta`, `depth_cm`, `suction_cm`, and `log10(suction_cm)` distributions
  - show per-source min/max/quantiles

5) **Site maps (quicklook)**
- If project shapefiles exist (see `poster/training_data_map.py`), render:
  - global scatter
  - CONUS scatter
- Fallback: if only tabular lat/lon exist, scatter those.

---

## Notebook 2: `earth_engine.ipynb`

Purpose: document the **spatial covariates**, show the **EE stack composition**, and provide a **toy example** around a ReESH site (smallest source repo), specifically `US-CDM`, including a ~4 km neighborhood view.

### Sections / cells

1) **EE initialization (guarded)**
- Prominent warning: EE calls use quotas.
- Default `RUN_EE = False`.
- When enabled, call `map.data.call_ee.is_authorized()` or run `ee.Initialize(...)`.

2) **Select a ReESH station: `US-CDM`**
- Load ReESH site points from the same shapefile used elsewhere:
  - `DATA_ROOT/soil_potential_obs/reesh/shapefile/reesh_sites_mgrs.shp` (or `*_5070.shp` depending on which exists)
- Pick `site_id == "US-CDM"` and extract lat/lon.
- Explicitly note any ID normalization conventions used downstream (hyphen/underscore), consistent with `map/data/build_training_table.py`.

3) **Define ROI: 4 km buffer**
- `roi = ee.Geometry.Point(lon, lat).buffer(4000)`
- Optionally show the bounding box coordinates and note scale vs display resolution.

4) **Build the project stack and list bands**
- Call `stack_bands_climatology(roi, region=...)` from `map/data/call_ee.py`.
- Print:
  - number of bands
  - a labeled subset using `map.data.ee_feature_list.label_feature`

5) **Neighborhood visualization helpers**
- Thumbnail display in-notebook (pattern from `/home/dgketchum/code/openet-ptjpl/examples/single_image.ipynb`):
  - `img.visualize(...).getThumbURL(...)` → download → display
- Gridded neighborhood plot:
  - Use an EE method to fetch an array (e.g., `sampleRectangle` at `scale=RESOLUTION_M`) and plot with `matplotlib`

6) **Demonstration layers (from current stack)**
- Landsat composite index (e.g., `nd_mean_gs`)
- Sentinel-1 (`VV_mean` or `VH_mean`)
- SMAP L4 (e.g., `sm_profile_mean`)
- Terrain (elevation/slope)
- SoilGrids example (clay)

7) **Optional: Sentinel-2 “quicklook”**
- Only if desired for outreach, add a clearly-marked cell that loads Sentinel-2 imagery for visualization.
- Explicitly label: “Not part of the current training covariate stack unless added to `map/data/call_ee.py`.”

---

## Notebook 3: `training_table.ipynb`

Purpose: show how the repo joins **observations + EE features** into a unified table, and provide distributions + spatial coverage at CONUS and global scales.

### Sections / cells

1) **Load or build the unified observation-level table**
- Prefer loading an existing parquet (fast demo).
- If building (guarded):
  - call `build_unified_table(...)` from `map/data/build_training_table.py`
  - sources likely include: `["gshp", "ncss", "mt_mesonet", "reesh"]`

2) **Schema + join health**
- Report:
  - total observations, unique `sample_id`, per-source counts
  - per-`rosetta_level` counts
- Diagnose missing-feature joins (e.g., how many rows have many NaNs).
- Call out ReESH ID normalization (site portion extraction), consistent with `map/data/build_training_table.py`.

3) **Core distributions**
- Plot by source:
  - `theta`
  - `log10_suction_cm` (target)
  - `depth_cm` and/or `rosetta_level`

4) **Covariate distributions by feature groups**
- Use `map.data.features.get_feature_columns()` + `classify_feature()` to:
  - count features by group (landsat, sentinel1, smap, gridmet, soilgrids, fao, polaris, terrain, embeddings)
  - plot a small set of representative feature histograms
- Explicitly note sentinel handling (e.g., `-9999`) where relevant to downstream models.

5) **Maps: Global + CONUS**
- Preferred: call `poster.training_data_map.plot_training_data_map(...)` if shapefiles exist.
- Fallback: scatter from available `lat`/`lon` columns in the table.

---

## Maintenance Expectations (Avoid Notebook Rot)

These notebooks are part of the project surface area and must be maintained.

When changing schemas, paths, join keys, or feature stack composition, explicitly check notebook impact, especially for changes in:
- `retention_curve/standardize_swp.py`
- `utils/ncss.py`
- `map/data/source_registry.py`
- `map/data/call_ee.py`
- `map/data/ee_export.py`
- `map/data/ee_tables.py`
- `map/data/build_training_table.py`

Minimum maintenance standard after relevant changes:
- Run each notebook through the “load + summarize + plot” sections (EE calls optional).

