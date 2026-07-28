# Reproducing a SWAP-Stress release

The pipeline is nine stages, one console command each. `reproduce.sh` chains
them; each stage is also runnable on its own.

```bash
uv sync --all-extras          # installs the package and its stage commands
./reproduce.sh --dry-run      # resolve every path and config, run nothing
./reproduce.sh --from 03 --to 06
```

`uv run swapstress` lists the stages. Every stage takes `--config <file.toml>`,
accepts CLI overrides of the same names, understands `--dry-run`, and writes a
`provenance.json` (or `provenance_<stage>.json` where several stages share an
output directory) recording the merged config, the software version, and the git
commit it ran from.

## The stages

| # | Command | What it does | Needs | Rough runtime |
|---|---|---|---|---|
| 00 | `swapstress-standardize` | Harmonizes each source's raw files to `(theta, suction_cm, depth_cm)` | raw source data | minutes |
| 01 | `swapstress-extract` | Samples the covariate stack at every site, then folds the exports into per-source feature tables | Earth Engine credentials, a writable GCS bucket | hours, mostly waiting on EE |
| 02 | `swapstress-build-table` | Joins features to observations into the observation-level training table | stages 00–01 | minutes |
| 03 | `swapstress-train` | Fits the quantile random forest, `features + theta -> log10(suction_cm)` | stage 02, many cores | tens of minutes |
| 04 | `swapstress-validate` | Blocked CV, per-source skill, conditional bias, the PTF baseline | stage 03 | tens of minutes |
| 05 | `swapstress-predict` | Applies the model to the gridded stack, day by day | stage 03, the EASE-Grid2 static rasters, daily SMAP L3 | hours to days over a full record |
| 06 | `swapstress-gapfill` | Interpolates the retrieval gaps along the time axis | stage 05 | hours |
| 07 | `swapstress-package` | Writes the released files: the model output in three representations, plus uncertainty and gap-fill flag | stages 05–06 | minutes to an hour |
| 08 | `swapstress-figures` | Renders the descriptor's figures | stages 03–06 | minutes |

No stage needs a GPU. Stage 03's `--n-jobs` and stage 05's `--n-jobs` are the
two knobs that matter for wall time.

## What `--dry-run` checks

Each stage resolves every input and output path from the config and the source
registry, prints them, and marks which inputs are present:

```
02 build-table
  in   gshp features          [ok     ] /nas/soils/swapstress/training/gshp_ee_data_9km_global.parquet
  in   gshp observations      [ok     ] /nas/soils/soil_potential_obs/preprocessed/gshp
  out  training table                  /nas/soils/swapstress/training/obs_level_training_9km_global.parquet
```

A `MISSING` input in a full-chain dry run is not necessarily a fault: a stage's
inputs are an earlier stage's outputs, so a clean checkout reports most of them
missing until the chain has run. It is a fault when the missing file is a raw
input to stage 00, or when you are running one stage against outputs that are
supposed to already exist.

## Configuration

Run configs live in `configs/`. A stage's TOML keys are its CLI flag names with
underscores, so `--obs-table` is `obs_table`. CLI flags override the file.

`reproduce.sh` reads these from the environment, so a different release is a
matter of exporting a few variables rather than editing the script:

| Variable | Default |
|---|---|
| `RELEASE` | `global_pruned_refresh_20260520` |
| `DATA_ROOT` | `/nas/soils` |
| `MODEL_DIR` | `$DATA_ROOT/swapstress/models/direct_rf_9km_global_pruned` |
| `RELEASE_DIR` | `$DATA_ROOT/swapstress/releases/$RELEASE` |
| `INFERENCE_DIR` | `$RELEASE_DIR/inference` — Level 1 |
| `GAPFILL_DIR` | `$RELEASE_DIR/gapfill` — Level 2 |
| `PRODUCT_DIR` | `$RELEASE_DIR/product` |
| `FIG_DIR` | `figs/descriptor` |
| `TRAIN_CONFIG` | `configs/train_9km_global_pruned.toml` |
| `PREDICT_CONFIG` | `configs/predict_9km_global_pruned.toml` |
| `GAPFILL_CONFIG` | `configs/gapfill_9km_global_pruned.toml` |
| `CONTAINER` | `netcdf` — stage 07's output form; `geotiff` for per-day files |
| `RUNNER` | `uv run` — set to empty if the console scripts are on `PATH` |

## Where the paths come from

`swapstress/sources/registry.py` is the one place that knows a source's layout:
its raw inputs, its site shapefile, its MGRS index, its Earth Engine export
prefix, and where its standardized observations and feature table land. Adding a
source is a registry entry plus one standardizer; no stage hardcodes a path.

## Stage 01 is two steps

The Earth Engine export starts one batch task per MGRS tile writing CSVs to
Cloud Storage. Those have to be synced down before they can be folded into
feature tables, and the wait is manual, so the stage splits:

```bash
uv run swapstress-extract --step export     # submits the EE tasks
# ... wait, then sync gs://<bucket>/swapstress/... to
#     $DATA_ROOT/swapstress/inference/global_features/<source>/
uv run swapstress-extract --step tables     # builds the parquets
```

`--step all` does both back to back, which is only useful when the exports have
already landed.

## What stage 07 releases

The model predicts one quantity, `log10_suction_cm`. Stage 07 writes it in the
three representations Table 2 documents, plus the two conditional bands:

| Band | Sign | Units | Derivation |
|---|---|---|---|
| `log10_suction_cm` | positive, ~0–6 | log10(cm H₂O) | the model's own output, unchanged |
| `matric_potential_MPa` | **negative** | MPa | `−10**log10_suction_cm / 10197.16` |
| `suction_cm` | positive | cm H₂O | `10**log10_suction_cm`; omitted by `--drop-linear-suction` |
| `uncertainty` | — | log10 units | QRF interval width; present only if the model was run with quantiles |
| `gapfill_flag` | — | 1 | Level 2 only; 1 where the value was interpolated |

Two sign conventions coexist by design — suction head is positive and rises as
soil dries, matric potential is negative and approaches zero as soil wets. Each
band states its own convention in its metadata.

The MPa conversion is exact, not approximate: because the target is a base-10
logarithm, the change of unit is an additive shift in log space, so R², RMSE, and
interval widths in log units are identical either way. Nothing retrains and no
reported metric moves.

**Levels.** Level 1 is the direct model output with retrieval gaps left as gaps.
Level 2 is the gap-filled series. Because the gapfill stage records only a
per-file `is_gap_filled` tag, the per-pixel flag is recovered by comparing the
two levels — which is why `--level 2` also needs `--level1-dir`, and errors
rather than guessing if it is absent.

**CF `standard_name` is not set by default.** Several plausible names exist for
soil water potential, and picking one belongs to the release rather than to the
code. Pin it when the choice is made:

```bash
uv run swapstress-package --source-dir <dir> --output-dir <dir> --level 1 \
  --standard-name matric_potential_MPa=soil_water_potential
```

Every other CF attribute — `units`, `long_name`, `_FillValue`, `grid_mapping`,
`Conventions`, the grid identification, and a link back to the run's
`provenance_package.json` — is written unconditionally.

**Containers.** `build_bands()` computes the stack without touching disk, and two
writers consume the same band list. Neither changes what a band means.

| | `--container netcdf` | `--container geotiff` |
|---|---|---|
| Layout | time-stacked, one file per year | one file per day |
| Role | the deposit, archive of record | derived convenience form for GIS |
| Time | a real CF coordinate | encoded in the filename |
| Dtypes | per variable — `gapfill_flag` is `uint8` | one for all bands, so the flag is float32 |
| Compression | deflate | zstd |

NetCDF is the citable deposit. CF is native there rather than riding in TIFF
tags, a point time series is one chunked read instead of thousands of file
opens, and the file count suits a repository. It uses deflate rather than the
zstd we use internally: worse ratio, but a deposit should optimize for opening
everywhere in ten years over bytes saved.

Chunking is fixed at write time and is the one costly thing to get wrong —
`(time=1, y=all, x=all)` makes maps fast and point series pathological, and the
reverse also fails. The shipped shape is `(time=30, y=128, x=128)`, about 2 MB
uncompressed per chunk, which serves both. Each chunk dimension clamps to the
array, so a short year is not padded.

A year of the M09 grid across the released bands is tens of gigabytes, so the
writer creates the file with its full time dimension up front and writes each
day into its slice as it is read; nothing beyond one day is held in memory. If a
day fails part way through, the partial file is deleted rather than left looking
finished.

**Redundant bands.** The three representations are exact transforms of one
another, so storing all three costs 3× for no added information.
`--drop-linear-suction` omits `suction_cm`, which is the one to drop first: it
spans 0–10⁶ linearly, so it is high-entropy and compresses worst, while
`log10_suction_cm` is smooth. The log band stays regardless — it is the model's
actual output and the units `uncertainty` is expressed in. `reproduce.sh` drops
it for NetCDF and keeps it for GeoTIFF.

## Running one analysis or one figure

Stages 04 and 08 are drivers over a set:

```bash
uv run swapstress-validate --model-dir <dir> --analysis loso regional
uv run swapstress-figures --figure koppen
```

`--analysis all` covers everything except `ptf-baseline`, which has its own
prep/eval subcommands and reads an external Rosetta grid. Extra arguments after
a single `--analysis` or `--figure` are forwarded to that module.
