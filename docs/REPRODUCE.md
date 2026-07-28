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
| 07 | `swapstress-package` | Writes the released product: dual-band MPa and log10(cm), CF attributes | stage 06 | **not yet implemented** |
| 08 | `swapstress-figures` | Renders the descriptor's figures | stages 03–06 | minutes |

No stage needs a GPU. Stage 03's `--n-jobs` and stage 05's `--n-jobs` are the
two knobs that matter for wall time.

Stage 07 arrives with the product writer; `reproduce.sh` names it and skips it
rather than pretending the chain is shorter than it is.

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
| `FIG_DIR` | `figs/descriptor` |
| `TRAIN_CONFIG` | `configs/train_9km_global_pruned.toml` |
| `PREDICT_CONFIG` | `configs/predict_9km_global_pruned.toml` |
| `GAPFILL_CONFIG` | `configs/gapfill_9km_global_pruned.toml` |
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

## Running one analysis or one figure

Stages 04 and 08 are drivers over a set:

```bash
uv run swapstress-validate --model-dir <dir> --analysis loso regional
uv run swapstress-figures --figure koppen
```

`--analysis all` covers everything except `ptf-baseline`, which has its own
prep/eval subcommands and reads an external Rosetta grid. Extra arguments after
a single `--analysis` or `--figure` are forwarded to that module.
