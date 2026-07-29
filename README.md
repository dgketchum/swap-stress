# SWAP-Stress

SWAP-Stress maps **soil water potential** rather than soil water content. The
same potential means the same water availability in any soil; the same water
content means a wet sand or a dry clay depending on texture.

The conventional route from a satellite soil-moisture retrieval to potential
pushes that retrieval through a van Genuchten retention curve whose parameters
come from a pedotransfer function. That composition is poorly conditioned near
saturation, diverges at the dry end, and compounds the PTF's own error. This
project bypasses the inversion entirely and learns the relationship directly:

```
(static landscape covariates, theta) -> log10(suction in cm H2O)
```

A random forest is trained on paired water-content / matric-potential
observations from laboratory retention curves and in-situ sensor networks, each
joined to a stack of Earth Engine covariates on the 9 km EASE-Grid 2.0. At
inference the trained model is driven by daily SMAP L3 soil moisture. Fitting the
forest as a quantile regression forest (`--quantile`) leaves its mean prediction
unchanged and lets the same trees return prediction intervals. No van Genuchten
inversion appears anywhere in the chain.

Only SMAP L3 direct retrievals (SPL3SMP_E) are used as the water-content input.
L4 is deliberately excluded: it assimilates brightness temperatures into a
catchment land surface model whose soil hydraulics are themselves PTF-derived, so
using it would reintroduce exactly the assumptions this approach avoids.

This repository is the code behind that dataset. Everything on the reproduction
path lives in the `swapstress` package and is reachable from `reproduce.sh`;
preserved exploratory science lives in `research/` and is explicitly not part of
the reproduction claim.

## Install

The project uses [uv](https://docs.astral.sh/uv/) exclusively — not conda, pip,
or virtualenv.

```bash
uv sync --all-extras     # installs the package and its stage commands
uv run swapstress        # lists the pipeline stages
uv run pytest tests/     # the core test suite
```

## Reproduce a release

The pipeline is nine numbered stages, one console command each. `reproduce.sh`
chains them; each stage is also runnable on its own.

```bash
./reproduce.sh --dry-run          # resolve every path and config, run nothing
./reproduce.sh --from 03 --to 06  # run a slice of the chain
./reproduce.sh                    # run the whole thing
```

| # | Command | What it does |
|---|---|---|
| 00 | `swapstress-standardize` | Harmonize each source's raw observations to `(theta, suction_cm, depth_cm)` |
| 01 | `swapstress-extract` | Sample the covariate stack at every site, then fold the exports into per-source parquets |
| 02 | `swapstress-build-table` | Join features to observations into the observation-level training table |
| 03 | `swapstress-train` | Fit the quantile random forest, `features + theta -> log10(suction_cm)` |
| 04 | `swapstress-validate` | Blocked CV, per-source skill, conditional bias, distribution shift, sensitivity; the Rosetta/POLARIS PTF baseline on request |
| 05 | `swapstress-predict` | Apply the model to the gridded covariate stack, day by day |
| 06 | `swapstress-gapfill` | Interpolate the retrieval gaps along the time axis |
| 07 | `swapstress-package` | Write the released files: CF metadata, MPa and log10(cm) bands |
| 08 | `swapstress-figures` | Render the descriptor's figures |

Every stage takes a TOML run config (`--config`, from `configs/`), accepts CLI
overrides of the same names, understands `--dry-run`, and writes a provenance
record naming the merged config, the software version, and the git commit it ran
from. Stage 01 needs Earth Engine credentials and a writable GCS bucket; no
stage needs a GPU. `docs/REPRODUCE.md` documents each stage's inputs, outputs,
and runtime.

## The released product

The model predicts one quantity, `log10_suction_cm`. Stage 07 writes it as
**NetCDF-4/CF, time-stacked one file per year** — the citable deposit and archive
of record. Per-day GeoTIFF (`--container geotiff`) is a derived convenience form
for GIS use; it is the same band list in a different container.

| Band | Sign | Units | Derivation |
|---|---|---|---|
| `log10_suction_cm` | positive, ~0–6 | log10(cm H₂O) | the model's own output, unchanged |
| `matric_potential_MPa` | **negative** | MPa | `−10**log10_suction_cm / 10197.16` |
| `suction_cm` | positive | cm H₂O | `10**log10_suction_cm`; dropped from the NetCDF deposit |
| `uncertainty` | — | log10 units | QRF interval width; present only if the model ran with quantiles |
| `gapfill_flag` | — | 1 | Level 2 only, `uint8`; 1 where the value was interpolated |

Two sign conventions coexist by design — suction head is positive and rises as
soil dries, matric potential is negative and approaches zero as soil wets. Each
band states its own convention in its metadata. The MPa conversion is exact
rather than approximate: because the target is a base-10 logarithm, the change of
unit is an additive shift in log space, so R², RMSE, and interval widths in log
units are identical either way. Nothing retrains and no reported metric moves.
`swapstress/units.py` holds the constant and the argument.

**Levels.** Level 1 is the direct model output, valid wherever a SMAP retrieval
was available that day, with gaps left as gaps. Level 2 is the gap-filled series
and carries the per-pixel `gapfill_flag`. Both are written on EASE-Grid 2.0 M09
(EPSG:6933, 9,008 m).

## Repository layout

```
swapstress/          the citable package — everything on the reproduction path
  cli.py             the stage table; reproduce.sh and docs/REPRODUCE.md follow it
  config.py          TOML run configs and provenance records
  units.py           cm H2O <-> MPa, and the exact log-space shift
  swrc.py            the canonical van Genuchten implementation
  sources/           per-source harmonization to (theta, suction_cm, depth_cm)
  features/          the 9 km Earth Engine covariate stack and the training table
  model/             training, cross-validation, importance, partial dependence
  inference/         daily prediction, gap-filling, and the product writer
  validation/        technical validation, including the Rosetta/POLARIS baseline
  figures/           the descriptor's Figs 1–6, plus the source summaries
research/            preserved exploratory work — not part of the reproduction path
configs/             TOML run configs
notebooks/           01–08, onboarding
tests/               core tests (research/ has its own, not run by default)
reproduce.sh         the driver
```

## Where to look next

- **`docs/REPRODUCE.md`** — the stage-by-stage reproduction guide: inputs,
  outputs, runtimes, credentials, the release environment variables, and what
  stage 07 writes.
- **`docs/DATA_SOURCES.md`** — the five training sources plus ISMN: what each
  one is, lab versus in-situ, and the unit and standardization conventions
  applied to it.
- **`notebooks/`** — eight onboarding notebooks over the same library the stages
  call: `01_source_data`, `02_earth_engine`, `03_amsr_vod`, `04_training_table`,
  `05_smap_l3`, `06_9k_feature_data`, `07_inference`, `08_validation`.
- **`research/README.md`** — the preserved exploratory subtree: flux and
  ecosystem-response analyses, satellite sensor intercomparisons, site-scale van
  Genuchten inversion, neural experiments, and presentation figures. It may
  import the core package; the core package never imports it. Not covered by the
  default test run, and not part of what the descriptor claims to reproduce.
