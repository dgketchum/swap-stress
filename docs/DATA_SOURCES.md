# Training data sources

Five sources supply the paired water-content / matric-potential observations the
model trains on. Stage 00 (`swapstress-standardize`) harmonizes each one to the
same three columns, and nothing downstream needs to know which source a row came
from except for the per-source skill reporting.

```
theta       volumetric water content, fraction, 0-1
suction_cm  suction head, cm H2O, positive and increasing as the soil dries
depth_cm    measurement depth, cm; horizon midpoint where the source reports an interval
```

The internal convention is **positive suction head in cm H₂O** throughout, and
the model target is its base-10 logarithm. The signed, negative
`matric_potential_MPa` convention appears only at the very end, when stage 07
writes the released product; see `swapstress/units.py`.

## Conventions applied to every source

| Constant | Value | Where |
|---|---|---|
| MPa → cm H₂O | × 10197.16 | ReESH, and the released MPa band |
| kPa → cm H₂O | × 10.19716 | MT Mesonet, LaCADIAN |
| bar → cm H₂O | × 1019.72 | NCSS |
| m → cm | × 100 | GSHP |

Every source's raw potential column is passed through `abs()` after conversion,
so a source that reports negative matric potential and one that reports positive
suction head land in the same place.

`apply_physical_filters` in `swapstress/sources/standardize.py` then runs on all
of them and drops rows outside physical bounds, reporting the counts:

- `suction_cm` must be > 0 and ≤ 10⁶ cm (100 MPa) — beyond any real soil
  measurement.
- `theta` must be in [0, 1] by definition.

`swapstress/features/build_training_table.py` re-checks these when it assembles
the observation-level table, and additionally requires `suction_cm ≥ 10⁻³` so the
log transform is safe.

Depth is standardized to `depth_cm` and mapped to a Rosetta level 1–7
(`swapstress/sources/depth.py`) in the training table. Each source declares the
identifier the spatial holdout blocks on — `index_col` / `group_col` in
`swapstress/sources/registry.py`, which is also the one place that knows where a
source's raw files, site shapefile, MGRS index, Earth Engine export prefix, and
standardized outputs live. Adding a source is a registry entry plus one
standardizer.

`--minimum-points` (default 4) drops a curve at a given depth with too few
retention points. It applies only to the two lab curve sources, GSHP and NCSS.

| Source | Type | Grouped on | Raw input |
|---|---|---|---|
| GSHP | laboratory retention curves | `profile_id` | `WRC_dataset_surya_et_al_2021_final.csv` |
| NCSS | laboratory characterization | `profile_id` | `ncss_selection.parquet` |
| MT Mesonet | laboratory retention curves | `station` | `swp.csv` + `station_metadata.csv` |
| ReESH | laboratory retention curves | `site_id` | `*_SoilWaterRetentionCurves.csv` |
| LaCADIAN | in-situ paired sensors | `station` | `swp.csv` + `station_metadata.csv` |

A sixth entry, `rosetta`, is registered but is a gridded pedotransfer prior used
for 250 m pretraining rather than an observation source. It is not in
`DEFAULT_SOURCES` and has to be opted into explicitly.

---

## GSHP — Global Soil Hydraulic Properties

Global compilation of laboratory-measured water retention curves (Gupta et al.,
2021). Measurement type: **laboratory**. The largest single contributor to the
training table.

**Standardization** (`standardize_gshp`). Rows are kept where
`data_flag == "good quality estimate"`, and rows missing `lab_head_m` or
`lab_wrc` are dropped. `lab_head_m` is tabulated in **metres**, so suction is
`|lab_head_m| × 100`; a guard drops layers with `|lab_head_m| > 10⁴ m`, which
exist in at least one contributing study (values of 10¹⁹–10³¹ m). `lab_wrc` is
already volumetric, so `theta` is taken as-is. Depth is the horizon midpoint,
`(hzn_top + hzn_bot) / 2`.

`SWCC_classes`, texture (`sand_tot_psa`, `silt_tot_psa`, `clay_tot_psa`), bulk
density (`db_od`), and `climate_classes` are carried through the standardized
files for downstream diagnostics only. Texture and bulk density are deliberately
**not** in the feature set: they are lab measurements, unavailable at inference.

### Parameters are taken from the published dataset, not refit

`swapstress/sources/gshp.py::load_published_params` reads the van Genuchten
parameters Gupta et al. published rather than refitting the curves.

The published `alpha` is in **1/m**, because GSHP tabulates head in metres, and
the loader is **the only place in the codebase that converts it to 1/cm** (along
with `se_alpha` and the alpha quantile columns). Getting this wrong is not
subtle but it is silent: evaluating the published parameters against the
published observations gives a median RMSE of 0.006 m³/m³ under the metre
convention and 0.21 if `alpha` is read as 1/cm.

Refitting was tried and rejected. Over the 10,411 layers where both existed,
median RMSE against the same observations was 0.0074 (ours) versus 0.0082
(published) — equivalent in fit — but 933 of our fits, 9.0%, returned
`theta_r >= theta_s`, which `swapstress.swrc` correctly refuses to evaluate. The
published set has none. Taking them directly also raises coverage from 10,411 to
15,259 layers and brings published uncertainty (`se_alpha`, `se_n`, and
quantiles) that deterministic refits never produced.

The quality filter is not decoration: `alpha` is capped at exactly 100 1/m and
`n` at exactly 7, which are the optimizer's bounds, and 1,045 layers rest on
them; a further 2,490 are flagged for a flat likelihood profile. In all, 3,554 of
15,259 layers (23.3%) are not `"good quality estimate"`, which is why
`good_quality_only` defaults to True. `at_param_bound` is computed before the
unit conversion, while the bounds are still in the units they were set in.

### Parameter-space comparisons must restrict to `SWCC_classes == "YWYD"`

`SWCC_classes` records which end of the curve was actually measured, and that
determined how freely GSHP could fit `theta_r` and `theta_s`:

| Class | Layers | Meaning |
|---|---|---|
| `YWYD` | 10,301 (67.5%) | both ends measured; `theta_r` and `theta_s` fit freely on [0, 1] |
| `NWYD` | 4,598 (30.1%) | no wet end; **`theta_s` confined to a prediction interval from a texture + bulk-density + climate-region regression** |
| `YWND` | 333 (2.2%) | no dry end; `theta_r` bounded by a specific surface area estimate |
| `NWND` | 27 (0.2%) | both constrained |

The 30.1% `NWYD` subset therefore carries a **pedotransfer prior on `theta_s`**.
That is harmless for training, which uses observation-level (theta, psi) pairs
and never sees these parameters — but it is **circular against a texture-based
PTF**. Any comparison of GSHP parameters against Rosetta or POLARIS must pass
`swcc_classes=("YWYD",)`, or it is partly measuring one texture regression
against another.

One further caveat: `WRC_dataset_surya_et_al_2021_final_clean.csv` is the
per-profile file written for Earth Engine point extraction. It collapses each
profile with `first`, so its parameter columns belong to whichever layer happened
to sort first. It is not a source of per-layer parameters — point
`load_published_params` at the full dataset.

---

## NCSS — USDA National Cooperative Soil Survey

Laboratory characterization data from the USDA Kellogg Soil Survey Laboratory.
Measurement type: **laboratory**.

**Standardization** (`swapstress/sources/ncss.py::ncss_to_standardized`).
Retention is reported at fixed pressures rather than as a continuous curve, so
each retention column maps to its pressure in bars — 0.06, 0.10, 0.33, 1, 2, 5,
and 15 bar — and `suction_cm = bar × 1019.72`. The wide table is melted so each
(sample, pressure) pair becomes one observation.

Water retention is reported **gravimetrically, as a percent**, so the volumetric
conversion needs bulk density:

```
theta = (wr_val / 100) * bulk_density_oven_dry
```

Rows whose oven-dry bulk density is missing or outside [0.5, 2.5] g/cm³ are
dropped rather than converted — without a valid bulk density there is no
volumetric water content to be had.

`profile_id` is `pedon_key` (falling back to `labsampnum`), and depth is
`hzn_mid_cm`, falling back to `(hzn_top + hzn_bot) / 2`.

NCSS has no `SWCC_classes` column of its own, so one is **derived** per profile
from the coverage actually present, mirroring GSHP's labels: the wet end counts
as measured if the minimum suction is ≤ 150 cm, the dry end if the maximum is
≥ 14,000 cm.

---

## MT Mesonet — Montana Mesonet

Automated environmental monitoring stations across Montana. Measurement type:
**laboratory** — retention data measured on soil samples collected at the
stations, using the same instrumentation as ReESH (HYPROP sample analysis), not
readings from the installed field probes. `swapstress/sources/mt_mesonet.py`
pulls the series from the Mesonet API and pivots the long-format response into
per-station tables; `swp.csv` is the paired extract, with columns
`station, Depth [cm], KPA, VWC`.

**Standardization** (`standardize_mt_mesonet`). `suction_cm = |KPA| × 10.19716`,
and `VWC` is already a volumetric fraction. Two source-specific filters run
before the shared physical ones:

- `VWC < 0` is dropped.
- `|KPA| > 200` is dropped — 200 kPa (≈ 2,000 cm) is a reasonable upper bound
  for a field sensor, and readings beyond it are sensor artefacts.

Grouped on `station`.

> MT Mesonet's held-out skill has been anomalously high in past runs, well above
> what the other sources achieve. Stage 04 reports per-source skill by default so
> this stays visible rather than being averaged away.

---

## ReESH

Ecosystem research sites, largely co-located with AmeriFlux towers, distributing
soil water retention curves measured on collected samples. Measurement type:
**laboratory** — the retention curves come from sample analysis (the source files
carry a HYPROP filter column and a sampling date), not from installed probes.

**Standardization** (`standardize_reesh`). Each site ships a
`*_SoilWaterRetentionCurves.csv` with `MPa_Abs` (matric potential magnitude, MPa)
and `Vol_Water` (volumetric water content, **percent**):

```
suction_cm = |MPa_Abs| * 10197.16
theta      = Vol_Water / 100
```

Depth comes from `Depth_cm`. Curves are grouped by `Plot` within a `Site`, and
`Sample_ID` is preferred as the identifier where present. Standardized files are
written as `<Site>_<Plot>.csv`; the registry groups the source on `site_id`.

---

## LaCADIAN — Louisiana Climate & Digital Ag Network

Automated field stations across Louisiana. Measurement type: **in-situ only** —
co-located soil moisture and matric potential probes at multiple depths, with no
laboratory analysis behind any of its observations. It is the one source with no
lab component at all.

`swapstress/sources/lacadian_download.py` retrieves the daily station files and
`lacadian.py` reshapes them. Soil moisture is recorded at 5, 10, 20, 50, and
100 cm and matric potential at 5, 20, and 50 cm, so the **paired depths are 5,
20, and 50 cm** — only those yield a (theta, suction) observation. `swp.csv` has
columns `station, KPA, VWC, depth_cm`, with `KPA` negative in the matric
potential convention.

**Standardization** (`standardize_lacadian`). Same conversion as MT Mesonet,
`suction_cm = |KPA| × 10.19716`, with the same `VWC < 0` and `|KPA| > 200`
filters, plus one that is specific to these sensors:

- readings at or above −0.15 kPa are dropped. The sensor's lower detection limit
  is −0.1 kPa, so values there are clamped at the floor rather than measured, and
  they would otherwise pile up as spuriously wet observations.

Grouped on `station`.

---

## ISMN — International Soil Moisture Network

**ISMN is not a training source.** ISMN stations report soil moisture (theta)
only; very few measure matric potential. Adding ISMN to the training table would
mean supplying suction values that were themselves imputed — the model would then
be trained partly on its own assumptions. It is deliberately absent from
`DEFAULT_SOURCES`.

What it is used for:

- `swapstress/sources/ismn.py::build_ismn_vwc_series` converts ISMN soil moisture
  to per-station daily VWC parquets (station keyed `network:station` to avoid
  collisions across networks), with per-sensor quality-flag plots.
- `swapstress/sources/ismn_sites.py` builds the station-metadata shapefile used
  to locate those stations.
- Those series back the independent in-situ evaluation of the theta input, and
  the satellite soil-moisture intercomparisons preserved under
  `research/sensors/` (SMAP L3/L4, SPL2SMAP_S, NISAR SME2, SMOS-IC).

Because ISMN is an independent evaluation set rather than a training input, it
stays usable as a check on the released product.
