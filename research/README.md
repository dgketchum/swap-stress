# research/

Preserved exploratory work. **Not part of the Scientific Data descriptor's
reproduction path, and not covered by the test suite.**

Everything here was real analysis that informed the released product, and its
history is intact — this subtree was created by `git mv`, not by re-adding
files. It lives in the same repository so there is one clone and one citable
archive, but it is excluded from `pytest` and from the ruff-enforced set, and
nothing in `swapstress/` imports from it.

The dependency arrow runs one way: `research/` may import the core package;
the core package never imports `research/`.

## Layout

| Directory | Contents |
|---|---|
| `flux/` | Eddy-covariance flux analyses — beta/stress models, drought stress, plant potential, transferability, quartile-binned MLR, and their four tests |
| `sensors/` | Satellite soil-moisture sensor intercomparison against ISMN — SMAP L4/L-band, NISAR SME2, SMOS-IC, and the download utilities |
| `vg_inversion/` | Site-scale van Genuchten inversion — `site_modeling/`, curve-fitting comparisons, the station training table |
| `neural/` | Neural experiments — tabular NN and masked autoencoder |
| `et/` | Evapotranspiration work (PT-JPL) |
| `figures/` | Presentation and poster figures that do not back SD Figs 1–6, plus the CONUS animations and empirical SWRC summaries |
| `extract/` | One-off extractions — CZE, FROM-GLC10, NCSS |

## Running it

Imports here were repaired in Phase 3's mechanical rewrite, so this subtree
runs: `pytest research/` collects and passes its 54 tests. It is still excluded
from the default `pytest` run and from the ruff-enforced set, because it is not
part of what the descriptor claims to reproduce — not because it is broken.

## What stayed behind

Six figure scripts went to `swapstress/figures/` instead, because they back the
descriptor's figures. Phase 6 renamed them to the descriptor's own numbering,
so the module name and the figure number no longer disagree:

| Was | Now | Role |
|---|---|---|
| `fig3_pipeline.py` | `fig01_pipeline.py` | Fig 1 |
| — | `fig02_coverage.py` | Fig 2 (new in Phase 6) |
| `fig11b_drought_timeseries.py` | `fig03_pixel_series.py` | Fig 3 (rewritten) |
| — | `fig04_validation_scatter.py` | Fig 4 (new in Phase 6) |
| `fig7_koppen_transferability.py` | `fig05_spatial_skill.py` | Fig 5 |
| `fig6b_error_map.py` | `fig06_uncertainty.py` | Fig 6 |
| `fig_vg_vs_direct.py` | `vg_vs_direct.py` | supporting |
| `fig5_kfold_validation.py` | `kfold_validation.py` | supporting |

The last two are analyses the plan listed under Fig 4, but the figure the
outline actually specifies is a scatter against the PTF baselines. They still
render on request and are excluded from `--figure all`.

`distributions.py` went there too. It was to fold into
`swapstress/figures/summaries.py`, but that is Phase 7 work, not Phase 6: both
modules back the notebooks and Table 1 rather than Figs 1-6, and neither is
imported anywhere in `swapstress/` yet. Merging them before the notebooks call
them would mean settling on an API with nothing exercising it.
