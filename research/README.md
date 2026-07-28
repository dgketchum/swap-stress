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

## Caveat on imports

Phase 2 of the refactor moved these files without touching their import
statements, so modules here still import from the pre-refactor top-level names
(`map.*`, `viz.*`). They are repaired in the single mechanical rewrite in
Phase 3, alongside the core package. Until then, expect imports in this subtree
to fail; the code and its history are what is being preserved, not its
immediate runnability.

## What stayed behind

Six scripts remain in `viz/presentation/` because they back the descriptor's
figures: `fig3_pipeline.py` (Fig 1), `fig11b_drought_timeseries.py` (Fig 3),
`fig_vg_vs_direct.py` and `fig5_kfold_validation.py` (Fig 4),
`fig7_koppen_transferability.py` (Fig 5), and `fig6b_error_map.py` (Fig 6).
`viz/emprical_summaries/distributions.py` also stayed; it folds into
`swapstress/figures/summaries.py` in Phase 3.
