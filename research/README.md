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
descriptor's figures: `fig3_pipeline.py` (Fig 1), `fig11b_drought_timeseries.py`
(Fig 3), `fig_vg_vs_direct.py` and `fig5_kfold_validation.py` (Fig 4),
`fig7_koppen_transferability.py` (Fig 5), and `fig6b_error_map.py` (Fig 6).
`distributions.py` went there too; it folds into
`swapstress/figures/summaries.py` in Phase 6.
