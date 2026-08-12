"""Stage 08: swapstress-figures -- render the descriptor's figures.

Every figure module here exposes ``main(argv)`` and accepts ``--output-dir``, so
this driver is a name-to-module table plus a loop. The modules are numbered by
the descriptor's Fig 1-7 paper order; the keys are what the CLI calls them, so a
later module rename does not change this interface.

The old module numbers (``fig3_``, ``fig5_``, ``fig7_``, ``fig11b_``) were a
presentation deck's ordering, not the paper's, and are gone; the 2026-08-11
rename aligned module numbers with the paper's Fig 1-7. ``pixel_series`` and
``vg_vs_direct`` are supporting analyses -- they still render on request but
are not part of ``all``.

A figure that fails is reported and the run continues. Most of these read a
released product or a trained model, and a missing one should not stop the rest
of the figure set from rendering.
"""

from __future__ import annotations

import argparse
import importlib
import traceback
from typing import List, Optional

# Descriptor figures (2026-08-12 lineup) -> module providing main(argv).
# Module numbers match the paper's Fig 1-7 order plus the supplement:
# the feature stack/importance figure took the Fig 3 slot from coverage,
# which moved to Supplementary Fig 1 but stays in ``all`` -- it is still
# a paper figure.
MAIN_FIGURES = {
    "pipeline": "swapstress.figures.fig01_pipeline",
    "training-sources": "swapstress.figures.fig02_training_sources",
    "features": "swapstress.figures.fig03_features",
    "kfold": "swapstress.figures.fig04_kfold_validation",
    "validation-scatter": "swapstress.figures.fig05_ptf_comparison",
    "spatial-skill": "swapstress.figures.fig06_regional_skill",
    "uncertainty": "swapstress.figures.fig07_product_maps",
    "coverage": "swapstress.figures.figS1_coverage",
}

# Rendered on request, not part of --figure all. pixel-series dropped from
# the mains to a supplementary candidate in the 2026-07-29 lineup.
SUPPORTING_FIGURES = {
    "pixel-series": "swapstress.figures.pixel_series",
    "vg-vs-direct": "swapstress.figures.vg_vs_direct",
}

FIGURES = {**MAIN_FIGURES, **SUPPORTING_FIGURES}

DEFAULT_OUTPUT_DIR = "figs/descriptor"


def build_parser() -> argparse.ArgumentParser:
    from swapstress.cli import add_common_args

    parser = argparse.ArgumentParser(
        prog="swapstress-figures",
        description="Stage 08: render the descriptor's figures.",
    )
    add_common_args(parser)
    parser.add_argument(
        "--figure",
        type=str,
        nargs="+",
        default=None,
        choices=["all", *FIGURES],
        help=f"Figures to render (default: all -> {' '.join(MAIN_FIGURES)}). "
        f"Supporting analyses, on request only: {' '.join(SUPPORTING_FIGURES)}.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Directory for rendered figures (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to a single named figure.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    from swapstress.cli import report_paths, resolve, stage_provenance

    config = resolve(build_parser(), argv)
    requested = config.get("figure") or ["all"]
    if "all" in requested:
        requested = list(MAIN_FIGURES)
    output_dir = config.get("output_dir") or DEFAULT_OUTPUT_DIR

    if config.get("dry_run"):
        report_paths("08 figures", {}, {"figures": output_dir})
        print(f"  renders {' '.join(requested)}")
        return

    extra = [a for a in config.get("rest", []) if a != "--"]
    if extra and len(requested) > 1:
        raise SystemExit(
            "Extra arguments only make sense with a single --figure; "
            f"got {len(requested)}: {' '.join(requested)}"
        )

    failed = []
    for name in requested:
        print(f"\n=== {name} ===")
        try:
            # Imported inside the guard: these modules pull in geopandas,
            # rasterio and scikit-learn at module level, and one missing
            # optional dependency should cost that figure, not the batch.
            module = importlib.import_module(FIGURES[name])
            module.main(["--output-dir", output_dir, *extra])
        except Exception:
            traceback.print_exc()
            failed.append(name)

    stage_provenance(
        output_dir,
        config,
        run_type="figures",
        extras={"figures": requested, "failed": failed},
    )
    if failed:
        raise SystemExit(f"{len(failed)} figure(s) failed: {' '.join(failed)}")


if __name__ == "__main__":
    main()
