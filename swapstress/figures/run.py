"""Stage 08: swapstress-figures -- render the descriptor's figures.

Every figure module here exposes ``main(argv)`` and accepts ``--output-dir``, so
this driver is a name-to-module table plus a loop. It keeps the pre-refactor
module names; Phase 6 of ``notes/refactor_plan.md`` is what maps them onto the
descriptor's Fig 1-6 numbering, and the keys below are the names the descriptor
uses so that renaming the modules will not change this interface.

A figure that fails is reported and the run continues. Most of these read a
released product or a trained model, and a missing one should not stop the rest
of the figure set from rendering.
"""

from __future__ import annotations

import argparse
import importlib
import traceback
from typing import List, Optional

# Descriptor figure -> module providing main(argv).
FIGURES = {
    "pipeline": "swapstress.figures.fig3_pipeline",
    "vg-vs-direct": "swapstress.figures.fig_vg_vs_direct",
    "kfold": "swapstress.figures.fig5_kfold_validation",
    "koppen": "swapstress.figures.fig7_koppen_transferability",
    "error-map": "swapstress.figures.fig6b_error_map",
    "drought": "swapstress.figures.fig11b_drought_timeseries",
}

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
        help=f"Figures to render (default: all -> {' '.join(FIGURES)}).",
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
        requested = list(FIGURES)
    output_dir = config.get("output_dir") or DEFAULT_OUTPUT_DIR

    if config["dry_run"]:
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
        module = importlib.import_module(FIGURES[name])
        try:
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
