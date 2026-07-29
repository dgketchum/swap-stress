"""Stage 04: swapstress-validate -- run the technical validation analyses.

Each analysis in this package is its own script with its own flags, because each
answers a different question and needs different inputs. This driver runs them
by name against one model directory, so the descriptor's validation section can
be regenerated with a single command instead of eight.

``--analysis all`` runs the set that only needs the trained model and its cached
test set. Two are excluded and asked for by name: the PTF baseline, which has
its own two-step prep/eval interface and reads an external Rosetta grid, and
quantile coverage, which needs a model trained with ``--quantile``.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from typing import List, Optional

# Analysis name -> module. Order is the order they run in and the order the
# descriptor presents them.
ANALYSES = {
    "baseline": "swapstress.validation.baseline_summary",
    "loso": "swapstress.validation.loso_cv",
    "regional": "swapstress.validation.regional_cv",
    "conditional-bias": "swapstress.validation.conditional_bias",
    "distribution-shift": "swapstress.validation.distribution_shift",
    "sensitivity": "swapstress.validation.sensitivity",
    "within-pixel": "swapstress.validation.within_pixel_variance",
    "error-lookup": "swapstress.validation.empirical_error_lookup",
    "quantile-coverage": "swapstress.validation.quantile_coverage",
    "ptf-baseline": "swapstress.validation.ptf_baseline",
}

# Not in --analysis all, each for its own reason. The PTF baseline has its own
# prep/eval interface and reads an external Rosetta grid. Quantile coverage
# needs a quantile forest: a plain RandomForestRegressor has no predictive
# distribution, so it would fail every run of a non-quantile model.
ON_REQUEST = {"ptf-baseline", "quantile-coverage"}

DEFAULT_ANALYSES = [name for name in ANALYSES if name not in ON_REQUEST]

# The PTF baseline dispatches on a prep/eval subcommand instead of taking the
# shared --model-dir/--output-dir pair, so it gets only what the caller passed.
SELF_ARGS = {"ptf-baseline"}


def _invoke(module_name: str, argv: List[str]) -> None:
    """Call a validation module's ``main()`` with *argv*.

    These modules parse ``sys.argv`` inside ``main()`` rather than taking an
    argv parameter, so the swap below is what lets one process run all of them.
    """
    module = importlib.import_module(module_name)
    saved = sys.argv
    sys.argv = [module_name.rsplit(".", 1)[-1], *argv]
    try:
        module.main()
    finally:
        sys.argv = saved


def build_parser() -> argparse.ArgumentParser:
    from swapstress.cli import add_common_args

    parser = argparse.ArgumentParser(
        prog="swapstress-validate",
        description="Stage 04: technical validation of a trained model.",
    )
    add_common_args(parser)
    parser.add_argument(
        "--analysis",
        type=str,
        nargs="+",
        default=None,
        choices=["all", *ANALYSES],
        help=f"Analyses to run (default: all -> {' '.join(DEFAULT_ANALYSES)}).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Trained model directory to validate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where results land (default: <model-dir>/error_analysis).",
    )
    parser.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to a single named analysis.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    from swapstress.cli import report_paths, resolve, stage_provenance

    config = resolve(build_parser(), argv, required=["model_dir"])
    requested = config.get("analysis") or ["all"]
    if "all" in requested:
        requested = DEFAULT_ANALYSES

    output_dir = config.get("output_dir") or os.path.join(
        config["model_dir"], "error_analysis"
    )

    if config["dry_run"]:
        report_paths(
            "04 validate",
            {"model dir": config["model_dir"]},
            {"results": output_dir},
        )
        print(f"  runs {' '.join(requested)}")
        return

    forwarded = ["--model-dir", config["model_dir"], "--output-dir", output_dir]
    extra = [a for a in config.get("rest", []) if a != "--"]
    if extra and len(requested) > 1:
        raise SystemExit(
            "Extra arguments only make sense with a single --analysis; "
            f"got {len(requested)}: {' '.join(requested)}"
        )

    for name in requested:
        print(f"\n=== {name} ===")
        _invoke(ANALYSES[name], extra if name in SELF_ARGS else forwarded + extra)

    stage_provenance(
        output_dir,
        config,
        run_type="validate",
        extras={"analyses": requested, "upstream": config["model_dir"]},
    )


if __name__ == "__main__":
    main()
