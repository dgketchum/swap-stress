"""The pipeline's console entry points, and the plumbing they share.

Every stage of the released workflow is one command. :data:`STAGES` is the list
of them in run order; ``reproduce.sh`` and ``docs/REPRODUCE.md`` are generated
from it rather than kept in sync by hand, so a stage cannot be added to the
pipeline without appearing in the documented chain.

Each stage takes its settings from a TOML file (``--config``), accepts CLI
overrides of the same names, and drops a ``provenance.json`` beside its outputs
recording the merged config, the software version, and the git commit. The
uniform ``--dry-run`` resolves and prints every input and output path without
touching them; that is what the pipeline-level dry run is built out of.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass(frozen=True)
class Stage:
    """One step of the reproduction chain."""

    number: str  # '00'..'08', the order they run in
    command: str  # console script name
    target: str  # 'module:function' entry point
    summary: str
    requires: str = ""  # credentials or hardware the stage needs
    implemented: bool = True

    @property
    def module(self) -> str:
        return self.target.split(":")[0]


STAGES: List[Stage] = [
    Stage(
        "00",
        "swapstress-standardize",
        "swapstress.sources.standardize:main",
        "Harmonize each source's raw observations to (theta, suction_cm, depth_cm).",
    ),
    Stage(
        "01",
        "swapstress-extract",
        "swapstress.features.ee_export:main",
        "Sample the covariate stack at every site, then fold the exports into "
        "per-source parquets.",
        requires="Earth Engine credentials and a writable GCS bucket",
    ),
    Stage(
        "02",
        "swapstress-build-table",
        "swapstress.features.build_training_table:main",
        "Join features to observations into the observation-level training table.",
    ),
    Stage(
        "03",
        "swapstress-train",
        "swapstress.model.train:main",
        "Fit the quantile random forest: features + theta -> log10(suction_cm).",
    ),
    Stage(
        "04",
        "swapstress-validate",
        "swapstress.validation.run:main",
        "Technical validation: blocked CV, per-source skill, and the PTF baseline.",
    ),
    Stage(
        "05",
        "swapstress-predict",
        "swapstress.inference.predict:main",
        "Apply the model to the gridded covariate stack, day by day.",
    ),
    Stage(
        "06",
        "swapstress-gapfill",
        "swapstress.inference.gapfill:main",
        "Fill the retrieval gaps in the daily prediction rasters.",
    ),
    Stage(
        "07",
        "swapstress-package",
        "swapstress.inference.product:main",
        "Write the released product: MPa and log10(cm) bands, CF attributes.",
    ),
    Stage(
        "08",
        "swapstress-figures",
        "swapstress.figures.run:main",
        "Render the descriptor's figures and tables.",
    ),
]


def get_stage(number: str) -> Stage:
    for stage in STAGES:
        if stage.number == number:
            return stage
    known = ", ".join(s.number for s in STAGES)
    raise KeyError(f"Unknown stage '{number}'. Known stages: {known}")


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the flags every stage accepts."""
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a TOML run config. CLI flags override its values.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Resolve and print every input and output path, then exit.",
    )
    return parser


def resolve(
    parser: argparse.ArgumentParser,
    argv: Optional[List[str]],
    required: Optional[List[str]] = None,
) -> dict:
    """Parse *argv*, merge it over ``--config``, and check required keys.

    ``argparse`` defaults of ``None`` mean "not given", so TOML values survive;
    anything else is a real default and wins over the file. Stage parsers
    therefore default to None for everything the config is allowed to set.
    """
    from swapstress.config import load_config

    args = parser.parse_args(argv)
    config = load_config(args.config, vars(args))
    for key in required or []:
        if not config.get(key):
            parser.error(f"--{key.replace('_', '-')} is required (via CLI or --config)")
    return config


def report_paths(title: str, inputs: dict, outputs: dict) -> None:
    """Print resolved paths for a dry run, marking which inputs are present.

    Missing inputs are reported, not raised on: a dry run of the whole chain is
    expected to name files that earlier stages have not produced yet.
    """
    print(f"\n{title}")
    for label, path in inputs.items():
        if path is None:
            print(f"  in   {label:<22s} (not configured)")
            continue
        mark = "ok     " if os.path.exists(path) else "MISSING"
        print(f"  in   {label:<22s} [{mark}] {path}")
    for label, path in outputs.items():
        print(f"  out  {label:<22s}          {path}")


def stage_provenance(
    output_dir: str,
    config: dict,
    run_type: str,
    extras: Optional[dict] = None,
) -> str:
    """Write a stage's provenance record, echoing where it landed.

    Named ``provenance_<run_type>.json`` because these stages write into shared
    directories -- standardize, extract, and build-table all land under the same
    tree, and a plain provenance.json would leave only the last one standing.
    """
    from swapstress.config import write_provenance

    path = write_provenance(
        output_dir,
        config,
        run_type=run_type,
        extras=extras,
        filename=f"provenance_{run_type}.json",
    )
    print(f"Wrote {path}")
    return path


def _load_entry(stage: Stage) -> Callable[..., None]:
    import importlib

    module_name, func_name = stage.target.split(":")
    return getattr(importlib.import_module(module_name), func_name)


def main(argv: Optional[List[str]] = None) -> None:
    """``swapstress`` -- list the stages, or dispatch to one by number."""
    parser = argparse.ArgumentParser(
        prog="swapstress",
        description="SWAP-Stress pipeline. Run a stage by number, or list them.",
    )
    parser.add_argument(
        "stage",
        nargs="?",
        default=None,
        help="Stage number to run (00-08). Omit to list the stages.",
    )
    parser.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the stage.",
    )
    args = parser.parse_args(argv)

    if args.stage is None:
        width = max(len(s.command) for s in STAGES)
        for stage in STAGES:
            mark = "" if stage.implemented else "  [not yet implemented]"
            print(f"{stage.number}  {stage.command:<{width}s}  {stage.summary}{mark}")
            if stage.requires:
                print(f"{'':<{width + 6}s}  requires: {stage.requires}")
        return

    stage = get_stage(args.stage)
    if not stage.implemented:
        raise SystemExit(
            f"Stage {stage.number} ({stage.command}) is not implemented yet."
        )
    _load_entry(stage)(args.rest)


if __name__ == "__main__":
    main()
