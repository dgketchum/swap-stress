"""
Generate an ablation-pruned TOML config from a base config and ablation CSV.

Reads the group_ablation.csv produced by feature_importance.py, drops groups
whose r2_drop is at or below the threshold, and writes a new TOML config with
the surviving feature_groups list.

Usage:
    uv run python -m swapstress.model.prune_config \
        --ablation-csv /nas/.../group_ablation.csv \
        --base-config configs/train_9km_global.toml \
        --threshold 0.0 \
        --output configs/train_9km_global_pruned.toml
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def read_ablation(path: str) -> list[dict]:
    """Read group_ablation.csv into a list of dicts."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def groups_to_keep(
    ablation_rows: list[dict],
    threshold: float,
) -> list[str]:
    """Return sorted list of groups with r2_drop > threshold."""
    keep = []
    for row in ablation_rows:
        group = row["excluded_group"]
        if group.startswith("none"):
            continue
        r2_drop = float(row["r2_drop"])
        if r2_drop > threshold:
            keep.append(group)
    return sorted(keep)


def write_toml(
    base_config: dict,
    feature_groups: list[str],
    output_dir: str | None,
    out_path: str,
) -> None:
    """Write a new TOML config with the pruned feature_groups."""
    lines = []
    lines.append(f'run_type = "{base_config.get("run_type", "train")}"')
    lines.append("")
    lines.append(f'obs_table = "{base_config["obs_table"]}"')

    if output_dir:
        lines.append(f'output_dir = "{output_dir}"')
    elif "output_dir" in base_config:
        lines.append(f'output_dir = "{base_config["output_dir"]}"')

    lines.append("")
    lines.append("feature_groups = [")
    for g in feature_groups:
        lines.append(f'    "{g}",')
    lines.append("]")

    # Copy numeric/boolean settings from base
    for key in ["n_estimators", "test_size", "random_state", "resolution_m"]:
        if key in base_config:
            val = base_config[key]
            if isinstance(val, float) and val == int(val):
                lines.append(f"{key} = {int(val)}")
            else:
                lines.append(f"{key} = {val}")

    lines.append("")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ablation-pruned TOML config from ablation CSV + base config.",
    )
    parser.add_argument(
        "--ablation-csv",
        type=str,
        required=True,
        help="Path to group_ablation.csv from feature_importance.py.",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        required=True,
        help="Path to base TOML config to prune.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Drop groups with r2_drop <= this value (default: 0.0).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override the output_dir in the generated TOML.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write the pruned TOML config.",
    )
    args = parser.parse_args()

    # Read ablation results
    ablation_rows = read_ablation(args.ablation_csv)
    keep = groups_to_keep(ablation_rows, args.threshold)

    # Read base config
    with open(args.base_config, "rb") as f:
        base_config = tomllib.load(f)

    # Determine which groups the ablation actually tested
    ablated_groups = {
        r["excluded_group"]
        for r in ablation_rows
        if not r["excluded_group"].startswith("none")
    }

    # Filter keep list to groups present in the base config
    base_groups = base_config.get("feature_groups")
    if base_groups is not None:
        keep = [g for g in keep if g in base_groups]
        # Preserve base config groups that were never ablated (no evidence to drop)
        not_ablated = [g for g in base_groups if g not in ablated_groups]
        keep = sorted(set(keep) | set(not_ablated))

    dropped = sorted(ablated_groups - set(keep))

    print(f"Ablation threshold: r2_drop <= {args.threshold}")
    print(f"Groups kept ({len(keep)}): {keep}")
    print(f"Groups dropped ({len(dropped)}): {dropped}")
    if base_groups:
        not_tested = sorted(set(base_groups) - ablated_groups)
        if not_tested:
            print(f"Groups not ablated (kept by default): {not_tested}")

    if not keep:
        print("ERROR: no groups survive pruning", file=sys.stderr)
        sys.exit(1)

    # Handle meta-group aliases: if "landsat" and "landsat_bands" are both
    # in keep, only emit "landsat_bands" (they have the same ablation row)
    if "landsat" in keep and "landsat_bands" in keep:
        keep.remove("landsat")

    write_toml(base_config, keep, args.output_dir, args.output)


if __name__ == "__main__":
    main()
