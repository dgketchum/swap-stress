"""
Run configuration loading and provenance tracking for SWAP-Stress pipeline.

Provides:
- load_config(): Merge TOML config with CLI overrides
- write_provenance(): Generate provenance.json for output directories
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

_VERSION_PATH = Path(__file__).resolve().parents[1] / "VERSION"
_MAX_CHECKSUM_BYTES = 500 * 1024 * 1024  # 500 MB


def load_config(
    config_path: str | None,
    cli_args: dict[str, Any],
) -> dict[str, Any]:
    """Load a TOML config file and merge with CLI argument overrides.

    Parameters
    ----------
    config_path : str or None
        Path to TOML file.  If None, returns cli_args with None values stripped.
    cli_args : dict
        CLI arguments from ``vars(argparse.Namespace)``.
        Keys with ``None`` values are treated as "not specified" and do not
        override TOML values.  ``False`` values are preserved.

    Returns
    -------
    dict
        Merged configuration.  CLI values override TOML values.
    """
    if config_path is not None:
        with open(config_path, "rb") as f:
            base = tomllib.load(f)
    else:
        base = {}

    # Strip None and the 'config' key itself from CLI args
    overrides = {k: v for k, v in cli_args.items() if v is not None and k != "config"}
    base.update(overrides)
    return base


def write_provenance(
    output_dir: str,
    config: dict[str, Any],
    run_type: str,
    extras: dict[str, Any] | None = None,
    filename: str = "provenance.json",
) -> str:
    """Write a ``provenance.json`` artifact to *output_dir*.

    Parameters
    ----------
    output_dir : str
        Directory to write provenance.json.
    config : dict
        Full merged config dict.
    run_type : str
        One of ``'train'``, ``'feature_importance'``, ``'predict'``,
        ``'gapfill'``.
    extras : dict, optional
        Additional top-level fields (``inputs``, ``outputs``, ``upstream``).
    filename : str
        Name to write under *output_dir*.  Stages that own their output
        directory keep the default; stages that share one (several write into
        ``swapstress/training/``) pass a stage-specific name so the last one to
        run does not erase the others' record.

    Returns
    -------
    str
        Path to written provenance.json.
    """
    doc: Dict[str, Any] = {
        "provenance_version": "1.0",
        "run_type": run_type,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "software_version": get_version(),
        "git_commit": get_git_commit(),
        "config": config,
    }
    if extras:
        doc.update(extras)

    out_path = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2, default=str)
    return out_path


def get_version() -> str:
    """Read software version from the VERSION file."""
    try:
        return _VERSION_PATH.read_text().strip()
    except FileNotFoundError:
        return "unknown"


def get_git_commit() -> Optional[str]:
    """Get current git commit hash, or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def input_checksum(path: str) -> Optional[str]:
    """Compute SHA-256 checksum for provenance tracking.

    Returns None for files larger than 500 MB or on error.
    """
    try:
        size = os.path.getsize(path)
        if size > _MAX_CHECKSUM_BYTES:
            return None
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def feature_groups_to_exclude(
    feature_groups: list[str],
) -> list[str]:
    """Convert a positive feature_groups list to an exclude_groups list.

    Parameters
    ----------
    feature_groups : list of str
        Groups to *include* in the model.

    Returns
    -------
    list of str
        Groups to *exclude*, suitable for ``filter_feature_groups()``.
    """
    from swapstress.features.features import FEATURE_GROUPS

    # Skip meta-aliases like "landsat" that overlap with sub-groups
    all_groups = {k for k in FEATURE_GROUPS if k not in ("landsat",)}
    return sorted(all_groups - set(feature_groups))
