"""Tests for swapstress.cli — the stage table and the entry points it names."""

import importlib
import tomllib
from pathlib import Path

import pytest

from swapstress.cli import STAGES, get_stage

REPO_ROOT = Path(__file__).resolve().parents[1]


def _declared_scripts():
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["scripts"]


class TestStageTable:
    """The stage table is what reproduce.sh and REPRODUCE.md are written from."""

    def test_numbers_are_sequential(self):
        assert [s.number for s in STAGES] == [f"{i:02d}" for i in range(len(STAGES))]

    def test_commands_are_unique(self):
        commands = [s.command for s in STAGES]
        assert len(set(commands)) == len(commands)

    def test_get_stage_rejects_unknown(self):
        with pytest.raises(KeyError, match="Unknown stage"):
            get_stage("99")


class TestConsoleScripts:
    """Every implemented stage must be installable and importable."""

    def test_declared_in_pyproject(self):
        scripts = _declared_scripts()
        for stage in STAGES:
            if stage.implemented:
                assert stage.command in scripts, f"{stage.command} not in the wheel"
                assert scripts[stage.command] == stage.target

    def test_unimplemented_stages_are_not_declared(self):
        """A registered-but-missing entry point would fail at install time."""
        scripts = _declared_scripts()
        for stage in STAGES:
            if not stage.implemented:
                assert stage.command not in scripts

    @pytest.mark.parametrize(
        "stage", [s for s in STAGES if s.implemented], ids=lambda s: s.command
    )
    def test_entry_point_resolves(self, stage):
        module_name, func_name = stage.target.split(":")
        module = importlib.import_module(module_name)
        assert callable(getattr(module, func_name))

    @pytest.mark.parametrize(
        "stage", [s for s in STAGES if s.implemented], ids=lambda s: s.command
    )
    def test_accepts_dry_run(self, stage):
        """--dry-run is the flag reproduce.sh passes to every stage."""
        module_name = stage.module
        module = importlib.import_module(module_name)
        options = module.build_parser()._option_string_actions
        assert "--dry-run" in options
        assert "--config" in options


def test_store_true_flags_default_none():
    """A store_true flag with a False default silently overrides a TOML
    ``key = true`` in load_config's merge, which treats only None as "not
    specified". Every boolean stage flag must default to None."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "swapstress"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text()
        for match in re.finditer(
            r'add_argument\(\s*"(--[\w-]+)"[^)]*?action="store_true"[^)]*?\)',
            text,
            re.S,
        ):
            if "default=None" not in match.group(0):
                offenders.append(f"{path.name}: {match.group(1)}")
    assert not offenders, (
        f"store_true flags whose False default would override TOML true: {offenders}"
    )
