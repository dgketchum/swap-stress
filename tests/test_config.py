"""Tests for swapstress.config — TOML loading and provenance generation."""

import json
import os


from swapstress.config import (
    feature_groups_to_exclude,
    get_git_commit,
    get_version,
    input_checksum,
    load_config,
    write_provenance,
)


class TestLoadConfig:
    """Test TOML loading and CLI merge."""

    def test_cli_only(self):
        """Without a TOML file, returns CLI args minus None values."""
        cli = {"obs_table": "/data/train.parquet", "output_dir": "/out", "config": None}
        result = load_config(None, cli)
        assert result["obs_table"] == "/data/train.parquet"
        assert result["output_dir"] == "/out"
        assert "config" not in result

    def test_toml_only(self, tmp_path):
        """Reads all keys from TOML when CLI is all None."""
        toml_file = tmp_path / "run.toml"
        toml_file.write_text(
            'run_type = "train"\n'
            'obs_table = "/data/train.parquet"\n'
            "n_estimators = 500\n"
        )
        result = load_config(str(toml_file), {"config": str(toml_file)})
        assert result["run_type"] == "train"
        assert result["obs_table"] == "/data/train.parquet"
        assert result["n_estimators"] == 500

    def test_cli_overrides_toml(self, tmp_path):
        """CLI values override TOML values."""
        toml_file = tmp_path / "run.toml"
        toml_file.write_text("n_estimators = 500\n")
        result = load_config(
            str(toml_file),
            {"config": str(toml_file), "n_estimators": 100},
        )
        assert result["n_estimators"] == 100

    def test_none_cli_does_not_override(self, tmp_path):
        """CLI None values don't clobber TOML values."""
        toml_file = tmp_path / "run.toml"
        toml_file.write_text("n_estimators = 500\n")
        result = load_config(
            str(toml_file),
            {"config": str(toml_file), "n_estimators": None, "obs_table": None},
        )
        assert result["n_estimators"] == 500
        assert "obs_table" not in result

    def test_false_cli_preserved(self, tmp_path):
        """CLI False values are not stripped (they're meaningful for booleans)."""
        toml_file = tmp_path / "run.toml"
        toml_file.write_text("overwrite = true\n")
        result = load_config(
            str(toml_file),
            {"config": str(toml_file), "overwrite": False},
        )
        assert result["overwrite"] is False

    def test_toml_list(self, tmp_path):
        """TOML list values are preserved."""
        toml_file = tmp_path / "run.toml"
        toml_file.write_text(
            'feature_groups = ["landsat_bands", "soilgrids", "worldclim"]\n'
        )
        result = load_config(str(toml_file), {"config": str(toml_file)})
        assert result["feature_groups"] == ["landsat_bands", "soilgrids", "worldclim"]


class TestWriteProvenance:
    """Test provenance.json generation."""

    def test_writes_json(self, tmp_path):
        """Provenance file is written with expected keys."""
        out = str(tmp_path / "model_out")
        config = {"obs_table": "/data/train.parquet", "n_estimators": 250}

        path = write_provenance(out, config, "train")
        assert os.path.exists(path)

        with open(path) as f:
            doc = json.load(f)

        assert doc["provenance_version"] == "1.0"
        assert doc["run_type"] == "train"
        assert "timestamp_utc" in doc
        assert "software_version" in doc
        assert "git_commit" in doc
        assert doc["config"]["obs_table"] == "/data/train.parquet"

    def test_extras_merged(self, tmp_path):
        """Extra fields (inputs, outputs, upstream) are included."""
        out = str(tmp_path / "model_out")
        extras = {
            "inputs": {"n_rows": 1000},
            "outputs": {"n_features": 50},
            "upstream": None,
        }
        path = write_provenance(out, {}, "train", extras=extras)

        with open(path) as f:
            doc = json.load(f)

        assert doc["inputs"]["n_rows"] == 1000
        assert doc["outputs"]["n_features"] == 50
        assert doc["upstream"] is None

    def test_creates_output_dir(self, tmp_path):
        """Output directory is created if it doesn't exist."""
        out = str(tmp_path / "nested" / "dir")
        write_provenance(out, {}, "predict")
        assert os.path.isdir(out)


class TestHelpers:
    """Test utility functions."""

    def test_get_version(self):
        """Version is a non-empty string."""
        v = get_version()
        assert isinstance(v, str)
        assert len(v) > 0

    def test_get_git_commit(self):
        """Git commit is a hex string or None."""
        commit = get_git_commit()
        if commit is not None:
            assert len(commit) == 40
            assert all(c in "0123456789abcdef" for c in commit)

    def test_input_checksum(self, tmp_path):
        """Checksum returns a hex string for a small file."""
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        result = input_checksum(str(f))
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256

    def test_input_checksum_missing(self):
        """Returns None for non-existent file."""
        assert input_checksum("/nonexistent/file") is None


class TestFeatureGroupsToExclude:
    """Test positive-to-negative group conversion."""

    def test_known_groups(self):
        """Including a subset excludes the rest."""
        include = ["landsat_bands", "soilgrids"]
        exclude = feature_groups_to_exclude(include)
        assert "landsat_bands" not in exclude
        assert "soilgrids" not in exclude
        assert "sentinel1" in exclude
        assert "worldclim" in exclude

    def test_all_groups_included(self):
        """Including all groups produces an empty exclude list."""
        from swapstress.features.features import FEATURE_GROUPS

        all_groups = [k for k in FEATURE_GROUPS if k != "landsat"]
        exclude = feature_groups_to_exclude(all_groups)
        assert exclude == []
