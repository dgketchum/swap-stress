"""Unit tests for Figure 5 applicability and common-subset logic."""

import numpy as np
import pandas as pd
import pytest

from swapstress.figures.fig05_ptf_comparison import (
    applicability_counts,
    common_strict_mask,
    common_subset_metrics,
    metrics,
)


@pytest.fixture
def comparison_rows():
    """Small table spanning every panel-a applicability category."""
    return pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(6)],
            "source": ["test"] * 6,
            "log10_suction_cm": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "rf_pred": [0.0, 2.0, 2.0, 3.0, 4.0, 5.0],
            "ros_log10_suction": [1.0, 1.0, 8.0, -2.0, np.nan, np.nan],
            "pol_depth_matched_log10_suction_capped": [
                0.0,
                1.0,
                2.0,
                8.0,
                -2.0,
                np.nan,
            ],
            "rosetta_domain_status": [
                "in_domain",
                "in_domain",
                "below_or_equal_theta_r",
                "above_or_equal_theta_s",
                "missing_parameters",
                "invalid_parameters",
            ],
            "polaris_domain_status": [
                "in_domain",
                "in_domain",
                "in_domain",
                "below_or_equal_theta_r",
                "above_or_equal_theta_s",
                "missing_parameters",
            ],
        }
    )


def test_applicability_categories_are_exhaustive(comparison_rows):
    counts = applicability_counts(comparison_rows)

    assert counts.loc["SWAP direct"].to_dict() == {
        "evaluable": 6,
        "below": 0,
        "above": 0,
        "unavailable": 0,
    }
    assert counts.loc["Rosetta"].to_dict() == {
        "evaluable": 2,
        "below": 1,
        "above": 1,
        "unavailable": 2,
    }
    assert counts.loc["POLARIS"].to_dict() == {
        "evaluable": 3,
        "below": 1,
        "above": 1,
        "unavailable": 1,
    }
    assert (counts.sum(axis=1) == len(comparison_rows)).all()


def test_common_strict_mask_uses_identical_rows(comparison_rows):
    mask = common_strict_mask(comparison_rows)
    assert mask.tolist() == [True, True, False, False, False, False]


def test_common_subset_metrics_share_one_denominator(comparison_rows):
    result = common_subset_metrics(comparison_rows)

    assert result["n"].tolist() == [2, 2, 2]
    assert result.loc["SWAP direct", "rmse"] == pytest.approx(np.sqrt(0.5))
    assert result.loc["SWAP direct", "mae"] == pytest.approx(0.5)
    assert result.loc["Rosetta", "rmse"] == pytest.approx(np.sqrt(0.5))
    assert result.loc["Rosetta", "mae"] == pytest.approx(0.5)
    assert result.loc["POLARIS", "rmse"] == pytest.approx(0.0)
    assert result.loc["POLARIS", "mae"] == pytest.approx(0.0)


def test_metrics_excludes_nonfinite_pairs():
    result = metrics(
        observed=np.array([0.0, 1.0, np.nan]),
        predicted=np.array([0.0, 2.0, 9.0]),
    )
    assert result["n"] == 2
    assert result["rmse"] == pytest.approx(np.sqrt(0.5))
    assert result["mae"] == pytest.approx(0.5)


def test_metrics_rejects_empty_input():
    with pytest.raises(ValueError, match="without finite pairs"):
        metrics(observed=np.array([np.nan]), predicted=np.array([np.nan]))
