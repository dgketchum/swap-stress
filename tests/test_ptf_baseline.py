"""Tests for the POLARIS alpha unit handling in the PTF baseline.

POLARIS distributes van Genuchten alpha as log10(kPa^-1) -- the readme
correction of 2019-06-02, not the log10(cm^-1) its early documentation
claimed -- while the pipeline composes the retention equation in cm. These
lock down the kPa->cm factor and the one-time correction applied to cached
prep outputs, which are never resampled from EE.
"""

import math

import numpy as np
import pandas as pd
import pytest

from swapstress.units import KPA_TO_CM, MPA_TO_CM
from swapstress.validation.ptf_baseline import (
    POL_ALPHA_UNITS_COL,
    fix_polaris_units_frame,
)


class TestKpaToCm:
    def test_derived_from_mpa_constant(self):
        assert KPA_TO_CM == MPA_TO_CM / 1000.0

    def test_value(self):
        assert KPA_TO_CM == pytest.approx(10.19716, abs=1e-5)

    def test_log10_shift_is_the_reported_bias_artifact(self):
        """Omitting the conversion understates suction by this much."""
        assert math.log10(KPA_TO_CM) == pytest.approx(1.0085, abs=1e-4)


class TestFixPolarisUnitsFrame:
    @pytest.fixture
    def prefix_table(self):
        """A prep output as written before the fix: pol_alpha in 1/kPa."""
        return pd.DataFrame(
            {
                "sample_id": ["a", "b", "c"],
                "ros_alpha": [0.02, 0.03, np.nan],
                "pol_alpha": [0.2, np.nan, 0.15],
                "pol_n": [1.4, 1.5, 1.6],
            }
        )

    def test_divides_pol_alpha_by_kpa_to_cm(self, prefix_table):
        fixed = fix_polaris_units_frame(prefix_table)
        expected = prefix_table["pol_alpha"] / KPA_TO_CM
        pd.testing.assert_series_equal(fixed["pol_alpha"], expected)

    def test_rosetta_alpha_untouched(self, prefix_table):
        """Rosetta grids ship log10(cm^-1); no conversion applies to them."""
        fixed = fix_polaris_units_frame(prefix_table)
        pd.testing.assert_series_equal(fixed["ros_alpha"], prefix_table["ros_alpha"])

    def test_stamps_units_marker(self, prefix_table):
        fixed = fix_polaris_units_frame(prefix_table)
        assert (fixed[POL_ALPHA_UNITS_COL] == "cm^-1").all()

    def test_refuses_double_application(self, prefix_table):
        fixed = fix_polaris_units_frame(prefix_table)
        with pytest.raises(ValueError, match="must not be divided again"):
            fix_polaris_units_frame(fixed)

    def test_input_frame_not_mutated(self, prefix_table):
        original = prefix_table.copy()
        fix_polaris_units_frame(prefix_table)
        pd.testing.assert_frame_equal(prefix_table, original)
