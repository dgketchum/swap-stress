"""Tests for the per-pixel gap-filling rule.

``interpolate_pixel`` is the single definition of what Level 2 means: stage 06
applies it to every pixel of the grid, and Fig 3 draws it at three of them. The
behaviour pinned here is therefore the product's, not one caller's.
"""

import numpy as np

from swapstress.inference.gapfill import NODATA_VALUE, interpolate_pixel


def series(*values):
    """A pixel's Level 1 record, with None for a day carrying no retrieval."""
    return np.array(
        [np.nan if v is None else v for v in values],
        dtype=np.float32,
    )


class TestInterpolatePixel:
    def test_interpolates_linearly_between_observations(self):
        day = np.arange(5, dtype=np.float32)
        out = interpolate_pixel(day, series(1.0, None, None, None, 5.0), day)
        assert np.allclose(out, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_observed_days_are_returned_unchanged(self):
        day = np.arange(4, dtype=np.float32)
        values = series(2.0, None, 3.5, 4.0)
        out = interpolate_pixel(day, values, day)
        observed = ~np.isnan(values)
        assert np.allclose(out[observed], values[observed])

    def test_leading_and_trailing_gaps_are_held_flat(self):
        # np.interp clamps rather than extrapolating, so the ends of the record
        # repeat the first and last retrieval. Fig 3 shades exactly this region.
        day = np.arange(6, dtype=np.float32)
        out = interpolate_pixel(day, series(None, None, 2.0, 3.0, None, None), day)
        assert np.allclose(out, [2.0, 2.0, 2.0, 3.0, 3.0, 3.0])

    def test_a_pixel_never_observed_stays_nodata(self):
        # Ocean and permanently masked pixels must not be invented.
        day = np.arange(3, dtype=np.float32)
        out = interpolate_pixel(day, series(None, None, None), day)
        assert np.all(out == NODATA_VALUE)

    def test_a_single_observation_fills_the_whole_span_flat(self):
        day = np.arange(4, dtype=np.float32)
        out = interpolate_pixel(day, series(None, 2.5, None, None), day)
        assert np.allclose(out, 2.5)

    def test_nodata_is_never_the_result_of_interpolation(self):
        # -9999 surviving into a filled series would be read as a real suction
        # by anything that later exponentiates it.
        day = np.arange(5, dtype=np.float32)
        out = interpolate_pixel(day, series(1.0, None, None, None, 5.0), day)
        assert not np.any(out == NODATA_VALUE)

    def test_write_days_need_not_be_the_observation_days(self):
        # The stage writes every calendar day in the span, including days with
        # no source raster at all.
        day = np.array([0.0, 4.0], dtype=np.float32)
        out = interpolate_pixel(day, series(1.0, 5.0), np.arange(5, dtype=np.float32))
        assert np.allclose(out, [1.0, 2.0, 3.0, 4.0, 5.0])
