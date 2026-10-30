import unittest

import numpy as np

from tslib.exp.granularity import (
    DAY_HOURS,
    GRANULARITIES,
    as_ns_int64,
    day_indices,
    event_masks,
)

_HOUR_NS = 3600 * 10**9


def _ns(hours):
    return (np.asarray(hours, dtype=float) * 3600e9).astype(np.int64)


class TestConstants(unittest.TestCase):
    def test_values(self):
        self.assertEqual(GRANULARITIES, ("point", "window", "day"))
        self.assertEqual(DAY_HOURS, 24)


class TestDayIndices(unittest.TestCase):
    def test_day_cut_greedy(self):
        h = np.array([0, 1, 23, 24, 25, 60, 84, 85, 108.5])
        ns = (h * 3600e9).astype(np.int64)
        np.testing.assert_array_equal(day_indices(ns), [0, 3, 5, 6, 8])

    def test_day_cut_unsorted_input(self):
        ns = (np.array([24, 0, 60]) * 3600e9).astype(np.int64)
        np.testing.assert_array_equal(day_indices(ns), [1, 0, 2])

    def test_gap_takes_first_window_after_it(self):
        # 5-minute windows with a 30-hour hole: the first window after the hole
        # is taken at once, then the 24 h spacing restarts from it
        minutes = np.concatenate([np.arange(0, 30 * 60, 5), np.arange(60 * 60, 90 * 60, 5)])
        ns = minutes.astype(np.int64) * 60 * 10**9
        picked = day_indices(ns)
        np.testing.assert_array_equal(minutes[picked] // 60, [0, 24, 60, 84])
        self.assertGreaterEqual(np.diff(ns[picked]).min(), 24 * _HOUR_NS)

    def test_exactly_one_day_later_is_taken(self):
        ns = np.array([0, 24 * _HOUR_NS - 1, 24 * _HOUR_NS], dtype=np.int64)
        np.testing.assert_array_equal(day_indices(ns), [0, 2])

    def test_equal_times_keep_first_position(self):
        ns = _ns([24, 0, 0, 24])
        np.testing.assert_array_equal(day_indices(ns), [1, 0])

    def test_custom_hours(self):
        np.testing.assert_array_equal(day_indices(_ns([0, 3, 6, 7, 12]), hours=6), [0, 2, 4])

    def test_empty_and_single(self):
        out = day_indices(np.array([], dtype=np.int64))
        self.assertEqual(out.shape, (0,))
        self.assertTrue(np.issubdtype(out.dtype, np.integer))
        np.testing.assert_array_equal(day_indices(_ns([5])), [0])

    def test_nonpositive_hours_rejected(self):
        with self.assertRaises(ValueError):
            day_indices(_ns([0, 1]), hours=0)


def _five_days_5min():
    import pandas as pd
    # pandas >= 3 builds this index as datetime64[us]; .values would be microseconds
    return pd.date_range("2020-01-01", periods=5 * 288, freq="5min")


class TestTimeUnits(unittest.TestCase):
    """forecast_start is int64 nanoseconds; a datetime64 of another unit is
    converted, never read as a nanosecond count (spec 4.4)."""

    def test_datetime64_us_gives_the_same_cut_as_ns(self):
        idx = _five_days_5min()
        ns = idx.as_unit("ns").asi8
        us = idx.as_unit("us").values
        self.assertEqual(us.dtype, np.dtype("datetime64[us]"))
        self.assertEqual(len(day_indices(ns)), 5)
        np.testing.assert_array_equal(day_indices(us), day_indices(ns))
        np.testing.assert_array_equal(as_ns_int64(us), ns)

    def test_pandas_times_are_converted(self):
        idx = _five_days_5min()
        ns = idx.as_unit("ns").asi8
        np.testing.assert_array_equal(as_ns_int64(idx.as_unit("us")), ns)
        np.testing.assert_array_equal(as_ns_int64(idx.tz_localize("UTC").as_unit("us")), ns)
        import pandas as pd
        np.testing.assert_array_equal(as_ns_int64(pd.Series(idx.as_unit("s"))), ns)

    def test_integers_are_taken_as_nanoseconds(self):
        out = as_ns_int64(np.array([1, 2], dtype=np.int32))
        self.assertEqual(out.dtype, np.int64)
        np.testing.assert_array_equal(out, [1, 2])

    def test_floats_and_other_types_are_rejected(self):
        for bad in (np.array([0.0, 3.6e12]), np.array(["2020-01-01"]), np.array([True])):
            with self.assertRaisesRegex(TypeError, "forecast_start", msg=str(bad.dtype)):
                as_ns_int64(bad)
            with self.assertRaises(TypeError):
                day_indices(bad)


class TestEventMasks(unittest.TestCase):
    def test_window_uses_horizon_max(self):
        p = np.array([[0.0, 2.0], [0.0, 0.5]]); t = np.array([[0.0, 0.0], [3.0, 0.0]])
        pm, tm = event_masks(p, t, 1.0, "window")
        np.testing.assert_array_equal(pm, [True, False]); np.testing.assert_array_equal(tm, [False, True])

    def test_threshold_inclusive(self):
        pm, _ = event_masks(np.array([[1.0]]), np.array([[0.0]]), 1.0, "point"); self.assertTrue(pm.all())

    def test_point_is_elementwise(self):
        p = np.array([[0.0, 2.0], [1.0, 0.5]]); t = np.array([[3.0, 0.0], [0.0, 1.0]])
        pm, tm = event_masks(p, t, 1.0, "point")
        np.testing.assert_array_equal(pm, [[False, True], [True, False]])
        np.testing.assert_array_equal(tm, [[True, False], [False, True]])

    def test_day_judges_selected_rows_like_window(self):
        p = np.array([[0.0, 2.0], [0.0, 0.0], [5.0, 0.0]])
        t = np.array([[0.0, 0.0], [9.0, 9.0], [0.0, 1.0]])
        pm, tm = event_masks(p, t, 1.0, "day", day_idx=np.array([2, 0]))
        np.testing.assert_array_equal(pm, [True, True])
        np.testing.assert_array_equal(tm, [True, False])

    def test_day_without_indices_rejected(self):
        with self.assertRaises(ValueError):
            event_masks(np.zeros((2, 2)), np.zeros((2, 2)), 1.0, "day")

    def test_unknown_granularity_rejected(self):
        with self.assertRaises(ValueError):
            event_masks(np.zeros((2, 2)), np.zeros((2, 2)), 1.0, "hour")


if __name__ == "__main__":
    unittest.main()
