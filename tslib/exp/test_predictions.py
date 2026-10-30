import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from tslib.exp.predictions import (
    PRED_FORMAT,
    load_predictions,
    prediction_path,
    save_predictions,
)


def _arrays(n=3, p=2, t=2):
    rng = np.random.default_rng(0)
    pred = rng.normal(size=(n, p, t)).astype(np.float32)
    true = rng.normal(size=(n, p, t)).astype(np.float32)
    fs = np.arange(n, dtype=np.int64) * 3_600_000_000_000
    return pred, true, fs


class TestPredictionPath(unittest.TestCase):
    def test_layout(self):
        self.assertEqual(prediction_path("runs/r", "dlinear"),
                         os.path.join("runs/r", "pred", "dlinear.npz"))


class TestRoundtrip(unittest.TestCase):
    def test_roundtrip(self):
        pred, true, fs = _arrays()
        meta = {"target_cols": ["a", "b"], "transform": "log10",
                "cadence_min": 5, "event_threshold": [10.0, 100.0]}
        with tempfile.TemporaryDirectory() as d:
            # the pred/ subfolder does not exist yet; save creates it
            path = prediction_path(d, "m")
            save_predictions(path, pred, true, fs, meta)
            p2, t2, f2, m2 = load_predictions(path)
        self.assertEqual(p2.dtype, np.float32)
        self.assertEqual(t2.dtype, np.float32)
        self.assertEqual(f2.dtype, np.int64)
        self.assertEqual(p2.shape, pred.shape)
        self.assertEqual(t2.shape, true.shape)
        self.assertEqual(f2.shape, fs.shape)
        np.testing.assert_array_equal(p2, pred)
        np.testing.assert_array_equal(t2, true)
        np.testing.assert_array_equal(f2, fs)
        self.assertEqual(m2.pop("format"), PRED_FORMAT)
        self.assertEqual(m2, meta)

    def test_inputs_are_cast_and_meta_not_mutated(self):
        pred, true, fs = _arrays()
        meta = {"fold": 1}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.npz")
            save_predictions(path, pred.astype(np.float64),
                             true.astype(np.float64), fs.astype(np.int32), meta)
            p2, t2, f2, _ = load_predictions(path)
        self.assertEqual((p2.dtype, t2.dtype, f2.dtype),
                         (np.float32, np.float32, np.int64))
        self.assertEqual(meta, {"fold": 1})

    def test_numpy_scalars_in_meta(self):
        pred, true, fs = _arrays()
        meta = {"pred_len": np.int64(2), "thr": np.array([1.0, 2.0])}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.npz")
            save_predictions(path, pred, true, fs, meta)
            m2 = load_predictions(path)[3]
        self.assertEqual(m2["pred_len"], 2)
        self.assertEqual(m2["thr"], [1.0, 2.0])


class TestValidation(unittest.TestCase):
    def _assert_raises_and_nothing_written(self, pred, true, fs):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.npz")
            with self.assertRaises(ValueError):
                save_predictions(path, pred, true, fs, {})
            self.assertEqual(os.listdir(d), [])

    def test_shape_mismatch_raises(self):
        fs = np.zeros(3, dtype=np.int64)
        self._assert_raises_and_nothing_written(
            np.zeros((3, 2, 1), np.float32), np.zeros((3, 2, 2), np.float32), fs)

    def test_forecast_start_length_mismatch_raises(self):
        self._assert_raises_and_nothing_written(
            np.zeros((3, 2, 1), np.float32), np.zeros((3, 2, 1), np.float32),
            np.zeros(4, dtype=np.int64))

    def test_not_three_dimensional_raises(self):
        self._assert_raises_and_nothing_written(
            np.zeros((3, 2), np.float32), np.zeros((3, 2), np.float32),
            np.zeros(3, dtype=np.int64))


class TestForecastStartUnit(unittest.TestCase):
    def test_datetime64_us_is_stored_as_ns(self):
        import pandas as pd
        pred, true, _ = _arrays()
        idx = pd.date_range("2020-01-01", periods=3, freq="h")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.npz")
            save_predictions(path, pred, true, idx.as_unit("us").values, {})
            f2 = load_predictions(path)[2]
        self.assertEqual(f2.dtype, np.int64)
        np.testing.assert_array_equal(f2, idx.as_unit("ns").asi8)
        self.assertEqual(int(f2[0]), 1577836800 * 10**9)

    def test_float_times_are_rejected_and_nothing_written(self):
        pred, true, fs = _arrays()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.npz")
            with self.assertRaisesRegex(TypeError, "forecast_start"):
                save_predictions(path, pred, true, fs.astype(float), {})
            self.assertEqual(os.listdir(d), [])


class TestAtomicity(unittest.TestCase):
    @staticmethod
    def _fail_midway(fh, **arrays):
        fh.write(b"partial")   # leave a half-written tmp behind, then die
        raise OSError("disk full")

    def test_no_partial_file_on_failure(self):
        pred, true, fs = _arrays()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.npz")
            with mock.patch("tslib.exp.predictions.np.savez",
                            side_effect=self._fail_midway):
                with self.assertRaises(OSError):
                    save_predictions(path, pred, true, fs, {})
            self.assertFalse(os.path.exists(path))
            self.assertFalse(os.path.exists(path + ".tmp"))

    def test_failure_keeps_previous_file(self):
        pred, true, fs = _arrays()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.npz")
            save_predictions(path, pred, true, fs, {"v": 1})
            with mock.patch("tslib.exp.predictions.np.savez",
                            side_effect=self._fail_midway):
                with self.assertRaises(OSError):
                    save_predictions(path, pred + 1, true, fs, {"v": 2})
            self.assertFalse(os.path.exists(path + ".tmp"))
            p2, _, _, m2 = load_predictions(path)
        np.testing.assert_array_equal(p2, pred)
        self.assertEqual(m2["v"], 1)

    def test_success_leaves_no_tmp(self):
        pred, true, fs = _arrays()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.npz")
            save_predictions(path, pred, true, fs, {})
            self.assertEqual(os.listdir(d), ["m.npz"])


if __name__ == "__main__":
    unittest.main()
