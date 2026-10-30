import math
import unittest

import numpy as np

from tslib.exp.metrics import (
    EVENT_SCORES,
    REGRESSION_METRICS,
    MetricContext,
    metric_kind,
    register_event_score,
    register_regression_metric,
    to_stored_space,
)


class TestEventScores(unittest.TestCase):
    T = (20, 10, 5, 65)   # tp, fp, fn, tn

    def test_values(self):
        exp = {"pod": 0.8, "far": 1/3, "tss": 0.6666666666666667, "hss": 0.625,
               "csi": 0.5714285714285714, "f1": 0.7272727272727273,
               "seds": 0.609423479734194, "sedi": 0.8164457768180353}
        for k, v in exp.items():
            self.assertAlmostEqual(EVENT_SCORES[k](*self.T), v, places=12, msg=k)

    def test_nan_rules(self):
        nan = lambda k, t: self.assertTrue(math.isnan(EVENT_SCORES[k](*t)), (k, t))
        nan("pod", (0, 3, 0, 7)); nan("tss", (0, 3, 0, 7)); nan("tss", (2, 0, 1, 0))
        nan("far", (0, 0, 4, 6)); nan("csi", (0, 0, 0, 9)); nan("f1", (0, 0, 0, 9))
        nan("hss", (0, 0, 0, 9)); nan("seds", (0, 3, 2, 5)); nan("seds", (9, 0, 0, 0))
        nan("sedi", (5, 0, 1, 9)); nan("sedi", (5, 1, 0, 9))      # F=0, H=1

    def test_csi_zero_when_only_false_alarms(self):
        self.assertEqual(EVENT_SCORES["csi"](0, 3, 0, 7), 0.0)

    def test_names_without_underscore(self):
        with self.assertRaises(ValueError):
            register_event_score("my_score")

    def test_order(self):
        self.assertEqual(list(EVENT_SCORES),
                         ["tss", "hss", "pod", "far", "csi", "f1", "seds", "sedi"])

    def test_returns_plain_float_and_never_raises(self):
        # every cell of a small count grid, including all the undefined corners
        for name, fn in EVENT_SCORES.items():
            for t in np.ndindex(3, 3, 3, 3):
                v = fn(*t)
                self.assertIsInstance(v, float, (name, t))

    def test_numpy_integer_counts(self):
        t = tuple(np.int64(x) for x in self.T)
        for name, fn in EVENT_SCORES.items():
            self.assertAlmostEqual(fn(*t), fn(*self.T), places=12, msg=name)

    def test_register_custom_score(self):
        @register_event_score("tmpscore")
        def _tmp(tp, fp, fn, tn):
            return float(tp)
        self.addCleanup(EVENT_SCORES.pop, "tmpscore")
        self.assertIs(EVENT_SCORES["tmpscore"], _tmp)
        self.assertEqual(metric_kind("tmpscore"), "event")


class TestRegressionMetrics(unittest.TestCase):
    def test_values(self):
        t = np.zeros(4); p = np.array([1.0, -1.0, 0.5, 0.0])
        R = REGRESSION_METRICS
        self.assertAlmostEqual(R["rmse"](p, t), 0.75)
        self.assertAlmostEqual(R["mae"](p, t), 0.625)
        self.assertAlmostEqual(R["msa"](p, t), 462.3413251903491, places=9)
        self.assertAlmostEqual(R["sspb"](p, t), 77.82794100389228, places=9)
        self.assertAlmostEqual(R["sspb"](np.array([-1, -1, -0.5, 0.0]), t), -462.3413251903491, places=9)
        self.assertAlmostEqual(R["logr"](np.array([0.1, 0.9, 2.2, 2.8]), np.array([0, 1, 2, 3.0])),
                               0.9908470001860923, places=12)

    def test_logr_constant_is_nan(self):
        self.assertTrue(math.isnan(REGRESSION_METRICS["logr"](np.ones(5), np.arange(5.0))))

    def test_logr_constant_with_inexact_mean_is_nan(self):
        # mean of 0.1 repeated need not equal 0.1 exactly; a constant is still constant
        const = np.full(1_000_003, 0.1)
        self.assertTrue(math.isnan(REGRESSION_METRICS["logr"](const, np.arange(const.size, dtype=float))))
        self.assertTrue(math.isnan(REGRESSION_METRICS["logr"](np.arange(const.size, dtype=float), const)))

    def test_logr_exact_in_float64_for_float32_input(self):
        x = np.arange(1000, dtype=np.float32) + np.float32(1e6)
        y = (2 * x).astype(np.float32)
        self.assertAlmostEqual(REGRESSION_METRICS["logr"](x, y), 1.0, places=12)

    def test_empty_is_nan_not_error(self):
        e = np.array([])
        for name, fn in REGRESSION_METRICS.items():
            self.assertTrue(math.isnan(fn(e, e)), name)

    def test_returns_plain_float(self):
        p = np.array([1.0, -1.0, 0.5, 0.0], dtype=np.float32); t = np.zeros(4, dtype=np.float32)
        for name, fn in REGRESSION_METRICS.items():
            self.assertIsInstance(fn(p, t), float, name)

    def test_order(self):
        self.assertEqual(list(REGRESSION_METRICS), ["rmse", "mae", "msa", "sspb", "logr"])

    def test_names_without_underscore(self):
        with self.assertRaises(ValueError):
            register_regression_metric("my_metric")

    def test_register_custom_metric(self):
        @register_regression_metric("tmpmetric")
        def _tmp(pred, true):
            return 0.0
        self.addCleanup(REGRESSION_METRICS.pop, "tmpmetric")
        self.assertEqual(metric_kind("tmpmetric"), "regression")

    def test_kinds(self):
        self.assertEqual(metric_kind("sedi"), "event"); self.assertEqual(metric_kind("msa"), "regression")

    def test_unknown_kind_is_keyerror(self):
        with self.assertRaises(KeyError):
            metric_kind("nope")


class TestContextAndThresholds(unittest.TestCase):
    def test_context_defaults(self):
        a, b = MetricContext(), MetricContext()
        self.assertIsNone(a.thresholds)
        self.assertIsNone(a.forecast_start)
        self.assertEqual(a.transform, "none")
        a.target_cols.append("x")
        self.assertEqual(b.target_cols, [])      # no shared mutable default

    def test_context_forecast_start(self):
        fs = np.array([0, 5 * 60 * 10**9], dtype=np.int64)
        self.assertIs(MetricContext(forecast_start=fs).forecast_start, fs)

    def test_to_stored_space(self):
        self.assertEqual(to_stored_space(100.0, "log10"), 2.0)
        self.assertEqual(to_stored_space(100.0, "none"), 100.0)
        with self.assertRaises(ValueError):
            to_stored_space(100.0, "ln")
