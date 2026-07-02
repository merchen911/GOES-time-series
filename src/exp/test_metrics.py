import unittest

import numpy as np

from exp.metrics import METRIC_REGISTRY, MetricContext


class TestRegressionMetrics(unittest.TestCase):
    def test_builtins_registered(self):
        for n in ["mse", "mae", "rmse"]:
            self.assertIn(n, METRIC_REGISTRY)
            self.assertEqual(METRIC_REGISTRY[n].kind, "regression")

    def test_regression_values(self):
        pred = np.zeros((4, 2, 1))
        true = np.ones((4, 2, 1))
        ctx = MetricContext()
        self.assertAlmostEqual(METRIC_REGISTRY["mse"].fn(pred, true, ctx), 1.0)
        self.assertAlmostEqual(METRIC_REGISTRY["mae"].fn(pred, true, ctx), 1.0)
        self.assertAlmostEqual(METRIC_REGISTRY["rmse"].fn(pred, true, ctx), 1.0)
