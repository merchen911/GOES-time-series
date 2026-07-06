import unittest

import numpy as np

from tslib.exp.metrics import METRIC_REGISTRY, MetricContext


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


class TestEventMetrics(unittest.TestCase):
    def _data(self):
        # log10 space; threshold physical=10 -> log10=1.0. Values above/below 1.0
        # give tp=1, fp=1, fn=1, tn=1 for a single channel over 4 samples.
        pred = np.array([2.0, 0.0, 2.0, 0.0]).reshape(4, 1, 1)
        true = np.array([2.0, 2.0, 0.0, 0.0]).reshape(4, 1, 1)
        ctx = MetricContext(thresholds=[10.0], transform="log10",
                            target_cols=["p_gt10"])
        return pred, true, ctx

    def test_event_registered_as_event(self):
        for n in ["tss", "hss", "pod", "far"]:
            self.assertIn(n, METRIC_REGISTRY)
            self.assertEqual(METRIC_REGISTRY[n].kind, "event")

    def test_skill_scores_and_per_channel_keys(self):
        pred, true, ctx = self._data()
        tss = METRIC_REGISTRY["tss"].fn(pred, true, ctx)
        hss = METRIC_REGISTRY["hss"].fn(pred, true, ctx)
        pod = METRIC_REGISTRY["pod"].fn(pred, true, ctx)
        far = METRIC_REGISTRY["far"].fn(pred, true, ctx)
        # per-channel dict keyed by target name
        self.assertEqual(set(tss), {"p_gt10"})
        # tp=fp=fn=tn=1 -> pod=.5, pofd=.5, tss=0 ; hss=0 ; far=.5
        self.assertAlmostEqual(tss["p_gt10"], 0.0, places=6)
        self.assertAlmostEqual(hss["p_gt10"], 0.0, places=6)
        self.assertAlmostEqual(pod["p_gt10"], 0.5, places=6)
        self.assertAlmostEqual(far["p_gt10"], 0.5, places=6)

    def test_missing_threshold_errors(self):
        pred = np.zeros((2, 1, 1)); true = np.zeros((2, 1, 1))
        with self.assertRaises(ValueError):
            METRIC_REGISTRY["tss"].fn(pred, true, MetricContext())


class TestRunMetrics(unittest.TestCase):
    def test_flattens_regression_and_event(self):
        import numpy as np
        from tslib.exp.metrics import run_metrics, MetricContext
        pred = np.zeros((4, 2, 1))
        true = np.full((4, 2, 1), 2.0)
        ctx = MetricContext(thresholds=[10.0], transform="log10",
                            target_cols=["p_gt10"])
        out = run_metrics(pred, true, ctx, ["mse", "tss"])
        self.assertIn("mse", out)
        self.assertIn("tss_p_gt10", out)
