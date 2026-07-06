import unittest
from types import SimpleNamespace

import numpy as np

from tslib.exp.strategy import run_strategy, StatisticalRunner


class TestRunStrategyRouting(unittest.TestCase):
    def test_unknown_strategy_rejected(self):
        with self.assertRaises(ValueError):
            run_strategy("bogus", "lstm", SimpleNamespace(),
                         SimpleNamespace(), "ckpt.pt")


def _stat_cfg():
    return SimpleNamespace(pred_len=4, arima_order=[1, 0, 0], ar_lags=2,
                           metrics=["mse"], transform="log10",
                           event_threshold=None)


def _fake_bundle():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(size=(3, 24, 1)), axis=1) + 5.0
    y = np.cumsum(rng.normal(size=(3, 4, 1)), axis=1) + 5.0
    bundle = SimpleNamespace(test_loader=[(x, y)], target_cols=["p_gt10"])
    return bundle


class TestStatisticalRunner(unittest.TestCase):
    def test_produces_result_with_nan_val_loss(self):
        runner = StatisticalRunner("ar", _stat_cfg())
        res = runner.fit_and_test(_fake_bundle(), "ar", "unused.pt")
        self.assertEqual(res.strategy, "statistic")
        self.assertTrue(np.isnan(res.best_val_loss))
        self.assertIn("mse", res.metrics)

    def test_dispatcher_routes_statistic(self):
        res = run_strategy("statistic", "ar", _fake_bundle(), _stat_cfg(),
                           "unused.pt")
        self.assertEqual(res.strategy, "statistic")
