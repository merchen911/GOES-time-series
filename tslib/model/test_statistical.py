import unittest
from types import SimpleNamespace

import numpy as np

from tslib.model.statistical import STAT_REGISTRY


def _cfg():
    return SimpleNamespace(arima_order=[1, 0, 0], ar_lags=2, pred_len=4)


class TestStatisticalModels(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.series = np.cumsum(rng.normal(size=64)) + 10.0

    def test_registry_has_builtins(self):
        self.assertEqual({"arima", "ar", "theta"} & set(STAT_REGISTRY),
                         {"arima", "ar", "theta"})

    def test_each_model_forecasts_correct_length(self):
        for name in ("arima", "ar", "theta"):
            model = STAT_REGISTRY[name](_cfg()).fit(self.series)
            fc = model.forecast(4)
            self.assertEqual(np.asarray(fc).shape, (4,), msg=name)
            self.assertTrue(np.all(np.isfinite(fc)), msg=name)
