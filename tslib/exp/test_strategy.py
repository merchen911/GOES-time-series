import unittest
from types import SimpleNamespace

from tslib.exp.strategy import run_strategy


class TestRunStrategyRouting(unittest.TestCase):
    def test_statistic_not_available_phase1(self):
        with self.assertRaises(ValueError):
            run_strategy("statistic", "arima", SimpleNamespace(),
                         SimpleNamespace(), "ckpt.pt")

    def test_unknown_strategy_rejected(self):
        with self.assertRaises(ValueError):
            run_strategy("bogus", "lstm", SimpleNamespace(),
                         SimpleNamespace(), "ckpt.pt")
