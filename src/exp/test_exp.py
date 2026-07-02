import unittest

from exp.exp import build_comparison
from exp.lightning_model import TrainResult


class TestBuildComparison(unittest.TestCase):
    def _results(self):
        return [
            TrainResult("lstm", 0.5, {"mse": 0.30, "tss_p_gt10": 0.7}, "a.pt"),
            TrainResult("timesnet", 0.4, {"mse": 0.20, "tss_p_gt10": 0.8}, "b.pt"),
        ]

    def test_dynamic_columns(self):
        df = build_comparison(self._results(), sort_metric="mse")
        self.assertEqual(set(df.columns),
                         {"model", "best_val_loss", "mse", "tss_p_gt10"})

    def test_sorts_by_metric(self):
        df = build_comparison(self._results(), sort_metric="mse")
        self.assertEqual(list(df["model"]), ["timesnet", "lstm"])  # 0.20 < 0.30

    def test_fallback_when_metric_absent(self):
        df = build_comparison(self._results(), sort_metric="nope")
        self.assertEqual(list(df["model"]), ["timesnet", "lstm"])  # by best_val_loss
