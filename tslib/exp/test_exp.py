import math
import unittest

import pandas as pd

from tslib.exp.exp import build_comparison, merge_comparisons
from tslib.exp.lightning_model import TrainResult


class TestBuildComparison(unittest.TestCase):
    def _results(self):
        return [
            TrainResult("lstm", 0.5, {"mse": 0.30, "tss_p_gt10": 0.7}, "a.pt"),
            TrainResult("timesnet", 0.4, {"mse": 0.20, "tss_p_gt10": 0.8}, "b.pt"),
        ]

    def test_dynamic_columns(self):
        df = build_comparison(self._results(), sort_metric="mse")
        self.assertEqual(set(df.columns),
                         {"strategy", "model", "best_val_loss", "mse", "tss_p_gt10"})

    def test_sorts_by_metric(self):
        df = build_comparison(self._results(), sort_metric="mse")
        self.assertEqual(list(df["model"]), ["timesnet", "lstm"])  # 0.20 < 0.30

    def test_fallback_when_metric_absent(self):
        df = build_comparison(self._results(), sort_metric="nope")
        self.assertEqual(list(df["model"]), ["timesnet", "lstm"])  # by best_val_loss


class TestStrategyColumn(unittest.TestCase):
    def test_strategy_column_present(self):
        from tslib.exp.exp import build_comparison
        res = [TrainResult("lstm", 0.5, {"mse": 0.3}, "a.pt", strategy="direct")]
        df = build_comparison(res, sort_metric="mse")
        self.assertEqual(set(df.columns), {"strategy", "model", "best_val_loss", "mse"})
        self.assertEqual(df.loc[0, "strategy"], "direct")

    def test_merge_across_strategies_nan_last(self):
        from tslib.exp.exp import build_comparison
        a = build_comparison(
            [TrainResult("lstm", 0.5, {"mse": 0.3}, "a.pt", strategy="recursive")],
            sort_metric="best_val_loss")
        b = build_comparison(
            [TrainResult("arima", float("nan"), {"mse": 0.9}, "", strategy="statistic")],
            sort_metric="best_val_loss")
        merged = merge_comparisons([a, b], sort_metric="best_val_loss")
        self.assertEqual(list(merged["model"]), ["lstm", "arima"])  # NaN last
        self.assertTrue(math.isnan(merged.iloc[-1]["best_val_loss"]))
