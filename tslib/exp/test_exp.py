import math
import unittest

import numpy as np

from tslib.exp.evaluate import Evaluation
from tslib.exp.exp import build_comparison, merge_comparisons
from tslib.exp.lightning_model import TrainResult
from tslib.exp.records import CELL_KEYS, event_columns, regression_columns


def _reg(channel="p_gt10", **values):
    return {"channel": channel, "n": 8, **values}


def _event(channel, granularity, hss):
    return {"channel": channel, "granularity": granularity, "threshold": 10.0, "n": 8,
            "n_events": 2, "tp": 1, "fp": 1, "fn": 1, "tn": 5, "n_nonfinite": 0,
            "hss": hss}


def _result(name, best_val_loss=0.5, strategy="direct", reg=None, event=()):
    reg = reg if reg is not None else [_reg()]
    return TrainResult(name, best_val_loss, Evaluation(regression=list(reg), event=list(event)),
                       f"{name}.pt", strategy=strategy)


class TestBuildComparison(unittest.TestCase):
    def _results(self):
        return [
            _result("lstm", 0.5, reg=[_reg(rmse=0.30, logr=0.7)]),
            _result("timesnet", 0.4, reg=[_reg(rmse=0.20, logr=0.8)]),
        ]

    def test_two_tables_with_record_columns(self):
        reg, ev = build_comparison(self._results(), sort_metric="rmse")
        # a cell's own table has no cell columns
        without_cell = lambda cols: [c for c in cols if c not in CELL_KEYS]
        self.assertEqual(list(reg.columns), without_cell(regression_columns()))
        self.assertEqual(list(ev.columns), without_cell(event_columns()))
        self.assertEqual(len(reg), 2)
        self.assertEqual(len(ev), 0)

    def test_sorts_by_metric(self):
        reg, _ = build_comparison(self._results(), sort_metric="rmse")
        self.assertEqual(list(reg["model"]), ["timesnet", "lstm"])  # 0.20 < 0.30

    def test_logr_sorts_descending(self):
        results = [_result("weak", reg=[_reg(rmse=0.1, logr=0.2)]),
                   _result("strong", reg=[_reg(rmse=0.9, logr=0.9)])]
        reg, _ = build_comparison(results, sort_metric="logr")
        self.assertEqual(list(reg["model"]), ["strong", "weak"])

    def test_fallback_when_metric_absent(self):
        reg, _ = build_comparison(self._results(), sort_metric="nope")
        self.assertEqual(list(reg["model"]), ["timesnet", "lstm"])  # by best_val_loss

    def test_event_table_by_channel_and_granularity_then_best_hss(self):
        results = [
            _result("a", event=[_event("xrs_long", "point", 0.1), _event("p_gt10", "window", 0.2),
                                _event("p_gt10", "point", 0.3)]),
            _result("b", event=[_event("xrs_long", "point", 0.4), _event("p_gt10", "window", 0.5),
                                _event("p_gt10", "point", float("nan"))]),
        ]
        _, ev = build_comparison(results, sort_metric="rmse")
        self.assertEqual(list(zip(ev["channel"], ev["granularity"], ev["model"])),
                         [("p_gt10", "point", "a"), ("p_gt10", "point", "b"),
                          ("p_gt10", "window", "b"), ("p_gt10", "window", "a"),
                          ("xrs_long", "point", "b"), ("xrs_long", "point", "a")])


class TestStrategyColumn(unittest.TestCase):
    def test_strategy_column_present(self):
        reg, _ = build_comparison([_result("lstm", reg=[_reg(rmse=0.3)])], sort_metric="rmse")
        self.assertEqual(reg.loc[0, "strategy"], "direct")

    def test_merge_across_strategies_nan_last(self):
        a, _ = build_comparison([_result("lstm", 0.5, strategy="recursive")],
                                sort_metric="best_val_loss")
        b, _ = build_comparison([_result("arima", float("nan"), strategy="statistic")],
                                sort_metric="best_val_loss")
        merged = merge_comparisons([a, b], sort_metric="best_val_loss")
        self.assertEqual(list(merged["model"]), ["lstm", "arima"])  # NaN last
        self.assertTrue(math.isnan(merged.iloc[-1]["best_val_loss"]))


class TestFailedModelColumns(unittest.TestCase):
    def test_nan_row_present_and_last(self):
        # a diverged model has NaN metrics; it stays in the table, at the bottom
        res = [_result("diverged", 0.1, reg=[_reg(rmse=float("nan"))]),
               _result("lstm", 0.3, reg=[_reg(rmse=0.2)])]
        reg, _ = build_comparison(res, sort_metric="rmse")
        self.assertEqual(list(reg["model"]), ["lstm", "diverged"])
        self.assertTrue(np.isnan(reg.iloc[-1]["rmse"]))


if __name__ == "__main__":
    unittest.main()
