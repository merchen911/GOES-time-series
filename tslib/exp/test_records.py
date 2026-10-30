import json
import unittest

import numpy as np
import pandas as pd

from tslib.exp.evaluate import Evaluation, evaluate
from tslib.exp.lightning_model import TrainResult
from tslib.exp.metrics import (
    EVENT_SCORES,
    REGRESSION_METRICS,
    MetricContext,
    register_event_score,
    register_regression_metric,
)
from tslib.exp.records import (
    RECORD_FORMAT,
    RUN_KEYS,
    event_columns,
    event_frame,
    regression_columns,
    regression_frame,
    result_record,
)

_HOUR_NS = 3600 * 10**9
_CELL = {"track": "uni_b", "seq_len": 288, "pred_len": 36, "fold": 0}
_CELL_KEYS = ["track", "seq_len", "pred_len", "fold"]


def _evaluation(metric_names=None, nan_channel=False):
    """Two channels, N=4 daily windows, P=1, log10; thresholds stored [1, -5].
    `nan_channel` makes channel 1 non-finite so its counts and scores are nan."""
    pred = np.stack([[2.0, 0.0, 2.0, 0.0], [-4.0, -6.0, -4.0, -6.0]], axis=-1)[:, None, :]
    true = np.stack([[2.0, 2.0, 0.0, 0.0], [-4.0, -4.0, -6.0, -6.0]], axis=-1)[:, None, :]
    if nan_channel:
        pred[1, 0, 1] = np.nan
    ctx = MetricContext(thresholds=[10.0, 1e-5], transform="log10",
                        target_cols=["p_gt10", "xrs_long"],
                        forecast_start=np.arange(4, dtype=np.int64) * 24 * _HOUR_NS)
    return evaluate(pred, true, ctx, metric_names)


def _result(model="dlinear", strategy="direct", best_val_loss=0.25, **kw):
    return TrainResult(model_name=model, best_val_loss=best_val_loss,
                       evaluation=_evaluation(**kw), ckpt_path="a.pt", strategy=strategy)


class TestResultRecord(unittest.TestCase):
    def test_record_layout(self):
        ev = _evaluation()
        rec = result_record(TrainResult("dlinear", 0.25, ev, "a.pt", strategy="recursive"))
        self.assertEqual(RECORD_FORMAT, 2)
        self.assertEqual(set(rec), {"format", "strategy", "model", "best_val_loss",
                                    "regression", "event"})
        self.assertEqual((rec["format"], rec["strategy"], rec["model"], rec["best_val_loss"]),
                         (2, "recursive", "dlinear", 0.25))
        self.assertEqual(rec["regression"], ev.regression)
        self.assertEqual(rec["event"], ev.event)

    def test_record_does_not_alias_the_evaluation(self):
        result = _result()
        rec = result_record(result)
        rec["regression"][0]["rmse"] = -1.0
        rec["event"][0]["tp"] = -1
        self.assertNotEqual(result.evaluation.regression[0]["rmse"], -1.0)
        self.assertNotEqual(result.evaluation.event[0]["tp"], -1)

    def test_result_without_evaluation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no evaluation"):
            result_record(TrainResult("lstm", 0.5, None, "a.pt"))

    def test_record_roundtrip_json(self):
        # a nan channel exercises NaN counts and scores through JSON
        recs = [result_record(_result("dlinear", nan_channel=True)),
                result_record(_result("lstm", "recursive", float("nan")))]
        again = json.loads(json.dumps(recs))
        pd.testing.assert_frame_equal(regression_frame(recs, _CELL),
                                      regression_frame(again, _CELL))
        pd.testing.assert_frame_equal(event_frame(recs, _CELL), event_frame(again, _CELL))
        ef = event_frame(again, _CELL)
        bad = ef[(ef["model"] == "dlinear") & (ef["channel"] == "xrs_long")]
        self.assertTrue(bad["tss"].isna().all() and bad["tp"].isna().all())
        self.assertTrue((bad["n_nonfinite"] > 0).any())
        self.assertTrue(np.isnan(regression_frame(again, _CELL)["best_val_loss"].iloc[-1]))


class TestFrames(unittest.TestCase):
    def setUp(self):
        self.recs = [result_record(_result("dlinear")),
                     result_record(_result("lstm", "recursive", 0.5))]

    def test_frames_columns_and_cell_keys(self):
        self.assertEqual(RUN_KEYS, _CELL_KEYS + ["strategy", "model"])
        reg = regression_frame(self.recs, _CELL)
        ev = event_frame(self.recs, _CELL)
        self.assertEqual(list(reg.columns), regression_columns())
        self.assertEqual(list(ev.columns), event_columns())
        self.assertEqual(len(reg), 2 * 2)       # models x channels
        self.assertEqual(len(ev), 2 * 2 * 3)    # models x channels x granularities
        for k, v in _CELL.items():
            self.assertTrue((reg[k] == v).all() and (ev[k] == v).all(), k)
        r = reg[(reg["model"] == "lstm") & (reg["channel"] == "p_gt10")].iloc[0]
        self.assertEqual((r["strategy"], r["best_val_loss"], r["n"]), ("recursive", 0.5, 4))
        e = ev[(ev["model"] == "dlinear") & (ev["channel"] == "p_gt10")
               & (ev["granularity"] == "point")].iloc[0]
        self.assertEqual((e["tp"], e["fp"], e["fn"], e["tn"], e["threshold"]), (1, 1, 1, 1, 10.0))

    def test_without_cell_omits_cell_columns(self):
        reg = regression_frame(self.recs)
        ev = event_frame(self.recs)
        self.assertEqual(list(reg.columns), [c for c in regression_columns() if c not in _CELL_KEYS])
        self.assertEqual(list(ev.columns), [c for c in event_columns() if c not in _CELL_KEYS])

    def test_unselected_metrics_are_nan_columns(self):
        recs = [result_record(_result(metric_names=["rmse", "tss"]))]
        reg = regression_frame(recs, _CELL)
        ev = event_frame(recs, _CELL)
        self.assertEqual(list(reg.columns), regression_columns())
        self.assertEqual(list(ev.columns), event_columns())
        self.assertTrue(reg["rmse"].notna().all())
        for c in ("mae", "msa", "sspb", "logr"):
            self.assertTrue(reg[c].isna().all(), c)
        self.assertTrue(ev["tss"].notna().all())
        for c in ("hss", "pod", "far", "csi", "f1", "seds", "sedi"):
            self.assertTrue(ev[c].isna().all(), c)
        # the contingency counts are always written
        self.assertTrue(ev[["n", "n_events", "tp", "fp", "fn", "tn", "n_nonfinite"]]
                        .notna().all().all())

    def test_no_records_gives_empty_frames_with_columns(self):
        self.assertEqual(list(regression_frame([], _CELL).columns), regression_columns())
        self.assertEqual(len(event_frame([], _CELL)), 0)

    def test_record_without_event_rows(self):
        # a run without thresholds has regression rows only
        ev = Evaluation(regression=[{"channel": "xrs_long", "n": 4, "rmse": 1.0}], event=[])
        recs = [result_record(TrainResult("dlinear", 0.1, ev, "a.pt"))]
        self.assertEqual(len(regression_frame(recs, _CELL)), 1)
        self.assertEqual(len(event_frame(recs, _CELL)), 0)

    def test_cell_must_carry_every_key(self):
        with self.assertRaisesRegex(ValueError, "fold"):
            regression_frame(self.recs, {"track": "uni_b", "seq_len": 288, "pred_len": 36})


_COUNTS = ["n", "n_events", "tp", "fp", "fn", "tn", "n_nonfinite"]


def _count_fields(csv_text, columns):
    """The text of `columns` in every data line of a CSV."""
    import csv
    import io
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    return [r[c] for r in rows for c in columns]


class TestCountsAreIntegers(unittest.TestCase):
    """A voided channel has NaN counts; that must not turn every count of the
    table into a float written as `20.0`."""

    def test_counts_print_as_integers_next_to_a_voided_channel(self):
        recs = [result_record(_result("dlinear", nan_channel=True)),
                result_record(_result("lstm"))]
        ev, reg = event_frame(recs, _CELL), regression_frame(recs, _CELL)
        for c in _COUNTS:
            self.assertEqual(str(ev[c].dtype), "Int64", c)
        self.assertEqual(str(reg["n"].dtype), "Int64")
        fields = _count_fields(ev.to_csv(index=False), _COUNTS)
        self.assertIn("", fields)                       # the voided counts stay empty
        self.assertTrue(all(f == "" or f.isdigit() for f in fields), fields)
        self.assertTrue(all(f.isdigit() for f in _count_fields(reg.to_csv(index=False), ["n"])))

    def test_empty_frames_keep_their_columns(self):
        self.assertEqual(list(event_frame([], _CELL).columns), event_columns())


class TestColumnsFollowTheRegistries(unittest.TestCase):
    """Adding a metric is registering one function (spec 4.5): its column must
    reach every table without editing a column list."""

    def test_built_in_columns_keep_the_published_order(self):
        self.assertEqual(regression_columns(),
                         RUN_KEYS + ["best_val_loss", "channel", "n",
                                     "rmse", "mae", "msa", "sspb", "logr"])
        self.assertEqual(event_columns(),
                         RUN_KEYS + ["channel", "granularity", "threshold", "n", "n_events",
                                     "tp", "fp", "fn", "tn", "n_nonfinite",
                                     "tss", "hss", "pod", "far", "csi", "f1", "seds", "sedi"])

    def _register_throwaway_metrics(self):
        register_event_score("tmpscore")(lambda tp, fp, fn, tn: 0.5)
        self.addCleanup(EVENT_SCORES.pop, "tmpscore")
        register_regression_metric("tmpreg")(lambda pred, true: 0.25)
        self.addCleanup(REGRESSION_METRICS.pop, "tmpreg")

    def test_a_registered_metric_gets_its_column_in_both_tables(self):
        self._register_throwaway_metrics()
        recs = [result_record(_result())]
        reg, ev = regression_frame(recs, _CELL), event_frame(recs, _CELL)
        self.assertEqual(list(reg.columns), regression_columns())
        self.assertEqual(list(ev.columns), event_columns())
        self.assertEqual((list(reg.columns)[-1], list(ev.columns)[-1]), ("tmpreg", "tmpscore"))
        self.assertTrue((reg["tmpreg"] == 0.25).all())
        self.assertTrue((ev["tmpscore"] == 0.5).all())

    def test_a_registered_metric_reaches_the_comparison_tables(self):
        from tslib.exp.exp import build_comparison
        self._register_throwaway_metrics()
        reg, ev = build_comparison([_result()], "best_val_loss")
        self.assertIn("tmpreg", reg.columns)
        self.assertIn("tmpscore", ev.columns)

    def test_cleanup_restores_the_columns(self):
        before = (regression_columns(), event_columns())
        with self.subTest("registered"):
            self._register_throwaway_metrics()
            self.assertNotEqual((regression_columns(), event_columns()), before)
        self.doCleanups()
        self.assertEqual((regression_columns(), event_columns()), before)


class TestRejectsV1Rows(unittest.TestCase):
    def test_rejects_v1_rows(self):
        v1 = {"model": "x", "rmse": 1.0}
        for frame in (regression_frame, event_frame):
            with self.assertRaisesRegex(ValueError, "v1 result rows"):
                frame([v1])
            with self.assertRaisesRegex(ValueError, "v1 result rows"):
                frame([result_record(_result()), v1], _CELL)

    def test_message_says_how_to_upgrade(self):
        with self.assertRaisesRegex(ValueError, r"delete score/rows/x\.json.*re-tested, not retrained"):
            regression_frame([{"model": "x", "rmse": 1.0}])

    def test_rejects_other_format_numbers(self):
        rec = result_record(_result())
        rec["format"] = 3
        with self.assertRaisesRegex(ValueError, "v1 result rows"):
            regression_frame([rec])


if __name__ == "__main__":
    unittest.main()
