import json
import math
import unittest
import warnings

import numpy as np

from tslib.exp.evaluate import Evaluation, evaluate
from tslib.exp.granularity import GRANULARITIES
from tslib.exp.metrics import EVENT_SCORES, REGRESSION_METRICS, MetricContext

_HOUR_NS = 3600 * 10**9
_COUNTS = ("n_events", "tp", "fp", "fn", "tn")


def _daily(n):
    return np.arange(n, dtype=np.int64) * 24 * _HOUR_NS


def _two_channel():
    # 2 channels, N=4, P=1, log10; thresholds [10, 1e-5] -> stored [1, -5].
    # channel 0: pred [2,0,2,0] true [2,2,0,0] -> tp=fp=fn=tn=1
    pred = np.stack([[2.0, 0.0, 2.0, 0.0], [-4.0, -6.0, -4.0, -6.0]], axis=-1)[:, None, :]
    true = np.stack([[2.0, 2.0, 0.0, 0.0], [-4.0, -4.0, -6.0, -6.0]], axis=-1)[:, None, :]
    ctx = MetricContext(thresholds=[10.0, 1e-5], transform="log10",
                        target_cols=["p_gt10", "xrs_long"], forecast_start=_daily(4))
    return pred, true, ctx


def _row(ev, channel, granularity):
    return next(r for r in ev.event
                if r["channel"] == channel and r["granularity"] == granularity)


class TestEvaluateRows(unittest.TestCase):
    def test_rows_and_counts(self):
        pred, true, ctx = _two_channel()
        ev = evaluate(pred, true, ctx)
        self.assertIsInstance(ev, Evaluation)
        self.assertEqual({(r["channel"], r["granularity"]) for r in ev.event},
                         {(c, g) for c in ["p_gt10", "xrs_long"] for g in GRANULARITIES})
        r = next(r for r in ev.event if r["channel"] == "p_gt10" and r["granularity"] == "point")
        self.assertEqual((r["tp"], r["fp"], r["fn"], r["tn"], r["n"], r["n_events"]), (1, 1, 1, 1, 4, 2))
        self.assertEqual(r["threshold"], 10.0)
        self.assertEqual(len(ev.regression), 2)
        # P=1: window and day (4 daily windows, all kept) judge the same samples
        for g in ("window", "day"):
            r = _row(ev, "p_gt10", g)
            self.assertEqual((r["tp"], r["fp"], r["fn"], r["tn"], r["n"], r["n_events"]),
                             (1, 1, 1, 1, 4, 2), g)
        x = _row(ev, "xrs_long", "point")
        self.assertEqual(x["threshold"], 1e-5)
        self.assertEqual((x["tp"], x["fp"], x["fn"], x["tn"]), (1, 1, 1, 1))

    def test_row_keys_and_order(self):
        pred, true, ctx = _two_channel()
        ev = evaluate(pred, true, ctx)
        self.assertEqual(list(ev.regression[0]),
                         ["channel", "n"] + list(REGRESSION_METRICS))
        self.assertEqual(list(ev.event[0]),
                         ["channel", "granularity", "threshold", "n", "n_events",
                          "tp", "fp", "fn", "tn", "n_nonfinite"] + list(EVENT_SCORES))
        self.assertEqual([(r["channel"], r["granularity"]) for r in ev.event],
                         [(c, g) for c in ["p_gt10", "xrs_long"] for g in GRANULARITIES])

    def test_rows_are_plain_json_values(self):
        # records are written with json; numpy scalars would not serialize
        pred, true, ctx = _two_channel()
        ev = evaluate(pred.astype(np.float32), true.astype(np.float32), ctx)
        for row in ev.regression + ev.event:
            for k, v in row.items():
                self.assertIn(type(v), (str, int, float), (k, type(v)))
        json.dumps({"regression": ev.regression, "event": ev.event})

    def test_scores_match_formulas(self):
        pred, true, ctx = _two_channel()
        ev = evaluate(pred, true, ctx)
        r = _row(ev, "p_gt10", "point")
        for k, fn in EVENT_SCORES.items():
            exp = fn(1, 1, 1, 1)
            if math.isnan(exp):
                self.assertTrue(math.isnan(r[k]), k)
            else:
                self.assertEqual(r[k], exp, k)

    def test_point_window_day_differ(self):
        # N=5, P=2. Window = horizon max; day keeps windows at 0, 24, 48 h
        p = np.array([[2, 0], [0, 0], [0, 2], [2, 2], [0, 0]], dtype=float)
        t = np.array([[0, 2], [0, 0], [0, 0], [2, 0], [2, 0]], dtype=float)
        fs = (np.array([0, 12, 24, 30, 48]) * _HOUR_NS).astype(np.int64)
        ctx = MetricContext(thresholds=[10.0], transform="log10", target_cols=["p"],
                            forecast_start=fs)
        ev = evaluate(p[..., None], t[..., None], ctx)
        counts = lambda g: tuple(_row(ev, "p", g)[k] for k in ("n", "n_events", "tp", "fp", "fn", "tn"))
        self.assertEqual(counts("point"), (10, 3, 1, 3, 2, 4))
        self.assertEqual(counts("window"), (5, 3, 2, 1, 1, 1))
        self.assertEqual(counts("day"), (3, 2, 1, 1, 1, 0))

    def test_channel_name_fallback(self):
        pred, true, ctx = _two_channel()
        ctx.target_cols = []
        ev = evaluate(pred, true, ctx)
        self.assertEqual([r["channel"] for r in ev.regression], ["c0", "c1"])
        self.assertEqual({r["channel"] for r in ev.event}, {"c0", "c1"})

    def test_regression_per_channel_values(self):
        pred, true, ctx = _two_channel()
        ev = evaluate(pred, true, ctx)
        r0, r1 = ev.regression
        self.assertEqual((r0["channel"], r0["n"]), ("p_gt10", 4))
        self.assertAlmostEqual(r0["rmse"], math.sqrt(2.0))       # deltas 0,-2,2,0
        self.assertAlmostEqual(r0["mae"], 1.0)
        self.assertAlmostEqual(r0["msa"], 100 * (10 ** 1.0 - 1), places=9)   # median |d| = 1
        self.assertEqual(r0["sspb"], 0.0)
        self.assertAlmostEqual(r1["rmse"], math.sqrt(2.0))
        self.assertAlmostEqual(r1["logr"], 0.0)


class TestEvaluateUndefined(unittest.TestCase):
    def test_no_observed_events(self):
        p = np.array([2.0, 0.0, 2.0, 0.0])[:, None, None]
        t = np.zeros((4, 1, 1))
        ctx = MetricContext(thresholds=[10.0], transform="log10", target_cols=["p"],
                            forecast_start=_daily(4))
        ev = evaluate(p, t, ctx)
        for g in GRANULARITIES:
            r = _row(ev, "p", g)
            self.assertEqual((r["n_events"], r["tp"], r["fp"], r["fn"], r["tn"]), (0, 0, 2, 0, 2), g)
            for k in ("pod", "tss", "seds", "sedi"):
                self.assertTrue(math.isnan(r[k]), (g, k))
            self.assertEqual(r["csi"], 0.0)
            self.assertEqual(r["far"], 1.0)

    def test_all_nan_prediction(self):
        n, p_len = 4, 3
        pred = np.full((n, p_len, 1), np.nan, dtype=np.float32)
        true = np.zeros((n, p_len, 1), dtype=np.float32)
        ctx = MetricContext(thresholds=[10.0], transform="log10", target_cols=["p"],
                            forecast_start=_daily(n))
        ev = evaluate(pred, true, ctx)
        reg = ev.regression[0]
        self.assertEqual(reg["n"], n * p_len)
        for k in REGRESSION_METRICS:
            self.assertTrue(math.isnan(reg[k]), k)
        for r in ev.event:
            for k in _COUNTS + tuple(EVENT_SCORES):
                self.assertTrue(math.isnan(r[k]), (r["granularity"], k))
        self.assertEqual(_row(ev, "p", "point")["n_nonfinite"], n * p_len)
        self.assertEqual(_row(ev, "p", "window")["n_nonfinite"], n)
        self.assertEqual(_row(ev, "p", "day")["n_nonfinite"], n)
        self.assertEqual(_row(ev, "p", "point")["n"], n * p_len)

    def test_one_nonfinite_point_voids_the_channel_only(self):
        # one inf in window 1, which the day cut does not keep (windows at 0, 12, 24 h)
        pred, true, ctx = _two_channel()
        pred = np.repeat(pred[:3], 2, axis=1)
        true = np.repeat(true[:3], 2, axis=1)
        pred[1, 1, 0] = np.inf
        ctx.forecast_start = (np.array([0, 12, 24]) * _HOUR_NS).astype(np.int64)
        ev = evaluate(pred, true, ctx)
        self.assertEqual([_row(ev, "p_gt10", g)["n_nonfinite"] for g in GRANULARITIES], [1, 1, 0])
        for g in GRANULARITIES:
            r = _row(ev, "p_gt10", g)
            for k in _COUNTS + tuple(EVENT_SCORES):
                self.assertTrue(math.isnan(r[k]), (g, k))
        for k in REGRESSION_METRICS:
            self.assertTrue(math.isnan(ev.regression[0][k]), k)
        # the other channel is untouched
        self.assertEqual(_row(ev, "xrs_long", "point")["n_nonfinite"], 0)
        self.assertFalse(math.isnan(_row(ev, "xrs_long", "point")["tp"]))
        self.assertFalse(math.isnan(ev.regression[1]["rmse"]))

    def test_undefined_cases_emit_no_warnings(self):
        pred = np.full((3, 2, 1), np.nan)
        true = np.zeros((3, 2, 1))
        ctx = MetricContext(thresholds=[10.0], transform="log10", target_cols=["p"],
                            forecast_start=_daily(3))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            evaluate(pred, true, ctx)
            evaluate(np.ones((3, 2, 1)), true, ctx)          # no events, constant series
            evaluate(np.zeros((0, 2, 1)), np.zeros((0, 2, 1)),
                     MetricContext(thresholds=[10.0], transform="log10",
                                   forecast_start=np.zeros(0, dtype=np.int64)))


class TestEvaluateOptions(unittest.TestCase):
    def test_metric_filter(self):
        pred, true, ctx = _two_channel()
        ev = evaluate(pred, true, ctx, metric_names=["rmse", "csi"])
        for r in ev.regression:
            self.assertEqual(set(r), {"channel", "n", "rmse"})
        for r in ev.event:
            scores = set(r) & (set(EVENT_SCORES) | set(REGRESSION_METRICS))
            self.assertEqual(scores, {"csi"})
            self.assertIn("tp", r)

    def test_regression_only_filter_keeps_counts(self):
        pred, true, ctx = _two_channel()
        ev = evaluate(pred, true, ctx, metric_names=["mae"])
        self.assertEqual(len(ev.event), 6)
        r = _row(ev, "p_gt10", "point")
        self.assertEqual((r["tp"], r["fp"], r["fn"], r["tn"]), (1, 1, 1, 1))
        self.assertFalse(set(r) & set(EVENT_SCORES))

    def test_unknown_metric_name(self):
        pred, true, ctx = _two_channel()
        with self.assertRaises(KeyError):
            evaluate(pred, true, ctx, metric_names=["nope"])

    def test_no_thresholds_no_event_rows(self):
        pred, true, ctx = _two_channel()
        for thr in (None, []):
            ctx.thresholds = thr
            ev = evaluate(pred, true, ctx)
            self.assertEqual(ev.event, [])
            self.assertEqual(len(ev.regression), 2)

    def test_float64_input_is_scored_like_its_float32_prediction_file(self):
        # prediction files store float32, so a float64 caller (the statistic
        # strategy) must get the numbers a re-score of its file gives:
        # -5.00000001 is below the M-class threshold (-5) in float64, equal in float32
        pred = np.array([-5.00000001, -5.00000001, -6.0]).reshape(3, 1, 1)
        true = np.array([-4.0, -6.0, -4.0]).reshape(3, 1, 1)
        ctx = MetricContext(thresholds=[1e-5], transform="log10",
                            target_cols=["xrs_long"], forecast_start=_daily(3))
        as64 = evaluate(pred, true, ctx)
        as32 = evaluate(pred.astype(np.float32), true.astype(np.float32), ctx)
        self.assertEqual(as64.event, as32.event)
        self.assertEqual(as64.regression, as32.regression)
        self.assertEqual(_row(as64, "xrs_long", "point")["tp"], 1)   # -5 >= -5 in float32

    def test_forecast_start_in_microseconds_gives_the_same_day_rows(self):
        pred, true, ctx = _two_channel()
        want = evaluate(pred, true, ctx)
        ctx.forecast_start = ctx.forecast_start.astype("datetime64[ns]").astype("datetime64[us]")
        self.assertEqual(evaluate(pred, true, ctx).event, want.event)
        ctx.forecast_start = ctx.forecast_start.astype("datetime64[ns]").astype(np.int64).astype(float)
        with self.assertRaisesRegex(TypeError, "forecast_start"):
            evaluate(pred, true, ctx)

    def test_no_forecast_start_no_day_rows(self):
        pred, true, ctx = _two_channel()
        ctx.forecast_start = None
        ev = evaluate(pred, true, ctx)
        self.assertEqual({r["granularity"] for r in ev.event}, {"point", "window"})
        self.assertEqual(len(ev.event), 4)

    def test_transform_none_log_metrics(self):
        true = np.array([1.0, 10.0, 100.0])[:, None, None]
        pred = np.full((3, 1, 1), 10.0)
        ev = evaluate(pred, true, MetricContext(transform="none", target_cols=["p"]))
        r = ev.regression[0]
        self.assertAlmostEqual(r["msa"], 900.0, places=9)
        self.assertEqual(r["sspb"], 0.0)
        self.assertTrue(math.isnan(r["logr"]))            # constant prediction
        self.assertAlmostEqual(r["rmse"], math.sqrt((81 + 0 + 8100) / 3))
        self.assertAlmostEqual(r["mae"], (9 + 0 + 90) / 3)

    def test_transform_none_nonpositive_voids_log_metrics(self):
        true = np.array([0.0, 10.0, 100.0])[:, None, None]
        pred = np.array([1.0, 20.0, 50.0])[:, None, None]
        r = evaluate(pred, true, MetricContext(transform="none")).regression[0]
        for k in ("msa", "sspb", "logr"):
            self.assertTrue(math.isnan(r[k]), k)
        self.assertAlmostEqual(r["mae"], (1 + 10 + 50) / 3)
        r = evaluate(true, pred, MetricContext(transform="none")).regression[0]
        self.assertTrue(math.isnan(r["msa"]))

    def test_transform_none_threshold_in_physical_units(self):
        pred = np.array([20.0, 5.0])[:, None, None]
        true = np.array([10.0, 50.0])[:, None, None]
        ev = evaluate(pred, true, MetricContext(thresholds=[10.0], transform="none"))
        r = _row(ev, "c0", "point")
        self.assertEqual((r["threshold"], r["tp"], r["fp"], r["fn"], r["tn"]), (10.0, 1, 0, 1, 0))

    def test_shape_errors(self):
        pred, true, ctx = _two_channel()
        with self.assertRaises(ValueError):
            evaluate(pred, true[:3], ctx)                   # pred/true differ
        with self.assertRaises(ValueError):
            evaluate(pred[..., 0], true[..., 0], ctx)       # not (N, P, T)
        ctx.thresholds = [10.0]
        with self.assertRaises(ValueError):
            evaluate(pred, true, ctx)                       # one threshold per channel
        ctx.thresholds = [10.0, 1e-5]
        ctx.forecast_start = _daily(3)
        with self.assertRaises(ValueError):
            evaluate(pred, true, ctx)                       # one start time per window


if __name__ == "__main__":
    unittest.main()
