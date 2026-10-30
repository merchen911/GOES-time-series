import json
import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from tslib.exp.evaluate import Evaluation
from tslib.exp.lightning_model import ForecastModule, TrainResult
from tslib.exp.metrics import MetricContext
from tslib.exp.records import result_record

_DAY_NS = 24 * 3600 * 10**9


def _cfg(**kw):
    base = dict(lr=1e-3, weight_decay=0.0, epochs=1, loss="mse",
                metrics=None, transform="log10",
                event_threshold=None, seq_len=4, pred_len=2, label_len=0)
    base.update(kw)
    return SimpleNamespace(**base)


class _ConstModel(torch.nn.Module):
    """Emits (B, pred_len, t) filled with a learnable scalar (device-aware)."""
    def __init__(self, pred_len, t, value=0.0):
        super().__init__()
        self.pred_len, self.t = pred_len, t
        self.p = torch.nn.Parameter(torch.full((1,), value))

    def forward(self, x):
        b = x.shape[0]
        return torch.zeros(b, self.pred_len, self.t, device=self.p.device) + self.p


def _event_ctx(n_windows):
    return MetricContext(thresholds=[10.0], transform="log10", target_cols=["p_gt10"],
                         forecast_start=np.arange(n_windows, dtype=np.int64) * _DAY_NS)


def _run_test_epoch(m, batches):
    for i, (x, y) in enumerate(batches):
        m.test_step((x, y), i)
    m.on_test_epoch_end()


class TestTrainResultFields(unittest.TestCase):
    def test_final_shape(self):
        ev = Evaluation(regression=[], event=[])
        r = TrainResult("lstm", 0.5, ev, "a.pt")
        self.assertIs(r.evaluation, ev)
        self.assertEqual((r.ckpt_path, r.strategy), ("a.pt", "direct"))
        r2 = TrainResult("lstm", 0.5, ev, "a.pt", strategy="recursive")
        self.assertEqual(r2.strategy, "recursive")
        for gone in ("metrics", "skipped", "est_train_hours"):
            self.assertFalse(hasattr(r, gone), gone)

    def test_evaluation_is_required(self):
        with self.assertRaises(TypeError):
            TrainResult(model_name="lstm", best_val_loss=0.5, ckpt_path="a.pt")


class TestTrainingStepAlignment(unittest.TestCase):
    def test_one_step_pred_against_multistep_target(self):
        # model emits (B,1,1); target is (B,2,1). training_step must align y to
        # the first step and return a finite loss.
        m = ForecastModule(_ConstModel(1, 1), _cfg(pred_len=2), None)
        x = torch.zeros(4, 4, 1)
        y = torch.full((4, 2, 1), 2.0)
        loss = m.training_step((x, y), 0)
        self.assertTrue(torch.isfinite(loss).item())


class TestTestEpochEvaluation(unittest.TestCase):
    def test_test_epoch_produces_evaluation(self):
        # two batches of 3 windows; constant prediction 2.0 (log10) against an
        # observed series that crosses the threshold log10(10) = 1
        m = ForecastModule(_ConstModel(2, 1, value=2.0), _cfg(event_threshold=[10.0]),
                           _event_ctx(6))
        x = torch.zeros(3, 4, 1)
        y1 = torch.full((3, 2, 1), 2.0)
        y2 = torch.zeros(3, 2, 1)
        _run_test_epoch(m, [(x, y1), (x, y2)])

        self.assertEqual(m.test_pred.shape, (6, 2, 1))
        self.assertEqual(m.test_true.shape, (6, 2, 1))
        np.testing.assert_array_equal(m.test_true[:3], 2.0)
        self.assertEqual(len(m.evaluation.regression), 1)
        reg = m.evaluation.regression[0]
        self.assertEqual((reg["channel"], reg["n"]), ("p_gt10", 12))
        self.assertAlmostEqual(reg["rmse"], math.sqrt(2.0))
        self.assertEqual([r["granularity"] for r in m.evaluation.event],
                         ["point", "window", "day"])
        for r in m.evaluation.event:
            # always forecasting an event: every observed event is a hit
            self.assertEqual((r["tp"], r["fn"]), (r["n"] // 2, 0), r["granularity"])
        # every metric is evaluated by default (config.metrics is None)
        self.assertIn("sedi", m.evaluation.event[0])
        self.assertIn("logr", reg)

    def test_metric_filter_from_config(self):
        m = ForecastModule(_ConstModel(2, 1), _cfg(metrics=["rmse", "hss"],
                                                    event_threshold=[10.0]),
                           _event_ctx(4))
        _run_test_epoch(m, [(torch.zeros(4, 4, 1), torch.full((4, 2, 1), 2.0))])
        self.assertIn("rmse", m.evaluation.regression[0])
        self.assertNotIn("mae", m.evaluation.regression[0])
        self.assertIn("hss", m.evaluation.event[0])
        self.assertNotIn("tss", m.evaluation.event[0])

    def test_nan_model_still_writes_record(self):
        m = ForecastModule(_ConstModel(2, 1, value=float("nan")),
                           _cfg(event_threshold=[10.0]), _event_ctx(4))
        _run_test_epoch(m, [(torch.zeros(4, 4, 1), torch.full((4, 2, 1), 2.0))])

        self.assertTrue(math.isnan(m.evaluation.regression[0]["rmse"]))
        point = m.evaluation.event[0]
        self.assertTrue(math.isnan(point["tss"]) and math.isnan(point["tp"]))
        self.assertEqual(point["n_nonfinite"], 8)
        rec = result_record(TrainResult("nanmodel", 0.5, m.evaluation, "a.pt"))
        again = json.loads(json.dumps(rec))
        self.assertEqual(again["model"], "nanmodel")
        self.assertTrue(math.isnan(again["regression"][0]["rmse"]))


class TestTrainerSmoke(unittest.TestCase):
    def test_fit_and_test_via_trainer(self):
        import pytorch_lightning as pl
        ctx = MetricContext(thresholds=None, transform="log10",
                            target_cols=["p_gt10"])
        m = ForecastModule(_ConstModel(2, 1), _cfg(), ctx)
        x = torch.zeros(8, 4, 1)
        y = torch.full((8, 2, 1), 2.0)
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        trainer = pl.Trainer(max_epochs=1, accelerator="cpu", devices=1,
                             limit_train_batches=2, limit_val_batches=1,
                             limit_test_batches=2, logger=False,
                             enable_checkpointing=False,
                             enable_progress_bar=False,
                             enable_model_summary=False)
        trainer.fit(m, loader, loader)
        trainer.test(m, loader)
        self.assertEqual(m.test_pred.shape, (8, 2, 1))
        self.assertEqual(m.evaluation.regression[0]["channel"], "p_gt10")
        self.assertEqual(m.evaluation.event, [])   # no thresholds, no event rows


if __name__ == "__main__":
    unittest.main()
