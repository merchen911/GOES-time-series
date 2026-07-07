import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from tslib.exp.lightning_model import ForecastModule, TrainResult
from tslib.exp.metrics import MetricContext


def _cfg(**kw):
    base = dict(lr=1e-3, weight_decay=0.0, epochs=1, loss="mse",
                metrics=["mse", "mae"], transform="log10",
                event_threshold=None, seq_len=4, pred_len=2, label_len=0)
    base.update(kw)
    return SimpleNamespace(**base)


class _ConstModel(torch.nn.Module):
    """Emits (B, pred_len, t) filled with a learnable scalar (device-aware)."""
    def __init__(self, pred_len, t):
        super().__init__()
        self.pred_len, self.t = pred_len, t
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b = x.shape[0]
        return torch.zeros(b, self.pred_len, self.t, device=self.p.device) + self.p


class TestTrainResultFields(unittest.TestCase):
    def test_new_fields_default_and_positional(self):
        r = TrainResult("lstm", 0.5, {"mse": 0.3}, "a.pt")
        self.assertFalse(r.skipped)
        self.assertTrue(np.isnan(r.est_train_hours))
        r2 = TrainResult("lstm", 0.5, {"mse": 0.3}, "a.pt", strategy="recursive")
        self.assertEqual(r2.strategy, "recursive")


class TestTrainingStepAlignment(unittest.TestCase):
    def test_one_step_pred_against_multistep_target(self):
        # model emits (B,1,1); target is (B,2,1). training_step must align y to
        # the first step and return a finite loss.
        m = ForecastModule(_ConstModel(1, 1), _cfg(pred_len=2), None)
        x = torch.zeros(4, 4, 1)
        y = torch.full((4, 2, 1), 2.0)
        loss = m.training_step((x, y), 0)
        self.assertTrue(torch.isfinite(loss).item())


class TestTestMetrics(unittest.TestCase):
    def test_regression_and_event_keys(self):
        ctx = MetricContext(thresholds=[10.0], transform="log10",
                            target_cols=["p_gt10"])
        m = ForecastModule(_ConstModel(2, 1),
                           _cfg(metrics=["mse", "tss"], event_threshold=[10.0]),
                           ctx)
        x = torch.zeros(4, 4, 1)
        y = torch.full((4, 2, 1), 2.0)  # log10 space, > log10(10)=1 → event
        m.test_step((x, y), 0)
        m.on_test_epoch_end()
        self.assertIn("mse", m.test_metrics)
        self.assertIn("tss_p_gt10", m.test_metrics)


class TestTrainerSmoke(unittest.TestCase):
    def test_fit_and_test_via_trainer(self):
        import pytorch_lightning as pl
        ctx = MetricContext(thresholds=None, transform="log10",
                            target_cols=["p_gt10"])
        m = ForecastModule(_ConstModel(2, 1), _cfg(metrics=["mse"]), ctx)
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
        self.assertIn("mse", m.test_metrics)
