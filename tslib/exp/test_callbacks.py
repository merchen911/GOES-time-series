import unittest
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

from tslib.exp.callbacks import estimate_train_hours, TimingGateCallback
from tslib.exp.lightning_model import ForecastModule
from tslib.exp.metrics import MetricContext


def _cfg(on_slow, max_hours=6.0, probe=2):
    return SimpleNamespace(lr=1e-3, weight_decay=0.0, epochs=1, loss="mse",
                           metrics=["mse"], transform="log10",
                           event_threshold=None, seq_len=4, pred_len=2,
                           label_len=0, max_train_hours=max_hours,
                           on_slow=on_slow, probe_batches=probe)


class _Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b = x.shape[0]
        return torch.zeros(b, 2, 1, device=self.p.device) + self.p


def _loader():
    x = torch.zeros(16, 4, 1)
    y = torch.full((16, 2, 1), 2.0)
    return DataLoader(TensorDataset(x, y), batch_size=4)


def _trainer(cb):
    return pl.Trainer(max_epochs=1, accelerator="cpu", devices=1,
                      limit_train_batches=8, limit_val_batches=1,
                      logger=False, enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False,
                      callbacks=[cb])


class TestEstimate(unittest.TestCase):
    def test_formula(self):
        self.assertAlmostEqual(estimate_train_hours(0.5, 100, 3), 150 / 3600.0)


class TestGatePolicy(unittest.TestCase):
    def test_skip_stops_training(self):
        m = ForecastModule(_Toy(), _cfg("skip", max_hours=1e-12), None)
        _trainer(TimingGateCallback(_cfg("skip", max_hours=1e-12))).fit(m, _loader(), _loader())
        self.assertTrue(m._gate_skipped)

    def test_abort_raises(self):
        m = ForecastModule(_Toy(), _cfg("abort", max_hours=1e-12), None)
        with self.assertRaises(RuntimeError):
            _trainer(TimingGateCallback(_cfg("abort", max_hours=1e-12))).fit(m, _loader(), _loader())

    def test_proceed_continues(self):
        m = ForecastModule(_Toy(), _cfg("proceed", max_hours=1e-12), None)
        _trainer(TimingGateCallback(_cfg("proceed", max_hours=1e-12))).fit(m, _loader(), _loader())
        self.assertFalse(m._gate_skipped)
        self.assertGreater(m._est_train_hours, 0.0)

    def test_fast_model_not_skipped(self):
        m = ForecastModule(_Toy(), _cfg("skip", max_hours=1e9), None)
        _trainer(TimingGateCallback(_cfg("skip", max_hours=1e9))).fit(m, _loader(), _loader())
        self.assertFalse(m._gate_skipped)
