import unittest
from types import SimpleNamespace

import numpy as np
import torch

from exp.lightning_model import pl_model
from exp.metrics import MetricContext


def _cfg(**kw):
    base = dict(lr=1e-3, weight_decay=0.0, epochs=1, loss="mse",
                metrics=["mse", "mae"], transform="log10",
                event_threshold=None, seq_len=4, pred_len=2, label_len=0)
    base.update(kw)
    return SimpleNamespace(**base)


class _ConstModel(torch.nn.Module):
    def __init__(self, pred_len, t):
        super().__init__()
        self.pred_len, self.t = pred_len, t
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b = x.shape[0]
        return torch.zeros(b, self.pred_len, self.t) + self.p


class TestPlModelMetrics(unittest.TestCase):
    def test_evaluate_regression_and_event(self):
        runner = pl_model(_ConstModel(2, 1),
                          _cfg(metrics=["mse", "tss"], event_threshold=[10.0]))
        # loader yields (x, y): pred is ~0, true=2.0 (log10 space, > log10(10)=1)
        x = torch.zeros(4, 4, 1)
        y = torch.full((4, 2, 1), 2.0)
        loader = [(x, y)]
        ctx = MetricContext(thresholds=[10.0], transform="log10",
                            target_cols=["p_gt10"])
        out = runner.evaluate(loader, ctx)
        self.assertIn("mse", out)            # regression scalar
        self.assertIn("tss_p_gt10", out)     # event, per-channel key
