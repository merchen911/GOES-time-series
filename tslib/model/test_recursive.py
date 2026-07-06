import unittest
from types import SimpleNamespace

import torch

from tslib.model import build_model, RecursiveForecastAdapter


def _cfg(seq_len, pred_len):
    return SimpleNamespace(task_name="long_term_forecast", label_len=0,
                           seq_len=seq_len, pred_len=pred_len,
                           d_model=16, num_layers=2, dropout=0.1)


class _EchoBase(torch.nn.Module):
    """One-step base that predicts the last input frame (persistence)."""
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return x_enc[:, -1:, :] + self.p  # (B, 1, C), on x_enc.device


class TestRecursiveAdapter(unittest.TestCase):
    def test_train_mode_single_step(self):
        m = build_model("lstm", _cfg(24, 6), input_size=2, target_indices=[0, 1],
                        strategy="recursive")
        m.train()
        out = m(torch.randn(4, 24, 2))
        self.assertEqual(tuple(out.shape), (4, 1, 2))

    def test_eval_mode_full_horizon(self):
        m = build_model("lstm", _cfg(24, 6), input_size=2, target_indices=[0, 1],
                        strategy="recursive")
        m.eval()
        out = m(torch.randn(4, 24, 2))
        self.assertEqual(tuple(out.shape), (4, 6, 2))

    def test_rollout_matches_manual_loop(self):
        cfg = _cfg(5, 3)
        adapter = RecursiveForecastAdapter(_EchoBase(), cfg,
                                           target_indices=[0], rollout_len=3)
        adapter.eval()
        x = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1)  # last frame = 4.0
        out = adapter(x)                       # echo => every step = 4.0
        self.assertEqual(tuple(out.shape), (1, 3, 1))
        self.assertTrue(torch.allclose(out, torch.full((1, 3, 1), 4.0)))
