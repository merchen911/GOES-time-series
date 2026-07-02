import unittest
from types import SimpleNamespace

import torch

from model import build_model


def _cfg(seq_len, pred_len):
    return SimpleNamespace(task_name="long_term_forecast", label_len=0,
                           seq_len=seq_len, pred_len=pred_len,
                           d_model=16, num_layers=2, dropout=0.1)


class TestLSTMMultiStep(unittest.TestCase):
    def test_output_honors_pred_len_multivariate(self):
        # multivariate input (C=2), single target -> (B, pred_len, 1)
        for seq_len, pred_len in [(24, 1), (288, 12), (864, 288)]:
            m = build_model("lstm", _cfg(seq_len, pred_len),
                            input_size=2, target_index=0)
            x = torch.randn(4, seq_len, 2)
            y = m(x)
            self.assertEqual(tuple(y.shape), (4, pred_len, 1),
                             msg=f"seq={seq_len} pred={pred_len}")

    def test_train_step_shapes_align(self):
        # loss between pred and (B, pred_len, 1) target must compute + backprop
        m = build_model("lstm", _cfg(576, 144), input_size=3, target_index=1)
        x = torch.randn(8, 576, 3)
        target = torch.randn(8, 144, 1)
        loss = torch.nn.functional.mse_loss(m(x), target)
        loss.backward()
        grads = [p.grad for p in m.parameters() if p.grad is not None]
        self.assertTrue(len(grads) > 0 and any(g.abs().sum() > 0 for g in grads))


if __name__ == "__main__":
    unittest.main()
