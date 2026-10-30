"""Shape smoke test for the v004 short-horizon grid.

The v003 sweep only ever ran pred_len 144/288, both multiples of SegRNN's
default seg_len=24. v004 adds pred_len 36 and 72 (3 h / 6 h on the common
5-min grid), and 24 does NOT divide 36 -- segrnn_thuml's decoder emits
seg_num_y * seg_len values and reshapes them to pred_len, so an indivisible
horizon is a hard RuntimeError, not a silent degradation. This test builds and
forward-passes EVERY sweep model at EVERY new (seq_len, pred_len) point so a
shape bug shows up in seconds instead of hours into a GPU sweep.
"""
import unittest

import torch

from tslib.benchmark.driver import DIRECT_UNI, PRED_LENS, SEG_LEN, SEQ_LENS
from tslib.configs.config import config_postprocess, exp_parser
from tslib.model import build_model


def _cfg(model, seq_len, pred_len, channels):
    argv = ["--data_path", "/tmp/x.parquet", "--target_col", "p_gt10",
            "--seq_len", str(seq_len), "--pred_len", str(pred_len),
            "--fold_numb", "0", "--forecast_strategy", "direct",
            "--event_threshold", *(["10"] * channels),
            "--metrics", "rmse", "mae",
            "--d_model", "128", "--num_layers", "2", "--dropout", "0.1",
            "--seg_len", str(SEG_LEN),
            "--models", model]
    if channels > 1:
        argv += ["--target_cols"] + [f"col{i}" for i in range(channels)]
    return config_postprocess(exp_parser().parse_args(argv))


class TestShortHorizons(unittest.TestCase):
    def test_every_model_every_new_horizon(self):
        for seq_len in SEQ_LENS:
            for pred_len in PRED_LENS:
                for channels in (1, 2):
                    for model in DIRECT_UNI:
                        with self.subTest(model=model, seq_len=seq_len,
                                          pred_len=pred_len, channels=channels):
                            cfg = _cfg(model, seq_len, pred_len, channels)
                            tgt = list(range(channels))
                            net = build_model(model, cfg, channels, tgt,
                                              strategy="direct").eval()
                            with torch.no_grad():
                                out = net(torch.zeros(2, seq_len, channels))
                            self.assertEqual(tuple(out.shape),
                                             (2, pred_len, channels))

    def test_seg_len_divides_every_lookback(self):
        """The one divisibility SegRNN still requires is seq_len % seg_len."""
        for seq_len in SEQ_LENS:
            self.assertEqual(seq_len % SEG_LEN, 0, f"seq_len={seq_len}")

    def test_indivisible_horizon_over_decodes_then_trims(self):
        """pred_len 36 at seg_len 24: decode 2 segments (48 steps), return 36.
        The old behaviour was a RuntimeError on the reshape."""
        cfg = _cfg("segrnn_thuml", 288, 36, 1)
        net = build_model("segrnn_thuml", cfg, 1, [0], strategy="direct").eval()
        inner = net.base_model if hasattr(net, "base_model") else net
        self.assertEqual((inner.seg_num_y, inner.dec_len), (2, 48))
        with torch.no_grad():
            self.assertEqual(tuple(net(torch.zeros(2, 288, 1)).shape), (2, 36, 1))

    def test_divisible_horizon_decodes_exactly(self):
        cfg = _cfg("segrnn_thuml", 288, 72, 1)
        net = build_model("segrnn_thuml", cfg, 1, [0], strategy="direct").eval()
        inner = net.base_model if hasattr(net, "base_model") else net
        self.assertEqual((inner.seg_num_y, inner.dec_len), (3, 72))

    def test_seg_len_not_dividing_the_lookback_is_a_named_error(self):
        with self.assertRaisesRegex(ValueError, "divide seq_len"):
            cfg = _cfg("segrnn_thuml", 288, 36, 1)
            cfg.seg_len = 7
            build_model("segrnn_thuml", cfg, 1, [0], strategy="direct")


if __name__ == "__main__":
    unittest.main()
