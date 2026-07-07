import unittest
from types import SimpleNamespace

import torch

from tslib.model.registry import MODEL_REGISTRY, register_model
from tslib.model import build_model


def _real_cfg(models, seq_len=96, pred_len=24, channels=1):
    """Build a real exp_parser config for the given model names (direct
    strategy). `channels>1` also sets --target_cols so multivariable model
    variants can be exercised (used by later model-zoo tasks)."""
    from tslib.configs.config import exp_parser, config_postprocess
    argv = ["--data_path", "/tmp/x.parquet", "--target_col", "p_gt10",
            "--seq_len", str(seq_len), "--pred_len", str(pred_len),
            "--fold_numb", "0", "--forecast_strategy", "direct",
            "--event_threshold", *(["10"] * channels),
            "--metrics", "rmse", "mae", "tss", "hss", "pod", "far",
            "--models", *models]
    if channels > 1:
        argv += ["--target_cols"] + [f"col{i}" for i in range(channels)]
    return config_postprocess(exp_parser().parse_args(argv))


def _assert_builds(testcase, names, channels=1):
    """Construct + forward each named model from a real config; assert the
    output shape is (batch, pred_len, n_targets)."""
    cfg = _real_cfg(names, channels=channels)
    x = torch.zeros(2, cfg.seq_len, channels)
    tgt = list(range(channels)) if channels > 1 else [0]
    for name in names:
        net = build_model(name, cfg, channels, tgt, strategy="direct")
        net.eval()
        with torch.no_grad():
            out = net(x)
        testcase.assertEqual(tuple(out.shape), (2, cfg.pred_len, len(tgt)), msg=name)


class TestModelRegistry(unittest.TestCase):
    def test_legacy_models_registered(self):
        for n in ["lstm", "timesnet", "patchtst"]:
            self.assertIn(n, MODEL_REGISTRY)

    def test_register_and_lookup(self):
        @register_model("dummy_reg_model")
        def ctor(cfg):
            return torch.nn.Identity()
        self.assertIn("dummy_reg_model", MODEL_REGISTRY)
        self.assertEqual(MODEL_REGISTRY["dummy_reg_model"].adapter, "standard")

    def test_unknown_model_errors(self):
        with self.assertRaises(ValueError):
            build_model("no_such_model", SimpleNamespace(), input_size=1,
                        target_indices=[0])

    def test_build_lstm_routes_through_registry(self):
        cfg = SimpleNamespace(task_name="long_term_forecast", label_len=0,
                              seq_len=24, pred_len=1, d_model=16,
                              num_layers=2, dropout=0.1)
        m = build_model("lstm", cfg, input_size=2, target_indices=[0])
        self.assertEqual(tuple(m(torch.randn(2, 24, 2)).shape), (2, 1, 1))

    def test_benchmark_direct_models_construct_and_forward(self):
        # Guards against a config attr being read by a model but never defined
        # in exp_parser (which silently made itransformer/timemixer unrunnable).
        from tslib.configs.config import exp_parser, config_postprocess
        argv = ["--data_path", "/tmp/x.parquet", "--target_col", "p_gt10",
                "--seq_len", "96", "--pred_len", "24", "--fold_numb", "0",
                "--forecast_strategy", "direct", "--event_threshold", "10",
                "--metrics", "rmse", "mae", "tss", "hss", "pod", "far",
                "--models", "patchtst"]
        cfg = config_postprocess(exp_parser().parse_args(argv))
        x = torch.zeros(2, cfg.seq_len, 1)
        for name in ["lstm", "patchtst", "itransformer", "timemixer"]:
            model = build_model(name, cfg, input_size=1, target_indices=[0],
                                strategy="direct")
            model.eval()
            with torch.no_grad():
                out = model(x)
            self.assertEqual(tuple(out.shape), (2, 24, 1), msg=name)


class TestEnableGroup1(unittest.TestCase):
    def test_builds(self):
        _assert_builds(self, ["micn", "nonstationary_transformer", "scinet"])


class TestEnableGroup2(unittest.TestCase):
    def test_builds(self):
        _assert_builds(self, ["patchmixer", "segrnn", "xpatch"])

    def test_segrnn_thuml_builds(self):
        # tslib/model/segrnn_thuml.py previously diverged from THUML's
        # SegRNN.py: encoder() had a padding branch (`if seq_len % seg_len ==
        # 0: seg_num_x += 1`) that incremented seg_num_x WITHOUT zero-padding
        # the tensor, and the condition was inverted (padding is only ever
        # needed when seq_len is NOT evenly divisible by seg_len), making the
        # `x.reshape(-1, seg_num_x, seg_len)` shape-invalid. Fixed to match
        # upstream (which has no such branch and instead requires seg_len to
        # evenly divide seq_len). Now builds like any other registered model.
        _assert_builds(self, ["segrnn_thuml"])
