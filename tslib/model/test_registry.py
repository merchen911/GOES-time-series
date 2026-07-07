import unittest
from types import SimpleNamespace

import torch

from tslib.model.registry import MODEL_REGISTRY, register_model
from tslib.model import build_model


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
