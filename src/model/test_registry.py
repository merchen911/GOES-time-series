import unittest
from types import SimpleNamespace

import torch

from model.registry import MODEL_REGISTRY, register_model
from model import build_model


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
