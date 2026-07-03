import unittest
from types import SimpleNamespace

import torch

from tslib.exp.losses import LOSS_REGISTRY, build_loss


class TestLosses(unittest.TestCase):
    def test_registry_has_builtins(self):
        for n in ["mse", "mae", "huber", "weighted_mse"]:
            self.assertIn(n, LOSS_REGISTRY)

    def test_default_mse_value(self):
        loss = build_loss(SimpleNamespace(loss="mse"))
        pred = torch.zeros(4, 3, 1)
        true = torch.ones(4, 3, 1)
        self.assertAlmostEqual(float(loss(pred, true)), 1.0, places=6)

    def test_unknown_loss_errors(self):
        with self.assertRaises(ValueError):
            build_loss(SimpleNamespace(loss="nope"))

    def test_weighted_mse_backprops(self):
        loss = build_loss(SimpleNamespace(loss="weighted_mse"))
        pred = torch.zeros(2, 2, 1, requires_grad=True)
        true = torch.ones(2, 2, 1)
        loss(pred, true).backward()
        self.assertIsNotNone(pred.grad)
