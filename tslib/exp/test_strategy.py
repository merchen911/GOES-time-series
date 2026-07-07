import unittest
from types import SimpleNamespace

import numpy as np

from tslib.exp.strategy import run_strategy, StatisticalRunner


class TestRunStrategyRouting(unittest.TestCase):
    def test_unknown_strategy_rejected(self):
        with self.assertRaises(ValueError):
            run_strategy("bogus", "lstm", SimpleNamespace(),
                         SimpleNamespace(), "ckpt.pt")


def _stat_cfg():
    return SimpleNamespace(pred_len=4, arima_order=[1, 0, 0], ar_lags=2,
                           metrics=["mse"], transform="log10",
                           event_threshold=None)


def _fake_bundle():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(size=(3, 24, 1)), axis=1) + 5.0
    y = np.cumsum(rng.normal(size=(3, 4, 1)), axis=1) + 5.0
    bundle = SimpleNamespace(test_loader=[(x, y)], target_cols=["p_gt10"],
                             target_indices=[0])
    return bundle


class TestStatisticalRunner(unittest.TestCase):
    def test_produces_result_with_nan_val_loss(self):
        runner = StatisticalRunner("ar", _stat_cfg())
        res = runner.fit_and_test(_fake_bundle(), "ar", "unused.pt")
        self.assertEqual(res.strategy, "statistic")
        self.assertTrue(np.isnan(res.best_val_loss))
        self.assertIn("mse", res.metrics)

    def test_dispatcher_routes_statistic(self):
        res = run_strategy("statistic", "ar", _fake_bundle(), _stat_cfg(),
                           "unused.pt")
        self.assertEqual(res.strategy, "statistic")

    def test_fits_target_channel_not_channel_zero(self):
        # 2-channel input: channel 0 = 100.0 (decoy), channel 1 (target) = 5.0.
        # y = 5.0. If the runner incorrectly fits channel 0, the forecast
        # will be ~100.0 and the mse will be huge; if it correctly fits the
        # target channel (index 1), the forecast will be ~5.0 and mse ~0.
        x = np.zeros((2, 24, 2))
        x[:, :, 0] = 100.0
        x[:, :, 1] = 5.0
        y = np.full((2, 4, 1), 5.0)
        cfg = SimpleNamespace(pred_len=4, arima_order=[1, 0, 0], ar_lags=2,
                              metrics=["mse"], transform="none",
                              event_threshold=None)
        bundle = SimpleNamespace(test_loader=[(x, y)],
                                 target_cols=["p_gt10"],
                                 target_indices=[1])
        res = StatisticalRunner("ar", cfg).fit_and_test(bundle, "ar",
                                                         "unused.pt")
        self.assertLess(res.metrics["mse"], 1.0)


class TestNeuralStrategyLightning(unittest.TestCase):
    def test_direct_routes_through_trainer(self):
        import tempfile, os
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from types import SimpleNamespace
        import tslib.exp.strategy as strat

        class _Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x):
                b = x.shape[0]
                return torch.zeros(b, 2, 1, device=self.p.device) + self.p

        # avoid needing full backbone config: build_model returns a toy model
        orig = strat.build_model
        strat.build_model = lambda *a, **k: _Toy()
        try:
            x = torch.zeros(8, 4, 1)
            y = torch.full((8, 2, 1), 2.0)
            loader = DataLoader(TensorDataset(x, y), batch_size=4)
            bundle = SimpleNamespace(train_loader=loader, val_loader=loader,
                                     test_loader=loader, input_size=1,
                                     target_indices=[0], target_cols=["p_gt10"])
            cfg = SimpleNamespace(lr=1e-3, weight_decay=0.0, epochs=1, loss="mse",
                                  metrics=["mse"], transform="log10",
                                  event_threshold=None, seq_len=4, pred_len=2,
                                  label_len=0, max_train_hours=1e6,
                                  on_slow="proceed", probe_batches=2)
            with tempfile.TemporaryDirectory() as d:
                res = strat.run_strategy("direct", "toy", bundle, cfg,
                                         os.path.join(d, "toy.pt"))
            self.assertEqual(res.strategy, "direct")
            self.assertFalse(res.skipped)
            self.assertIn("mse", res.metrics)
        finally:
            strat.build_model = orig


class TestGateSkipInStrategy(unittest.TestCase):
    def test_slow_model_skipped(self):
        import tempfile, os
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from types import SimpleNamespace
        import tslib.exp.strategy as strat

        class _Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x):
                b = x.shape[0]
                return torch.zeros(b, 2, 1, device=self.p.device) + self.p

        orig = strat.build_model
        strat.build_model = lambda *a, **k: _Toy()
        try:
            x = torch.zeros(16, 4, 1)
            y = torch.full((16, 2, 1), 2.0)
            loader = DataLoader(TensorDataset(x, y), batch_size=4)
            bundle = SimpleNamespace(train_loader=loader, val_loader=loader,
                                     test_loader=loader, input_size=1,
                                     target_indices=[0], target_cols=["p_gt10"])
            cfg = SimpleNamespace(lr=1e-3, weight_decay=0.0, epochs=1, loss="mse",
                                  metrics=["mse"], transform="log10",
                                  event_threshold=None, seq_len=4, pred_len=2,
                                  label_len=0, max_train_hours=1e-12,
                                  on_slow="skip", probe_batches=2)
            with tempfile.TemporaryDirectory() as d:
                res = strat.run_strategy("direct", "toy", bundle, cfg,
                                         os.path.join(d, "toy.pt"))
            self.assertTrue(res.skipped)
            self.assertGreater(res.est_train_hours, 0.0)
        finally:
            strat.build_model = orig
