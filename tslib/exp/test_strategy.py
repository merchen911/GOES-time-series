import contextlib
import io
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from tslib.exp.predictions import load_predictions
from tslib.exp.strategy import run_strategy, StatisticalRunner

_DAY_NS = 24 * 3600 * 10**9
_META_KEYS = {"model", "run_name", "strategy", "seq_len", "pred_len", "fold_numb",
              "target_cols", "transform", "cadence_min", "event_threshold",
              "tslib_version", "format"}


class TestRunStrategyRouting(unittest.TestCase):
    def test_unknown_strategy_rejected(self):
        with self.assertRaises(ValueError):
            run_strategy("bogus", "lstm", SimpleNamespace(),
                         SimpleNamespace(), "ckpt.pt")


def _stat_cfg(**kw):
    base = dict(pred_len=4, arima_order=[1, 0, 0], ar_lags=2,
                metrics=None, transform="log10", event_threshold=None,
                run_name="cell", seq_len=24, fold_numb=0, cadence_min=5,
                save_predictions=True)
    base.update(kw)
    return SimpleNamespace(**base)


def _fake_bundle():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(size=(3, 24, 1)), axis=1) + 5.0
    y = np.cumsum(rng.normal(size=(3, 4, 1)), axis=1) + 5.0
    bundle = SimpleNamespace(test_loader=[(x, y)], target_cols=["p_gt10"],
                             target_indices=[0],
                             test_forecast_start=np.arange(3, dtype=np.int64) * _DAY_NS)
    return bundle


class TestStatisticalRunner(unittest.TestCase):
    def test_produces_result_with_nan_val_loss(self):
        runner = StatisticalRunner("ar", _stat_cfg())
        res = runner.fit_and_test(_fake_bundle(), "ar", "unused.pt")
        self.assertEqual(res.strategy, "statistic")
        self.assertTrue(np.isnan(res.best_val_loss))
        self.assertEqual(res.evaluation.regression[0]["channel"], "p_gt10")
        self.assertIn("rmse", res.evaluation.regression[0])

    def test_dispatcher_routes_statistic(self):
        res = run_strategy("statistic", "ar", _fake_bundle(), _stat_cfg(),
                           "unused.pt")
        self.assertEqual(res.strategy, "statistic")

    def test_context_carries_forecast_start(self):
        # the statistic path builds its context with _metric_ctx too, so the
        # 24-hour cut (day rows) is available there as well
        res = run_strategy("statistic", "ar", _fake_bundle(),
                           _stat_cfg(event_threshold=[1e5]), "unused.pt")
        self.assertEqual([r["granularity"] for r in res.evaluation.event],
                         ["point", "window", "day"])
        self.assertEqual(res.evaluation.event[-1]["n"], 3)

    def test_fits_target_channel_not_channel_zero(self):
        # 2-channel input: channel 0 = 100.0 (decoy), channel 1 (target) = 5.0.
        # y = 5.0. If the runner incorrectly fits channel 0, the forecast
        # will be ~100.0 and the rmse will be huge; if it correctly fits the
        # target channel (index 1), the forecast will be ~5.0 and rmse ~0.
        x = np.zeros((2, 24, 2))
        x[:, :, 0] = 100.0
        x[:, :, 1] = 5.0
        y = np.full((2, 4, 1), 5.0)
        cfg = _stat_cfg(transform="none")
        bundle = SimpleNamespace(test_loader=[(x, y)],
                                 target_cols=["p_gt10"],
                                 target_indices=[1],
                                 test_forecast_start=np.arange(2, dtype=np.int64))
        res = StatisticalRunner("ar", cfg).fit_and_test(bundle, "ar",
                                                         "unused.pt")
        self.assertLess(res.evaluation.regression[0]["rmse"], 1.0)

    def test_statistic_saves_predictions(self):
        bundle = _fake_bundle()
        cfg = _stat_cfg(event_threshold=[1e5])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pred", "ar.npz")
            res = StatisticalRunner("ar", cfg).fit_and_test(bundle, "ar", "unused.pt",
                                                            pred_path=path)
            self.assertTrue(os.path.exists(path))
            pred, true, fs, meta = load_predictions(path)
        self.assertEqual(pred.shape, (3, 4, 1))
        np.testing.assert_allclose(true, bundle.test_loader[0][1], rtol=1e-6)
        np.testing.assert_array_equal(fs, bundle.test_forecast_start)
        self.assertEqual(set(meta), _META_KEYS)
        self.assertEqual((meta["model"], meta["strategy"], meta["run_name"]),
                         ("ar", "statistic", "cell"))
        self.assertEqual((meta["seq_len"], meta["pred_len"], meta["fold_numb"]), (24, 4, 0))
        self.assertEqual((meta["target_cols"], meta["transform"], meta["cadence_min"]),
                         (["p_gt10"], "log10", 5))
        self.assertEqual(meta["event_threshold"], [1e5])
        self.assertIsInstance(meta["tslib_version"], str)
        self.assertEqual(res.evaluation.regression[0]["n"], 12)

    def test_inline_rows_equal_a_rescore_of_the_saved_file(self):
        # one implementation, one number: the row written at fit time and a
        # re-score of the float32 prediction file must agree exactly
        from tslib.exp.evaluate import evaluate
        from tslib.exp.metrics import MetricContext
        bundle = _fake_bundle()
        cfg = _stat_cfg(event_threshold=[1e5])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pred", "ar.npz")
            res = StatisticalRunner("ar", cfg).fit_and_test(bundle, "ar", "unused.pt",
                                                            pred_path=path)
            pred, true, fs, meta = load_predictions(path)
        again = evaluate(pred, true, MetricContext(
            thresholds=meta["event_threshold"], transform=meta["transform"],
            target_cols=meta["target_cols"], forecast_start=fs))
        self.assertEqual(res.evaluation.regression, again.regression)
        self.assertEqual(res.evaluation.event, again.event)

    def test_no_save_when_disabled(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pred", "ar.npz")
            run_strategy("statistic", "ar", _fake_bundle(),
                         _stat_cfg(save_predictions=False), "unused.pt", pred_path=path)
            self.assertFalse(os.path.exists(path))
            self.assertFalse(os.path.exists(os.path.join(d, "pred")))

    def test_no_save_without_forecast_start(self):
        # data without timestamps (CSV loader) cannot make a self-contained
        # prediction file; the run still produces its evaluation
        bundle = _fake_bundle()
        bundle.test_forecast_start = None
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(out):
            path = os.path.join(d, "pred", "ar.npz")
            res = run_strategy("statistic", "ar", bundle, _stat_cfg(), "unused.pt",
                               pred_path=path)
            self.assertFalse(os.path.exists(path))
        self.assertIn("[warn] ar: predictions not saved", out.getvalue())
        self.assertEqual(res.evaluation.regression[0]["n"], 12)

    def test_failed_save_keeps_the_result(self):
        # a prediction file that cannot be written must not cost the run its
        # result (on the neural paths that would mean a retrain)
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(out):
            blocker = os.path.join(d, "pred")
            open(blocker, "w").close()           # a file where the directory goes
            res = run_strategy("statistic", "ar", _fake_bundle(), _stat_cfg(),
                               "unused.pt", pred_path=os.path.join(blocker, "ar.npz"))
        self.assertIn("[warn] ar: predictions not saved to", out.getvalue())
        self.assertEqual(res.evaluation.regression[0]["n"], 12)


class TestNeuralStrategyLightning(unittest.TestCase):
    def setUp(self):
        import torch
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
        self.addCleanup(setattr, strat, "build_model", orig)
        self.strat = strat

    def _bundle(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        x = torch.zeros(8, 4, 1)
        y = torch.full((8, 2, 1), 2.0)
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        return SimpleNamespace(train_loader=loader, val_loader=loader,
                               test_loader=loader, input_size=1,
                               target_indices=[0], target_cols=["p_gt10"],
                               test_forecast_start=np.arange(8, dtype=np.int64) * _DAY_NS)

    def _cfg(self):
        return SimpleNamespace(lr=1e-3, weight_decay=0.0, epochs=1, loss="mse",
                               metrics=None, transform="log10",
                               event_threshold=[10.0], seq_len=4, pred_len=2,
                               label_len=0, early_stop_patience=10,
                               run_name="cell", fold_numb=0, cadence_min=5)

    def test_direct_routes_through_trainer(self):
        with tempfile.TemporaryDirectory() as d:
            pred_path = os.path.join(d, "pred", "toy.npz")
            res = self.strat.run_strategy("direct", "toy", self._bundle(), self._cfg(),
                                          os.path.join(d, "toy.pt"), pred_path=pred_path)
            # the Lightning log directory is keyed on the model, not on an
            # auto-incrementing version_N that concurrent processes would
            # race for
            self.assertTrue(os.path.isdir(
                os.path.join(d, "lightning", "toy")))
            self.assertFalse(os.path.exists(
                os.path.join(d, "lightning", "version_0")))
            pred, true, fs, meta = load_predictions(pred_path)
        self.assertEqual(res.strategy, "direct")
        self.assertEqual([r["granularity"] for r in res.evaluation.event],
                         ["point", "window", "day"])
        self.assertEqual(pred.shape, (8, 2, 1))
        np.testing.assert_array_equal(true, 2.0)
        np.testing.assert_array_equal(fs, np.arange(8, dtype=np.int64) * _DAY_NS)
        self.assertEqual((meta["model"], meta["strategy"]), ("toy", "direct"))

    def test_test_only_recovery_saves_predictions(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt_path = os.path.join(d, "toy.pt")
            trained = self.strat.run_strategy("direct", "toy", self._bundle(), self._cfg(),
                                              ckpt_path)
            pred_path = os.path.join(d, "pred", "toy.npz")
            res = self.strat.test_only_neural("direct", "toy", self._bundle(), self._cfg(),
                                              ckpt_path, os.path.join(d, "toy.ckpt"),
                                              pred_path=pred_path)
            pred, _, _, meta = load_predictions(pred_path)
        self.assertEqual(pred.shape, (8, 2, 1))
        self.assertEqual(meta["model"], "toy")
        self.assertEqual(res.best_val_loss, trained.best_val_loss)
        self.assertEqual(res.evaluation.regression[0]["rmse"],
                         trained.evaluation.regression[0]["rmse"])

    def test_module_arrays_released_after_save(self):
        import torch
        from tslib.exp.lightning_model import ForecastModule
        from tslib.exp.metrics import MetricContext
        m = ForecastModule(torch.nn.Identity(), self._cfg(), MetricContext())
        m.test_pred = np.zeros((2, 2, 1), dtype=np.float32)
        m.test_true = np.zeros((2, 2, 1), dtype=np.float32)
        bundle = SimpleNamespace(target_cols=["p_gt10"],
                                 test_forecast_start=np.arange(2, dtype=np.int64))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "toy.npz")
            self.strat._save_and_release(m, path, bundle, self._cfg(), "toy", "direct")
            self.assertTrue(os.path.exists(path))
        self.assertIsNone(m.test_pred)
        self.assertIsNone(m.test_true)


if __name__ == "__main__":
    unittest.main()


class TestCodeVersion(unittest.TestCase):
    """Prediction files record the code version (spec 4.6); without an
    installed package it was "unknown" in all 1,746 archive files."""

    def setUp(self):
        import tslib.exp.strategy as strat
        self.strat = strat
        strat._tslib_version.cache_clear()
        self.addCleanup(strat._tslib_version.cache_clear)

    def test_falls_back_to_the_package_version_without_an_install(self):
        import importlib.metadata
        from unittest import mock
        import tslib
        with mock.patch.object(self.strat.importlib.metadata, "version",
                               side_effect=importlib.metadata.PackageNotFoundError("tslib")):
            version = self.strat._tslib_version()
        self.assertNotEqual(version, "unknown")
        self.assertTrue(version.startswith(tslib.__version__), version)

    def test_a_git_checkout_adds_its_commit(self):
        from pathlib import Path
        if not (Path(__file__).resolve().parents[2] / ".git").exists():
            self.skipTest("not a git checkout")
        self.assertRegex(self.strat._tslib_version(), r"^\d+\.\d+\.\d+\+g[0-9a-f]{7,}(\.dirty)?$")

    def test_package_version_matches_pyproject(self):
        import tomllib
        from pathlib import Path
        import tslib
        with open(Path(__file__).resolve().parents[2] / "pyproject.toml", "rb") as f:
            self.assertEqual(tomllib.load(f)["project"]["version"], tslib.__version__)
