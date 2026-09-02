import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from tslib.exp import exp as expmod
from tslib.exp.lightning_model import TrainResult


def _cfg(runroot, models):
    # run_name is absolute, so save_dir_name's os.path.join("runs", run_name)
    # resolves to the temp dir (os.path.join drops the "runs" prefix when the
    # second arg is absolute) — keeps the test off the real runs/ tree.
    return SimpleNamespace(seed=42, run_name=os.path.join(runroot, "run"),
                           models=models, sort_metric="tss_p_gt10",
                           forecast_strategy="direct")


def _fake_bundle():
    return SimpleNamespace(input_size=1, target_indices=[0],
                           target_cols=["p_gt10"])


def _result(name):
    return TrainResult(model_name=name, best_val_loss=0.5,
                       metrics={"rmse": 0.1, "tss_p_gt10": 0.3},
                       ckpt_path="x", strategy="direct")


class TestRunExperimentResume(unittest.TestCase):
    def _run(self, cfg, calls):
        def fake_run_strategy(strategy, model_name, data_bundle, config, ckpt_path):
            calls.append(model_name)
            return _result(model_name)
        with mock.patch.object(expmod, "DataModule") as DM, \
                mock.patch.object(expmod, "run_strategy", fake_run_strategy):
            DM.return_value.setup.return_value = _fake_bundle()
            return expmod.run_experiment(cfg)

    def test_persists_per_model_and_resumes_without_retraining(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(d, ["a", "b", "c"])
            calls = []
            comp, comp_path = self._run(cfg, calls)

            # first run trains all three, in order
            self.assertEqual(calls, ["a", "b", "c"])
            # one persisted row per model, plus a comparison.csv with 3 rows
            rows_dir = os.path.join(d, "run", "score", "rows")
            self.assertEqual(sorted(os.listdir(rows_dir)),
                             ["a.json", "b.json", "c.json"])
            self.assertEqual(len(pd.read_csv(comp_path)), 3)

            # simulate a crash that lost only model c's row (a, b survived)
            os.remove(os.path.join(rows_dir, "c.json"))
            calls.clear()
            self._run(cfg, calls)
            # resume retrains ONLY c; a and b are skipped from their rows
            self.assertEqual(calls, ["c"])
            self.assertEqual(len(pd.read_csv(comp_path)), 3)

    def test_recovers_from_completed_unlogged_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(d, ["a"])
            # a trained checkpoint AND its completion marker exist, but no
            # logged row -> training finished, crash before logging -> recover
            ckpt_dir = os.path.join(d, "run", "ckpt")
            os.makedirs(ckpt_dir, exist_ok=True)
            open(os.path.join(ckpt_dir, "a.ckpt"), "w").close()
            open(os.path.join(ckpt_dir, "a.done"), "w").close()

            recovered, trained = [], []

            def fake_test_only(strategy, name, bundle, config, ckpt_path, ckpt_file):
                recovered.append(name)
                return _result(name)

            def fake_run_strategy(strategy, name, bundle, config, ckpt_path):
                trained.append(name)
                return _result(name)

            with mock.patch.object(expmod, "DataModule") as DM, \
                    mock.patch.object(expmod, "test_only_neural", fake_test_only), \
                    mock.patch.object(expmod, "run_strategy", fake_run_strategy):
                DM.return_value.setup.return_value = _fake_bundle()
                expmod.run_experiment(cfg)

            self.assertEqual(recovered, ["a"])   # recovered via test-only
            self.assertEqual(trained, [])        # never retrained
            self.assertTrue(os.path.exists(
                os.path.join(d, "run", "score", "rows", "a.json")))

    def test_orphan_without_marker_is_discarded_and_retrained(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(d, ["a"])
            # a checkpoint with NO completion marker = interrupted mid-training
            # orphan; it must NOT be recovered as final -> retrain, after first
            # clearing the stale orphan file.
            ckpt_dir = os.path.join(d, "run", "ckpt")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt = os.path.join(ckpt_dir, "a.ckpt")
            open(ckpt, "w").close()

            recovered, trained = [], []

            def fake_test_only(*a, **k):
                recovered.append(a[1])
                return _result("a")

            def fake_run_strategy(strategy, name, bundle, config, ckpt_path):
                # by the time we retrain, the orphan must already be cleared
                self.assertFalse(os.path.exists(ckpt),
                                 "orphan .ckpt not cleared before retrain")
                trained.append(name)
                return _result(name)

            with mock.patch.object(expmod, "DataModule") as DM, \
                    mock.patch.object(expmod, "test_only_neural", fake_test_only), \
                    mock.patch.object(expmod, "run_strategy", fake_run_strategy):
                DM.return_value.setup.return_value = _fake_bundle()
                expmod.run_experiment(cfg)

            self.assertEqual(recovered, [])      # never recovered the orphan
            self.assertEqual(trained, ["a"])     # retrained from scratch
            # a fresh completion marker is dropped after retraining
            self.assertTrue(os.path.exists(os.path.join(ckpt_dir, "a.done")))

    def test_recover_failure_falls_back_to_retrain(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(d, ["a"])
            ckpt_dir = os.path.join(d, "run", "ckpt")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt = os.path.join(ckpt_dir, "a.ckpt")
            open(ckpt, "w").close()
            open(os.path.join(ckpt_dir, "a.done"), "w").close()  # recovery attempted

            trained = []

            def boom(*a, **k):
                raise RuntimeError("bad checkpoint")

            def fake_run_strategy(strategy, name, bundle, config, ckpt_path):
                # the fallback retrain must clear the bad orphan first
                self.assertFalse(os.path.exists(ckpt),
                                 "bad orphan .ckpt not cleared before fallback retrain")
                trained.append(name)
                return _result(name)

            with mock.patch.object(expmod, "DataModule") as DM, \
                    mock.patch.object(expmod, "test_only_neural", boom), \
                    mock.patch.object(expmod, "run_strategy", fake_run_strategy):
                DM.return_value.setup.return_value = _fake_bundle()
                expmod.run_experiment(cfg)

            # recovery raised -> fell back to a full retrain, cell survived
            self.assertEqual(trained, ["a"])
            self.assertTrue(os.path.exists(
                os.path.join(d, "run", "score", "rows", "a.json")))


class TestBestOnTopSort(unittest.TestCase):
    def _frame(self, sort_metric):
        rows = [
            {"model": "hi", "best_val_loss": 0.9, "tss_p_gt10": 0.80,
             "rmse": 0.5, "far_p_gt10": 0.3},
            {"model": "lo", "best_val_loss": 0.1, "tss_p_gt10": 0.20,
             "rmse": 0.1, "far_p_gt10": 0.1},
        ]
        return expmod._rows_to_frame(rows, sort_metric)

    def test_skill_score_sorts_descending_best_on_top(self):
        # tss: higher is better -> the 0.80 model must be on top
        df = self._frame("tss_p_gt10")
        self.assertEqual(list(df["model"]), ["hi", "lo"])

    def test_far_sorts_ascending_best_on_top(self):
        # far: lower is better -> the 0.1 model on top
        df = self._frame("far_p_gt10")
        self.assertEqual(list(df["model"]), ["lo", "hi"])

    def test_error_metric_sorts_ascending_best_on_top(self):
        for key in ("rmse", "best_val_loss"):
            df = self._frame(key)
            self.assertEqual(list(df["model"]), ["lo", "hi"], msg=key)

    def test_nan_rows_stay_at_bottom_regardless_of_direction(self):
        rows = [
            {"model": "good", "tss_p_gt10": 0.9},
            {"model": "failed", "tss_p_gt10": float("nan")},
            {"model": "mid", "tss_p_gt10": 0.5},
        ]
        df = expmod._rows_to_frame(rows, "tss_p_gt10")
        self.assertEqual(list(df["model"]), ["good", "mid", "failed"])


class TestCkptHygiene(unittest.TestCase):
    def test_clear_ckpts_only_touches_this_model(self):
        with tempfile.TemporaryDirectory() as d:
            # 'segrnn' must not sweep 'segrnn_thuml' (prefix collision) but must
            # clear its own canonical + versioned ckpt + done marker.
            for name in ("segrnn.ckpt", "segrnn-v1.ckpt", "segrnn.done",
                         "segrnn_thuml.ckpt", "segrnn_thuml.done"):
                open(os.path.join(d, name), "w").close()
            expmod._clear_ckpts(d, "segrnn")
            left = set(os.listdir(d))
            self.assertEqual(left, {"segrnn_thuml.ckpt", "segrnn_thuml.done"})

    def test_write_csv_atomic_leaves_no_tmp_and_roundtrips(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "comparison.csv")
            df = pd.DataFrame([{"model": "a", "tss_p_gt10": 0.8},
                               {"model": "b", "tss_p_gt10": 0.2}])
            expmod._write_csv_atomic(df, path)
            self.assertTrue(os.path.exists(path))
            self.assertFalse(os.path.exists(path + ".tmp"))
            self.assertEqual(list(pd.read_csv(path)["model"]), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
