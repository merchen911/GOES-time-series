import io
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from tslib.exp import exp as expmod
from tslib.exp.evaluate import evaluate
from tslib.exp.lightning_model import TrainResult
from tslib.exp.metrics import MetricContext
from tslib.exp.predictions import prediction_path
from tslib.exp.records import RECORD_FORMAT, result_record


def _cfg(runroot, models):
    # run_name is absolute, so save_dir_name's os.path.join("runs", run_name)
    # resolves to the temp dir (os.path.join drops the "runs" prefix when the
    # second arg is absolute) — keeps the test off the real runs/ tree.
    return SimpleNamespace(seed=42, run_name=os.path.join(runroot, "run"),
                           models=models, sort_metric="rmse",
                           forecast_strategy="direct")


def _fake_bundle():
    return SimpleNamespace(input_size=1, target_indices=[0], target_cols=["p_gt10"],
                           test_forecast_start=np.arange(4, dtype=np.int64))


def _result(name):
    # one channel, four daily windows: one regression row, three event rows
    pred = np.array([2.0, 0.0, 2.0, 0.0]).reshape(4, 1, 1)
    true = np.array([2.0, 2.0, 0.0, 0.0]).reshape(4, 1, 1)
    ctx = MetricContext(thresholds=[10.0], transform="log10", target_cols=["p_gt10"],
                        forecast_start=np.arange(4, dtype=np.int64) * 24 * 3600 * 10**9)
    return TrainResult(model_name=name, best_val_loss=0.5,
                       evaluation=evaluate(pred, true, ctx),
                       ckpt_path="x", strategy="direct")


def _score(d, name):
    return os.path.join(d, "run", "score", name)


class _QuietRun(unittest.TestCase):
    """run_experiment narrates every model and prints both tables; keep that
    out of the test report."""

    def setUp(self):
        out = mock.patch("sys.stdout", io.StringIO())
        out.start()
        self.addCleanup(out.stop)


class TestRunExperimentResume(_QuietRun):
    def _run(self, cfg, calls):
        def fake_run_strategy(strategy, model_name, data_bundle, config, ckpt_path,
                              pred_path=None):
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
            regression, event = self._run(cfg, calls)

            # first run trains all three, in order
            self.assertEqual(calls, ["a", "b", "c"])
            # one persisted record per model, plus the two comparison tables
            rows_dir = _score(d, "rows")
            self.assertEqual(sorted(os.listdir(rows_dir)),
                             ["a.json", "b.json", "c.json"])
            with open(os.path.join(rows_dir, "a.json")) as f:
                self.assertEqual(json.load(f)["format"], RECORD_FORMAT)
            self.assertEqual(len(pd.read_csv(_score(d, "comparison_regression.csv"))), 3)
            self.assertEqual(len(pd.read_csv(_score(d, "comparison_event.csv"))), 9)
            self.assertEqual((len(regression), len(event)), (3, 9))
            self.assertFalse(os.path.exists(_score(d, "comparison.csv")))

            # simulate a crash that lost only model c's row (a, b survived)
            os.remove(os.path.join(rows_dir, "c.json"))
            calls.clear()
            self._run(cfg, calls)
            # resume retrains ONLY c; a and b are skipped from their rows
            self.assertEqual(calls, ["c"])
            self.assertEqual(len(pd.read_csv(_score(d, "comparison_regression.csv"))), 3)

    def test_passes_the_prediction_path_to_every_path(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(d, ["fresh", "recovered"])
            ckpt_dir = os.path.join(d, "run", "ckpt")
            os.makedirs(ckpt_dir, exist_ok=True)
            open(os.path.join(ckpt_dir, "recovered.ckpt"), "w").close()
            open(os.path.join(ckpt_dir, "recovered.done"), "w").close()
            seen = {}

            def fake_run_strategy(strategy, name, bundle, config, ckpt_path, pred_path=None):
                seen[name] = pred_path
                return _result(name)

            def fake_test_only(strategy, name, bundle, config, ckpt_path, ckpt_file,
                               pred_path=None):
                seen[name] = pred_path
                return _result(name)

            with mock.patch.object(expmod, "DataModule") as DM, \
                    mock.patch.object(expmod, "test_only_neural", fake_test_only), \
                    mock.patch.object(expmod, "run_strategy", fake_run_strategy):
                DM.return_value.setup.return_value = _fake_bundle()
                expmod.run_experiment(cfg)

            root = os.path.join(d, "run")
            self.assertEqual(seen, {m: prediction_path(root, m)
                                    for m in ("fresh", "recovered")})

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

            def fake_test_only(strategy, name, bundle, config, ckpt_path, ckpt_file,
                               pred_path=None):
                recovered.append(name)
                return _result(name)

            def fake_run_strategy(strategy, name, bundle, config, ckpt_path, pred_path=None):
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

            def fake_run_strategy(strategy, name, bundle, config, ckpt_path, pred_path=None):
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

            def fake_run_strategy(strategy, name, bundle, config, ckpt_path, pred_path=None):
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


_V1_ROW = {"strategy": "direct", "model": "old", "best_val_loss": 0.2,
           "rmse": 0.1, "mae": 0.05, "tss_p_gt10": 0.0}   # a v1.0.0 row file (no format)


class TestMissingPredictionFile(_QuietRun):
    """A logged row whose prediction file is missing (its save failed) is not
    a finished model: it could never be re-scored."""

    def _logged_cell(self, d, models, with_checkpoint):
        dirs = expmod.save_dir_name(SimpleNamespace(run_name=os.path.join(d, "run")),
                                    makedir=True)
        for name in models:
            stale = result_record(_result(name))
            stale["best_val_loss"] = 0.75
            expmod._write_row(expmod._row_path(dirs, name), stale)
            if with_checkpoint:
                for ext in ("ckpt", "done"):
                    open(os.path.join(dirs["ckpt"], f"{name}.{ext}"), "w").close()
        return dirs

    def _run(self, cfg, tested, trained, out=None):
        def fake_test_only(strategy, name, bundle, config, ckpt_path, ckpt_file,
                           pred_path=None):
            tested.append((name, pred_path))
            return _result(name)

        def fake_run_strategy(strategy, name, bundle, config, ckpt_path, pred_path=None):
            trained.append(name)
            return _result(name)
        with mock.patch.object(expmod, "DataModule") as DM, \
                mock.patch.object(expmod, "test_only_neural", fake_test_only), \
                mock.patch.object(expmod, "run_strategy", fake_run_strategy), \
                mock.patch("sys.stdout", out or io.StringIO()):
            DM.return_value.setup.return_value = _fake_bundle()
            return expmod.run_experiment(cfg)

    def test_regenerated_from_checkpoint_without_retraining(self):
        with tempfile.TemporaryDirectory() as d:
            dirs = self._logged_cell(d, ["a"], with_checkpoint=True)
            tested, trained = [], []
            self._run(_cfg(d, ["a"]), tested, trained)
            self.assertEqual(tested, [("a", prediction_path(dirs["root"], "a"))])
            self.assertEqual(trained, [])
            with open(expmod._row_path(dirs, "a")) as f:
                self.assertEqual(json.load(f)["best_val_loss"], 0.5)   # rewritten

    def test_without_checkpoint_the_row_stays_and_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            dirs = self._logged_cell(d, ["a"], with_checkpoint=False)
            tested, trained, out = [], [], io.StringIO()
            self._run(_cfg(d, ["a"]), tested, trained, out)
            self.assertEqual((tested, trained), ([], []))
            with open(expmod._row_path(dirs, "a")) as f:
                self.assertEqual(json.load(f)["best_val_loss"], 0.75)  # kept
            text = out.getvalue()
            self.assertIn("a.npz", text)
            self.assertIn("no prediction file", text)

    def test_with_saving_off_a_logged_row_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            self._logged_cell(d, ["a"], with_checkpoint=True)
            cfg = _cfg(d, ["a"])
            cfg.save_predictions = False
            tested, trained, out = [], [], io.StringIO()
            self._run(cfg, tested, trained, out)
            self.assertEqual((tested, trained), ([], []))
            self.assertIn("SKIP a", out.getvalue())

    def test_no_regeneration_when_the_data_cannot_make_a_file(self):
        # CSV input without --time_col has no forecast start times, so no
        # prediction file can be written: re-testing on every rerun is useless
        with tempfile.TemporaryDirectory() as d:
            self._logged_cell(d, ["a"], with_checkpoint=True)
            tested, trained, out = [], [], io.StringIO()
            bundle = _fake_bundle()
            bundle.test_forecast_start = None
            with mock.patch.object(expmod, "DataModule") as DM, \
                    mock.patch.object(expmod, "test_only_neural",
                                      lambda *a, **k: tested.append(a[1])), \
                    mock.patch.object(expmod, "run_strategy",
                                      lambda *a, **k: trained.append(a[1])), \
                    mock.patch("sys.stdout", out):
                DM.return_value.setup.return_value = bundle
                expmod.run_experiment(_cfg(d, ["a"]))
            self.assertEqual((tested, trained), ([], []))
            self.assertIn("SKIP a", out.getvalue())

    def test_a_present_prediction_file_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            dirs = self._logged_cell(d, ["a"], with_checkpoint=True)
            os.makedirs(os.path.join(dirs["root"], "pred"))
            open(prediction_path(dirs["root"], "a"), "w").close()
            tested, trained = [], []
            self._run(_cfg(d, ["a"]), tested, trained)
            self.assertEqual((tested, trained), ([], []))


class TestUpgradeFromV1Rows(_QuietRun):
    """A run directory of v1.0.0 holds row files without `format: 2`. They must
    not block a v2 run there: the model is re-tested from its checkpoint (or
    retrained), and other models are not held up."""

    def _run(self, cfg, tested, trained):
        def fake_test_only(strategy, name, bundle, config, ckpt_path, ckpt_file,
                           pred_path=None):
            tested.append(name)
            return _result(name)

        def fake_run_strategy(strategy, name, bundle, config, ckpt_path, pred_path=None):
            trained.append(name)
            return _result(name)
        with mock.patch.object(expmod, "DataModule") as DM, \
                mock.patch.object(expmod, "test_only_neural", fake_test_only), \
                mock.patch.object(expmod, "run_strategy", fake_run_strategy):
            DM.return_value.setup.return_value = _fake_bundle()
            return expmod.run_experiment(cfg)

    def _v1_cell(self, d, with_checkpoint):
        rows = os.path.join(d, "run", "score", "rows")
        os.makedirs(rows)
        with open(os.path.join(rows, "old.json"), "w") as f:
            json.dump(_V1_ROW, f)
        if with_checkpoint:
            ckpt_dir = os.path.join(d, "run", "ckpt")
            os.makedirs(ckpt_dir)
            for name in ("old.ckpt", "old.done"):
                open(os.path.join(ckpt_dir, name), "w").close()
        return rows

    def test_v1_row_with_checkpoint_is_re_tested_not_retrained(self):
        with tempfile.TemporaryDirectory() as d:
            rows = self._v1_cell(d, with_checkpoint=True)
            tested, trained = [], []
            regression, event = self._run(_cfg(d, ["old", "new"]), tested, trained)
            self.assertEqual((tested, trained), (["old"], ["new"]))
            for name in ("old", "new"):
                with open(os.path.join(rows, f"{name}.json")) as f:
                    self.assertEqual(json.load(f)["format"], RECORD_FORMAT)
            self.assertEqual(sorted(set(regression["model"])), ["new", "old"])

    def test_v1_row_without_checkpoint_is_retrained(self):
        with tempfile.TemporaryDirectory() as d:
            self._v1_cell(d, with_checkpoint=False)
            tested, trained = [], []
            self._run(_cfg(d, ["old"]), tested, trained)
            self.assertEqual((tested, trained), ([], ["old"]))

    def test_v1_row_of_a_model_not_run_does_not_stop_the_cell(self):
        with tempfile.TemporaryDirectory() as d:
            self._v1_cell(d, with_checkpoint=False)
            tested, trained = [], []
            out = io.StringIO()
            with mock.patch("sys.stdout", out):
                regression, event = self._run(_cfg(d, ["a", "b"]), tested, trained)
            self.assertEqual(trained, ["a", "b"])
            self.assertEqual(sorted(set(regression["model"])), ["a", "b"])
            self.assertEqual(sorted(set(event["model"])), ["a", "b"])
            self.assertIn("old.json", out.getvalue())          # the skipped row is named

    def test_rows_on_disk_skips_v1_rows(self):
        with tempfile.TemporaryDirectory() as d:
            dirs = expmod.save_dir_name(SimpleNamespace(run_name="run", runs_root=d),
                                        makedir=True)
            expmod._write_row(expmod._row_path(dirs, "new"), result_record(_result("new")))
            expmod._write_row(expmod._row_path(dirs, "old"), _V1_ROW)
            out = io.StringIO()
            with mock.patch("sys.stdout", out):
                self.assertEqual([r["model"] for r in expmod._rows_on_disk(dirs)], ["new"])
            self.assertIn("[warn]", out.getvalue())


class TestBestOnTopSort(unittest.TestCase):
    def _frame(self, sort_metric):
        rows = [
            {"model": "hi", "best_val_loss": 0.9, "tss_p_gt10": 0.80,
             "rmse": 0.5, "far_p_gt10": 0.3, "logr": 0.9, "seds": 0.8,
             "sedi": 0.7, "msa": 90.0},
            {"model": "lo", "best_val_loss": 0.1, "tss_p_gt10": 0.20,
             "rmse": 0.1, "far_p_gt10": 0.1, "logr": 0.2, "seds": 0.1,
             "sedi": 0.1, "msa": 10.0},
        ]
        return expmod._rows_to_frame(rows, sort_metric)

    def test_skill_score_sorts_descending_best_on_top(self):
        # tss: higher is better -> the 0.80 model must be on top
        df = self._frame("tss_p_gt10")
        self.assertEqual(list(df["model"]), ["hi", "lo"])

    def test_new_higher_is_better_metrics_sort_descending(self):
        for key in ("logr", "seds", "sedi"):
            df = self._frame(key)
            self.assertEqual(list(df["model"]), ["hi", "lo"], msg=key)

    def test_far_sorts_ascending_best_on_top(self):
        # far: lower is better -> the 0.1 model on top
        df = self._frame("far_p_gt10")
        self.assertEqual(list(df["model"]), ["lo", "hi"])

    def test_error_metric_sorts_ascending_best_on_top(self):
        for key in ("rmse", "msa", "best_val_loss"):
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


class TestConcurrentWriters(unittest.TestCase):
    """The scheduler and the main.py processes of one cell rewrite the same
    comparison tables; a shared temp name made one writer's os.replace fail
    (FileNotFoundError) or interleave two writes in one file."""

    def _hammer(self, write, path, payloads, n_threads=8, rounds=20):
        import threading
        errors = []

        def work(k):
            for _ in range(rounds):
                try:
                    write(payloads[k % len(payloads)], path)
                except Exception as e:  # noqa: BLE001 - collected for the assertion
                    errors.append(repr(e))
        threads = [threading.Thread(target=work, args=(k,)) for k in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return errors

    def test_concurrent_table_writes_never_fail_or_mix(self):
        frames = [pd.DataFrame({"model": [f"m{k}"] * 4000, "v": np.arange(4000) + k})
                  for k in range(2)]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "comparison_regression.csv")
            errors = self._hammer(expmod._write_csv_atomic, path, frames)
            self.assertEqual(errors, [])
            back = pd.read_csv(path)
            self.assertEqual(len(back), 4000)
            self.assertEqual(back["model"].nunique(), 1)       # one writer's table, whole
            self.assertEqual([f for f in os.listdir(d) if f.endswith(".tmp")], [])

    def test_concurrent_row_writes_never_fail(self):
        rows = [{"format": 2, "model": "a", "pad": "x" * 200_000},
                {"format": 2, "model": "a", "pad": "y" * 200_000}]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "rows", "a.json")
            errors = self._hammer(lambda row, p: expmod._write_row(p, row), path, rows)
            self.assertEqual(errors, [])
            with open(path) as f:
                self.assertIn(json.load(f)["pad"][0], "xy")
            self.assertEqual(os.listdir(os.path.dirname(path)), ["a.json"])


class TestRunsRoot(unittest.TestCase):
    def test_default_is_runs(self):
        # configs that predate --runs_root (no attribute) and configs that leave
        # it unset both resolve to the historical "runs" directory
        dirs = expmod.save_dir_name(SimpleNamespace(run_name="r"))
        self.assertEqual(dirs["root"], os.path.join("runs", "r"))
        dirs = expmod.save_dir_name(SimpleNamespace(run_name="r", runs_root=None))
        self.assertEqual(dirs["root"], os.path.join("runs", "r"))

    def test_runs_root_redirects_every_directory(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = SimpleNamespace(run_name="bench/c", runs_root=d)
            dirs = expmod.save_dir_name(cfg, makedir=True)
            for key, path in dirs.items():
                self.assertTrue(path.startswith(os.path.join(d, "bench", "c")),
                                f"{key} -> {path}")
                self.assertTrue(os.path.isdir(path), key)


class TestRowFileMerge(_QuietRun):
    """The comparison tables are rebuilt from the row files on disk, so processes
    that share a run directory (v004 runs one model per process per GPU) merge
    into one table instead of each overwriting it with only its own row."""

    def _dirs(self, d):
        return expmod.save_dir_name(
            SimpleNamespace(run_name="run", runs_root=d), makedir=True)

    def test_rows_on_disk_reads_every_row_file(self):
        with tempfile.TemporaryDirectory() as d:
            dirs = self._dirs(d)
            for name in ("b", "a"):
                expmod._write_row(expmod._row_path(dirs, name),
                                  result_record(_result(name)))
            rows = expmod._rows_on_disk(dirs)
            self.assertEqual([r["model"] for r in rows], ["a", "b"])  # name order

    def test_rows_on_disk_empty_before_any_row(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(expmod._rows_on_disk(self._dirs(d)), [])

    def test_rows_on_disk_skips_unreadable_row(self):
        with tempfile.TemporaryDirectory() as d:
            dirs = self._dirs(d)
            expmod._write_row(expmod._row_path(dirs, "good"),
                              result_record(_result("good")))
            with open(expmod._row_path(dirs, "torn"), "w") as f:
                f.write('{"model": "to')       # half-written, not valid JSON
            with open(expmod._row_path(dirs, "tmp") + ".tmp", "w") as f:
                f.write("{}")                   # a tmp file is not a row
            self.assertEqual([r["model"] for r in expmod._rows_on_disk(dirs)],
                             ["good"])

    def test_comparison_includes_rows_written_by_another_process(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = SimpleNamespace(seed=42, run_name="run", runs_root=d,
                                  models=["a"], sort_metric="rmse",
                                  forecast_strategy="direct")
            # another process (another GPU) already finished model "other" in
            # this same run directory
            dirs = self._dirs(d)
            expmod._write_row(expmod._row_path(dirs, "other"),
                              result_record(_result("other")))

            def fake_run_strategy(strategy, name, bundle, config, ckpt_path, pred_path=None):
                return _result(name)

            with mock.patch.object(expmod, "DataModule") as DM, \
                    mock.patch.object(expmod, "run_strategy", fake_run_strategy):
                DM.return_value.setup.return_value = _fake_bundle()
                regression, event = expmod.run_experiment(cfg)

            # this process only ran "a", yet both tables have both models
            reg_path, event_path = expmod.comparison_paths(dirs)
            self.assertEqual((reg_path, event_path),
                             (_score(d, "comparison_regression.csv"),
                              _score(d, "comparison_event.csv")))
            for df in (regression, event, pd.read_csv(reg_path), pd.read_csv(event_path)):
                self.assertEqual(sorted(set(df["model"])), ["a", "other"])


if __name__ == "__main__":
    unittest.main()
