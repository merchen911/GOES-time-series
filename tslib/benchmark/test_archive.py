import collections
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tslib.benchmark import archive
from tslib.benchmark.driver import parse_cell_name
from tslib.exp.evaluate import evaluate
from tslib.exp.lightning_model import TrainResult
from tslib.exp.metrics import MetricContext
from tslib.exp.predictions import save_predictions
from tslib.exp.records import RECORD_FORMAT

_REPO = Path(__file__).resolve().parents[2]
_UNI_B = {"track": "uni_b", "seq_len": 288, "pred_len": 36, "fold": 0,
          "strategy": "direct"}
_UNI_B_NAME = "uni_b_seq288_pred36_f0_direct"


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _old_tree(root):
    """An old run tree as v003/v004 left it: cells with checkpoints, Lightning's
    versioned siblings, completion markers and logs, plus folders and files
    that are not cells."""
    bench = Path(root) / "bench"
    for cell, models in [("uni_b_seq288_pred36_f0_direct", ["segrnn_thuml", "dlinear"]),
                         ("multi_seq288_pred144_f1_direct", ["patchtst"]),
                         ("uni_a_seq864_pred144_f0_direct", ["tide"]),
                         ("uni_a_seq288_pred144_f0_direct", ["dlinear"])]:
        for m in models:
            _touch(bench / cell / "ckpt" / f"{m}.ckpt")
            _touch(bench / cell / "ckpt" / f"{m}.done")
    _touch(bench / "uni_b_seq288_pred36_f0_direct" / "ckpt" / "dlinear-v1.ckpt")
    _touch(bench / "uni_b_seq288_pred36_f0_direct" / "ckpt" / "lightning" / "dlinear"
           / "metrics.csv")
    _touch(bench / "uni_b_seq288_pred36_f0_direct" / "score" / "rows" / "dlinear.json")
    _touch(bench / "uni_b_seq288_pred144_f0_direct" / "score" / "comparison.csv")  # no ckpt
    _touch(bench / "smoke" / "ckpt" / "dlinear.ckpt")                              # not a cell
    _touch(bench / "results_master.csv")
    return bench


class TestDiscoverCheckpoints(unittest.TestCase):
    def test_discover_skips_versioned_and_non_cells(self):
        with tempfile.TemporaryDirectory() as d:
            bench = _old_tree(d)
            found = archive.discover_checkpoints(bench)
        expected = [  # (fold, seq_len, pred_len, track, model) order
            ("uni_b_seq288_pred36_f0_direct", "dlinear"),
            ("uni_b_seq288_pred36_f0_direct", "segrnn_thuml"),
            ("uni_a_seq288_pred144_f0_direct", "dlinear"),
            ("uni_a_seq864_pred144_f0_direct", "tide"),
            ("multi_seq288_pred144_f1_direct", "patchtst"),
        ]
        self.assertEqual([(archive.cell_name(c), m) for c, m, _ in found], expected)
        for cell, model, ckpt in found:
            name = archive.cell_name(cell)
            self.assertEqual(cell, parse_cell_name(name))
            self.assertEqual(ckpt, bench / name / "ckpt" / f"{model}.ckpt")

    def test_group_cells_keeps_the_sorted_order(self):
        with tempfile.TemporaryDirectory() as d:
            bench = _old_tree(d)
            cells = archive.group_cells(archive.discover_checkpoints(bench))
        self.assertEqual([archive.cell_name(c) for c, _ in cells],
                         ["uni_b_seq288_pred36_f0_direct",
                          "uni_a_seq288_pred144_f0_direct",
                          "uni_a_seq864_pred144_f0_direct",
                          "multi_seq288_pred144_f1_direct"])
        self.assertEqual([m for m, _ in cells[0][1]], ["dlinear", "segrnn_thuml"])


class TestArchiveConfig(unittest.TestCase):
    def test_archive_config_values(self):
        out_bench = Path("/tmp/x/runs/archive/v003/bench")
        cfg = archive.archive_config(_UNI_B, ["dlinear", "segrnn_thuml"], out_bench)
        self.assertEqual(cfg.event_threshold, [1e-5])
        self.assertEqual(cfg.seg_len, 24)
        self.assertEqual(cfg.batch_size, 64)
        self.assertEqual(cfg.transform, "log10")
        self.assertIs(cfg.save_predictions, True)
        self.assertEqual(cfg.num_workers, 8)
        self.assertEqual(cfg.models, ["dlinear", "segrnn_thuml"])
        self.assertEqual(cfg.target_col, "xrs_long")
        self.assertEqual((cfg.seq_len, cfg.pred_len, cfg.fold_numb, cfg.forecast_strategy),
                         (288, 36, 0, "direct"))
        self.assertEqual(Path(cfg.runs_root), out_bench.parent)
        self.assertEqual(Path(cfg.runs_root) / cfg.run_name, out_bench / _UNI_B_NAME)

    def test_multi_track_has_both_channels(self):
        cell = {**_UNI_B, "track": "multi", "pred_len": 288}
        cfg = archive.archive_config(cell, ["patchtst"], Path("/tmp/x/bench"))
        self.assertEqual(cfg.event_threshold, [10.0, 1e-5])
        self.assertEqual(cfg.target_cols, ["p_gt10", "xrs_long"])
        self.assertEqual(len(cfg.channels), 2)

    def test_out_bench_must_be_named_bench(self):
        # run_name is bench/<cell>, so any other last folder would put the
        # output beside --out instead of inside it
        with self.assertRaisesRegex(ValueError, "bench"):
            archive.archive_config(_UNI_B, ["dlinear"], Path("/tmp/x/runs/archive/v003"))


class TestShard(unittest.TestCase):
    def test_shard_partition(self):
        with tempfile.TemporaryDirectory() as d:
            cells = archive.group_cells(archive.discover_checkpoints(_old_tree(d)))
        names = [archive.cell_name(c) for c, _ in cells]
        self.assertEqual(archive.select_shard(cells, 0, 1), cells)
        for n in (1, 2, 3, 4, 7):
            seen = collections.Counter(
                archive.cell_name(c) for i in range(n)
                for c, _ in archive.select_shard(cells, i, n))
            self.assertEqual(set(seen), set(names), n)
            self.assertTrue(all(v == 1 for v in seen.values()), (n, seen))
        # round robin over the sorted cells
        self.assertEqual([archive.cell_name(c) for c, _ in archive.select_shard(cells, 1, 2)],
                         names[1::2])

    def test_parse_shard(self):
        self.assertEqual(archive.parse_shard("0/1"), (0, 1))
        self.assertEqual(archive.parse_shard("3/4"), (3, 4))
        for bad in ("4/4", "-1/4", "0/0", "1", "a/b", "1/2/3"):
            with self.assertRaises(ValueError, msg=bad):
                archive.parse_shard(bad)


def _result(name, best_val_loss=0.25):
    pred = np.array([2.0, 0.0, 2.0, 0.0]).reshape(4, 1, 1)
    true = np.array([2.0, 2.0, 0.0, 0.0]).reshape(4, 1, 1)
    ctx = MetricContext(thresholds=[1e-5], transform="log10", target_cols=["xrs_long"],
                        forecast_start=np.arange(4, dtype=np.int64) * 24 * 3600 * 10**9)
    return TrainResult(model_name=name, best_val_loss=best_val_loss,
                       evaluation=evaluate(pred, true, ctx), ckpt_path="x",
                       strategy="direct")


def _write_npz(path):
    save_predictions(path, np.zeros((2, 3, 1)), np.zeros((2, 3, 1)),
                     np.arange(2, dtype=np.int64), {"model": Path(path).stem})


class _FakeInference:
    """Stands in for `test_only_neural`: records each call and writes the
    prediction file unless the model is listed in `fail` or `no_file`."""

    def __init__(self, fail=(), no_file=()):
        self.calls = []
        self.fail, self.no_file = set(fail), set(no_file)

    def __call__(self, strategy, model_name, data_bundle, config, ckpt_path,
                 ckpt_file, pred_path=None):
        self.calls.append({"strategy": strategy, "model": model_name,
                           "bundle": data_bundle, "ckpt_file": ckpt_file,
                           "pred_path": pred_path})
        if model_name in self.fail:
            raise RuntimeError(f"cannot load {model_name}")
        if model_name not in self.no_file:
            _write_npz(pred_path)
        return _result(model_name)


class TestDumpCell(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.src_ckpt = root / "old" / "bench" / _UNI_B_NAME / "ckpt"
        self.items = [(m, _touch(self.src_ckpt / f"{m}.ckpt"))
                      for m in ("dlinear", "segrnn_thuml")]
        self.out_bench = root / "archive" / "v003" / "bench"
        self.cell_dir = self.out_bench / _UNI_B_NAME
        out = mock.patch("sys.stdout", io.StringIO())
        out.start()
        self.addCleanup(out.stop)

    def _dump(self, fake):
        with mock.patch.object(archive, "DataModule") as DM, \
                mock.patch.object(archive, "test_only_neural", fake):
            DM.return_value.setup.return_value = SimpleNamespace(name="bundle")
            done = archive.dump_cell(_UNI_B, self.items, self.out_bench)
        return done, DM

    def _row(self, model):
        return self.cell_dir / "score" / "rows" / f"{model}.json"

    def _npz(self, model):
        return self.cell_dir / "pred" / f"{model}.npz"

    def test_dumps_every_model_from_one_data_load(self):
        fake = _FakeInference()
        done, DM = self._dump(fake)
        self.assertEqual(done, ["dlinear", "segrnn_thuml"])
        self.assertEqual(DM.call_count, 1)
        cfg = DM.call_args.args[0]
        self.assertEqual(Path(cfg.runs_root) / cfg.run_name, self.cell_dir)
        for call, (model, ckpt) in zip(fake.calls, self.items):
            self.assertEqual(call["model"], model)
            self.assertEqual(call["strategy"], "direct")
            self.assertEqual(call["bundle"].name, "bundle")
            self.assertEqual(Path(call["ckpt_file"]), ckpt)
            self.assertEqual(Path(call["pred_path"]), self._npz(model))
            row = json.loads(self._row(model).read_text())
            self.assertEqual(row["format"], RECORD_FORMAT)
            self.assertEqual((row["model"], row["best_val_loss"]), (model, 0.25))
        # nothing is written next to the source checkpoints
        self.assertEqual(sorted(p.name for p in self.src_ckpt.iterdir()),
                         ["dlinear.ckpt", "segrnn_thuml.ckpt"])

    def test_skips_models_already_dumped(self):
        _write_npz(str(self._npz("dlinear")))
        _touch(self._row("dlinear")).write_text("{}")
        fake = _FakeInference()
        done, _ = self._dump(fake)
        self.assertEqual(done, ["segrnn_thuml"])
        self.assertEqual([c["model"] for c in fake.calls], ["segrnn_thuml"])
        self.assertEqual(self._row("dlinear").read_text(), "{}")

    def test_a_dumped_cell_does_not_load_its_data(self):
        for m, _ in self.items:
            _write_npz(str(self._npz(m)))
            _touch(self._row(m))
        done, DM = self._dump(_FakeInference())
        self.assertEqual(done, [])
        DM.assert_not_called()

    def test_prediction_file_without_its_row_is_dumped_again(self):
        # interrupted between the two writes: the row (with best_val_loss from
        # the checkpoint) would otherwise never be written
        _write_npz(str(self._npz("dlinear")))
        fake = _FakeInference()
        done, _ = self._dump(fake)
        self.assertEqual(done, ["dlinear", "segrnn_thuml"])
        self.assertTrue(self._row("dlinear").exists())

    def test_a_failing_model_does_not_stop_the_cell(self):
        fake = _FakeInference(fail={"dlinear"})
        with mock.patch("sys.stderr", io.StringIO()), \
                self.assertRaisesRegex(RuntimeError, "dlinear"):
            self._dump(fake)
        self.assertEqual([c["model"] for c in fake.calls], ["dlinear", "segrnn_thuml"])
        self.assertFalse(self._row("dlinear").exists())
        self.assertTrue(self._row("segrnn_thuml").exists())
        self.assertTrue(self._npz("segrnn_thuml").exists())

    def test_an_unsaved_prediction_file_is_a_failure(self):
        # test_only_neural only warns when the save fails; the dump exists for the file
        fake = _FakeInference(no_file={"dlinear"})
        with mock.patch("sys.stderr", io.StringIO()), \
                self.assertRaisesRegex(RuntimeError, "dlinear"):
            self._dump(fake)
        self.assertFalse(self._row("dlinear").exists())
        self.assertTrue(self._row("segrnn_thuml").exists())


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "dump_predictions_script", _REPO / "scripts" / "dump_predictions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDumpScript(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.src = _old_tree(self.root / "old")
        self.out = self.root / "archive" / "v003" / "bench"
        self.calls = []

    def _main(self, *extra, fail=()):
        def fake_dump(cell, items, out_bench):
            name = archive.cell_name(cell)
            self.calls.append((name, [m for m, _ in items], Path(out_bench)))
            if name in fail:
                raise RuntimeError(f"{name} broke")
            return [m for m, _ in items]

        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(archive, "dump_cell", fake_dump), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = _load_script().main(["--src", str(self.src), "--out", str(self.out),
                                        *extra])
        return code, out.getvalue(), err.getvalue()

    def test_dumps_every_cell_in_sorted_order(self):
        code, out, _ = self._main()
        self.assertEqual(code, 0, out)
        self.assertEqual([c[0] for c in self.calls],
                         ["uni_b_seq288_pred36_f0_direct",
                          "uni_a_seq288_pred144_f0_direct",
                          "uni_a_seq864_pred144_f0_direct",
                          "multi_seq288_pred144_f1_direct"])
        self.assertTrue(all(c[2] == self.out for c in self.calls))
        self.assertEqual(self.calls[0][1], ["dlinear", "segrnn_thuml"])

    def test_shard_takes_every_nth_cell(self):
        self.assertEqual(self._main("--shard", "1/2")[0], 0)
        self.assertEqual([c[0] for c in self.calls],
                         ["uni_a_seq288_pred144_f0_direct",
                          "multi_seq288_pred144_f1_direct"])

    def test_only_and_models_filter(self):
        code, _, _ = self._main("--only", "uni_b_seq288_pred36_f0_direct",
                                "uni_a_seq288_pred144_f0_direct",
                                "--models", "segrnn_thuml")
        self.assertEqual(code, 0)
        # uni_a_seq288 has no segrnn_thuml checkpoint, so it drops out
        self.assertEqual([(c[0], c[1]) for c in self.calls],
                         [("uni_b_seq288_pred36_f0_direct", ["segrnn_thuml"])])

    def test_unknown_filter_names_are_an_error(self):
        code, _, err = self._main("--only", "uni_b_seq288_pred36_f9_direct")
        self.assertEqual(code, 1)
        self.assertIn("uni_b_seq288_pred36_f9_direct", err)
        code, _, err = self._main("--models", "no_such_model")
        self.assertEqual(code, 1)
        self.assertIn("no_such_model", err)
        self.assertEqual(self.calls, [])

    def test_a_failed_cell_is_reported_and_the_others_still_run(self):
        code, _, err = self._main(fail={"uni_a_seq288_pred144_f0_direct"})
        self.assertEqual(code, 1)
        self.assertIn("uni_a_seq288_pred144_f0_direct broke", err)
        self.assertEqual(len(self.calls), 4)

    def test_out_must_end_in_bench(self):
        self.out = self.root / "archive" / "v003"
        with self.assertRaisesRegex(ValueError, "bench"):
            self._main()
        self.assertEqual(self.calls, [])

    def test_out_must_not_be_in_the_source_tree(self):
        for out in (self.src, self.src / "archive" / "bench"):
            self.out = out
            with self.assertRaisesRegex(ValueError, "read-only"):
                self._main()
        self.assertEqual(self.calls, [])

    def test_bad_shard_is_a_usage_error(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            _load_script().parse_args(["--src", "a", "--out", "b/bench", "--shard", "4/4"])


if __name__ == "__main__":
    unittest.main()
