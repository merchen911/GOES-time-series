import contextlib
import importlib.util
import io
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tslib.benchmark.driver import rebuild_master_from_dirs
from tslib.exp.evaluate import evaluate
from tslib.exp.lightning_model import TrainResult
from tslib.exp.metrics import MetricContext
from tslib.exp.predictions import prediction_path, save_predictions
from tslib.exp.records import event_columns, regression_columns, result_record
from tslib.exp.rescore import merge_masters, rescore_cell

_HOUR_NS = 3600 * 10**9
_TARGETS = ["p_gt10", "xrs_long"]
_THRESHOLDS = [10.0, 1e-5]
_REPO = Path(__file__).resolve().parents[2]


def _arrays(seed=0, n=96, p=6):
    """Two log10 channels (N=96 windows 2 h apart, so the 24 h cut keeps ~8) with
    observed values scattered around each threshold (log10: 1 and -5)."""
    rng = np.random.default_rng(seed)
    true = np.stack([rng.normal(0.0, 1.0, (n, p)), rng.normal(-5.0, 1.0, (n, p))],
                    axis=-1).astype(np.float32)
    pred = (true + rng.normal(0.0, 0.8, true.shape)).astype(np.float32)
    forecast_start = np.arange(n, dtype=np.int64) * 2 * _HOUR_NS
    return pred, true, forecast_start


def _meta(model, strategy="direct", **over):
    return {"model": model, "strategy": strategy, "transform": "log10",
            "target_cols": _TARGETS, "event_threshold": _THRESHOLDS,
            "seq_len": 288, "pred_len": 6, "fold_numb": 0, "cadence_min": 5,
            **over}


def _write_pred(cell_dir, model, arrays, strategy="direct"):
    pred, true, forecast_start = arrays
    save_predictions(prediction_path(str(cell_dir), model), pred, true,
                     forecast_start, _meta(model, strategy))


def _expected(model, arrays, strategy="direct", best_val_loss=float("nan")):
    """The record the training path would write for these predictions."""
    pred, true, forecast_start = arrays
    ctx = MetricContext(thresholds=_THRESHOLDS, transform="log10",
                        target_cols=_TARGETS, forecast_start=forecast_start)
    return result_record(TrainResult(model_name=model, best_val_loss=best_val_loss,
                                     evaluation=evaluate(pred, true, ctx),
                                     ckpt_path="", strategy=strategy))


def _row_file(cell_dir, model):
    return Path(cell_dir) / "score" / "rows" / f"{model}.json"


def _canon(record):
    """Text form for comparing records that hold nan (nan != nan)."""
    return json.dumps(record, sort_keys=True)


def _read_row(cell_dir, model):
    return json.loads(_row_file(cell_dir, model).read_text())


def _with_nan_channel(arrays):
    pred, true, fs = arrays
    pred = pred.copy()
    pred[3, 2, 0] = np.nan
    return pred, true, fs


class TestRescoreCell(unittest.TestCase):
    def test_rescore_matches_inline_evaluation(self):
        a, b = _arrays(1), _with_nan_channel(_arrays(2))
        with tempfile.TemporaryDirectory() as d:
            cell = Path(d) / "multi_seq288_pred6_f0_direct"
            _write_pred(cell, "dlinear", a)
            _write_pred(cell, "lstm", b, strategy="recursive")
            self.assertEqual(rescore_cell(cell), 2)

            want_a = _expected("dlinear", a)
            want_b = _expected("lstm", b, strategy="recursive")
            # the comparison is only meaningful if the fixtures exercise a nan
            self.assertTrue(any(math.isnan(r["tss"]) for r in want_b["event"]))
            self.assertTrue(any(not math.isnan(r["tss"]) for r in want_a["event"]))
            self.assertEqual(_canon(_read_row(cell, "dlinear")), _canon(want_a))
            self.assertEqual(_canon(_read_row(cell, "lstm")), _canon(want_b))

            score = cell / "score"
            reg = pd.read_csv(score / "comparison_regression.csv")
            ev = pd.read_csv(score / "comparison_event.csv")
            self.assertEqual(list(reg.columns),
                             [c for c in regression_columns() if c not in
                              ("track", "seq_len", "pred_len", "fold")])
            self.assertEqual(sorted(set(reg["model"])), ["dlinear", "lstm"])
            self.assertEqual(len(reg), 2 * 2)          # models x channels
            self.assertEqual(len(ev), 2 * 2 * 3)       # models x channels x granularities

    def test_keeps_best_val_loss(self):
        a, b = _arrays(1), _arrays(2)
        with tempfile.TemporaryDirectory() as d:
            cell = Path(d) / "multi_seq288_pred6_f0_direct"
            _write_pred(cell, "dlinear", a)
            _write_pred(cell, "tsmixer", b)
            # a row from the original run: its training loss, and metrics that
            # the re-scoring must replace
            stale = _expected("dlinear", _arrays(9), best_val_loss=0.123)
            _row_file(cell, "dlinear").parent.mkdir(parents=True)
            _row_file(cell, "dlinear").write_text(json.dumps(stale))

            self.assertEqual(rescore_cell(cell), 2)
            kept = _read_row(cell, "dlinear")
            self.assertEqual(kept["best_val_loss"], 0.123)
            self.assertEqual(_canon(kept),
                             _canon(_expected("dlinear", a, best_val_loss=0.123)))
            self.assertNotEqual(kept["regression"], stale["regression"])
            # a model without a previous row has no training loss to keep
            self.assertTrue(math.isnan(_read_row(cell, "tsmixer")["best_val_loss"]))
            reg = pd.read_csv(cell / "score" / "comparison_regression.csv")
            self.assertEqual(reg.loc[reg["model"] == "dlinear", "best_val_loss"].tolist(),
                             [0.123, 0.123])

    def test_ignores_tmp_files(self):
        a = _arrays(1)
        with tempfile.TemporaryDirectory() as d:
            cell = Path(d) / "multi_seq288_pred6_f0_direct"
            _write_pred(cell, "dlinear", a)
            pred_dir = cell / "pred"
            # what save_predictions leaves if it dies mid-write
            (pred_dir / "tsmixer.npz.tmp").write_bytes(b"half a zip")
            (pred_dir / "dlinear.npz.tmp").write_bytes(b"half a zip")
            self.assertEqual(rescore_cell(cell), 1)
            self.assertEqual(sorted(p.name for p in (cell / "score" / "rows").iterdir()),
                             ["dlinear.json"])
            self.assertEqual((pred_dir / "tsmixer.npz.tmp").read_bytes(), b"half a zip")
            self.assertEqual(_canon(_read_row(cell, "dlinear")),
                             _canon(_expected("dlinear", a)))

    def test_cell_without_predictions_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as d:
            cell = Path(d) / "multi_seq288_pred6_f0_direct"
            (cell / "pred").mkdir(parents=True)
            self.assertEqual(rescore_cell(cell), 0)
            self.assertFalse((cell / "score").exists())

    def test_metric_filter(self):
        a = _arrays(1)
        with tempfile.TemporaryDirectory() as d:
            cell = Path(d) / "multi_seq288_pred6_f0_direct"
            _write_pred(cell, "dlinear", a)
            rescore_cell(cell, ["tss", "rmse"])
            row = _read_row(cell, "dlinear")
            self.assertEqual(set(row["regression"][0]) - {"channel", "n"}, {"rmse"})
            self.assertIn("tss", row["event"][0])
            self.assertNotIn("hss", row["event"][0])
            self.assertIn("tp", row["event"][0])       # the counts are always written

    def test_a_row_without_a_prediction_file_is_refused(self):
        # re-scoring around it would leave that run on an older rule in the
        # tables, with nothing to show it (spec section 2)
        with tempfile.TemporaryDirectory() as d:
            cell = Path(d) / "multi_seq288_pred6_f0_direct"
            _write_pred(cell, "dlinear", _arrays(1))
            stale = _expected("tsmixer", _arrays(9), best_val_loss=0.3)
            _row_file(cell, "tsmixer").parent.mkdir(parents=True)
            _row_file(cell, "tsmixer").write_text(json.dumps(stale))
            with self.assertRaisesRegex(ValueError, "tsmixer"):
                rescore_cell(cell)
            # nothing was rewritten
            self.assertEqual(sorted(p.name for p in _row_file(cell, "x").parent.iterdir()),
                             ["tsmixer.json"])
            self.assertFalse((cell / "score" / "comparison_regression.csv").exists())

    def test_meta_without_transform_is_an_error(self):
        pred, true, fs = _arrays(1)
        with tempfile.TemporaryDirectory() as d:
            cell = Path(d) / "multi_seq288_pred6_f0_direct"
            save_predictions(prediction_path(str(cell), "dlinear"), pred, true, fs,
                             {k: v for k, v in _meta("dlinear").items() if k != "transform"})
            with self.assertRaisesRegex(ValueError, "transform"):
                rescore_cell(cell)

    def test_no_thresholds_gives_regression_rows_only(self):
        pred, true, fs = _arrays(1)
        with tempfile.TemporaryDirectory() as d:
            cell = Path(d) / "multi_seq288_pred6_f0_direct"
            save_predictions(prediction_path(str(cell), "dlinear"), pred, true, fs,
                             _meta("dlinear", event_threshold=None))
            self.assertEqual(rescore_cell(cell), 1)
            row = _read_row(cell, "dlinear")
            self.assertEqual(len(row["regression"]), 2)
            self.assertEqual(row["event"], [])


def _master(root, folds, name="bench"):
    """A re-scored tree with one cell per fold; returns (bench_root, master_dir)."""
    bench = Path(root) / name
    for fold in folds:
        cell = bench / f"multi_seq288_pred6_f{fold}_direct"
        _write_pred(cell, "dlinear", _arrays(fold))
        _write_pred(cell, "tsmixer", _arrays(10 + fold))
        rescore_cell(cell)
    master = Path(root) / f"{name}_master"
    rebuild_master_from_dirs(bench, master)
    return bench, master


class TestMergeMasters(unittest.TestCase):
    def test_merge_masters_concatenates(self):
        with tempfile.TemporaryDirectory() as d:
            _, m0 = _master(Path(d) / "a", [0])
            _, m1 = _master(Path(d) / "b", [1, 2])
            out = Path(d) / "out"
            counts = merge_masters([m0, m1], out)

            reg = pd.read_csv(out / "results_regression.csv")
            ev = pd.read_csv(out / "results_event.csv")
            # 3 cells x 2 models x 2 channels (x 3 granularities)
            self.assertEqual(counts, (12, 36))
            self.assertEqual((len(reg), len(ev)), counts)
            self.assertEqual(list(reg.columns), regression_columns())
            self.assertEqual(list(ev.columns), event_columns())
            self.assertEqual(sorted(set(reg["fold"])), [0, 1, 2])
            # the same run in two folders is refused
            with self.assertRaisesRegex(ValueError, "more than once"):
                merge_masters([m0, m0], Path(d) / "dup")

    def test_masters_without_events_merge_to_empty_event_table(self):
        pred, true, fs = _arrays(1)
        with tempfile.TemporaryDirectory() as d:
            masters = []
            for fold in (0, 1):
                bench = Path(d) / f"b{fold}"
                cell = bench / f"multi_seq288_pred6_f{fold}_direct"
                save_predictions(prediction_path(str(cell), "dlinear"), pred, true, fs,
                                 _meta("dlinear", event_threshold=None))
                rescore_cell(cell)
                masters.append(Path(d) / f"m{fold}")
                rebuild_master_from_dirs(bench, masters[-1])
            self.assertEqual(merge_masters(masters, Path(d) / "out"), (4, 0))
            ev = pd.read_csv(Path(d) / "out" / "results_event.csv")
            self.assertEqual(list(ev.columns), event_columns())

    def test_merge_is_the_master_of_the_union(self):
        # dir order does not matter, and joining two masters gives the table a
        # single rebuild over both trees would
        with tempfile.TemporaryDirectory() as d:
            a, m0 = _master(Path(d) / "a", [2])
            b, m1 = _master(Path(d) / "b", [0, 1])
            union = Path(d) / "union"
            union.mkdir()
            for cell in [*a.iterdir(), *b.iterdir()]:
                if cell.is_dir():
                    (union / cell.name).symlink_to(cell, target_is_directory=True)
            rebuild_master_from_dirs(union, Path(d) / "union_master")

            merged = {}
            for tag, dirs in (("ab", [m0, m1]), ("ba", [m1, m0])):
                merge_masters(dirs, Path(d) / tag)
                merged[tag] = {f: (Path(d) / tag / f).read_text()
                               for f in ("results_regression.csv", "results_event.csv")}
            self.assertEqual(merged["ab"], merged["ba"])
            for f, text in merged["ab"].items():
                self.assertEqual(text, (Path(d) / "union_master" / f).read_text(), f)

    def test_duplicate_run_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            _, m0 = _master(Path(d) / "a", [0])
            _, m1 = _master(Path(d) / "b", [0, 1])      # fold 0 again
            out = Path(d) / "out"
            with self.assertRaisesRegex(ValueError, "more than once"):
                merge_masters([m0, m1], out)
            self.assertFalse(out.exists())               # nothing half-merged
            with self.assertRaisesRegex(ValueError, "more than once"):
                merge_masters([m0, m0], out)

    def test_other_layout_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            _, m0 = _master(Path(d) / "a", [0])
            _, m1 = _master(Path(d) / "b", [1])
            reg = pd.read_csv(m1 / "results_regression.csv").drop(columns=["logr"])
            reg.to_csv(m1 / "results_regression.csv", index=False)
            with self.assertRaisesRegex(ValueError, "logr"):
                merge_masters([m0, m1], Path(d) / "out")

    def test_counts_are_written_as_integers(self):
        # a master written before the count dtype was fixed holds counts as
        # floats (`396040.0`, forced by a voided run's NaN counts); the merge
        # writes them as integers and keeps a NaN count empty
        with tempfile.TemporaryDirectory() as d:
            _, m0 = _master(Path(d) / "a", [0])
            _, m1 = _master(Path(d) / "b", [1])
            ev = pd.read_csv(m1 / "results_event.csv")
            ev["tp"] = ev["tp"].astype(float)
            ev.loc[0, ["tp", "fp", "fn", "tn", "n_events"]] = float("nan")
            ev.to_csv(m1 / "results_event.csv", index=False)
            self.assertIn(".0,", (m1 / "results_event.csv").read_text())
            out = Path(d) / "out"
            merge_masters([m0, m1], out)
            import csv
            counts = ["n", "n_events", "tp", "fp", "fn", "tn", "n_nonfinite"]
            with open(out / "results_event.csv") as f:
                fields = [r[c] for r in csv.DictReader(f) for c in counts]
            self.assertIn("", fields)
            self.assertTrue(all(x == "" or x.isdigit() for x in fields), fields[:20])

    def test_no_directories_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                merge_masters([], Path(d) / "out")


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "rescore_script", _REPO / "scripts" / "rescore.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestRescoreScript(unittest.TestCase):
    def _tree(self, d, folds=(0, 1)):
        bench = Path(d) / "bench"
        for fold in folds:
            cell = bench / f"multi_seq288_pred6_f{fold}_direct"
            _write_pred(cell, "dlinear", _arrays(fold))
        # not cells: a folder with predictions but another name, and a stray file
        _write_pred(bench / "smoke", "dlinear", _arrays(5))
        (bench / "notes.txt").write_text("x")
        return bench

    def test_rescores_every_cell_and_rebuilds_the_master(self):
        script = _load_script()
        with tempfile.TemporaryDirectory() as d:
            bench = self._tree(d)
            master = Path(d) / "master"
            with contextlib.redirect_stdout(io.StringIO()) as out:
                code = script.main(["--runs", str(bench), "--master-dir", str(master),
                                    "--workers", "1"])
            self.assertEqual(code, 0)
            self.assertIn("2 cells", out.getvalue())
            reg = pd.read_csv(master / "results_regression.csv")
            self.assertEqual(sorted(reg["fold"]), [0, 0, 1, 1])
            self.assertFalse((bench / "smoke" / "score").exists())
            self.assertTrue(_row_file(bench / "multi_seq288_pred6_f1_direct",
                                      "dlinear").exists())

    def test_master_defaults_to_the_runs_folder(self):
        script = _load_script()
        with tempfile.TemporaryDirectory() as d:
            bench = self._tree(d, folds=(0,))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(script.main(["--runs", str(bench), "--workers", "1"]), 0)
            self.assertTrue((bench / "results_regression.csv").exists())
            self.assertTrue((bench / "results_event.csv").exists())

    def test_a_failed_cell_is_reported_and_the_master_left_alone(self):
        script = _load_script()
        with tempfile.TemporaryDirectory() as d:
            bench = self._tree(d)
            (bench / "multi_seq288_pred6_f1_direct" / "pred" / "dlinear.npz").write_bytes(
                b"not a zip")
            master = Path(d) / "master"
            err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                code = script.main(["--runs", str(bench), "--master-dir", str(master),
                                    "--workers", "1"])
            self.assertEqual(code, 1)
            self.assertIn("multi_seq288_pred6_f1_direct", err.getvalue())
            self.assertFalse(master.exists())
            # the healthy cell was still re-scored
            self.assertTrue(_row_file(bench / "multi_seq288_pred6_f0_direct",
                                      "dlinear").exists())

    def test_a_cell_with_rows_but_no_pred_folder_fails_the_master(self):
        script = _load_script()
        with tempfile.TemporaryDirectory() as d:
            bench = self._tree(d, folds=(0,))
            cell = bench / "multi_seq288_pred6_f1_direct"     # saving was off here
            _row_file(cell, "dlinear").parent.mkdir(parents=True)
            _row_file(cell, "dlinear").write_text(json.dumps(_expected("dlinear", _arrays(3))))
            master = Path(d) / "master"
            err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                code = script.main(["--runs", str(bench), "--master-dir", str(master),
                                    "--workers", "1"])
            self.assertEqual(code, 1)
            self.assertIn("multi_seq288_pred6_f1_direct", err.getvalue())
            self.assertFalse(master.exists())

    def test_runs_folder_without_prediction_files_is_an_error(self):
        script = _load_script()
        with tempfile.TemporaryDirectory() as d:
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(script.main(["--runs", d]), 1)
            self.assertIn("no cells", err.getvalue())

    def test_merge_masters_mode(self):
        script = _load_script()
        with tempfile.TemporaryDirectory() as d:
            _, m0 = _master(Path(d) / "a", [0])
            _, m1 = _master(Path(d) / "b", [1])
            out = Path(d) / "out"
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = script.main(["--merge-masters", str(m0), str(m1), "--out", str(out)])
            self.assertEqual(code, 0)
            self.assertIn("8 regression rows", stdout.getvalue())
            self.assertEqual(len(pd.read_csv(out / "results_regression.csv")), 8)
            # the same run twice: a message and a nonzero exit, not a traceback
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(script.main(["--merge-masters", str(m0), str(m0),
                                              "--out", str(Path(d) / "dup")]), 1)
            self.assertIn("more than once", err.getvalue())

    def test_merge_masters_needs_out(self):
        script = _load_script()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            script.main(["--merge-masters", "a", "b"])

    def test_command_line_with_a_process_pool(self):
        # as a user runs it: from another folder, several worker processes
        with tempfile.TemporaryDirectory() as d:
            bench = self._tree(d)
            master = Path(d) / "master"
            run = subprocess.run(
                [sys.executable, str(_REPO / "scripts" / "rescore.py"),
                 "--runs", str(bench), "--master-dir", str(master), "--workers", "2"],
                cwd=d, capture_output=True, text=True, timeout=300)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertEqual(len(pd.read_csv(master / "results_regression.csv")), 4)
            for fold in (0, 1):
                self.assertTrue(_row_file(bench / f"multi_seq288_pred6_f{fold}_direct",
                                          "dlinear").exists())


if __name__ == "__main__":
    unittest.main()
