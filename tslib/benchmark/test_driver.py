import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from tslib.benchmark import driver
from tslib.exp.evaluate import evaluate
from tslib.exp.lightning_model import TrainResult
from tslib.exp.metrics import MetricContext
from tslib.exp.records import event_columns, regression_columns, result_record

_HOUR_NS = 3600 * 10**9


class TestEnumerateCells(unittest.TestCase):
    def test_direct_fold0_full_sweep_count(self):
        cells = driver.enumerate_cells(
            tracks=list(driver.TRACKS), seq_lens=driver.SEQ_LENS,
            pred_lens=driver.PRED_LENS, folds=[0], strategies=["direct"])
        # 3 tracks x 3 seq x 4 pred x 1 fold x 1 strategy = 36 (all direct-applicable)
        self.assertEqual(len(cells), 36)
        self.assertTrue(all(c["strategy"] == "direct" for c in cells))

    def test_recursive_covers_all_tracks(self):
        cells = driver.enumerate_cells(
            tracks=list(driver.TRACKS), seq_lens=[288], pred_lens=[144],
            folds=[0], strategies=["recursive"])
        # every track has a recursive lstm now — multi forecasts its 2 input
        # channels (targets == channels), so recursive applies to it too.
        tracks = {c["track"] for c in cells}
        self.assertEqual(tracks, {"uni_a", "uni_b", "multi"})
        self.assertEqual(len(cells), 3)


class TestBuildCmd(unittest.TestCase):
    def test_uni_a_direct_flags(self):
        cell = {"track": "uni_a", "seq_len": 288, "pred_len": 144,
                "fold": 0, "strategy": "direct"}
        cmd = driver.build_cmd(cell, epochs=30)
        self.assertIn("--forecast_strategy", cmd)
        self.assertEqual(cmd[cmd.index("--forecast_strategy") + 1], "direct")
        # direct univariate model set = the fast tier (no lstm; slow models excluded)
        mi = cmd.index("--models")
        n = len(driver.DIRECT_UNI)
        self.assertEqual(cmd[mi + 1:mi + 1 + n], driver.DIRECT_UNI)
        self.assertNotIn("timemixer", cmd)
        self.assertNotIn("segrnn", cmd[mi + 1:mi + 1 + n])  # segrnn dropped (redundant)
        self.assertEqual(cmd[cmd.index("--event_threshold") + 1], "10")
        self.assertEqual(cmd[cmd.index("--seq_len") + 1], "288")
        self.assertEqual(cmd[cmd.index("--pred_len") + 1], "144")
        self.assertEqual(cmd[cmd.index("--fold_numb") + 1], "0")
        self.assertEqual(cmd[cmd.index("--epochs") + 1], "30")
        self.assertEqual(cmd[cmd.index("--run_name") + 1],
                         "bench/uni_a_seq288_pred144_f0_direct")
        # gate flags removed; early-stop present
        self.assertNotIn("--max_train_hours", cmd)
        self.assertEqual(cmd[cmd.index("--early_stop_patience") + 1], "10")
        self.assertEqual(cmd[cmd.index("--epochs") + 1], "30")
        # univariate has no channels
        self.assertNotIn("--channels", cmd)

    def test_multi_direct_has_two_channels_and_thresholds(self):
        cell = {"track": "multi", "seq_len": 864, "pred_len": 288,
                "fold": 0, "strategy": "direct"}
        cmd = driver.build_cmd(cell, epochs=30)
        ci = cmd.index("--channels")
        # multivariable = exactly 2 channels (p_gt10 + xrs_long), matching targets
        self.assertTrue(cmd[ci + 1].endswith(":p_gt10"))
        self.assertTrue(cmd[ci + 2].endswith(":xrs_long"))
        self.assertEqual(cmd[ci + 3], "--target_cols")  # only 2 channels listed
        ti = cmd.index("--target_cols")
        self.assertEqual(cmd[ti + 1:ti + 3], ["p_gt10", "xrs_long"])
        ei = cmd.index("--event_threshold")
        self.assertEqual(cmd[ei + 1:ei + 3], ["10", "1e-5"])
        mi = cmd.index("--models")
        n = len(driver.DIRECT_MULTI)
        self.assertEqual(cmd[mi + 1:mi + 1 + n], driver.DIRECT_MULTI)
        self.assertNotIn("lstm", cmd[mi + 1:mi + 1 + n])  # lstm is recursive, not direct

    def test_multi_recursive_forecasts_all_channels(self):
        cell = {"track": "multi", "seq_len": 288, "pred_len": 144,
                "fold": 0, "strategy": "recursive"}
        cmd = driver.build_cmd(cell, epochs=30)
        self.assertEqual(cmd[cmd.index("--forecast_strategy") + 1], "recursive")
        mi = cmd.index("--models")
        self.assertEqual(cmd[mi + 1], "lstm")
        # recursive needs targets == channels: 2 channels, 2 targets
        ci = cmd.index("--channels")
        self.assertEqual(cmd[ci + 3], "--target_cols")
        ti = cmd.index("--target_cols")
        self.assertEqual(cmd[ti + 1:ti + 3], ["p_gt10", "xrs_long"])


def v2_record(model, strategy="direct", channels=("p_gt10",), best_val_loss=0.25):
    """The row-file content run_experiment writes for one model: a real
    `evaluate` of four daily windows, so one regression row per channel and
    point/window/day event rows per channel. Shared with test_scheduler."""
    pred = np.stack([[2.0, 0.0, 2.0, 0.0], [-4.0, -6.0, -4.0, -6.0]],
                    axis=-1)[:, None, :len(channels)]
    true = np.stack([[2.0, 2.0, 0.0, 0.0], [-4.0, -4.0, -6.0, -6.0]],
                    axis=-1)[:, None, :len(channels)]
    ctx = MetricContext(thresholds=[10.0, 1e-5][:len(channels)], transform="log10",
                        target_cols=list(channels),
                        forecast_start=np.arange(4, dtype=np.int64) * 24 * _HOUR_NS)
    return result_record(TrainResult(model, best_val_loss, evaluate(pred, true, ctx),
                                     "a.pt", strategy=strategy))


def _write_row(cell, runs_root, record):
    """Persist one model's result record exactly the way run_experiment does —
    the completion/rebuild source of truth since v004 runs one model per
    process across several GPUs."""
    p = driver.row_path(cell, record["model"], runs_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record))


def _write_pred_file(cell, runs_root, model):
    """Stand-in for the run's pred/<model>.npz (completion only checks it exists)."""
    p = driver.prediction_file(cell, model, runs_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"npz")


class TestShortHorizonGrid(unittest.TestCase):
    def test_seg_len_is_the_same_at_every_horizon(self):
        """A seg_len that moved with pred_len would confound the horizon axis;
        segrnn_thuml absorbs an indivisible horizon by over-decoding instead."""
        for pred_len in (36, 72, 144, 288):
            cell = {"track": "uni_a", "seq_len": 288, "pred_len": pred_len,
                    "fold": 0, "strategy": "direct"}
            cmd = driver.build_cmd(cell, epochs=1, models=["segrnn_thuml"])
            self.assertEqual(cmd[cmd.index("--seg_len") + 1], "24")

    def test_xpatch_excluded_only_at_the_7_day_lookback(self):
        base = {"track": "uni_a", "pred_len": 36, "fold": 0, "strategy": "direct"}
        self.assertIn("xpatch", driver.models_for_cell({**base, "seq_len": 864}))
        self.assertNotIn("xpatch", driver.models_for_cell({**base, "seq_len": 2016}))
        self.assertEqual(len(driver.models_for_cell({**base, "seq_len": 2016})),
                         len(driver.models_for_cell({**base, "seq_len": 864})) - 1)


class TestFixedHyperparameters(unittest.TestCase):
    def test_fixed_model_hparams_match_v1(self):
        """The sweep's model hyperparameters are the v1 (v003/v004) values; a
        silent change would make v005 rows incomparable with the archived runs."""
        f = driver.FIXED
        for flag, val in [("--d_model", "128"), ("--num_layers", "2"), ("--dropout", "0.1"),
                          ("--batch_size", "64"), ("--seg_len", "24"), ("--transform", "log10"),
                          ("--lr", "1e-3"), ("--early_stop_patience", "10")]:
            self.assertEqual(f[f.index(flag) + 1], val)

    def test_every_track_passes_config_validation(self):
        """A cell whose flags main.py rejects fails every job of the sweep; the
        regression table is ranked by best_val_loss (the --sort_metric default)."""
        from tslib.configs.config import config_postprocess, exp_parser
        for track in driver.TRACKS:
            cell = {"track": track, "seq_len": 288, "pred_len": 36, "fold": 0,
                    "strategy": "direct"}
            argv = driver.cell_argv(cell, 1, ["dlinear"])
            cfg = config_postprocess(exp_parser().parse_args(argv))
            self.assertEqual(cfg.sort_metric, "best_val_loss", track)

    def test_every_registered_metric_is_computed_and_the_sort_metric_is_the_default(self):
        """The sweep records everything (spec 4.5): no --metrics filter, and the
        regression comparison is ranked by the --sort_metric default."""
        from tslib.configs.config import config_postprocess, exp_parser
        self.assertNotIn("--metrics", driver.FIXED)
        for track in driver.TRACKS:
            self.assertNotIn("sort_metric", driver.TRACKS[track])
            cell = {"track": track, "seq_len": 288, "pred_len": 36, "fold": 0,
                    "strategy": "direct"}
            argv = driver.cell_argv(cell, 1, ["dlinear"])
            self.assertNotIn("--metrics", argv)
            self.assertNotIn("--sort_metric", argv)
            cfg = config_postprocess(exp_parser().parse_args(argv))
            self.assertIsNone(cfg.metrics, track)


class TestRunsRootIsHonouredEndToEnd(unittest.TestCase):
    """The scheduler bookkeeps at `runs_root` while a subprocess resolves its
    own output paths, so the two MUST agree. When they did not, a job that
    finished cleanly (exit 0, checkpoint written) was reported as FAILED
    because its row landed in the default runs/ tree instead."""

    CELL = {"track": "uni_a", "seq_len": 288, "pred_len": 36, "fold": 0,
            "strategy": "direct"}

    def test_build_cmd_passes_runs_root_to_the_subprocess(self):
        cmd = driver.build_cmd(self.CELL, epochs=1, models=["dlinear"],
                               runs_root="/scratch/runs")
        self.assertEqual(cmd[cmd.index("--runs_root") + 1], "/scratch/runs")

    def test_omitted_runs_root_leaves_the_default_untouched(self):
        self.assertNotIn("--runs_root",
                         driver.build_cmd(self.CELL, epochs=1, models=["dlinear"]))

    def test_subprocess_writes_where_the_driver_looks(self):
        from tslib.configs.config import config_postprocess, exp_parser
        from tslib.exp.exp import save_dir_name
        root = "/scratch/runs"
        argv = driver.cell_argv(self.CELL, 1, ["dlinear"], runs_root=root)
        cfg = config_postprocess(exp_parser().parse_args(argv))
        self.assertEqual(save_dir_name(cfg)["score"],
                         str(driver.score_dir(self.CELL, Path(root))))


def _run_name(cell):
    return driver.run_name_for(cell["track"], cell["seq_len"], cell["pred_len"],
                               cell["fold"], cell["strategy"])


class TestParseCellName(unittest.TestCase):
    def test_parse_cell_name_roundtrip(self):
        for track in driver.TRACKS:
            for strategy in ("direct", "recursive", "statistic"):
                cell = {"track": track, "seq_len": 864, "pred_len": 36,
                        "fold": 3, "strategy": strategy}
                name = _run_name(cell)
                self.assertEqual(driver.parse_cell_name(name), cell, name)
                # the cell directory's own name parses to the same cell
                self.assertEqual(driver.parse_cell_name(Path(name).name), cell, name)

    def test_values_are_ints(self):
        cell = driver.parse_cell_name("multi_seq2016_pred288_f4_recursive")
        self.assertEqual(cell, {"track": "multi", "seq_len": 2016, "pred_len": 288,
                                "fold": 4, "strategy": "recursive"})
        for k in ("seq_len", "pred_len", "fold"):
            self.assertIsInstance(cell[k], int)

    def test_rejects_names_that_are_not_cells(self):
        for bad in ("logs", "smoke", "uni_c_seq288_pred36_f0_direct",
                    "uni_a_seq288_pred36_f0_ensemble",
                    "uni_a_seq288_pred36_f0_direct_old",
                    "uni_a_seq288_pred36_f0_direct.bak",
                    "uni_a_seq288_pred36_f0_direct\n",
                    "uni_a_seq_pred36_f0_direct", "uni_a_seq288_pred36_fx_direct",
                    "other/uni_a_seq288_pred36_f0_direct", ""):
            with self.assertRaises(ValueError, msg=bad):
                driver.parse_cell_name(bad)


class TestRebuildMaster(unittest.TestCase):
    UNI = {"track": "uni_a", "seq_len": 288, "pred_len": 144,
           "fold": 0, "strategy": "direct"}
    MULTI = {"track": "multi", "seq_len": 864, "pred_len": 36,
             "fold": 1, "strategy": "direct"}

    def _two_cells(self, runs_root):
        """uni_a with two models x one channel, multi with two models x two."""
        _write_row(self.UNI, runs_root, v2_record("patchtst"))
        _write_row(self.UNI, runs_root, v2_record("dlinear", best_val_loss=0.5))
        two = ("p_gt10", "xrs_long")
        _write_row(self.MULTI, runs_root, v2_record("patchtst", channels=two))
        _write_row(self.MULTI, runs_root, v2_record("dlinear", channels=two))

    def test_rebuild_master_writes_two_files(self):
        with tempfile.TemporaryDirectory() as d:
            runs_root, master_dir = Path(d) / "runs", Path(d) / "runs" / "bench"
            self._two_cells(runs_root)
            n = driver.rebuild_master([self.UNI, self.MULTI], runs_root, master_dir)
            self.assertEqual(n, 2)
            self.assertEqual(sorted(p.name for p in master_dir.glob("*.csv")),
                             ["results_event.csv", "results_regression.csv"])
            reg = pd.read_csv(master_dir / "results_regression.csv")
            ev = pd.read_csv(master_dir / "results_event.csv")
            self.assertEqual(list(reg.columns), regression_columns())
            self.assertEqual(list(ev.columns), event_columns())
            n_reg = 2 * 1 + 2 * 2       # runs x channels, summed over the two cells
            self.assertEqual(len(reg), n_reg)
            self.assertEqual(len(ev), n_reg * 3)   # point, window, day
            self.assertEqual(set(ev["granularity"]), {"point", "window", "day"})
            # every row carries its own cell's keys
            for df in (reg, ev):
                multi = df[df["track"] == "multi"]
                self.assertEqual(set(multi["channel"]), {"p_gt10", "xrs_long"})
                self.assertTrue((multi[["seq_len", "pred_len", "fold"]]
                                 == [864, 36, 1]).all().all())
                self.assertEqual(set(df[df["track"] == "uni_a"]["channel"]), {"p_gt10"})
            r = reg[(reg["track"] == "uni_a") & (reg["model"] == "dlinear")].iloc[0]
            self.assertEqual((r["strategy"], r["best_val_loss"]), ("direct", 0.5))
            # written atomically: no temp file is left next to the tables
            self.assertEqual([p.name for p in master_dir.glob("*.tmp")], [])

    def test_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            runs_root, master_dir = Path(d) / "runs", Path(d) / "runs" / "bench"
            self._two_cells(runs_root)
            cells = [self.UNI, self.MULTI]
            driver.rebuild_master(cells, runs_root, master_dir)
            first = [(master_dir / f).read_bytes()
                     for f in ("results_regression.csv", "results_event.csv")]
            n = driver.rebuild_master(cells, runs_root, master_dir)
            second = [(master_dir / f).read_bytes()
                      for f in ("results_regression.csv", "results_event.csv")]
            self.assertEqual(n, 2)
            self.assertEqual(first, second)   # no duplicate rows

    def test_skips_a_cell_with_no_rows(self):
        cell = {"track": "uni_b", "seq_len": 288, "pred_len": 144,
                "fold": 0, "strategy": "direct"}
        with tempfile.TemporaryDirectory() as d:
            runs_root, master_dir = Path(d) / "runs", Path(d) / "runs" / "bench"
            self.assertEqual(driver.rebuild_master([cell], runs_root, master_dir), 0)
            self.assertEqual(list(master_dir.glob("*.csv")), [])
            # a cell with rows next to an empty one still counts only itself
            _write_row(self.UNI, runs_root, v2_record("patchtst"))
            self.assertEqual(driver.rebuild_master([cell, self.UNI], runs_root, master_dir), 1)

    def test_run_without_thresholds_has_regression_rows_only(self):
        with tempfile.TemporaryDirectory() as d:
            runs_root, master_dir = Path(d) / "runs", Path(d) / "runs" / "bench"
            rec = v2_record("patchtst")
            rec["event"] = []
            _write_row(self.UNI, runs_root, rec)
            driver.rebuild_master([self.UNI], runs_root, master_dir)
            self.assertEqual(len(pd.read_csv(master_dir / "results_regression.csv")), 1)
            ev = pd.read_csv(master_dir / "results_event.csv")
            self.assertEqual((len(ev), list(ev.columns)), (0, event_columns()))

    def test_counts_are_written_as_integers_next_to_a_voided_run(self):
        with tempfile.TemporaryDirectory() as d:
            runs_root, master_dir = Path(d) / "runs", Path(d) / "runs" / "bench"
            voided = v2_record("xpatch")
            for row in voided["event"]:
                row.update(n_events=float("nan"), tp=float("nan"), fp=float("nan"),
                           fn=float("nan"), tn=float("nan"))
            _write_row(self.UNI, runs_root, voided)
            _write_row(self.UNI, runs_root, v2_record("dlinear"))
            driver.rebuild_master([self.UNI], runs_root, master_dir)
            text = (master_dir / "results_event.csv").read_text()
            # dlinear's p_gt10 point row: n 4, 2 events, tp fp fn tn 1 each, 0 non-finite
            self.assertIn(",4,2,1,1,1,1,0,", text)
            self.assertIn(",4,,,,,,0,", text)          # the voided counts stay empty

    def test_a_v1_row_is_left_out_with_a_warning(self):
        # an upgrader's v1.0.0 row must not stop the master rebuild (and with it
        # every scheduler job of the sweep); the model re-runs because its
        # prediction file is missing, and its v2 row then joins the master
        with tempfile.TemporaryDirectory() as d:
            runs_root, master_dir = Path(d) / "runs", Path(d) / "runs" / "bench"
            _write_row(self.UNI, runs_root, v2_record("patchtst"))
            _write_row(self.UNI, runs_root, {"strategy": "direct", "model": "old",
                                             "best_val_loss": 0.2, "rmse": 0.1})
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(driver.rebuild_master([self.UNI], runs_root, master_dir), 1)
            reg = pd.read_csv(master_dir / "results_regression.csv")
            self.assertEqual(reg["model"].tolist(), ["patchtst"])
            self.assertIn("old.json", out.getvalue())
            self.assertIn("[warn]", out.getvalue())


class TestRebuildMasterFromDirs(unittest.TestCase):
    def test_rebuild_from_dirs_ignores_non_cells(self):
        cell = {"track": "uni_a", "seq_len": 2016, "pred_len": 72,
                "fold": 2, "strategy": "recursive"}
        other = {"track": "uni_a", "seq_len": 288, "pred_len": 72,
                 "fold": 0, "strategy": "direct"}
        with tempfile.TemporaryDirectory() as d:
            runs_root = Path(d) / "runs"
            bench_root = runs_root / "bench"
            _write_row(cell, runs_root, v2_record("lstm", "recursive"))
            _write_row(other, runs_root, v2_record("dlinear"))
            # not cells: row files under a wrong name, a cell-named folder
            # without any row file, and a stray file
            for name in ("smoke", "logs"):
                (bench_root / name / "score" / "rows").mkdir(parents=True)
                (bench_root / name / "score" / "rows" / "x.json").write_text(
                    json.dumps(v2_record("x")))
            (bench_root / "multi_seq288_pred36_f0_direct" / "score" / "rows").mkdir(parents=True)
            (bench_root / "results_regression.csv").write_text("stale\n")

            master_dir = Path(d) / "master"
            n = driver.rebuild_master_from_dirs(bench_root, master_dir)
            self.assertEqual(n, 2)
            reg = pd.read_csv(master_dir / "results_regression.csv")
            self.assertEqual(sorted(reg["model"]), ["dlinear", "lstm"])
            ev = pd.read_csv(master_dir / "results_event.csv")
            self.assertEqual(len(ev), 2 * 3)
            row = reg[reg["model"] == "lstm"].iloc[0]
            self.assertEqual((row["track"], row["seq_len"], row["pred_len"],
                              row["fold"], row["strategy"]),
                             ("uni_a", 2016, 72, 2, "recursive"))
            # numeric order, not name order (seq2016 sorts before seq288 as text)
            self.assertEqual(reg["seq_len"].tolist(), [288, 2016])

    def test_empty_bench_root_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(driver.rebuild_master_from_dirs(Path(d), Path(d) / "m"), 0)
            self.assertFalse((Path(d) / "m").exists())


class TestRunBenchmarkWiring(unittest.TestCase):
    CELL = {"track": "uni_a", "seq_len": 288, "pred_len": 36, "fold": 0,
            "strategy": "direct"}

    def test_a_finished_cell_updates_both_master_tables_and_shows_both_comparisons(self):
        from tslib.exp.exp import _write_comparisons
        with tempfile.TemporaryDirectory() as d:
            runs_root = Path(d) / "runs"
            master_dir = runs_root / "bench"

            def fake_main_py(cmd, cwd=None, env=None):
                # what main.py does: row file, then the two comparison tables
                _write_row(self.CELL, runs_root, v2_record("dlinear"))
                _write_comparisons(
                    {"score": str(driver.score_dir(self.CELL, runs_root))},
                    "best_val_loss")
                return mock.Mock(returncode=0)

            out = io.StringIO()
            with mock.patch.object(driver.subprocess, "run", fake_main_py), \
                    contextlib.redirect_stdout(out):
                driver.run_benchmark([(self.CELL, ["dlinear"])], 1, runs_root,
                                     master_dir)
            self.assertEqual(len(pd.read_csv(master_dir / "results_regression.csv")), 1)
            self.assertEqual(len(pd.read_csv(master_dir / "results_event.csv")), 3)
            text = out.getvalue()
            self.assertIn("best_val_loss", text)   # regression comparison table
            self.assertIn("granularity", text)     # event comparison table
            self.assertIn(str(master_dir), text)

    def test_main_puts_the_master_tables_under_runs_bench(self):
        with mock.patch.object(driver, "run_benchmark") as run, \
                contextlib.redirect_stdout(io.StringIO()):
            driver.main(["--tracks", "uni_a", "--seq-lens", "288",
                         "--pred-lens", "36", "--dry-run"])
        args = run.call_args.args
        self.assertEqual(args[2], driver.REPO / "runs")
        self.assertEqual(args[3], driver.REPO / "runs" / "bench")


class TestBothEntryPointsRunTheSameGrid(unittest.TestCase):
    """REPRODUCE.md documents the scheduler and the sequential driver for the
    full sweep; both must plan the same 1,740 runs (MODEL_EXCLUSIONS applied,
    so no xPatch at the 7-day input)."""

    FOLDS = ["--folds", "0", "1", "2", "3", "4"]

    def _driver_jobs(self):
        with mock.patch.object(driver, "run_benchmark") as run, \
                contextlib.redirect_stdout(io.StringIO()):
            driver.main([*self.FOLDS, "--dry-run"])
        runs = run.call_args.args[0]
        return [(_run_name(c), m) for c, models in runs for m in models]

    def _scheduler_jobs(self):
        import re
        from tslib.benchmark import scheduler
        out = io.StringIO()
        with mock.patch.object(scheduler, "_require_data"), \
                contextlib.redirect_stdout(out):
            self.assertEqual(scheduler.main([*self.FOLDS, "--dry-run"]), 0)
        return [tuple(m.groups()) for m in
                re.finditer(r"(bench/\S+)::(\S+)", out.getvalue())]

    def test_same_1740_jobs_without_xpatch_at_2016(self):
        drv, sch = self._driver_jobs(), self._scheduler_jobs()
        self.assertEqual(len(drv), 1740)
        self.assertEqual(len(set(drv)), 1740)
        self.assertEqual(sorted(drv), sorted(sch))
        self.assertFalse([j for j in drv if j[1] == "xpatch" and "_seq2016_" in j[0]])
        self.assertEqual(sum(1 for j in drv if j[1] == "xpatch"), 120)   # 288 and 864 only

    def test_build_cmd_default_models_apply_the_exclusions(self):
        cell = {"track": "uni_a", "seq_len": 2016, "pred_len": 36, "fold": 0,
                "strategy": "direct"}
        cmd = driver.build_cmd(cell, epochs=1)
        models = cmd[cmd.index("--models") + 1:cmd.index("--n_fold")]
        self.assertEqual(models, driver.models_for_cell(cell))
        self.assertNotIn("xpatch", models)


class TestManifest(unittest.TestCase):
    def test_cells_from_manifest_groups_models(self):
        manifest = {"approved": [
            {"track": "uni_a", "seq_len": 288, "pred_len": 144, "fold": 0,
             "strategy": "direct", "model": "patchtst"},
            {"track": "uni_a", "seq_len": 288, "pred_len": 144, "fold": 0,
             "strategy": "direct", "model": "timemixer"},
            {"track": "uni_b", "seq_len": 288, "pred_len": 144, "fold": 0,
             "strategy": "direct", "model": "patchtst"},
        ]}
        out = driver.cells_from_manifest(manifest)
        self.assertEqual(len(out), 2)  # two distinct cells
        cell0, models0 = out[0]
        self.assertEqual(cell0["track"], "uni_a")
        self.assertEqual(models0, ["patchtst", "timemixer"])

    def test_build_cmd_honours_explicit_models(self):
        cell = {"track": "uni_a", "seq_len": 288, "pred_len": 144,
                "fold": 0, "strategy": "direct"}
        cmd = driver.build_cmd(cell, epochs=30, models=["timemixer"])
        mi = cmd.index("--models")
        self.assertEqual(cmd[mi + 1], "timemixer")
        self.assertNotIn("patchtst", cmd)


class TestCellComplete(unittest.TestCase):
    def test_complete_only_when_all_models_present(self):
        cell = {"track": "uni_a", "seq_len": 288, "pred_len": 144,
                "fold": 0, "strategy": "direct"}
        with tempfile.TemporaryDirectory() as d:
            runs_root = Path(d) / "runs"
            self.assertFalse(driver.cell_is_complete(cell, ["patchtst"], runs_root))
            for model in ("patchtst", "timemixer"):
                _write_row(cell, runs_root, v2_record(model))
                _write_pred_file(cell, runs_root, model)
            self.assertTrue(driver.cell_is_complete(cell, ["patchtst"], runs_root))
            self.assertTrue(driver.cell_is_complete(cell, ["patchtst", "timemixer"], runs_root))
            self.assertFalse(driver.cell_is_complete(cell, ["itransformer"], runs_root))

    def test_a_row_without_its_prediction_file_is_not_complete(self):
        # a run whose npz save failed must not stay "done" forever: it would
        # never be re-scored and would keep old-rule numbers in the tables
        cell = {"track": "uni_a", "seq_len": 288, "pred_len": 144,
                "fold": 0, "strategy": "direct"}
        with tempfile.TemporaryDirectory() as d:
            runs_root = Path(d) / "runs"
            _write_row(cell, runs_root, v2_record("patchtst"))
            self.assertFalse(driver.model_is_complete(cell, "patchtst", runs_root))
            self.assertTrue(driver.model_is_complete(cell, "patchtst", runs_root,
                                                     require_predictions=False))
            self.assertFalse(driver.cell_is_complete(cell, ["patchtst"], runs_root))
            _write_pred_file(cell, runs_root, "patchtst")
            self.assertTrue(driver.model_is_complete(cell, "patchtst", runs_root))
            self.assertEqual(driver.prediction_file(cell, "patchtst", runs_root),
                             runs_root / "bench" / "uni_a_seq288_pred144_f0_direct"
                             / "pred" / "patchtst.npz")


if __name__ == "__main__":
    unittest.main()
