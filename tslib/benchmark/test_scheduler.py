"""Queue-scheduler behaviour, exercised with fake one-second 'training' jobs.

No GPU and no model: build_cmd is replaced by a python -c that writes the same
per-model row file a real run writes, so the scheduler's contract (pull, skip,
retry, rebuild) is tested for real while the work itself is trivial."""
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from tslib.benchmark import scheduler as sched
from tslib.benchmark.driver import prediction_file, row_path
from tslib.benchmark.test_driver import v2_record


def _jobs(folds=(0, 1), models=("a", "b", "c")):
    return [{"track": "uni_a", "seq_len": 288, "pred_len": 36, "fold": f,
             "strategy": "direct", "model": m} for f in folds for m in models]


def _fake_cmd(runs_root, fail_models=(), no_pred_models=()):
    """Return a build_cmd stand-in that writes the v2 row file and the
    prediction file (or exits 1; or, for `no_pred_models`, writes only the row,
    like a run whose prediction save failed)."""
    def build(cell, epochs, models, runs_root=None):
        model = models[0]
        rp = row_path(cell, model, Path(runs_root))
        pp = prediction_file(cell, model, Path(runs_root))
        if model in fail_models:
            return [sys.executable, "-c", "import sys; sys.exit(1)"]
        record = json.dumps(v2_record(model, best_val_loss=0.5))
        code = (f"import os;p={str(rp)!r};"
                f"os.makedirs(os.path.dirname(p),exist_ok=True);"
                f"open(p,'w').write({record!r})")
        if model not in no_pred_models:
            code += (f";q={str(pp)!r};os.makedirs(os.path.dirname(q),exist_ok=True);"
                     f"open(q,'wb').write(b'npz')")
        return [sys.executable, "-c", code]
    return build


class SchedulerHarness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.runs = Path(self.tmp.name) / "runs"
        self.master_dir = self.runs / "bench"
        self.logs = self.runs / "bench" / "logs"
        self._orig = sched.build_cmd
        self.addCleanup(setattr, sched, "build_cmd", self._orig)
        self.addCleanup(self.tmp.cleanup)
        # The scheduler narrates every job on stdout (from its worker threads);
        # keep that out of the test report.
        out = mock.patch("sys.stdout", io.StringIO())
        out.start()
        self.addCleanup(out.stop)

    def make(self, jobs, fail_models=(), gpus=(0, 1, 2), no_pred_models=()):
        sched.build_cmd = _fake_cmd(self.runs, fail_models, no_pred_models)
        return sched.Scheduler(jobs, gpus, 1, self.runs, self.master_dir,
                               self.logs, min_free_gb=0.0, stagger_s=0.0)

    def master(self):
        """(regression, event) master tables as written by the scheduler."""
        return (pd.read_csv(self.master_dir / "results_regression.csv"),
                pd.read_csv(self.master_dir / "results_event.csv"))


class TestOrdering(unittest.TestCase):
    def test_fold_major_then_cheapest_first(self):
        jobs = [{"track": "multi", "seq_len": 2016, "pred_len": 72, "fold": 0,
                 "strategy": "direct", "model": "z"},
                {"track": "uni_a", "seq_len": 288, "pred_len": 36, "fold": 1,
                 "strategy": "direct", "model": "a"},
                {"track": "uni_a", "seq_len": 288, "pred_len": 36, "fold": 0,
                 "strategy": "direct", "model": "a"}]
        order = [(j["fold"], j["seq_len"]) for j in sorted(jobs, key=sched.job_key)]
        self.assertEqual(order, [(0, 288), (0, 2016), (1, 288)])

    def test_every_fold_0_job_precedes_every_fold_1_job(self):
        ordered = sorted(_jobs(), key=sched.job_key)
        folds = [j["fold"] for j in ordered]
        self.assertEqual(folds, sorted(folds))


class TestQueueRun(SchedulerHarness):
    def test_all_jobs_run_across_gpus_and_master_is_built(self):
        jobs = _jobs()
        s = self.make(jobs)
        self.assertEqual(s.run(), 0)
        self.assertEqual(s.done, len(jobs))
        self.assertEqual(s.failed, [])
        regression, event = self.master()
        self.assertEqual(len(regression), len(jobs))      # one channel each
        self.assertEqual(len(event), len(jobs) * 3)       # point, window, day
        for table in (regression, event):
            self.assertEqual(sorted(table["fold"].unique().tolist()), [0, 1])
            self.assertEqual(sorted(table["model"].unique().tolist()), ["a", "b", "c"])

    def test_finished_jobs_are_skipped_on_rerun(self):
        jobs = _jobs()
        self.make(jobs).run()
        s2 = self.make(jobs)
        self.assertEqual(s2.run(), 0)
        # a skipped job never launches a process, so no log file is created
        logged = list(self.logs.glob("*.log"))
        self.assertEqual(len(logged), len(jobs))  # only the first run's logs
        regression, event = self.master()
        self.assertEqual((len(regression), len(event)), (len(jobs), len(jobs) * 3))

    def test_failure_is_retried_once_then_reported(self):
        jobs = _jobs(folds=(0,), models=("a", "boom"))
        s = self.make(jobs, fail_models=("boom",))
        self.assertEqual(s.run(), 1)
        self.assertEqual([j["model"] for j in s.failed], ["boom"])
        # the healthy job still landed, and the failure did not poison the table
        regression, event = self.master()
        self.assertEqual(regression["model"].tolist(), ["a"])
        self.assertEqual(set(event["model"]), {"a"})
        statuses = [json.loads(l)["status"]
                    for l in self.progress_lines()]
        self.assertEqual(statuses.count("failed"), 2)  # original + one retry
        self.assertEqual(statuses.count("ok"), 1)

    def test_comparison_tables_merge_every_model_of_a_cell(self):
        jobs = _jobs(folds=(0,))
        self.make(jobs).run()
        score = self.runs / "bench" / "uni_a_seq288_pred36_f0_direct" / "score"
        regression = pd.read_csv(score / "comparison_regression.csv")
        event = pd.read_csv(score / "comparison_event.csv")
        self.assertEqual(sorted(regression["model"]), ["a", "b", "c"])
        self.assertEqual(len(event), 3 * 3)
        self.assertEqual(sorted(set(event["model"])), ["a", "b", "c"])
        self.assertFalse((score / "comparison.csv").exists())   # the v1 table is gone

    def test_a_row_without_predictions_is_a_failure_not_a_skip(self):
        jobs = _jobs(folds=(0,), models=("a", "nopred"))
        out = io.StringIO()
        with mock.patch("sys.stdout", out):
            s = self.make(jobs, no_pred_models=("nopred",))
            self.assertEqual(s.run(), 1)
        self.assertEqual([j["model"] for j in s.failed], ["nopred"])
        self.assertIn("row without predictions", out.getvalue())
        # a rerun does not take the row as done: the job is launched again
        s2 = self.make(jobs, no_pred_models=("nopred",))
        with mock.patch("sys.stdout", io.StringIO()):
            s2.run()
        self.assertEqual([j["model"] for j in s2.failed], ["nopred"])

    def progress_lines(self):
        return (self.logs / "queue_progress.jsonl").read_text().splitlines()


class TestWorkerResilience(SchedulerHarness):
    def test_a_raising_job_does_not_kill_its_gpu(self):
        """One bad job must cost one job, not a GPU for the rest of the sweep."""
        jobs = _jobs(folds=(0,), models=("a", "b", "c"))
        s = self.make(jobs, gpus=(0,))
        good = sched.build_cmd

        def exploding(cell, epochs, models, runs_root=None):
            if models[0] == "b":
                raise RuntimeError("transient NFS hiccup")
            return good(cell, epochs, models, runs_root)

        sched.build_cmd = exploding
        self.assertEqual(s.run(), 1)          # reports the failure
        self.assertEqual([j["model"] for j in s.failed], ["b"])
        # a and c still ran on the same worker, after b blew up
        self.assertEqual(sorted(self.master()[0]["model"]), ["a", "c"])


class TestTableRefresh(SchedulerHarness):
    def test_a_v1_row_in_the_cell_does_not_fail_finished_jobs(self):
        jobs = _jobs(folds=(0,), models=("a", "b"))
        cell = sched.cell_of(jobs[0])
        v1 = row_path(cell, "old", self.runs)
        v1.parent.mkdir(parents=True)
        v1.write_text(json.dumps({"model": "old", "strategy": "direct",
                                  "best_val_loss": 0.1, "rmse": 0.2}))
        s = self.make(jobs)
        self.assertEqual(s.run(), 0)
        self.assertEqual(s.failed, [])
        self.assertEqual(sorted(self.master()[0]["model"]), ["a", "b"])

    def test_a_failed_refresh_keeps_the_job_and_counts_it_once(self):
        jobs = _jobs(folds=(0,), models=("a", "b"))
        s = self.make(jobs, gpus=(0,))
        with mock.patch.object(s, "_refresh_tables", side_effect=OSError("stale NFS handle")):
            self.assertEqual(s.run(), 0)
        self.assertEqual((s.done, s.total, s.failed), (2, 2, []))
        statuses = [json.loads(l)["status"] for l in
                    (self.logs / "queue_progress.jsonl").read_text().splitlines()]
        self.assertEqual(statuses, ["ok", "ok"])


class TestMemoryGuard(unittest.TestCase):
    def test_reports_a_plausible_headroom(self):
        free = sched._mem_available_gb()
        self.assertGreater(free, 0)
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
        self.assertLessEqual(free, total)

    def test_guard_releases_once_memory_is_available(self):
        """The wait loop must be a loop, not a one-shot check: it has to exit
        the moment headroom appears (min_free 0 is always satisfied)."""
        s = sched.Scheduler([], [0], 1, "/tmp", "/tmp/m",
                            tempfile.mkdtemp(), min_free_gb=0.0, stagger_s=0.0)
        s._wait_for_memory(0)   # returns immediately or the test times out


if __name__ == "__main__":
    unittest.main()
