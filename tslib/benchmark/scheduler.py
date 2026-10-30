"""Multi-GPU queue scheduler for the benchmark sweep (introduced for the v004
short-horizon sweep).

v003 ran the sweep as one sequential process per CELL — a cell being all ~10
models at one (track, seq_len, pred_len, fold). Two properties of that design
do not survive being spread over several GPUs:

  * **Cell granularity strands a GPU.** A cell is only as fast as the sum of its
    models, and the model runtimes inside one cell differ by more than an order
    of magnitude. Statically splitting cells across GPUs (or even pulling whole
    cells from a queue) leaves one GPU grinding through a straggler while the
    others idle. So the unit of work here is ONE MODEL, pulled from a shared
    queue the moment a GPU frees up: the tail is a single model, not a cell.
    Measured cost of the finer granularity: each job re-reads and re-grids its
    parquet, 6.6 s (uni_a/seq288) to 58.6 s (multi/seq2016) against a job that
    runs for tens of minutes.

  * **One run directory, many writers.** Models of the same cell now execute
    concurrently in different processes that share ``runs/bench/<cell>/``.
    Everything they touch is already keyed by model name (``ckpt/<model>.ckpt``,
    ``score/rows/<model>.json``, and — after the v004 fix — the Lightning CSV
    logger directory), so the only shared artefacts are the derived tables, and
    those are rebuilt here from the per-model row files rather than written
    from any single process's view.

Resume is free and total: a job is skipped when its result row and its
prediction file already exist, so re-running this module after any crash, kill, or NFS flap continues exactly
where it stopped. That also makes it safe for a watchdog to relaunch blindly.

    python3.12 -m tslib.benchmark.scheduler --gpus 0 1 2

Without ``--manifest`` the jobs are the driver's grid (``models_for_cell``, so
MODEL_EXCLUSIONS applies); ``--manifest FILE`` runs the one-model jobs of the
file's ``approved`` list instead.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from tslib.benchmark.driver import (REPO, build_cmd, enumerate_cells,
                                    model_is_complete, models_for_cell,
                                    rebuild_master, row_path, run_name_for,
                                    score_dir)

TRACK_RANK = {"uni_a": 0, "uni_b": 1, "multi": 2}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def job_key(job) -> tuple:
    """Queue order: fold-major, then cheapest-first inside a fold.

    Fold-major is the important half. The sweep is long enough that it will be
    read before it is finished, and a partial sweep is only publishable if the
    folds it did finish are COMPLETE — every track/seq_len/pred_len/model at
    fold 0, then all of fold 1, and so on. Ordering by cell instead would leave
    every fold half-done at any interruption. Within a fold, short lookbacks and
    short horizons run first so the cheap results land early."""
    return (job["fold"], job["seq_len"], job["pred_len"],
            TRACK_RANK.get(job["track"], 9), job["model"])


def cell_of(job) -> dict:
    return {k: job[k] for k in ("track", "seq_len", "pred_len", "fold", "strategy")}


def _mem_available_gb() -> float:
    """MemAvailable from /proc/meminfo — the kernel's own estimate of what a new
    process can claim without swapping, which is the number that matters here
    (plain 'free' counts reclaimable page cache as used)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return float("inf")  # unreadable: never block on a guard we cannot evaluate


class Scheduler:
    def __init__(self, jobs, gpus, epochs, runs_root, master_dir, log_dir,
                 min_free_gb, stagger_s, verbose_tail=40):
        self.q = queue.Queue()
        for job in sorted(jobs, key=job_key):
            self.q.put(job)
        self.total = self.q.qsize()
        self.gpus = list(gpus)
        self.epochs = epochs
        self.runs_root = Path(runs_root)
        self.master_dir = Path(master_dir)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.min_free_gb = min_free_gb
        self.stagger_s = stagger_s
        self.verbose_tail = verbose_tail
        self.cells = self._distinct_cells(jobs)
        self.launch_lock = threading.Lock()   # serialises the memory-heavy startup
        self.table_lock = threading.Lock()    # serialises derived-table rebuilds
        self.print_lock = threading.Lock()
        self.stop = threading.Event()
        self.procs: dict[int, subprocess.Popen] = {}
        self.done = 0
        self.failed: list[dict] = []
        self.progress_path = self.log_dir / "queue_progress.jsonl"

    @staticmethod
    def _distinct_cells(jobs) -> list[dict]:
        seen, cells = set(), []
        for job in jobs:
            cell = cell_of(job)
            key = tuple(cell.values())
            if key not in seen:
                seen.add(key)
                cells.append(cell)
        return cells

    def say(self, msg: str) -> None:
        with self.print_lock:
            print(f"[{_now()}] {msg}", flush=True)

    def record(self, **fields) -> None:
        line = json.dumps({"ts": _now(), **fields})
        with self.table_lock:
            with open(self.progress_path, "a") as f:
                f.write(line + "\n")

    def _wait_for_memory(self, gpu: int) -> None:
        """Hold a launch until the host has headroom. Only the STARTUP of a job
        is memory-hungry (parquet read + resample to the common grid); once
        training begins the resident set is a few GB. The launch lock already
        serialises startups, so this only has to cover the case where the rest
        of the machine — this box is shared — is under pressure."""
        warned = False
        while not self.stop.is_set():
            free = _mem_available_gb()
            if free >= self.min_free_gb:
                return
            if not warned:
                self.say(f"gpu{gpu}: waiting for RAM "
                         f"({free:.0f} GB available < {self.min_free_gb} GB)")
                warned = True
            self.stop.wait(30)

    def _launch(self, job, gpu: int, log_path: Path):
        cmd = build_cmd(cell_of(job), self.epochs, [job["model"]],
                        runs_root=self.runs_root)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        with self.launch_lock:
            self._wait_for_memory(gpu)
            if self.stop.is_set():
                return None
            log = open(log_path, "a")
            log.write(f"\n=== {_now()} gpu={gpu} {' '.join(cmd)} ===\n")
            log.flush()
            # start_new_session: the job gets its own process group, so a
            # scheduler shutdown can kill the whole training tree (dataloader
            # workers included) instead of orphaning workers on the GPU.
            proc = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=log,
                                    stderr=subprocess.STDOUT,
                                    start_new_session=True)
            self.procs[gpu] = proc
            # Stay in the lock through the startup window so two jobs never
            # load their parquets at the same instant.
            self.stop.wait(self.stagger_s)
        return proc, log

    def _refresh_tables(self, job) -> int:
        """Rebuild the derived tables from the per-model row files: the job's
        cell gets its two comparison tables (regression, event), then the two
        master tables are rebuilt. All are pure functions of what is on disk, so
        a concurrent rebuild can at worst be momentarily stale and the next
        job's rebuild repairs it."""
        from tslib.exp.exp import _write_comparisons
        cell = cell_of(job)
        with self.table_lock:
            # best_val_loss: the --sort_metric default, which the jobs run with
            _write_comparisons({"score": str(score_dir(cell, self.runs_root))},
                               "best_val_loss")
            return rebuild_master(self.cells, self.runs_root, self.master_dir)

    def worker(self, gpu: int) -> None:
        """Keep this GPU busy until the queue drains.

        A bare thread that raises dies silently and takes its GPU out of the
        pool for the rest of the sweep — days of idle hardware and no error
        anywhere. So every job is wrapped: an unexpected failure is logged
        against that job and the worker goes back for the next one."""
        while not self.stop.is_set():
            try:
                job = self.q.get_nowait()
            except queue.Empty:
                return
            try:
                self._run_one(job, gpu)
            except Exception as exc:  # noqa: BLE001 - a GPU must never be lost
                with self.table_lock:
                    self.done += 1
                self.failed.append(job)
                self.say(f"gpu{gpu} ERROR on {job}: {exc!r} — continuing")
                self.record(gpu=gpu, job=job, status="error", error=repr(exc))

    def _run_one(self, job, gpu: int) -> None:
        cell = cell_of(job)
        name = run_name_for(cell["track"], cell["seq_len"], cell["pred_len"],
                            cell["fold"], cell["strategy"])
        tag = f"{name}::{job['model']}"
        if model_is_complete(cell, job["model"], self.runs_root):
            with self.table_lock:
                self.done += 1
                n = self.done
            self.say(f"[{n}/{self.total}] gpu{gpu} SKIP (done): {tag}")
            return
        log_path = self.log_dir / (
            f"{name.replace('/', '_')}__{job['model']}.log")
        self.say(f"[{self.done}/{self.total}] gpu{gpu} RUN: {tag} "
                 f"-> {log_path.name}")
        t0 = time.time()
        launched = self._launch(job, gpu, log_path)
        if launched is None:
            self.q.put(job)          # shutting down: leave it for a resume
            return
        proc, log = launched
        rc = proc.wait()
        log.close()
        self.procs.pop(gpu, None)
        mins = (time.time() - t0) / 60
        with self.table_lock:
            self.done += 1
            n = self.done
        # exit 0 is not enough: the row file is what the sweep is made of, and
        # a run that "succeeds" without leaving one has written somewhere the
        # scheduler is not looking; a row without its prediction file (a
        # failed save) could never be re-scored.
        ok = rc == 0 and model_is_complete(cell, job["model"], self.runs_root)
        why = f"exit {rc}"
        if rc == 0 and not ok:
            why += (", row without predictions"
                    if row_path(cell, job["model"], self.runs_root).exists()
                    else ", no result row")
        if ok:
            # The job is done whatever happens to the derived tables: a refresh
            # error is reported here (not by the worker's catch-all, which would
            # count the job twice and mark it failed); the next refresh repairs
            # the tables.
            try:
                cells = f"master has {self._refresh_tables(job)} cells"
            except Exception as exc:  # noqa: BLE001
                cells = f"table refresh failed: {exc!r}"
            self.say(f"[{n}/{self.total}] gpu{gpu} DONE ({mins:.1f} min): "
                     f"{tag} — {cells}")
        else:
            self.failed.append(job)
            self.say(f"[{n}/{self.total}] gpu{gpu} FAILED ({why}, "
                     f"{mins:.1f} min): {tag} — see {log_path}")
            self.say("    " + self._tail(log_path))
        self.record(gpu=gpu, job=job, status="ok" if ok else "failed",
                    returncode=rc, minutes=round(mins, 2))

    def _tail(self, path: Path) -> str:
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            return "(log unreadable)"
        return " | ".join(x.strip() for x in lines[-self.verbose_tail:]
                          if x.strip())[-2000:]

    def shutdown(self, *_):
        if self.stop.is_set():
            return
        self.stop.set()
        self.say("shutdown requested — terminating running jobs")
        for proc in list(self.procs.values()):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

    def run(self) -> int:
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        self.say(f"queue: {self.total} model-jobs over gpus {self.gpus} "
                 f"(epochs={self.epochs}, stagger={self.stagger_s}s, "
                 f"min_free={self.min_free_gb} GB)")
        threads = [threading.Thread(target=self.worker, args=(g,),
                                    name=f"gpu{g}", daemon=True)
                   for g in self.gpus]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # One retry pass: the long tail of failures on this host is transient
        # (the shared network mount flaps), and a re-queued job that fails
        # deterministically only costs its own runtime once more.
        if self.failed and not self.stop.is_set():
            retry, self.failed = self.failed, []
            self.say(f"retrying {len(retry)} failed job(s) once")
            for job in retry:
                self.q.put(job)
            self.total += len(retry)
            threads = [threading.Thread(target=self.worker, args=(g,),
                                        name=f"gpu{g}-retry", daemon=True)
                       for g in self.gpus]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        n = rebuild_master(self.cells, self.runs_root, self.master_dir)
        self.say(f"sweep finished: {self.done}/{self.total} jobs, "
                 f"{n} cells in {self.master_dir}, {len(self.failed)} failed")
        for job in self.failed:
            self.say(f"  FAILED: {job}")
        return 1 if self.failed else 0


def jobs_from_manifest(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)["approved"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Multi-GPU queue scheduler for the benchmark sweep.")
    p.add_argument("--manifest", type=str, default=None,
                   help="JSON with an 'approved' list of one-model jobs")
    p.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--tracks", nargs="+", default=None)
    p.add_argument("--seq-lens", nargs="+", type=int, default=None)
    p.add_argument("--pred-lens", nargs="+", type=int, default=None)
    p.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--strategies", nargs="+", default=["direct"])
    p.add_argument("--min-free-gb", type=float, default=25.0,
                   help="hold a launch while MemAvailable is below this")
    p.add_argument("--stagger-s", type=float, default=90.0,
                   help="seconds to hold the launch lock after starting a job, "
                        "so two jobs never load their data at the same instant")
    p.add_argument("--runs-root", type=str, default=None,
                   help="write runs/logs/tables here instead of <repo>/runs "
                        "(use a scratch path for pilots so a short-epoch probe "
                        "cannot mark real sweep jobs as finished)")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _require_data() -> None:
    """Fail at startup, not eight hours into the queue. The parquet pair lives
    outside the repo and is located by SW_DATA_DIR; an unset or stale value
    otherwise surfaces as every single job dying on its own."""
    from tslib.benchmark.driver import DATA_DIR, PARTICLE, XRAY
    missing = [p for p in (PARTICLE, XRAY) if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            "data file(s) not found: " + ", ".join(missing) +
            f"\nSW_DATA_DIR is {DATA_DIR!r} — point it at the directory "
            "holding kswrc_particle_5m.parquet and kswrc_xray_1m.parquet.")


def main(argv=None):
    args = parse_args(argv)
    runs_root = Path(args.runs_root) if args.runs_root else REPO / "runs"
    master_dir = runs_root / "bench"
    log_dir = runs_root / "bench" / "logs"
    if args.manifest:
        jobs = jobs_from_manifest(args.manifest)
    else:
        from tslib.benchmark.driver import PRED_LENS, SEQ_LENS, TRACKS
        cells = enumerate_cells(args.tracks or list(TRACKS),
                                args.seq_lens or SEQ_LENS,
                                args.pred_lens or PRED_LENS,
                                args.folds,
                                args.strategies)
        jobs = [{**c, "model": m} for c in cells for m in models_for_cell(c)]
    _require_data()
    if args.dry_run:
        for i, job in enumerate(sorted(jobs, key=job_key), 1):
            done = model_is_complete(cell_of(job), job["model"], runs_root)
            print(f"[{i}/{len(jobs)}] {'DONE ' if done else '     '}"
                  f"{run_name_for(job['track'], job['seq_len'], job['pred_len'], job['fold'], job['strategy'])}"
                  f"::{job['model']}")
        return 0
    return Scheduler(jobs, args.gpus, args.epochs, runs_root, master_dir,
                     log_dir, args.min_free_gb, args.stagger_s).run()


if __name__ == "__main__":
    raise SystemExit(main())
