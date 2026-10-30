# v004 — short-horizon sweep (pred_len 36 / 72) on a multi-GPU queue

> **v2.0.0 note.** This page describes the v004 sweep as it was run. Since v2.0.0
> the per-cell `score/comparison.csv` is replaced by `comparison_regression.csv` and
> `comparison_event.csv`, and the sweep's `results_master.csv` by
> `results_regression.csv` and `results_event.csv`; they are still rebuilt from
> `score/rows/<model>.json`, whose records now hold both the regression and the event
> rows, and the completion check is unchanged. The HSS values in "Result" were scored
> under the v1 rules (per-point, undefined = 0.0) and are not v2 values; the v1 files
> are kept in [`results/v1/`](../results/v1/). See [`metrics.md`](metrics.md).

v003 established the benchmark at horizons of 144 and 288 steps (12 h / 24 h on
the common 5-min grid). v004 extends the horizon axis **downward**, to 36 and 72
steps (3 h / 6 h), holding every other condition of
[`benchmark-conditions.md`](benchmark-conditions.md) fixed so the four horizons
form one comparable series.

Grid: 3 tracks x seq_len {288, 864, 2016} x pred_len {36, 72} x folds 0-4,
direct strategy, the 10-model fast tier — **870 model-runs** (xpatch is excluded
at seq_len 2016, where it diverges, exactly as in v003).

> **In this repository** `driver.PRED_LENS` is the full four-horizon series
> `[36, 72, 144, 288]`, so the scheduler's default grid is 36 cells per fold
> (1,740 model-runs over five folds). The v004 sweep itself is the `36 72` subset:
> `python3.12 -m tslib.benchmark.scheduler --pred-lens 36 72`.

## What changed in the code, and why

### 1. SegRNN absorbs an indivisible horizon instead of retuning `seg_len`

SegRNN segments both the lookback and the horizon. Upstream requires `seg_len`
to divide `pred_len` and floor-divides, so at pred_len 36 with the default
`seg_len=24` the decoder produces 48 values and dies reshaping them to 36:

    RuntimeError: shape '[-1, 1, 36]' is invalid for input of size 48

The fix is in the model, not the sweep configuration: `seg_num_y` now
**ceils**, so the decoder emits the next whole number of segments (2 x 24 = 48
steps) and the forward pass trims the tail to `pred_len`. The trimmed steps are
never compared against a target, so they carry no gradient; the parameters that
produce them are shared with the retained steps and train normally.

`seg_len` therefore stays **24 at every horizon in the study** — 36, 72, 144 and
288 alike. That is the point. Deriving `seg_len` from `pred_len` (the first
attempt here used 12 at pred_len 36) makes a model hyperparameter move with the
very axis the sweep measures, so a horizon effect and a capacity effect become
impossible to separate. A rule of the shape `min(48, pred_len)` is also unsafe
in this grid for a subtler reason: it happens to work at 36, 144 and 288 and
crashes at 72, because 48 does not divide 72.

The one divisibility SegRNN still requires is `seq_len % seg_len == 0` (the
encoder reshape). 24 divides 288, 864 and 2016; a future lookback that breaks it
raises a named `ValueError` at construction rather than an opaque reshape error.
`test_horizons.py` builds and forward-passes every model at every grid point and
pins the over-decode arithmetic (`seg_num_y=2`, `dec_len=48`, output 36).

### 2. One model per job, pulled from a shared queue

v003 ran one process per **cell** (all ~10 models of a track/seq_len/pred_len/
fold). Spread over several GPUs that strands hardware: a cell is as slow as the
sum of its models, and model runtimes inside a cell differ by more than an order
of magnitude, so one straggler idles a GPU for hours. `tslib/benchmark/
scheduler.py` makes the unit of work **one model** and hands it to whichever GPU
just went free. The cost of the finer grain is one extra data preparation per
job: measured at 6.6 s (uni_a/seq288) to 58.6 s (multi/seq2016) against jobs that
run for tens of minutes.

Queue order is **fold-major, cheapest-first within a fold**. The sweep will be
read before it finishes, and a partial sweep is only publishable if the folds it
did finish are complete, so fold 0 lands whole before fold 1 starts.

### 3. Concurrency fixes in the shared run directory

Models of one cell now execute at the same time in different processes that
share `runs/bench/<cell>/`. Three artefacts needed to become writer-safe:

| Artefact | v003 | v004 |
|---|---|---|
| `score/comparison.csv` | rebuilt from the writing process's own models — a second process overwrote it with only its row | rebuilt from `score/rows/*.json`, so every write is a merge |
| `ckpt/lightning/` | `CSVLogger` auto-versioning (`version_N`) raced: two processes could pick the same directory | `version=<model>`, race-free and attributable |
| completion check | "is this model in `comparison.csv`?" | "does `score/rows/<model>.json` exist?" — one writer per file |

### 4. `--runs_root`

A pilot at `--epochs 1` used to write into the real sweep tree and mark those
jobs finished. `--runs_root` redirects a run's output, and it is passed **through
to the subprocess** — the driver and the process it launches must resolve the
same path, or a job that succeeds reads as a failure. `test_driver.py` asserts
that equality directly.

## Running it

    bash scripts/run_queue.sh            # GPUs 0 1 2, detached, resumable
    bash scripts/watchdog.sh 60 0 1 2 &  # resource log + auto-resume

    tail -f runs/bench/queue_sweep.log
    column -t -s, runs/bench/resource_monitor.csv | tail

Re-running `run_queue.sh` is the resume: any job whose result row exists is
skipped (since v2.0.0, only when its `pred/<model>.npz` exists too; a row
without it is re-tested from the checkpoint, not retrained), so a crash, a kill, or an NFS flap costs at most the jobs that were
in flight. `kill $(cat runs/bench/queue_sweep.pid)` stops it cleanly — SIGTERM
tears down the whole training tree instead of orphaning dataloader workers.

Guards worth knowing about: a launch waits while `MemAvailable` is under
`--min-free-gb` (25 GB), holds a lock for `--stagger-s` (90 s) so two jobs never
read their parquets at the same instant, an unhandled exception in a worker
fails that job instead of silently retiring its GPU, and each failed job is
retried once at the end of the queue before being reported.

## Result (sweep completed 2026-09-11 18:30 UTC)

870/870 model-runs, 5 complete folds in every cell, 7.6 days wall-clock, 546
GPU-hours, GPU load 181/182/183 h — the queue held the three devices within
0.5% of each other across a week.

The v004 table was aggregated by two analysis scripts that lived in the sweep's
own `runs/bench/` folder and are not part of this repository (the v005
evaluation tooling replaces them). They pooled v004 with v003 into the
four-horizon picture (36/72/144/288 = 3/6/12/24 h, 36 base-cells x 5 folds) and
ran the paired significance test. Pooling is legitimate: every condition in
`FIXED` matches except `--num_workers`, and the ceil-and-trim change to
segrnn_thuml is verified bit-identical to v003 at pred_len 72/144/288 under a
fixed seed.

Headline numbers, best model per cell, 5-fold mean HSS:

| track | 3 h | 6 h | 12 h | 24 h |
|---|---|---|---|---|
| proton (p>10) | 0.935 | 0.900 | 0.834 | 0.728 |
| xray (xrs_long) | 0.277 | 0.166 | 0.101 | 0.062 |
| multi | 0.907 | 0.851 | 0.762 | 0.586 |

The X-ray track is **horizon-limited rather than unskillful**: its HSS falls 78%
from 3 h to 24 h against the proton track's 22%. A 24-hour-only view reads as a
near-total failure; the 3-hour horizon shows real skill.

`segrnn_thuml` takes 29 of the 36 base-cells and beats every one of the nine
rivals under a one-sided Wilcoxon signed-rank test paired on cells (n=36,
Holm-corrected p < 2.5e-7 for all nine). The overlapping per-cell 95% CIs are an
unpaired view and do not contradict this — the paired test is the correct one.
