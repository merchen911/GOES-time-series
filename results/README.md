# Benchmark results

Two generations of result files live here:

- **v2 (this folder):** the tables of the v2.0.0 evaluation, described below.
  Metric definitions, NaN rules and the three event units are in
  [`../docs/metrics.md`](../docs/metrics.md).
- **v1 ([`v1/`](v1/)):** the files published up to v1.0.0, plus the v004
  short-horizon runs, kept unchanged. [`v1/README.md`](v1/README.md) says what
  each file contains and how its event columns were computed. v1 and v2 values
  are not directly comparable where a v1 score was an undefined `0.0`.

## `results_regression.csv`

One row per run and target channel: `track` x `seq_len` x `pred_len` x `fold` x
`strategy` x `model` x `channel`. A run is one model trained in one benchmark
cell. The cells cover three tracks (`uni_a` proton `p_gt10`, `uni_b` X-ray
`xrs_long`, `multi` both), input lengths 288 / 864 / 2016 steps (1 / 3 / 7 days),
forecast lengths `pred_len` 36 / 72 / 144 / 288 steps (3 h / 6 h / 12 h / 24 h on
the 5-minute grid) and the five folds 0-4 of the rotating term-based split.

| column | meaning |
|---|---|
| `best_val_loss` | best validation loss of the training run |
| `channel`, `n` | target channel and number of values scored |
| `rmse` `mae` | root-mean-square and mean absolute error, log10 flux |
| `msa` `sspb` | median symmetric accuracy and symmetric signed percentage bias, % |
| `logr` | correlation between log10 predicted and observed flux |

## `results_event.csv`

One row per run, channel and aggregation unit (`granularity`: `point`, `window`
or `day`), at the operational thresholds: proton >= 10 pfu (`p_gt10`), X-ray >=
1e-5 W m-2, the M-class flare level (`xrs_long`).

| column | meaning |
|---|---|
| `granularity`, `threshold` | unit and threshold (physical units) |
| `n` `n_events` `tp` `fp` `fn` `tn` | samples, observed events and the 2x2 contingency counts |
| `n_nonfinite` | non-finite predicted samples in the unit; a non-finite prediction anywhere voids every score and count of the channel |
| `tss` `hss` `pod` `far` `csi` `f1` `seds` `sedi` | event scores; **NaN where undefined** |

Because the counts are stored, any further 2x2 score can be computed from this
file alone. A NaN is a score that has no value (for example `far` for a model
that never forecast an event), not a zero.

## How the tables are used

The README leaderboard is built from these two files by
`scripts/make_leaderboard.py`: one table per forecast length and track, each cell
the mean of a model's runs with a defined value over the folds and input lengths
of that forecast length, regression columns RMSE, MSA and logR per channel and
per-day HSS and TSS. `k/n` marks a cell where only k of the n runs of the
forecast length (5 folds x 3 input lengths = 15) have a value. xPatch is left
out at the 7-day input (`seq_len` 2016), where its training diverges, so its
cells average 10 runs (1-day and 3-day inputs) and show `(10/15)`; its rows are
marked with a dagger and footnoted at every forecast length, because they are
not directly comparable with the others. The six xPatch runs at the 7-day input
that did train (fold 0, from v1.0.0, 12 h and 24 h) diverged; they stay in the
tables as NaN rows and are left out of the leaderboard.

The tables can be rebuilt from saved predictions without a GPU
(`scripts/rescore.py`, see [`../REPRODUCE.md`](../REPRODUCE.md)). Older runs
(v1.0.0 and v004) were re-scored from predictions dumped from their checkpoints
(by the code of commit `966700c`; the files' `tslib_version` reads `unknown`,
see [`../docs/metrics.md`](../docs/metrics.md) section 6).
