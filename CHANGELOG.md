# Changelog

## 2.0.0 - 2026-10-30

v2.0.0 reworks how forecasts are evaluated and recorded, and merges the v004
short-horizon work into the main line. **Published benchmark numbers change**
(see "Changes to published numbers"); the v1 files are kept unchanged in
[`results/v1/`](results/v1/). Metric definitions: [`docs/metrics.md`](docs/metrics.md).

### Changed (affects numbers)

- **Undefined event scores are NaN instead of 0.0.** `tss`, `hss`, `pod` and
  `far` now return NaN when their formula has no value (for example `far` when a
  model never forecasts an event). v1 wrote `0.0`, which could mean "no skill" or
  "undefined".
- **A NaN or infinite prediction voids the channel.** Every event score and count
  and every regression score of that channel is NaN, and `n_nonfinite` records
  how many samples were affected. v1 counted a NaN prediction as "no event" and
  scored it 0.0, so diverged runs looked like runs without skill.
- **Regression scores are per channel.** The multivariate track used to report
  one `rmse`/`mae` pooled over both channels; v2 reports each channel separately.
- **The leaderboard is built per forecast length** (3 h, 6 h, 12 h, 24 h) from the
  new tables. Each cell is the mean over the folds and input lengths of the runs
  with a defined value, with a `k/n` count when fewer than all n runs of the
  forecast length have one. xPatch is left out at the 7-day input (seq_len
  2016), where it diverges, so its cells average 10 of the 15 runs, show
  `(10/15)` and carry a footnote at every forecast length; the diverged v1.0.0
  runs stay in the tables as NaN rows. Event columns are per-day HSS and TSS.
- **No rounding before averaging.** Scores are stored and averaged unrounded; the
  manuscript's re-scoring rounded each fold's score to three decimals first.

### Added

- **Event evaluation with more scores and three units.** New scores `csi`, `f1`,
  `seds` and `sedi`, the raw contingency counts (`tp` `fp` `fn` `tn` `n`
  `n_events` `n_nonfinite`), and the units `point`, `window` and `day`. `day` is
  one forecast per 24 hours, chosen by a 24-hour cut over the test windows'
  forecast start times and applied to every forecast length.
- **Regression evaluation per channel:** `rmse`, `mae`, `msa`, `sspb`, `logr`.
- **Two result tables:** `results_regression.csv` (one row per run and channel)
  and `results_event.csv` (one row per run, channel and unit), plus
  `comparison_regression.csv` and `comparison_event.csv` for each benchmark cell,
  replacing `results_master.csv` and `comparison.csv`.
- **Saved predictions.** Every run writes `pred/<model>.npz` (predictions,
  observations, each window's forecast start time and run metadata); turn off
  with `--no-save_predictions`. CSV input needs `--time_col` for this and for
  the `day` rows; a CSV run without a time column writes neither.
- **Re-scoring without a GPU:** `scripts/rescore.py` rebuilds the row files and
  both tables from the saved predictions, so a new metric is evaluated on all
  existing runs in minutes.
- **Archive tools:** `scripts/dump_predictions.py` creates prediction files from
  the checkpoints of v1.0.0 and v004 runs, and `scripts/verify_archive.py`
  checks them against the v1 result tables.
- **Metric registries:** `register_event_score` (one function of the four
  contingency counts) and `register_regression_metric` (one function of a
  channel's predictions and observations). `--metrics` is now an optional filter;
  by default every registered metric is computed.
- **Four forecast lengths in one benchmark:** `pred_len` 36 / 72 / 144 / 288
  (3 h / 6 h / 12 h / 24 h), with the v004 multi-GPU job scheduler
  (`tslib/benchmark/scheduler.py`), `--runs_root`, `scripts/run_queue.sh` and
  `scripts/watchdog.sh`. SegRNN decodes whole segments and trims the output to the
  forecast length, so its segment length stays 24 at every forecast length.
- **Documentation:** [`docs/metrics.md`](docs/metrics.md) (formulas, NaN rules,
  units, file formats and the differences from v1 and the manuscript) and
  [`docs/v004-short-horizon-sweep.md`](docs/v004-short-horizon-sweep.md).

### Removed

- `run_metrics`, the `@register_metric` registry and the `mse` metric (the `mse`
  loss is unchanged), and the `metrics`, `skipped` and `est_train_hours` fields of
  `TrainResult`. Code that registered a metric must move to `register_event_score`
  or `register_regression_metric`, whose names must not contain `_`.
- The single `results_master.csv` and the `comparison.csv` of each cell. The v1
  tables live on in `results/v1/`.

### Fixed

- `results/README.md` described the logged event columns as per-window; they were
  per-point (see [`results/v1/README.md`](results/v1/README.md)).
- Documentation of the removed train-time gate, the `@register_metric` API and
  the old result files was corrected (`README.md`, `tslib/README.md`,
  `tslib/exp/README.md`, `DATA.md`, `REPRODUCE.md`).

### Versioning and dependencies

- `pyproject.toml` stood at 0.3.0 while the repository was tagged v1.0.0; the
  package version is now 2.0.0, matching the tag and `CITATION.cff`.
- `statsmodels` is limited to versions below 0.15: 0.15 removed the `old_names`
  argument of `AutoReg`, which the `statistic` strategy (`ar` model) uses.

### Upgrading from v1

- **Run directories written by v1.0.0.** Their `score/rows/<model>.json` files
  are not v2 result records. A v2 run in such a directory (for example the
  README command again, whose default `--run_name` is `default_run`) re-tests a
  model from its checkpoint when `ckpt/<model>.ckpt` and `ckpt/<model>.done`
  exist, and retrains it otherwise; the new row and `pred/<model>.npz` replace
  the old row. Until then an old row is left out of the comparison and master
  tables with a warning. To start over instead, delete `score/rows/<model>.json`.

### Changes to published numbers

**Six diverged xPatch runs.** In v1.0.0, xPatch diverged at the 7-day input
(`seq_len` 2016, fold 0) at both forecast lengths (`pred_len` 144 and 288) in the
`uni_a`, `uni_b` and `multi` tracks. `results/v1/results_master.csv` holds these
six rows with `best_val_loss` `inf`, `rmse` and `mae` NaN, and `0.0` for every
event score of their channels, so the v1 leaderboard averaged them in as runs
without skill. In the v2 tables the same six rows hold NaN in every regression
and event score (all their predictions are non-finite, `n_nonfinite` equals `n`),
and the leaderboard leaves them out with a footnote.

**Everything else, cell by cell.** All 1,746 v1 runs (876 at 12 h / 24 h, 870 at
3 h / 6 h) were re-scored from their checkpoints and compared with the v1 tables at
the point granularity, which is what v1 recorded
(see [`results/v2_verification_summary.md`](results/v2_verification_summary.md)):

| Score | Cells (run × channel) | 0.0 → NaN, xPatch runs | 0.0 → NaN, other runs | Other changes |
|---|---|---|---|---|
| `tss` | 2,328 | 8 | 0 | none |
| `hss` | 2,328 | 8 | 0 | none (largest difference 1.7e-14) |
| `pod` | 2,328 | 8 | 0 | none |
| `far` | 2,328 | 8 | 241 (in 204 runs) | none |
| `rmse`, `mae` | 1,746 runs | NaN in both | — | relative difference ≤ 1.6e-7 |

- **`far` is NaN where a model never forecast an event.** The 241 `far` cells
  outside the xPatch runs belong to runs that issued no alarm. v1 reported them as
  `far = 0.0`, a perfect score. Most are in the X-ray channel (`uni_b` 95 cells,
  `multi` 81) and the rest in the proton channel (`multi` 49, `uni_a` 16). `tss`
  and `hss` of these runs stay 0.0, because they are defined (no skill).
- **No value changed otherwise.** No defined v1 score became NaN, no NaN became a
  value, and no score moved by more than 0.001.
- **Leaderboard at 12 h and 24 h.**
  - The RMSE order is the same as in v1 in every track table.
  - Only the xPatch rows move, because the diverged 7-day runs are now left out
    of the mean instead of counting as 0.0. Examples: proton HSS 0.723 → 0.795
    (12 h, `uni_a`) and 0.609 → 0.670 (24 h).
  - The v2 leaderboard is published per forecast length. It marks that xPatch
    averages 10 runs where the other models average 15 (no 7-day input).
