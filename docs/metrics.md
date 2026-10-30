# Metrics and evaluation (v2)

This page defines every number the benchmark reports from v2.0.0 on: the event
(yes/no) scores, the regression scores, the three ways of counting events, the
rules for values that cannot be computed, and the files the results and the
saved predictions are written to. The code is `tslib/exp/metrics.py` (formulas),
`tslib/exp/granularity.py` (aggregation units) and `tslib/exp/evaluate.py`
(the evaluation that ties them together). The training path and the re-scoring
tool both call the same `evaluate`, so a number has one implementation.

Two kinds of evaluation are recorded separately for every run:

- **Event evaluation** turns a forecast into a yes/no call ("will the flux reach
  the threshold?") and scores it on a 2x2 contingency table.
- **Regression evaluation** compares predicted and observed flux values.

## 1. Event scores

An *event* is a value at or above a threshold (`>=`, the threshold itself
counts). The threshold is given per target channel in physical units with
`--event_threshold` (for example `10` pfu for `p_gt10`, `1e-5` W m-2 for
`xrs_long`, the M-class flare level) and is converted to the stored space
(log10 by default) before the comparison.

For one channel and one aggregation unit (section 3) every sample is a hit,
false alarm, miss or correct negative:

| | observed event | observed no event |
|---|---|---|
| **forecast event** | a = TP | b = FP |
| **forecast no event** | c = FN | d = TN |

with n = a + b + c + d, hit rate H = a / (a + c) and false-alarm rate
F = b / (b + d).

| name | formula | NaN when | better |
|---|---|---|---|
| `pod` | a / (a + c) | a + c = 0 (no observed event) | higher |
| `far` | b / (a + b) | a + b = 0 (never forecast an event) | lower |
| `tss` | H - F | a + c = 0 or b + d = 0 | higher |
| `hss` | 2(ad - bc) / [(a + c)(c + d) + (a + b)(b + d)] | denominator = 0 | higher |
| `csi` | a / (a + b + c) | a + b + c = 0 | higher |
| `f1` | 2a / (2a + b + c) | 2a + b + c = 0 | higher |
| `seds` | [ln((a + b)/n) + ln((a + c)/n)] / ln(a/n) - 1 | a = 0 or a = n | higher |
| `sedi` | [ln F - ln H - ln(1 - F) + ln(1 - H)] / [ln F + ln H + ln(1 - F) + ln(1 - H)] | H or F is 0 or 1 (or undefined), or the denominator is 0 | higher |

`tss` is the true skill statistic (Hanssen-Kuipers), `hss` the Heidke skill
score, `pod` the probability of detection, `far` the false-alarm ratio, `csi`
the critical success index (threat score), `seds` the symmetric extremal
dependence score and `sedi` the symmetric extremal dependence index. SEDS and
SEDI are designed for rare events, which is why they are reported next to the
scores that rare events push towards zero (Stephenson et al. 2008, *Meteorol.
Appl.* 15; Ferro and Stephenson 2011, *Weather and Forecasting* 26).

Together with the scores, the raw counts are always written: `tp`, `fp`, `fn`,
`tn`, `n` (samples in the unit), `n_events` (= tp + fn, observed events) and
`n_nonfinite`. A score that is not in the list above can be computed from the
counts alone, so a new 2x2 score never needs the predictions.

## 2. Regression scores

Regression scores are computed per target channel over every window and lead
time of the test split. With delta = predicted - observed in log10 space:

| name | formula | note |
|---|---|---|
| `rmse` | sqrt(mean(delta^2)) | lower is better |
| `mae` | mean(abs(delta)) | lower is better |
| `msa` | 100 (10^median(abs(delta)) - 1) % | median symmetric accuracy; lower is better |
| `sspb` | 100 sign(median(delta)) (10^abs(median(delta)) - 1) % | symmetric signed percentage bias; 0 is best, positive = over-forecast, negative = under-forecast |
| `logr` | Pearson r between log10 predicted and log10 observed | higher is better; NaN when either series is constant |

`msa` and `sspb` are the log-accuracy-ratio measures of Morley, Brito and
Welling (2018, *Space Weather* 16): a value of 100 % means the typical forecast
is off by a factor of 2.

With the default `--transform log10` the stored values already are log10 values.
With `--transform none` (flux stored in physical units) `rmse` and `mae` are
computed on the stored values, as in v1, while `msa`, `sspb` and `logr` take
log10 first and are NaN when any predicted or observed value is zero or negative.

`sspb` is signed, so neither sort direction ranks models by it and it is
rejected as `--sort_metric`. With a `--metrics` filter, `--sort_metric` must be
one of the metrics computed (or `best_val_loss`), otherwise the sort column
would be empty.

## 3. Aggregation units for events

Event scores are computed for three units, each a row of the event table:

| unit | samples | an event is |
|---|---|---|
| `point` | every (test window, lead time) value, N x `pred_len` of them | a single value >= threshold, judged separately for forecast and observation |
| `window` | every test window (N; consecutive windows start 5 minutes apart and overlap) | the maximum over the forecast range >= threshold, judged separately for forecast and observation |
| `day` | the test windows kept by the 24-hour cut below | the same maximum rule as `window` |

**The 24-hour cut.** The test windows are sorted by forecast start time (the
UTC time of the first forecast step). The first window is kept; then, again and
again, the first window that starts at least 24 hours after the last kept one.
After a gap in the data (windows missing for lack of valid input) the next
window after the gap is kept immediately. Among selections whose spacing is at
least 24 hours this keeps the most windows. The choice depends only on the
observed times, so every model of a benchmark cell is scored on the same
windows. A `day` sample is therefore "the forecast issued once per 24 hours".

The cut is the same for every forecast length (3 h, 6 h, 12 h, 24 h): a 3 h
forecast issued once per day is judged on its own 3 hours, not on the whole day.
The cut needs the windows' times: parquet input always has them, CSV input only
with `--time_col`. A CSV run without a time column writes `point` and `window`
rows but no `day` rows.
For the reference cell (X-ray at the M-class threshold, input 288 steps,
forecast 144 steps, fold 0) the cut keeps 876 windows, 145 of them with an
observed event, and at least 24.0 hours lie between consecutive kept windows.
The number of kept windows differs from cell to cell.

## 4. Values that cannot be computed

A metric function never raises and never returns a placeholder number: when its
formula has no value, it is `NaN` (the NaN column of the tables above).

- **v1 wrote 0.0 for these cases, v2 writes NaN.** A `0.0` in v1 could mean "no
  skill" or "undefined"; a NaN is always "undefined".
- **Non-finite predictions void a channel.** If the predictions of a channel
  contain any NaN or infinity, every event score *and* every count of that
  channel is NaN, in all three units, and all of its regression scores are NaN.
  `n_nonfinite` records how many non-finite samples the unit itself holds
  (values for `point`, windows for `window` and `day`). A `day` row can therefore
  show `n_nonfinite = 0` with NaN counts when the non-finite value sits in a
  window the 24-hour cut dropped: the channel is voided as a whole, not unit by
  unit. In v1 a NaN prediction was counted as "no event" and scored 0.0.
- **Nothing is rounded.** Scores are stored and averaged unrounded.
- **Averages leave NaN out.** The README leaderboard averages the per-run values
  of a model and skips NaN ones; each cell shows `k/n` when only k of the n runs
  of its forecast length (folds x input lengths) have a value, so a model is not
  rewarded for undefined scores. A model with fewer runs than that (xPatch, left
  out at the 7-day input) is marked with a dagger and footnoted, because its
  means are over other inputs than the other rows'.

## 5. Result files

One run (a model in one benchmark cell) writes its rows to
`<runs_root>/<run_name>/score/rows/<model>.json`; the benchmark driver folds all
cells into two master tables, `results_regression.csv` and `results_event.csv`
(the released ones are in [`results/`](../results/)). The run key is
`track`, `seq_len`, `pred_len`, `fold`, `strategy`, `model`.

`results_regression.csv`, one row per run and channel:

| column | meaning |
|---|---|
| run key | `track`, `seq_len`, `pred_len`, `fold`, `strategy`, `model` |
| `best_val_loss` | best validation loss of the training (not recomputable from predictions) |
| `channel` | target channel (`p_gt10`, `xrs_long`) |
| `n` | number of values scored (windows x lead times) |
| `rmse` `mae` `msa` `sspb` `logr` | section 2 |

`results_event.csv`, one row per run, channel and unit:

| column | meaning |
|---|---|
| run key | as above |
| `channel` | target channel |
| `granularity` | `point`, `window` or `day` |
| `threshold` | the event threshold in physical units |
| `n` `n_events` `tp` `fp` `fn` `tn` `n_nonfinite` | counts, section 1 and 4 |
| `tss` `hss` `pod` `far` `csi` `f1` `seds` `sedi` | section 1 |

In the multivariate track each channel has its own rows; the pooled
two-channel `rmse`/`mae` of v1 does not exist in v2. Each benchmark cell also
keeps its own `score/comparison_regression.csv` and `score/comparison_event.csv`
(the cell's models, best first).

## 6. Prediction files

After testing, every run writes its test predictions to
`<runs_root>/<run_name>/pred/<model>.npz` (on by default; turn off with
`--no-save_predictions`). The file needs each window's forecast start time, so
a CSV run without `--time_col` writes no prediction file (it prints a
warning); parquet runs and CSV runs with a time column always can. The file is self-contained, so a run can be re-scored
without its model, its data or a GPU:

| array | content |
|---|---|
| `pred` | float32, shape (N, `pred_len`, T): predicted values in the stored space |
| `true` | float32, same shape: observed values |
| `forecast_start` | int64, shape (N,): UTC time of each window's first forecast step, nanoseconds since the epoch |
| `meta` | JSON text: `model`, `strategy`, `run_name`, `seq_len`, `pred_len`, `fold_numb`, `transform`, `cadence_min`, `event_threshold`, `target_cols`, `tslib_version`, `format` |

The file is written to a temporary name and renamed, so an interrupted run never
leaves a half-written file. `tslib.exp.predictions.load_predictions(path)`
returns `(pred, true, forecast_start, meta)`. When you pass times yourself (to
`save_predictions`, or as `MetricContext.forecast_start` to `evaluate`), give
int64 nanoseconds or a `datetime64` array of any unit, which is converted; a
float array is rejected. pandas 3 builds `datetime64[us]` indexes, so their
`.asi8` or `.view("int64")` would be microseconds and would break the 24-hour cut.

Re-scoring (no GPU) rebuilds the row files, the comparison tables and the two
master tables from the prediction files of a tree of benchmark cells:

```bash
python3.12 scripts/rescore.py --runs runs/bench --master-dir runs/bench --workers 8
```

A cell where a result row has no prediction file (a failed save, or a run
with `--no-save_predictions`) is refused and the master tables are not
rebuilt, because that row would keep the numbers of the older rule; re-run the
model (a checkpoint with its `.done` marker is re-tested, not retrained) or
delete the row. The benchmark scheduler and driver likewise count a run as
finished only when both its row and its prediction file exist.

`--metrics NAME ...` re-scores only some metrics; `--merge-masters DIR ... --out
DIR` joins master tables built from separate trees. This is how a new metric is
evaluated on all existing runs: register it (see
[`tslib/exp/README.md`](../tslib/exp/README.md)) and re-score.

`tslib_version` is the installed package version, or the version in
`tslib/__init__.py` when the package is run from a checkout without being
installed, followed by `+g<commit>` (and `.dirty` for uncommitted changes) when
the code runs from a git checkout.

Checkpoints of runs made before v2 (v1.0.0 and the v004 short-horizon sweep)
were turned into prediction files once with `scripts/dump_predictions.py`. Those
1,746 archived files carry `tslib_version` `unknown`: the package was not
installed for the dump, and the fallback above came later. They were written by
the code of commit `966700c` (where the dump tool was added); the inference,
evaluation and file-writing code it used is unchanged up to `18d2c80`, the
commit the archive was re-scored with. `scripts/verify_archive.py` checks the
dumped predictions against the v1 result tables: `rmse` and `mae` must agree to
a relative 1e-4, and an event score must agree up to at most five flipped threshold-boundary points (GPU arithmetic is not
bit-reproducible).

## 7. Differences from v1 and from the paper

| | v1 (up to v1.0.0, v004) | v2 |
|---|---|---|
| undefined event score | 0.0 | NaN |
| NaN prediction | counted as "no event" | whole channel NaN, `n_nonfinite` recorded |
| event units | `point` only | `point`, `window`, `day` |
| event scores | `tss` `hss` `pod` `far` | plus `csi` `f1` `seds` `sedi` and the raw counts |
| regression scores | `rmse`, `mae` pooled over all target channels | `rmse` `mae` `msa` `sspb` `logr` per channel |
| result files | one `results_master.csv` | `results_regression.csv` and `results_event.csv` |
| predictions | not kept | `pred/<model>.npz` per run |
| leaderboard | mean of all runs, undefined scores as 0.0 | per forecast length, undefined runs left out and counted |

The manuscript's event tables (`results/v1/event_skill_5fold_ci*.csv`) were
produced by re-running inference from checkpoints with a separate script. Their
numbers differ from v2 values in three ways:

- **`per-day` is defined differently.** The manuscript took every 288th window
  of the test list (796 samples in the reference cell above); v2 applies the
  24-hour cut (876 samples).
- **Rounding.** The manuscript rounded each fold's score to three decimals before
  averaging; v2 does not round.
- **Inference path.** The manuscript's re-scoring forwarded batches of 512 windows
  directly, the benchmark test (and the v2 archive dump) uses Lightning's
  `trainer.test` with batches of 64. A forecast that sits exactly on a threshold
  can fall on either side, so contingency counts can differ by a few points.

The v1 result files are kept unchanged in [`results/v1/`](../results/v1/) so the
published v1.0.0 numbers stay reproducible.
