# v1 results (preserved)

These are the result files produced by the v1 evaluation (releases up to
v1.0.0, plus the v004 short-horizon runs). They are kept unchanged so the
published numbers stay reproducible and so the v2 evaluation can be checked
against them. New runs write the v2 result files described in
[`../README.md`](../README.md) and [`../../docs/metrics.md`](../../docs/metrics.md),
not these.

## How the v1 event columns were computed

The event columns in `results_master*.csv` (`tss_*`, `hss_*`, `pod_*`, `far_*`
for `p_gt10` and `xrs_long`) were logged at test time, with these rules:

- **Aggregation unit: per-point.** Every window x lead-time sample is one
  contingency-table entry. There is no per-window or per-day variant in these
  columns. (Older documentation that called these columns "per-window" was wrong.)
- **Undefined ratios were replaced by 0.0.** When a ratio had a zero
  denominator (no observed events, no alarms, or a degenerate table) it was
  set to `0.0` instead of a missing value: `pod` and `far` became `0.0`
  directly, `hss` became `0.0`, and `tss` was computed as POD - POFD with an
  undefined POD or POFD taken as `0.0`. A `0.0` in these columns can therefore
  mean "no skill" or "undefined". A channel that does not exist in a track
  (for example `xrs_long` in the proton-only track `uni_a`) is blank, not
  `0.0`.
- **NaN predictions counted as "no event"**, which could also produce `0.0`
  scores for diverged runs.
- `rmse` and `mae` are over all target channels pooled together, in log10
  space.

The v2 evaluation (see [`../../docs/metrics.md`](../../docs/metrics.md)) returns NaN for undefined
scores, scores each channel separately, and reports three aggregation units
(point, window, day). Values in the files below are therefore not directly
comparable to v2 values for undefined cells.

## Files

### Benchmark result tables

| File | Contents |
|---|---|
| `results_master.csv` | v003 / v1.0.0 benchmark: one row per `track` x `seq_len` x `pred_len` x `fold` x `model` (876 rows; `pred_len` 144 and 288, `seq_len` 288/864/2016, folds 0-4, ten models). Columns: validation loss, `rmse`, `mae`, and the per-point event columns above. |
| `results_master_v004_pred36_72.csv` | Same schema for the v004 short-horizon runs, `pred_len` 36 (3 h) and 72 (6 h) (870 rows). Together with `results_master.csv` it covers the four horizons 3 h / 6 h / 12 h / 24 h. |

### Manuscript event-skill tables

| File | Contents |
|---|---|
| `event_skill_5fold_ci.csv` | Event skill **re-scored at the reference cell** `seq_len=288`, `pred_len=144` (12 h): 5-fold mean and 95% confidence-interval half-width (Student-t, df=4) of TSS, HSS, CSI, MCC, SEDI, precision and FB. One row per `track` x `model` x `gran` (`per-point`, `per-window`, `per-day`). |
| `event_skill_5fold_ci_24h.csv` | Same, at `pred_len=288` (24 h). |

Tracks are `Proton SPE` (>= 10 pfu), `XRS M-class` (>= 1e-5 W m-2) and
`XRS X-class` (>= 1e-4 W m-2), plus the `(multi)` variants scored from joint
multivariate input. `naive (persistence)` is the flat-value persistence floor.
Blank cells are undefined metrics. These are the paper's re-scoring, which
re-ran inference from checkpoints; they are not the logged columns of
`results_master.csv`.

### Text reports (v004, 2026-09-14)

| File | Contents |
|---|---|
| `aggregate_5fold_v004_2026-09-14.txt` | 5-fold aggregate of the 870 v004 rows (3 h / 6 h): per-cell champion with 95% CI, model standings, track difficulty and statistically tied cells. Metric: HSS (the primary per track). |
| `horizon_scaling_2026-09-14.txt` | The same rows pooled with the 876 v003 rows (1746 rows): skill versus forecast horizon (3 h to 24 h), one-sided Wilcoxon signed-rank tests of the champion against each rival, and champion count over all 36 base cells. |

### `analysis/` - manuscript re-scoring outputs (2026-09-15)

These CSVs are outputs of the paper's re-scoring (inference re-run from
checkpoints; the confusion-table method of the manuscript's analysis scripts,
which are not part of this repository). They record
every per-fold row, with the contingency counts (`TP FP FN TN`, `n`,
`n_events`, `base`) beside the scores, so other scores can be derived from
them later. Per-fold scores are rounded to three decimals. The `gran` column is
`per-point`, `per-window` or `per-day`. The model name `segrnn` here is
`segrnn_thuml` in `results_master*.csv`.

| File | Contents |
|---|---|
| `model_skill_short_horizon_perfold_2026-09-15.csv` | All ten models plus `naive (persistence)`, `pred_len` 36 / 72, reference lookback `seq_len=288`, folds 0-4, uni and `(multi)` tracks. |
| `model_skill_short_horizon_summary_2026-09-15.csv` | 5-fold mean and 95% CI half-width of HSS, TSS and SEDI for the file above. |
| `model_skill_history_perfold_2026-09-15.csv` | The same models at the longer lookbacks `seq_len` 864 (3 days) and 2016 (7 days), `pred_len` 36 / 72 / 144 / 288, folds 0-4, uni tracks. |
| `floors_short_horizon_perfold_2026-09-15.csv` | Persistence floors (`flat-value persistence`: the last input value held flat; `occurrence-persistence`: whether the preceding window of equal length had an event) at `pred_len` 36 / 72, `per-window` and `per-day`, folds 0-4. Pure data, no model inference. |
| `floors_short_horizon_summary_2026-09-15.csv` | 5-fold mean and 95% CI half-width of HSS, TSS, SEDI and base rate for the file above. |
| `floors_history_perfold_2026-09-15.csv` | The persistence floors at `seq_len` 864 and 2016, `pred_len` 36 / 72 / 144 / 288, folds 0-4. |

The scripts that produced these tables hard-coded machine-specific paths and
are not part of the repository. The v2 re-scoring tool (`scripts/rescore.py`)
does not rebuild these tables: it re-scores the per-run rows of the benchmark
runs from their prediction files (unrounded, with the v2 NaN rules), into
`../results_regression.csv` and `../results_event.csv`. The persistence floors
have no prediction files, and the 5-fold confidence-interval summaries are not
part of the tool, so these files stay the record of the manuscript's analysis.
The v004 working directory never committed these six CSVs
(they were untracked there), so the copies here are the committed record of
what the manuscript's short-horizon and history analyses were scored with.
