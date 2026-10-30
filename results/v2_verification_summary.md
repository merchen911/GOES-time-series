# v2 results: dump and verification summary

The v2 tables `results_regression.csv` and `results_event.csv` were not produced by
re-training. Each of the 1,746 published v1 runs was re-scored from its original
checkpoint:

1. `scripts/dump_predictions.py` ran one test pass per checkpoint (Lightning
   `trainer.test`, batch size 64, one GPU), the same inference path that produced
   the v1 numbers, and saved the predictions.
2. `scripts/verify_archive.py` checked the saved predictions against the v1 tables.
3. `scripts/rescore.py` built the v2 tables from the saved predictions.

## Runs

| Sweep | Forecast lengths | Runs dumped | Failures | Prediction files |
|---|---|---|---|---|
| v003 (published v1.0.0) | 144, 288 (12 h, 24 h) | 876 / 876 | 0 | 245 GB |
| v004 | 36, 72 (3 h, 6 h) | 870 / 870 | 0 | 72 GB |

The dump took about 1.4 h per sweep on four TITAN Xp GPUs. The prediction files
are not part of the repository.

## Fidelity check against v1 (pass condition)

The check recomputes the v1 rules from the dumped predictions:
- per-point event scores, with 0.0 for undefined values;
- RMSE and MAE pooled over all channels.

It compares the result with `v1/results_master.csv` (v003) and
`v1/results_master_v004_pred36_72.csv` (v004). The pass conditions are:
- **RMSE and MAE:** relative difference ≤ 1e-4.
- **Event scores:** the difference must be explainable by at most 5 forecast
  points flipping at the threshold.

| Sweep | Runs | Metric rows | Failing | Event scores identical | Largest RMSE / MAE difference |
|---|---|---|---|---|---|
| v003 | 876 | 6,424 | 0 | all (difference 0.0) | 2.4e-6 / 2.8e-7 |
| v004 | 870 | 6,380 | 0 | all (difference 0.0) | 7.1e-7 / 3.5e-7 |

Every v1 event score (`tss`, `hss`, `pod`, `far`) was reproduced exactly. In other
words, no forecast point flipped at a threshold. The only differences are in the
last digits of RMSE and MAE, from float accumulation order.

## Cross-check with the manuscript's re-scoring (reference)

- **Formula check (pass condition):** the paper's per-fold tables
  (`event_skill{,_p288}_fold<k>.csv`, seq 288, 6,600 rows) were recomputed from
  their own TP/FP/FN/TN counts with the v2 formulas. TSS, HSS, CSI and SEDI agree
  within the paper's 3-decimal rounding in every row (0 failing).
- **Count comparison (reference only):** the same 800 model × channel × granularity
  tables, per-point and per-window, were compared with the dumped predictions.
  - 754 are identical.
  - 41 differ by 1 count, 2 by 2, 2 by 3 and 1 by 4.

  The manuscript re-scored with a different inference path (raw forward pass at
  batch size 512). That is why it differs from v1 at a few threshold-boundary
  points, while the dump reproduces v1 exactly.

## v2 tables

- **`results_regression.csv`:** 2,328 rows, one per run × channel. That is 876 + 870
  runs, and the multi track has two channels.
- **`results_event.csv`:** 6,984 rows, one per run × channel × granularity
  (point, window, day).

The six diverged xPatch runs (seq 2016, fold 0, v003) predict NaN. Under the v2
rule their regression and event scores are NaN, not 0.0. They account for 8 rows
per table and granularity.
