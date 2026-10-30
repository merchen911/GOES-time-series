# tslib.exp — training loop, losses, metrics

`lightning_model.py` runs training/eval; `exp.py` runs multiple models and
writes the comparison tables (`comparison_regression.csv`,
`comparison_event.csv`) and, per model, a prediction file. Losses and metrics
are pluggable via registries.

## Add a loss (`losses.py`)

A loss is `fn(pred, true) -> scalar tensor`, computed in the stored (log10)
space. Register and select with `--loss`:

```python
from tslib.exp.losses import register_loss

@register_loss("my_loss")
def _my_loss(pred, true):
    return ...
```
Built-ins: `mse` (default), `mae`, `huber`, `weighted_mse`.

## Add a metric (`metrics.py`)

There are two registries, one per kind of evaluation (definitions:
[`docs/metrics.md`](../../docs/metrics.md)). A metric is a plain function that
returns a `float` and never raises; a value that cannot be computed is
`float("nan")`. The name must **not contain `_`** (the sort direction and the
leaderboard read the token before the first `_`), otherwise registration raises.

- **An event (2x2 contingency) score** is a function of the four counts
  `(tp, fp, fn, tn)`; register it with `register_event_score`. The evaluation
  counts the table for every channel and aggregation unit (point / window / day)
  and calls it, so the score is computed everywhere at once.
- **A regression metric** is a function of one channel's flattened
  `(pred, true)` arrays in log10 space; register it with
  `register_regression_metric`.

```python
import numpy as np
from tslib.exp.metrics import register_event_score, register_regression_metric

@register_event_score("gss")                 # Gilbert skill score, from the counts
def _score_gss(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    hits_random = (tp + fp) * (tp + fn) / n if n else 0.0
    denom = tp + fp + fn - hits_random
    return (tp - hits_random) / denom if denom else float("nan")

@register_regression_metric("medae")         # median absolute error
def _reg_medae(pred, true):
    return float(np.median(np.abs(pred - true))) if pred.size else float("nan")
```

Built-ins: events `tss hss pod far csi f1 seds sedi`; regression `rmse mae msa
sspb logr`. Put a new function next to them in `metrics.py`, so that every
process (training, `scripts/rescore.py`) registers it. Every registered metric is
computed for every run and kept in the run's `score/rows/<model>.json`;
`--metrics NAME ...` is an optional filter (the others become NaN) and the
contingency counts are always written. Event metrics need `--event_threshold`
(physical units, one per `--target_cols` entry; converted to log10 automatically).

Registering is all it takes to get a column: the metric columns of
`results_event.csv` / `results_regression.csv` and of the comparison tables are
the registered metrics, in registration order, after the key and count columns
(`records.event_columns()` / `records.regression_columns()`). A
higher-is-better regression metric that should rank the regression comparison
(`--sort_metric`) must be added to `_HIGHER_IS_BETTER` in `exp.py`; `sspb` is signed
and cannot be a `--sort_metric`.

To get a **new metric on runs that already exist**, add it as above and re-score
the saved prediction files (no GPU): `python3.12 scripts/rescore.py --runs
runs/bench --master-dir runs/bench`.

Other pieces of the evaluation: `evaluate.py` (`evaluate(pred, true, ctx)` turns
predictions into regression and event rows; the training path and the re-scoring
tool both call it), `granularity.py` (point / window / day and the 24-hour cut),
`records.py` (row files and the two table layouts), `predictions.py` (the
`pred/<model>.npz` files), `rescore.py` (re-scoring and master merging).

See `losses.py`, `metrics.py`, `evaluate.py`, and their tests for reference.

## Forecast strategies (`strategy.py`, `--forecast_strategy`)

`--forecast_strategy` selects how the model generates predictions across the
forecast horizon:

- `direct` (default) — model emits full `pred_len` in one forward pass.
- `recursive` — train a one-step model, roll out at inference; requires
  forecasting all input channels.
- `statistic` — classical statsmodels (ARIMA/AR/Theta), per-window fit,
  univariate only.

`run_strategy` (in `strategy.py`) dispatches the chosen strategy. To compare
families of runs across strategies (e.g., direct vs. recursive), use
`merge_comparisons(frames, sort_metric)` in `exp.py` — it combines regression
comparison tables into one, keeping the `strategy` column. The `rows/*.json`
records of a run directory also carry the strategy, and the master tables of
the benchmark driver key every run by `strategy` and `model`.
