# tslib.exp — training loop, losses, metrics

`lightning_model.py` runs training/eval; `exp.py` runs multiple models and
writes a comparison table. Losses and metrics are pluggable via registries.

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

A metric is `fn(pred, true, ctx) -> float | dict`, where `ctx` is a
`MetricContext(thresholds, transform, target_cols)`. Two kinds:

- `kind="regression"` → return a scalar (see `mse`/`mae`/`rmse`).
- `kind="event"` → threshold-crossing skill score; return a per-channel
  `dict` keyed by target name (see `tss`/`hss`/`pod`/`far`). Physical-unit
  thresholds come from `--event_threshold` and are converted to log10 space.

```python
from tslib.exp.metrics import register_metric

@register_metric("my_score", kind="event")
def _my_score(pred, true, ctx): ...
```
Select with `--metrics mse my_score`; event metrics require `--event_threshold`
aligned to `--target_cols`. Results become dynamic columns in the comparison
table (`build_comparison`), event metrics as `myscore_<channel>`.

See `losses.py`, `metrics.py`, and their tests for reference.
