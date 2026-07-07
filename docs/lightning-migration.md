# Lightning Migration (v002 → v003)

**Status:** shipped. This document describes the training architecture as it
exists in v003 — a real `pytorch_lightning` `Trainer`/`LightningModule` pair,
replacing v002's hand-rolled training loop.

## Why

v002's `pl_model` was a hand-written training loop: its own `for epoch in
range(...)` over batches, its own `.train()`/`.eval()` toggling, its own
best-checkpoint tracking by watching a validation loss variable, and no
built-in way to bail out of a run that turned out to be far too slow to
finish. Every one of those concerns had to be reimplemented and re-debugged
for each new experiment.

v003 uses `pytorch_lightning` 2.6.5 directly. The training loop, validation
loop, device placement, and checkpoint bookkeeping are all `Trainer`
responsibilities now. Model authors interact with the framework through the
standard Lightning hooks (`training_step`, `validation_step`, `test_step`,
`configure_optimizers`) instead of a bespoke loop, so anything a Lightning
user already knows — callbacks, loggers, precision flags, multi-GPU
strategies — applies here without extra glue code.

## Architecture

The entry point for a neural model is `run_strategy` in
`tslib/exp/strategy.py`. For `strategy in ("direct", "recursive")` it
dispatches to `_run_neural`, which:

1. Builds the underlying model via `build_model(...)` (the same
   backbone/adapter registry as before — LSTM, TimesNet, PatchTST,
   iTransformer, MICN, etc.).
2. Wraps it in `ForecastModule(model, config, ctx, strategy=strategy)`, a
   `pl.LightningModule` defined in `tslib/exp/lightning_model.py`.
3. Constructs a `pl.Trainer` with two callbacks — a `ModelCheckpoint` and a
   `TimingGateCallback` (see below) — plus a `CSVLogger`, and calls
   `trainer.fit(module, data_bundle.train_loader, data_bundle.val_loader)`.
4. If the run wasn't gated off, calls
   `trainer.test(module, data_bundle.test_loader, ckpt_path="best")` and reads
   back `module.test_metrics`.

`ForecastModule` itself is small: `forward` delegates to the wrapped model,
`training_step`/`validation_step` compute the configured loss
(`tslib/exp/losses.py`) against `y[:, :pred.shape[1], :]`, `test_step`
accumulates predictions/targets, and `on_test_epoch_end` runs
`run_metrics(...)` (`tslib/exp/metrics.py`) once over the full concatenated
test set. `configure_optimizers` returns a plain `torch.optim.Adam` built
from `config.lr` / `config.weight_decay`.

For `strategy == "statistic"`, `run_strategy` instead builds a
`StatisticalRunner` (also in `tslib/exp/strategy.py`). This path is
unchanged by the Lightning migration: it is a plain per-window
fit/forecast loop over `STAT_REGISTRY` (ARIMA/AR/Theta via `statsmodels`),
with no `Trainer`, no `LightningModule`, and no GPU — it doesn't train
gradient-based weights, so there is nothing for Lightning to manage.

Both paths return the same `TrainResult` dataclass (`model_name`,
`best_val_loss`, `metrics`, `ckpt_path`, `strategy`, `skipped`,
`est_train_hours`), so `tslib/exp/exp.py`'s `build_comparison` can rank
neural and statistical runs side by side.

## Customizing

Model-specific training behavior lives entirely in `ForecastModule`
(`tslib/exp/lightning_model.py`) — edit it in place, or subclass it and swap
it in at the point `_run_neural` constructs
`ForecastModule(model, config, ctx, strategy=strategy)`, if you need a
per-model variant. The hooks to change:

- **`training_step`** — the loss computed per training batch (currently the
  configured criterion against the target, truncated to the prediction
  length).
- **`validation_step`** — logs `val_loss`; this is the metric
  `ModelCheckpoint` (below) selects the best checkpoint on, so changing what
  it computes changes what "best" means.
- **`test_step`** / **`on_test_epoch_end`** — controls which metrics get
  written into the comparison table; `run_metrics` and the metric registry
  live in `tslib/exp/metrics.py`.
- **`configure_optimizers`** — swap in a different optimizer, add an LR
  scheduler, etc.

One detail worth calling out: the wrapped model/adapter's own train/eval
branching is preserved automatically. `RecursiveForecastAdapter`, for
example, does 1-step-ahead prediction while in training mode but a full
autoregressive rollout while in eval mode (used for `validation_step` and
`test_step`). Because `ForecastModule` is a `pl.LightningModule`, Lightning
itself calls `.train()` before the training loop and `.eval()` before
validation/test loops — so the adapter sees the correct mode without
`ForecastModule` having to toggle it explicitly.

## Train-time gate

Some backbones (TimesNet in particular) can be dramatically slower per
epoch than others on the same hardware, especially at long `--seq_len` /
`--pred_len`. Running a full multi-model comparison would otherwise risk one
slow model consuming the whole time budget. The `TimingGateCallback`
(`tslib/exp/callbacks.py`) is a `pl.Callback` attached to every neural run
that estimates the full training time from a handful of probe batches and
applies a policy before the run is allowed to continue.

It works by timing `on_train_batch_start`/`on_train_batch_end` for the first
`--probe_batches` training batches (CUDA-synchronizing around each so GPU
work is actually accounted for), taking the **median** duration to absorb
first-batch CUDA/cuDNN warmup, then extrapolating:

```
est_train_hours = median(sec/batch) * trainer.num_training_batches * trainer.max_epochs / 3600
```

The three flags governing this (all in `tslib/configs/config.py`):

| flag | default | meaning |
|---|---|---|
| `--max_train_hours` | `6.0` | estimated training time above which the gate policy fires |
| `--on_slow` | `skip` | policy when the estimate exceeds the limit: `skip` / `abort` / `proceed` |
| `--probe_batches` | `3` | number of training batches timed to produce the estimate |

Regardless of outcome, the gate always prints one line once the estimate is
available:

```
[gate] model=<ModelClassName> s/batch=<sec> est_train~<hours>h (limit <max_train_hours>h)
```

If the estimate is within the limit, training proceeds normally and nothing
else changes. If it exceeds the limit, the configured `--on_slow` policy
applies:

- **`abort`** (fail loudly) — raises `RuntimeError` immediately, stopping the
  whole run.
- **`skip`** (default) — sets `trainer.should_stop = True` (training stops
  after the current step) and marks the module as gated. Back in
  `_run_neural`, `module._gate_skipped` is checked after `trainer.fit(...)`
  returns; if set, the model is short-circuited straight to a `TrainResult`
  with `skipped=True`, `est_train_hours` set to the estimate, empty
  `metrics`, and **no** `trainer.test(...)` call. In `tslib/exp/exp.py`,
  `build_comparison` includes `skipped` and `est_train_hours` as columns in
  the comparison table for every run, and `run_experiment` prints one line
  per skipped model to stdout:

  ```
  SKIPPED (too slow): <model_name> ~<est_train_hours>h — re-include with --on_slow proceed
  ```

- **`proceed`** — log-only: the estimate is still printed and still recorded
  in `est_train_hours`, but training continues to completion and the model
  is scored normally.

To force a model that was previously skipped back into a run, pass
`--on_slow proceed` (there is no per-model override — it applies to every
model in that invocation).

**Scope of the estimate:** the gate estimates **training** time only — the
epochs the `Trainer` will run, extrapolated from probed training batches. It
says nothing about evaluation cost. For the `recursive` strategy in
particular, evaluation is far more expensive than training per window (a
model like LSTM rolls `pred_len` sequential single-step calls per test
window during `validation_step`/`test_step`, vs. one step during
`training_step`), and that cost is outside what this gate measures.
Recursive-rollout evaluation cost is instead controlled separately, at the
benchmark-design level, by sub-sampling the test windows fed to the eval
loop (see `docs/benchmark-conditions.md`, "Deferred: statistic" section, for
the same sub-sampling pattern applied to the `statistic` strategy).

## Checkpointing

`_run_neural` attaches a `pytorch_lightning.callbacks.ModelCheckpoint`
configured as `ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1,
dirpath=<ckpt_dir>, filename=<model>)`, where `<ckpt_dir>` /  `<model>` are
derived from the `ckpt_path` (`runs/<run_name>/ckpt/<model>.pt`) passed in
by `run_experiment`. Concretely this means: after every validation epoch,
Lightning compares the just-logged `val_loss` (the value `validation_step`
logs) against the best seen so far, and if it improves, overwrites the saved
checkpoint — so at most one checkpoint file, the best one, exists per model
at `runs/<run_name>/ckpt/<model>.ckpt`.

Once `trainer.fit(...)` completes (and the run wasn't gated off), the test
pass reloads that best checkpoint rather than whatever weights happen to be
in memory at the end of training:

```python
trainer.test(module, data_bundle.test_loader, ckpt_path="best")
```

`ckpt_path="best"` tells the `Trainer` to look up the path tracked by the
`ModelCheckpoint` callback it was constructed with and restore those weights
before running the test loop — so a model that overfit in its final epochs
is still scored at its best validation checkpoint, not its last one. The
scalar val loss at that checkpoint is read back via
`ckpt_cb.best_model_score` and reported as `best_val_loss` in the comparison
table.
