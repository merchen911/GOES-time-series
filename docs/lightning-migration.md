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
3. Constructs a `pl.Trainer` with two callbacks — a `ModelCheckpoint` and an
   `EarlyStopping` (see "Early stopping + long epochs" below) — plus a
   `CSVLogger`, and calls
   `trainer.fit(module, data_bundle.train_loader, data_bundle.val_loader)`.
4. Calls `trainer.test(module, data_bundle.test_loader, ckpt_path="best")` and
   reads back `module.test_metrics`.

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

## Early stopping + long epochs

v003 no longer estimates a run's total training time up front and gates
slow models off before they start. Instead, every neural run is simply
given a long training budget (`--epochs`, default `10000`, in
`tslib/configs/config.py`) and relies on `EarlyStopping` to stop it once it
stops improving:

```python
early_cb = EarlyStopping(monitor="val_loss", mode="min",
                         patience=config.early_stop_patience)
```

(`tslib/exp/strategy.py`, `_run_neural`). `--early_stop_patience` (default
`10`) is the number of epochs `val_loss` is allowed to fail to improve
before `Trainer.fit(...)` stops the run — so `--epochs` is a ceiling that is
essentially never reached in practice; the real stopping point is whichever
comes first, `patience` epochs without a `val_loss` improvement or
`--epochs` total epochs.

`EarlyStopping` and `ModelCheckpoint` are both passed to the `Trainer` as
callbacks:

```python
ckpt_cb = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1,
                          dirpath=ckpt_dir, filename=ckpt_name)
early_cb = EarlyStopping(monitor="val_loss", mode="min",
                         patience=config.early_stop_patience)
trainer = pl.Trainer(max_epochs=config.epochs, ..., callbacks=[ckpt_cb, early_cb], ...)
```

Every run — regardless of how long it trained — ends with
`trainer.test(module, data_bundle.test_loader, ckpt_path="best")`, which
reloads the best-`val_loss` checkpoint saved by `ModelCheckpoint` before
scoring (see "Checkpointing" below). There is no run outcome where the test
pass is skipped or the model is excluded from the comparison table on
timing grounds.

## Runtime pre-test

Because there's no more automatic gate to cap a slow model's time budget,
`tslib/benchmark/pretest.py` gives you a way to check projected runtimes
*before* committing to a full sweep:

```bash
python3.12 -m tslib.benchmark.pretest
```

For each (track, seq_len, pred_len, fold, strategy) cell in the benchmark
sweep, it builds every model in that cell, times a handful of real training
steps (`--probe-batches`, default `3`) to get a median seconds/batch, and
projects a runtime as `per_epoch_time × --nominal-epochs` (default `50`) —
a yardstick for comparing models, not a prediction of the actual
early-stopped duration. Any model whose projection exceeds
`--threshold-hours` (default `12.0`) triggers an interactive prompt
(`... ~<hours>h projected — proceed? [y/N]`); models at or under the
threshold are auto-approved. Once every cell has been probed, the approved
(cell, model) entries are written to `runs/bench/manifest.json`.

`tslib/benchmark/driver.py` then consumes that file via `--manifest`:

```bash
python3.12 -m tslib.benchmark.driver --manifest runs/bench/manifest.json
```

so a full sweep only ever launches the models a human has looked at and
approved (or that were cheap enough not to need review).

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

Once `trainer.fit(...)` completes (whether it ran to `--epochs` or, more
typically, stopped early via `EarlyStopping`), the test pass reloads that
best checkpoint rather than whatever weights happen to be in memory at the
end of training:

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
