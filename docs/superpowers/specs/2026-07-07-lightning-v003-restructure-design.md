# v003 — PyTorch Lightning Restructure + Train-Time Gate — Design

**Date:** 2026-07-07

## Goal

Replace v002's hand-rolled training loop (`pl_model` in `tslib/exp/lightning_model.py`
— named "lightning" but actually a pure-PyTorch loop) with a genuine
**`pytorch_lightning`** implementation, so humans can customize behavior through the
standard Lightning hooks. Add a training-time **pre-flight gate** that probes a few
training batches, estimates total training time, and — when the estimate exceeds a
configured limit — skips / aborts / proceeds per policy and records the decision so a
human can later choose to re-include the model. Ship as a new working directory
`sw-framework-v003` that carries over v002's git history and remote, and update the
docs alongside.

**Motivation (observed):** In the benchmark pilot, `timesnet` took ~14.5 h to train 3
epochs on the full dataset, and the recursive LSTM rollout eval took ~1 h. Full-scale
benchmarking is infeasible without a way to catch pathologically slow models before
committing to a full run.

## Environment (verified)

- `pytorch_lightning` **2.6.5** installed in `python3.12`; `torch` 2.4.1+cu124.
- `sw-framework-v002` is a git repo on `main`, remote
  `git@github.com:merchen911/GOES-time-series.git`. `sw-framework-v001` is a separate
  legacy repo (`master`, no remote).
- `DataBundle` already exposes torch `DataLoader`s (`train_loader`/`val_loader`/
  `test_loader`) plus `input_size`, `target_indices`, `target_cols` — directly usable by
  a Lightning `Trainer`.
- GPU fixed to device 1 for runs via `CUDA_VISIBLE_DEVICES=1` (unchanged).

## Architecture (after restructure)

```
main.py → run_experiment(config)
  DataModule.setup() → DataBundle(train/val/test DataLoader, input_size,
                                  target_indices, target_cols)
  for model_name in config.models:
    run_strategy(strategy, model_name, bundle, config, ckpt_path)
      ├ direct | recursive:
      │    model  = build_model(model_name, config, input_size,
      │                         target_indices, strategy)      # adapters reused, unchanged
      │    module = ForecastModule(model, config, metric_ctx, strategy)   # real LightningModule
      │    trainer = pl.Trainer(max_epochs=config.epochs,
      │                 accelerator="gpu" if cuda else "cpu", devices=1,
      │                 callbacks=[ModelCheckpoint(monitor="val_loss", save_top_k=1),
      │                            TimingGateCallback(config)],
      │                 logger=CSVLogger(...), enable_progress_bar=…)
      │    trainer.fit(module, train_loader, val_loader)
      │    if module._gate_skipped:   → return skipped TrainResult (no test)
      │    else trainer.test(module, test_loader, ckpt_path="best")  → module.test_metrics
      │    → TrainResult(model_name, best_val_loss, metrics, ckpt_path,
      │                  strategy, skipped, est_train_hours)
      └ statistic: StatisticalRunner  (NOT Lightning — statsmodels per-window; unchanged)
  build_comparison(results) + SKIPPED summary to stdout → comparison.csv
```

The forecast-strategy dispatch, event/regression metrics, comparison table, and
`merge_comparisons` from v002 are preserved. Only the neural train/eval mechanism moves
onto Lightning.

## Components

### A. `ForecastModule(pl.LightningModule)` — `tslib/exp/lightning_model.py` (full replace)

- `__init__(self, model, config, metric_ctx, strategy)` — stores the wrapped
  `nn.Module` (`StandardForecastAdapter`/`RecursiveForecastAdapter`), the loss
  (`build_loss(config)`), config, `metric_ctx`, and `strategy`. Initializes CPU-side
  buffers (`self._test_pred`, `self._test_true`) and `self._gate_skipped = False`.
- `forward(x)` → `self.model(x)`. The adapter's own `training`/`eval` branch handles
  recursive 1-step (train) vs full rollout (eval); Lightning toggles the mode, so no
  extra logic here.
- `training_step(batch, idx)` — `x, y = batch`; `pred = self(x)`;
  `loss = self.criterion(pred, y[:, :pred.shape[1], :])` (the `:pred.shape[1]`
  alignment is a no-op for direct, and aligns `y` to the single step for recursive
  1-step training); `self.log("train_loss", loss)`; return `loss`.
- `validation_step(batch, idx)` — eval-mode forward (recursive rolls out to
  `(B, pred_len, T)`); `val_loss = self.criterion(pred, y[:, :pred.shape[1], :])`;
  `self.log("val_loss", val_loss, prog_bar=True)`. `ModelCheckpoint` monitors
  `val_loss`.
- `test_step(batch, idx)` — append `pred.cpu().numpy()` and `np.asarray(y)` to the
  buffers (matches v002 `evaluate`).
- `on_test_epoch_end` — concat buffers, `self.test_metrics = run_metrics(pred, true,
  self.metric_ctx, self.config.metrics)`, clear buffers.
- `configure_optimizers` — `torch.optim.Adam(self.model.parameters(), lr=config.lr,
  weight_decay=config.weight_decay)`.

### B. `TimingGateCallback(pl.Callback)` + `estimate_train_hours` — `tslib/exp/callbacks.py` (new)

- Pure helper: `estimate_train_hours(sec_per_batch, n_batches, epochs) -> float`
  = `sec_per_batch * n_batches * epochs / 3600.0`. Unit-tested directly.
- `on_train_batch_start(...)` records `t0` (after `torch.cuda.synchronize()` when on
  CUDA). `on_train_batch_end(...)` records `dt` (sync then `perf_counter` delta) into a
  list; once `len(durations) >= config.probe_batches`, compute
  `sec_per_batch = median(durations)` and
  `est = estimate_train_hours(sec_per_batch, trainer.num_training_batches,
  trainer.max_epochs)`.
  - Log one line: `gate: model=<name> s/batch=<x> est_train≈<y>h (limit <max_train_hours>h)`.
  - If `est > config.max_train_hours`:
    - `on_slow == "skip"`   → set `pl_module._gate_skipped = True` and
      `pl_module._est_train_hours = est`; `trainer.should_stop = True`.
    - `on_slow == "abort"`  → `raise RuntimeError(...)` with the estimate.
    - `on_slow == "proceed"`→ log a warning, disable further probing, continue.
  - Otherwise store `pl_module._est_train_hours = est` and stop probing.
- Timing captures the full batch (forward + backward + optimizer step, which occur
  between the start/end hooks). `median` over `probe_batches` absorbs the first-batch
  CUDA/cuDNN warmup spike.

### C. Config — `tslib/configs/config.py`

New flags in `exp_parser`:

| flag | type | default |
|---|---|---|
| `--max_train_hours` | float | `6.0` |
| `--on_slow` | choice `{skip,abort,proceed}` | `skip` |
| `--probe_batches` | int | `3` |

`config_postprocess` validation: `max_train_hours > 0`, `on_slow` in the choice set
(argparse enforces), `probe_batches >= 1`. Existing forecast-strategy flags and
validation are unchanged.

### D. Dispatch / comparison — `tslib/exp/strategy.py`, `tslib/exp/exp.py`

- `run_strategy` (direct/recursive branch) rewritten to build `ForecastModule` +
  `pl.Trainer` with the two callbacks, `fit`, gate-check, then `test`. `statistic`
  branch unchanged.
- `TrainResult` gains `skipped: bool = False` and `est_train_hours: float = nan`
  (appended as last fields to preserve positional construction in existing tests).
- `best_val_loss` comes from the `ModelCheckpoint.best_model_score` (fallback `nan`).
- `build_comparison` adds `skipped` / `est_train_hours` columns; skipped rows carry
  `NaN` metrics and sort last. `run_experiment` prints a `SKIPPED (too slow): <model>
  ~<est>h — re-include with --on_slow proceed` summary. `merge_comparisons` unchanged.

### E. Checkpointing

v002's manual best-val `torch.save` is replaced by
`ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, dirpath=<ckpt dir>,
filename=<model_name>)`. `trainer.test(ckpt_path="best")` loads the best checkpoint for
the test pass, matching v002 semantics.

## v003 creation & git

1. `cp -a sw-framework-v002 sw-framework-v003` (including `.git`); delete `runs/`,
   `__pycache__/`, `*.egg-info/`, and other git-ignored scratch from the copy.
2. In `sw-framework-v003`, create branch `feat/lightning-restructure`; implement;
   get tests green; merge to `main`; push (continues the GOES-time-series remote).
3. The `sw-framework-v002` directory is frozen as the v002 snapshot (already pushed).

## Error handling

- Invalid config rejected up front in `config_postprocess` (existing pattern).
- Gate `abort` raises `RuntimeError` with the estimate; the run stops.
- Gate `skip` halts training cleanly via `trainer.should_stop`, skips the test pass,
  and returns a `skipped` `TrainResult` so downstream code and the comparison table
  stay consistent.
- `statistic` per-window fit failures still fall back to persistence (unchanged).

## Testing (unittest, no pytest)

- **ForecastModule:** synthetic tiny dataset + `pl.Trainer(fast_dev_run=True,
  accelerator="cpu", logger=False)` runs training/validation/test steps once; device-
  aware toy model (places tensors on the parameter device, per prior lesson);
  `test_metrics` populated after `test`.
- **Gate:** `estimate_train_hours` value assertions; policy behavior driven by setting
  `--max_train_hours` extremely low so any real per-batch time exceeds it → assert
  `skip` sets `_gate_skipped`/`should_stop`, `abort` raises, `proceed` continues (no
  need to fake timing).
- **Dispatch/comparison:** `run_strategy` via a real (tiny) `Trainer` returns a
  `TrainResult`; `build_comparison` includes a skipped row; `merge_comparisons`
  unchanged behavior.
- Existing `test_config`, `test_metrics`, `test_recursive`, `test_statistical` remain
  green (adapters, metrics, statistical registry untouched).

## Docs to update

- `README.md` — framework overview now Lightning-based; how to customize via
  `ForecastModule` hooks.
- `pyproject.toml` / `requirements.txt` — add `pytorch_lightning` dependency; bump
  version to v003.
- `tslib/exp/README` — document `ForecastModule` and `TimingGateCallback`.
- `docs/benchmark-conditions.md` — reflect any run-mechanics changes (gate flags).
- New `docs/` entry — Lightning migration + train-time gate rationale and usage.

## Out of scope (YAGNI)

- Eval/rollout time gating (training-only probe, per decision). Recursive eval cost is
  addressed separately via test-window subsampling.
- Lightning-izing the `statistic` strategy (no gradient training — left as-is).
- LR schedulers, AMP/mixed precision, multi-GPU, EarlyStopping — plain `Trainer` only
  this round.
- Auto batch-size / model tuning.
