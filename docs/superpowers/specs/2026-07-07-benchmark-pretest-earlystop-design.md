# Benchmark Pre-test + Early Stopping (replace inline gate) — Design

**Date:** 2026-07-07

## Goal

Replace the inline auto-skip training gate (`TimingGateCallback`, merged in v003)
with two things the user wants instead:

1. A **separate, interactive pre-test tool** that estimates each model's runtime
   before the benchmark and, for models projected to take too long, asks the user
   to confirm (`진짜 진행?`) rather than silently skipping. Its output is an
   **approved manifest** the benchmark driver consumes.
2. **Long training with early stopping** in the actual runs: `--epochs` defaults
   very high (10000) and a Lightning `EarlyStopping` callback halts on `val_loss`
   plateau, so models train as long as useful and stop themselves.

## Motivation

The inline gate estimated full-training time from the first few batches and
auto-skipped over a threshold. The user prefers: don't silently skip — probe
separately, double-check the slow ones interactively, and let the real runs go
long with early stopping to decide when to stop.

## Key reconciliation (estimate basis)

With `max_epochs = 10000`, "full training time = sec/batch × steps × 10000" is
meaningless (astronomically large — everything would exceed any threshold). So
the pre-test projects time as **`per_epoch_time × nominal_epochs`** (nominal
default **50**), i.e. "if this trained 50 epochs, it would take X h." Actual
training still uses `max_epochs=10000 + EarlyStopping`; `nominal_epochs` is only
the yardstick for the "too long" double-check. It is a pre-test parameter.

## Components

### 1. Remove the inline gate

**Files:** `tslib/exp/strategy.py`, `tslib/configs/config.py`,
`tslib/exp/callbacks.py`, `tslib/exp/test_callbacks.py`.

- `_run_neural`: remove `TimingGateCallback` from the trainer callbacks and the
  `if module._gate_skipped:` early-return branch.
- Delete `TimingGateCallback` (class) and `tslib/exp/test_callbacks.py`. Move the
  pure helper `estimate_train_hours(sec_per_batch, n_batches, epochs)` to
  `tslib/benchmark/pretest.py` (reused there). `callbacks.py` is deleted once
  empty.
- `config.py`: remove `--max_train_hours` and `--on_slow`. Keep `--probe_batches`
  (used by the pre-test). Remove their `config_postprocess` validation
  (`max_train_hours`), keep `probe_batches >= 1`.
- `TrainResult.skipped` / `est_train_hours` fields stay (the driver marks
  manifest-rejected cells as skipped in the master table). `ForecastModule`'s
  `_gate_skipped` / `_est_train_hours` attrs are removed (no longer set by any
  callback).

### 2. Early stopping + long epochs

**Files:** `tslib/exp/strategy.py`, `tslib/configs/config.py`.

- `_run_neural` trainer callbacks become `[ckpt_cb, early_stop_cb]` where
  `early_stop_cb = EarlyStopping(monitor="val_loss", mode="min",
  patience=config.early_stop_patience)`. `ModelCheckpoint(monitor="val_loss")`
  stays. `trainer = pl.Trainer(max_epochs=config.epochs, ...)`.
- `config.py`: `--epochs` default changes to **10000**; add
  `--early_stop_patience` (int, default **10**).
- **Steps-per-epoch floor (200):** the full benchmark data yields ~13,500
  train steps/epoch, so the floor is already met. `_run_neural` logs a warning
  if `len(train_loader) < 200` (guard only; no behavior change for benchmark
  data).

### 3. Pre-test tool (new, interactive)

**File:** `tslib/benchmark/pretest.py`.

- Shares the driver's cell enumeration (`enumerate_cells`) and track defs.
- CLI mirrors the driver: `--tracks --seq-lens --pred-lens --folds --strategies
  --probe-batches (3) --nominal-epochs (50) --threshold-hours (12.0)
  --out (runs/bench/manifest.json)`.
- For each cell: `DataModule(config).setup()` once (reused across the cell's
  models) → `steps_per_epoch = len(train_loader)`.
- For each model in the cell: build the model, run `probe_batches` real training
  steps (forward + backward + optimizer step) timing each with CUDA sync,
  `sec_per_batch = median(durations)`,
  `projected_hours = estimate_train_hours(sec_per_batch, steps_per_epoch,
  nominal_epochs)`.
  - `projected_hours <= threshold_hours` → auto-approve.
  - else → interactive prompt: `[<cell>] <model> ~<projected>h (@ <nominal>
    epochs) — proceed? [y/N]`; approve iff the answer is `y`/`yes`.
- Write the manifest: JSON list of approved entries
  `{track, seq_len, pred_len, fold, strategy, model, projected_hours}` plus the
  run parameters. Print a summary (approved / rejected counts).
- Decision logic is a pure function `decide(projected_hours, threshold_hours,
  prompt_fn) -> bool` so tests inject `sec_per_batch`/answers without a GPU.

### 4. Driver consumes the manifest

**File:** `tslib/benchmark/driver.py`.

- New `--manifest <path>`: load approved `(cell, model)` entries. For each cell,
  build `--models` from the approved models for that cell; a cell with no
  approved models is skipped (logged). Without `--manifest`, behavior is
  unchanged (runs the full enumerated model set) — the manifest is an optional
  filter.
- `FIXED`: drop the gate flags (`--max_train_hours`, `--on_slow`); add
  `--epochs 10000` and `--early_stop_patience 10`. `--probe_batches` is not
  passed to `main.py` (it is a pre-test-only concern).
- `--epochs` override flag on the driver still works (defaults to 10000).

### 5. Documentation

**Files:** `docs/lightning-migration.md`, `docs/benchmark-conditions.md`.

- `lightning-migration.md`: replace the "train-time gate" section with
  "early stopping" (long epochs + `EarlyStopping`) and point to the pre-test for
  runtime double-checks.
- `benchmark-conditions.md`: remove the gate-flags subsection; document the
  pre-test → manifest → driver workflow, `--epochs 10000` + `--early_stop_patience
  10`, and the `nominal_epochs`/`threshold_hours` double-check.

## Workflow

```bash
# 1) interactive pre-test → approved manifest
python3.12 -m tslib.benchmark.pretest --folds 0 --strategies direct
# 2) run only the approved cells/models (background, GPU 1)
CUDA_VISIBLE_DEVICES=1 python3.12 -m tslib.benchmark.driver \
    --manifest runs/bench/manifest.json
```

## Error handling

- Pre-test model build / probe failure on a cell: log it, mark that model
  rejected (not approved), continue — one bad model doesn't abort the pre-test.
- Driver: a manifest with no approved models for a cell → skip cell with a log
  line. Missing manifest file → clear error.
- Early stopping requires `val_loss` to be logged (it is, in
  `ForecastModule.validation_step`).

## Testing (unittest, no pytest)

- **Gate removal:** `_run_neural` builds a trainer whose callbacks are
  `ModelCheckpoint` + `EarlyStopping` and no `TimingGateCallback`; existing
  neural-path strategy test still passes. `test_callbacks.py` removed.
- **Early stop wiring:** a test asserts the trainer’s callbacks include an
  `EarlyStopping` monitoring `val_loss` (inspect constructed callbacks, or a
  small integration run with `patience` and a flat loss).
- **Pre-test:** `estimate_train_hours` value test; `decide(projected, threshold,
  prompt_fn)` — under threshold auto-approves without prompting; over threshold
  calls `prompt_fn` and honors y/n; manifest write/read round-trips approved
  entries.
- **Driver manifest:** with a manifest approving a subset, `build_cmd`/cell
  planning includes only approved models and skips cells with none; `FIXED` no
  longer contains gate flags and contains `--epochs`/`--early_stop_patience`.
- Full suite stays green.

## Out of scope (YAGNI)

- Inline auto-skip gate (removed).
- folds 1-4, recursive strategy, `pred_len 864` (later).
- Training-set subsampling.
- Auto-tuning `nominal_epochs` from observed convergence.
