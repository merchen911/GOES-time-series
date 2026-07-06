# Forecast Strategy — Design

**Date:** 2026-07-06

## Goal

Let a single config flag `--forecast_strategy` select how a run produces its
forecast horizon, across three families of methods, all evaluated on the same
term split and all landing in one shared comparison table:

- `direct` (default) — neural many-to-many; model emits `(B, pred_len, T)` in
  one forward. **Current behavior, unchanged.**
- `recursive` — neural; trained one-step-ahead, rolled out `pred_len` times at
  inference.
- `statistic` — classical statistical models (statsmodels ARIMA / AR / Theta),
  univariate, fit per test window on its own history, forecast `pred_len`. No
  gradient training, no GPU.

## Architecture

`run_experiment` today does, per model:
`build_model → pl_model.fit_and_test → TrainResult → build_comparison`.

We insert **one dispatch point** that routes each model to the right runner.
All three runners return the **same `TrainResult`**, so metrics, comparison
table, and persistence are shared and mostly unchanged.

```
run_experiment(config)
  └─ for model_name in config.models:
       run_strategy(config.forecast_strategy, model_name, data_bundle, config, ckpt_path)
         ├─ "direct"     → pl_model.fit_and_test              [neural MODEL_REGISTRY]
         ├─ "recursive"  → pl_model.fit_and_test              [neural MODEL_REGISTRY]
         │                 (model wrapped in RecursiveForecastAdapter)
         └─ "statistic"  → StatisticalRunner.fit_and_test     [STAT_REGISTRY, no torch]
       → TrainResult(model_name, best_val_loss, metrics, ckpt_path)
  └─ build_comparison(results, sort_metric)   # + "strategy" column
```

The strategy is **run-level**: one `--forecast_strategy` per run. Comparing
families (classical vs recursive vs direct) is done by running each strategy on
the **same split** and merging their comparison rows at the end
(`merge_comparisons`). Mixing strategies within one run (per-model strategy) is
explicitly out of scope.

## Component A — strategy spine

**Files:** `tslib/configs/config.py`, `tslib/exp/strategy.py` (new),
`tslib/exp/exp.py`.

- `config.py`
  - `exp_parser`: `--forecast_strategy`, choices `{direct, recursive, statistic}`,
    default `direct`.
  - `config_postprocess` validation:
    - `statistic` ⇒ every name in `config.models` must be in `STAT_REGISTRY`
      (raise `ValueError` listing registered statistical models otherwise).
    - `direct` / `recursive` ⇒ every name must be in the neural `MODEL_REGISTRY`.
    - `recursive` ⇒ the resolved target set must equal the full input-channel
      set (all channels forecast). Raise `ValueError` otherwise. (The
      channel/target resolution already lives in the data layer; validation
      compares target count to channel count.)
- `strategy.py`
  - `run_strategy(strategy, model_name, data_bundle, config, ckpt_path) -> TrainResult`
    — pure dispatch to the three runners. No business logic beyond routing.
- `exp.py`
  - `run_experiment` calls `run_strategy(...)` instead of building `pl_model`
    directly.
  - `build_comparison(results, sort_metric)` gains a `strategy` column
    (read from each `TrainResult`; add `strategy: str` to `TrainResult`).
  - `merge_comparisons(paths_or_frames) -> DataFrame` — concatenate comparison
    frames from separate runs (same split) into one table keyed by
    `(strategy, model)`, sorted by `sort_metric`. Tolerates `NaN`
    `best_val_loss` (statistic runs have no validation loss).

## Component B — recursive (neural, no new dependency)

**Files:** `tslib/model/__init__.py`, `tslib/exp/lightning_model.py`.

- `RecursiveForecastAdapter(nn.Module)` in `tslib/model/__init__.py`
  - Base model is built with `pred_len = 1` (its head emits a single step). The
    adapter stores the **real** `pred_len` for rollout.
  - Reuses the existing single-step machinery (build zero time-marks + decoder
    input, call base model) — factored out of `StandardForecastAdapter` as a
    shared `_one_step(x) -> (B, 1, C)` helper.
  - `forward(x)`:
    - **train mode** (`self.training`): return `_one_step(x)` → `(B, 1, C)`.
    - **eval mode**: rollout —
      ```
      window = x
      outs = []
      for _ in range(pred_len):
          yhat = _one_step(window)              # (B, 1, C)
          outs.append(yhat)
          window = cat([window[:, 1:], yhat], 1)
      pred = cat(outs, 1)                        # (B, pred_len, C)
      return pred[..., target_indices]           # targets == all channels
      ```
    Since `recursive` requires targets == all channels, `yhat` fills the whole
    next frame; no exogenous-channel imputation is needed.
- `build_model(model_name, config, input_size, target_indices, strategy="direct")`
  - `strategy == "recursive"` ⇒ construct the base with a `pred_len = 1` config
    (do not mutate the caller's config permanently) and wrap in
    `RecursiveForecastAdapter`.
  - `strategy == "direct"` ⇒ `StandardForecastAdapter` (unchanged).
- `pl_model._run_epoch`: one line — `loss = criterion(pred, y[:, :pred.shape[1], :])`.
  - `direct`: `pred.shape[1] == pred_len == y length` → no-op, fully backward
    compatible.
  - `recursive` train: `pred.shape[1] == 1` → aligns `y` to its first step.
  - `evaluate` is unchanged: the recursive adapter rolls out in eval mode, so it
    returns `(B, pred_len, T)` — the same interface `evaluate` already expects.

**Why this is safe:** the only recursive-specific code is the adapter's
train/eval branch and the one-line `y` alignment. Optimizer, epoch loop,
checkpointing, losses, metrics, and comparison are shared and unchanged.
Training one-step while evaluating full-rollout is the accepted exposure-bias
trade-off.

## Component C — statistical subsystem (new dependency)

**Files:** `pyproject.toml`, `tslib/model/statistical.py` (new),
`tslib/exp/strategy.py`.

- Dependency: add **statsmodels** to `pyproject.toml`. Not currently installed
  in the target env — Phase 2 begins with `python3.12 -m pip install statsmodels`.
- `tslib/model/statistical.py`
  - `STAT_REGISTRY: Dict[str, Callable]` mapping name → a builder producing an
    object with:
    - `fit(history: np.ndarray) -> self` (history is 1-D, stored/log10 space)
    - `forecast(steps: int) -> np.ndarray` (length `steps`, 1-D)
  - `register_stat(name)` decorator (mirrors `register_model`).
  - Builtins wrapping statsmodels:
    - `arima` — `ARIMA(history, order=config.arima_order)`, default `(1,0,0)`.
    - `ar` — `AutoReg(history, lags=config.ar_lags)`, default `1`.
    - `theta` — `ThetaModel(history)`.
  - Config: `--arima_order` (three ints, default `1 0 0`), `--ar_lags`
    (int, default `1`). Added in `exp_parser`.
- `StatisticalRunner` in `strategy.py`
  - `fit_and_test(data_bundle, model_name, ckpt_path) -> TrainResult`
  - Iterates `data_bundle.test_loader`; for each univariate window takes the
    input history `x` (`(B, seq_len, 1)`), fits the statistical model per series
    in the batch (in stored/log10 space, same as neural), forecasts `pred_len`,
    collects `pred (N, pred_len, 1)` and `true (N, pred_len, 1)`.
  - Runs the **same** metrics via `MetricContext` (thresholds, transform,
    target_cols) — identical to `pl_model.evaluate`.
  - No training, no validation: `best_val_loss = float("nan")`, no checkpoint
    written.
  - Univariate only: if more than one target/channel is configured under
    `statistic`, raise `ValueError` (multivariate classical models are VAR-class,
    out of scope).

## Data flow (shared)

All three paths consume the same `DataModule`/`DataBundle`: windows are already
gridded to the 5-min cadence, term-split, and transformed (log10). Statistical
fits and neural forecasts both operate in the stored (log10) space, and all
metrics run through the same `MetricContext`, so physical-unit thresholds and
event scores are consistent across strategies.

## Error handling

- Invalid strategy/model combinations are rejected up front in
  `config_postprocess` with explicit `ValueError` messages (wrong registry,
  recursive with partial targets, statistic with multivariate targets).
- `merge_comparisons` and `build_comparison` tolerate `NaN` `best_val_loss`.
- statsmodels fit failures on a degenerate window (e.g. constant series) are
  caught per window and fall back to persistence (`forecast = last value`),
  logged once; the run does not crash.

## Testing

- **A (dispatch/validation):** `run_strategy` routes each name to the correct
  runner; `config_postprocess` rejects recursive-with-partial-targets,
  statistic-with-neural-name, and statistic-with-multivariate; `merge_comparisons`
  merges two frames with a `strategy` column and `NaN` val loss.
- **B (recursive):** adapter train forward is `(B, 1, C)`; eval rollout is
  `(B, pred_len, T)`; rollout equals a hand-written 3-step loop on a toy
  1-step model; device-aware (GPU fixtures — toy models place tensors on the
  parameter device, per prior lesson).
- **C (statistic):** each builtin fits and forecasts the correct length on a
  synthetic series; runner emits a comparison row with `NaN` val loss and
  regression metrics matching a direct statsmodels call; degenerate-window
  fallback returns persistence without raising.

## Phasing

One design document (this file); implementation and merge in **two phases**:

- **Phase 1 — A + B** (neural, no new dependency): strategy spine, config flag +
  validation, `RecursiveForecastAdapter`, `_run_epoch` alignment, comparison
  `strategy` column, `merge_comparisons`. Independently mergeable.
- **Phase 2 — C** (statsmodels): dependency, `statistical.py`,
  `StatisticalRunner`, statistic validation and config. Merged after Phase 1.

## Out of scope (YAGNI)

- Mixing strategies within a single run (per-model strategy).
- BPTT / scheduled-sampling recursive training (we train one-step only).
- Multivariate classical models (VAR-class).
- Exogenous-channel imputation for recursive (targets == all channels required).
- Automatic order selection (auto-ARIMA); orders come from config with defaults.
```
