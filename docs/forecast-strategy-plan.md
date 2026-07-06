# Forecast Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a run-level `--forecast_strategy` flag that routes each model to one of three forecast families — `direct` (current), `recursive` (1-step train / rollout eval), `statistic` (statsmodels ARIMA/AR/Theta) — all sharing one comparison table.

**Architecture:** A `run_strategy` dispatcher replaces the direct `build_model → pl_model` call in `run_experiment`. `direct`/`recursive` reuse the neural `pl_model` training loop; the only recursive-specific logic is a new `RecursiveForecastAdapter` (train→1-step, eval→rollout) plus a one-line target alignment in `_run_epoch`. `statistic` uses a separate `StatisticalRunner` that fits per test window with no torch. All runners return the same `TrainResult`.

**Tech Stack:** Python 3.12, PyTorch, NumPy, pandas, statsmodels (Phase 2 only), unittest.

## Global Constraints

- Tests use **unittest only** (no pytest). Full suite: `python -m unittest discover -t . -s tslib -p 'test_*.py'`. Single module: `python -m unittest tslib.<pkg>.test_<name> -v`.
- Interpreter is **python3.12** (has torch/numpy/pandas; statsmodels installed in Phase 2).
- **Backward compatibility is mandatory.** The `direct` path must be behaviorally unchanged. New parameters (`build_model(..., strategy=)`, `TrainResult.strategy`) get defaults so existing positional callers keep working.
- **Toy test models must be device-aware:** create tensors on the parameter/input device (`device=self.p.device` or derive from `x.device`), never bare `torch.zeros(...)`.
- All metrics are computed in the **stored (log10) space**; `MetricContext(thresholds, transform, target_cols)` converts physical thresholds.
- `recursive` requires forecasting **all** input channels (targets == channels); reject otherwise.
- `statistic` is **univariate** (exactly one target) and its `--models` names resolve in `STAT_REGISTRY`, not the neural `MODEL_REGISTRY`.
- **Phasing:** Phase 1 = Tasks 1–3 (neural, no new dependency), independently mergeable. Phase 2 = Tasks 4–5 (statsmodels). After Task 3, `--forecast_strategy statistic` must fail with a clear message, not a crash.

---

### Task 1: Config flag + recursive validation

**Files:**
- Modify: `tslib/configs/config.py` (add flag in `exp_parser`, validate in `config_postprocess`)
- Test: `tslib/configs/test_config.py`

**Interfaces:**
- Consumes: existing `exp_parser()`, `config_postprocess(config)`, and `tslib.model.registry.MODEL_REGISTRY` (dict keyed by neural model name).
- Produces: `config.forecast_strategy ∈ {"direct","recursive","statistic"}` (default `"direct"`); `config_postprocess` raises `ValueError` for `recursive` when target count ≠ channel count, and for `direct`/`recursive` when a model name is not in `MODEL_REGISTRY`.

- [ ] **Step 1: Write the failing tests**

Add to `tslib/configs/test_config.py`:

```python
class TestForecastStrategy(unittest.TestCase):
    def _parse(self, *extra):
        argv = ["--data_path", "x.parquet", "--target_col", "p_gt10", *extra]
        return exp_parser().parse_args(argv)

    def test_default_is_direct(self):
        self.assertEqual(self._parse().forecast_strategy, "direct")

    def test_choice_parses(self):
        self.assertEqual(
            self._parse("--forecast_strategy", "recursive").forecast_strategy,
            "recursive")

    def test_recursive_univariate_ok(self):
        c = self._parse("--forecast_strategy", "recursive")  # 1 channel, 1 target
        self.assertIs(config_postprocess(c), c)

    def test_recursive_all_channels_ok(self):
        c = self._parse("--forecast_strategy", "recursive",
                        "--channels", "a.parquet:x", "b.parquet:y",
                        "--target_cols", "x", "y")
        self.assertIs(config_postprocess(c), c)

    def test_recursive_partial_targets_rejected(self):
        c = self._parse("--forecast_strategy", "recursive",
                        "--channels", "a.parquet:x", "b.parquet:y",
                        "--target_cols", "x")
        with self.assertRaises(ValueError):
            config_postprocess(c)

    def test_unknown_neural_model_rejected(self):
        c = self._parse("--models", "not_a_model")
        with self.assertRaises(ValueError):
            config_postprocess(c)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m unittest tslib.configs.test_config -v`
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'forecast_strategy'` (and the reject tests do not raise).

- [ ] **Step 3: Add the flag**

In `tslib/configs/config.py`, inside `exp_parser()`, in the "model comparison" block right after the `--models` line:

```python
    parser.add_argument("--forecast_strategy", type=str, default="direct",
                        choices=["direct", "recursive", "statistic"],
                        help="how the forecast horizon is produced")
```

- [ ] **Step 4: Add the validation**

In `config_postprocess(config)`, immediately before `return config`:

```python
    strategy = getattr(config, "forecast_strategy", "direct")
    if strategy in ("direct", "recursive"):
        from tslib.model.registry import MODEL_REGISTRY
        for m in config.models:
            if m not in MODEL_REGISTRY:
                raise ValueError(
                    f"unknown model '{m}' for strategy '{strategy}'; "
                    f"registered: {sorted(MODEL_REGISTRY)}")
    if strategy == "recursive":
        n_ch = len(config.channels) if getattr(config, "channels", None) else 1
        n_tgt = len(config.target_cols) if getattr(config, "target_cols", None) else 1
        if n_tgt != n_ch:
            raise ValueError(
                "recursive strategy must forecast all input channels: "
                f"target count {n_tgt} != channel count {n_ch}")
```

- [ ] **Step 5: Run to verify they pass**

Run: `python -m unittest tslib.configs.test_config -v`
Expected: PASS (all, including the pre-existing config tests).

- [ ] **Step 6: Commit**

```bash
git add tslib/configs/config.py tslib/configs/test_config.py
git commit -m "feat(config): --forecast_strategy flag + recursive target validation"
```

---

### Task 2: RecursiveForecastAdapter + build_model strategy + _run_epoch alignment

**Files:**
- Modify: `tslib/model/__init__.py` (extract `_call_base` helper, add `RecursiveForecastAdapter`, add `strategy=` to `build_model`)
- Modify: `tslib/exp/lightning_model.py` (`_run_epoch` target alignment)
- Test: `tslib/model/test_recursive.py` (new)
- Test: `tslib/exp/test_lightning_model.py` (add 1-step alignment test)

**Interfaces:**
- Consumes: `tslib.model.registry.MODEL_REGISTRY` (`spec.ctor`, `spec.adapter`); config attrs `seq_len`, `label_len`, `pred_len`, `enc_in/dec_in/c_out` (set inside `build_model`).
- Produces:
  - `_call_base(base_model, config, x, dec_len) -> Tensor` — module-level helper building zero time-marks + zero decoder input of length `label_len + dec_len` and calling the base model.
  - `RecursiveForecastAdapter(base_model, config, target_indices, rollout_len)` — `nn.Module`; `forward(x)` returns `(B, 1, T)` in train mode and `(B, rollout_len, T)` in eval mode, where `T = len(target_indices)`.
  - `build_model(model_name, config, input_size, target_indices, strategy="direct")` — `strategy="recursive"` builds the base with `pred_len=1` and wraps in `RecursiveForecastAdapter`.

- [ ] **Step 1: Write the failing tests (adapter)**

Create `tslib/model/test_recursive.py`:

```python
import unittest
from types import SimpleNamespace

import torch

from tslib.model import build_model, RecursiveForecastAdapter


def _cfg(seq_len, pred_len):
    return SimpleNamespace(task_name="long_term_forecast", label_len=0,
                           seq_len=seq_len, pred_len=pred_len,
                           d_model=16, num_layers=2, dropout=0.1)


class _EchoBase(torch.nn.Module):
    """One-step base that predicts the last input frame (persistence)."""
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return x_enc[:, -1:, :] + self.p  # (B, 1, C), on x_enc.device


class TestRecursiveAdapter(unittest.TestCase):
    def test_train_mode_single_step(self):
        m = build_model("lstm", _cfg(24, 6), input_size=2, target_indices=[0, 1],
                        strategy="recursive")
        m.train()
        out = m(torch.randn(4, 24, 2))
        self.assertEqual(tuple(out.shape), (4, 1, 2))

    def test_eval_mode_full_horizon(self):
        m = build_model("lstm", _cfg(24, 6), input_size=2, target_indices=[0, 1],
                        strategy="recursive")
        m.eval()
        out = m(torch.randn(4, 24, 2))
        self.assertEqual(tuple(out.shape), (4, 6, 2))

    def test_rollout_matches_manual_loop(self):
        cfg = _cfg(5, 3)
        adapter = RecursiveForecastAdapter(_EchoBase(), cfg,
                                           target_indices=[0], rollout_len=3)
        adapter.eval()
        x = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1)  # last frame = 4.0
        out = adapter(x)                       # echo => every step = 4.0
        self.assertEqual(tuple(out.shape), (1, 3, 1))
        self.assertTrue(torch.allclose(out, torch.full((1, 3, 1), 4.0)))
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m unittest tslib.model.test_recursive -v`
Expected: FAIL — `ImportError: cannot import name 'RecursiveForecastAdapter'`.

- [ ] **Step 3: Refactor `StandardForecastAdapter` onto a shared helper and add the recursive adapter**

Replace the body of `tslib/model/__init__.py` with:

```python
from __future__ import annotations

import copy

from torch import nn


def _build_time_mark(batch: int, length: int, device):
    import torch
    return torch.zeros((batch, length, 5), dtype=torch.long, device=device)


def _call_base(base_model, config, x, dec_len: int):
    """Build zero time-marks + zero decoder input (length label_len + dec_len)
    and call a legacy-style 4-arg base model. Returns its raw output."""
    import torch
    b, dev = x.shape[0], x.device
    x_mark_enc = _build_time_mark(b, config.seq_len, dev)
    x_dec = torch.zeros((b, config.label_len + dec_len, x.shape[-1]), device=dev)
    x_mark_dec = _build_time_mark(b, config.label_len + dec_len, dev)
    return base_model(x, x_mark_enc, x_dec, x_mark_dec)


class StandardForecastAdapter(nn.Module):
    """Direct many-to-many: base emits full pred_len; select target channel(s)."""

    def __init__(self, base_model: nn.Module, config, target_indices) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.target_indices = list(target_indices)

    def forward(self, x):
        pred = _call_base(self.base_model, self.config, x, self.config.pred_len)
        if pred.ndim == 3 and pred.shape[-1] > 1:
            pred = pred[..., self.target_indices]
        return pred


class RecursiveForecastAdapter(nn.Module):
    """Train: emit one step. Eval: roll the one-step prediction forward
    rollout_len times, feeding the full predicted frame back into the window."""

    def __init__(self, base_model: nn.Module, config, target_indices,
                 rollout_len: int) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.target_indices = list(target_indices)
        self.rollout_len = int(rollout_len)

    def forward(self, x):
        import torch
        if self.training:
            step = _call_base(self.base_model, self.config, x, 1)  # (B,1,C)
            return step[..., self.target_indices]
        window, outs = x, []
        for _ in range(self.rollout_len):
            step = _call_base(self.base_model, self.config, window, 1)  # (B,1,C)
            outs.append(step)
            window = torch.cat([window[:, 1:, :], step], dim=1)
        pred = torch.cat(outs, dim=1)               # (B, rollout_len, C)
        return pred[..., self.target_indices]        # (B, rollout_len, T)


def build_model(model_name: str, config, input_size: int, target_indices,
                strategy: str = "direct"):
    from tslib.model.registry import MODEL_REGISTRY
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"unknown model '{model_name}'; registered: {sorted(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[model_name]
    config.model = model_name
    config.enc_in = input_size
    config.dec_in = input_size
    config.c_out = input_size
    if spec.adapter != "standard":
        raise ValueError(f"unknown adapter '{spec.adapter}'")
    if strategy == "recursive":
        base_cfg = copy.copy(config)
        base_cfg.pred_len = 1                      # 1-step head
        base_model = spec.ctor(base_cfg)
        return RecursiveForecastAdapter(base_model, config, target_indices,
                                        rollout_len=config.pred_len)
    base_model = spec.ctor(config)
    return StandardForecastAdapter(base_model=base_model, config=config,
                                   target_indices=target_indices)
```

- [ ] **Step 4: Run to verify adapter tests pass**

Run: `python -m unittest tslib.model.test_recursive tslib.model.test_lstm tslib.model.test_registry -v`
Expected: PASS (recursive tests pass; existing lstm/registry tests still pass — direct path unchanged).

- [ ] **Step 5: Write the failing test (training alignment)**

Add to `tslib/exp/test_lightning_model.py`:

```python
class TestOneStepTargetAlignment(unittest.TestCase):
    def test_one_step_pred_against_multistep_target(self):
        # model emits (B,1,1); loader target is (B,pred_len=2,1). Training must
        # align y to the first step and run without a shape error.
        runner = pl_model(_ConstModel(1, 1), _cfg(pred_len=2, epochs=1))
        x = torch.zeros(4, 4, 1)
        y = torch.full((4, 2, 1), 2.0)
        loss = runner._run_epoch([(x, y)], train=True)
        self.assertTrue(np.isfinite(loss))
```

- [ ] **Step 6: Run to verify it fails**

Run: `python -m unittest tslib.exp.test_lightning_model -v`
Expected: FAIL — `RuntimeError` on shape mismatch `(4,1,1)` vs `(4,2,1)` in the loss.

- [ ] **Step 7: Align the target in `_run_epoch`**

In `tslib/exp/lightning_model.py`, `_run_epoch`, change the loss line:

```python
            pred = self.model(x)
            loss = self.criterion(pred, y[:, :pred.shape[1], :])
```

- [ ] **Step 8: Run to verify it passes**

Run: `python -m unittest tslib.exp.test_lightning_model -v`
Expected: PASS (new test passes; `TestPlModelMetrics` still passes — direct `pred.shape[1] == pred_len` makes the slice a no-op).

- [ ] **Step 9: Commit**

```bash
git add tslib/model/__init__.py tslib/model/test_recursive.py \
        tslib/exp/lightning_model.py tslib/exp/test_lightning_model.py
git commit -m "feat(model): recursive forecast adapter + 1-step target alignment"
```

---

### Task 3: Strategy dispatcher + comparison strategy column + merge

**Files:**
- Create: `tslib/exp/strategy.py`
- Modify: `tslib/exp/lightning_model.py` (`TrainResult.strategy`)
- Modify: `tslib/exp/exp.py` (`run_experiment` uses dispatcher; `build_comparison` strategy column; `merge_comparisons`)
- Test: `tslib/exp/test_strategy.py` (new)
- Test: `tslib/exp/test_exp.py` (update column expectation; add merge + NaN tests)

**Interfaces:**
- Consumes: `build_model(..., strategy=)`, `pl_model(model, config).fit_and_test(bundle, model_name, ckpt_path)`, `DataBundle` (`input_size`, `target_indices`).
- Produces:
  - `TrainResult` gains `strategy: str = "direct"` (last field, keeps positional construction valid).
  - `run_strategy(strategy, model_name, data_bundle, config, ckpt_path) -> TrainResult`. `statistic` raises `ValueError("statistic strategy is not available until Phase 2")` (replaced in Task 5).
  - `build_comparison(results, sort_metric) -> DataFrame` with a leading `strategy` column.
  - `merge_comparisons(frames, sort_metric="best_val_loss") -> DataFrame` — concatenates comparison frames (same split), sorts, `NaN` sorts last.

- [ ] **Step 1: Write the failing tests**

Create `tslib/exp/test_strategy.py`:

```python
import unittest
from types import SimpleNamespace

from tslib.exp.strategy import run_strategy


class TestRunStrategyRouting(unittest.TestCase):
    def test_statistic_not_available_phase1(self):
        with self.assertRaises(ValueError):
            run_strategy("statistic", "arima", SimpleNamespace(),
                         SimpleNamespace(), "ckpt.pt")

    def test_unknown_strategy_rejected(self):
        with self.assertRaises(ValueError):
            run_strategy("bogus", "lstm", SimpleNamespace(),
                         SimpleNamespace(), "ckpt.pt")
```

Add to `tslib/exp/test_exp.py`:

```python
import math
import pandas as pd
from tslib.exp.exp import merge_comparisons


class TestStrategyColumn(unittest.TestCase):
    def test_strategy_column_present(self):
        from tslib.exp.exp import build_comparison
        res = [TrainResult("lstm", 0.5, {"mse": 0.3}, "a.pt", strategy="direct")]
        df = build_comparison(res, sort_metric="mse")
        self.assertEqual(set(df.columns), {"strategy", "model", "best_val_loss", "mse"})
        self.assertEqual(df.loc[0, "strategy"], "direct")

    def test_merge_across_strategies_nan_last(self):
        from tslib.exp.exp import build_comparison
        a = build_comparison(
            [TrainResult("lstm", 0.5, {"mse": 0.3}, "a.pt", strategy="recursive")],
            sort_metric="best_val_loss")
        b = build_comparison(
            [TrainResult("arima", float("nan"), {"mse": 0.9}, "", strategy="statistic")],
            sort_metric="best_val_loss")
        merged = merge_comparisons([a, b], sort_metric="best_val_loss")
        self.assertEqual(list(merged["model"]), ["lstm", "arima"])  # NaN last
        self.assertTrue(math.isnan(merged.iloc[-1]["best_val_loss"]))
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m unittest tslib.exp.test_strategy tslib.exp.test_exp -v`
Expected: FAIL — `ModuleNotFoundError: tslib.exp.strategy`; `TrainResult` has no `strategy`; `test_dynamic_columns` fails (missing `strategy` column).

- [ ] **Step 3: Add `strategy` to `TrainResult`**

In `tslib/exp/lightning_model.py`, extend the dataclass (new field last, with default):

```python
@dataclass
class TrainResult:
    model_name: str
    best_val_loss: float
    metrics: Dict[str, float]
    ckpt_path: str
    strategy: str = "direct"
```

- [ ] **Step 4: Create the dispatcher**

Create `tslib/exp/strategy.py`:

```python
from __future__ import annotations

from tslib.model import build_model
from tslib.exp.lightning_model import pl_model, TrainResult


def run_strategy(strategy, model_name, data_bundle, config, ckpt_path) -> TrainResult:
    if strategy in ("direct", "recursive"):
        model = build_model(model_name, config, data_bundle.input_size,
                            data_bundle.target_indices, strategy=strategy)
        result = pl_model(model, config).fit_and_test(
            data_bundle, model_name=model_name, ckpt_path=ckpt_path)
        result.strategy = strategy
        return result
    if strategy == "statistic":
        raise ValueError("statistic strategy is not available until Phase 2")
    raise ValueError(f"unknown forecast_strategy '{strategy}'")
```

- [ ] **Step 5: Wire `run_experiment` and add the comparison helpers**

In `tslib/exp/exp.py`: update the import line `from .lightning_model import pl_model, TrainResult` to also expose the dispatcher, and replace `build_comparison` + the model loop:

```python
from .strategy import run_strategy


def build_comparison(results, sort_metric):
    rows = [{"strategy": r.strategy, "model": r.model_name,
             "best_val_loss": r.best_val_loss, **r.metrics}
            for r in results]
    df = pd.DataFrame(rows)
    key = sort_metric if sort_metric in df.columns else "best_val_loss"
    return df.sort_values(by=key, na_position="last").reset_index(drop=True)


def merge_comparisons(frames, sort_metric="best_val_loss"):
    df = pd.concat(list(frames), ignore_index=True)
    key = sort_metric if sort_metric in df.columns else "best_val_loss"
    return df.sort_values(by=key, na_position="last").reset_index(drop=True)
```

Then inside `run_experiment`, replace the `for model_name in config.models:` body:

```python
    strategy = getattr(config, "forecast_strategy", "direct")
    results = []
    for model_name in config.models:
        ckpt_path = os.path.join(dirs["ckpt"], f"{model_name}.pt")
        results.append(run_strategy(strategy, model_name, data_bundle,
                                    config, ckpt_path))
```

The model loop no longer calls `build_model` or `pl_model` directly (the dispatcher does). Delete the now-unused top-of-file imports `from tslib.model import build_model` and `from .lightning_model import pl_model, TrainResult` (nothing in `exp.py` references them after this change) to keep the module clean.

- [ ] **Step 6: Run to verify they pass**

Run: `python -m unittest tslib.exp.test_strategy tslib.exp.test_exp -v`
Expected: PASS. Note: `test_dynamic_columns` in `test_exp.py` must now expect `strategy` in its column set — update that assertion to `{"strategy", "model", "best_val_loss", "mse", "tss_p_gt10"}`.

- [ ] **Step 7: Run the full suite (Phase 1 gate)**

Run: `python -m unittest discover -t . -s tslib -p 'test_*.py'`
Expected: PASS (all). This is the Phase 1 merge point.

- [ ] **Step 8: Commit**

```bash
git add tslib/exp/strategy.py tslib/exp/exp.py tslib/exp/lightning_model.py \
        tslib/exp/test_strategy.py tslib/exp/test_exp.py
git commit -m "feat(exp): forecast-strategy dispatcher + comparison strategy column + merge"
```

---

### Task 4: Statistical models registry + config (Phase 2)

**Files:**
- Modify: `pyproject.toml` (add `statsmodels` dependency)
- Create: `tslib/model/statistical.py`
- Modify: `tslib/configs/config.py` (`--arima_order`, `--ar_lags`, statistic validation)
- Test: `tslib/model/test_statistical.py` (new)
- Test: `tslib/configs/test_config.py` (statistic validation tests)

**Interfaces:**
- Consumes: config attrs `arima_order` (list of 3 ints), `ar_lags` (int), `pred_len`.
- Produces:
  - `STAT_REGISTRY: Dict[str, type]` and `register_stat(name)` decorator.
  - Each registered class: `__init__(config)`, `fit(history_1d: np.ndarray) -> self`, `forecast(steps: int) -> np.ndarray` (length `steps`, 1-D).
  - Builtins: `arima`, `ar`, `theta`.
  - `config_postprocess` rejects `statistic` with a non-`STAT_REGISTRY` model name or with more than one target.

- [ ] **Step 1: Install the dependency**

Run: `python3.12 -m pip install statsmodels`
Expected: statsmodels installs; `python3.12 -c "import statsmodels; print(statsmodels.__version__)"` prints a version.

- [ ] **Step 2: Write the failing tests (models)**

Create `tslib/model/test_statistical.py`:

```python
import unittest
from types import SimpleNamespace

import numpy as np

from tslib.model.statistical import STAT_REGISTRY


def _cfg():
    return SimpleNamespace(arima_order=[1, 0, 0], ar_lags=2, pred_len=4)


class TestStatisticalModels(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.series = np.cumsum(rng.normal(size=64)) + 10.0

    def test_registry_has_builtins(self):
        self.assertEqual({"arima", "ar", "theta"} & set(STAT_REGISTRY),
                         {"arima", "ar", "theta"})

    def test_each_model_forecasts_correct_length(self):
        for name in ("arima", "ar", "theta"):
            model = STAT_REGISTRY[name](_cfg()).fit(self.series)
            fc = model.forecast(4)
            self.assertEqual(np.asarray(fc).shape, (4,), msg=name)
            self.assertTrue(np.all(np.isfinite(fc)), msg=name)
```

- [ ] **Step 3: Run to verify they fail**

Run: `python -m unittest tslib.model.test_statistical -v`
Expected: FAIL — `ModuleNotFoundError: tslib.model.statistical`.

- [ ] **Step 4: Implement the statistical models**

Create `tslib/model/statistical.py`:

```python
from __future__ import annotations

from typing import Dict

import numpy as np

STAT_REGISTRY: Dict[str, type] = {}


def register_stat(name):
    def deco(cls):
        STAT_REGISTRY[name] = cls
        return cls
    return deco


@register_stat("arima")
class ArimaModel:
    def __init__(self, config):
        self.order = tuple(getattr(config, "arima_order", [1, 0, 0]))

    def fit(self, history):
        from statsmodels.tsa.arima.model import ARIMA
        self._res = ARIMA(np.asarray(history, dtype=float),
                          order=self.order).fit()
        return self

    def forecast(self, steps):
        return np.asarray(self._res.forecast(steps), dtype=float)


@register_stat("ar")
class ArModel:
    def __init__(self, config):
        self.lags = int(getattr(config, "ar_lags", 1))

    def fit(self, history):
        from statsmodels.tsa.ar_model import AutoReg
        self._hist = np.asarray(history, dtype=float)
        self._res = AutoReg(self._hist, lags=self.lags, old_names=False).fit()
        return self

    def forecast(self, steps):
        n = len(self._hist)
        return np.asarray(self._res.predict(start=n, end=n + steps - 1),
                          dtype=float)


@register_stat("theta")
class ThetaModelWrap:
    def __init__(self, config):
        pass

    def fit(self, history):
        from statsmodels.tsa.forecasting.theta import ThetaModel
        self._res = ThetaModel(np.asarray(history, dtype=float),
                               period=1).fit()
        return self

    def forecast(self, steps):
        return np.asarray(self._res.forecast(steps), dtype=float)
```

- [ ] **Step 5: Run to verify model tests pass**

Run: `python -m unittest tslib.model.test_statistical -v`
Expected: PASS.

- [ ] **Step 6: Write the failing tests (config validation)**

Add to `tslib/configs/test_config.py`:

```python
class TestStatisticStrategyConfig(unittest.TestCase):
    def _parse(self, *extra):
        argv = ["--data_path", "x.parquet", "--target_col", "p_gt10", *extra]
        return exp_parser().parse_args(argv)

    def test_arima_order_and_ar_lags_defaults(self):
        c = self._parse()
        self.assertEqual(c.arima_order, [1, 0, 0])
        self.assertEqual(c.ar_lags, 1)

    def test_statistic_ok(self):
        c = self._parse("--forecast_strategy", "statistic", "--models", "arima")
        self.assertIs(config_postprocess(c), c)

    def test_statistic_neural_model_rejected(self):
        c = self._parse("--forecast_strategy", "statistic", "--models", "lstm")
        with self.assertRaises(ValueError):
            config_postprocess(c)

    def test_statistic_multivariate_rejected(self):
        c = self._parse("--forecast_strategy", "statistic", "--models", "arima",
                        "--channels", "a.parquet:x", "b.parquet:y",
                        "--target_cols", "x", "y")
        with self.assertRaises(ValueError):
            config_postprocess(c)
```

- [ ] **Step 7: Run to verify they fail**

Run: `python -m unittest tslib.configs.test_config -v`
Expected: FAIL — no `arima_order` attribute; statistic branch not validated.

- [ ] **Step 8: Add config flags + statistic validation**

In `exp_parser()`, after the `--forecast_strategy` line:

```python
    parser.add_argument("--arima_order", nargs=3, type=int, default=[1, 0, 0],
                        help="ARIMA (p,d,q) for statistic strategy")
    parser.add_argument("--ar_lags", type=int, default=1,
                        help="AR lag count for statistic strategy")
```

In `config_postprocess`, extend the strategy block with a `statistic` branch (alongside the `direct/recursive` and `recursive` branches from Task 1):

```python
    if strategy == "statistic":
        from tslib.model.statistical import STAT_REGISTRY
        for m in config.models:
            if m not in STAT_REGISTRY:
                raise ValueError(
                    f"unknown statistical model '{m}'; "
                    f"registered: {sorted(STAT_REGISTRY)}")
        n_tgt = len(config.target_cols) if getattr(config, "target_cols", None) else 1
        if n_tgt != 1:
            raise ValueError(
                f"statistic strategy is univariate; got {n_tgt} targets")
```

- [ ] **Step 9: Run to verify they pass**

Run: `python -m unittest tslib.configs.test_config -v`
Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml tslib/model/statistical.py \
        tslib/configs/config.py tslib/model/test_statistical.py \
        tslib/configs/test_config.py
git commit -m "feat(model): statsmodels ARIMA/AR/Theta registry + statistic config"
```

---

### Task 5: StatisticalRunner + wire into dispatcher (Phase 2)

**Files:**
- Modify: `tslib/exp/metrics.py` (extract shared `run_metrics` helper)
- Modify: `tslib/exp/lightning_model.py` (`evaluate` uses `run_metrics`)
- Modify: `tslib/exp/strategy.py` (`StatisticalRunner`, wire `statistic` branch)
- Test: `tslib/exp/test_strategy.py` (StatisticalRunner test)
- Test: `tslib/exp/test_metrics.py` (run_metrics test)

**Interfaces:**
- Consumes: `STAT_REGISTRY`, `MetricContext`, `TrainResult`, `data_bundle.test_loader` (yields `(x, y)` with `x=(B,seq_len,1)`, `y=(B,pred_len,1)`), `data_bundle.target_cols`.
- Produces:
  - `run_metrics(pred, true, ctx, metric_names) -> Dict[str, float]` in `metrics.py` (flattens event per-channel dicts to `f"{name}_{ch}"` keys) — used by both `pl_model.evaluate` and `StatisticalRunner`.
  - `StatisticalRunner(model_name, config).fit_and_test(data_bundle, model_name, ckpt_path) -> TrainResult` with `best_val_loss = nan`, `strategy="statistic"`; degenerate-window fits fall back to persistence.

- [ ] **Step 1: Write the failing test (run_metrics)**

Add to `tslib/exp/test_metrics.py`:

```python
class TestRunMetrics(unittest.TestCase):
    def test_flattens_regression_and_event(self):
        import numpy as np
        from tslib.exp.metrics import run_metrics, MetricContext
        pred = np.zeros((4, 2, 1))
        true = np.full((4, 2, 1), 2.0)
        ctx = MetricContext(thresholds=[10.0], transform="log10",
                            target_cols=["p_gt10"])
        out = run_metrics(pred, true, ctx, ["mse", "tss"])
        self.assertIn("mse", out)
        self.assertIn("tss_p_gt10", out)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m unittest tslib.exp.test_metrics -v`
Expected: FAIL — `ImportError: cannot import name 'run_metrics'`.

- [ ] **Step 3: Extract `run_metrics` and reuse it in `evaluate`**

Add to `tslib/exp/metrics.py`:

```python
def run_metrics(pred, true, ctx, metric_names):
    out = {}
    for name in metric_names:
        val = METRIC_REGISTRY[name].fn(pred, true, ctx)
        if isinstance(val, dict):
            for ch, v in val.items():
                out[f"{name}_{ch}"] = float(v)
        else:
            out[name] = float(val)
    return out
```

In `tslib/exp/lightning_model.py`, replace the metric loop in `evaluate` with:

```python
        from tslib.exp.metrics import run_metrics
        return run_metrics(pred, true, ctx, self.config.metrics)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m unittest tslib.exp.test_metrics tslib.exp.test_lightning_model -v`
Expected: PASS (evaluate behavior unchanged).

- [ ] **Step 5: Write the failing test (StatisticalRunner)**

First **delete** the now-obsolete Phase-1 test `test_statistic_not_available_phase1` from `tslib/exp/test_strategy.py` (statistic is wired in this task, so it no longer raises). Then add to `tslib/exp/test_strategy.py`:

```python
import numpy as np
from tslib.exp.strategy import StatisticalRunner


def _stat_cfg():
    return SimpleNamespace(pred_len=4, arima_order=[1, 0, 0], ar_lags=2,
                           metrics=["mse"], transform="log10",
                           event_threshold=None)


def _fake_bundle():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(size=(3, 24, 1)), axis=1) + 5.0
    y = np.cumsum(rng.normal(size=(3, 4, 1)), axis=1) + 5.0
    bundle = SimpleNamespace(test_loader=[(x, y)], target_cols=["p_gt10"])
    return bundle


class TestStatisticalRunner(unittest.TestCase):
    def test_produces_result_with_nan_val_loss(self):
        runner = StatisticalRunner("ar", _stat_cfg())
        res = runner.fit_and_test(_fake_bundle(), "ar", "unused.pt")
        self.assertEqual(res.strategy, "statistic")
        self.assertTrue(np.isnan(res.best_val_loss))
        self.assertIn("mse", res.metrics)

    def test_dispatcher_routes_statistic(self):
        res = run_strategy("statistic", "ar", _fake_bundle(), _stat_cfg(),
                           "unused.pt")
        self.assertEqual(res.strategy, "statistic")
```

- [ ] **Step 6: Run to verify it fails**

Run: `python -m unittest tslib.exp.test_strategy -v`
Expected: FAIL — `ImportError: cannot import name 'StatisticalRunner'`; and `test_dispatcher_routes_statistic` still hits the Phase-1 `ValueError`.

- [ ] **Step 7: Implement `StatisticalRunner` and wire the dispatcher**

In `tslib/exp/strategy.py`, add imports and the runner, and replace the `statistic` branch:

```python
import numpy as np

from tslib.exp.metrics import MetricContext, run_metrics


class StatisticalRunner:
    def __init__(self, model_name, config):
        from tslib.model.statistical import STAT_REGISTRY
        self.builder = STAT_REGISTRY[model_name]
        self.model_name = model_name
        self.config = config

    def _forecast(self, history):
        steps = self.config.pred_len
        try:
            fc = self.builder(self.config).fit(history).forecast(steps)
            fc = np.asarray(fc, dtype=float)
            if fc.shape != (steps,) or not np.all(np.isfinite(fc)):
                raise ValueError("bad forecast")
            return fc
        except Exception:
            return np.full(steps, history[-1], dtype=float)  # persistence

    def fit_and_test(self, data_bundle, model_name, ckpt_path) -> TrainResult:
        preds, trues = [], []
        for x, y in data_bundle.test_loader:
            xb = np.asarray(x, dtype=float)   # (B, seq_len, 1)
            yb = np.asarray(y, dtype=float)   # (B, pred_len, 1)
            for i in range(xb.shape[0]):
                preds.append(self._forecast(xb[i, :, 0]).reshape(-1, 1))
                trues.append(yb[i])
        pred = np.stack(preds, axis=0)        # (N, pred_len, 1)
        true = np.stack(trues, axis=0)
        ctx = MetricContext(
            thresholds=getattr(self.config, "event_threshold", None),
            transform=getattr(self.config, "transform", "none"),
            target_cols=list(getattr(data_bundle, "target_cols", []) or []))
        metrics = run_metrics(pred, true, ctx, self.config.metrics)
        return TrainResult(model_name=model_name, best_val_loss=float("nan"),
                           metrics=metrics, ckpt_path=ckpt_path,
                           strategy="statistic")
```

Replace the `statistic` branch of `run_strategy`:

```python
    if strategy == "statistic":
        return StatisticalRunner(model_name, config).fit_and_test(
            data_bundle, model_name=model_name, ckpt_path=ckpt_path)
```

- [ ] **Step 8: Run to verify it passes**

Run: `python -m unittest tslib.exp.test_strategy -v`
Expected: PASS.

- [ ] **Step 9: Run the full suite (Phase 2 gate)**

Run: `python -m unittest discover -t . -s tslib -p 'test_*.py'`
Expected: PASS (all).

- [ ] **Step 10: Commit**

```bash
git add tslib/exp/metrics.py tslib/exp/lightning_model.py tslib/exp/strategy.py \
        tslib/exp/test_strategy.py tslib/exp/test_metrics.py
git commit -m "feat(exp): StatisticalRunner (per-window fit) wired into dispatcher"
```

---

## Documentation follow-up (fold into the last task of each phase)

- Phase 1 (with Task 3): update `tslib/exp/README.md` and `tslib/configs/README.md` to mention `--forecast_strategy {direct,recursive,statistic}`, the recursive all-channels rule, and `merge_comparisons` for cross-strategy tables.
- Phase 2 (with Task 5): update `tslib/model/README.md` to describe the statistical registry (`register_stat`, `STAT_REGISTRY`, `fit`/`forecast` contract) and note statsmodels as a dependency.

Keep these to the brief "pointer" style already used in those READMEs.
```
