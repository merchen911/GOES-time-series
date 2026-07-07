# v003 Lightning Restructure + Train-Time Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the neural training path on genuine `pytorch_lightning` (a real `LightningModule` + `Trainer`) and add a training-time pre-flight gate that skips/aborts/proceeds on models whose estimated full training time exceeds a limit — shipped as a new `sw-framework-v003` directory that carries over v002's git history.

**Architecture:** `run_strategy` builds a `ForecastModule(pl.LightningModule)` wrapping the existing forecast adapters, drives it with a `pl.Trainer` (ModelCheckpoint + a `TimingGateCallback`), and returns the same `TrainResult`. The forecast-strategy dispatch, metrics, comparison table, and `merge_comparisons` are preserved; only the neural train/eval mechanism changes. The `statistic` strategy stays a plain statsmodels runner (no Lightning).

**Tech Stack:** Python 3.12, PyTorch 2.4.1, pytorch_lightning 2.6.5, statsmodels, unittest.

## Global Constraints

- **Tests:** unittest only (no pytest). Discover with `python3.12 -m unittest discover -t . -s tslib -p 'test_*.py'`; run one module with `python3.12 -m unittest tslib.exp.test_lightning_model -v`.
- **Interpreter:** always `python3.12` (has torch/numpy/pandas/statsmodels/pytorch_lightning). Never `python3` (3.10, no libs).
- **Lightning package:** `import pytorch_lightning as pl` (the classic package, already installed at 2.6.5). Do NOT use the unified `lightning` package.
- **Backward compatibility:** `TrainResult` new fields are appended after `strategy` as keyword-defaulted (`skipped: bool = False`, `est_train_hours: float = nan`) so positional construction `TrainResult("lstm", 0.5, {...}, "a.pt")` and `TrainResult(..., strategy="direct")` keep working.
- **Device-aware toy models:** test models must place output tensors on their parameter's device (`+ self.p` pattern), per prior GPU lesson.
- **Metrics run in stored (log10) space** via `MetricContext(thresholds, transform, target_cols)`; physical thresholds auto-convert. Per-channel event keys are `f"{name}_{ch}"`.
- **Recursive requires targets == all input channels** (unchanged rule). **Statistic is univariate** (unchanged rule).
- **GPU fixed to device 1** for real runs via `CUDA_VISIBLE_DEVICES=1`; `Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)`.
- **Gate defaults (fixed):** `--max_train_hours 6.0`, `--on_slow skip`, `--probe_batches 3`.
- **Never commit datasets** (`.gitignore` already excludes `*.parquet`, `runs/`, `*.pt`, etc.). Commit/push only the source + docs.

---

## File Structure

- `tslib/exp/lightning_model.py` — **replaced**: `ForecastModule(pl.LightningModule)` + `TrainResult` dataclass. (removes `pl_model`, `_run_epoch`, `evaluate`.)
- `tslib/exp/callbacks.py` — **new**: `estimate_train_hours(...)` pure helper + `TimingGateCallback(pl.Callback)`.
- `tslib/exp/strategy.py` — **modified**: neural branch rewritten to build `ForecastModule` + `pl.Trainer`; `statistic` branch unchanged.
- `tslib/exp/exp.py` — **modified**: `build_comparison` gains `skipped`/`est_train_hours` columns; `run_experiment` prints a skipped summary. (existing `_set_seed` seeding of torch/numpy/random is retained unchanged.)
- `tslib/configs/config.py` — **modified**: `--max_train_hours`, `--on_slow`, `--probe_batches` + validation.
- `tslib/model/__init__.py` — **unchanged** (adapters reused).
- `pyproject.toml`, `requirements.txt`, `README.md`, `docs/benchmark-conditions.md`, `docs/lightning-migration.md` (new) — docs/deps.
- Tests: `tslib/exp/test_lightning_model.py` (rewritten), `tslib/exp/test_callbacks.py` (new), `tslib/exp/test_strategy.py` (extended), `tslib/exp/test_exp.py` (extended), `tslib/configs/test_config.py` (extended).

---

### Task 1: v003 scaffold, dependency, version bump

**Files:**
- Create: `../sw-framework-v003/` (copy of v002 including `.git`)
- Modify: `pyproject.toml`, `requirements.txt` (in v003)

**Interfaces:**
- Consumes: nothing.
- Produces: the working directory `sw-framework-v003` on branch `feat/lightning-restructure`, with `pytorch_lightning` declared as a dependency. All subsequent tasks run inside `sw-framework-v003`.

- [ ] **Step 1: Create v003 as a copy of v002 (carrying git history)**

```bash
cd /NAS/ioGuard3/vol3/spaceai/SW_framework/workdir
cp -a sw-framework-v002 sw-framework-v003
# remove git-ignored scratch that must not seed the new tree
rm -rf sw-framework-v003/runs sw-framework-v003/.superpowers
find sw-framework-v003 -type d -name __pycache__ -prune -exec rm -rf {} +
find sw-framework-v003 -type d -name '.ipynb_checkpoints' -prune -exec rm -rf {} +
find sw-framework-v003 -type d -name '*.egg-info' -prune -exec rm -rf {} +
```

- [ ] **Step 2: Create the working branch**

```bash
cd /NAS/ioGuard3/vol3/spaceai/SW_framework/workdir/sw-framework-v003
git checkout -b feat/lightning-restructure
git status --short   # expect clean (scratch was git-ignored)
```

- [ ] **Step 3: Verify pytorch_lightning imports and existing tests pass**

Run:
```bash
cd /NAS/ioGuard3/vol3/spaceai/SW_framework/workdir/sw-framework-v003
python3.12 -c "import pytorch_lightning as pl; print(pl.__version__)"
python3.12 -m unittest discover -t . -s tslib -p 'test_*.py'
```
Expected: prints `2.6.5`; all existing tests pass (pl_model still present at this point).

- [ ] **Step 4: Declare the dependency and bump version**

In `pyproject.toml` change line 7 and line 10:
```toml
version = "0.3.0"
```
```toml
dependencies = ["numpy", "pandas", "torch", "statsmodels", "pytorch_lightning"]
```

In `requirements.txt` (currently `numpy` / `pandas` / `torch`), make it:
```
numpy
pandas
torch
statsmodels
pytorch_lightning
```

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml requirements.txt
git commit -m "chore(v003): scaffold Lightning restructure branch + declare pytorch_lightning"
```

---

### Task 2: `ForecastModule` LightningModule (replaces `pl_model`)

**Files:**
- Modify (full replace): `tslib/exp/lightning_model.py`
- Test (rewrite): `tslib/exp/test_lightning_model.py`

**Interfaces:**
- Consumes: `build_loss(config)` from `tslib.exp.losses`; `run_metrics(pred, true, ctx, metric_names)` and `MetricContext` from `tslib.exp.metrics`. `config` provides `lr, weight_decay, loss, metrics, transform, event_threshold, seq_len, pred_len, label_len`.
- Produces:
  - `TrainResult(model_name, best_val_loss, metrics, ckpt_path, strategy="direct", skipped=False, est_train_hours=nan)` — dataclass.
  - `ForecastModule(model, config, metric_ctx, strategy="direct")` — a `pl.LightningModule` with `training_step`, `validation_step`, `test_step`, `on_test_epoch_end` (populates `self.test_metrics: Dict[str,float]`), `configure_optimizers`, and public attrs `_gate_skipped: bool` and `_est_train_hours: float` for the gate to set.

- [ ] **Step 1: Write the failing tests**

Replace the entire contents of `tslib/exp/test_lightning_model.py` with:
```python
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from tslib.exp.lightning_model import ForecastModule, TrainResult
from tslib.exp.metrics import MetricContext


def _cfg(**kw):
    base = dict(lr=1e-3, weight_decay=0.0, epochs=1, loss="mse",
                metrics=["mse", "mae"], transform="log10",
                event_threshold=None, seq_len=4, pred_len=2, label_len=0)
    base.update(kw)
    return SimpleNamespace(**base)


class _ConstModel(torch.nn.Module):
    """Emits (B, pred_len, t) filled with a learnable scalar (device-aware)."""
    def __init__(self, pred_len, t):
        super().__init__()
        self.pred_len, self.t = pred_len, t
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b = x.shape[0]
        return torch.zeros(b, self.pred_len, self.t, device=self.p.device) + self.p


class TestTrainResultFields(unittest.TestCase):
    def test_new_fields_default_and_positional(self):
        r = TrainResult("lstm", 0.5, {"mse": 0.3}, "a.pt")
        self.assertFalse(r.skipped)
        self.assertTrue(np.isnan(r.est_train_hours))
        r2 = TrainResult("lstm", 0.5, {"mse": 0.3}, "a.pt", strategy="recursive")
        self.assertEqual(r2.strategy, "recursive")


class TestTrainingStepAlignment(unittest.TestCase):
    def test_one_step_pred_against_multistep_target(self):
        # model emits (B,1,1); target is (B,2,1). training_step must align y to
        # the first step and return a finite loss.
        m = ForecastModule(_ConstModel(1, 1), _cfg(pred_len=2), None)
        x = torch.zeros(4, 4, 1)
        y = torch.full((4, 2, 1), 2.0)
        loss = m.training_step((x, y), 0)
        self.assertTrue(torch.isfinite(loss).item())


class TestTestMetrics(unittest.TestCase):
    def test_regression_and_event_keys(self):
        ctx = MetricContext(thresholds=[10.0], transform="log10",
                            target_cols=["p_gt10"])
        m = ForecastModule(_ConstModel(2, 1),
                           _cfg(metrics=["mse", "tss"], event_threshold=[10.0]),
                           ctx)
        x = torch.zeros(4, 4, 1)
        y = torch.full((4, 2, 1), 2.0)  # log10 space, > log10(10)=1 → event
        m.test_step((x, y), 0)
        m.on_test_epoch_end()
        self.assertIn("mse", m.test_metrics)
        self.assertIn("tss_p_gt10", m.test_metrics)


class TestTrainerSmoke(unittest.TestCase):
    def test_fit_and_test_via_trainer(self):
        import pytorch_lightning as pl
        ctx = MetricContext(thresholds=None, transform="log10",
                            target_cols=["p_gt10"])
        m = ForecastModule(_ConstModel(2, 1), _cfg(metrics=["mse"]), ctx)
        x = torch.zeros(8, 4, 1)
        y = torch.full((8, 2, 1), 2.0)
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        trainer = pl.Trainer(max_epochs=1, accelerator="cpu", devices=1,
                             limit_train_batches=2, limit_val_batches=1,
                             limit_test_batches=2, logger=False,
                             enable_checkpointing=False,
                             enable_progress_bar=False,
                             enable_model_summary=False)
        trainer.fit(m, loader, loader)
        trainer.test(m, loader)
        self.assertIn("mse", m.test_metrics)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3.12 -m unittest tslib.exp.test_lightning_model -v`
Expected: FAIL — `ImportError: cannot import name 'ForecastModule'`.

- [ ] **Step 3: Replace `lightning_model.py`**

Replace the entire contents of `tslib/exp/lightning_model.py` with:
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import pytorch_lightning as pl

from tslib.exp.losses import build_loss
from tslib.exp.metrics import run_metrics


@dataclass
class TrainResult:
    model_name: str
    best_val_loss: float
    metrics: Dict[str, float]
    ckpt_path: str
    strategy: str = "direct"
    skipped: bool = False
    est_train_hours: float = float("nan")


class ForecastModule(pl.LightningModule):
    """Wraps a forecast adapter/model and drives it under PyTorch Lightning.

    The wrapped model's own train/eval branch (e.g. RecursiveForecastAdapter's
    1-step train vs full rollout eval) is honored automatically because
    Lightning toggles module train/eval mode around the step hooks.
    """

    def __init__(self, model, config, metric_ctx, strategy: str = "direct") -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.metric_ctx = metric_ctx
        self.strategy = strategy
        self.criterion = build_loss(config)
        self._test_pred: List[np.ndarray] = []
        self._test_true: List[np.ndarray] = []
        self.test_metrics: Dict[str, float] = {}
        # gate hooks (set by TimingGateCallback)
        self._gate_skipped: bool = False
        self._est_train_hours: float = float("nan")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y[:, :pred.shape[1], :])
        self.log("train_loss", loss, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y[:, :pred.shape[1], :])
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        self._test_pred.append(pred.detach().cpu().numpy())
        self._test_true.append(y.detach().cpu().numpy())

    def on_test_epoch_end(self):
        pred = np.concatenate(self._test_pred, axis=0)
        true = np.concatenate(self._test_true, axis=0)
        self.test_metrics = run_metrics(pred, true, self.metric_ctx,
                                        self.config.metrics)
        self._test_pred.clear()
        self._test_true.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config.lr,
                                weight_decay=self.config.weight_decay)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3.12 -m unittest tslib.exp.test_lightning_model -v`
Expected: PASS (4 tests). Note: `test_test_metrics` and `test_training_step_alignment` call the step methods directly (no Trainer); `TestTrainerSmoke` exercises the real Lightning wiring.

- [ ] **Step 5: Commit**

```bash
git add tslib/exp/lightning_model.py tslib/exp/test_lightning_model.py
git commit -m "feat(exp): ForecastModule LightningModule replaces hand-rolled pl_model"
```

---

### Task 3: Lightning-backed `run_strategy` (neural branch)

**Files:**
- Modify: `tslib/exp/strategy.py`
- Test (extend): `tslib/exp/test_strategy.py`

**Interfaces:**
- Consumes: `ForecastModule`, `TrainResult` (Task 2); `build_model(model_name, config, input_size, target_indices, strategy)` from `tslib.model`; `MetricContext` from `tslib.exp.metrics`; `data_bundle` with `.train_loader/.val_loader/.test_loader/.input_size/.target_indices/.target_cols`; `config.epochs/lr/weight_decay/...`.
- Produces: `run_strategy(strategy, model_name, data_bundle, config, ckpt_path) -> TrainResult` — direct/recursive route through a `pl.Trainer` with a `ModelCheckpoint(monitor="val_loss")`; `statistic` routes to `StatisticalRunner` (unchanged); unknown strategy raises `ValueError`. Best val loss comes from `ModelCheckpoint.best_model_score`. (No gate yet; `skipped` is always `False` and `est_train_hours` is `nan` until Task 6.)

- [ ] **Step 1: Write the failing test**

Append to `tslib/exp/test_strategy.py`:
```python
class TestNeuralStrategyLightning(unittest.TestCase):
    def test_direct_routes_through_trainer(self):
        import tempfile, os
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from types import SimpleNamespace
        import tslib.exp.strategy as strat

        class _Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x):
                b = x.shape[0]
                return torch.zeros(b, 2, 1, device=self.p.device) + self.p

        # avoid needing full backbone config: build_model returns a toy model
        orig = strat.build_model
        strat.build_model = lambda *a, **k: _Toy()
        try:
            x = torch.zeros(8, 4, 1)
            y = torch.full((8, 2, 1), 2.0)
            loader = DataLoader(TensorDataset(x, y), batch_size=4)
            bundle = SimpleNamespace(train_loader=loader, val_loader=loader,
                                     test_loader=loader, input_size=1,
                                     target_indices=[0], target_cols=["p_gt10"])
            cfg = SimpleNamespace(lr=1e-3, weight_decay=0.0, epochs=1, loss="mse",
                                  metrics=["mse"], transform="log10",
                                  event_threshold=None, seq_len=4, pred_len=2,
                                  label_len=0)
            with tempfile.TemporaryDirectory() as d:
                res = strat.run_strategy("direct", "toy", bundle, cfg,
                                         os.path.join(d, "toy.pt"))
            self.assertEqual(res.strategy, "direct")
            self.assertFalse(res.skipped)
            self.assertIn("mse", res.metrics)
        finally:
            strat.build_model = orig
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3.12 -m unittest tslib.exp.test_strategy.TestNeuralStrategyLightning -v`
Expected: FAIL — the current `run_strategy` builds a real `pl_model` (removed in Task 2), so this raises `ImportError`/`AttributeError`.

- [ ] **Step 3: Rewrite the neural branch of `strategy.py`**

Replace the imports and the `run_strategy` function in `tslib/exp/strategy.py`. Keep `StatisticalRunner` exactly as-is. Change the top imports from:
```python
from tslib.model import build_model
from tslib.exp.lightning_model import pl_model, TrainResult
from tslib.exp.metrics import MetricContext, run_metrics
```
to:
```python
import os

from tslib.model import build_model
from tslib.exp.lightning_model import ForecastModule, TrainResult
from tslib.exp.metrics import MetricContext, run_metrics
```

Then replace `run_strategy` (the last function) with:
```python
def _run_neural(strategy, model_name, data_bundle, config, ckpt_path) -> TrainResult:
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

    model = build_model(model_name, config, data_bundle.input_size,
                        data_bundle.target_indices, strategy=strategy)
    ctx = MetricContext(
        thresholds=getattr(config, "event_threshold", None),
        transform=getattr(config, "transform", "none"),
        target_cols=list(getattr(data_bundle, "target_cols", []) or []))
    module = ForecastModule(model, config, ctx, strategy=strategy)

    ckpt_dir = os.path.dirname(ckpt_path) or "."
    ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    ckpt_cb = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1,
                              dirpath=ckpt_dir, filename=ckpt_name)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=config.epochs, accelerator=accelerator, devices=1,
        callbacks=[ckpt_cb],
        logger=CSVLogger(save_dir=ckpt_dir, name="lightning"),
        enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(module, data_bundle.train_loader, data_bundle.val_loader)

    trainer.test(module, data_bundle.test_loader, ckpt_path="best")
    best = (float(ckpt_cb.best_model_score)
            if ckpt_cb.best_model_score is not None else float("nan"))
    return TrainResult(model_name=model_name, best_val_loss=best,
                       metrics=module.test_metrics, ckpt_path=ckpt_path,
                       strategy=strategy)


def run_strategy(strategy, model_name, data_bundle, config, ckpt_path) -> TrainResult:
    if strategy in ("direct", "recursive"):
        return _run_neural(strategy, model_name, data_bundle, config, ckpt_path)
    if strategy == "statistic":
        return StatisticalRunner(model_name, config).fit_and_test(
            data_bundle, model_name=model_name, ckpt_path=ckpt_path)
    raise ValueError(f"unknown forecast_strategy '{strategy}'")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3.12 -m unittest tslib.exp.test_strategy -v`
Expected: PASS (all: unknown-strategy, statistic runner, target-channel fit, new neural routing).

- [ ] **Step 5: Commit**

```bash
git add tslib/exp/strategy.py tslib/exp/test_strategy.py
git commit -m "feat(exp): run_strategy drives ForecastModule via pl.Trainer + ModelCheckpoint"
```

---

### Task 4: Gate config flags + validation

**Files:**
- Modify: `tslib/configs/config.py`
- Test (extend): `tslib/configs/test_config.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `exp_parser()` accepts `--max_train_hours` (float, default `6.0`), `--on_slow` (choice `{skip,abort,proceed}`, default `skip`), `--probe_batches` (int, default `3`). `config_postprocess` raises `ValueError` when `max_train_hours <= 0` or `probe_batches < 1`.

- [ ] **Step 1: Write the failing tests**

Append to `tslib/configs/test_config.py`:
```python
class TestGateFlags(unittest.TestCase):
    def _base_argv(self, *extra):
        argv = ["--data_path", "x.parquet", "--target_col", "p_gt10", *extra]
        from tslib.configs.config import exp_parser
        return exp_parser().parse_args(argv)

    def test_gate_defaults(self):
        cfg = self._base_argv()
        self.assertEqual(cfg.max_train_hours, 6.0)
        self.assertEqual(cfg.on_slow, "skip")
        self.assertEqual(cfg.probe_batches, 3)

    def test_rejects_nonpositive_max_train_hours(self):
        from tslib.configs.config import config_postprocess
        cfg = self._base_argv("--max_train_hours", "0")
        with self.assertRaises(ValueError):
            config_postprocess(cfg)

    def test_rejects_probe_batches_below_one(self):
        from tslib.configs.config import config_postprocess
        cfg = self._base_argv("--probe_batches", "0")
        with self.assertRaises(ValueError):
            config_postprocess(cfg)
```
(If `test_config.py` does not already `import unittest`, add it at the top.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3.12 -m unittest tslib.configs.test_config.TestGateFlags -v`
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'max_train_hours'`.

- [ ] **Step 3: Add the flags and validation**

In `tslib/configs/config.py`, in `exp_parser` right after the `--hidden_size` line (line 53), add:
```python
    # train-time pre-flight gate
    parser.add_argument("--max_train_hours", type=float, default=6.0,
                        help="skip/abort a model if estimated full training "
                             "time exceeds this many hours")
    parser.add_argument("--on_slow", type=str, default="skip",
                        choices=["skip", "abort", "proceed"],
                        help="action when est. train time exceeds --max_train_hours")
    parser.add_argument("--probe_batches", type=int, default=3,
                        help="number of training batches timed for the gate estimate")
```

In `config_postprocess`, right before the final `return config`, add:
```python
    if getattr(config, "max_train_hours", 1.0) <= 0:
        raise ValueError("max_train_hours must be > 0.")
    if getattr(config, "probe_batches", 1) < 1:
        raise ValueError("probe_batches must be >= 1.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3.12 -m unittest tslib.configs.test_config -v`
Expected: PASS (existing + 3 new).

- [ ] **Step 5: Commit**

```bash
git add tslib/configs/config.py tslib/configs/test_config.py
git commit -m "feat(config): train-time gate flags (--max_train_hours/--on_slow/--probe_batches)"
```

---

### Task 5: `TimingGateCallback` + `estimate_train_hours`

**Files:**
- Create: `tslib/exp/callbacks.py`
- Test (new): `tslib/exp/test_callbacks.py`

**Interfaces:**
- Consumes: `config.max_train_hours/on_slow/probe_batches`; a `pl_module` exposing `.device`, `.model`, and writable `_gate_skipped`/`_est_train_hours` (Task 2 provides these); `trainer.num_training_batches`, `trainer.max_epochs`, `trainer.should_stop`.
- Produces:
  - `estimate_train_hours(sec_per_batch, n_batches, epochs) -> float` = `sec_per_batch * n_batches * epochs / 3600.0`.
  - `TimingGateCallback(config)` — a `pl.Callback` that times the first `probe_batches` training batches, computes the median per-batch time and the estimated full training time, sets `pl_module._est_train_hours`, and on exceeding the limit performs the `on_slow` policy: `skip` → set `pl_module._gate_skipped=True` and `trainer.should_stop=True`; `abort` → raise `RuntimeError`; `proceed` → log and continue.

- [ ] **Step 1: Write the failing tests**

Create `tslib/exp/test_callbacks.py`:
```python
import unittest
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

from tslib.exp.callbacks import estimate_train_hours, TimingGateCallback
from tslib.exp.lightning_model import ForecastModule
from tslib.exp.metrics import MetricContext


def _cfg(on_slow, max_hours=6.0, probe=2):
    return SimpleNamespace(lr=1e-3, weight_decay=0.0, epochs=1, loss="mse",
                           metrics=["mse"], transform="log10",
                           event_threshold=None, seq_len=4, pred_len=2,
                           label_len=0, max_train_hours=max_hours,
                           on_slow=on_slow, probe_batches=probe)


class _Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b = x.shape[0]
        return torch.zeros(b, 2, 1, device=self.p.device) + self.p


def _loader():
    x = torch.zeros(16, 4, 1)
    y = torch.full((16, 2, 1), 2.0)
    return DataLoader(TensorDataset(x, y), batch_size=4)


def _trainer(cb):
    return pl.Trainer(max_epochs=1, accelerator="cpu", devices=1,
                      limit_train_batches=8, limit_val_batches=1,
                      logger=False, enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False,
                      callbacks=[cb])


class TestEstimate(unittest.TestCase):
    def test_formula(self):
        self.assertAlmostEqual(estimate_train_hours(0.5, 100, 3), 150 / 3600.0)


class TestGatePolicy(unittest.TestCase):
    def test_skip_stops_training(self):
        m = ForecastModule(_Toy(), _cfg("skip", max_hours=1e-12), None)
        _trainer(TimingGateCallback(_cfg("skip", max_hours=1e-12))).fit(m, _loader(), _loader())
        self.assertTrue(m._gate_skipped)

    def test_abort_raises(self):
        m = ForecastModule(_Toy(), _cfg("abort", max_hours=1e-12), None)
        with self.assertRaises(RuntimeError):
            _trainer(TimingGateCallback(_cfg("abort", max_hours=1e-12))).fit(m, _loader(), _loader())

    def test_proceed_continues(self):
        m = ForecastModule(_Toy(), _cfg("proceed", max_hours=1e-12), None)
        _trainer(TimingGateCallback(_cfg("proceed", max_hours=1e-12))).fit(m, _loader(), _loader())
        self.assertFalse(m._gate_skipped)
        self.assertGreater(m._est_train_hours, 0.0)

    def test_fast_model_not_skipped(self):
        m = ForecastModule(_Toy(), _cfg("skip", max_hours=1e9), None)
        _trainer(TimingGateCallback(_cfg("skip", max_hours=1e9))).fit(m, _loader(), _loader())
        self.assertFalse(m._gate_skipped)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3.12 -m unittest tslib.exp.test_callbacks -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tslib.exp.callbacks'`.

- [ ] **Step 3: Create `callbacks.py`**

```python
from __future__ import annotations

import statistics
import time

import torch
import pytorch_lightning as pl


def estimate_train_hours(sec_per_batch: float, n_batches: float,
                         epochs: int) -> float:
    """Wall-clock hours to train `epochs` full epochs at `sec_per_batch`."""
    return float(sec_per_batch) * float(n_batches) * float(epochs) / 3600.0


class TimingGateCallback(pl.Callback):
    """Times the first `probe_batches` training batches, estimates full
    training time, and applies the `on_slow` policy if it exceeds the limit.

    median over the probed batches absorbs the first-batch CUDA/cuDNN warmup.
    """

    def __init__(self, config) -> None:
        self.max_train_hours = float(config.max_train_hours)
        self.on_slow = config.on_slow
        self.probe_batches = int(config.probe_batches)
        self._durations: list[float] = []
        self._t0 = None
        self._done = False

    def _sync(self, device) -> None:
        if device is not None and str(device).startswith("cuda"):
            torch.cuda.synchronize()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self._done:
            return
        self._sync(pl_module.device)
        self._t0 = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._done or self._t0 is None:
            return
        self._sync(pl_module.device)
        self._durations.append(time.perf_counter() - self._t0)
        if len(self._durations) < self.probe_batches:
            return
        self._done = True
        sec = statistics.median(self._durations)
        est = estimate_train_hours(sec, trainer.num_training_batches,
                                   trainer.max_epochs)
        pl_module._est_train_hours = est
        name = type(getattr(pl_module, "model", pl_module)).__name__
        print(f"[gate] model={name} s/batch={sec:.3f} "
              f"est_train~{est:.2f}h (limit {self.max_train_hours}h)")
        if est <= self.max_train_hours:
            return
        if self.on_slow == "abort":
            raise RuntimeError(
                f"train-time gate: est {est:.2f}h > limit "
                f"{self.max_train_hours}h for {name}")
        if self.on_slow == "skip":
            pl_module._gate_skipped = True
            trainer.should_stop = True
        # "proceed": log-only, continue training
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3.12 -m unittest tslib.exp.test_callbacks -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add tslib/exp/callbacks.py tslib/exp/test_callbacks.py
git commit -m "feat(exp): TimingGateCallback + estimate_train_hours train-time gate"
```

---

### Task 6: Wire the gate into `run_strategy` + surface skips in the comparison

**Files:**
- Modify: `tslib/exp/strategy.py` (`_run_neural`)
- Modify: `tslib/exp/exp.py` (`build_comparison`, `run_experiment`)
- Test (extend): `tslib/exp/test_strategy.py`, `tslib/exp/test_exp.py`

**Interfaces:**
- Consumes: `TimingGateCallback` (Task 5); `ForecastModule._gate_skipped/_est_train_hours` (Task 2); `TrainResult.skipped/est_train_hours` (Task 2).
- Produces: neural runs add `TimingGateCallback(config)` to the trainer; when `module._gate_skipped`, `_run_neural` returns `TrainResult(..., skipped=True, est_train_hours=...)` and skips the test pass. `build_comparison` includes `skipped` and `est_train_hours` columns (skipped rows keep `NaN` metrics and sort last). `run_experiment` prints a `SKIPPED (too slow): <model> ~<est>h — re-include with --on_slow proceed` summary line per skipped model.

- [ ] **Step 1: Write the failing tests**

Append to `tslib/exp/test_exp.py`:
```python
class TestSkippedColumns(unittest.TestCase):
    def test_skipped_row_present_and_last(self):
        from tslib.exp.exp import build_comparison
        res = [
            TrainResult("lstm", 0.3, {"mse": 0.2}, "a.pt", strategy="direct"),
            TrainResult("timesnet", float("nan"), {}, "b.pt",
                        strategy="direct", skipped=True, est_train_hours=14.0),
        ]
        df = build_comparison(res, sort_metric="mse")
        self.assertIn("skipped", df.columns)
        self.assertIn("est_train_hours", df.columns)
        # NaN metric sorts last
        self.assertEqual(list(df["model"]), ["lstm", "timesnet"])
        self.assertTrue(bool(df.iloc[-1]["skipped"]))
```

Update the existing exact-column assertions in `tslib/exp/test_exp.py` to include the two new columns:
- In `TestBuildComparison.test_dynamic_columns`, change the expected set to
  `{"strategy", "model", "best_val_loss", "skipped", "est_train_hours", "mse", "tss_p_gt10"}`.
- In `TestStrategyColumn.test_strategy_column_present`, change the expected set to
  `{"strategy", "model", "best_val_loss", "skipped", "est_train_hours", "mse"}`.

Append to `tslib/exp/test_strategy.py`:
```python
class TestGateSkipInStrategy(unittest.TestCase):
    def test_slow_model_skipped(self):
        import tempfile, os
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from types import SimpleNamespace
        import tslib.exp.strategy as strat

        class _Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x):
                b = x.shape[0]
                return torch.zeros(b, 2, 1, device=self.p.device) + self.p

        orig = strat.build_model
        strat.build_model = lambda *a, **k: _Toy()
        try:
            x = torch.zeros(16, 4, 1)
            y = torch.full((16, 2, 1), 2.0)
            loader = DataLoader(TensorDataset(x, y), batch_size=4)
            bundle = SimpleNamespace(train_loader=loader, val_loader=loader,
                                     test_loader=loader, input_size=1,
                                     target_indices=[0], target_cols=["p_gt10"])
            cfg = SimpleNamespace(lr=1e-3, weight_decay=0.0, epochs=1, loss="mse",
                                  metrics=["mse"], transform="log10",
                                  event_threshold=None, seq_len=4, pred_len=2,
                                  label_len=0, max_train_hours=1e-12,
                                  on_slow="skip", probe_batches=2)
            with tempfile.TemporaryDirectory() as d:
                res = strat.run_strategy("direct", "toy", bundle, cfg,
                                         os.path.join(d, "toy.pt"))
            self.assertTrue(res.skipped)
            self.assertGreater(res.est_train_hours, 0.0)
        finally:
            strat.build_model = orig
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3.12 -m unittest tslib.exp.test_exp tslib.exp.test_strategy -v`
Expected: FAIL — `build_comparison` lacks `skipped`/`est_train_hours` columns; `_run_neural` has no gate so `res.skipped` is `False`.

- [ ] **Step 3: Add the gate to `_run_neural`**

In `tslib/exp/strategy.py`, inside `_run_neural`, add the import and callback, and the skip branch. Change the callback line and add the gate import near the other trainer imports:
```python
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from tslib.exp.callbacks import TimingGateCallback
```
Change the `callbacks=[ckpt_cb]` line to:
```python
        callbacks=[ckpt_cb, TimingGateCallback(config)],
```
Immediately after `trainer.fit(...)` and before `trainer.test(...)`, insert:
```python
    if module._gate_skipped:
        return TrainResult(model_name=model_name, best_val_loss=float("nan"),
                           metrics={}, ckpt_path=ckpt_path, strategy=strategy,
                           skipped=True, est_train_hours=module._est_train_hours)
```
And change the final (non-skipped) `TrainResult(...)` return to carry the estimate:
```python
    return TrainResult(model_name=model_name, best_val_loss=best,
                       metrics=module.test_metrics, ckpt_path=ckpt_path,
                       strategy=strategy, est_train_hours=module._est_train_hours)
```

- [ ] **Step 4: Add columns + skip summary in `exp.py`**

In `tslib/exp/exp.py`, replace `build_comparison` (lines 14-20) with:
```python
def build_comparison(results, sort_metric):
    rows = [{"strategy": r.strategy, "model": r.model_name,
             "best_val_loss": r.best_val_loss,
             "skipped": r.skipped, "est_train_hours": r.est_train_hours,
             **r.metrics}
            for r in results]
    df = pd.DataFrame(rows)
    key = sort_metric if sort_metric in df.columns else "best_val_loss"
    return df.sort_values(by=key, na_position="last").reset_index(drop=True)
```
In `run_experiment`, after `comparison = build_comparison(results, config.sort_metric)` and before writing the CSV, add:
```python
    for r in results:
        if getattr(r, "skipped", False):
            print(f"SKIPPED (too slow): {r.model_name} "
                  f"~{r.est_train_hours:.2f}h — re-include with --on_slow proceed")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3.12 -m unittest tslib.exp.test_exp tslib.exp.test_strategy -v`
Expected: PASS (updated column sets, new skipped-row test, gate-skip strategy test).

- [ ] **Step 6: Full suite green**

Run: `python3.12 -m unittest discover -t . -s tslib -p 'test_*.py'`
Expected: PASS (entire suite).

- [ ] **Step 7: Commit**

```bash
git add tslib/exp/strategy.py tslib/exp/exp.py tslib/exp/test_exp.py tslib/exp/test_strategy.py
git commit -m "feat(exp): wire TimingGateCallback into run_strategy; surface skips in comparison"
```

---

### Task 7: Documentation

**Files:**
- Modify: `README.md`, `docs/benchmark-conditions.md`
- Create: `docs/lightning-migration.md`

**Interfaces:**
- Consumes: the shipped behavior from Tasks 1-6.
- Produces: user-facing docs describing the Lightning architecture, how to customize via `ForecastModule` hooks, and the train-time gate flags.

- [ ] **Step 1: Write `docs/lightning-migration.md`**

Create `docs/lightning-migration.md` with these sections (write real prose, no placeholders):
- **Why:** v002's `pl_model` was a hand-rolled loop; v003 uses real `pytorch_lightning` (2.6.5) so users customize via standard hooks.
- **Architecture:** `run_strategy` → `ForecastModule(pl.LightningModule)` + `pl.Trainer(ModelCheckpoint + TimingGateCallback)`; `statistic` stays a plain statsmodels runner.
- **Customizing:** subclass or edit `ForecastModule` (`training_step`/`validation_step`/`test_step`/`configure_optimizers`) in `tslib/exp/lightning_model.py`; the wrapped adapter's train/eval branch (recursive rollout) is preserved because Lightning toggles module mode.
- **Train-time gate:** what it does; flags `--max_train_hours` (default 6.0), `--on_slow` (`skip`/`abort`/`proceed`, default `skip`), `--probe_batches` (default 3); how a skipped model appears (`skipped=True`, `est_train_hours` in the comparison table + a `SKIPPED (too slow): …` stdout line); how to re-include (`--on_slow proceed`). Note the gate estimates **training** time only; recursive rollout **eval** cost is handled separately via test-window subsampling.
- **Checkpointing:** `ModelCheckpoint(monitor="val_loss")`; best checkpoint (`<model>.ckpt`) reloaded for the test pass via `trainer.test(ckpt_path="best")`.

- [ ] **Step 2: Update `docs/benchmark-conditions.md`**

Add a short subsection under "Fixed conditions (all runs)" noting the gate flags and their defaults, and that timesnet-class models may be auto-skipped unless `--on_slow proceed` is passed. Reference `docs/lightning-migration.md`.

- [ ] **Step 3: Update `README.md`**

In `README.md`, update any description of the training/experiment mechanism to say the framework runs on `pytorch_lightning`, and add a one-line pointer to `docs/lightning-migration.md`. Add `pytorch_lightning` to any stated install/requirements list.

- [ ] **Step 4: Verify docs reference real flags/paths**

Run:
```bash
cd /NAS/ioGuard3/vol3/spaceai/SW_framework/workdir/sw-framework-v003
grep -n "max_train_hours\|on_slow\|probe_batches" tslib/configs/config.py docs/lightning-migration.md
```
Expected: the three flags appear in both `config.py` and the new doc.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/benchmark-conditions.md docs/lightning-migration.md
git commit -m "docs(v003): Lightning migration + train-time gate documentation"
```

---

## Final steps (after all tasks)

- [ ] Run the full suite once more: `python3.12 -m unittest discover -t . -s tslib -p 'test_*.py'` — expect all green.
- [ ] Whole-branch review (superpowers:requesting-code-review), then superpowers:finishing-a-development-branch to merge `feat/lightning-restructure` → `main` and push (continues the GOES-time-series remote).
