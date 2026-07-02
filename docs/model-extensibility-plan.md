# Model-extensibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make model backbones, loss functions, and evaluation metrics pluggable via three independent registries, wired through the training loop and comparison table.

**Architecture:** Three `name → callable` registries with decorators — `model/registry.py` (backbones + adapter selection), `exp/losses.py` (loss functions), `exp/metrics.py` (regression + threshold-crossing event metrics). `pl_model` and `exp.py` only assemble these from config; they hold no model/loss/metric logic.

**Tech Stack:** Python 3.12, PyTorch, NumPy, pandas; `unittest` (no pytest).

## Global Constraints

- Run tests from `src/` with `PYTHONPATH=.`: `cd src && PYTHONPATH=. python3.12 -m unittest <module> -v`. Use `python3.12` (has torch/numpy/pandas); plain `python3` does NOT.
- Real-data tests are `@unittest.skipUnless` gated and read `SW_DATA_DIR` (data is shared separately, not in the repo).
- **Default behavior must not change:** `--loss mse`, `--metrics mse mae`; existing tests (loader 29, model 4, config 6) and legacy symbols stay green.
- Loss and metrics operate in the **stored (log10) space**; event-metric thresholds are given in **physical units** and converted via `config.transform`.
- Registries raise a clear `ValueError` listing registered keys on an unknown name.
- Backbones are registered via a **central table** in `registry.py`; the existing `model/*.py` files are NOT edited.
- TDD: write the failing test first. Commit after each task. Co-author trailer on every commit: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## File structure

- Create `src/model/registry.py` — backbone registry, `ModelSpec`, `register_model`, central `_LEGACY` table.
- Modify `src/model/__init__.py` — `build_model` looks up the registry (was `importlib`+`getattr`).
- Create `src/exp/losses.py` — `LOSS_REGISTRY`, `register_loss`, built-ins, `build_loss`.
- Create `src/exp/metrics.py` — `METRIC_REGISTRY`, `register_metric`, `MetricContext`, regression + event metrics.
- Modify `src/configs/config.py` — `--loss/--metrics/--event_threshold/--sort_metric` + validation.
- Modify `src/exp/lightning_model.py` — `build_loss`, metric-driven `evaluate`, generalized `TrainResult`.
- Modify `src/data/loader.py` — `DataBundle.target_cols` (for per-channel metric naming).
- Modify `src/exp/exp.py` — dynamic comparison table via a pure `build_comparison` helper.

---

### Task 1: Backbone registry + `build_model` rewrite

**Files:**
- Create: `src/model/registry.py`
- Modify: `src/model/__init__.py`
- Test: `src/model/test_registry.py`

**Interfaces:**
- Produces:
  - `ModelSpec(ctor: Callable, adapter: str = "standard")` dataclass.
  - `MODEL_REGISTRY: dict[str, ModelSpec]`.
  - `register_model(name, *, adapter="standard") -> decorator` (decorates a `ctor(config) -> nn.Module`).
  - `build_model(model_name, config, input_size, target_indices) -> nn.Module` — now registry-backed; unchanged signature and `StandardForecastAdapter` wrapping.

- [ ] **Step 1: Write the failing test**

Create `src/model/test_registry.py`:

```python
import unittest
from types import SimpleNamespace

import torch

from model.registry import MODEL_REGISTRY, register_model
from model import build_model


class TestModelRegistry(unittest.TestCase):
    def test_legacy_models_registered(self):
        for n in ["lstm", "timesnet", "patchtst"]:
            self.assertIn(n, MODEL_REGISTRY)

    def test_register_and_lookup(self):
        @register_model("dummy_reg_model")
        def ctor(cfg):
            return torch.nn.Identity()
        self.assertIn("dummy_reg_model", MODEL_REGISTRY)
        self.assertEqual(MODEL_REGISTRY["dummy_reg_model"].adapter, "standard")

    def test_unknown_model_errors(self):
        with self.assertRaises(ValueError):
            build_model("no_such_model", SimpleNamespace(), input_size=1,
                        target_indices=[0])

    def test_build_lstm_routes_through_registry(self):
        cfg = SimpleNamespace(task_name="long_term_forecast", label_len=0,
                              seq_len=24, pred_len=1, d_model=16,
                              num_layers=2, dropout=0.1)
        m = build_model("lstm", cfg, input_size=2, target_indices=[0])
        self.assertEqual(tuple(m(torch.randn(2, 24, 2)).shape), (2, 1, 1))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src && PYTHONPATH=. python3.12 -m unittest model.test_registry -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'model.registry'`).

- [ ] **Step 3: Create the registry**

Create `src/model/registry.py`:

```python
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, Dict


@dataclass
class ModelSpec:
    ctor: Callable          # config -> base nn.Module
    adapter: str = "standard"


MODEL_REGISTRY: Dict[str, ModelSpec] = {}


def register_model(name, *, adapter="standard"):
    def deco(ctor):
        MODEL_REGISTRY[name] = ModelSpec(ctor=ctor, adapter=adapter)
        return ctor
    return deco


# Central table: register existing THU-ML-style backbones WITHOUT editing their
# files. The ctor lazily imports the module so registering does not import torch
# models until one is actually built.
_LEGACY = ["lstm", "timesnet", "patchtst", "itransformer", "micn",
           "nonstationary_transformer", "patchmixer", "scinet", "segrnn",
           "segrnn_thuml", "timemixer", "etsformer", "xpatch"]


def _legacy_ctor(name):
    return lambda cfg, _n=name: importlib.import_module(f"model.{_n}").Model(cfg)


for _n in _LEGACY:
    register_model(_n)(_legacy_ctor(_n))
```

- [ ] **Step 4: Rewrite `build_model` to use the registry**

In `src/model/__init__.py`, replace the `build_model` function (the current
`importlib`+`getattr` version) with:

```python
def build_model(model_name: str, config, input_size: int, target_indices):
    from model.registry import MODEL_REGISTRY
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"unknown model '{model_name}'; registered: {sorted(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[model_name]
    # legacy-style configs
    config.model = model_name
    config.enc_in = input_size
    config.dec_in = input_size
    config.c_out = input_size
    base_model = spec.ctor(config)
    if spec.adapter == "standard":
        return StandardForecastAdapter(base_model=base_model, config=config,
                                       target_indices=target_indices)
    raise ValueError(f"unknown adapter '{spec.adapter}'")
```

Leave the `import importlib` line and `StandardForecastAdapter` class as-is.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd src && PYTHONPATH=. python3.12 -m unittest model.test_registry model.test_lstm -v`
Expected: PASS (registry tests + the existing LSTM tests still route through `build_model`).

- [ ] **Step 6: Commit**

```bash
cd /NAS/ioGuard3/vol3/spaceai/SW_framework/workdir/sw-framework-v002
git add src/model/registry.py src/model/__init__.py src/model/test_registry.py
git commit -m "feat(model): backbone registry + registry-backed build_model

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Loss registry

**Files:**
- Create: `src/exp/losses.py`
- Test: `src/exp/test_losses.py`

**Interfaces:**
- Produces:
  - `LOSS_REGISTRY: dict[str, Callable]` — `callable(pred, true) -> scalar tensor`.
  - `register_loss(name) -> decorator`.
  - `build_loss(config) -> Callable` — reads `config.loss` (default `"mse"`); raises on unknown.
  - Built-ins: `mse`, `mae`, `huber`, `weighted_mse`.

- [ ] **Step 1: Write the failing test**

Create `src/exp/test_losses.py`:

```python
import unittest
from types import SimpleNamespace

import torch

from exp.losses import LOSS_REGISTRY, build_loss


class TestLosses(unittest.TestCase):
    def test_registry_has_builtins(self):
        for n in ["mse", "mae", "huber", "weighted_mse"]:
            self.assertIn(n, LOSS_REGISTRY)

    def test_default_mse_value(self):
        loss = build_loss(SimpleNamespace(loss="mse"))
        pred = torch.zeros(4, 3, 1)
        true = torch.ones(4, 3, 1)
        self.assertAlmostEqual(float(loss(pred, true)), 1.0, places=6)

    def test_unknown_loss_errors(self):
        with self.assertRaises(ValueError):
            build_loss(SimpleNamespace(loss="nope"))

    def test_weighted_mse_backprops(self):
        loss = build_loss(SimpleNamespace(loss="weighted_mse"))
        pred = torch.zeros(2, 2, 1, requires_grad=True)
        true = torch.ones(2, 2, 1)
        loss(pred, true).backward()
        self.assertIsNotNone(pred.grad)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src && PYTHONPATH=. python3.12 -m unittest exp.test_losses -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'exp.losses'`).

- [ ] **Step 3: Implement**

Create `src/exp/losses.py`:

```python
from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn.functional as F

LOSS_REGISTRY: Dict[str, Callable] = {}


def register_loss(name):
    def deco(fn):
        LOSS_REGISTRY[name] = fn
        return fn
    return deco


@register_loss("mse")
def _mse(pred, true):
    return F.mse_loss(pred, true)


@register_loss("mae")
def _mae(pred, true):
    return F.l1_loss(pred, true)


@register_loss("huber")
def _huber(pred, true):
    return F.huber_loss(pred, true)


@register_loss("weighted_mse")
def _weighted_mse(pred, true):
    # Up-weight high-magnitude (rare/energetic) targets. Data is log10-flux, so
    # larger `true` = larger flux; weight grows with positive log10 magnitude.
    # Simple, tunable heuristic (not learned).
    w = 1.0 + true.clamp(min=0.0)
    return (w * (pred - true) ** 2).mean()


def build_loss(config):
    name = getattr(config, "loss", "mse")
    if name not in LOSS_REGISTRY:
        raise ValueError(f"unknown loss '{name}'; registered: {sorted(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[name]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd src && PYTHONPATH=. python3.12 -m unittest exp.test_losses -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
cd /NAS/ioGuard3/vol3/spaceai/SW_framework/workdir/sw-framework-v002
git add src/exp/losses.py src/exp/test_losses.py
git commit -m "feat(exp): pluggable loss registry (mse/mae/huber/weighted_mse)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Metrics registry — regression metrics

**Files:**
- Create: `src/exp/metrics.py`
- Test: `src/exp/test_metrics.py`

**Interfaces:**
- Produces:
  - `MetricContext(thresholds: list[float] | None = None, transform: str = "none", target_cols: list[str] = [])` dataclass.
  - `Metric(fn: Callable, kind: str)` dataclass; `kind ∈ {"regression","event"}`.
  - `METRIC_REGISTRY: dict[str, Metric]`.
  - `register_metric(name, kind="regression") -> decorator`; wrapped fn signature `fn(pred, true, ctx) -> float | dict`. `pred`/`true` are NumPy arrays shaped `(N, pred_len, T)`.
  - Regression built-ins: `mse`, `mae`, `rmse`.

- [ ] **Step 1: Write the failing test**

Create `src/exp/test_metrics.py`:

```python
import unittest

import numpy as np

from exp.metrics import METRIC_REGISTRY, MetricContext


class TestRegressionMetrics(unittest.TestCase):
    def test_builtins_registered(self):
        for n in ["mse", "mae", "rmse"]:
            self.assertIn(n, METRIC_REGISTRY)
            self.assertEqual(METRIC_REGISTRY[n].kind, "regression")

    def test_regression_values(self):
        pred = np.zeros((4, 2, 1))
        true = np.ones((4, 2, 1))
        ctx = MetricContext()
        self.assertAlmostEqual(METRIC_REGISTRY["mse"].fn(pred, true, ctx), 1.0)
        self.assertAlmostEqual(METRIC_REGISTRY["mae"].fn(pred, true, ctx), 1.0)
        self.assertAlmostEqual(METRIC_REGISTRY["rmse"].fn(pred, true, ctx), 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src && PYTHONPATH=. python3.12 -m unittest exp.test_metrics -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'exp.metrics'`).

- [ ] **Step 3: Implement (regression only for now)**

Create `src/exp/metrics.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np


@dataclass
class MetricContext:
    thresholds: Optional[List[float]] = None   # physical units, aligned to target_cols
    transform: str = "none"
    target_cols: List[str] = field(default_factory=list)


@dataclass
class Metric:
    fn: Callable
    kind: str          # "regression" | "event"


METRIC_REGISTRY: Dict[str, Metric] = {}


def register_metric(name, kind="regression"):
    def deco(fn):
        METRIC_REGISTRY[name] = Metric(fn=fn, kind=kind)
        return fn
    return deco


@register_metric("mse", "regression")
def _mse(pred, true, ctx):
    return float(np.mean((pred - true) ** 2))


@register_metric("mae", "regression")
def _mae(pred, true, ctx):
    return float(np.mean(np.abs(pred - true)))


@register_metric("rmse", "regression")
def _rmse(pred, true, ctx):
    return float(np.sqrt(np.mean((pred - true) ** 2)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd src && PYTHONPATH=. python3.12 -m unittest exp.test_metrics -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
cd /NAS/ioGuard3/vol3/spaceai/SW_framework/workdir/sw-framework-v002
git add src/exp/metrics.py src/exp/test_metrics.py
git commit -m "feat(exp): metric registry + regression metrics (mse/mae/rmse)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Event skill-score metrics + threshold conversion

**Files:**
- Modify: `src/exp/metrics.py`
- Test: `src/exp/test_metrics.py` (append)

**Interfaces:**
- Consumes: `register_metric`, `MetricContext` (Task 3).
- Produces: event metrics `tss`, `hss`, `pod`, `far` (`kind="event"`). Each returns a `dict[str, float]` keyed by target-channel name (from `ctx.target_cols`, else `c{i}`). Thresholds in `ctx.thresholds` are physical-unit values converted to the stored space via `ctx.transform`.

- [ ] **Step 1: Write the failing test**

Append to `src/exp/test_metrics.py`:

```python
class TestEventMetrics(unittest.TestCase):
    def _data(self):
        # log10 space; threshold physical=10 -> log10=1.0. Values above/below 1.0
        # give tp=1, fp=1, fn=1, tn=1 for a single channel over 4 samples.
        pred = np.array([2.0, 0.0, 2.0, 0.0]).reshape(4, 1, 1)
        true = np.array([2.0, 2.0, 0.0, 0.0]).reshape(4, 1, 1)
        ctx = MetricContext(thresholds=[10.0], transform="log10",
                            target_cols=["p_gt10"])
        return pred, true, ctx

    def test_event_registered_as_event(self):
        for n in ["tss", "hss", "pod", "far"]:
            self.assertIn(n, METRIC_REGISTRY)
            self.assertEqual(METRIC_REGISTRY[n].kind, "event")

    def test_skill_scores_and_per_channel_keys(self):
        pred, true, ctx = self._data()
        tss = METRIC_REGISTRY["tss"].fn(pred, true, ctx)
        hss = METRIC_REGISTRY["hss"].fn(pred, true, ctx)
        pod = METRIC_REGISTRY["pod"].fn(pred, true, ctx)
        far = METRIC_REGISTRY["far"].fn(pred, true, ctx)
        # per-channel dict keyed by target name
        self.assertEqual(set(tss), {"p_gt10"})
        # tp=fp=fn=tn=1 -> pod=.5, pofd=.5, tss=0 ; hss=0 ; far=.5
        self.assertAlmostEqual(tss["p_gt10"], 0.0, places=6)
        self.assertAlmostEqual(hss["p_gt10"], 0.0, places=6)
        self.assertAlmostEqual(pod["p_gt10"], 0.5, places=6)
        self.assertAlmostEqual(far["p_gt10"], 0.5, places=6)

    def test_missing_threshold_errors(self):
        pred = np.zeros((2, 1, 1)); true = np.zeros((2, 1, 1))
        with self.assertRaises(ValueError):
            METRIC_REGISTRY["tss"].fn(pred, true, MetricContext())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src && PYTHONPATH=. python3.12 -m unittest exp.test_metrics.TestEventMetrics -v`
Expected: FAIL (`KeyError: 'tss'`).

- [ ] **Step 3: Implement — append to `src/exp/metrics.py`**

```python
def _to_stored_space(threshold, transform):
    if transform == "log10":
        return float(np.log10(threshold))
    if transform == "none":
        return float(threshold)
    raise ValueError(f"unknown transform: {transform}")


def _confusion(pred_ch, true_ch, thr):
    p = pred_ch >= thr
    t = true_ch >= thr
    tp = int(np.sum(p & t))
    fp = int(np.sum(p & ~t))
    fn = int(np.sum(~p & t))
    tn = int(np.sum(~p & ~t))
    return tp, fp, fn, tn


def _event_metric(score_fn):
    def fn(pred, true, ctx):
        if not ctx.thresholds:
            raise ValueError("event metric requires ctx.thresholds (physical units)")
        out = {}
        n_targets = pred.shape[-1]
        for i in range(n_targets):
            thr = _to_stored_space(ctx.thresholds[i], ctx.transform)
            tp, fp, fn_, tn = _confusion(pred[..., i], true[..., i], thr)
            name = ctx.target_cols[i] if i < len(ctx.target_cols) else f"c{i}"
            out[name] = score_fn(tp, fp, fn_, tn)
        return out
    return fn


def _tss(tp, fp, fn, tn):
    pod = tp / (tp + fn) if (tp + fn) else 0.0
    pofd = fp / (fp + tn) if (fp + tn) else 0.0
    return pod - pofd


def _hss(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    if n == 0:
        return 0.0
    exp = ((tp + fn) * (tp + fp) + (tn + fn) * (tn + fp)) / n
    denom = n - exp
    return (tp + tn - exp) / denom if denom else 0.0


def _pod(tp, fp, fn, tn):
    return tp / (tp + fn) if (tp + fn) else 0.0


def _far(tp, fp, fn, tn):
    return fp / (tp + fp) if (tp + fp) else 0.0


register_metric("tss", "event")(_event_metric(_tss))
register_metric("hss", "event")(_event_metric(_hss))
register_metric("pod", "event")(_event_metric(_pod))
register_metric("far", "event")(_event_metric(_far))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src && PYTHONPATH=. python3.12 -m unittest exp.test_metrics -v`
Expected: PASS (regression + event tests).

- [ ] **Step 5: Commit**

```bash
cd /NAS/ioGuard3/vol3/spaceai/SW_framework/workdir/sw-framework-v002
git add src/exp/metrics.py src/exp/test_metrics.py
git commit -m "feat(exp): event skill-score metrics (tss/hss/pod/far) with physical-unit thresholds

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Config flags + validation

**Files:**
- Modify: `src/configs/config.py`
- Test: `src/configs/test_config.py` (append)

**Interfaces:**
- Consumes: `METRIC_REGISTRY` (Task 3/4).
- Produces: config attributes `loss: str="mse"`, `metrics: list[str]=["mse","mae"]`, `event_threshold: list[float]|None=None`, `sort_metric: str="best_val_loss"`. `config_postprocess` rejects unknown metric names and event metrics without correctly-sized thresholds.

- [ ] **Step 1: Write the failing test**

Append to `src/configs/test_config.py` (uses the file's existing parse helper; if none, use the pattern below):

```python
import unittest
from configs.config import exp_parser, config_postprocess


def _parse(*extra):
    argv = ["--data_path", "x.parquet", "--target_col", "p_gt10", *extra]
    return config_postprocess(exp_parser().parse_args(argv))


class TestModelingFlags(unittest.TestCase):
    def test_defaults(self):
        c = _parse()
        self.assertEqual(c.loss, "mse")
        self.assertEqual(c.metrics, ["mse", "mae"])
        self.assertEqual(c.sort_metric, "best_val_loss")

    def test_unknown_metric_errors(self):
        with self.assertRaises(ValueError):
            _parse("--metrics", "mse", "bogus")

    def test_event_metric_requires_threshold(self):
        with self.assertRaises(ValueError):
            _parse("--metrics", "tss")

    def test_event_threshold_length_must_match_targets(self):
        # two targets but one threshold -> error
        with self.assertRaises(ValueError):
            _parse("--channels", "a.parquet:p_gt10", "b.parquet:xrs_long",
                   "--target_cols", "p_gt10", "xrs_long",
                   "--metrics", "tss", "--event_threshold", "10")

    def test_event_metric_ok_with_matching_threshold(self):
        c = _parse("--metrics", "mse", "tss", "--event_threshold", "10")
        self.assertEqual(c.event_threshold, [10.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src && PYTHONPATH=. python3.12 -m unittest configs.test_config.TestModelingFlags -v`
Expected: FAIL (`AttributeError: 'Namespace' object has no attribute 'loss'`).

- [ ] **Step 3: Add the flags**

In `src/configs/config.py`, inside `exp_parser()` (after the existing model
flags, before `return parser`), add:

```python
    # modeling: pluggable loss / metrics
    parser.add_argument("--loss", type=str, default="mse")
    parser.add_argument("--metrics", nargs="+", default=["mse", "mae"])
    parser.add_argument("--event_threshold", nargs="*", type=float, default=None)
    parser.add_argument("--sort_metric", type=str, default="best_val_loss")
```

- [ ] **Step 4: Add validation in `config_postprocess`**

In `src/configs/config.py`, insert before the final `return config` of
`config_postprocess`:

```python
    from exp.metrics import METRIC_REGISTRY
    for m in config.metrics:
        if m not in METRIC_REGISTRY:
            raise ValueError(
                f"unknown metric '{m}'; registered: {sorted(METRIC_REGISTRY)}")
    event = [m for m in config.metrics
             if METRIC_REGISTRY[m].kind == "event"]
    if event:
        n_targets = (len(config.target_cols)
                     if getattr(config, "target_cols", None) else 1)
        if not config.event_threshold:
            raise ValueError(f"event metrics {event} require --event_threshold")
        if len(config.event_threshold) != n_targets:
            raise ValueError(
                f"--event_threshold length {len(config.event_threshold)} "
                f"!= number of targets {n_targets}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd src && PYTHONPATH=. python3.12 -m unittest configs.test_config -v`
Expected: PASS (new modeling-flag tests + existing config tests).

- [ ] **Step 6: Commit**

```bash
cd /NAS/ioGuard3/vol3/spaceai/SW_framework/workdir/sw-framework-v002
git add src/configs/config.py src/configs/test_config.py
git commit -m "feat(config): --loss/--metrics/--event_threshold/--sort_metric + validation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Wire loss + metrics into `pl_model`; add `DataBundle.target_cols`

**Files:**
- Modify: `src/data/loader.py` (`DataBundle` + `_setup_parquet`)
- Modify: `src/exp/lightning_model.py`
- Test: `src/data/test_loader.py` (append), `src/exp/test_lightning_model.py` (create)

**Interfaces:**
- Consumes: `build_loss` (Task 2), `METRIC_REGISTRY`/`MetricContext` (Tasks 3-4).
- Produces:
  - `DataBundle.target_cols: list[str]` (default `[]`); populated by `_setup_parquet` with the resolved target column names.
  - `TrainResult(model_name: str, best_val_loss: float, metrics: dict[str,float], ckpt_path: str)`.
  - `pl_model.evaluate(loader, ctx) -> dict[str,float]` — runs `config.metrics`; regression → `{name: value}`, event → `{f"{name}_{channel}": value}`.

- [ ] **Step 1: Write the failing test (DataBundle.target_cols)**

Append to `src/data/test_loader.py`:

```python
class TestDataBundleTargetCols(unittest.TestCase):
    def test_target_cols_default_empty(self):
        from data.loader import DataBundle
        b = DataBundle(train_loader=None, val_loader=None, test_loader=None,
                       input_size=1, target_index=0)
        self.assertEqual(b.target_cols, [])
```

Also add the real-data assertion to the existing `TestSetupMultivar.test_two_channels_multi_target` (append inside it):

```python
        self.assertEqual(b.target_cols, ["p_gt10", "xrs_long"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src && PYTHONPATH=. python3.12 -m unittest data.test_loader.TestDataBundleTargetCols -v`
Expected: FAIL (`TypeError: __init__() got an unexpected keyword` is NOT raised, but `AttributeError: 'DataBundle' object has no attribute 'target_cols'`).

- [ ] **Step 3: Add `target_cols` to `DataBundle` and populate it**

In `src/data/loader.py`, add a field to the `DataBundle` dataclass (after
`target_indices`):

```python
    target_cols: List[str] = field(default_factory=list)
```

In `_setup_parquet`, change the final `return DataBundle(...)` to also pass
`target_cols=target_cols`:

```python
        return DataBundle(
            train_loader=loaders["train"], val_loader=loaders["val"],
            test_loader=loaders["test"], input_size=len(cols),
            target_index=target_idx[0], output_size=len(target_cols),
            target_indices=target_idx, target_cols=target_cols)
```

- [ ] **Step 4: Run the DataBundle test to verify it passes**

Run: `cd src && PYTHONPATH=. python3.12 -m unittest data.test_loader.TestDataBundleTargetCols -v`
Expected: PASS. (Real-data `TestSetupMultivar` passes when `SW_DATA_DIR` is set; otherwise it skips.)

- [ ] **Step 5: Write the failing test (pl_model metric wiring)**

Create `src/exp/test_lightning_model.py`:

```python
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from exp.lightning_model import pl_model
from exp.metrics import MetricContext


def _cfg(**kw):
    base = dict(lr=1e-3, weight_decay=0.0, epochs=1, loss="mse",
                metrics=["mse", "mae"], transform="log10",
                event_threshold=None, seq_len=4, pred_len=2, label_len=0)
    base.update(kw)
    return SimpleNamespace(**base)


class _ConstModel(torch.nn.Module):
    def __init__(self, pred_len, t):
        super().__init__()
        self.pred_len, self.t = pred_len, t
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b = x.shape[0]
        return torch.zeros(b, self.pred_len, self.t) + self.p


class TestPlModelMetrics(unittest.TestCase):
    def test_evaluate_regression_and_event(self):
        runner = pl_model(_ConstModel(2, 1),
                          _cfg(metrics=["mse", "tss"], event_threshold=[10.0]))
        # loader yields (x, y): pred is ~0, true=2.0 (log10 space, > log10(10)=1)
        x = torch.zeros(4, 4, 1)
        y = torch.full((4, 2, 1), 2.0)
        loader = [(x, y)]
        ctx = MetricContext(thresholds=[10.0], transform="log10",
                            target_cols=["p_gt10"])
        out = runner.evaluate(loader, ctx)
        self.assertIn("mse", out)            # regression scalar
        self.assertIn("tss_p_gt10", out)     # event, per-channel key
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd src && PYTHONPATH=. python3.12 -m unittest exp.test_lightning_model -v`
Expected: FAIL (`evaluate()` takes 2 positional args / returns only mse+mae).

- [ ] **Step 7: Rewire `pl_model`**

In `src/exp/lightning_model.py`:

Replace the `TrainResult` dataclass with:

```python
@dataclass
class TrainResult:
    model_name: str
    best_val_loss: float
    metrics: Dict[str, float]
    ckpt_path: str
```

Add imports near the top (after existing imports):

```python
from exp.losses import build_loss
from exp.metrics import METRIC_REGISTRY, MetricContext
```

In `__init__`, replace `self.criterion = nn.MSELoss()` with:

```python
        self.criterion = build_loss(config)
```

Replace the `evaluate` method with:

```python
    @torch.no_grad()
    def evaluate(self, loader, ctx) -> Dict[str, float]:
        self.model.eval()
        all_pred, all_true = [], []
        for x, y in loader:
            pred = self.model(x.to(self.device)).cpu().numpy()
            all_pred.append(pred)
            all_true.append(np.asarray(y))
        pred = np.concatenate(all_pred, axis=0)
        true = np.concatenate(all_true, axis=0)
        out: Dict[str, float] = {}
        for name in self.config.metrics:
            val = METRIC_REGISTRY[name].fn(pred, true, ctx)
            if isinstance(val, dict):
                for ch, v in val.items():
                    out[f"{name}_{ch}"] = float(v)
            else:
                out[name] = float(val)
        return out
```

Replace `fit_and_test` with:

```python
    def fit_and_test(self, datamodule, model_name: str, ckpt_path: str) -> TrainResult:
        best_val = float("inf")
        for _ in range(self.config.epochs):
            self._run_epoch(datamodule.train_loader, train=True)
            val_loss = self._run_epoch(datamodule.val_loader, train=False)
            if val_loss < best_val:
                best_val = val_loss
                torch.save(self.model.state_dict(), ckpt_path)

        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        ctx = MetricContext(
            thresholds=getattr(self.config, "event_threshold", None),
            transform=getattr(self.config, "transform", "none"),
            target_cols=list(getattr(datamodule, "target_cols", []) or []))
        metrics = self.evaluate(datamodule.test_loader, ctx)
        return TrainResult(
            model_name=model_name,
            best_val_loss=best_val,
            metrics=metrics,
            ckpt_path=ckpt_path,
        )
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd src && PYTHONPATH=. python3.12 -m unittest exp.test_lightning_model data.test_loader.TestDataBundleTargetCols -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
cd /NAS/ioGuard3/vol3/spaceai/SW_framework/workdir/sw-framework-v002
git add src/data/loader.py src/data/test_loader.py src/exp/lightning_model.py src/exp/test_lightning_model.py
git commit -m "feat(exp): wire pluggable loss + metrics into pl_model; DataBundle.target_cols

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Dynamic comparison table in `exp.py`

**Files:**
- Modify: `src/exp/exp.py`
- Test: `src/exp/test_exp.py` (create)

**Interfaces:**
- Consumes: `TrainResult(model_name, best_val_loss, metrics, ckpt_path)` (Task 6).
- Produces: `build_comparison(results: list[TrainResult], sort_metric: str) -> pd.DataFrame` — one row per result with columns `model`, `best_val_loss`, and one column per metric key (dynamic, e.g. `mse`, `tss_p_gt10`); sorts ascending by `sort_metric`, falling back to `best_val_loss` if the column is absent.

- [ ] **Step 1: Write the failing test**

Create `src/exp/test_exp.py`:

```python
import unittest

from exp.exp import build_comparison
from exp.lightning_model import TrainResult


class TestBuildComparison(unittest.TestCase):
    def _results(self):
        return [
            TrainResult("lstm", 0.5, {"mse": 0.30, "tss_p_gt10": 0.7}, "a.pt"),
            TrainResult("timesnet", 0.4, {"mse": 0.20, "tss_p_gt10": 0.8}, "b.pt"),
        ]

    def test_dynamic_columns(self):
        df = build_comparison(self._results(), sort_metric="mse")
        self.assertEqual(set(df.columns),
                         {"model", "best_val_loss", "mse", "tss_p_gt10"})

    def test_sorts_by_metric(self):
        df = build_comparison(self._results(), sort_metric="mse")
        self.assertEqual(list(df["model"]), ["timesnet", "lstm"])  # 0.20 < 0.30

    def test_fallback_when_metric_absent(self):
        df = build_comparison(self._results(), sort_metric="nope")
        self.assertEqual(list(df["model"]), ["timesnet", "lstm"])  # by best_val_loss
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src && PYTHONPATH=. python3.12 -m unittest exp.test_exp -v`
Expected: FAIL (`ImportError: cannot import name 'build_comparison'`).

- [ ] **Step 3: Implement `build_comparison` and use it in `run_experiment`**

In `src/exp/exp.py`, add this function (top-level, after the imports):

```python
def build_comparison(results, sort_metric):
    rows = [{"model": r.model_name, "best_val_loss": r.best_val_loss, **r.metrics}
            for r in results]
    df = pd.DataFrame(rows)
    key = sort_metric if sort_metric in df.columns else "best_val_loss"
    return df.sort_values(by=key).reset_index(drop=True)
```

In `run_experiment`, replace the results-collection + comparison block. Change
the loop body to collect `TrainResult` objects:

```python
    results = []
    for model_name in config.models:
        model = build_model(
            model_name=model_name,
            config=config,
            input_size=data_bundle.input_size,
            target_indices=data_bundle.target_indices,
        )
        runner = pl_model(model, config)
        ckpt_path = os.path.join(dirs["ckpt"], f"{model_name}.pt")
        results.append(runner.fit_and_test(data_bundle, model_name=model_name,
                                           ckpt_path=ckpt_path))

    comparison = build_comparison(results, config.sort_metric)
    comparison_path = os.path.join(dirs["score"], "comparison.csv")
    comparison.to_csv(comparison_path, index=False)
    return comparison, comparison_path
```

(Delete the old `rows = []` / per-row dict / `pd.DataFrame(rows).sort_values(...)`
code that referenced `result.test_mse`/`result.test_mae`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src && PYTHONPATH=. python3.12 -m unittest exp.test_exp -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Full regression**

Run: `cd src && PYTHONPATH=. python3.12 -m unittest discover -s . -p 'test_*.py'`
Expected: PASS. With `SW_DATA_DIR` unset, real-data tests skip; set it to run the full suite.
Also verify legacy symbols: `cd src && PYTHONPATH=. python3.12 -c "from data.loader import DataModule, SequenceDataset, _build_windows; print('legacy OK')"`.

- [ ] **Step 6: Commit**

```bash
cd /NAS/ioGuard3/vol3/spaceai/SW_framework/workdir/sw-framework-v002
git add src/exp/exp.py src/exp/test_exp.py
git commit -m "feat(exp): dynamic comparison table over configured metrics

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-review notes

- **Spec coverage:** backbone registry + central table (Task 1); loss registry (Task 2); regression metrics (Task 3); event skill scores + physical→log10 thresholds + per-channel (Task 4); config flags + validation (Task 5); `pl_model` loss/metric wiring + `MetricContext` + `DataBundle.target_cols` (Task 6); dynamic comparison table (Task 7). Module boundaries match design §7. Out-of-scope items (new adapter kinds, task heads, embeddings, optimizers) are untouched.
- **Type consistency:** `register_model`/`MODEL_REGISTRY`/`ModelSpec`, `build_loss`/`LOSS_REGISTRY`, `register_metric`/`METRIC_REGISTRY`/`Metric`/`MetricContext`, `TrainResult(..., metrics, ...)`, `build_comparison(results, sort_metric)`, `DataBundle.target_cols` are used identically across tasks. `--sort_metric` default is `best_val_loss` (matches the actual column; design's `val_loss` label is realized as `best_val_loss`).
- **Defaults unchanged:** `loss="mse"`, `metrics=["mse","mae"]` reproduce current behavior; existing suites stay green.
