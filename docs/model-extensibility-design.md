# Model-extensibility code structure — design

**Date:** 2026-07-01
**Repo:** `workdir/sw-framework-v002/` (branch off `master`)
**Status:** approved, pre-implementation

## 1. Goal

Move the framework into its modeling phase with a code structure built for
extension along the two axes the user prioritized:

1. **Pluggable loss & evaluation metrics** — replace the hardcoded `MSELoss`
   and inline `mse`/`mae` with configurable, registry-based losses and
   metrics. Metrics span **regression** (MSE/MAE/RMSE) **and event skill
   scores** (TSS/HSS/POD/FAR) built on threshold-crossing.
2. **New backbones** — replace the implicit `importlib` name convention with an
   explicit **registry + interface contract**, so adding an architecture is
   safe and declared.

**Out of scope (YAGNI):** new task heads (e.g. classification), input
embeddings / real time-features (time-marks stay zero), config-file/Hydra
systems, per-channel distinct transforms, missing-value options 2/3 (Option 1
is fixed — see `docs/`).

## 2. Chosen approach

**Approach A — three independent registries + thin factories.** Each axis is a
small, self-contained module with a `name → callable` registry and a decorator.
`pl_model` and `exp.py` only *assemble* these from config; they hold no
model/loss/metric logic. Rejected: a single combined `registry.py` (mixes
concerns, grows large) and a Hydra-style config-object system (over-engineered,
departs from the existing argparse + flat-namespace style).

Rationale: matches the existing per-file-module + argparse structure, isolates
the three axes so each is understood and tested independently, and keeps the
change small.

## 3. Component 1 — backbone registry (`src/model/registry.py`)

```python
@dataclass
class ModelSpec:
    ctor: Callable            # config -> base nn.Module
    adapter: str = "standard" # which adapter wraps it

MODEL_REGISTRY: dict[str, ModelSpec] = {}

def register_model(name, *, adapter="standard"):
    def deco(ctor):
        MODEL_REGISTRY[name] = ModelSpec(ctor, adapter)
        return ctor
    return deco
```

**Interface contract (documented protocol):**
- `adapter="standard"` backbone: `Model(config)` constructor +
  `forward(x_enc, x_mark_enc, x_dec, x_mark_dec) -> (B, pred_len, C)`. All
  current THU-ML models satisfy this.
- Only the `standard` adapter is implemented now. The `adapter` field is the
  polymorphism seam: a future `forward(x)`-only model registers with a new
  adapter kind and a sibling adapter class next to `StandardForecastAdapter`.

**Legacy models are folded in via a central table** (chosen over per-file
decorators): the 15 existing `model/*.py` files are NOT modified.

```python
_LEGACY = ["lstm", "timesnet", "patchtst", "itransformer", "micn",
           "nonstationary_transformer", "patchmixer", "scinet", "segrnn",
           "segrnn_thuml", "timemixer", "etsformer", "xpatch"]  # candidate set
for n in _LEGACY:
    register_model(n)(lambda cfg, _n=n: importlib.import_module(f"model.{_n}").Model(cfg))
```
The exact list is finalized at implementation time by an importability check
(models that fail to import cleanly — e.g. `scinet_old`, `_xpatch_rev`, or any
needing unavailable deps — are excluded and logged). Adding a new backbone =
write `model/newnet.py` + add `"newnet"` to `_LEGACY` (or register a custom
ctor with an explicit spec).

**`build_model` change:** look up `MODEL_REGISTRY[name]` instead of
`importlib`+`getattr`; instantiate `spec.ctor(config)`; wrap per `spec.adapter`.
Unregistered name → clear error listing registered keys. The existing
`StandardForecastAdapter` (zero time-marks/x_dec, `pred[..., target_indices]`
target selection) is unchanged.

## 4. Component 2 — loss registry (`src/exp/losses.py`)

```python
LOSS_REGISTRY = {}                 # name -> callable(pred, true) -> scalar tensor
def register_loss(name): ...       # @register_loss("mse")
def build_loss(config): return LOSS_REGISTRY[config.loss]
```
- Built-ins: `mse` (default, current behavior), `mae`, `huber`,
  `weighted_mse` (up-weights high/rare values for SEP events).
- Computed in the **log10-transformed space** (pred/true both log10), matching
  current behavior.
- `pl_model` uses `build_loss(config)` instead of `nn.MSELoss()`.

## 5. Component 3 — metrics registry (`src/exp/metrics.py`)

```python
METRIC_REGISTRY = {}   # name -> Metric(fn, kind)   kind ∈ {"regression","event"}
def register_metric(name, kind="regression"): ...

@dataclass
class MetricContext:
    thresholds: list[float] | None   # physical-unit thresholds, aligned to target_cols
    transform: str                   # e.g. "log10" — how pred/true are encoded
    target_cols: list[str]

def metric(pred, true, ctx) -> float | dict   # unified signature
```
- **Regression metrics** (`mse`, `mae`, `rmse`): ignore `ctx`, reduce
  `(pred, true)` error to a scalar.
- **Event metrics** (`tss`, `hss`, `pod`, `far`): binarize pred/true by
  threshold-crossing (value ≥ threshold → event), build a confusion matrix,
  compute the skill score.

**Threshold handling (decided):**
- `--event_threshold` is given in **physical units** (e.g. proton `10` pfu).
  Because `metrics.py` receives `ctx.transform`, it **converts the threshold to
  the pred/true space automatically** (log10). Users think in physical units.
- **Multi-target:** `--event_threshold` is a list aligned to `--target_cols`
  order (e.g. `--event_threshold 10 1e-5`). Event metrics are computed **per
  target channel** and reported **per channel** (not averaged).

## 6. Component 4 — wiring & config

**`pl_model` (`src/exp/lightning_model.py`):**
- `self.criterion = build_loss(config)`.
- `evaluate()` iterates the configured metric list via `metrics.py`. Regression
  metrics → scalar; event metrics → **per-target-channel** dict. Builds a
  `MetricContext` from `DataBundle` (target_cols) + config (thresholds,
  transform). The train/eval loop stays agnostic to metric count/kind.

**`exp.py` — dynamic comparison table:**
- Row becomes `{model, best_val_loss, **metrics_dict}`; `metrics_dict` holds
  dynamic columns: `mse`, `mae`, and per-channel `tss_p_gt10`, `hss_p_gt10`,
  `tss_xrs_long`, … (single-target → no channel suffix). Sort by
  `--sort_metric` (default `val_loss`). Missing columns → NaN allowed.

**`config.py` new flags:**

| flag | default | meaning |
|---|---|---|
| `--loss` | `mse` | `LOSS_REGISTRY` key |
| `--metrics` | `["mse","mae"]` | metric keys to report |
| `--event_threshold` | `None` | physical-unit thresholds aligned to `--target_cols` |
| `--sort_metric` | `val_loss` | comparison-table sort key |

`config_postprocess`: if any event metric is in `--metrics` but
`--event_threshold` is missing or its length ≠ number of target columns → clear
error.

## 7. Module boundaries

- `model/registry.py` — name → backbone spec + adapter selection. In: config.
  Out: `nn.Module`.
- `exp/losses.py` — name → loss callable. In: pred/true. Out: scalar.
- `exp/metrics.py` — name → metric callable + `MetricContext`. In: pred/true/ctx.
  Out: float / per-channel dict.
- `pl_model` / `exp.py` — assemble the three from config; no model/loss/metric
  logic of their own.

## 8. Testing

- Registries: register/lookup round-trip, unregistered-name error, `build_model`
  routes through registry, existing `test_lstm.py` passes via the new path.
- Loss: numeric check on known inputs + backward for each built-in.
- Metrics: regression numeric checks; event metrics validated against a
  hand-built confusion matrix (TSS/HSS/POD/FAR ground truth); physical→log10
  threshold conversion; multi-target per-channel separation.
- Integration: `exp.py` produces a dynamic comparison table including event
  metrics (small synthetic or small real-parquet run).
- Regression: existing 39 tests (loader 29 / model 4 / config 6) + legacy
  symbol imports remain green.

## 9. Out of scope / follow-on

New adapter kinds beyond `standard`, task heads (classification/probabilistic),
input embeddings & real time-features, custom optimizers/schedulers (Adam stays
for now), per-channel losses. Each is a clean future extension against these
seams.
