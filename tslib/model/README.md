# tslib.model — adding a backbone

Models are looked up in a **registry** (`registry.py`). `build_model()`
(`__init__.py`) resolves a name to a constructor, instantiates it, and wraps it
in a forecast **adapter** that selects the target channel(s).

## Contract (`adapter="standard"`)

A backbone is a `Model(config)` `nn.Module` whose forward is:

```python
def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
    # returns (B, pred_len, enc_in)
```

`config.enc_in / dec_in / c_out` are set to the number of input channels before
construction. The adapter (`StandardForecastAdapter` in `__init__.py`) builds
the zero time-marks / decoder input and then slices the predicted target
channels — so a backbone just needs to output all `enc_in` channels.

## Steps to add one

1. Write `tslib/model/<name>.py` with a `Model(config)` class meeting the
   contract. Shared building blocks live in [`layers/`](layers/); import them
   relatively, e.g. `from .layers.embed import DataEmbedding`.
2. Register the name in `registry.py` — either add it to the central `_LEGACY`
   list (lazy import), or decorate a custom constructor:
   ```python
   from tslib.model.registry import register_model
   @register_model("mymodel")
   def _build(config): ...
   ```
3. Run it: `--models mymodel`.

A model needing a different call signature (e.g. `forward(x)` only) would add a
new adapter kind alongside `StandardForecastAdapter` and register with
`adapter="<kind>"`. Only `"standard"` exists today.

See `registry.py`, `__init__.py`, and any existing model (e.g. `lstm.py`,
`patchtst.py`) for reference.

## Statistical models (`--forecast_strategy statistic`)

Non-neural, per-window backbones live in `statistical.py` behind their own
registry, separate from the neural `registry.py` above:

```python
from tslib.model.statistical import register_stat
@register_stat("myarima")
class MyModel:
    def fit(self, history): ...      # returns self
    def forecast(self, steps): ...   # returns an array of length `steps`
```

`STAT_REGISTRY` maps a name to the class; `StatisticalRunner`
(`tslib/exp/strategy.py`) constructs it per test window, calls
`fit(history).forecast(pred_len)`, and falls back to persistence
(last observed value repeated) if the fit raises or returns a bad-shape /
non-finite forecast. Built-ins: `arima`, `ar`, `theta` — all backed by
`statsmodels`, an added dependency for this strategy.
