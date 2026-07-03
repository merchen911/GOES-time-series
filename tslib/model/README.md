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
