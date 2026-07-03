# tslib

The forecasting framework package. Import as `tslib` after `pip install -e .`
from the repo root.

```python
from tslib.data.loader import DataModule
from tslib.model import build_model
```

## Package map

| Subpackage | Responsibility | Extend it to… |
|---|---|---|
| [`configs/`](configs/) | CLI flags + validation (`exp_parser`, `config_postprocess`) | add a new option |
| [`data/`](data/) | parquet loading, half-year split, common-grid channel join | add / wire a new time series (channel) |
| [`model/`](model/) | backbone registry + forecast adapter | add a model architecture |
| [`exp/`](exp/) | training loop, pluggable **losses** and **metrics**, comparison table | add a loss or an evaluation metric |
| [`preprocessing/`](preprocessing/) | term-split, fold-sample counting, table generation | add an offline analysis/prep script |

Each subpackage has its own `README.md` with the concrete steps. Below is the
one-paragraph version of the three most common extensions; **see the linked
folder for details.**

## Add a model
Drop a `Model(config)` class in `tslib/model/<name>.py` following the adapter
contract, then register the name. → [`model/README.md`](model/)

## Add a time series (new channel / dataset)
If the new series is a parquet with the standard schema
(`time_utc`, `role`, value columns), no code change is needed — just point
`--channels PATH:COL` at it. A different schema needs a small tweak in the
loader. → [`data/README.md`](data/)

## Add a loss or metric
Decorate a function with `@register_loss(...)` or `@register_metric(...)` and
select it with `--loss` / `--metrics`. → [`exp/README.md`](exp/)
