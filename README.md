# GOES-time-series

A PyTorch forecasting framework for GOES space-weather time series (proton flux
and X-ray flux). It provides a leakage-free data pipeline and a pluggable model
layer aimed at short- to medium-horizon forecasting of solar energetic-particle
and flare-related quantities.

**Highlights**
- **Multivariate parquet loading** — join N channels (e.g. proton `p_gt10` +
  X-ray `xrs_long`) onto a common 5-minute grid.
- **Leakage-free cross-validation** — half-year (H1/H2) term tagging with a
  rotating k-fold split, so no time period is shared across train/val/test.
- **Configurable horizons** — independent look-back (`--seq_len`) and forecast
  (`--pred_len`) lengths, in 5-minute steps.
- **Single- or multi-target forecasting** — predict one channel or several at
  once (`--target_cols`).
- **Multiple backbones** — LSTM, TimesNet, PatchTST, iTransformer, MICN, and
  more, run and compared in one command.
- **PyTorch Lightning training** — neural models train under a real
  `pytorch_lightning` `Trainer`, with checkpointing and a train-time gate
  that auto-skips models estimated to run too long (see
  [`docs/lightning-migration.md`](docs/lightning-migration.md)).

> **Data is not included** in this repository and is shared separately. Point
> `SW_DATA_DIR` at your processed-parquet directory before running (see below).

## Installation

The code lives in the importable `tslib` package. Install it editable so
`import tslib` works from any notebook or script:

```bash
pip install -e .        # Python 3.12 recommended (netCDF/parquet libs)
```

Core dependencies (see `requirements.txt` / `pyproject.toml`): `numpy`,
`pandas`, `torch`, `statsmodels`, and `pytorch_lightning` (the neural
training loop runs on `pytorch_lightning`'s `Trainer`/`LightningModule`;
see [`docs/lightning-migration.md`](docs/lightning-migration.md)).

Then, e.g. in a notebook under `notebook/`:

```python
from tslib.data.loader import DataModule
from tslib.model import build_model
```

## Quick start

Set the data location (data is distributed separately):

```bash
export SW_DATA_DIR=/path/to/goes_data/processed
```

Run one or more models and compare them:

```bash
# Univariate: proton p_gt10, 7-day look-back -> 1-day forecast
python main.py --data_path $SW_DATA_DIR/kasi_swpc_particle_5m_v02.parquet \
    --target_col p_gt10 --seq_len 2016 --pred_len 288 \
    --n_fold 5 --fold_numb 0 --models lstm patchtst

# Multivariate input -> single target (proton + X-ray in, proton out)
python main.py \
    --channels $SW_DATA_DIR/kasi_swpc_particle_5m_v02.parquet:p_gt10 \
               $SW_DATA_DIR/kasi_swpc_xray_1m_v02.parquet:xrs_long \
    --target_cols p_gt10 --seq_len 864 --pred_len 288 --models lstm patchtst

# Multivariate input -> multi-target (forecast both channels)
python main.py --channels $SW_DATA_DIR/...:p_gt10 $SW_DATA_DIR/...:xrs_long \
    --target_cols p_gt10 xrs_long --seq_len 864 --pred_len 288 --models lstm
```

## Key options

| Option | Default | Description |
|---|---|---|
| `--data_path` / `--target_col` | (required) | Single channel; synthesized into one channel when `--channels` is absent |
| `--channels PATH:COL [...]` | `None` | Input channels; one channel = univariate input |
| `--target_cols COL [...]` | `[first channel col]` | Forecast targets (subset of channels); 1 = univariate output, ≥2 = multivariate |
| `--split_type` | `year_half` | `year_half` / `year` / `ratio` |
| `--n_fold` / `--fold_numb` | `5` / `0` | Rotating k-fold count and fold index (0..n_fold-1) |
| `--seq_len` / `--pred_len` | `24` / `1` | Look-back / forecast length. **Parquet uses 5-min steps** (1 day = 288, 7 days = 2016) |
| `--cadence_min` | `5` | Common-grid spacing (minutes) |
| `--min_bin_count` | `1` | Min native samples for a 5-min bin to be valid |
| `--transform` | `log10` | `none` / `log10` (applied after 5-min averaging) |

> The `--seq_len`/`--pred_len` defaults (24/1) are legacy hourly values; for
> 5-minute parquet, pass step counts explicitly (1 day = 288, 3 days = 864).

## Data format

- **parquet** (space-weather pipeline): time column defaults to `time_utc`, with
  a `role` column (filtered to `primary` by default). Values are raw flux;
  `--transform log10` is applied after 5-minute averaging.
- **CSV** (generic): a target column (`--target_col`) and optional time column
  (`--time_col`).

## Repository layout

```
.
├── tslib/            # the importable package (pip install -e .)
│   ├── configs/      # CLI / experiment configuration
│   ├── data/         # loading, windowing, splitting, common-grid channel join
│   ├── model/        # model sources + build_model / forecast adapter
│   ├── exp/          # training loop and multi-model comparison
│   └── preprocessing/# term-split, fold-sample counting, table generation
├── notebook/         # exploratory / analysis notebooks (import tslib)
├── docs/             # design documents
└── main.py           # CLI entry point (imports tslib)
```

## Testing

```bash
python -m unittest discover -t . -s tslib -p 'test_*.py'   # from repo root
```
Real-data tests are skipped unless `SW_DATA_DIR` is set.

## Design documents

See `docs/`:
- `dataloader-split-{design,plan}.md` — half-year split + parquet `DataModule`
- `multivar-join-{design,plan}.md` — multivariate common-grid join, variable
  lengths, multi-target output
- `model-extensibility-design.md` — backbone / loss / metric registries
- `lightning-migration.md` — pytorch_lightning training architecture,
  customizing `ForecastModule`, the train-time gate, and checkpointing

## Outputs

- `runs/<run_name>/ckpt/<model>.ckpt`
- `runs/<run_name>/score/comparison.csv`

## License

See [LICENSE](LICENSE).
