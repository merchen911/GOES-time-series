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

## Benchmark leaderboard

<!-- LEADERBOARD:START -->

_Mean over the 5-fold rotating split and all input-length/horizon cells; lower RMSE is better. Full per-configuration results in [`results/results_master.csv`](results/results_master.csv)._

**SXR 1--8 A (univariate)**

| Model | RMSE (log-flux) | SXR HSS / TSS |
|---|---|---|
| SegRNN | 0.224 | 0.063 / 0.034 |
| iTransformer | 0.236 | 0.060 / 0.038 |
| PatchTST | 0.239 | 0.055 / 0.036 |
| xPatch | 0.241 | 0.058 / 0.036 |
| RLinear | 0.256 | 0.078 / 0.050 |
| TiDE | 0.259 | 0.080 / 0.051 |
| FReTS | 0.342 | 0.030 / 0.024 |
| PatchMixer | 0.618 | 0.062 / 0.043 |
| DLinear | 0.774 | 0.049 / 0.094 |
| TSMixer | 0.809 | 0.000 / 0.000 |

**Proton >=10 MeV (univariate)**

| Model | RMSE (log-flux) | Proton HSS / TSS |
|---|---|---|
| SegRNN | 0.226 | 0.781 / 0.729 |
| xPatch | 0.248 | 0.666 / 0.621 |
| iTransformer | 0.265 | 0.674 / 0.616 |
| PatchTST | 0.327 | 0.490 / 0.444 |
| TSMixer | 0.361 | 0.326 / 0.252 |
| DLinear | 0.525 | 0.551 / 0.595 |
| FReTS | 0.560 | 0.273 / 0.194 |
| PatchMixer | 0.685 | 0.528 / 0.511 |
| TiDE | 0.723 | 0.537 / 0.568 |
| RLinear | 0.739 | 0.546 / 0.584 |

**Multivariate (joint proton + SXR)**

| Model | RMSE (log-flux) | SXR HSS / TSS | Proton HSS / TSS |
|---|---|---|---|
| SegRNN | 0.209 | 0.061 / 0.035 | 0.674 / 0.611 |
| xPatch | 0.216 | 0.060 / 0.038 | 0.578 / 0.504 |
| iTransformer | 0.259 | 0.039 / 0.029 | 0.389 / 0.341 |
| PatchMixer | 0.261 | 0.075 / 0.049 | 0.472 / 0.409 |
| PatchTST | 0.271 | 0.037 / 0.026 | 0.384 / 0.332 |
| DLinear | 0.326 | 0.076 / 0.096 | 0.515 / 0.471 |
| RLinear | 0.328 | 0.087 / 0.083 | 0.526 / 0.491 |
| TiDE | 0.336 | 0.089 / 0.089 | 0.529 / 0.495 |
| TSMixer | 0.723 | 0.012 / 0.023 | 0.105 / 0.088 |
| FReTS | 2.112 | 0.032 / 0.021 | 0.232 / 0.160 |

<!-- LEADERBOARD:END -->

Regenerate with `python3.12 scripts/make_leaderboard.py`.

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
- `benchmark-conditions.md` — the benchmark task: targets, tracks, splits, metrics
- `dataloader-split-design.md` — half-year split + parquet `DataModule`
- `multivar-join-design.md` — multivariate common-grid join, variable
  lengths, multi-target output
- `model-extensibility-design.md` — backbone / loss / metric registries
- `lightning-migration.md` — pytorch_lightning training architecture,
  customizing `ForecastModule`, the train-time gate, and checkpointing

## Outputs

- `runs/<run_name>/ckpt/<model>.ckpt`
- `runs/<run_name>/score/comparison.csv`

## License

See [LICENSE](LICENSE).
