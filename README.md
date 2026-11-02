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
  `pytorch_lightning` `Trainer`, with checkpointing and early stopping (see
  [`docs/lightning-migration.md`](docs/lightning-migration.md)).
- **Separate regression and event evaluation** — per-channel RMSE, MAE, MSA,
  SSPB and log R, plus event scores (TSS, HSS, POD, FAR, CSI, F1, SEDS, SEDI)
  per point, window and day. Undefined scores are NaN, not 0
  ([`docs/metrics.md`](docs/metrics.md)).
- **Saved predictions** — every run stores its test predictions, so new metrics
  are evaluated on existing runs without a GPU (`scripts/rescore.py`).

> **Data is not included** in this repository. Download it from Zenodo
> ([10.5281/zenodo.22245285](https://doi.org/10.5281/zenodo.22245285)) or
> reconstruct it from public NOAA sources — see [DATA.md](DATA.md). Point
> `SW_DATA_DIR` at the processed-parquet directory before running.

## Benchmark leaderboard

<!-- LEADERBOARD:START -->

_Representative scores of the univariate tracks. Each cell is the mean over the 5 folds and the 3 input lengths (1, 3 and 7 days) at one forecast length; forecast lengths are never averaged together. The best model of each column is in bold._

**Regression — RMSE of log10 flux (lower is better)**

<table>
<thead>
<tr><th rowspan="2" align="left"><sub>Model</sub></th><th colspan="4"><sub>SXR 1–8 Å</sub></th><th colspan="4"><sub>Proton ≥10 MeV</sub></th></tr>
<tr><th><sub>3 h</sub></th><th><sub>6 h</sub></th><th><sub>12 h</sub></th><th><sub>24 h</sub></th><th><sub>3 h</sub></th><th><sub>6 h</sub></th><th><sub>12 h</sub></th><th><sub>24 h</sub></th></tr>
</thead>
<tbody>
<tr><td><sub>DLinear</sub></td><td align="right"><sub>0.675</sub></td><td align="right"><sub>0.792</sub></td><td align="right"><sub>0.805</sub></td><td align="right"><sub>0.744</sub></td><td align="right"><sub>0.421</sub></td><td align="right"><sub>0.455</sub></td><td align="right"><sub>0.508</sub></td><td align="right"><sub>0.543</sub></td></tr>
<tr><td><sub>FReTS</sub></td><td align="right"><sub>0.313</sub></td><td align="right"><sub>0.317</sub></td><td align="right"><sub>0.328</sub></td><td align="right"><sub>0.356</sub></td><td align="right"><sub>0.363</sub></td><td align="right"><sub>0.401</sub></td><td align="right"><sub>0.527</sub></td><td align="right"><sub>0.593</sub></td></tr>
<tr><td><sub>iTransformer</sub></td><td align="right"><sub>0.194</sub></td><td align="right"><sub>0.209</sub></td><td align="right"><sub>0.225</sub></td><td align="right"><sub>0.248</sub></td><td align="right"><sub>0.183</sub></td><td align="right"><sub>0.200</sub></td><td align="right"><sub>0.234</sub></td><td align="right"><sub>0.296</sub></td></tr>
<tr><td><sub>PatchMixer</sub></td><td align="right"><sub>0.280</sub></td><td align="right"><sub>0.286</sub></td><td align="right"><sub>0.522</sub></td><td align="right"><sub>0.714</sub></td><td align="right"><sub>0.479</sub></td><td align="right"><sub>0.402</sub></td><td align="right"><sub>0.454</sub></td><td align="right"><sub>0.916</sub></td></tr>
<tr><td><sub>PatchTST</sub></td><td align="right"><sub>0.210</sub></td><td align="right"><sub>0.213</sub></td><td align="right"><sub>0.228</sub></td><td align="right"><sub>0.251</sub></td><td align="right"><sub>0.280</sub></td><td align="right"><sub>0.278</sub></td><td align="right"><sub>0.304</sub></td><td align="right"><sub>0.349</sub></td></tr>
<tr><td><sub>RLinear</sub></td><td align="right"><sub>0.202</sub></td><td align="right"><sub>0.218</sub></td><td align="right"><sub>0.238</sub></td><td align="right"><sub>0.274</sub></td><td align="right"><sub>0.400</sub></td><td align="right"><sub>0.597</sub></td><td align="right"><sub>0.767</sub></td><td align="right"><sub>0.710</sub></td></tr>
<tr><td><sub>SegRNN</sub></td><td align="right"><sub><b>0.179</b></sub></td><td align="right"><sub><b>0.196</b></sub></td><td align="right"><sub><b>0.213</b></sub></td><td align="right"><sub><b>0.235</b></sub></td><td align="right"><sub><b>0.132</b></sub></td><td align="right"><sub><b>0.155</b></sub></td><td align="right"><sub><b>0.195</b></sub></td><td align="right"><sub><b>0.257</b></sub></td></tr>
<tr><td><sub>TiDE</sub></td><td align="right"><sub>0.203</sub></td><td align="right"><sub>0.219</sub></td><td align="right"><sub>0.242</sub></td><td align="right"><sub>0.275</sub></td><td align="right"><sub>0.377</sub></td><td align="right"><sub>0.527</sub></td><td align="right"><sub>0.743</sub></td><td align="right"><sub>0.703</sub></td></tr>
<tr><td><sub>TSMixer</sub></td><td align="right"><sub>0.782</sub></td><td align="right"><sub>0.882</sub></td><td align="right"><sub>0.810</sub></td><td align="right"><sub>0.807</sub></td><td align="right"><sub>0.272</sub></td><td align="right"><sub>0.314</sub></td><td align="right"><sub>0.321</sub></td><td align="right"><sub>0.401</sub></td></tr>
<tr><td><sub>xPatch†</sub></td><td align="right"><sub>0.212</sub></td><td align="right"><sub>0.220</sub></td><td align="right"><sub>0.233</sub></td><td align="right"><sub>0.249</sub></td><td align="right"><sub>0.155</sub></td><td align="right"><sub>0.176</sub></td><td align="right"><sub>0.217</sub></td><td align="right"><sub>0.279</sub></td></tr>
</tbody>
</table>

**Classification — HSS, one forecast per day (higher is better)**

<table>
<thead>
<tr><th rowspan="2" align="left"><sub>Model</sub></th><th colspan="4"><sub>SXR ≥ M-class</sub></th><th colspan="4"><sub>Proton ≥ 10 pfu</sub></th></tr>
<tr><th><sub>3 h</sub></th><th><sub>6 h</sub></th><th><sub>12 h</sub></th><th><sub>24 h</sub></th><th><sub>3 h</sub></th><th><sub>6 h</sub></th><th><sub>12 h</sub></th><th><sub>24 h</sub></th></tr>
</thead>
<tbody>
<tr><td><sub>DLinear</sub></td><td align="right"><sub>0.183</sub></td><td align="right"><sub>0.164</sub></td><td align="right"><sub><b>0.161</b></sub></td><td align="right"><sub><b>0.257</b></sub></td><td align="right"><sub>0.696</sub></td><td align="right"><sub>0.674</sub></td><td align="right"><sub>0.613</sub></td><td align="right"><sub>0.547</sub></td></tr>
<tr><td><sub>FReTS</sub></td><td align="right"><sub>0.077</sub></td><td align="right"><sub>0.039</sub></td><td align="right"><sub>0.096</sub></td><td align="right"><sub>0.060</sub></td><td align="right"><sub>0.558</sub></td><td align="right"><sub>0.589</sub></td><td align="right"><sub>0.498</sub></td><td align="right"><sub>0.362</sub></td></tr>
<tr><td><sub>iTransformer</sub></td><td align="right"><sub>0.167</sub></td><td align="right"><sub>0.104</sub></td><td align="right"><sub>0.062</sub></td><td align="right"><sub>0.028</sub></td><td align="right"><sub>0.793</sub></td><td align="right"><sub>0.774</sub></td><td align="right"><sub>0.713</sub></td><td align="right"><sub>0.580</sub></td></tr>
<tr><td><sub>PatchMixer</sub></td><td align="right"><sub>0.266</sub></td><td align="right"><sub>0.181</sub></td><td align="right"><sub>0.112</sub></td><td align="right"><sub>0.102</sub></td><td align="right"><sub>0.635</sub></td><td align="right"><sub>0.653</sub></td><td align="right"><sub>0.642</sub></td><td align="right"><sub>0.493</sub></td></tr>
<tr><td><sub>PatchTST</sub></td><td align="right"><sub>0.126</sub></td><td align="right"><sub>0.078</sub></td><td align="right"><sub>0.035</sub></td><td align="right"><sub>0.027</sub></td><td align="right"><sub>0.591</sub></td><td align="right"><sub>0.569</sub></td><td align="right"><sub>0.534</sub></td><td align="right"><sub>0.471</sub></td></tr>
<tr><td><sub>RLinear</sub></td><td align="right"><sub>0.227</sub></td><td align="right"><sub>0.159</sub></td><td align="right"><sub>0.117</sub></td><td align="right"><sub>0.104</sub></td><td align="right"><sub>0.646</sub></td><td align="right"><sub>0.620</sub></td><td align="right"><sub>0.595</sub></td><td align="right"><sub>0.536</sub></td></tr>
<tr><td><sub>SegRNN</sub></td><td align="right"><sub>0.246</sub></td><td align="right"><sub>0.175</sub></td><td align="right"><sub>0.097</sub></td><td align="right"><sub>0.055</sub></td><td align="right"><sub><b>0.917</b></sub></td><td align="right"><sub><b>0.874</b></sub></td><td align="right"><sub><b>0.807</b></sub></td><td align="right"><sub><b>0.659</b></sub></td></tr>
<tr><td><sub>TiDE</sub></td><td align="right"><sub><b>0.279</b></sub></td><td align="right"><sub><b>0.183</b></sub></td><td align="right"><sub>0.142</sub></td><td align="right"><sub>0.125</sub></td><td align="right"><sub>0.680</sub></td><td align="right"><sub>0.638</sub></td><td align="right"><sub>0.616</sub></td><td align="right"><sub>0.544</sub></td></tr>
<tr><td><sub>TSMixer</sub></td><td align="right"><sub>0.000</sub></td><td align="right"><sub>0.008</sub></td><td align="right"><sub>0.000</sub></td><td align="right"><sub>0.000</sub></td><td align="right"><sub>0.555</sub></td><td align="right"><sub>0.463</sub></td><td align="right"><sub>0.471</sub></td><td align="right"><sub>0.260</sub></td></tr>
<tr><td><sub>xPatch†</sub></td><td align="right"><sub>0.191</sub></td><td align="right"><sub>0.134</sub></td><td align="right"><sub>0.088</sub></td><td align="right"><sub>0.044</sub></td><td align="right"><sub>0.897</sub></td><td align="right"><sub>0.863</sub></td><td align="right"><sub>0.792</sub></td><td align="right"><sub>0.648</sub></td></tr>
</tbody>
</table>

_† xPatch is left out at the 7-day input (seq_len 2016) because its training diverges there, so its cells average 10 runs (1-day and 3-day inputs) instead of all 15 runs of this forecast length and are not directly comparable with the other rows._

**Full performance tables** — every metric (RMSE, MSA, logR, HSS, TSS), the multivariate track and one table per forecast length — are in [`results/LEADERBOARD.md`](results/LEADERBOARD.md). Per-run values: [`results/results_regression.csv`](results/results_regression.csv), [`results/results_event.csv`](results/results_event.csv).

<!-- LEADERBOARD:END -->

Regenerate with `python3.12 scripts/make_leaderboard.py`.

## Data & reproduction

- **Data** — download from Zenodo or rebuild from NOAA sources: [DATA.md](DATA.md)
- **Reproduce** the benchmark end-to-end: [REPRODUCE.md](REPRODUCE.md)
- **Reconstruction pipeline** (public NOAA/NCEI download + ingest): [data_pipeline/](data_pipeline/)

## Installation

The code lives in the importable `tslib` package. Install it editable so
`import tslib` works from any notebook or script:

```bash
pip install -e .        # Python 3.12 recommended (netCDF/parquet libs)
```

Core dependencies (see `requirements.txt` / `pyproject.toml`): `numpy`,
`pandas`, `torch`, `statsmodels` (`<0.15`), and `pytorch_lightning` (the neural
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
# Univariate: proton p_gt10, 7-day look-back -> 1-day forecast, with the
# >= 10 pfu event scores (one threshold per target, physical units)
python main.py --data_path $SW_DATA_DIR/kswrc_particle_5m.parquet \
    --target_col p_gt10 --seq_len 2016 --pred_len 288 \
    --event_threshold 10 \
    --n_fold 5 --fold_numb 0 --models lstm patchtst

# Multivariate input -> single target (proton + X-ray in, proton out)
python main.py \
    --channels $SW_DATA_DIR/kswrc_particle_5m.parquet:p_gt10 \
               $SW_DATA_DIR/kswrc_xray_1m.parquet:xrs_long \
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
| `--metrics NAME [...]` | all registered | Optional filter on the metrics computed: regression `rmse mae msa sspb logr`, event `tss hss pod far csi f1 seds sedi`. Without it every metric is computed |
| `--event_threshold X [...]` | `None` | Event threshold per target, aligned to `--target_cols`, in **physical units** (e.g. `10` pfu, `1e-5` W m⁻²). Without it no event rows are written |
| `--save_predictions` | on | Write `<runs_root>/<run_name>/pred/<model>.npz` after testing; `--no-save_predictions` turns it off |

> The `--seq_len`/`--pred_len` defaults (24/1) are legacy hourly values; for
> 5-minute parquet, pass step counts explicitly (1 day = 288, 3 days = 864).

## Data format

- **parquet** (space-weather pipeline): time column defaults to `time_utc`, with
  a `role` column (filtered to `primary` by default). Values are raw flux;
  `--transform log10` is applied after 5-minute averaging.
- **CSV** (generic): a target column (`--target_col`) and optional time column
  (`--time_col`; naive times are taken as UTC). With a time column a CSV run is
  evaluated and saved like a parquet run. Without one it has no `day` event
  rows and writes no `pred/<model>.npz`, because both need each test window's
  forecast start time.

## Repository layout

```
.
├── tslib/            # the importable package (pip install -e .)
│   ├── configs/      # CLI / experiment configuration
│   ├── data/         # loading, windowing, splitting, common-grid channel join
│   ├── model/        # model sources + build_model / forecast adapter
│   ├── exp/          # training loop and multi-model comparison
│   └── preprocessing/# term-split, fold-sample counting, table generation
├── data_pipeline/    # public NOAA/NCEI download + ingest -> parquet
├── results/          # results_regression.csv + results_event.csv (v2 tables); LEADERBOARD.md (full tables); v1/ (v1 files)
├── scripts/          # fetch_data.py, make_leaderboard.py, rescore.py, dump_predictions.py, verify_archive.py
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
- `metrics.md` — metric definitions, NaN rules, aggregation units, result and
  prediction file formats, differences from v1
- `benchmark-conditions.md` — the benchmark task: targets, tracks, splits, metrics
- `dataloader-split-design.md` — half-year split + parquet `DataModule`
- `multivar-join-design.md` — multivariate common-grid join, variable
  lengths, multi-target output
- `model-extensibility-design.md` — backbone / loss / metric registries
- `lightning-migration.md` — pytorch_lightning training architecture,
  customizing `ForecastModule`, early stopping, and checkpointing
- `v004-short-horizon-sweep.md` — the 3 h / 6 h horizons and the multi-GPU job queue

## Outputs

- `runs/<run_name>/ckpt/<model>.ckpt`
- `runs/<run_name>/pred/<model>.npz` — test predictions, observations and forecast start times (not for CSV input without `--time_col`)
- `runs/<run_name>/score/comparison_regression.csv` — per-channel regression scores, one row per model and channel
- `runs/<run_name>/score/comparison_event.csv` — event scores and counts per model, channel and aggregation unit
- `runs/<run_name>/score/rows/<model>.json` — the model's result record

The benchmark driver gathers all runs into `results_regression.csv` and
`results_event.csv` (see [`results/README.md`](results/README.md)).

## Citation

If you use this benchmark, code, or dataset, please cite both (see [CITATION.cff](CITATION.cff)):

- **Code / benchmark** — this repository
- **Dataset** — Zenodo DOI [10.5281/zenodo.22245285](https://doi.org/10.5281/zenodo.22245285)

## License

See [LICENSE](LICENSE).
