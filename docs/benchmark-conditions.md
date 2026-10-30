# GOES Forecast-Strategy Benchmark — Conditions

**Date:** 2026-07-06 (grid updated for v2.0.0)
**Goal:** Benchmark neural forecasting on GOES space-weather series across a
single-variable and a multivariable track.

> **The released benchmark (v2.0.0)** is the direct-strategy sweep of ten models
> (the fast tier below plus `rlinear`) over three tracks, input lengths 288 / 864 /
> 2016 (1 / 3 / 7 days), forecast lengths 36 / 72 / 144 / 288 (3 / 6 / 12 / 24 h)
> and folds 0-4, minus xPatch at the 7-day input, where its training diverges:
> **1,740 runs** (`python3.12 -m tslib.benchmark.scheduler` or
> `python3.12 -m tslib.benchmark.driver --folds 0 1 2 3 4`). The result tables also
> keep the six diverged v1.0.0 xPatch runs at `seq_len` 2016 (fold 0, 12 h and
> 24 h) as NaN rows, 1,746 runs in all. The recursive `lstm` and the `statistic`
> models are supported by the driver but are not part of the released results.

## Data

Location: `$SW_DATA_DIR` (default `./data/goes_data/processed`)

| file | cadence | columns used |
|---|---|---|
| `kswrc_particle_5m.parquet` | 5 min | `p_gt10`, `p_gt100` (proton integral flux, pfu) |
| `kswrc_xray_1m.parquet` | 1 min | `xrs_long`, `xrs_short` (W/m²) |

Common 5-min grid (`--cadence_min 5`, xray resample-mean to 5 min), `role=primary`,
`--transform log10`. Multivariable channels span both files (outer-join on the grid).

## Compute

Each run uses one GPU. Pick it with `CUDA_VISIBLE_DEVICES` (torch then sees that
GPU as `cuda:0`); the multi-GPU scheduler sets it per job (`--gpus`). An empty
`CUDA_VISIBLE_DEVICES=""` disables the GPU.

## Fixed conditions (all runs)

| flag | value |
|---|---|
| `--cadence_min` | `5` |
| `--transform` | `log10` |
| `--split_type` | `year_half` (leakage-free rotating k-fold) |
| `--n_fold` | `5` |
| `--seq_len` (lookback) | **sweep** `2016` (7 d) · `864` (3 d) · `288` (1 d) |
| `--pred_len` (horizon) | **sweep** `36` (3 h) · `72` (6 h) · `144` (12 h) · `288` (24 h); v003 ran `144`/`288`, v004 added `36`/`72` — see [v004-short-horizon-sweep.md](v004-short-horizon-sweep.md) |
| `--loss` | `mse` (log10 space); `weighted_mse` optional for rare-event runs |
| `--metrics` | omitted: every registered metric is computed (see [Metrics](#metrics)) |
| `--seed` | `42` (repeat 3 seeds for variance if time allows) |
| neural HP | `--epochs 10000 --early_stop_patience 10 --batch_size 64 --lr 1e-3 --d_model 128 --num_layers 2 --dropout 0.1` |

### Pre-test + early stopping

v003 no longer gates a run's training time from inside `main.py`. Instead
there are two steps:

1. **Pre-test** — before launching the full sweep, run
   `python3.12 -m tslib.benchmark.pretest` (`tslib/benchmark/pretest.py`).
   It probes every cell's models with a few real training steps
   (`--probe-batches`, default `3`), projects a runtime as
   `per_epoch_time × --nominal-epochs` (default `50`) — a yardstick, not the
   actual early-stopped duration — and, for any model whose projection
   exceeds `--threshold-hours` (default `12.0`), asks interactively whether
   to proceed. Approved (cell, model) entries are written to
   `runs/bench/manifest.json`.
2. **Driver** — run the sweep against only the approved entries:
   `python3.12 -m tslib.benchmark.driver --manifest runs/bench/manifest.json`.

Actual training uses `--epochs 10000` with
`EarlyStopping(monitor="val_loss", patience=--early_stop_patience [10])`, so
most runs stop long before the 10000-epoch ceiling once `val_loss` stops
improving for 10 epochs.

**`timesnet` is excluded from this benchmark entirely** — it is far too slow
(it alone took ~14.5 h for 3 epochs on the full dataset in the pilot), so it
is dropped from every model set below. The pre-test's `--nominal-epochs` /
`--threshold-hours` double-check is the safety net for the rest: any
remaining model whose projected runtime is too high at the long
`seq_len`/`pred_len` sweep points is caught and confirmed (or declined)
before it ever gets a driver invocation, rather than silently eating the
whole benchmark's time budget. See `docs/lightning-migration.md` for the
full pre-test/manifest and early-stopping mechanics.

> **Deferred (note only):** `--pred_len 864` (3-day horizon) — run later once the
> 0.5 d / 1 d horizons are validated. The 3-day recursive rollout is the heaviest config.
> **Deferred:** the `statistic` strategy (ARIMA/AR/Theta) — separate later benchmark; see
> the "Deferred: statistic" section.

**Folds:** pilot on `--fold_numb 0`, then run `0..4`. Results are reported as the
mean over folds (and input lengths) of the per-run values; runs whose value is
undefined (NaN) are left out of a mean and the number of runs that remain is shown
(`k/n`, n = folds x input lengths), see [Metrics](#metrics). A model with fewer
runs than that (xPatch, see below) is marked and footnoted. No standard deviation
is reported.

## Strategy assignment (per model — fixed)

Model set = the **fast tier**, selected by a direct-mode per-epoch speed probe
(2026-07-07) across every seq_len/pred_len combo. A model is included only if it
stays under ~10 min/epoch in **every** cell.

- **Direct backbones** (10 — same set for both tracks): the fast tier
  `dlinear`, `segrnn_thuml`, `tsmixer`, `patchmixer`, `tide`, `xpatch`,
  `patchtst`, `frets`, `itransformer`, plus `rlinear`. `xpatch` is not run at
  `seq_len` 2016: its training diverges at the 7-day input (every v1.0.0 fold),
  so the driver's `MODEL_EXCLUSIONS` drops it there.
- **Recursive** (optional, not in the released results): `lstm`, possible in
  every track since each track forecasts exactly its input channels
  (targets == channels); `--strategies recursive`.

**Excluded (≥10 min/epoch in ≥1 combo):** `timemixer`, `etsformer`, `micn`,
`scinet`, `timesnet`, and `nonstationary_transformer` (also OOMs at
`seq_len ≥ 864` on a 12 GB GPU). `segrnn` omitted as redundant with
`segrnn_thuml`. All excluded models remain registered and can be run explicitly.

`--forecast_strategy` is run-level, so a cell that also runs the recursive
`lstm` takes two invocations, one per strategy, whose regression tables
`merge_comparisons` can join. The driver (`tslib.benchmark.driver`) and the
scheduler build the model lists automatically.

## Metrics

Every run is evaluated twice, and both evaluations use all registered metrics:

- **Regression**, per target channel: `rmse` `mae` `msa` `sspb` `logr`.
- **Event** (yes/no at the threshold below), per target channel and per
  aggregation unit `point` / `window` / `day`: `tss` `hss` `pod` `far` `csi` `f1`
  `seds` `sedi`, together with the contingency counts `tp` `fp` `fn` `tn`.

`day` is one forecast per 24 hours (a 24-hour cut over the test windows) and is
used for every forecast length. An undefined score (for example `far` when no
event is forecast) is NaN, and so is every score of a channel whose predictions
contain a NaN. Formulas, the NaN rules and the file formats are in
[metrics.md](metrics.md). Results go to `results_regression.csv` and
`results_event.csv`; each run also saves its predictions (`pred/<model>.npz`).

## Event thresholds (physical units → log10 auto-converted by MetricContext)

| target | threshold | meaning |
|---|---|---|
| `p_gt10` | `10` | NOAA S1 (≥10 MeV, ≥10 pfu) SEP |
| `xrs_long` | `1e-5` | M-class flare (`1e-4` = X-class, alt run) |

`--event_threshold` is a list aligned to `--target_cols` order. An event is a
value at or above the threshold.

## Track 1 — Univariate

Two independent sub-tracks (single input channel = the target):

- **UNI-A:** `p_gt10` (particle), `--event_threshold 10`
- **UNI-B:** `xrs_long` (xray), `--event_threshold 1e-5`

For **each** sub-track × seq_len `{2016,864,288}` × pred_len `{36,72,144,288}`:
- `direct`: `dlinear segrnn_thuml tsmixer patchmixer tide xpatch patchtst frets itransformer rlinear` (DIRECT_UNI, 10; no `xpatch` at seq_len 2016)
- optional `recursive`: `lstm` (1 channel ⇒ target == all-channels holds)

## Track 2 — Multivariable (2-variable joint forecast)

- Input channels: `p_gt10` (particle) + `xrs_long` (xray) — **2 channels**
- Targets: `p_gt10`, `xrs_long` (**== the input channels**)
- `--event_threshold 10 1e-5` (aligned to `--target_cols p_gt10 xrs_long`)

Because targets == input channels, **recursive applies here too** (unlike a
partial-target multivariable setup). For **each** seq_len `{2016,864,288}` ×
pred_len `{36,72,144,288}`:
- `direct`: the same 10 models (DIRECT_MULTI; no `xpatch` at seq_len 2016)
- optional `recursive`: `lstm` (rolls the full 2-channel predicted frame forward)

> **Note:** `p_gt100` / `xrs_short` are intentionally **not** modelled — only the
> two targets of interest (`p_gt10`, `xrs_long`) are forecast, which keeps
> targets == channels so recursive is applicable.

## Practical constraints

- **Long lookback (`seq_len 2016`, 7 d):** heavy attention/memory for the transformer
  backbones; reduce `--batch_size` if a 12 GB GPU runs out of memory.
- **Recursive cost:** `lstm` rolls `pred_len` sequential single-step calls per window
  (288 for the 1-day horizon) — the slowest neural config; the deferred 3-day horizon
  (864) is heavier still.
- Neural models on the GPU; the deferred `statistic` models run on the CPU.

## Example commands

```bash
export SW_DATA_DIR=./data/goes_data/processed
P=$SW_DATA_DIR/kswrc_particle_5m.parquet
X=$SW_DATA_DIR/kswrc_xray_1m.parquet
COMMON="--seq_len 288 --pred_len 288 --n_fold 5 --fold_numb 0 --transform log10 \
  --epochs 10000 --early_stop_patience 10 --batch_size 64 --lr 1e-3"

MODELS="dlinear segrnn_thuml tsmixer patchmixer tide xpatch patchtst frets itransformer rlinear"

# --- UNI-A: p_gt10, the ten direct models ---
python main.py --data_path $P --target_col p_gt10 $COMMON \
  --event_threshold 10 --forecast_strategy direct --models $MODELS

# --- UNI-B: xrs_long (swap file/target/threshold) ---
python main.py --data_path $X --target_col xrs_long $COMMON \
  --event_threshold 1e-5 --forecast_strategy direct --models $MODELS

# --- Track 2: multivariable (2ch p_gt10+xrs_long) ---
python main.py --data_path $P --target_col p_gt10 $COMMON \
  --channels $P:p_gt10 $X:xrs_long --target_cols p_gt10 xrs_long \
  --event_threshold 10 1e-5 --forecast_strategy direct --models $MODELS

# optional: the recursive lstm of a cell, in its own invocation
python main.py --data_path $P --target_col p_gt10 $COMMON \
  --event_threshold 10 --forecast_strategy recursive --models lstm
```

(Sweep `--seq_len {2016,864,288}` and `--pred_len {36,72,144,288}` over these, leaving
`xpatch` out at `--seq_len 2016`; the driver and the scheduler do this for you.
`--data_path`/`--target_col` are required by the parser; when `--channels` is given the
loader uses the channel set, so they act only as the required placeholder.)

## Run matrix size (per fold, released grid)

- Track 1: 2 sub-tracks × 4 pred_len × (3 seq_len × 10 direct − 1 `xpatch` at 2016) = **232 model-runs**
- Track 2: 4 pred_len × (3 seq_len × 10 direct − 1) = **116 model-runs**
- Total **348 model-runs/fold** → ×5 folds = **1,740**. Pilot fold 0 first. (All are fast-tier-speed models — <10 min/epoch — with EarlyStopping.)
- The released tables add the six diverged v1.0.0 xPatch runs at `seq_len` 2016 (fold 0,
  `pred_len` 144 and 288, three tracks) as NaN rows: **1,746 runs** (v1.0.0 876 + v004 870).
  The optional recursive `lstm` would add 3 runs (one per track) per (seq_len, pred_len) and fold.

## Deferred: statistic (later, separate)

Classical `--forecast_strategy statistic --models arima ar theta`, univariate only
(UNI-A, UNI-B). Per-window fit × ~350k test windows is expensive → sub-sample test
windows (e.g. every 12th ≈ 1-hour stride) and apply the same sub-sample to the neural
eval of the compared sub-track for fairness. Run on CPU.

## Deliverable per (track, seq_len, pred_len)

Two comparison tables, written to the cell's `score/` folder:
`comparison_regression.csv` (`strategy`, `model`, `best_val_loss`, then one row per
target channel with `rmse mae msa sspb logr`, sorted by `--sort_metric`, default
`best_val_loss`) and `comparison_event.csv` (one row per model, channel and
aggregation unit with the counts and the eight event scores, best `hss` first
within each channel and unit). Across cells and folds the benchmark driver
collects every run into `results_regression.csv` and `results_event.csv`, and the
README leaderboard averages them per forecast length (mean over folds and input
lengths, undefined runs left out and counted; a model with fewer runs than the
grid, xPatch, is marked and footnoted).
