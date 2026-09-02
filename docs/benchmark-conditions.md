# GOES Forecast-Strategy Benchmark — Conditions

**Date:** 2026-07-06
**Goal:** Benchmark neural forecasting on GOES space-weather series, with **LSTM run
recursively** and the other backbones run **directly**, across a single-variable and a
multivariable track. (Classical `statistic` models are deferred to a separate later run.)

## Data

Location: `$SW_DATA_DIR` (default `./data/goes_data/processed`)

| file | cadence | columns used |
|---|---|---|
| `kasi_swpc_particle_5m.parquet` | 5 min | `p_gt10`, `p_gt100` (proton integral flux, pfu) |
| `kasi_swpc_xray_1m.parquet` | 1 min | `xrs_long`, `xrs_short` (W/m²) |

Common 5-min grid (`--cadence_min 5`, xray resample-mean to 5 min), `role=primary`,
`--transform log10`. Multivariable channels span both files (outer-join on the grid).

## Compute

**GPU is fixed to device 1** — prefix every run with `CUDA_VISIBLE_DEVICES=1` (torch then
sees GPU 1 as `cuda:0`). Do not use empty `CUDA_VISIBLE_DEVICES=""` (that disables GPU).

## Fixed conditions (all runs)

| flag | value |
|---|---|
| `--cadence_min` | `5` |
| `--transform` | `log10` |
| `--split_type` | `year_half` (leakage-free rotating k-fold) |
| `--n_fold` | `5` |
| `--seq_len` (lookback) | **sweep** `2016` (7 d) · `864` (3 d) · `288` (1 d) |
| `--pred_len` (horizon) | **sweep** `144` (0.5 d) · `288` (1 d) |
| `--loss` | `mse` (log10 space); `weighted_mse` optional for rare-event runs |
| `--metrics` | `rmse mae tss hss pod far` |
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

**Folds:** pilot on `--fold_numb 0`, then run `0..4` and average (report mean ± std).

## Strategy assignment (per model — fixed)

Model set = the **fast tier**, selected by a direct-mode per-epoch speed probe
(2026-07-07) across every seq_len/pred_len combo. A model is included only if it
stays under ~10 min/epoch in **every** cell.

- **Direct backbones** (the fast tier, 9 — same set for both tracks):
  `dlinear`, `segrnn_thuml`, `tsmixer`, `patchmixer`, `tide`, `xpatch`,
  `patchtst`, `frets`, `itransformer`.
- **Recursive**: `lstm` — in **every** track (univariate and multivariable),
  since each track forecasts exactly its input channels (targets == channels).

**Excluded (≥10 min/epoch in ≥1 combo):** `timemixer`, `etsformer`, `micn`,
`scinet`, `timesnet`, and `nonstationary_transformer` (also OOMs at
`seq_len ≥ 864` on a 12 GB GPU). `segrnn` omitted as redundant with
`segrnn_thuml`. All excluded models remain registered and can be run explicitly.

Because `--forecast_strategy` is run-level, **every** cell (univariate and
multivariable) is **two invocations** — one `recursive --models lstm`, one
`direct --models <fast tier>` — merged with `merge_comparisons`. The driver
(`tslib.benchmark.driver`) builds these model lists automatically.

## Event thresholds (physical units → log10 auto-converted by MetricContext)

| target | threshold | meaning |
|---|---|---|
| `p_gt10` | `10` | NOAA S1 (≥10 MeV, ≥10 pfu) SEP |
| `xrs_long` | `1e-5` | M-class flare (`1e-4` = X-class, alt run) |

`--event_threshold` is a list aligned to `--target_cols` order.

## Track 1 — Univariate

Two independent sub-tracks (single input channel = the target):

- **UNI-A:** `p_gt10` (particle), `--event_threshold 10`
- **UNI-B:** `xrs_long` (xray), `--event_threshold 1e-5`

For **each** sub-track × seq_len `{2016,864,288}` × pred_len `{144,288}`:
- `recursive`: `lstm` (1 channel ⇒ target == all-channels holds)
- `direct`: `dlinear segrnn_thuml tsmixer patchmixer tide xpatch patchtst frets itransformer` (DIRECT_UNI, 9)

Merge the two runs per cell → one table ranking all 10 model-runs by `strategy`+`model`.

## Track 2 — Multivariable (2-variable joint forecast)

- Input channels: `p_gt10` (particle) + `xrs_long` (xray) — **2 channels**
- Targets: `p_gt10`, `xrs_long` (**== the input channels**)
- `--event_threshold 10 1e-5` (aligned to `--target_cols p_gt10 xrs_long`)

Because targets == input channels, **recursive applies here too** (unlike a
partial-target multivariable setup). For **each** seq_len `{2016,864,288}` ×
pred_len `{144,288}`:
- `recursive`: `lstm` (rolls the full 2-channel predicted frame forward)
- `direct`: `dlinear segrnn_thuml tsmixer patchmixer tide xpatch patchtst frets itransformer` (DIRECT_MULTI, 9)

Merge the two runs per cell → one table ranking all 10 model-runs.

> **Note:** `p_gt100` / `xrs_short` are intentionally **not** modelled — only the
> two targets of interest (`p_gt10`, `xrs_long`) are forecast, which keeps
> targets == channels so recursive is applicable.

## Practical constraints

- **Long lookback (`seq_len 2016`, 7 d):** heavy attention/memory for the transformer
  backbones; reduce `--batch_size` if OOM on GPU 1.
- **Recursive cost:** `lstm` rolls `pred_len` sequential single-step calls per window
  (288 for the 1-day horizon) — the slowest neural config; the deferred 3-day horizon
  (864) is heavier still.
- Neural on GPU 1; nothing on CPU in this phase (statistic deferred).

## Example commands

```bash
export SW_DATA_DIR=./data/goes_data/processed
P=$SW_DATA_DIR/kasi_swpc_particle_5m.parquet
X=$SW_DATA_DIR/kasi_swpc_xray_1m.parquet
COMMON="--seq_len 288 --pred_len 288 --n_fold 5 --fold_numb 0 --transform log10 \
  --metrics rmse mae tss hss pod far --epochs 10000 --early_stop_patience 10 \
  --batch_size 64 --lr 1e-3"

# --- UNI-A: p_gt10 — LSTM recursive + others direct (same split), then merge ---
CUDA_VISIBLE_DEVICES=1 python main.py --data_path $P --target_col p_gt10 $COMMON \
  --event_threshold 10 --forecast_strategy recursive --models lstm
CUDA_VISIBLE_DEVICES=1 python main.py --data_path $P --target_col p_gt10 $COMMON \
  --event_threshold 10 --forecast_strategy direct \
  --models dlinear segrnn_thuml tsmixer patchmixer tide xpatch patchtst frets itransformer

# --- UNI-B: xrs_long (swap file/target/threshold, same two-run pattern) ---
CUDA_VISIBLE_DEVICES=1 python main.py --data_path $X --target_col xrs_long $COMMON \
  --event_threshold 1e-5 --forecast_strategy recursive --models lstm
CUDA_VISIBLE_DEVICES=1 python main.py --data_path $X --target_col xrs_long $COMMON \
  --event_threshold 1e-5 --forecast_strategy direct \
  --models dlinear segrnn_thuml tsmixer patchmixer tide xpatch patchtst frets itransformer

# --- Track 2: multivariable (2ch p_gt10+xrs_long) — recursive lstm + direct, then merge ---
CUDA_VISIBLE_DEVICES=1 python main.py --data_path $P --target_col p_gt10 $COMMON \
  --channels $P:p_gt10 $X:xrs_long --target_cols p_gt10 xrs_long \
  --event_threshold 10 1e-5 --forecast_strategy recursive --models lstm
CUDA_VISIBLE_DEVICES=1 python main.py --data_path $P --target_col p_gt10 $COMMON \
  --channels $P:p_gt10 $X:xrs_long --target_cols p_gt10 xrs_long \
  --event_threshold 10 1e-5 --forecast_strategy direct --models dlinear segrnn_thuml tsmixer patchmixer tide xpatch patchtst frets itransformer
```

(Sweep `--seq_len {2016,864,288}` and `--pred_len {144,288}` over these. `--data_path`/
`--target_col` are required by the parser; when `--channels` is given the loader uses the
channel set, so they act only as the required placeholder.)

## Run matrix size (per fold, current scope)

- Track 1: 2 sub-tracks × 3 seq_len × 2 pred_len × (1 recursive + 9 direct) = **120 model-runs**
- Track 2: 3 seq_len × 2 pred_len × (1 recursive + 9 direct) = **60 model-runs**
- Total ≈ **180 model-runs/fold** → ×5 folds ≈ **900**. Pilot fold 0 first. (All are fast-tier models — <10 min/epoch — with EarlyStopping.)

## Deferred: statistic (later, separate)

Classical `--forecast_strategy statistic --models arima ar theta`, univariate only
(UNI-A, UNI-B). Per-window fit × ~350k test windows is expensive → sub-sample test
windows (e.g. every 12th ≈ 1-hour stride) and apply the same sub-sample to the neural
eval of the compared sub-track for fairness. Run on CPU.

## Deliverable per (track, seq_len, pred_len)

One merged comparison table (`strategy` + `model` + `rmse mae` + `tss_<t> hss_<t>
pod_<t> far_<t>` per target), sorted by the chosen `--sort_metric` (e.g. `tss_p_gt10`),
mean ± std over folds.
