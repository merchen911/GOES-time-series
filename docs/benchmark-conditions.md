# GOES Forecast-Strategy Benchmark — Conditions

**Date:** 2026-07-06
**Goal:** Benchmark neural forecasting on GOES space-weather series, with **LSTM run
recursively** and the other backbones run **directly**, across a single-variable and a
multivariable track. (Classical `statistic` models are deferred to a separate later run.)

## Data

Location: `$SW_DATA_DIR` = `/NAS/ioGuard3/vol3/spaceai/SW_framework/data/goes_data/processed`

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
| neural HP | `--epochs 30 --batch_size 64 --lr 1e-3 --d_model 128 --num_layers 2 --dropout 0.1` |

### Train-time gate

Every neural run goes through the v003 Lightning train-time gate
(`TimingGateCallback`), which probes the first few training batches and
extrapolates a full-training-time estimate before letting a model run to
completion. Defaults used here (not overridden in the example commands
below): `--max_train_hours 6.0`, `--on_slow skip`, `--probe_batches 3`.

**`timesnet` is excluded from this benchmark entirely** — it is far too slow
(it alone took ~14.5 h for 3 epochs on the full dataset in the pilot), so it
is dropped from every model set below. The gate remains as a safety net for
the rest: any remaining model estimated to exceed the limit at the long
`seq_len`/`pred_len` sweep points is auto-skipped (excluded from
`comparison.csv`, with a `SKIPPED (too slow): ...` line on stdout) rather
than silently eating the whole benchmark's time budget. To force a skipped
cell to run anyway, re-run that specific invocation with `--on_slow proceed`.
See `docs/lightning-migration.md` for the full gate mechanics and the other
`--on_slow` policies.

> **Deferred (note only):** `--pred_len 864` (3-day horizon) — run later once the
> 0.5 d / 1 d horizons are validated. The 3-day recursive rollout is the heaviest config.
> **Deferred:** the `statistic` strategy (ARIMA/AR/Theta) — separate later benchmark; see
> the "Deferred: statistic" section.

**Folds:** pilot on `--fold_numb 0`, then run `0..4` and average (report mean ± std).

## Strategy assignment (per model — fixed)

| model | strategy | why |
|---|---|---|
| `lstm` | **recursive** | RNN rolled one step at a time |
| `patchtst` | direct | one-shot |
| `itransformer` | direct | one-shot |
| `timemixer` | direct | one-shot |

`timesnet` excluded (too slow — see the train-time gate note above). `etsformer`
excluded (pre-existing broken `layers` imports). Because `--forecast_strategy`
is run-level, each (track, seq_len, pred_len) cell is **two invocations** — one
`--forecast_strategy recursive --models lstm`, one `--forecast_strategy direct --models
patchtst itransformer timemixer` — merged with `merge_comparisons`.

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
- `direct`: `patchtst itransformer timemixer`

Merge the two runs per cell → one table ranking all 4 model-runs by `strategy`+`model`.

## Track 2 — Multivariable (direct, multi-target)

- Input channels: `p_gt10`, `p_gt100` (particle) + `xrs_long`, `xrs_short` (xray)
- Targets (main): `p_gt10`, `xrs_long` (multi-target output)
- `--event_threshold 10 1e-5` (aligned to `--target_cols p_gt10 xrs_long`)

For **each** seq_len `{2016,864,288}` × pred_len `{144,288}`:
- `direct`: `lstm patchtst itransformer timemixer`

> **Note:** here `lstm` runs **direct**, not recursive — recursive needs targets == all
> input channels (2 ≠ 4), so it is not applicable to this multi-target multivariable config.
> The recursive-LSTM comparison lives in Track 1.

## Practical constraints

- **Long lookback (`seq_len 2016`, 7 d):** heavy attention/memory for the transformer
  backbones; reduce `--batch_size` if OOM on GPU 1.
- **Recursive cost:** `lstm` rolls `pred_len` sequential single-step calls per window
  (288 for the 1-day horizon) — the slowest neural config; the deferred 3-day horizon
  (864) is heavier still.
- Neural on GPU 1; nothing on CPU in this phase (statistic deferred).

## Example commands

```bash
export SW_DATA_DIR=/NAS/ioGuard3/vol3/spaceai/SW_framework/data/goes_data/processed
P=$SW_DATA_DIR/kasi_swpc_particle_5m.parquet
X=$SW_DATA_DIR/kasi_swpc_xray_1m.parquet
COMMON="--seq_len 288 --pred_len 288 --n_fold 5 --fold_numb 0 --transform log10 \
  --metrics rmse mae tss hss pod far --epochs 30 --batch_size 64 --lr 1e-3"

# --- UNI-A: p_gt10 — LSTM recursive + others direct (same split), then merge ---
CUDA_VISIBLE_DEVICES=1 python main.py --data_path $P --target_col p_gt10 $COMMON \
  --event_threshold 10 --forecast_strategy recursive --models lstm
CUDA_VISIBLE_DEVICES=1 python main.py --data_path $P --target_col p_gt10 $COMMON \
  --event_threshold 10 --forecast_strategy direct \
  --models patchtst itransformer timemixer

# --- UNI-B: xrs_long (swap file/target/threshold, same two-run pattern) ---
CUDA_VISIBLE_DEVICES=1 python main.py --data_path $X --target_col xrs_long $COMMON \
  --event_threshold 1e-5 --forecast_strategy recursive --models lstm
CUDA_VISIBLE_DEVICES=1 python main.py --data_path $X --target_col xrs_long $COMMON \
  --event_threshold 1e-5 --forecast_strategy direct \
  --models patchtst itransformer timemixer

# --- Track 2: multivariable, multi-target (direct, lstm included as direct) ---
CUDA_VISIBLE_DEVICES=1 python main.py --data_path $P --target_col p_gt10 $COMMON \
  --channels $P:p_gt10 $P:p_gt100 $X:xrs_long $X:xrs_short \
  --target_cols p_gt10 xrs_long --event_threshold 10 1e-5 \
  --forecast_strategy direct --models lstm patchtst itransformer timemixer
```

(Sweep `--seq_len {2016,864,288}` and `--pred_len {144,288}` over these. `--data_path`/
`--target_col` are required by the parser; when `--channels` is given the loader uses the
channel set, so they act only as the required placeholder.)

## Run matrix size (per fold, current scope)

- Track 1: 2 sub-tracks × 3 seq_len × 2 pred_len × (1 recursive + 3 direct) = **48 model-runs**
- Track 2: 3 seq_len × 2 pred_len × 4 direct = **24 model-runs**
- Total ≈ **72 model-runs/fold** → ×5 folds ≈ **360**. Pilot fold 0 first.

## Deferred: statistic (later, separate)

Classical `--forecast_strategy statistic --models arima ar theta`, univariate only
(UNI-A, UNI-B). Per-window fit × ~350k test windows is expensive → sub-sample test
windows (e.g. every 12th ≈ 1-hour stride) and apply the same sub-sample to the neural
eval of the compared sub-track for fairness. Run on CPU.

## Deliverable per (track, seq_len, pred_len)

One merged comparison table (`strategy` + `model` + `rmse mae` + `tss_<t> hss_<t>
pod_<t> far_<t>` per target), sorted by the chosen `--sort_metric` (e.g. `tss_p_gt10`),
mean ± std over folds.
