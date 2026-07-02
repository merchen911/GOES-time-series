# Multivariate Common-Grid Join — Design Spec

- **Date:** 2026-07-01
- **Status:** approved (design), implementation pending
- **Workspace:** `workdir/sw-framework-v002/` (extends the univariate loader; see
  [dataloader-split-design.md](dataloader-split-design.md))
- **Scope:** extend `DataModule` to build a **multivariate** input by joining N
  parquet channels (e.g. particle `p_gt10` + xray `xrs_long`) on a common
  5-min grid, with configurable prediction targets (univariate OR multivariate
  output). Backward-compatible: a single channel reproduces the current
  univariate behavior.

## 1. Goal & context

The univariate loader consumes one parquet channel of the `primary` series. For
multivariate forecasting we need several channels that live in **separate
parquet files at different native cadences** (particle 5-min, xray 1-min),
aligned onto one regular grid. The join and gap rule were prototyped and
validated in `workdir/preprocessing/count_fold_samples.py::load_validity`; this
spec moves that logic into the training `DataModule` with one refinement:
**5-min mean resampling** instead of exact-mark sampling.

## 2. Decisions (locked during brainstorming)

| # | Decision |
|---|---|
| M1 | Config-driven channels: `--channels PATH:COL [PATH:COL ...]`. One channel → univariate (current behavior). |
| M2 | Prediction targets are a **subset** of the input channels: `--target_cols COL [COL ...]`. `len==1` → univariate output; `>1` → multivariate output. Default = first channel's column. |
| M3 | Grid building = **`resample("{cadence_min}min").mean()`** per term (bin mean), replacing exact-mark `reindex`. Unifies alignment + downsample. particle univariate at cadence 5 is unchanged (1 sample/bin → mean = value). |
| M4 | Transform order: **mean on RAW flux first, THEN `log10`** (i.e. log10 of the 5-min mean flux — not the mean of logs). |
| M5 | A 5-min bin is valid for a channel iff it has **≥ `--min_bin_count`** native samples (default `1`); otherwise that bin is NaN for that channel. Enforced via `resample(...).count()` masking the mean. |
| M6 | Row validity = **all channels present** (`notna().all(axis=1)`). Window rule unchanged: whole `L=seq_len+pred_len` all-present, never crossing a half-year term boundary (option 1). |
| M7 | Backward compat: single channel + single target reproduces the existing univariate path; existing `--data_path`/`--target_col` still work (synthesized into one channel when `--channels` is absent). |
| M8 | ~~Out of scope: adapting models to consume >1 output channel.~~ **DONE (2026-07-01, follow-on `c87c813`)** — see §9. The forecast adapter now threads a full `target_indices` list and selects `pred[..., target_indices]` → `(B,pred_len,T)`, matching the loader's `y=(pred,T)`/`output_size=T`. Multi-target via `--target_cols COL COL ...`. |

## 3. Architecture

Touched file: **`workdir/sw-framework-v002/src/data/loader.py`** (+ `config.py`,
+ tests). `exp.py` interface stays compatible (new `DataBundle.output_size`
defaults to 1). The `.parquet` branch of `setup()` handles both 1 and N
channels; the legacy CSV/`ratio` path is untouched.

```
config: channels=[(path,col)...], target_cols⊆cols, role, transform,
        cadence_min, min_bin_count, split_type, n_fold, fold_numb,
        seq_len, pred_len
   │
   ▼
_setup_parquet()
   1. parse channels; per channel: _prepare_series(role, dedup) RAW (no transform)
   2. outer-join channels on time_utc → DataFrame [col_0..col_{C-1}]
   3. terms from _term_labels(joined.index); _fold_indices → {train,val,test}
   4. per split, per term:
        resample(cadence).mean()  and  resample(cadence).count()
        mean[count < min_bin_count] = NaN            (M5)
        transform == log10 → log10(mean.where(>0))   (M4, per column)
        valid_row = mean.notna().all(axis=1)         (M6)
        starts = _valid_starts(valid_row, L) + offset ; concat grid (G,C)
   5. target_idx = [channel-order index of each col in target_cols]
      WindowDataset(values(G,C), starts, seq_len, pred_len, target_idx)
   6. log per-split window counts
   │
   ▼
DataBundle(train/val/test loaders, input_size=C, target_index=target_idx[0],
           output_size=len(target_cols))
```

## 4. Components (changes)

- **`config.py`**: add `--channels` (nargs+, `PATH:COL`), `--target_cols`
  (nargs+), `--min_bin_count` (int, default 1). Keep `--data_path`/`--target_col`.
  `config_postprocess`: if `--channels` absent but `data_path` ends `.parquet`,
  synthesize `channels=[f"{data_path}:{target_col}"]`; default `target_cols` to
  the first channel's column; validate every target_col is among the channel
  columns; validate each channel spec has exactly one `:` splitting an existing
  file path from a column name.
- **`_parse_channels(channels_arg) -> list[tuple[str,str]]`** — split each
  `PATH:COL` on the last `:` (paths may contain none; column names don't contain
  `:`). Returns ordered `(path, col)` list.
- **`_prepare_series(...)`** — add a `transform="none"` call site (transform now
  applied later, after resample); function itself unchanged.
- **`_grid_and_starts(joined, terms, cadence_min, seq_len, pred_len, split_type,
  transform, min_bin_count)`** — generalized to a multi-column DataFrame:
  per term `resample(f"{cadence_min}min")` → `.mean()` and `.count()`; mask; per-column
  transform; `valid = mean.notna().all(axis=1).to_numpy()`; `_valid_starts`;
  concat `mean.to_numpy()` (shape `(G,C)`). Returns `(values(G,C), starts)`.
- **`WindowDataset(values, starts, seq_len, pred_len, target_idx=None)`** —
  `values` may be 2-D `(G,C)`; `x = values[s:s+seq_len]` → `(seq_len,C)`;
  `y = values[s+seq_len:s+L]`, then if `target_idx` is not None select those
  columns → `(pred_len, T)`. `target_idx=None` keeps current `(·,1)` behavior for
  the plain 1-D case (unsqueeze).
- **`DataBundle`** — add `output_size: int = 1` (non-breaking default).

## 5. Config summary

| flag | default | meaning |
|---|---|---|
| `--channels` | (from `--data_path:--target_col`) | input channels `PATH:COL …` |
| `--target_cols` | `[first channel col]` | prediction targets ⊆ channel cols |
| `--min_bin_count` | `1` | min native samples per bin to keep it |
| (reused) `--role` `--transform` `--cadence_min` `--seq_len` `--pred_len` `--split_type` `--n_fold` `--fold_numb` | | as in univariate |

For multivariate mixing 1-min + 5-min data, `--cadence_min 5` is required
(coarser cadence); the 1-min channel is mean-aggregated into 5-min bins.

## 6. Testing (`src/data/test_loader.py`, append)

1. **`_parse_channels`** — `a/b.parquet:p_gt10` → `("a/b.parquet","p_gt10")`; path with no colon errors clearly.
2. **mean+min_bin_count** — synthetic 1-min series over one 5-min bin: mean equals arithmetic mean; with `min_bin_count=5` a 4-sample bin → NaN.
3. **transform order** — mean of `[10,1000]` = 505 → `log10(505)` (NOT mean of `[1,3]`); confirms M4.
4. **multivar validity/shape** — two synthetic channels; a bin missing in one channel invalidates that row; `x:(N,seq,2)`, `y:(N,pred,T)`.
5. **univariate regression** — single channel `p_gt10`, particle v02, seq 288 + pred 12, fold 0: window counts EXACTLY equal the recorded univariate figures (train 1,076,731 / val 370,516 / test 350,429), proving resample-mean == prior reindex for native-cadence data.
6. **real multivar integration** — channels `p_gt10 + xrs_long`, target `[p_gt10]`: `input_size==2`, `output_size==1`, `y` shape `(·,pred,1)`, all splits non-empty, leakage-free terms.

## 7. Out of scope

~~Multi-output model heads (M8)~~ — **now implemented, see §9.** Still out of
scope: `count_fold_samples.py` realignment to mean (the analysis tool keeps
exact-mark; note the divergence), per-channel distinct transforms, non-`primary`
role mixing.

## 8. Progress log (continuously updated)

- **2026-07-01** — Design approved (M1–M8). *Next: implementation plan
  (writing-plans).*
- **2026-07-01** — Implementation plan written
  (`multivar-join-plan.md`) and executed task-by-task via
  subagent-driven-development on branch `feat/multivar-join`
  (baseline `ffbf4ef`):
  - Task 1 (`eeff038`) — config flags `--channels PATH:COL`,
    `--target_cols`, `--min_bin_count` (default 1); `--data_path`/
    `--target_col` retained for backward compat.
  - Task 2 (`d15d9f3`) — `_parse_channels` (split on last `:`) and
    `_resolve_channels(cfg) -> (channels, target_cols)` (defaults +
    target⊆channels validation).
  - Task 3 (`eda2020`) — `WindowDataset` generalized to 2-D `(G,C)`
    values with `target_idx` (1-D promoted to `(G,1)`; `y=(pred,T)`);
    `DataBundle.output_size: int = 1` (non-breaking).
  - Task 4 (`693cf67`) — frame-based `_grid_and_starts`: per-term
    `resample("{cadence}min").mean()`, `.count()`-based `min_bin_count`
    masking (M5), mean-then-log10 (M4), AND-rule row validity (M6),
    no cross-term windows. Prior `TestGridAndStarts` starts
    (`[3,4]`, `[0,1,4,5]`) unchanged → resample-mean == old reindex on
    5-min-native data.
  - Task 5 (`61e4944`) — multivariate `DataModule._setup_parquet`:
    per-channel `_prepare_series(..., "none")` (raw), outer-join into
    one frame, term-level leakage-free k-fold, `transform`/
    `min_bin_count` applied in `_grid_and_starts`, `target_idx` subset;
    returns `DataBundle(input_size=C, target_index=target_idx[0],
    output_size=len(target_cols))`; per-split window counts printed at
    setup. Legacy CSV/`ratio` path and symbols untouched.
  - Tests: **26/26 pass** (`python3.12 -m unittest data.test_loader`);
    legacy-symbol import OK. **Univariate regression**
    (`TestSetupUnivariateRegression`, particle `p_gt10`, seq 288 +
    pred 12, fold 0) reproduces the recorded univariate counts
    EXACTLY — train 1,076,731 / val 370,516 / test 350,429 — proving
    resample-mean equals the prior reindex on real data. **Real
    multivariate integration** (`TestSetupMultivar`, channels
    `p_gt10 + xrs_long`, target `[p_gt10]`): `input_size==2`,
    `output_size==1`, `x=(·,288,2)`, `y=(·,12,1)`, all splits
    non-empty, leakage-free. Each task got a spec+quality review
    (all clean, Minor-only). *Next: final whole-branch review.*
  - **Implementation note (spec refinement):** §4/M7 anticipated that
    `config_postprocess` would synthesize the default channel, default
    `target_cols`, and validate `target_cols ⊆ channel columns`. In the
    implementation that synthesis + subset validation lives in
    `_resolve_channels` in the loader (`config.py` only declares the
    argparse flags) — functionally equivalent and keeps validation next
    to its point of use. `_parse_channels` also enforces the single-`:`
    `PATH:COL` rule there.
- **2026-07-01** — Final whole-branch review (opus): **READY TO MERGE** —
  no Critical/Important. Cross-task composition, backward compatibility
  (legacy CSV/`ratio` path untouched; univariate exact counts; legacy
  symbols importable), and mixed-cadence (5-min + 1-min) union-join
  correctness all verified (per-channel resample-mean ignores the other
  channel's union NaNs; `.count()` mask is per-column; AND-rule row
  validity; shared UTC timeline). 26/26 tests pass on merged state. All
  minors deferred. **Status: complete.**

## 9. Follow-on: forecast flexibility (variable length + multi-target)

After the join merged, an end-to-end check of the **model** path (not just the
loader) surfaced two gaps in adjustable time-series length and multi-target
output. Both fixed on branch `fix/lstm-multistep-forecast`, merged to master
(`f343e65`).

- **Variable forecast length — `fix(lstm)` `0e73291`.** The custom
  `src/model/lstm.py::forecast` projected only the last LSTM step
  (`o[:,-1:]`), so it always emitted **1** step regardless of `pred_len` —
  any horizon >1 mismatched the `(B,pred_len,T)` target (the legacy hourly
  `pred_len=1` hid this). Fix: project the last hidden state to
  `pred_len*enc_in` and reshape to `[B,pred_len,enc_in]`. The THU-ML models
  (timesnet, patchtst, itransformer, micn, …) already honored `pred_len` via
  their forecast heads; LSTM was the lone exception. `--seq_len` (past
  reference) and `--pred_len` (horizon) are in **5-min steps**.

- **Multi-target output (M8) — `feat(model)` `c87c813`.** The loader already
  emitted `y=(B,pred_len,T)` for `len(target_cols)==T`, but
  `StandardForecastAdapter` reduced the prediction to a single column via
  `target_index`. Threaded the full list end-to-end:
  - `DataBundle` gains `target_indices: List[int]`; `__post_init__` defaults
    it to `[target_index]` (backward compat with the legacy CSV path and any
    single-target caller).
  - `_setup_parquet` passes the resolved `target_idx` list.
  - `build_model(..., target_indices)` and the adapter select
    `pred[..., target_indices]` → `(B,pred_len,T)`. The base model still emits
    all `enc_in=C` channels; the adapter subsets — no model-internal change.
  - `exp.py` forwards `data_bundle.target_indices`. The `pl_model` runner is
    target-count-agnostic (`MSELoss(pred,y)`), so it needed no change.
  - Pick multi-target with `--target_cols p_gt10 xrs_long`.

- **Verification.** Real multivar multi-target (channels `p_gt10 + xrs_long`,
  both as targets): `input_size==2`, `output_size==2`,
  `target_indices==[0,1]`, `pred == y == (B,12,2)`, loss backprops.
  Regression tests: `data/test_loader.py` **29** (added `target_indices`
  default/explicit + real multi-target setup), `model/test_lstm.py` **4**
  (multi-step horizons + multi-target shape/backprop), `configs/test_config.py`
  **6** — all pass. Univariate and single-target multivariate paths unchanged.

**Capability matrix (as-merged):**

| Requirement | Flag | Status |
|---|---|---|
| Univariate / multivariate **input** | `--channels PATH:COL ...` | ✅ |
| Univariate / multivariate **target** | `--target_cols COL ...` | ✅ (loader + model) |
| CV interval split | `--split_type year_half --n_fold --fold_numb` | ✅ |
| Past-reference length | `--seq_len` (5-min steps) | ✅ |
| Forecast horizon | `--pred_len` (5-min steps) | ✅ |
