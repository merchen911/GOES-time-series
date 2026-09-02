# DataLoader ↔ Strict Half-Year Split — Design Spec

- **Date:** 2026-06-30
- **Status:** approved (design), implementation pending
- **Module:** `tslib/data/loader.py`
- **Scope:** wire the new KASI SWPC parquet + strict half-year (term) train/val/test split into the existing PyTorch-Lightning `DataModule`. **Univariate first** (multivariate deferred).

## 1. Goal & context

`tslib/data/loader.py::DataModule` already carries the split
machinery (`split_type ∈ {year_half, year, ratio}`, rotating k-fold via
`_fold_indices`), but it (a) reads **CSV**, (b) assumes a single **wide**
series (one row per timestamp), and (c) builds windows **without gap
awareness**. The ingested datasets are **long-format parquet** at
`data/goes_data/processed/` (multiple `series`/satellites share a timestamp),
5-min cadence, with scattered missingness.

This work upgrades the `DataModule` (in-place, in **v002**) to consume the
parquet, apply the strict half-year split at the *term* level (leakage-free),
and produce gap-free sliding windows for one target channel of the
`primary`-role series.

## 2. Decisions (locked during brainstorming)

| # | Decision |
|---|---|
| D1 | Extend the existing `DataModule` in `tslib/data/loader.py` in place. |
| D2 | **In-place** extension of `DataModule` (not a parallel module); existing `ratio`/CSV path preserved. |
| D3 | **Univariate first**; multivariate (particle+xray join) deferred. |
| D4 | Target transform is config-selectable: `--transform {none,log10}`, **default `log10`** (normalization left to model RevIN/StandardNorm). |
| D5 | Window rule = **option 1** (legacy `nonvalue2nan`): the whole `L=seq_len+pred_len` window must be non-missing, within one term. |
| D6 | Reuse the loader's own `_add_term_column`/`_fold_indices` (identical math to `preprocessing/term_split.py`); **no cross-repo import** → framework stays self-contained. |
| D7 | Docs in `docs/` are updated continuously (this file's §8 log). |

## 3. Architecture

Single touched file: **`tslib/data/loader.py`**
(plus a new test file). `exp.py`/`main.py` interface is unchanged
(`DataModule(config).setup() -> DataBundle`).

```
config (parquet path, target_col, role, transform, cadence_min,
        split_type=year_half, n_fold, fold_numb, seq_len, pred_len)
   │
   ▼
DataModule.setup()
   1. read parquet (cols: time_col, role, target only) | or CSV (legacy path)
   2. filter role==config.role ; dedup timestamps ; sort
   3. transform (log10|none) applied to target values
   4. _add_term_column (year_half) → terms
   5. _fold_indices(n_terms, n_fold, fold_numb) → {train,val,test} term sets
   6. per split: per term → reindex to regular `cadence_min` grid (NaN gaps)
                 → concat into one 1-D value array + term-boundary marks
                 → compute VALID window-start positions (full L non-NaN,
                   no term crossing)  ← gap-aware, option 1
   7. wrap each split in a LAZY WindowDataset(values, starts, seq_len, pred_len)
   8. log per-split sample counts (must match count_fold_samples.py)
   │
   ▼
DataBundle(train_loader, val_loader, test_loader, input_size=1, target_index=0)
```

## 4. Components

- **`_read_table(path, cols)`** — `.parquet`→`read_parquet(columns=cols)`, else
  `read_csv`. Keeps the legacy CSV path alive.
- **`_prepare_series(df, cfg)`** — role filter, dedup (`keep="last"`, matching
  `count_fold_samples.py`), sort, transform; returns a `time_utc`-indexed Series
  of the (possibly log10) target. On the parquet path `use_cols=[target]`
  (univariate) and `input_size=1`.
- **`_add_term_column` / `_fold_indices`** — unchanged (already present).
- **`_grid_and_starts(series, terms_in_split, cadence, L)`** — per term: build a
  regular-cadence grid, place values (NaN at gaps); concatenate into one array;
  return `(values, valid_start_positions)` where each start `i` satisfies
  window `[i, i+L)` is inside one term and fully non-NaN.
- **`WindowDataset(values, starts, seq_len, pred_len)`** — lazy
  `torch.utils.data.Dataset`; `__getitem__` slices
  `x=values[s:s+seq_len, None]`, `y=values[s+seq_len:s+L, None]`.
  Memory ≈ grid array (tens of MB) + int32 starts, **not** the dense window
  tensor (which would be multi-GB at seq_len=2016).

## 5. Config additions (`src/configs/config.py`)

| flag | default | meaning |
|---|---|---|
| `--role` | `primary` | which `role` rows to use |
| `--transform` | `log10` | `none` or `log10` on the target |
| `--cadence_min` | `5` | regular-grid step (minutes) for reindex/windowing |
| `--time_col` (default) | `time_utc` | (was `None`) parquet timestamp column |

`seq_len`/`pred_len` stay in **steps** (5-min units): 7d=2016, 3d=864, 1d=288.
Existing `ratio` validation untouched; for parquet the `year_half` path is canonical.

## 6. Testing (`src/data/test_loader.py`, new)

1. **gap exclusion** — synthetic series with a hole drops exactly the windows that span it; count matches a hand rule.
2. **leakage-free split** — no term appears in >1 split; per-split term sets disjoint.
3. **shapes** — `x:(N,seq_len,1)`, `y:(N,pred_len,1)`; `input_size==1`.
4. **transform** — log10 maps a known value; NaN stays NaN; `none` is identity.
5. **count parity** — total windows equal `count_fold_samples.py` for a small term subset.

## 7. Out of scope

Multivariate (particle+xray common-grid join), model/training-loop changes,
target inverse-transform at eval (model side). `ratio`/CSV legacy path kept but
not gap-upgraded.

## 8. Progress log (continuously updated)

- **2026-06-30** — Design approved. v002 created as copy of v001 (v001/.git had
  no commits; folder copy is the backup). Spec written. *Next: implementation
  plan (writing-plans).*
- **2026-06-30** — Implemented and executed task-by-task on branch
  `feat/dataloader-split` (baseline `12cae1e`):
  - Task 1 (`d269c4c`) — config flags `--role`/`--transform`/`--cadence_min`;
    parquet bypasses the `--time_col` requirement.
  - Task 2 (`ef0379c`) — `_read_table`, gap-aware `_valid_starts`.
  - Task 3 (`4031014`) — `_term_labels`, `_prepare_series` (role/dedup/log10).
  - Task 4 (`f70e39b`) — `_grid_and_starts` (per-term 5-min grid, gap-free,
    no cross-term windows).
  - Task 5 (`9c79d01`) — lazy `WindowDataset`.
  - Task 6 (`19161e0`) — `DataModule._setup_parquet`; `.parquet` routes here,
    legacy CSV/`ratio` path untouched.
  - Tests: **13/13 pass** (`python3.12 -m unittest data.test_loader`),
    legacy-import regression OK. Each task got a spec+quality review (all
    clean, Minor-only). Real-data check on `kasi_swpc_particle_5m_v02.parquet`
    (`p_gt10`, primary, log10, `year_half`, fold 0, seq 288 + pred 12):
    train 33 terms / 1,076,731 windows, val 11 / 370,516, test 11 / 350,429 —
    leakage-free term partition confirmed. *Next: final whole-branch review.*
- **2026-06-30** — Final whole-branch review (opus): **READY TO MERGE** — no
  Critical/Important, all Minors deferred, 17/17 tests pass, real-data counts
  match, leakage-free partition verified across n_term 3–39 × n_fold 3–7.
  Merged `feat/dataloader-split` → `master` (`ffbf4ef`, --no-ff); feature
  branch deleted; 17/17 tests pass on merged master. **Status: complete.**
  Deferred Minors: CSV-branch `columns`
  ignored (parquet path unaffected); no `__getitem__` bounds guard (starts are
  precomputed valid); a few test-scope/readability nits. Next stream:
  multivariate (particle+xray common-grid join), out of this scope.
