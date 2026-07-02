# DataLoader ↔ Strict Half-Year Split — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `sw-framework-v002`'s `DataModule` to load the KASI SWPC parquet, apply the strict half-year (term) train/val/test split, and emit gap-free univariate sliding windows for one `primary`-role target channel.

**Architecture:** In-place extension of `src/data/loader.py`. A parquet `data_path` routes to a new `_setup_parquet()` path: prepare one primary series (role filter + dedup + optional `log10`), tag half-year terms, assign terms to folds with the existing `_fold_indices`, reindex each term to a regular 5-min grid, and index the gap-free window start positions into a **lazy** `WindowDataset`. The legacy CSV/`ratio` path is untouched.

**Tech Stack:** Python 3.12, pandas 3.0.3, pyarrow 24.0, numpy 2.4.3, torch 2.4.1, unittest.

## Global Constraints

- Work ONLY in `workdir/sw-framework-v002/`; never modify `sw-framework-v001/` (backup) or the read-only KASI DB at `/NAS/ioGuard3/vol0/...`.
- Data lives at `/NAS/ioGuard3/vol3/spaceai/SW_framework/data/goes_data/processed/` (e.g. `kasi_swpc_particle_5m_v02.parquet`).
- Run tests with `python3.12` (3.10 lacks the libs). From `workdir/sw-framework-v002/src`: `python3.12 -m unittest data.test_loader -v`.
- Window rule = option 1: the whole `L = seq_len + pred_len` window must be non-missing and within one term. `seq_len`/`pred_len` are in 5-min STEPS (7d=2016, 3d=864, 1d=288).
- Do NOT cross-import `preprocessing/term_split.py`; keep the framework self-contained.
- After each task, append a dated line to `docs/sw-framework-v002/dataloader-split-design.md` §8 progress log.
- Commit messages end with: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

---

### Task 1: Config flags (`--role`, `--transform`, `--cadence_min`) + postprocess relax

**Files:**
- Modify: `workdir/sw-framework-v002/src/configs/config.py:20` (time_col stays), add flags after line 28; relax `config_postprocess` at line 70-71.
- Test: `workdir/sw-framework-v002/src/configs/test_config.py` (create)

**Interfaces:**
- Produces: `config.role: str="primary"`, `config.transform: str="log10"`, `config.cadence_min: int=5`. `config_postprocess` no longer raises for `split_type in {year,year_half}` when `data_path` ends with `.parquet` (time_col auto-defaults later).

- [ ] **Step 1: Write the failing test**

```python
# src/configs/test_config.py
import unittest
from configs.config import exp_parser, config_postprocess


class TestConfig(unittest.TestCase):
    def _parse(self, *extra):
        argv = ["--data_path", "x.parquet", "--target_col", "p_gt10", *extra]
        return exp_parser().parse_args(argv)

    def test_new_defaults(self):
        c = self._parse()
        self.assertEqual(c.role, "primary")
        self.assertEqual(c.transform, "log10")
        self.assertEqual(c.cadence_min, 5)

    def test_transform_choices(self):
        c = self._parse("--transform", "none")
        self.assertEqual(c.transform, "none")

    def test_parquet_year_half_without_time_col_ok(self):
        c = self._parse("--split_type", "year_half")  # no --time_col
        self.assertIs(config_postprocess(c), c)  # must not raise

    def test_csv_year_half_without_time_col_still_errors(self):
        argv = ["--data_path", "x.csv", "--target_col", "t", "--split_type", "year_half"]
        c = exp_parser().parse_args(argv)
        with self.assertRaises(ValueError):
            config_postprocess(c)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd workdir/sw-framework-v002/src && python3.12 -m unittest configs.test_config -v`
Expected: FAIL (`AttributeError: 'Namespace' object has no attribute 'role'`).

- [ ] **Step 3: Add the flags**

In `src/configs/config.py`, after line 28 (`--shuffle_train`):

```python
    parser.add_argument("--role", type=str, default="primary",
                        help="parquet: which 'role' rows to use (primary/secondary)")
    parser.add_argument("--transform", type=str, default="log10",
                        choices=["none", "log10"], help="target transform")
    parser.add_argument("--cadence_min", type=int, default=5,
                        help="regular-grid step in minutes for parquet windowing")
```

- [ ] **Step 4: Relax the postprocess guard**

Replace `config.py:70-71`:

```python
    if (config.split_type in {"year", "year_half"} and not config.time_col
            and not str(config.data_path).endswith(".parquet")):
        raise ValueError("time-based k-fold requires --time_col.")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd workdir/sw-framework-v002/src && python3.12 -m unittest configs.test_config -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
cd workdir/sw-framework-v002
git add src/configs/config.py src/configs/test_config.py
git commit -m "feat(config): add --role/--transform/--cadence_min; allow parquet year_half without --time_col

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Pure helpers `_read_table` and `_valid_starts`

**Files:**
- Modify: `workdir/sw-framework-v002/src/data/loader.py` (add functions after line 9 imports)
- Test: `workdir/sw-framework-v002/src/data/test_loader.py` (create)

**Interfaces:**
- Produces:
  - `_read_table(path: str, columns=None) -> pd.DataFrame` — `.parquet`→`read_parquet(columns=)`, else `read_csv`.
  - `_valid_starts(valid: np.ndarray, L: int) -> np.ndarray` — int64 positions `i` where `valid[i:i+L]` is all True.

- [ ] **Step 1: Write the failing test**

```python
# src/data/test_loader.py
import os, tempfile, unittest
import numpy as np
import pandas as pd
from data.loader import _read_table, _valid_starts

PROC = "/NAS/ioGuard3/vol3/spaceai/SW_framework/data/goes_data/processed"
PARTICLE = os.path.join(PROC, "kasi_swpc_particle_5m_v02.parquet")


class TestValidStarts(unittest.TestCase):
    def test_all_valid(self):
        v = np.ones(10, dtype=bool)
        np.testing.assert_array_equal(_valid_starts(v, 3), np.arange(8))

    def test_gap_excludes_spanning_windows(self):
        v = np.ones(10, dtype=bool); v[5] = False
        # windows of len 3 not covering index 5: starts 0,1,2 and 6,7
        np.testing.assert_array_equal(_valid_starts(v, 3), np.array([0, 1, 2, 6, 7]))

    def test_too_short(self):
        self.assertEqual(len(_valid_starts(np.ones(2, dtype=bool), 3)), 0)


class TestReadTable(unittest.TestCase):
    def test_parquet_columns(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.parquet")
            pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]}).to_parquet(p)
            df = _read_table(p, columns=["a", "c"])
            self.assertEqual(list(df.columns), ["a", "c"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd workdir/sw-framework-v002/src && python3.12 -m unittest data.test_loader -v`
Expected: FAIL (`ImportError: cannot import name '_read_table'`).

- [ ] **Step 3: Implement the helpers**

Add to `src/data/loader.py` after line 9:

```python
def _read_table(path, columns=None):
    if str(path).endswith(".parquet"):
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path)


def _valid_starts(valid, L):
    """int64 positions i where valid[i:i+L] is entirely True."""
    n = len(valid) - L + 1
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    invalid = (~np.asarray(valid)).astype(np.int64)
    csum = np.concatenate([[0], np.cumsum(invalid)])
    cnt = csum[L:L + n] - csum[:n]
    return np.nonzero(cnt == 0)[0].astype(np.int64)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd workdir/sw-framework-v002/src && python3.12 -m unittest data.test_loader -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
cd workdir/sw-framework-v002
git add src/data/loader.py src/data/test_loader.py
git commit -m "feat(loader): add _read_table and gap-aware _valid_starts helpers

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `_term_labels` and `_prepare_series`

**Files:**
- Modify: `workdir/sw-framework-v002/src/data/loader.py` (add after `_valid_starts`)
- Test: `workdir/sw-framework-v002/src/data/test_loader.py` (append)

**Interfaces:**
- Consumes: nothing from Task 2 directly.
- Produces:
  - `_term_labels(index: pd.DatetimeIndex, split_type: str) -> np.ndarray` — str labels (`"2003"` or `"2003-H1"`).
  - `_prepare_series(df, time_col, target_col, role, transform, role_col="role") -> pd.Series` — primary series indexed by tz-aware datetime, deduped (`keep="last"`), optionally `log10` (non-positive→NaN).

- [ ] **Step 1: Write the failing test**

```python
# append to src/data/test_loader.py
from data.loader import _term_labels, _prepare_series


class TestTermLabels(unittest.TestCase):
    def test_year_half(self):
        idx = pd.to_datetime(["2003-06-30T23:59", "2003-07-01T00:00", "2020-01-01"], utc=True)
        np.testing.assert_array_equal(
            _term_labels(idx, "year_half"), np.array(["2003-H1", "2003-H2", "2020-H1"]))

    def test_year(self):
        idx = pd.to_datetime(["2003-07-01"], utc=True)
        np.testing.assert_array_equal(_term_labels(idx, "year"), np.array(["2003"]))


class TestPrepareSeries(unittest.TestCase):
    def _df(self):
        return pd.DataFrame({
            "time_utc": pd.to_datetime(
                ["2020-01-01T00:00", "2020-01-01T00:00", "2020-01-01T00:05"], utc=True),
            "role": ["primary", "secondary", "primary"],
            "p_gt10": [10.0, 999.0, 100.0],
        })

    def test_role_filter_dedup_and_log10(self):
        s = _prepare_series(self._df(), "time_utc", "p_gt10", "primary", "log10")
        self.assertEqual(len(s), 2)               # secondary dropped
        self.assertAlmostEqual(float(s.iloc[0]), 1.0)   # log10(10)
        self.assertAlmostEqual(float(s.iloc[1]), 2.0)   # log10(100)

    def test_transform_none_identity(self):
        s = _prepare_series(self._df(), "time_utc", "p_gt10", "primary", "none")
        self.assertAlmostEqual(float(s.iloc[0]), 10.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd workdir/sw-framework-v002/src && python3.12 -m unittest data.test_loader -v`
Expected: FAIL (`ImportError: cannot import name '_term_labels'`).

- [ ] **Step 3: Implement**

Add to `src/data/loader.py` after `_valid_starts`:

```python
def _term_labels(index, split_type):
    year = index.year.to_numpy().astype("U4")
    if split_type == "year":
        return year
    if split_type == "year_half":
        half = np.where(index.month.to_numpy() <= 6, "H1", "H2")
        return np.char.add(np.char.add(year, "-"), half)
    raise ValueError(f"Unsupported split_type: {split_type}")


def _prepare_series(df, time_col, target_col, role, transform, role_col="role"):
    sub = df
    if role is not None and role_col in df.columns:
        sub = df[df[role_col] == role]
    sub = sub[[time_col, target_col]].copy()
    sub[time_col] = pd.to_datetime(sub[time_col])
    sub = (sub.dropna(subset=[time_col])
              .drop_duplicates(time_col, keep="last")
              .sort_values(time_col))
    s = sub.set_index(time_col)[target_col].astype("float64")
    if transform == "log10":
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.log10(s.where(s > 0))
    elif transform != "none":
        raise ValueError(f"unknown transform: {transform}")
    return s
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd workdir/sw-framework-v002/src && python3.12 -m unittest data.test_loader -v`
Expected: PASS (8 tests total).

- [ ] **Step 5: Commit**

```bash
cd workdir/sw-framework-v002
git add src/data/loader.py src/data/test_loader.py
git commit -m "feat(loader): add _term_labels and _prepare_series (role/dedup/log10)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `_grid_and_starts` (per-term regular grid + gap-free starts)

**Files:**
- Modify: `workdir/sw-framework-v002/src/data/loader.py` (add after `_prepare_series`)
- Test: `workdir/sw-framework-v002/src/data/test_loader.py` (append)

**Interfaces:**
- Consumes: `_term_labels`, `_valid_starts`.
- Produces: `_grid_and_starts(series, terms, cadence_min, seq_len, pred_len, split_type) -> (values: np.ndarray, starts: np.ndarray)`. `values` is the per-term regular-grid concatenation (NaN at gaps); `starts` are global positions whose `L`-window is gap-free AND inside one term (never crosses a term boundary).

- [ ] **Step 1: Write the failing test**

```python
# append to src/data/test_loader.py
from data.loader import _grid_and_starts


class TestGridAndStarts(unittest.TestCase):
    def _series(self, times, vals):
        return pd.Series(vals, index=pd.to_datetime(times, utc=True), dtype="float64")

    def test_gap_breaks_windows(self):
        # 6 points at 5-min, but one missing -> grid has a NaN hole
        t = ["2020-01-01T00:00", "2020-01-01T00:05", "2020-01-01T00:15",
             "2020-01-01T00:20", "2020-01-01T00:25", "2020-01-01T00:30"]
        s = self._series(t, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # 00:10 missing
        # grid 00:00..00:30 = 7 slots; slot idx2 (00:10) is NaN
        values, starts = _grid_and_starts(s, ["2020-H1"], 5, 2, 1, "year_half")
        self.assertEqual(len(values), 7)
        self.assertTrue(np.isnan(values[2]))
        # L=3 windows avoiding idx2: starts 0? no (covers 0,1,2). valid: 3,4 (->len? n=5)
        np.testing.assert_array_equal(starts, np.array([3, 4]))

    def test_no_cross_term_boundary(self):
        # two terms; each fully dense; window must not span the join
        t1 = pd.date_range("2020-06-29", periods=4, freq="5min", tz="UTC")   # H1
        t2 = pd.date_range("2020-07-01", periods=4, freq="5min", tz="UTC")   # H2
        s = pd.Series(np.arange(8.0),
                      index=t1.append(t2), dtype="float64")
        values, starts = _grid_and_starts(s, ["2020-H1", "2020-H2"], 5, 2, 1, "year_half")
        L = 3
        # term grids are length 4 each, concatenated -> offsets 0 and 4
        # valid starts per term: [0,1], second term +4 -> [4,5]
        np.testing.assert_array_equal(starts, np.array([0, 1, 4, 5]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd workdir/sw-framework-v002/src && python3.12 -m unittest data.test_loader -v`
Expected: FAIL (`ImportError: cannot import name '_grid_and_starts'`).

- [ ] **Step 3: Implement**

Add to `src/data/loader.py` after `_prepare_series`:

```python
def _grid_and_starts(series, terms, cadence_min, seq_len, pred_len, split_type):
    L = seq_len + pred_len
    step = pd.Timedelta(minutes=cadence_min)
    labels = _term_labels(series.index, split_type)
    all_vals, all_starts, offset = [], [], 0
    for term in sorted(terms):
        sub = series[labels == term]
        if sub.empty:
            continue
        grid = pd.date_range(sub.index.min(), sub.index.max(), freq=step)
        g = sub.reindex(grid).to_numpy(dtype="float64")
        starts = _valid_starts(~np.isnan(g), L)
        all_vals.append(g)
        if len(starts):
            all_starts.append(starts + offset)
        offset += len(g)
    values = np.concatenate(all_vals) if all_vals else np.empty(0, dtype="float64")
    starts = np.concatenate(all_starts) if all_starts else np.empty(0, dtype=np.int64)
    return values, starts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd workdir/sw-framework-v002/src && python3.12 -m unittest data.test_loader -v`
Expected: PASS (10 tests total).

- [ ] **Step 5: Commit**

```bash
cd workdir/sw-framework-v002
git add src/data/loader.py src/data/test_loader.py
git commit -m "feat(loader): add _grid_and_starts (per-term grid, gap-free window starts)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Lazy `WindowDataset`

**Files:**
- Modify: `workdir/sw-framework-v002/src/data/loader.py` (add after `SequenceDataset`, around line 22)
- Test: `workdir/sw-framework-v002/src/data/test_loader.py` (append)

**Interfaces:**
- Produces: `WindowDataset(values, starts, seq_len, pred_len)` — `Dataset`; `__len__()==len(starts)`; `__getitem__(i)->(x,y)` with `x:(seq_len,1)`, `y:(pred_len,1)` float32, sliced from `values[starts[i]:starts[i]+L]`.

- [ ] **Step 1: Write the failing test**

```python
# append to src/data/test_loader.py
import torch
from data.loader import WindowDataset


class TestWindowDataset(unittest.TestCase):
    def test_shapes_and_values(self):
        values = np.arange(10.0)
        ds = WindowDataset(values, np.array([0, 2]), seq_len=2, pred_len=1)
        self.assertEqual(len(ds), 2)
        x, y = ds[1]                      # start=2, L=3 -> [2,3,4]
        self.assertEqual(tuple(x.shape), (2, 1))
        self.assertEqual(tuple(y.shape), (1, 1))
        self.assertEqual(x.dtype, torch.float32)
        np.testing.assert_array_equal(x.squeeze(-1).numpy(), np.array([2.0, 3.0]))
        np.testing.assert_array_equal(y.squeeze(-1).numpy(), np.array([4.0]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd workdir/sw-framework-v002/src && python3.12 -m unittest data.test_loader.TestWindowDataset -v`
Expected: FAIL (`ImportError: cannot import name 'WindowDataset'`).

- [ ] **Step 3: Implement**

Add to `src/data/loader.py` after the `SequenceDataset` class:

```python
class WindowDataset(Dataset):
    """Lazy sliding-window dataset over a 1-D value array and precomputed,
    gap-free start positions. Slices windows on access (low memory)."""

    def __init__(self, values, starts, seq_len, pred_len) -> None:
        self.values = torch.as_tensor(np.asarray(values), dtype=torch.float32)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        s = int(self.starts[idx])
        L = self.seq_len + self.pred_len
        w = self.values[s:s + L].unsqueeze(-1)
        return w[:self.seq_len], w[self.seq_len:]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd workdir/sw-framework-v002/src && python3.12 -m unittest data.test_loader.TestWindowDataset -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd workdir/sw-framework-v002
git add src/data/loader.py src/data/test_loader.py
git commit -m "feat(loader): add lazy WindowDataset

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Wire `DataModule.setup()` parquet branch + integration test

**Files:**
- Modify: `workdir/sw-framework-v002/src/data/loader.py` (`DataModule.setup` at line 83; add `_setup_parquet`)
- Test: `workdir/sw-framework-v002/src/data/test_loader.py` (append)

**Interfaces:**
- Consumes: `_read_table`, `_prepare_series`, `_term_labels`, `_fold_indices`, `_grid_and_starts`, `WindowDataset`.
- Produces: `DataModule.setup()` returns a `DataBundle` with `input_size=1, target_index=0` for parquet inputs; per-split window counts printed.

- [ ] **Step 1: Write the failing test**

```python
# append to src/data/test_loader.py
from types import SimpleNamespace
from data.loader import DataModule


def _cfg(**kw):
    base = dict(data_path=PARTICLE, target_col="p_gt10", time_col=None,
                role="primary", transform="log10", cadence_min=5,
                split_type="year_half", n_fold=5, fold_numb=0,
                seq_len=288, pred_len=12, batch_size=32, shuffle_train=False,
                num_workers=0, feature_cols=None,
                train_ratio=0.7, val_ratio=0.15)
    base.update(kw)
    return SimpleNamespace(**base)


@unittest.skipUnless(os.path.exists(PARTICLE), "real parquet not present")
class TestSetupParquet(unittest.TestCase):
    def test_bundle_shapes_and_nonempty(self):
        b = DataModule(_cfg()).setup()
        self.assertEqual(b.input_size, 1)
        self.assertEqual(b.target_index, 0)
        for loader in (b.train_loader, b.val_loader, b.test_loader):
            self.assertGreater(len(loader.dataset), 0)
        xb, yb = next(iter(b.train_loader))
        self.assertEqual(tuple(xb.shape[1:]), (288, 1))
        self.assertEqual(tuple(yb.shape[1:]), (12, 1))

    def test_leakage_free_terms(self):
        from data.loader import _read_table, _prepare_series, _term_labels, _fold_indices
        c = _cfg()
        s = _prepare_series(_read_table(c.data_path, [ "time_utc", "role", c.target_col]),
                            "time_utc", c.target_col, c.role, c.transform)
        terms = sorted(pd.unique(_term_labels(s.index, c.split_type)).tolist())
        fold = _fold_indices(len(terms), c.n_fold, c.fold_numb)
        sets = [set(idx.tolist()) for idx in fold.values()]
        self.assertEqual(sum(len(x) for x in sets), len(terms))   # partition
        self.assertEqual(len(sets[0] & sets[1]), 0)
        self.assertEqual(len(sets[0] & sets[2]), 0)
        self.assertEqual(len(sets[1] & sets[2]), 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd workdir/sw-framework-v002/src && python3.12 -m unittest data.test_loader.TestSetupParquet -v`
Expected: FAIL (setup still reads CSV → error reading parquet as csv, or wrong input_size).

- [ ] **Step 3: Route parquet in `setup` and add `_setup_parquet`**

In `src/data/loader.py`, make `setup` branch at its top (line 83). Replace the first line `df = pd.read_csv(self.config.data_path)` region by inserting the branch BEFORE it:

```python
    def setup(self) -> DataBundle:
        if str(self.config.data_path).endswith(".parquet"):
            return self._setup_parquet()
        df = pd.read_csv(self.config.data_path)
        # ... (existing CSV/legacy body unchanged) ...
```

Add the new method to the `DataModule` class (after `setup`):

```python
    def _setup_parquet(self) -> DataBundle:
        cfg = self.config
        time_col = cfg.time_col or "time_utc"
        df = _read_table(cfg.data_path, columns=[time_col, "role", cfg.target_col])
        for c in (time_col, cfg.target_col):
            if c not in df.columns:
                raise ValueError(f"column '{c}' not found in {cfg.data_path}")
        series = _prepare_series(df, time_col, cfg.target_col, cfg.role, cfg.transform)
        terms = sorted(pd.unique(_term_labels(series.index, cfg.split_type)).tolist())
        if len(terms) < cfg.n_fold:
            raise ValueError(f"Not enough terms ({len(terms)}) for n_fold={cfg.n_fold}.")
        fold = _fold_indices(len(terms), cfg.n_fold, cfg.fold_numb)
        split_terms = {k: [terms[i] for i in idxs.tolist()] for k, idxs in fold.items()}

        loaders = {}
        for name in ("train", "val", "test"):
            values, starts = _grid_and_starts(
                series, split_terms[name], cfg.cadence_min,
                cfg.seq_len, cfg.pred_len, cfg.split_type)
            ds = WindowDataset(values, starts, cfg.seq_len, cfg.pred_len)
            print(f"[DataModule] {name}: {len(split_terms[name])} terms, "
                  f"{len(ds):,} windows")
            loaders[name] = DataLoader(
                ds, batch_size=cfg.batch_size,
                shuffle=(cfg.shuffle_train if name == "train" else False),
                num_workers=cfg.num_workers)

        return DataBundle(
            train_loader=loaders["train"], val_loader=loaders["val"],
            test_loader=loaders["test"], input_size=1, target_index=0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd workdir/sw-framework-v002/src && python3.12 -m unittest data.test_loader -v`
Expected: PASS (all tests, ~13).

- [ ] **Step 5: Full regression — legacy path still imports/parses**

Run: `cd workdir/sw-framework-v002/src && python3.12 -c "from data.loader import DataModule, SequenceDataset, _build_windows; print('legacy symbols OK')"`
Expected: prints `legacy symbols OK` (no import breakage).

- [ ] **Step 6: Commit**

```bash
cd workdir/sw-framework-v002
git add src/data/loader.py src/data/test_loader.py
git commit -m "feat(loader): route parquet to gap-aware half-year split (_setup_parquet)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Update docs progress log

**Files:**
- Modify: `docs/sw-framework-v002/dataloader-split-design.md` §8

- [ ] **Step 1: Append progress entries**

Add under §8, one line per completed task, e.g.:

```markdown
- **2026-06-30** — Implemented Tasks 1–6: config flags; `_read_table`,
  `_valid_starts`, `_term_labels`, `_prepare_series`, `_grid_and_starts`,
  lazy `WindowDataset`; `DataModule._setup_parquet`. Tests: <N> pass
  (`python3.12 -m unittest data.test_loader`). Per-fold window counts logged
  at setup; verified leakage-free term partition. Legacy CSV/ratio path intact.
```

- [ ] **Step 2: Verify the design doc reads cleanly**

Run: `sed -n '/## 8/,$p' docs/sw-framework-v002/dataloader-split-design.md`
Expected: progress entries present, no placeholders.

---

## Self-Review

**Spec coverage:** D1 (v002 copy) done pre-plan; D2 in-place → Task 6; D3 univariate → Tasks 3–6 (single target, input_size=1); D4 transform → Tasks 1,3; D5 option-1 no-gap → Tasks 2,4; D6 self-contained (no cross-import) → all tasks use local helpers; D7 docs → Task 7. Config additions → Task 1. Components `_read_table`/`_prepare_series`/`_grid_and_starts`/`WindowDataset` → Tasks 2–5. Testing (gap, leakage, shapes, transform, integration) → Tasks 2,3,4,5,6. All spec sections covered.

**Placeholder scan:** No TBD/TODO; every code step has complete code; every test has assertions.

**Type consistency:** `_valid_starts`/`_grid_and_starts`/`WindowDataset`/`_term_labels`/`_prepare_series`/`_read_table` signatures match across definition (Tasks 2–5) and use (Task 6). `DataBundle(input_size=1, target_index=0)` matches the dataclass fields. Term-label strings (`"2020-H1"`) consistent between Task 3 and Task 4 tests.
