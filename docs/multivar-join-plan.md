# Multivariate Common-Grid Join — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `sw-framework-v002`'s `DataModule` to build a multivariate input by joining N parquet channels on a common 5-min mean grid, with configurable univariate-or-multivariate prediction targets; a single channel reproduces the current univariate behavior.

**Architecture:** Generalize the existing parquet path: `_grid_and_starts` becomes frame-based (`resample("{cadence}min").mean()` per term, `min_bin_count` gating, transform AFTER mean); `WindowDataset` accepts a 2-D `(G,C)` grid + a `target_idx`; `_setup_parquet` resolves `--channels`, joins them, and emits `x=(seq,C)`, `y=(pred,T)`. The legacy CSV/`ratio` path is untouched.

**Tech Stack:** Python 3.12, pandas 3.0.3, pyarrow 24.0, numpy 2.4.3, torch 2.4.1, unittest.

## Global Constraints

- Work ONLY in `workdir/sw-framework-v002/`; never modify `sw-framework-v001/` or the read-only KASI DB at `/NAS/ioGuard3/vol0/...`.
- Data at `/NAS/ioGuard3/vol3/spaceai/SW_framework/data/goes_data/processed/`: `kasi_swpc_particle_5m_v02.parquet` (`p_gt10`), `kasi_swpc_xray_1m_v02.parquet` (`xrs_long`).
- Grid = `resample("{cadence_min}min").mean()` per term; a bin is valid for a channel iff it has `>= min_bin_count` native samples (mask via `resample(...).count()`), else NaN.
- Transform order: **mean on RAW flux first, THEN log10** (`np.log10(mean.where(mean > 0))`).
- Row validity = ALL channels present (`~np.isnan(vals).any(axis=1)`); window rule unchanged (whole `L=seq_len+pred_len` valid, never crossing a half-year term boundary).
- Backward compat: single channel + single target reproduces univariate counts EXACTLY (train 1,076,731 / val 370,516 / test 350,429 at `p_gt10`, primary, log10, year_half, fold 0, seq 288 + pred 12).
- `seq_len`/`pred_len` are in 5-min steps. Channel column names are assumed distinct.
- Run tests with `python3.12` from `workdir/sw-framework-v002/src`: `python3.12 -m unittest data.test_loader configs.test_config -v`.
- Commit messages end with: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` (commit via `git -c user.name='Claude Code' -c user.email='noreply@anthropic.com' commit`).
- After each task append a dated line to `docs/sw-framework-v002/multivar-join-design.md` §8.

---

### Task 1: Config flags `--channels`, `--target_cols`, `--min_bin_count`

**Files:**
- Modify: `src/configs/config.py` (add args after line 34, the `--cadence_min` line)
- Test: `src/configs/test_config.py` (append)

**Interfaces:**
- Produces: `config.channels: list[str]|None` (default None), `config.target_cols: list[str]|None` (default None), `config.min_bin_count: int` (default 1).

- [ ] **Step 1: Write the failing test**

Append to `src/configs/test_config.py`:

```python
class TestMultivarFlags(unittest.TestCase):
    def _parse(self, *extra):
        argv = ["--data_path", "x.parquet", "--target_col", "p_gt10", *extra]
        return exp_parser().parse_args(argv)

    def test_defaults(self):
        c = self._parse()
        self.assertIsNone(c.channels)
        self.assertIsNone(c.target_cols)
        self.assertEqual(c.min_bin_count, 1)

    def test_lists_parse(self):
        c = self._parse("--channels", "a.parquet:p_gt10", "b.parquet:xrs_long",
                        "--target_cols", "p_gt10", "--min_bin_count", "3")
        self.assertEqual(c.channels, ["a.parquet:p_gt10", "b.parquet:xrs_long"])
        self.assertEqual(c.target_cols, ["p_gt10"])
        self.assertEqual(c.min_bin_count, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd workdir/sw-framework-v002/src && python3.12 -m unittest configs.test_config -v`
Expected: FAIL (`AttributeError: 'Namespace' object has no attribute 'channels'`).

- [ ] **Step 3: Add the flags**

In `src/configs/config.py`, after line 34 (the `--cadence_min` `add_argument` call, before the `# model comparison` comment):

```python
    parser.add_argument("--channels", nargs="*", default=None,
                        help="multivariate input channels as PATH:COL (>=1). "
                             "If omitted, uses --data_path:--target_col.")
    parser.add_argument("--target_cols", nargs="*", default=None,
                        help="prediction target columns (subset of channels). "
                             "Default: first channel's column.")
    parser.add_argument("--min_bin_count", type=int, default=1,
                        help="min native samples per resample bin to keep it")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd workdir/sw-framework-v002/src && python3.12 -m unittest configs.test_config -v`
Expected: PASS (all TestConfig + TestMultivarFlags).

- [ ] **Step 5: Commit**

```bash
cd workdir/sw-framework-v002
git add src/configs/config.py src/configs/test_config.py
git commit -m "feat(config): add --channels/--target_cols/--min_bin_count

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `_parse_channels` and `_resolve_channels`

**Files:**
- Modify: `src/data/loader.py` (add after `_read_table`, around line 16)
- Test: `src/data/test_loader.py` (append)

**Interfaces:**
- Produces:
  - `_parse_channels(specs: list[str]) -> list[tuple[str,str]]` — split each `PATH:COL` on the LAST `:`; raise `ValueError` if no `:` or empty part.
  - `_resolve_channels(cfg) -> tuple[list[tuple[str,str]], list[str]]` — returns `(channels, target_cols)`. `channels` from `cfg.channels` or `[(cfg.data_path, cfg.target_col)]`; `target_cols` from `cfg.target_cols` or `[channels[0][1]]`; raise if any target col not among channel columns.

- [ ] **Step 1: Write the failing test**

Append to `src/data/test_loader.py`:

```python
from types import SimpleNamespace as _NS
from data.loader import _parse_channels, _resolve_channels


class TestParseChannels(unittest.TestCase):
    def test_parse_ok(self):
        self.assertEqual(
            _parse_channels(["/a/b.parquet:p_gt10", "/c.parquet:xrs_long"]),
            [("/a/b.parquet", "p_gt10"), ("/c.parquet", "xrs_long")])

    def test_parse_no_colon_errors(self):
        with self.assertRaises(ValueError):
            _parse_channels(["/a/b.parquet"])

    def test_resolve_defaults_from_data_path(self):
        cfg = _NS(channels=None, target_cols=None,
                  data_path="p.parquet", target_col="p_gt10")
        ch, tgt = _resolve_channels(cfg)
        self.assertEqual(ch, [("p.parquet", "p_gt10")])
        self.assertEqual(tgt, ["p_gt10"])

    def test_resolve_explicit_and_target_subset(self):
        cfg = _NS(channels=["p.parquet:p_gt10", "x.parquet:xrs_long"],
                  target_cols=["xrs_long"], data_path=None, target_col=None)
        ch, tgt = _resolve_channels(cfg)
        self.assertEqual([c for _, c in ch], ["p_gt10", "xrs_long"])
        self.assertEqual(tgt, ["xrs_long"])

    def test_resolve_bad_target_errors(self):
        cfg = _NS(channels=["p.parquet:p_gt10"], target_cols=["nope"],
                  data_path=None, target_col=None)
        with self.assertRaises(ValueError):
            _resolve_channels(cfg)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd workdir/sw-framework-v002/src && python3.12 -m unittest data.test_loader -v`
Expected: FAIL (`ImportError: cannot import name '_parse_channels'`).

- [ ] **Step 3: Implement**

Add to `src/data/loader.py` after `_read_table` (after line 15):

```python
def _parse_channels(specs):
    out = []
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"channel spec must be PATH:COL, got {spec!r}")
        path, col = spec.rsplit(":", 1)
        if not path or not col:
            raise ValueError(f"invalid channel spec {spec!r}")
        out.append((path, col))
    return out


def _resolve_channels(cfg):
    if getattr(cfg, "channels", None):
        channels = _parse_channels(cfg.channels)
    else:
        channels = [(cfg.data_path, cfg.target_col)]
    cols = [c for _, c in channels]
    if getattr(cfg, "target_cols", None):
        target_cols = list(cfg.target_cols)
    else:
        target_cols = [cols[0]]
    for t in target_cols:
        if t not in cols:
            raise ValueError(f"target col {t!r} not among channels {cols}")
    return channels, target_cols
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd workdir/sw-framework-v002/src && python3.12 -m unittest data.test_loader -v`
Expected: PASS (prior tests + 5 new).

- [ ] **Step 5: Commit**

```bash
cd workdir/sw-framework-v002
git add src/data/loader.py src/data/test_loader.py
git commit -m "feat(loader): add _parse_channels and _resolve_channels

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `DataBundle.output_size` + generalize `WindowDataset` to 2-D + `target_idx`

**Files:**
- Modify: `src/data/loader.py` (`WindowDataset` at lines 90-107; `DataBundle` at lines 110-116)
- Test: `src/data/test_loader.py` (append)

**Interfaces:**
- Produces:
  - `WindowDataset(values, starts, seq_len, pred_len, target_idx=None)` — `values` may be 1-D `(G,)` (promoted to `(G,1)`) or 2-D `(G,C)`. `__getitem__` returns `x=(seq_len,C)`, `y=(pred_len,T)` where `T=len(target_idx)` (default all C columns).
  - `DataBundle` gains `output_size: int = 1` (last field, defaulted → non-breaking).

- [ ] **Step 1: Write the failing test**

Append to `src/data/test_loader.py`:

```python
class TestWindowDatasetMV(unittest.TestCase):
    def test_2d_values_and_target_idx(self):
        values = np.arange(20.0).reshape(10, 2)   # (G=10, C=2)
        ds = WindowDataset(values, np.array([0, 2]), seq_len=2, pred_len=1,
                           target_idx=[1])
        x, y = ds[1]                               # start=2, L=3 rows 2,3,4
        self.assertEqual(tuple(x.shape), (2, 2))   # (seq, C)
        self.assertEqual(tuple(y.shape), (1, 1))   # (pred, T=1)
        np.testing.assert_array_equal(x.numpy(), np.array([[4., 5.], [6., 7.]]))
        np.testing.assert_array_equal(y.squeeze(-1).numpy(), np.array([9.]))  # col 1 of row 4

    def test_1d_values_still_work(self):
        ds = WindowDataset(np.arange(10.0), np.array([0]), seq_len=2, pred_len=1)
        x, y = ds[0]
        self.assertEqual(tuple(x.shape), (2, 1))
        self.assertEqual(tuple(y.shape), (1, 1))


class TestDataBundleOutputSize(unittest.TestCase):
    def test_default_output_size(self):
        from data.loader import DataBundle
        b = DataBundle(train_loader=None, val_loader=None, test_loader=None,
                       input_size=2, target_index=0)
        self.assertEqual(b.output_size, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd workdir/sw-framework-v002/src && python3.12 -m unittest data.test_loader.TestWindowDatasetMV data.test_loader.TestDataBundleOutputSize -v`
Expected: FAIL (`__init__() got an unexpected keyword argument 'target_idx'` and/or missing `output_size`).

- [ ] **Step 3: Implement**

Replace the `WindowDataset` class body (lines 90-107) with:

```python
class WindowDataset(Dataset):
    """Lazy sliding-window dataset over a (G,) or (G,C) value array and
    precomputed, gap-free start positions. Slices windows on access."""

    def __init__(self, values, starts, seq_len, pred_len, target_idx=None) -> None:
        v = np.asarray(values)
        if v.ndim == 1:
            v = v[:, None]
        self.values = torch.as_tensor(v, dtype=torch.float32)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.target_idx = (list(range(v.shape[1])) if target_idx is None
                           else list(target_idx))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        s = int(self.starts[idx])
        L = self.seq_len + self.pred_len
        w = self.values[s:s + L]
        return w[:self.seq_len], w[self.seq_len:][:, self.target_idx]
```

Replace the `DataBundle` dataclass (lines 110-116) with:

```python
@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    input_size: int
    target_index: int
    output_size: int = 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd workdir/sw-framework-v002/src && python3.12 -m unittest data.test_loader -v`
Expected: PASS (all prior WindowDataset/setup tests still pass — the 1-D path and `target_idx=None` preserve old shapes — plus 3 new).

- [ ] **Step 5: Commit**

```bash
cd workdir/sw-framework-v002
git add src/data/loader.py src/data/test_loader.py
git commit -m "feat(loader): WindowDataset 2-D + target_idx; DataBundle.output_size

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Frame-based `_grid_and_starts` (resample-mean + min_bin_count + transform)

**Files:**
- Modify: `src/data/loader.py` (`_grid_and_starts` at lines 57-75)
- Test: `src/data/test_loader.py` (append)

**Interfaces:**
- Produces: `_grid_and_starts(data, terms, cadence_min, seq_len, pred_len, split_type, transform="none", min_bin_count=1) -> (values, starts)`. `data` is a `pd.Series` (→ 1 column) or `pd.DataFrame` (C columns). Per term: `resample("{cadence_min}min").mean()`, mask bins with `count < min_bin_count`, apply `log10` (mean.where>0) if `transform=="log10"`, row validity = all columns non-NaN. `values` shape `(G, C)`; `starts` never cross a term boundary. Backward compatible with the positional Series call used by prior tests.

- [ ] **Step 1: Write the failing test**

Append to `src/data/test_loader.py`:

```python
class TestGridResampleMean(unittest.TestCase):
    def _frame(self, times, **cols):
        idx = pd.to_datetime(times, utc=True)
        return pd.DataFrame(cols, index=idx)

    def test_mean_then_log10(self):
        # two 1-min samples in one 5-min bin: mean(10,1000)=505 -> log10(505)
        f = self._frame(["2020-01-01T00:00", "2020-01-01T00:01"], a=[10.0, 1000.0])
        values, starts = _grid_and_starts(f, ["2020-H1"], 5, 1, 0, "year_half",
                                          transform="log10", min_bin_count=1)
        # L=1 window on a single valid bin -> one row, value log10(505)
        self.assertEqual(values.shape, (1, 1))
        self.assertAlmostEqual(float(values[0, 0]), np.log10(505.0), places=6)

    def test_min_bin_count_masks_sparse_bin(self):
        # 4 one-min samples in a 5-min bin; require >=5 -> bin becomes NaN
        f = self._frame([f"2020-01-01T00:0{m}" for m in range(4)], a=[1.0, 2.0, 3.0, 4.0])
        values, starts = _grid_and_starts(f, ["2020-H1"], 5, 1, 0, "year_half",
                                          transform="none", min_bin_count=5)
        self.assertTrue(np.isnan(values[0, 0]))
        self.assertEqual(len(starts), 0)

    def test_multichannel_and_validity(self):
        # channel b missing its second bin -> that row invalid for AND rule
        idx = pd.date_range("2020-01-01T00:00", periods=3, freq="5min", tz="UTC")
        f = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, np.nan, 3.0]}, index=idx)
        values, starts = _grid_and_starts(f, ["2020-H1"], 5, 1, 0, "year_half",
                                          transform="none", min_bin_count=1)
        self.assertEqual(values.shape, (3, 2))
        np.testing.assert_array_equal(starts, np.array([0, 2]))  # row 1 invalid
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd workdir/sw-framework-v002/src && python3.12 -m unittest data.test_loader.TestGridResampleMean -v`
Expected: FAIL (`_grid_and_starts() got an unexpected keyword argument 'transform'`).

- [ ] **Step 3: Implement**

Replace `_grid_and_starts` (lines 57-75) with:

```python
def _grid_and_starts(data, terms, cadence_min, seq_len, pred_len, split_type,
                     transform="none", min_bin_count=1):
    frame = data.to_frame() if isinstance(data, pd.Series) else data
    L = seq_len + pred_len
    rule = f"{cadence_min}min"
    labels = _term_labels(frame.index, split_type)
    all_vals, all_starts, offset = [], [], 0
    for term in sorted(terms):
        sub = frame[labels == term]
        if sub.empty:
            continue
        mean = sub.resample(rule).mean()
        cnt = sub.resample(rule).count()
        mean = mean.mask(cnt < min_bin_count)
        if transform == "log10":
            with np.errstate(divide="ignore", invalid="ignore"):
                mean = np.log10(mean.where(mean > 0))
        elif transform != "none":
            raise ValueError(f"unknown transform: {transform}")
        vals = mean.to_numpy(dtype="float64")
        valid = ~np.isnan(vals).any(axis=1)
        starts = _valid_starts(valid, L)
        all_vals.append(vals)
        if len(starts):
            all_starts.append(starts + offset)
        offset += len(vals)
    C = frame.shape[1]
    values = np.concatenate(all_vals) if all_vals else np.empty((0, C), dtype="float64")
    starts = np.concatenate(all_starts) if all_starts else np.empty(0, dtype=np.int64)
    return values, starts
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd workdir/sw-framework-v002/src && python3.12 -m unittest data.test_loader -v`
Expected: PASS. The prior `TestGridAndStarts` positional Series calls still pass (defaults `transform="none"`, `min_bin_count=1`; resample-mean on 5-min data equals the old reindex — starts `[3,4]` and `[0,1,4,5]` unchanged; `values` now `(G,1)`, and `len(values)`/`np.isnan(values[2])` assertions still hold), plus 3 new.

- [ ] **Step 5: Commit**

```bash
cd workdir/sw-framework-v002
git add src/data/loader.py src/data/test_loader.py
git commit -m "feat(loader): frame-based _grid_and_starts (resample-mean, min_bin_count, transform)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Multivariate `_setup_parquet` + univariate regression + real integration

**Files:**
- Modify: `src/data/loader.py` (`_setup_parquet` at lines 268-297)
- Test: `src/data/test_loader.py` (append)

**Interfaces:**
- Consumes: `_read_table`, `_prepare_series`, `_resolve_channels`, `_term_labels`, `_fold_indices`, `_grid_and_starts` (frame form), `WindowDataset` (target_idx), `DataBundle` (output_size).
- Produces: `_setup_parquet` returns `DataBundle(input_size=C, target_index=target_idx[0], output_size=len(target_cols))`; per-split window counts printed.

- [ ] **Step 1: Write the failing test**

Append to `src/data/test_loader.py`:

```python
XRAY = os.path.join(PROC, "kasi_swpc_xray_1m_v02.parquet")


def _cfg_mv(**kw):
    base = dict(data_path=PARTICLE, target_col="p_gt10", channels=None,
                target_cols=None, time_col=None, role="primary",
                transform="log10", cadence_min=5, min_bin_count=1,
                split_type="year_half", n_fold=5, fold_numb=0,
                seq_len=288, pred_len=12, batch_size=32, shuffle_train=False,
                num_workers=0, feature_cols=None, train_ratio=0.7, val_ratio=0.15)
    base.update(kw)
    return SimpleNamespace(**base)


@unittest.skipUnless(os.path.exists(PARTICLE), "real parquet not present")
class TestSetupUnivariateRegression(unittest.TestCase):
    def test_counts_match_recorded(self):
        b = DataModule(_cfg_mv()).setup()
        self.assertEqual(b.input_size, 1)
        self.assertEqual(b.output_size, 1)
        self.assertEqual(len(b.train_loader.dataset), 1_076_731)
        self.assertEqual(len(b.val_loader.dataset), 370_516)
        self.assertEqual(len(b.test_loader.dataset), 350_429)


@unittest.skipUnless(os.path.exists(PARTICLE) and os.path.exists(XRAY),
                     "real parquet not present")
class TestSetupMultivar(unittest.TestCase):
    def test_two_channels_one_target(self):
        cfg = _cfg_mv(channels=[f"{PARTICLE}:p_gt10", f"{XRAY}:xrs_long"],
                      target_cols=["p_gt10"])
        b = DataModule(cfg).setup()
        self.assertEqual(b.input_size, 2)
        self.assertEqual(b.output_size, 1)
        for loader in (b.train_loader, b.val_loader, b.test_loader):
            self.assertGreater(len(loader.dataset), 0)
        xb, yb = next(iter(b.train_loader))
        self.assertEqual(tuple(xb.shape[1:]), (288, 2))
        self.assertEqual(tuple(yb.shape[1:]), (12, 1))
```

Note: `SimpleNamespace` and `PARTICLE`/`PROC` are already imported/defined in `test_loader.py` from earlier tasks.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd workdir/sw-framework-v002/src && python3.12 -m unittest data.test_loader.TestSetupMultivar -v`
Expected: FAIL (`_setup_parquet` ignores `channels` / no `output_size` wiring → input_size 1 not 2).

- [ ] **Step 3: Implement**

Replace `_setup_parquet` (lines 268-297) with:

```python
    def _setup_parquet(self) -> DataBundle:
        cfg = self.config
        time_col = cfg.time_col or "time_utc"
        channels, target_cols = _resolve_channels(cfg)
        cols, series_list = [], []
        for path, col in channels:
            df = _read_table(path, columns=[time_col, "role", col])
            for c in (time_col, col):
                if c not in df.columns:
                    raise ValueError(f"column '{c}' not found in {path}")
            series_list.append(_prepare_series(df, time_col, col, cfg.role, "none").rename(col))
            cols.append(col)
        frame = pd.concat(series_list, axis=1).sort_index()
        frame.columns = cols

        terms = sorted(pd.unique(_term_labels(frame.index, cfg.split_type)).tolist())
        if len(terms) < cfg.n_fold:
            raise ValueError(f"Not enough terms ({len(terms)}) for n_fold={cfg.n_fold}.")
        fold = _fold_indices(len(terms), cfg.n_fold, cfg.fold_numb)
        split_terms = {k: [terms[i] for i in idxs.tolist()] for k, idxs in fold.items()}
        target_idx = [cols.index(t) for t in target_cols]

        loaders = {}
        for name in ("train", "val", "test"):
            values, starts = _grid_and_starts(
                frame, split_terms[name], cfg.cadence_min, cfg.seq_len, cfg.pred_len,
                cfg.split_type, transform=cfg.transform, min_bin_count=cfg.min_bin_count)
            ds = WindowDataset(values, starts, cfg.seq_len, cfg.pred_len,
                               target_idx=target_idx)
            print(f"[DataModule] {name}: {len(split_terms[name])} terms, "
                  f"{len(ds):,} windows")
            loaders[name] = DataLoader(
                ds, batch_size=cfg.batch_size,
                shuffle=(cfg.shuffle_train if name == "train" else False),
                num_workers=cfg.num_workers)

        return DataBundle(
            train_loader=loaders["train"], val_loader=loaders["val"],
            test_loader=loaders["test"], input_size=len(cols),
            target_index=target_idx[0], output_size=len(target_cols))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd workdir/sw-framework-v002/src && python3.12 -m unittest data.test_loader -v`
Expected: PASS. `TestSetupUnivariateRegression` confirms resample-mean reproduces the recorded univariate counts exactly (single channel, particle 5-min, mean=identity, log10 after). `TestSetupMultivar` yields `input_size=2`, `output_size=1`, `x=(·,288,2)`, `y=(·,12,1)`, all splits non-empty.

- [ ] **Step 5: Legacy regression**

Run: `cd workdir/sw-framework-v002/src && python3.12 -c "from data.loader import DataModule, SequenceDataset, _build_windows; print('legacy symbols OK')"`
Expected: prints `legacy symbols OK`.

- [ ] **Step 6: Commit**

```bash
cd workdir/sw-framework-v002
git add src/data/loader.py src/data/test_loader.py
git commit -m "feat(loader): multivariate _setup_parquet (channel join, target subset)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Update docs progress log

**Files:**
- Modify: `docs/sw-framework-v002/multivar-join-design.md` §8

- [ ] **Step 1: Append progress entries**

Add under §8, one line per completed task summarizing Tasks 1–5 (config flags; `_parse_channels`/`_resolve_channels`; `WindowDataset` 2-D + `target_idx` + `DataBundle.output_size`; frame-based `_grid_and_starts`; multivariate `_setup_parquet`), the final test count, the univariate regression result (counts match), and the real multivariate integration result (`input_size=2`, `output_size=1`, per-split window counts printed at setup).

- [ ] **Step 2: Verify the design doc reads cleanly**

Run: `sed -n '/## 8/,$p' docs/sw-framework-v002/multivar-join-design.md`
Expected: entries present, no placeholders.

---

## Self-Review

**Spec coverage:** M1 channels config → Task 1; M2 target subset → Tasks 1,2,5 (`target_cols`, `target_idx`, `output_size`); M3 resample-mean grid → Task 4; M4 mean-then-log10 → Task 4 (`test_mean_then_log10`); M5 min_bin_count → Tasks 1,4 (`test_min_bin_count_masks_sparse_bin`); M6 AND validity + option-1 window → Task 4 (`test_multichannel_and_validity`); M7 backward compat → Tasks 2,5 (`_resolve_channels` default, `TestSetupUnivariateRegression` exact counts); M8 output ready (`y=(pred,T)`, `output_size`) → Tasks 3,5. Config summary → Task 1. All spec sections covered.

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every test asserts concrete values.

**Type consistency:** `_grid_and_starts(..., transform, min_bin_count)` returns `(values(G,C), starts)` consumed by `WindowDataset(values, starts, seq, pred, target_idx)` in Task 5; `_resolve_channels -> (channels, target_cols)` feeds `cols`/`target_idx`; `DataBundle(..., output_size=len(target_cols))` matches the Task 3 dataclass field. `WindowDataset` `target_idx` default None (Task 3) is what old call sites relied on before Task 5 passes it explicitly. Names consistent across tasks.
