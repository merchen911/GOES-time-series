# tslib.preprocessing — offline analysis / prep scripts

Standalone utilities that inspect the data and compute split statistics. They
read the processed parquet directory from the `SW_DATA_DIR` environment
variable (data is shared separately, not in the repo).

## What's here

- `term_split.py` — the shared split logic: `add_term` (half-year tagging),
  `make_fold_indices` (rotating leakage-free k-fold), `assign_split`,
  `split_frames`, `term_distribution`. The training loader reuses the same
  math.
- `count_fold_samples.py` — counts trainable windows per fold for univariate
  and multivariate channel configs.
- `make_latex_tables.py` — regenerates the paper tables (`SW_TABLES_OUT`).

## Add a script

1. Create `tslib/preprocessing/<name>.py`.
2. Reuse the split helpers: `from tslib.preprocessing import term_split as ts`.
3. Resolve data via `os.environ["SW_DATA_DIR"]` (don't hardcode absolute paths).
4. If it has logic worth locking down, add `test_<name>.py` next to it
   (`test_term_split.py` is the pattern) — it's picked up by
   `python -m unittest discover -t . -s tslib -p 'test_*.py'`.

See `term_split.py` and `test_term_split.py` for reference.
