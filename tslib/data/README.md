# tslib.data — adding a time series / channel

`loader.py` turns parquet files into windowed train/val/test loaders. Channels
are joined onto a common 5-minute grid; each is one `PATH:COL` pair.

## Pipeline (per channel, in `_setup_parquet`)

`_read_table` (read parquet) → `_prepare_series` (filter `role`, dedup, apply
transform) → outer-join all channels → half-year term split
(`_term_labels` + `_fold_indices`) → `_grid_and_starts` (resample-mean to the
5-min grid, `min_bin_count` gate, gap-free windows) → `WindowDataset`.

## Add a channel with the standard schema — no code needed

If the new series is a parquet with `time_utc`, a `role` column, and value
column(s), just add it to `--channels`:

```
--channels $SW_DATA_DIR/particle.parquet:p_gt10 $SW_DATA_DIR/newseries.parquet:my_col
--target_cols p_gt10        # or include my_col to forecast it too
```

`input_size` grows automatically; targets are any subset of channel columns.

## Add a channel with a different schema — small tweak

- Different time column → pass `--time_col`, or adjust `_read_table`.
- No `role` column, or different filtering → adjust `_prepare_series`
  (`role_col` / dedup / transform).
- A new transform beyond `none`/`log10` → add a branch in `_grid_and_starts`.

The split logic (`_resolve_channels`, `_fold_indices`) and windowing are
schema-agnostic and rarely need changes.

See `loader.py` and `test_loader.py` for the exact signatures and examples.
