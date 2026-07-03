# tslib.configs — adding a CLI option

`config.py` defines `exp_parser()` (argparse flags) and `config_postprocess()`
(defaults + validation). The parsed config object is passed throughout the
framework.

## Add a flag

1. Add it in `exp_parser()`:
   ```python
   parser.add_argument("--my_flag", type=int, default=8)
   ```
2. (Optional) validate / derive it in `config_postprocess()` before
   `return config` — raise `ValueError` with a clear message on bad input.
   Registry-backed choices (models, losses, metrics) are validated here by
   looking them up in their registries.
3. Consume `config.my_flag` where needed.

Existing flags cover data (`--channels`, `--target_cols`, `--transform`,
`--min_bin_count`), windows (`--seq_len`, `--pred_len`, `--cadence_min`), the
split (`--split_type`, `--n_fold`, `--fold_numb`), and modeling (`--loss`,
`--metrics`, `--event_threshold`, `--sort_metric`, `--models`).

See `config.py` and `test_config.py` for the full list and validation examples.
