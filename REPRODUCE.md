# Reproducing the benchmark

End-to-end steps to reproduce `results/results_regression.csv`,
`results/results_event.csv` and the leaderboard (README summary and
`results/LEADERBOARD.md`). Python 3.12 recommended
(netCDF / parquet libraries).

```bash
# 1. install the package (statsmodels is limited to <0.15, see the notes below)
pip install -e .

# 2. get the data (~254 MB) — see DATA.md
export SW_DATA_DIR=./data/goes_data/processed
python3.12 scripts/fetch_data.py

# 3. run the benchmark sweep: 1,740 runs =
#    3 tracks x 10 models x {seq 288,864,2016} x {pred 36,72,144,288} x 5 folds,
#    minus xPatch at seq 2016 (60 runs; its training diverges there).
#    The four forecast lengths are 3 h / 6 h / 12 h / 24 h on the 5-minute grid.
#    Every run saves its predictions (runs/bench/<cell>/pred/<model>.npz) and the
#    sweep writes two master tables, runs/bench/results_regression.csv and
#    runs/bench/results_event.csv. Both commands below plan the same 1,740 runs
#    and are resumable (a run counts as done when its row and npz exist).
python3.12 -m tslib.benchmark.scheduler --gpus 0 1 2    # multi-GPU job queue
#    …or the sequential driver (fold 0 only unless --folds is given):
python3.12 -m tslib.benchmark.driver --folds 0 1 2 3 4  # full sweep (long; GPU recommended)

#    …or a single configuration via the CLI:
python main.py --data_path $SW_DATA_DIR/kswrc_particle_5m.parquet \
    --target_col p_gt10 --seq_len 288 --pred_len 144 \
    --event_threshold 10 \
    --n_fold 5 --fold_numb 0 --models segrnn_thuml dlinear

# 4. (optional) re-score from the saved predictions — no GPU needed
python3.12 scripts/rescore.py --runs runs/bench --master-dir runs/bench --workers 8

# 5. regenerate the leaderboard (README summary + results/LEADERBOARD.md)
cp runs/bench/results_regression.csv runs/bench/results_event.csv results/
python3.12 scripts/make_leaderboard.py

# 6. (optional) rebuild the data from public NOAA sources instead of downloading
#    see data_pipeline/README.md
```

## Notes

- The split is a leakage-controlled term-based rotating k-fold (`year_half`
  tagging); every fold spans all solar-cycle phases. See `docs/benchmark-conditions.md`.
- Each run is evaluated twice: **regression** per target channel (`rmse`, `mae`,
  `msa`, `sspb`, `logr`) and **events** per channel and aggregation unit
  (`point`, `window`, `day`) at the operational thresholds (M-class flares, 10 pfu
  SPE) on the 2x2 contingency table (`tss`, `hss`, `pod`, `far`, `csi`, `f1`,
  `seds`, `sedi`, plus the raw counts). Undefined scores are NaN. Definitions:
  [`docs/metrics.md`](docs/metrics.md).
- **Two result files.** `results_regression.csv` has one row per run and channel,
  `results_event.csv` one row per run, channel and aggregation unit. The v1 files
  (`results/v1/results_master.csv` and the v004 table next to it) are kept unchanged.
- **Re-scoring.** Predictions are stored per run, so a metric can be added or
  changed without training again: edit `tslib/exp/metrics.py` and run
  `scripts/rescore.py` (step 4). It rebuilds `score/rows/<model>.json`, the
  comparison tables and the two master tables from the `pred/*.npz` files; use
  `--merge-masters DIR DIR --out DIR` to join master tables built from separate
  trees. Runs from before v2 (v1.0.0 and the v004 short-horizon sweep) have no
  prediction files; `scripts/dump_predictions.py` creates them from the
  checkpoints and `scripts/verify_archive.py` checks the result against the v1
  tables. The dump needs the old checkpoints and, in practice, a GPU; the check
  needs only the dumped files and the v1 table.
- **Run counts.** Step 3 runs 1,740 runs with either entry point (xPatch is
  skipped at the 7-day input, `seq_len` 2016, where it diverges). The released
  tables hold 1,746: those 1,740 plus the six diverged v1.0.0 xPatch runs at
  `seq_len` 2016, fold 0 (`pred_len` 144 and 288 in each of the three tracks),
  which stay in the tables as NaN rows. A fresh sweep therefore reproduces
  1,740 of the 1,746 runs.
- **`statsmodels<0.15`.** The `ar` model of the `statistic` strategy calls
  `AutoReg(..., old_names=False)`, an argument that statsmodels 0.15 removed.
  `requirements.txt` and `pyproject.toml` therefore pin `statsmodels<0.15`. Only
  the `statistic` strategy (`--forecast_strategy statistic`) imports statsmodels,
  so the neural benchmark is unaffected, but with 0.15 or later the `ar` model
  fails.
- Checkpoints are **not** distributed (56 GB); step 3 retrains from scratch.
  Prediction files are not distributed either (about 0.35 TB for all runs: the
  876 runs at 12 h and 24 h measured about 0.27 TB as uncompressed float32
  prediction and observation arrays, the 870 shorter-horizon runs add about
  0.07 TB).
