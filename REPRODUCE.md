# Reproducing the benchmark

End-to-end steps to reproduce `results/results_master.csv` and the README
leaderboard. Python 3.12 recommended (netCDF / parquet libraries).

```bash
# 1. install the package
pip install -e .

# 2. get the data (~254 MB) — see DATA.md
export SW_DATA_DIR=./data/goes_data/processed
python3.12 scripts/fetch_data.py

# 3. run the benchmark sweep
#    3 tracks x 10 models x {seq 288,864,2016} x {pred 144,288} x 5 folds.
#    The driver writes runs/bench/results_master.csv.
python3.12 -m tslib.benchmark.driver          # full sweep (long; GPU recommended)

#    …or a single configuration via the CLI:
python main.py --data_path $SW_DATA_DIR/kswrc_particle_5m.parquet \
    --target_col p_gt10 --seq_len 288 --pred_len 144 \
    --n_fold 5 --fold_numb 0 --models segrnn_thuml dlinear

# 4. regenerate the leaderboard from the results table
cp runs/bench/results_master.csv results/results_master.csv
python3.12 scripts/make_leaderboard.py

# 5. (optional) rebuild the data from public NOAA sources instead of downloading
#    see data_pipeline/README.md
```

## Notes

- The split is a leakage-controlled term-based rotating k-fold (`year_half`
  tagging); every fold spans all solar-cycle phases. See `docs/benchmark-conditions.md`.
- Event skill is scored at the operational thresholds (M/X flares, ≥10 pfu SPE)
  on the 2×2 contingency table; metric definitions are in the paper's appendix.
- Checkpoints are **not** distributed (56 GB); step 3 retrains from scratch.
