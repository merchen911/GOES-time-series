# GOES-time-series

A PyTorch forecasting framework for GOES space-weather time series (proton flux
and X-ray flux). It provides a leakage-free data pipeline and a pluggable model
layer aimed at short- to medium-horizon forecasting of solar energetic-particle
and flare-related quantities.

**Highlights**
- **Multivariate parquet loading** — join N channels (e.g. proton `p_gt10` +
  X-ray `xrs_long`) onto a common 5-minute grid.
- **Leakage-free cross-validation** — half-year (H1/H2) term tagging with a
  rotating k-fold split, so no time period is shared across train/val/test.
- **Configurable horizons** — independent look-back (`--seq_len`) and forecast
  (`--pred_len`) lengths, in 5-minute steps.
- **Single- or multi-target forecasting** — predict one channel or several at
  once (`--target_cols`).
- **Multiple backbones** — LSTM, TimesNet, PatchTST, iTransformer, MICN, and
  more, run and compared in one command.
- **PyTorch Lightning training** — neural models train under a real
  `pytorch_lightning` `Trainer`, with checkpointing and early stopping (see
  [`docs/lightning-migration.md`](docs/lightning-migration.md)).
- **Separate regression and event evaluation** — per-channel RMSE, MAE, MSA,
  SSPB and log R, plus event scores (TSS, HSS, POD, FAR, CSI, F1, SEDS, SEDI)
  per point, window and day. Undefined scores are NaN, not 0
  ([`docs/metrics.md`](docs/metrics.md)).
- **Saved predictions** — every run stores its test predictions, so new metrics
  are evaluated on existing runs without a GPU (`scripts/rescore.py`).

> **Data is not included** in this repository. Download it from Zenodo
> ([10.5281/zenodo.22245285](https://doi.org/10.5281/zenodo.22245285)) or
> reconstruct it from public NOAA sources — see [DATA.md](DATA.md). Point
> `SW_DATA_DIR` at the processed-parquet directory before running.

## Benchmark leaderboard

<!-- LEADERBOARD:START -->

_Each cell is the mean of a model's runs at that forecast length, over all folds and input lengths unless a † footnote says otherwise; runs with an undefined value are left out of the mean. `k/n` marks a cell where only k of the n runs of that forecast length (folds x input lengths) have a value, and † marks a model with fewer runs than that, whose rows are not directly comparable with the others (see the note under the section). RMSE (log10 flux) and MSA (median symmetric accuracy, %) are better lower; logR (correlation of the log10 values), HSS and TSS are better higher. HSS and TSS are per-day scores (one forecast per 24 h) at the M-class (SXR) and 10 pfu (proton) thresholds. Rows are ordered by the first channel's RMSE. Per-run results: [`results/results_regression.csv`](results/results_regression.csv), [`results/results_event.csv`](results/results_event.csv)._

### 3 h (pred_len 36)

**SXR 1--8 A (univariate)**

| Model | SXR RMSE | SXR MSA | SXR logR | SXR HSS | SXR TSS |
|---|---|---|---|---|---|
| SegRNN | 0.179 | 13.690 | 0.982 | 0.246 | 0.154 |
| iTransformer | 0.194 | 16.264 | 0.979 | 0.167 | 0.104 |
| RLinear | 0.202 | 16.918 | 0.978 | 0.227 | 0.152 |
| TiDE | 0.203 | 16.898 | 0.977 | 0.279 | 0.199 |
| PatchTST | 0.210 | 17.481 | 0.976 | 0.126 | 0.076 |
| xPatch† | 0.212 (10/15) | 17.288 (10/15) | 0.973 (10/15) | 0.191 (10/15) | 0.117 (10/15) |
| PatchMixer | 0.280 | 19.876 | 0.956 | 0.266 | 0.190 |
| FReTS | 0.313 | 45.208 | 0.951 | 0.077 | 0.074 |
| DLinear | 0.675 | 296.873 | 0.855 | 0.183 | 0.233 |
| TSMixer | 0.782 | 280.768 | 0.742 | 0.000 | 0.000 |

**Proton >=10 MeV (univariate)**

| Model | Proton RMSE | Proton MSA | Proton logR | Proton HSS | Proton TSS |
|---|---|---|---|---|---|
| SegRNN | 0.132 | 14.967 | 0.964 | 0.917 | 0.859 |
| xPatch† | 0.155 (10/15) | 18.679 (10/15) | 0.955 (10/15) | 0.897 (10/15) | 0.835 (10/15) |
| iTransformer | 0.183 | 17.267 | 0.923 | 0.793 | 0.727 |
| TSMixer | 0.272 | 39.185 | 0.859 | 0.555 | 0.493 |
| PatchTST | 0.280 | 19.893 | 0.801 | 0.591 | 0.544 |
| FReTS | 0.363 | 48.839 | 0.672 | 0.558 | 0.567 |
| TiDE | 0.377 | 19.617 | 0.817 | 0.680 | 0.865 |
| RLinear | 0.400 | 19.150 | 0.821 | 0.646 | 0.857 |
| DLinear | 0.421 | 48.081 | 0.824 | 0.696 | 0.870 |
| PatchMixer | 0.479 | 26.145 | 0.706 | 0.635 | 0.859 |

**Multivariate (joint proton + SXR)**

| Model | Proton RMSE | Proton MSA | Proton logR | Proton HSS | Proton TSS | SXR RMSE | SXR MSA | SXR logR | SXR HSS | SXR TSS |
|---|---|---|---|---|---|---|---|---|---|---|
| SegRNN | 0.112 | 12.683 | 0.959 | 0.840 | 0.785 | 0.176 | 14.573 | 0.974 | 0.242 | 0.154 |
| xPatch† | 0.135 (10/15) | 15.480 (10/15) | 0.945 (10/15) | 0.873 (10/15) | 0.823 (10/15) | 0.183 (10/15) | 15.083 (10/15) | 0.981 (10/15) | 0.257 (10/15) | 0.159 (10/15) |
| PatchMixer | 0.153 | 14.954 | 0.930 | 0.805 | 0.818 | 0.191 | 16.849 | 0.970 | 0.243 | 0.158 |
| RLinear | 0.163 | 14.287 | 0.915 | 0.800 | 0.825 | 0.203 | 19.454 | 0.963 | 0.261 | 0.170 |
| TiDE | 0.164 | 14.599 | 0.916 | 0.807 | 0.827 | 0.203 | 19.401 | 0.963 | 0.257 | 0.168 |
| DLinear | 0.167 | 14.877 | 0.917 | 0.835 | 0.802 | 0.245 | 39.956 | 0.959 | 0.254 | 0.186 |
| PatchTST | 0.186 | 14.232 | 0.879 | 0.657 | 0.610 | 0.204 | 21.481 | 0.964 | 0.061 | 0.037 |
| iTransformer | 0.192 | 15.369 | 0.878 | 0.610 | 0.552 | 0.217 | 22.266 | 0.957 | 0.071 | 0.046 |
| TSMixer | 0.512 | 108.880 | 0.354 | 0.291 | 0.227 | 0.779 | 177.285 | 0.655 | 0.040 | 0.039 |
| FReTS | 1.673 | 20.347 | 0.772 | 0.703 | 0.692 | 0.257 | 33.018 | 0.927 | 0.076 | 0.062 |

_† xPatch is left out at the 7-day input (seq_len 2016) because its training diverges there, so its cells average 10 runs (1-day and 3-day inputs) instead of all 15 runs of this forecast length and are not directly comparable with the other rows._

### 6 h (pred_len 72)

**SXR 1--8 A (univariate)**

| Model | SXR RMSE | SXR MSA | SXR logR | SXR HSS | SXR TSS |
|---|---|---|---|---|---|
| SegRNN | 0.196 | 16.578 | 0.979 | 0.175 | 0.107 |
| iTransformer | 0.209 | 18.738 | 0.976 | 0.104 | 0.062 |
| PatchTST | 0.213 | 19.341 | 0.975 | 0.078 | 0.047 |
| RLinear | 0.218 | 19.721 | 0.974 | 0.159 | 0.103 |
| TiDE | 0.219 | 19.625 | 0.974 | 0.183 | 0.121 |
| xPatch† | 0.220 (10/15) | 19.189 (10/15) | 0.971 (10/15) | 0.134 (10/15) | 0.080 (10/15) |
| PatchMixer | 0.286 | 22.669 | 0.954 | 0.181 | 0.127 |
| FReTS | 0.317 | 47.834 | 0.950 | 0.039 | 0.024 |
| DLinear | 0.792 | 346.929 | 0.798 | 0.164 | 0.184 |
| TSMixer | 0.882 | 263.265 | 0.880 | 0.008 | 0.006 |

**Proton >=10 MeV (univariate)**

| Model | Proton RMSE | Proton MSA | Proton logR | Proton HSS | Proton TSS |
|---|---|---|---|---|---|
| SegRNN | 0.155 | 15.607 | 0.951 | 0.874 | 0.792 |
| xPatch† | 0.176 (10/15) | 19.071 (10/15) | 0.942 (10/15) | 0.863 (10/15) | 0.782 (10/15) |
| iTransformer | 0.200 | 17.920 | 0.916 | 0.774 | 0.679 |
| PatchTST | 0.278 | 19.431 | 0.798 | 0.569 | 0.503 |
| TSMixer | 0.314 | 43.747 | 0.820 | 0.463 | 0.385 |
| FReTS | 0.401 | 49.014 | 0.648 | 0.589 | 0.612 |
| PatchMixer | 0.402 | 24.748 | 0.732 | 0.653 | 0.835 |
| DLinear | 0.455 | 44.738 | 0.793 | 0.674 | 0.824 |
| TiDE | 0.527 | 20.874 | 0.770 | 0.638 | 0.850 |
| RLinear | 0.597 | 21.144 | 0.743 | 0.620 | 0.829 |

**Multivariate (joint proton + SXR)**

| Model | Proton RMSE | Proton MSA | Proton logR | Proton HSS | Proton TSS | SXR RMSE | SXR MSA | SXR logR | SXR HSS | SXR TSS |
|---|---|---|---|---|---|---|---|---|---|---|
| SegRNN | 0.132 | 13.132 | 0.943 | 0.812 | 0.735 | 0.193 | 17.599 | 0.969 | 0.163 | 0.104 |
| xPatch† | 0.152 (10/15) | 15.568 (10/15) | 0.932 (10/15) | 0.857 (10/15) | 0.847 (10/15) | 0.197 (10/15) | 17.104 (10/15) | 0.978 (10/15) | 0.166 (10/15) | 0.101 (10/15) |
| PatchMixer | 0.185 | 15.443 | 0.882 | 0.794 | 0.797 | 0.216 | 21.715 | 0.959 | 0.175 | 0.112 |
| DLinear | 0.190 | 15.439 | 0.888 | 0.826 | 0.769 | 0.282 | 47.954 | 0.931 | 0.218 | 0.151 |
| iTransformer | 0.205 | 15.666 | 0.861 | 0.559 | 0.482 | 0.229 | 24.555 | 0.953 | 0.049 | 0.029 |
| RLinear | 0.212 | 14.799 | 0.878 | 0.739 | 0.802 | 0.237 | 26.390 | 0.949 | 0.236 | 0.171 |
| TiDE | 0.215 | 14.939 | 0.882 | 0.742 | 0.810 | 0.239 | 26.588 | 0.948 | 0.233 | 0.169 |
| PatchTST | 0.226 | 15.032 | 0.818 | 0.507 | 0.440 | 0.235 | 25.726 | 0.947 | 0.032 | 0.018 |
| TSMixer | 0.449 | 84.106 | 0.381 | 0.348 | 0.301 | 0.821 | 221.434 | 0.599 | 0.022 | 0.028 |
| FReTS | 2.117 | 21.597 | 0.739 | 0.624 | 0.593 | 0.262 | 38.479 | 0.931 | 0.086 | 0.068 |

_† xPatch is left out at the 7-day input (seq_len 2016) because its training diverges there, so its cells average 10 runs (1-day and 3-day inputs) instead of all 15 runs of this forecast length and are not directly comparable with the other rows._

### 12 h (pred_len 144)

**SXR 1--8 A (univariate)**

| Model | SXR RMSE | SXR MSA | SXR logR | SXR HSS | SXR TSS |
|---|---|---|---|---|---|
| SegRNN | 0.213 | 19.614 | 0.975 | 0.097 | 0.059 |
| iTransformer | 0.225 | 20.860 | 0.972 | 0.062 | 0.037 |
| PatchTST | 0.228 | 21.903 | 0.971 | 0.035 | 0.021 |
| xPatch† | 0.233 (10/15) | 21.031 (10/15) | 0.968 (10/15) | 0.088 (10/15) | 0.052 (10/15) |
| RLinear | 0.238 | 23.122 | 0.969 | 0.117 | 0.075 |
| TiDE | 0.242 | 23.314 | 0.968 | 0.142 | 0.094 |
| FReTS | 0.328 | 52.119 | 0.946 | 0.096 | 0.073 |
| PatchMixer | 0.522 | 25.005 | 0.877 | 0.112 | 0.073 |
| DLinear | 0.805 | 384.340 | 0.762 | 0.161 | 0.188 |
| TSMixer | 0.810 | 260.895 | 0.755 | 0.000 | 0.000 |

**Proton >=10 MeV (univariate)**

| Model | Proton RMSE | Proton MSA | Proton logR | Proton HSS | Proton TSS |
|---|---|---|---|---|---|
| SegRNN | 0.195 | 16.806 | 0.922 | 0.807 | 0.692 |
| xPatch† | 0.217 (10/15) | 20.146 (10/15) | 0.912 (10/15) | 0.792 (10/15) | 0.685 (10/15) |
| iTransformer | 0.234 | 18.438 | 0.886 | 0.713 | 0.606 |
| PatchTST | 0.304 | 19.409 | 0.771 | 0.534 | 0.458 |
| TSMixer | 0.321 | 45.101 | 0.815 | 0.471 | 0.369 |
| PatchMixer | 0.454 | 25.877 | 0.745 | 0.642 | 0.791 |
| DLinear | 0.508 | 46.264 | 0.679 | 0.613 | 0.714 |
| FReTS | 0.527 | 51.819 | 0.512 | 0.498 | 0.455 |
| TiDE | 0.743 | 22.319 | 0.713 | 0.616 | 0.794 |
| RLinear | 0.767 | 22.906 | 0.716 | 0.595 | 0.778 |

**Multivariate (joint proton + SXR)**

| Model | Proton RMSE | Proton MSA | Proton logR | Proton HSS | Proton TSS | SXR RMSE | SXR MSA | SXR logR | SXR HSS | SXR TSS |
|---|---|---|---|---|---|---|---|---|---|---|
| SegRNN | 0.166 | 13.778 | 0.910 | 0.691 | 0.603 | 0.210 | 20.777 | 0.963 | 0.088 | 0.055 |
| xPatch† | 0.179 (10/15) | 15.784 (10/15) | 0.903 (10/15) | 0.807 (10/15) | 0.715 (10/15) | 0.211 (10/15) | 19.414 (10/15) | 0.975 (10/15) | 0.086 (10/15) | 0.053 (10/15) |
| DLinear | 0.228 | 16.834 | 0.837 | 0.807 | 0.747 | 0.357 | 100.078 | 0.916 | 0.194 | 0.167 |
| iTransformer | 0.237 | 15.997 | 0.813 | 0.447 | 0.354 | 0.246 | 27.604 | 0.946 | 0.034 | 0.021 |
| PatchMixer | 0.243 | 19.231 | 0.778 | 0.722 | 0.703 | 0.243 | 26.694 | 0.946 | 0.138 | 0.091 |
| PatchTST | 0.262 | 15.151 | 0.751 | 0.491 | 0.408 | 0.257 | 29.123 | 0.945 | 0.027 | 0.017 |
| RLinear | 0.283 | 15.541 | 0.799 | 0.704 | 0.736 | 0.282 | 34.767 | 0.928 | 0.164 | 0.123 |
| TiDE | 0.300 | 15.856 | 0.792 | 0.722 | 0.787 | 0.295 | 37.221 | 0.921 | 0.191 | 0.154 |
| TSMixer | 0.536 | 63.802 | 0.298 | 0.190 | 0.161 | 0.767 | 204.514 | 0.618 | 0.018 | 0.014 |
| FReTS | 2.345 | 25.850 | 0.678 | 0.543 | 0.503 | 0.290 | 44.992 | 0.925 | 0.138 | 0.111 |

_† xPatch is left out at the 7-day input (seq_len 2016) because its training diverges there, so its cells average 10 runs (1-day and 3-day inputs) instead of all 15 runs of this forecast length and are not directly comparable with the other rows. The 3 diverged 7-day runs at this forecast length stay in the result tables as NaN rows._

### 24 h (pred_len 288)

**SXR 1--8 A (univariate)**

| Model | SXR RMSE | SXR MSA | SXR logR | SXR HSS | SXR TSS |
|---|---|---|---|---|---|
| SegRNN | 0.235 | 22.619 | 0.970 | 0.055 | 0.035 |
| iTransformer | 0.248 | 24.626 | 0.966 | 0.028 | 0.018 |
| xPatch† | 0.249 (10/15) | 23.768 (10/15) | 0.964 (10/15) | 0.044 (10/15) | 0.028 (10/15) |
| PatchTST | 0.251 | 25.052 | 0.965 | 0.027 | 0.017 |
| RLinear | 0.274 | 27.808 | 0.960 | 0.104 | 0.074 |
| TiDE | 0.275 | 28.166 | 0.959 | 0.125 | 0.089 |
| FReTS | 0.356 | 54.547 | 0.940 | 0.060 | 0.040 |
| PatchMixer | 0.714 | 31.321 | 0.829 | 0.102 | 0.074 |
| DLinear | 0.744 | 217.688 | 0.795 | 0.257 | 0.325 |
| TSMixer | 0.807 | 273.065 | 0.748 | 0.000 | 0.000 |

**Proton >=10 MeV (univariate)**

| Model | Proton RMSE | Proton MSA | Proton logR | Proton HSS | Proton TSS |
|---|---|---|---|---|---|
| SegRNN | 0.257 | 19.251 | 0.862 | 0.659 | 0.518 |
| xPatch† | 0.279 (10/15) | 22.530 (10/15) | 0.851 (10/15) | 0.648 (10/15) | 0.505 (10/15) |
| iTransformer | 0.296 | 21.452 | 0.818 | 0.580 | 0.441 |
| PatchTST | 0.349 | 19.433 | 0.706 | 0.471 | 0.378 |
| TSMixer | 0.401 | 49.337 | 0.667 | 0.260 | 0.191 |
| DLinear | 0.543 | 46.359 | 0.630 | 0.547 | 0.599 |
| FReTS | 0.593 | 59.579 | 0.393 | 0.362 | 0.377 |
| TiDE | 0.703 | 23.023 | 0.638 | 0.544 | 0.620 |
| RLinear | 0.710 | 22.757 | 0.644 | 0.536 | 0.618 |
| PatchMixer | 0.916 | 46.235 | 0.600 | 0.493 | 0.612 |

**Multivariate (joint proton + SXR)**

| Model | Proton RMSE | Proton MSA | Proton logR | Proton HSS | Proton TSS | SXR RMSE | SXR MSA | SXR logR | SXR HSS | SXR TSS |
|---|---|---|---|---|---|---|---|---|---|---|
| SegRNN | 0.221 | 15.277 | 0.833 | 0.498 | 0.380 | 0.232 | 24.202 | 0.955 | 0.052 | 0.035 |
| xPatch† | 0.235 (10/15) | 16.194 (10/15) | 0.829 (10/15) | 0.605 (10/15) | 0.452 (10/15) | 0.235 (10/15) | 22.997 (10/15) | 0.969 (10/15) | 0.030 (10/15) | 0.019 (10/15) |
| DLinear | 0.276 | 16.781 | 0.767 | 0.536 | 0.449 | 0.403 | 92.895 | 0.900 | 0.205 | 0.178 |
| PatchMixer | 0.279 | 17.877 | 0.734 | 0.518 | 0.447 | 0.272 | 32.945 | 0.938 | 0.110 | 0.079 |
| iTransformer | 0.282 | 15.839 | 0.731 | 0.347 | 0.252 | 0.263 | 30.455 | 0.938 | 0.011 | 0.007 |
| PatchTST | 0.292 | 14.502 | 0.696 | 0.359 | 0.270 | 0.269 | 31.648 | 0.937 | 0.016 | 0.011 |
| TiDE | 0.387 | 16.680 | 0.727 | 0.589 | 0.530 | 0.345 | 62.430 | 0.904 | 0.132 | 0.100 |
| RLinear | 0.388 | 16.596 | 0.732 | 0.519 | 0.459 | 0.344 | 62.788 | 0.903 | 0.135 | 0.103 |
| TSMixer | 0.579 | 84.508 | 0.182 | 0.080 | 0.065 | 0.851 | 213.005 | 0.492 | 0.031 | 0.025 |
| FReTS | 3.412 | 24.989 | 0.640 | 0.388 | 0.310 | 0.369 | 51.948 | 0.891 | 0.157 | 0.138 |

_† xPatch is left out at the 7-day input (seq_len 2016) because its training diverges there, so its cells average 10 runs (1-day and 3-day inputs) instead of all 15 runs of this forecast length and are not directly comparable with the other rows. The 3 diverged 7-day runs at this forecast length stay in the result tables as NaN rows._

<!-- LEADERBOARD:END -->

Regenerate with `python3.12 scripts/make_leaderboard.py`.

## Data & reproduction

- **Data** — download from Zenodo or rebuild from NOAA sources: [DATA.md](DATA.md)
- **Reproduce** the benchmark end-to-end: [REPRODUCE.md](REPRODUCE.md)
- **Reconstruction pipeline** (public NOAA/NCEI download + ingest): [data_pipeline/](data_pipeline/)

## Installation

The code lives in the importable `tslib` package. Install it editable so
`import tslib` works from any notebook or script:

```bash
pip install -e .        # Python 3.12 recommended (netCDF/parquet libs)
```

Core dependencies (see `requirements.txt` / `pyproject.toml`): `numpy`,
`pandas`, `torch`, `statsmodels` (`<0.15`), and `pytorch_lightning` (the neural
training loop runs on `pytorch_lightning`'s `Trainer`/`LightningModule`;
see [`docs/lightning-migration.md`](docs/lightning-migration.md)).

Then, e.g. in a notebook under `notebook/`:

```python
from tslib.data.loader import DataModule
from tslib.model import build_model
```

## Quick start

Set the data location (data is distributed separately):

```bash
export SW_DATA_DIR=/path/to/goes_data/processed
```

Run one or more models and compare them:

```bash
# Univariate: proton p_gt10, 7-day look-back -> 1-day forecast, with the
# >= 10 pfu event scores (one threshold per target, physical units)
python main.py --data_path $SW_DATA_DIR/kswrc_particle_5m.parquet \
    --target_col p_gt10 --seq_len 2016 --pred_len 288 \
    --event_threshold 10 \
    --n_fold 5 --fold_numb 0 --models lstm patchtst

# Multivariate input -> single target (proton + X-ray in, proton out)
python main.py \
    --channels $SW_DATA_DIR/kswrc_particle_5m.parquet:p_gt10 \
               $SW_DATA_DIR/kswrc_xray_1m.parquet:xrs_long \
    --target_cols p_gt10 --seq_len 864 --pred_len 288 --models lstm patchtst

# Multivariate input -> multi-target (forecast both channels)
python main.py --channels $SW_DATA_DIR/...:p_gt10 $SW_DATA_DIR/...:xrs_long \
    --target_cols p_gt10 xrs_long --seq_len 864 --pred_len 288 --models lstm
```

## Key options

| Option | Default | Description |
|---|---|---|
| `--data_path` / `--target_col` | (required) | Single channel; synthesized into one channel when `--channels` is absent |
| `--channels PATH:COL [...]` | `None` | Input channels; one channel = univariate input |
| `--target_cols COL [...]` | `[first channel col]` | Forecast targets (subset of channels); 1 = univariate output, ≥2 = multivariate |
| `--split_type` | `year_half` | `year_half` / `year` / `ratio` |
| `--n_fold` / `--fold_numb` | `5` / `0` | Rotating k-fold count and fold index (0..n_fold-1) |
| `--seq_len` / `--pred_len` | `24` / `1` | Look-back / forecast length. **Parquet uses 5-min steps** (1 day = 288, 7 days = 2016) |
| `--cadence_min` | `5` | Common-grid spacing (minutes) |
| `--min_bin_count` | `1` | Min native samples for a 5-min bin to be valid |
| `--transform` | `log10` | `none` / `log10` (applied after 5-min averaging) |
| `--metrics NAME [...]` | all registered | Optional filter on the metrics computed: regression `rmse mae msa sspb logr`, event `tss hss pod far csi f1 seds sedi`. Without it every metric is computed |
| `--event_threshold X [...]` | `None` | Event threshold per target, aligned to `--target_cols`, in **physical units** (e.g. `10` pfu, `1e-5` W m⁻²). Without it no event rows are written |
| `--save_predictions` | on | Write `<runs_root>/<run_name>/pred/<model>.npz` after testing; `--no-save_predictions` turns it off |

> The `--seq_len`/`--pred_len` defaults (24/1) are legacy hourly values; for
> 5-minute parquet, pass step counts explicitly (1 day = 288, 3 days = 864).

## Data format

- **parquet** (space-weather pipeline): time column defaults to `time_utc`, with
  a `role` column (filtered to `primary` by default). Values are raw flux;
  `--transform log10` is applied after 5-minute averaging.
- **CSV** (generic): a target column (`--target_col`) and optional time column
  (`--time_col`; naive times are taken as UTC). With a time column a CSV run is
  evaluated and saved like a parquet run. Without one it has no `day` event
  rows and writes no `pred/<model>.npz`, because both need each test window's
  forecast start time.

## Repository layout

```
.
├── tslib/            # the importable package (pip install -e .)
│   ├── configs/      # CLI / experiment configuration
│   ├── data/         # loading, windowing, splitting, common-grid channel join
│   ├── model/        # model sources + build_model / forecast adapter
│   ├── exp/          # training loop and multi-model comparison
│   └── preprocessing/# term-split, fold-sample counting, table generation
├── data_pipeline/    # public NOAA/NCEI download + ingest -> parquet
├── results/          # results_regression.csv + results_event.csv (v2 tables); v1/ (v1 files)
├── scripts/          # fetch_data.py, make_leaderboard.py, rescore.py, dump_predictions.py, verify_archive.py
├── notebook/         # exploratory / analysis notebooks (import tslib)
├── docs/             # design documents
└── main.py           # CLI entry point (imports tslib)
```

## Testing

```bash
python -m unittest discover -t . -s tslib -p 'test_*.py'   # from repo root
```
Real-data tests are skipped unless `SW_DATA_DIR` is set.

## Design documents

See `docs/`:
- `metrics.md` — metric definitions, NaN rules, aggregation units, result and
  prediction file formats, differences from v1
- `benchmark-conditions.md` — the benchmark task: targets, tracks, splits, metrics
- `dataloader-split-design.md` — half-year split + parquet `DataModule`
- `multivar-join-design.md` — multivariate common-grid join, variable
  lengths, multi-target output
- `model-extensibility-design.md` — backbone / loss / metric registries
- `lightning-migration.md` — pytorch_lightning training architecture,
  customizing `ForecastModule`, early stopping, and checkpointing
- `v004-short-horizon-sweep.md` — the 3 h / 6 h horizons and the multi-GPU job queue

## Outputs

- `runs/<run_name>/ckpt/<model>.ckpt`
- `runs/<run_name>/pred/<model>.npz` — test predictions, observations and forecast start times (not for CSV input without `--time_col`)
- `runs/<run_name>/score/comparison_regression.csv` — per-channel regression scores, one row per model and channel
- `runs/<run_name>/score/comparison_event.csv` — event scores and counts per model, channel and aggregation unit
- `runs/<run_name>/score/rows/<model>.json` — the model's result record

The benchmark driver gathers all runs into `results_regression.csv` and
`results_event.csv` (see [`results/README.md`](results/README.md)).

## Citation

If you use this benchmark, code, or dataset, please cite both (see [CITATION.cff](CITATION.cff)):

- **Code / benchmark** — this repository
- **Dataset** — Zenodo DOI [10.5281/zenodo.22245285](https://doi.org/10.5281/zenodo.22245285)

## License

See [LICENSE](LICENSE).
