# Data

The benchmark data is **not** stored in this repository. It is archived on
Zenodo and can either be downloaded directly or reconstructed from public NOAA
products.

## Download (recommended)

**DOI: [10.5281/zenodo.22245285](https://doi.org/10.5281/zenodo.22245285)**

```bash
export SW_DATA_DIR=./data/goes_data/processed
python3.12 scripts/fetch_data.py          # downloads + MD5-verifies into $SW_DATA_DIR
```

Two Apache Parquet files (~254 MB total):

| File | Cadence | Span | Columns |
|---|---|---|---|
| `kswrc_xray_1m.parquet` | 1 min | 2002–2026 | `time_utc`, `role`, `xrs_long` (1–8 Å), `xrs_short` (0.5–4 Å) |
| `kswrc_particle_5m.parquet` | 5 min | 1999–2026 | `time_utc`, `role`, `p_gt10` (≥10 MeV), `p_gt100` (≥100 MeV) |

`role` is the per-satellite designation (`primary` / `secondary`); the loader
filters to `primary` by default. Values are raw flux — pass `--transform log10`.
These are the exact files the benchmark was run on (`results/results_regression.csv`
and `results/results_event.csv` for v2; the v1.0.0 table is `results/v1/results_master.csv`).

## Reconstruct from source (optional)

The underlying measurements are NOAA public-domain GOES products. `data_pipeline/`
downloads the science-quality equivalents from NCEI/NGDC and rebuilds the parquet;
see [`data_pipeline/README.md`](data_pipeline/README.md). Product levels differ
(operational vs science-quality), so a fresh rebuild may not be byte-identical.

## Provenance & license

Assembled from NOAA/SWPC GOES products via the operational archive of the Korea Space Weather Research Center
([KSWRC](https://kswrc.kasi.re.kr/ko)); each file ships
with a per-source provenance manifest (sha256) and a QC report. Released under
**CC-BY-4.0** (the source measurements are NOAA public domain).
