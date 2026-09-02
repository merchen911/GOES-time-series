# Data reconstruction pipeline

Rebuild the two benchmark targets — GOES soft X-ray (`xrs_long`, 1-min) and
integral proton flux (`p_gt10`, 5-min) — from **public NOAA products**. The
processed parquet used by the benchmark is distributed separately (see the data
DOI in the top-level README); this pipeline lets anyone reconstruct an
equivalent dataset from scratch.

## Public sources

| Product | Host |
|---|---|
| XRS science-quality L2 (GOES-R 16–19) | `https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/` |
| XRS science-quality L2 (legacy 08–15) | `https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs/` |
| Proton SGPS / EPEAD / EPS | `https://data.ngdc.noaa.gov/…` and `https://www.ncei.noaa.gov/data/goes-space-environment-monitor/` |
| Proton 7-day live JSON (recent) | `https://services.swpc.noaa.gov/json/goes/` |

All are NOAA public-domain products. Access politely (the downloaders throttle,
back off on 429/5xx, and record per-file provenance with sha256).

## Steps

```bash
# 0. where downloads / archive live (defaults are repo-relative)
export SW_DATA_ROOT=./data/goes_data                 # raw/ + processed/ + manifest/
export SW_GOES_ARCHIVE_XRAY=$SW_DATA_ROOT/raw/xrs    # daily XRS files to ingest
export SW_GOES_ARCHIVE_PARTICLE=$SW_DATA_ROOT/raw/particle

# 1. download public products into raw/
python3.12 xrs_download_v02/xrs_download_v02.py
python3.12 proton_download_v02/proton_download_v02.py

# 2. build the primary/secondary designation table (proton)
python3.12 proton_download_v02/build_designation_table.py

# 3. ingest the daily archive into one parquet per target (+ provenance manifest + QC report)
python3.12 xray_ingest_v01/build_xray_dataset.py
python3.12 particle_ingest_v01/build_particle_dataset.py

# 4. (optional) re-grid onto the common analysis grid
python3.12 build_v02_grid.py --in <processed>.parquet --out <processed>_v02.parquet
```

Each ingest writes `processed/*.parquet`, `manifest/*_provenance.jsonl` (one line
per source file, with sha256), and `manifest/*_qc_report.md`.

## Provenance note

The parquet distributed with this benchmark was assembled from the NOAA/SWPC
GOES **daily archive** (operational text products). This pipeline fetches the
NOAA/NCEI **science-quality** equivalents; the two product levels can differ
slightly (recalibration, reprocessing), so a fresh reconstruction may not be
byte-identical to the released files. The parser handles both on-disk formats.

## Tests

Parser unit tests need a local archive and skip otherwise:

```bash
cd xray_ingest_v01     && python3.12 -m unittest test_parse_xray -v      # skips unless SW_GOES_ARCHIVE_XRAY is set
cd particle_ingest_v01 && python3.12 -m unittest test_parse_particle -v  # skips unless SW_GOES_ARCHIVE_PARTICLE is set
cd proton_download_v02 && python3.12 -m unittest test_designation_table -v  # stdlib only, always runs
```
