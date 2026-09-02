#!/usr/bin/env python3.12
"""Combine the SWPC GOES X-ray (XRS) daily archive into one Parquet dataset.

Sibling of particle_ingest_v01/build_particle_dataset.py. Reads every daily
1-minute X-ray file under the SWPC archive directory (set via
$SW_GOES_ARCHIVE_XRAY or --db), parses both on-disk formats
(see parse_xray.py), records per-file provenance with sha256, and emits:

    goes_data/processed/kasi_swpc_xray_1m.parquet
    goes_data/manifest/xray_provenance.jsonl
    goes_data/manifest/xray_qc_report.md

HARD RULE: the source DB is never written/modified.

Output row = one (timestamp x series) sample at native 1-min UTC. Columns:
  time_utc, series, role, source_sat, location, era, xrs_short, xrs_long, src_file
  (xrs_short = XRS-A 0.05-0.4nm, xrs_long = XRS-B 0.1-0.8nm, units W/m^2)
Missing -> NaN. role: Gp->primary, Gs->secondary; per-satellite era (2002-2009)
labelled via designation_table.csv (the spacecraft-level XRS primary chronology
from the GOES XRS readme Table 2 -- the same intervals the proton rows reuse).

NOTE: this is the SWPC *operational* X-ray text product. The science-quality
XRS netCDF (NCEI) already lives in goes_data/raw/xrs/ and is the research canon;
this dataset is the operational/real-time parallel to the particle ingest.

Run: python3.12 build_xray_dataset.py            # full build
     python3.12 build_xray_dataset.py --years 2024 2025
     python3.12 build_xray_dataset.py --copy-raw
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                       # data_download/
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "particle_ingest_v01"))
from goes_download_common import Manifest, Logger, now_utc       # noqa: E402
from build_particle_dataset import load_designation, assign_roles  # noqa: E402 (reuse)

from parse_xray import CHANNEL_KEYS, parse_file, series_from_name  # noqa: E402

DEFAULT_DB = os.environ.get("SW_GOES_ARCHIVE_XRAY", "")
DEFAULT_OUT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(_HERE))), "data", "goes_data")
SOURCE_LABEL = "NOAA/SWPC GOES daily archive"

OUTPUT_COLUMNS = (["time_utc", "series", "role", "source_sat", "location", "era"]
                  + CHANNEL_KEYS + ["src_file"])


def era_of(fmt: str) -> str:
    return "kasi_2020_" if fmt == "kasi" else "noaa_2002_2019"


def list_files(db: str, years: list[str] | None) -> list[str]:
    """Daily X-ray 1-min files only; exclude misfiled *part_5m and *xr_5m."""
    out = []
    yrs = years or sorted(d for d in os.listdir(db) if re.fullmatch(r"\d{4}", d))
    for y in yrs:
        for p in sorted(glob.glob(os.path.join(db, y, "*xr_1m.txt"))):
            base = os.path.basename(p)
            try:
                series_from_name(base)
            except ValueError:
                continue
            out.append(p)
    return out


def build_dataframe(files, log, manifest, out_root, copy_raw):
    frames = []
    n = len(files)
    raw_root = os.path.join(out_root, "raw", "xrs", "kasi_swpc")
    for i, path in enumerate(files, 1):
        base = os.path.basename(path)
        with open(path, "rb") as fh:
            raw_bytes = fh.read()
        sha = hashlib.sha256(raw_bytes).hexdigest()
        text = raw_bytes.decode("utf-8", errors="replace")
        if not text.strip():
            manifest.append({
                "local_path": path, "source": SOURCE_LABEL, "instrument": "xrs",
                "series": series_from_name(base), "n_rows": 0, "sha256": sha,
                "note": "empty source file", "retrieved_utc": now_utc()})
            log(f"  skip empty file: {base}")
            continue
        parsed = parse_file(path, text=text)
        m = parsed.meta
        if parsed.rows:
            fdf = pd.DataFrame(parsed.rows)
            fdf["series"] = m.series
            fdf["source_sat"] = m.source_sat
            fdf["location"] = m.location
            fdf["era"] = era_of(m.fmt)
            fdf["src_file"] = base
            frames.append(fdf)
        ts = [r["time_utc"] for r in parsed.rows]
        manifest.append({
            "local_path": path, "source": SOURCE_LABEL,
            "satellite": f"goes{m.source_sat}" if m.source_sat else None,
            "instrument": "xrs", "series": m.series, "format": m.fmt,
            "prepared_by": m.prepared_by, "location": m.location,
            "channels": {"xrs_short": "0.05-0.4nm (XRS-A)",
                         "xrs_long": "0.1-0.8nm (XRS-B)"},
            "native_cadence": "1min", "units": "W/m^2",
            "n_rows": len(parsed.rows),
            "time_start": ts[0].strftime("%Y-%m-%dT%H:%M:%SZ") if ts else None,
            "time_end": ts[-1].strftime("%Y-%m-%dT%H:%M:%SZ") if ts else None,
            "sha256": sha, "retrieved_utc": now_utc()})
        if copy_raw:
            dest_dir = os.path.join(raw_root, base[:4])
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, base)
            if not (os.path.exists(dest) and
                    hashlib.sha256(open(dest, "rb").read()).hexdigest() == sha):
                with open(dest, "wb") as out:
                    out.write(raw_bytes)
        if i % 1000 == 0 or i == n:
            log(f"  parsed {i}/{n} files ({base})")
    if not frames:
        raise SystemExit("no rows parsed")
    return pd.concat(frames, ignore_index=True)


# Physical ceiling for GOES XRS flux (W/m^2); larger values are corrupt/truncated
# source lines (e.g. a record cut mid-write to "5"). Masked to NaN, counted in QC.
XRS_MAX_PHYSICAL = 1.0e-2


def finalize(df, desig):
    df["role"] = assign_roles(df, desig)
    df["source_sat"] = df["source_sat"].astype("Int16")
    n_corrupt = 0
    for k in CHANNEL_KEYS:
        df[k] = df[k].astype("float64")        # X-ray flux spans many decades
        bad = df[k] > XRS_MAX_PHYSICAL
        n_corrupt += int(bad.sum())
        df.loc[bad, k] = np.nan
    finalize.n_corrupt = n_corrupt             # surfaced in the QC report
    for c in ("series", "role", "era", "location"):
        df[c] = df[c].astype("string")
    df = df[OUTPUT_COLUMNS]
    df = df.sort_values(["time_utc", "series"]).reset_index(drop=True)
    df = df.drop_duplicates(["time_utc", "series"], keep="last")
    return df


def write_qc(df, path, n_files):
    L = ["# KASI SWPC GOES X-ray (XRS) — combined dataset QC report", ""]
    L.append(f"- generated: {now_utc()}")
    L.append(f"- source files parsed: {n_files}")
    L.append(f"- total rows (time x series): {len(df):,}")
    L.append(f"- time coverage (UTC): {df['time_utc'].min()} .. {df['time_utc'].max()}")
    L.append("- channels: xrs_short = 0.05-0.4 nm (XRS-A), "
             "xrs_long = 0.1-0.8 nm (XRS-B); units W/m^2; native cadence 1-min")
    L += ["", "## rows by series / role", "| series | role | rows |", "|---|---|---|"]
    for _, r in df.groupby(["series", "role"], dropna=False).size()\
            .reset_index(name="n").iterrows():
        L.append(f"| {r['series']} | {r['role']} | {r['n']:,} |")
    L += ["", "## missing fraction per channel (NaN share)", "| channel | NaN % |",
          "|---|---|"]
    for k in CHANNEL_KEYS:
        L.append(f"| {k} | {float(df[k].isna().mean())*100:.1f} |")
    L += ["", "Masked as missing (NaN): exactly-0.0 no-data sentinels (KASI era) "
          f"and {getattr(finalize, 'n_corrupt', 0)} physically-impossible values "
          f"> {XRS_MAX_PHYSICAL:g} W/m^2 (corrupt/truncated source lines)."]
    L += ["", "## source_sat timeline (primary rows)", "| source_sat | first | last | rows |",
          "|---|---|---|---|"]
    prim = df[df["role"] == "primary"]
    for sat, row in prim.groupby("source_sat")["time_utc"]\
            .agg(["min", "max", "size"]).iterrows():
        L.append(f"| GOES-{sat} | {row['min']} | {row['max']} | {row['size']:,} |")
    L += ["", "## notes",
          "- Two formats: noaa_2002_2019 (8 cols, MJD/SOD index) and kasi_2020_ "
          "(7 cols, Sat# col). Both carry Short(XRS-A)+Long(XRS-B) only.",
          "- No cross-satellite blending; each row keeps its source_sat. "
          "primary/secondary from Gp/Gs labels (2009+) or designation_table.csv "
          "(spacecraft-level XRS primary, per-satellite era 2002-2009).",
          "- Operational SWPC product; science-quality XRS netCDF is in "
          "goes_data/raw/xrs/. Source DB read-only; sha256 provenance in "
          "manifest/xray_provenance.jsonl.",
          "- Excluded: misfiled *_part_5m (particle) and *_xr_5m (coarse, 2020 "
          "only) files; undated root-level rolling files."]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--years", nargs="*")
    ap.add_argument("--copy-raw", action="store_true")
    args = ap.parse_args(argv)

    out_root = args.out_root
    for sub in ("manifest", "processed", "logs"):
        os.makedirs(os.path.join(out_root, sub), exist_ok=True)
    log = Logger(Path(out_root) / "logs" /
                 f"xray_ingest_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.log")
    # Full (re)build regenerates every record -> start fresh (Manifest.append is
    # O(n^2) when re-appending paths that already exist in a populated manifest).
    mpath = Path(out_root) / "manifest" / "xray_provenance.jsonl"
    if not args.years:
        mpath.unlink(missing_ok=True)
    manifest = Manifest(mpath)

    files = list_files(args.db, args.years)
    log(f"discovered {len(files)} X-ray 1m files under {args.db}")
    if not files:
        raise SystemExit("no input files found")
    df = build_dataframe(files, log, manifest, out_root, args.copy_raw)
    desig = load_designation(out_root)
    log(f"designation_table: {'loaded' if desig is not None else 'MISSING'}")
    df = finalize(df, desig)

    out_parquet = os.path.join(out_root, "processed", "kasi_swpc_xray_1m.parquet")
    df.to_parquet(out_parquet, engine="pyarrow", compression="snappy", index=False)
    log(f"wrote {out_parquet} ({len(df):,} rows, "
        f"{os.path.getsize(out_parquet)/1e6:.1f} MB)")
    qc = os.path.join(out_root, "manifest", "xray_qc_report.md")
    write_qc(df, qc, len(files))
    log(f"wrote {qc}")
    log.close()
    print(f"\nDONE: {len(df):,} rows -> {out_parquet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
