#!/usr/bin/env python3.12
"""Combine the SWPC GOES particle daily archive into one Parquet dataset.

Reads every daily 5-minute particle file under the SWPC archive directory
(set via $SW_GOES_ARCHIVE_PARTICLE or --db), parses both on-disk
formats (see parse_particle.py), records per-file provenance with sha256, and
emits a single native-cadence (5-min UTC) columnar file:

    goes_data/processed/kasi_swpc_particle_5m.parquet

plus a QC report at goes_data/manifest/particle_qc_report.md and a provenance
manifest at goes_data/manifest/particle_provenance.jsonl.

HARD RULE: the source DB is never written/modified. We only read + derive.

Each output row is one (timestamp x series) sample. Columns:
  time_utc, series, role, source_sat, location, era,
  p_gt1 p_gt5 p_gt10 p_gt30 p_gt50 p_gt60 p_gt100 p_gt500,
  e_gt0p6 e_gt0p8 e_gt2 e_gt4, src_file
Missing / structurally-absent channels are NaN (distinguish via `era`).
`role`: Gp->primary, Gs->secondary; pre-Gp/Gs per-satellite files are labelled
primary/secondary/other by joining manifest/designation_table.csv (proton).

Run: python3.12 build_particle_dataset.py            # full build
     python3.12 build_particle_dataset.py --years 2024 2025   # subset
     python3.12 build_particle_dataset.py --copy-raw          # also mirror raw
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

# Reuse the project's provenance/sha256/log helpers.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from goes_download_common import Manifest, Logger, now_utc  # noqa: E402

from parse_particle import (  # noqa: E402
    ALL_CHANNEL_KEYS, parse_file, series_from_name,
)

DEFAULT_DB = os.environ.get("SW_GOES_ARCHIVE_PARTICLE", "")
DEFAULT_OUT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))), "data", "goes_data")
SOURCE_LABEL = "NOAA/SWPC GOES daily archive"

OUTPUT_COLUMNS = (
    ["time_utc", "series", "role", "source_sat", "location", "era"]
    + ALL_CHANNEL_KEYS + ["src_file"]
)


def era_of(fmt: str) -> str:
    return "kasi_2020_" if fmt == "kasi" else "noaa_1998_2019"


def list_files(db: str, years: list[str] | None) -> list[str]:
    """All daily particle 5m files, excluding the stray Gp_xr_1m X-ray files."""
    out = []
    yrs = years or sorted(d for d in os.listdir(db) if re.fullmatch(r"\d{4}", d))
    for y in yrs:
        for p in sorted(glob.glob(os.path.join(db, y, "*part_5m.txt"))):
            base = os.path.basename(p)
            if "_xr_" in base:
                continue
            try:
                series_from_name(base)
            except ValueError:
                continue
            out.append(p)
    return out


def load_designation(out_root: str) -> pd.DataFrame | None:
    path = os.path.join(out_root, "manifest", "designation_table.csv")
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    d = d[d["instrument"] == "proton"].copy()
    d["start_dt"] = pd.to_datetime(d["start_utc"], utc=True)
    d = d.sort_values("start_dt").reset_index(drop=True)
    d["primary_sat"] = d["primary_sat"].astype(str)
    d["secondary_sat"] = d["secondary_sat"].astype(str)
    return d[["start_dt", "primary_sat", "secondary_sat"]]


def assign_roles(df: pd.DataFrame, desig: pd.DataFrame | None) -> pd.Series:
    """Gp->primary, Gs->secondary; per-satellite files joined to designation."""
    role = pd.Series(pd.NA, index=df.index, dtype="object")
    role[df["series"] == "Gp"] = "primary"
    role[df["series"] == "Gs"] = "secondary"
    persat = df["series"].str.fullmatch(r"G\d{1,2}")
    if persat.any():
        if desig is None:
            role[persat] = "undesignated"
        else:
            sub = df.loc[persat, ["time_utc", "source_sat"]].sort_values("time_utc")
            orig_idx = sub.index                       # merge_asof drops the index
            merged = pd.merge_asof(
                sub.reset_index(drop=True), desig,
                left_on="time_utc", right_on="start_dt", direction="backward")
            sat = merged["source_sat"].astype("Int64").astype(str)
            r = np.where(sat == merged["primary_sat"], "primary",
                 np.where(sat == merged["secondary_sat"], "secondary", "other"))
            role.loc[orig_idx] = pd.Series(r, index=orig_idx)
    return role.astype("string")


def build_dataframe(files: list[str], log: Logger,
                    manifest: Manifest, db: str,
                    out_root: str, copy_raw: bool) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    n = len(files)
    raw_root = os.path.join(out_root, "raw", "proton", "kasi_swpc")
    for i, path in enumerate(files, 1):
        base = os.path.basename(path)
        with open(path, "rb") as fh:
            raw_bytes = fh.read()
        sha = hashlib.sha256(raw_bytes).hexdigest()
        text = raw_bytes.decode("utf-8", errors="replace")
        if not text.strip():
            manifest.append({
                "local_path": path, "source": SOURCE_LABEL,
                "instrument": "proton", "series": series_from_name(base),
                "n_rows": 0, "sha256": sha, "note": "empty source file",
                "retrieved_utc": now_utc(),
            })
            log(f"  skip empty file: {base}")
            continue
        parsed = parse_file(path, text=text)
        m = parsed.meta
        if parsed.rows:
            fdf = pd.DataFrame(parsed.rows)
            for k in ALL_CHANNEL_KEYS:
                if k not in fdf.columns:
                    fdf[k] = np.nan
            fdf["series"] = m.series
            fdf["source_sat"] = m.source_sat
            fdf["location"] = m.location
            fdf["era"] = era_of(m.fmt)
            fdf["src_file"] = base
            frames.append(fdf)
        # provenance (1 line per source file; key = original source path)
        ts = [r["time_utc"] for r in parsed.rows]
        manifest.append({
            "local_path": path,                      # immutable source path
            "source": SOURCE_LABEL,
            "satellite": f"goes{m.source_sat}" if m.source_sat else None,
            "instrument": "proton",
            "series": m.series,
            "format": m.fmt,
            "prepared_by": m.prepared_by,
            "location": m.location,
            "channels": m.channel_keys,
            "native_cadence": "5min",
            "units_proton": "protons/(cm^2 s sr)",
            "units_electron": "electrons/(cm^2 s sr)",
            "n_rows": len(parsed.rows),
            "time_start": ts[0].strftime("%Y-%m-%dT%H:%M:%SZ") if ts else None,
            "time_end": ts[-1].strftime("%Y-%m-%dT%H:%M:%SZ") if ts else None,
            "sha256": sha,
            "retrieved_utc": now_utc(),
        })
        if copy_raw:
            yr = base[:4]
            dest_dir = os.path.join(raw_root, yr)
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
    df = pd.concat(frames, ignore_index=True)
    return df


def finalize(df: pd.DataFrame, desig: pd.DataFrame | None) -> pd.DataFrame:
    df["role"] = assign_roles(df, desig)
    df["source_sat"] = df["source_sat"].astype("Int16")
    for k in ALL_CHANNEL_KEYS:
        df[k] = df[k].astype("float32")
    for c in ("series", "role", "era", "location"):
        df[c] = df[c].astype("string")
    df = df[OUTPUT_COLUMNS]
    df = df.sort_values(["time_utc", "series"]).reset_index(drop=True)
    df = df.drop_duplicates(["time_utc", "series"], keep="last")
    return df


def write_qc(df: pd.DataFrame, path: str, n_files: int) -> None:
    lines = ["# KASI SWPC GOES particle — combined dataset QC report", ""]
    lines.append(f"- generated: {now_utc()}")
    lines.append(f"- source files parsed: {n_files}")
    lines.append(f"- total rows (time x series): {len(df):,}")
    t0, t1 = df["time_utc"].min(), df["time_utc"].max()
    lines.append(f"- time coverage (UTC): {t0} .. {t1}")
    lines.append("")
    lines.append("## rows by series / role")
    sr = df.groupby(["series", "role"], dropna=False).size().reset_index(name="n")
    lines.append("| series | role | rows |")
    lines.append("|---|---|---|")
    for _, r in sr.iterrows():
        lines.append(f"| {r['series']} | {r['role']} | {r['n']:,} |")
    lines.append("")
    lines.append("## missing/absent fraction per channel (NaN share)")
    lines.append("| channel | NaN % |")
    lines.append("|---|---|")
    for k in ALL_CHANNEL_KEYS:
        frac = float(df[k].isna().mean()) * 100.0
        lines.append(f"| {k} | {frac:.1f} |")
    lines.append("")
    lines.append("## source_sat timeline (first/last UTC per satellite, primary)")
    prim = df[df["role"] == "primary"]
    g = prim.groupby("source_sat")["time_utc"].agg(["min", "max", "size"])
    lines.append("| source_sat | first | last | rows |")
    lines.append("|---|---|---|---|")
    for sat, row in g.iterrows():
        lines.append(f"| GOES-{sat} | {row['min']} | {row['max']} | {row['size']:,} |")
    lines.append("")
    lines.append("## notes")
    lines.append("- Two source formats: noaa_1998_2019 (6 proton thresholds, no "
                 ">60/>500; electrons E>0.6 or E>0.8, E>2.0, E>4.0) and "
                 "kasi_2020_ (8 thresholds incl >60/>500; electron E>2.0 only).")
    lines.append("- p_gt60/p_gt500 are structurally absent (NaN) before 2020; "
                 "e_gt0p6/e_gt0p8/e_gt4 are absent in the kasi era.")
    lines.append("- Missing masked to NaN: declared sentinel -1.00e+05 AND "
                 "exactly-0.0 no-data values (KASI era; non-physical for "
                 "integral flux).")
    lines.append("- No cross-satellite blending. Each row keeps its own "
                 "source_sat; primary/secondary follow the DB's Gp/Gs labelling "
                 "(2010+) or designation_table.csv (per-satellite era).")
    lines.append("- Source DB is read-only; provenance with sha256 in "
                 "manifest/particle_provenance.jsonl.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DEFAULT_DB, help="KASI particle DB root (read-only)")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="goes_data root")
    ap.add_argument("--years", nargs="*", help="restrict to these years")
    ap.add_argument("--copy-raw", action="store_true",
                    help="also mirror raw text files into goes_data/raw (durable)")
    args = ap.parse_args(argv)

    out_root = args.out_root
    os.makedirs(os.path.join(out_root, "manifest"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "processed"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "logs"), exist_ok=True)
    log = Logger(Path(out_root) / "logs" /
                 f"particle_ingest_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.log")
    # Full (re)build regenerates every record, so start the manifest fresh.
    # (Manifest.append rewrites the whole file when a local_path already exists,
    # which is O(n^2) on a rebuild against a populated manifest.)
    mpath = Path(out_root) / "manifest" / "particle_provenance.jsonl"
    if not args.years:
        mpath.unlink(missing_ok=True)
    manifest = Manifest(mpath)

    files = list_files(args.db, args.years)
    log(f"discovered {len(files)} particle 5m files under {args.db}")
    if not files:
        raise SystemExit("no input files found")

    df = build_dataframe(files, log, manifest, args.db, out_root, args.copy_raw)
    desig = load_designation(out_root)
    log(f"designation_table: {'loaded' if desig is not None else 'MISSING'}")
    df = finalize(df, desig)

    out_parquet = os.path.join(out_root, "processed", "kasi_swpc_particle_5m.parquet")
    df.to_parquet(out_parquet, engine="pyarrow", compression="snappy", index=False)
    sz = os.path.getsize(out_parquet) / 1e6
    log(f"wrote {out_parquet} ({len(df):,} rows, {sz:.1f} MB)")

    qc = os.path.join(out_root, "manifest", "particle_qc_report.md")
    write_qc(df, qc, len(files))
    log(f"wrote {qc}")
    log.close()
    print(f"\nDONE: {len(df):,} rows -> {out_parquet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
