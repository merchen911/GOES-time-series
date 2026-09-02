#!/usr/bin/env python3.12
"""Regrid a v01 KASI-SWPC parquet onto a regular native-cadence time grid (v02).

The v01 datasets contain a row only where the source had a sample, so the time
axis is not regular (missing 5-min / 1-min slots, and several series sharing a
timestamp in long format). This tool makes each series internally regular:

  * Per series, sort the existing timestamps and split into CONTIGUOUS SEGMENTS
    wherever the gap between consecutive samples exceeds --gap-min (default 1
    day). A gap that large is a real instrument/satellite outage, not a dropped
    sample, so we do NOT fabricate rows across it.
  * Within each segment, build the full native-step index (min..max of the
    segment) and reindex the series onto it. Small intra-segment gaps therefore
    become explicit NaN rows -> consecutive rows within a segment are exactly
    one step apart.

Meta columns on newly-created (missing) rows:
  * series          : the group key (known).
  * source_sat      : for per-satellite series (G8/G9/G10/G11/G12) this is the
                      satellite number from the name (definitionally constant);
                      for Gp/Gs it is left NA (which bird was primary at that
                      exact missing minute is unknown).
  * era             : derived from the timestamp (the 2020 noaa->kasi boundary).
  * role            : re-derived (Gp->primary, Gs->secondary, per-sat via
                      designation_table) so filled rows are labelled too.
  * location/src_file: NA on filled rows (no observation).

Channel values are NA on filled rows. v01 is left untouched.

Run: python3.12 build_v02_grid.py \
        --in  goes_data/processed/kasi_swpc_particle_5m.parquet \
        --out goes_data/processed/kasi_swpc_particle_5m_v02.parquet \
        --step-min 5
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "particle_ingest_v01"))
from build_particle_dataset import load_designation, assign_roles  # noqa: E402

META_COLS = ["series", "role", "source_sat", "location", "era", "src_file"]


def regrid(df: pd.DataFrame, step_min: int, gap_min: int) -> pd.DataFrame:
    step = pd.Timedelta(minutes=step_min)
    gap = pd.Timedelta(minutes=gap_min)
    freq = f"{step_min}min"
    channel_cols = [c for c in df.columns
                    if c not in (["time_utc"] + META_COLS)]
    out_parts = []
    for series, g in df.groupby("series", observed=True):
        g = g.drop_duplicates("time_utc").set_index("time_utc").sort_index()
        times = g.index.to_series()
        seg = (times.diff() > gap).cumsum()             # contiguous-segment id
        idx_parts = []
        for _, seg_times in times.groupby(seg.values):
            idx_parts.append(pd.date_range(seg_times.min(), seg_times.max(),
                                            freq=freq, tz="UTC"))
        full_idx = idx_parts[0]
        for extra in idx_parts[1:]:
            full_idx = full_idx.append(extra)
        gg = g.reindex(full_idx)
        gg["series"] = series                            # set group key on fill rows
        gg = gg.reset_index(names="time_utc")
        out_parts.append(gg[["time_utc"] + channel_cols + META_COLS])
    out = pd.concat(out_parts, ignore_index=True)

    # --- re-derive meta on filled rows ----------------------------------
    # per-satellite series carry a constant satellite number
    persat = out["series"].str.fullmatch(r"G\d{1,2}")
    out.loc[persat, "source_sat"] = (
        out.loc[persat, "series"].str.slice(1).astype("Int16"))
    out["source_sat"] = out["source_sat"].astype("Int16")
    # era from the 2020 boundary, using the labels already present in v01
    era_labels = [e for e in df["era"].dropna().unique()]
    noaa = next(e for e in era_labels if "noaa" in e)
    kasi = next(e for e in era_labels if "kasi" in e)
    boundary = pd.Timestamp("2020-01-01", tz="UTC")
    out["era"] = np.where(out["time_utc"] < boundary, noaa, kasi)
    # role re-derived for every row (incl. fills)
    desig = load_designation(os.path.join(
        os.path.dirname(os.path.dirname(_HERE)), "data", "goes_data"))
    out["role"] = assign_roles(out, desig)
    for c in ("series", "role", "era", "location"):
        out[c] = out[c].astype("string")
    out = out.sort_values(["time_utc", "series"]).reset_index(drop=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step-min", type=int, required=True)
    ap.add_argument("--gap-min", type=int, default=1440,
                    help="gaps larger than this (min) split segments; default 1 day")
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.inp)
    n_in = len(df)
    out = regrid(df, args.step_min, args.gap_min)
    out.to_parquet(args.out, engine="pyarrow", compression="snappy", index=False)

    # report
    chan = [c for c in df.columns if c not in (["time_utc"] + META_COLS)][0]
    filled = int(out[chan].isna().sum() - df[chan].isna().sum())
    print(f"{os.path.basename(args.inp)}: {n_in:,} rows -> "
          f"{len(out):,} rows  (+{len(out)-n_in:,}; ~{filled:,} new gap rows)")
    # cadence proof: within each series, all intra-segment diffs == step
    bad = 0
    for _, g in out.groupby("series", observed=True):
        d = g["time_utc"].sort_values().diff().dt.total_seconds().dropna() / 60
        d = d[d > 0]
        bad += int(((d != args.step_min) & (d <= args.gap_min)).sum())
    print(f"  intra-segment steps != {args.step_min}min (should be 0): {bad}")
    print(f"  wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
