#!/usr/bin/env python3
"""
QC report for the XRS v02 download (spec §7 / §8 acceptance).

Reads the provenance manifest and every downloaded yearly file, and reports:
  * manifest integrity: every raw .nc present is recorded with a matching
    sha256; no orphan files; no duplicate local_path lines;
  * per-satellite coverage: years, version(s), record count, UTC span,
    per-channel masked (missing/out-of-range) fraction, gap count;
  * regime boundaries (legacy v1-0-0 vs v2-2-1; GOES-16 transition; GOES-13
    science-start 2013-06-07);
  * maturity / product_line distribution;
  * inter-satellite temporal overlaps (informational; full cross-calibration
    concordance is a Phase-3/4 step, not part of download QC).

Requires ``netCDF4``. Run with python3.12.
Writes ``<root>/manifest/qc_report.md``.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import netCDF4  # type: ignore
import numpy as np  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = PROJECT_ROOT / "data" / "goes_data"
RE_PRIMARY = re.compile(r"^xrs([ab])_flux$", re.I)


def sha256_file(path: Path) -> str:
    import hashlib
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def masked_fraction(var) -> tuple[int, int]:
    var.set_auto_mask(False)
    raw = np.asarray(var[:], dtype="float64")
    mask = np.zeros(raw.shape, dtype=bool)
    for attr, op in (("_FillValue", "eq"), ("missing_value", "eq")):
        v = getattr(var, attr, None)
        if v is not None:
            mask |= np.isclose(raw, float(v))
    vr = getattr(var, "valid_range", None)
    if vr is not None and len(vr) == 2:
        mask |= (raw < float(vr[0])) | (raw > float(vr[1]))
    for attr, lo in (("valid_min", True), ("valid_max", False)):
        v = getattr(var, attr, None)
        if v is not None:
            mask |= (raw < float(v)) if lo else (raw > float(v))
    return int(mask.sum()), int(raw.size)


def inspect_file(path: Path) -> dict:
    ds = netCDF4.Dataset(str(path), "r")
    try:
        out: dict = {"vars": len(ds.variables)}
        tvar = ds.variables.get("time")
        if tvar is not None:
            tunits = getattr(tvar, "units", None)
            tcal = getattr(tvar, "calendar", "standard")
            tvar.set_auto_mask(False)
            tvals = np.asarray(tvar[:], dtype="float64")
            out["n"] = int(tvals.size)
            d = netCDF4.num2date([tvals[0], tvals[-1]], units=tunits, calendar=tcal)
            out["start"] = str(d[0])
            out["end"] = str(d[1])
            diffs = np.diff(tvals)
            if diffs.size:
                modal = float(np.median(diffs))
                out["cadence_s"] = modal
                out["gaps"] = int(np.sum(diffs > 1.5 * modal))
                out["monotonic"] = bool(np.all(diffs > 0))
        chans = {}
        for v in ds.variables:
            if RE_PRIMARY.match(v):
                nm, n = masked_fraction(ds.variables[v])
                chans[v] = (nm, n)
        out["chans"] = chans
        return out
    finally:
        ds.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    root = args.output_root.resolve()
    man_path = root / "manifest" / "provenance.jsonl"
    records = [json.loads(l) for l in man_path.read_text().splitlines() if l.strip()]
    flux = [r for r in records if r.get("instrument") == "xrs"
            and r.get("kind") != "documentation"]
    docs = [r for r in records if r.get("kind") == "documentation"]

    # ---- manifest integrity ---------------------------------------------- #
    integrity = []
    raw_ncs = sorted((root / "raw" / "xrs").rglob("*.nc"))
    manifest_paths = {r["local_path"] for r in records}
    seen_paths = [r["local_path"] for r in records]
    dups = {p for p in seen_paths if seen_paths.count(p) > 1}
    orphans = [str(p.relative_to(root)) for p in raw_ncs
               if str(p.relative_to(root)) not in manifest_paths]
    bad_sha = []
    for r in flux:
        fp = root / r["local_path"]
        if not fp.exists():
            bad_sha.append((r["local_path"], "MISSING FILE"))
        elif sha256_file(fp) != r.get("sha256"):
            bad_sha.append((r["local_path"], "SHA MISMATCH"))

    # ---- per-satellite aggregate ----------------------------------------- #
    bysat: dict[str, list] = defaultdict(list)
    for r in flux:
        bysat[r["satellite"]].append(r)
    sat_summ = {}
    sat_span = {}
    for sat in sorted(bysat):
        recs = sorted(bysat[sat], key=lambda r: r["year"])
        files = [(root / r["local_path"], r) for r in recs]
        agg_mask = defaultdict(lambda: [0, 0])
        total_n = 0
        gaps = 0
        starts, ends = [], []
        non_monotonic = []
        versions = sorted({r["product_version"] for r in recs})
        for fp, r in files:
            info = inspect_file(fp)
            total_n += info.get("n", 0)
            gaps += info.get("gaps", 0)
            if info.get("start"):
                starts.append(info["start"])
                ends.append(info["end"])
            if info.get("monotonic") is False:
                non_monotonic.append(r["year"])
            for c, (nm, n) in info.get("chans", {}).items():
                agg_mask[c][0] += nm
                agg_mask[c][1] += n
        sat_summ[sat] = {
            "years": [r["year"] for r in recs],
            "versions": versions,
            "n_files": len(recs),
            "total_records": total_n,
            "span": (min(starts) if starts else None, max(ends) if ends else None),
            "gaps": gaps,
            "non_monotonic_years": non_monotonic,
            "mask": {c: (nm / n if n else 0.0) for c, (nm, n) in agg_mask.items()},
            "regime_flags": sorted({f for r in recs for f in r.get("regime_flags", [])}),
        }
        if starts:
            sat_span[sat] = (min(starts), max(ends))

    # ---- inter-satellite overlaps ---------------------------------------- #
    sats = sorted(sat_span)
    overlaps = []
    for i in range(len(sats)):
        for j in range(i + 1, len(sats)):
            a, b = sats[i], sats[j]
            s = max(sat_span[a][0], sat_span[b][0])
            e = min(sat_span[a][1], sat_span[b][1])
            if s < e:
                overlaps.append((a, b, s, e))

    # ---- write report ---------------------------------------------------- #
    L = ["# GOES XRS v02 — QC report (spec §7)", ""]
    ok = not (dups or orphans or bad_sha)
    L.append(f"**Manifest integrity: {'OK' if ok else 'PROBLEMS'}** — "
             f"{len(flux)} flux files, {len(docs)} docs, "
             f"{len(raw_ncs)} raw .nc on disk.")
    if dups:
        L.append(f"- duplicate local_path lines: {sorted(dups)}")
    if orphans:
        L.append(f"- orphan files (on disk, not in manifest): {orphans}")
    if bad_sha:
        L.append(f"- checksum/missing problems: {bad_sha}")
    if ok:
        L.append("- every raw file recorded once with a verified sha256; no orphans.")
    L.append("")

    L.append("## Per-satellite coverage")
    L.append("")
    L.append("| sat | ver | files | records | UTC span | gaps | xrs-a masked | xrs-b masked | flags |")
    L.append("|-----|-----|-------|---------|----------|------|--------------|--------------|-------|")
    for sat in sorted(sat_summ):
        s = sat_summ[sat]
        a = s["mask"].get("xrsa_flux")
        b = s["mask"].get("xrsb_flux")
        span = f"{(s['span'][0] or '')[:10]} → {(s['span'][1] or '')[:10]}"
        L.append(f"| {sat} | {','.join(s['versions'])} | {s['n_files']} | "
                 f"{s['total_records']:,} | {span} | {s['gaps']} | "
                 f"{(f'{a:.2%}' if a is not None else '—')} | "
                 f"{(f'{b:.2%}' if b is not None else '—')} | "
                 f"{','.join(s['regime_flags']) or '—'} |")
    L.append("")
    nonmono = {sat: s["non_monotonic_years"] for sat, s in sat_summ.items()
               if s["non_monotonic_years"]}
    L.append(f"- non-monotonic time axes: {nonmono or 'none (all strictly increasing)'}")
    L.append("")

    L.append("## Regime boundaries (spec §3.2, §6)")
    L += [
        "- **Product-version split:** GOES-08..12 = `v1-0-0`; GOES-13..19 = `v2-2-1`.",
        "- **Calibration regime:** legacy GOES-08..15 (EPS-era) vs GOES-R 16..19; "
        "GOES-16 (~2017) marks the GOES-R XRS calibration transition.",
        "- **GOES-13:** science-quality begins 2013-06-07 (flagged "
        "`goes13_anomalous_mode`); earlier anomalous data must not be spliced in.",
        "- **GOES-19:** recent period may be provisional maturity "
        "(flag `maturity_check_goes19`); confirm against README/version table.",
        "",
    ]

    L.append("## Maturity / product line")
    lines_pl = defaultdict(int)
    for r in flux:
        lines_pl[r.get("product_line", "?")] += 1
    L.append(f"- product_line distribution: {dict(lines_pl)} "
             f"(science-quality preferred per §3.3).")
    L.append("- `validation_maturity` is null: GOES-R XRS files carry no per-file "
             "maturity attribute; maturity is documented in the README/version "
             "table saved under `raw/xrs/docs/`. file-derived `processing_level` "
             "(= 'Level 2') is recorded per file.")
    L.append("")

    L.append("## Inter-satellite temporal overlaps (informational)")
    L.append("Full cross-calibration concordance is a Phase-3/4 step; this only "
             "lists where mission spans overlap (useful for primary/secondary).")
    L.append("")
    for a, b, s, e in overlaps:
        L.append(f"- {a} ∩ {b}: {s[:10]} → {e[:10]}")
    L.append("")

    L.append("## Notes")
    L += [
        "- Time axis decoded from each file's CF `time:units`; GOES products "
        "**neglect leap seconds** (see `time:comments`), so cross-era UTC "
        "alignment is leap-second-approximate (sub-minute).",
        "- Masked fraction = missing (`_FillValue`/`missing_value`) + "
        "out-of-range (`valid_min`/`valid_max`/`valid_range`); eclipse outages "
        "near equinox dominate the XRS gaps.",
        "- Per-satellite full-mission aggregate files were intentionally skipped "
        "(yearly tiles cover the same span); recorded in each manifest line.",
    ]

    report = root / "manifest" / "qc_report.md"
    report.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nreport -> {report}  ({'OK' if ok else 'PROBLEMS'})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
