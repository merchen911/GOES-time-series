#!/usr/bin/env python3
"""
GOES proton flux downloader -- v02 (same methodology as the XRS v02 collector).

Implements the proton scope of `markdown/GOES_collection_spec.md`. Runtime tree
discovery (no guessed paths), provenance manifest + sha256, integrity-validated
downloads, resumable, polite, README-first, best-effort netCDF metadata.

Three authoritative sources (verified at runtime), with provenance making the
origin of every value explicit (spec §3.2/§3.4/§5):

  1. SGPS  (GOES-R 16-19) -- science-grade L2 differential proton flux (13 ch),
     differential alpha (11 ch) and the >500 MeV integral channel, East+West
     units. NGDC L2 tree, DAILY files under sgps-l2-avg1m/ and sgps-l2-avg5m/
     (YYYY/MM/). Coverage starts 2020 (g16/g17), 2022 (g18), 2024 (g19).
     sensor=sgps, product_line=L2 (sci_ prefix; no separate _science dir).

  2. Legacy EPEAD/EPS/HEPAD (GOES 08-15) -- NO science-quality reprocessing
     exists (access/science/particles/ is empty), so the operational
     access/avg/ tree is used and flagged product_line=operational (spec §3.3).
     MONTHLY files: gNN_eps_* (08-12, combined E/P/A incl. proton),
     gNN_epead_p17ew_* (13-15 proton P1-P7 E/W), gNN_epead_cpflux_* (corrected
     proton), gNN_hepad_* (high-energy proton/alpha). All differential.

  3. SWPC integral proton flux -- the standard multi-threshold integral flux
     (>=1,5,10,30,50,60,100,500 MeV, pfu) that is NOT in any NCEI instrument
     product. services.swpc.noaa.gov JSON, primary+secondary. OPERATIONAL and
     RECENT ONLY (~7-day rolling window); flagged source=swpc / operational /
     recent-snapshot. Also grabs instrument-sources.json & satellite-longitudes
     .json (machine-readable primary/secondary, useful for the §4 table).

NOTE (recorded in QC): a long-term multi-threshold integral proton series is not
freely available as a single authoritative product; the NCEI products are
differential. We do NOT derive integral from differential (spec forbids
self-computed quantities). The integral here is SWPC recent + SGPS >500 MeV.

Run with python3.12 (needs netCDF4 for metadata). See ../goes_download_common.py.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import goes_download_common as gc  # noqa: E402

# --------------------------------------------------------------------------- #
SGPS_BASE = ("https://data.ngdc.noaa.gov/platforms/solar-space-observing-"
             "satellites/goes/goes{nn}/l2/data/{product}/")
SGPS_PRODUCTS = {"avg1m": "sgps-l2-avg1m", "avg5m": "sgps-l2-avg5m"}
AVG_BASE = ("https://www.ncei.noaa.gov/data/goes-space-environment-monitor"
            "/access/avg/{yyyy}/{mm:02d}/goes{nn:02d}/netcdf/")
SWPC_BASE = "https://services.swpc.noaa.gov/json/goes/{which}/"
SWPC_FILES = [
    "integral-protons-7-day.json",
    "differential-protons-7-day.json",
    "integral-proton-fluence-7-day.json",
]
SWPC_AUX = ["instrument-sources.json", "satellite-longitudes.json"]
DOC_INDEXES = [
    "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites"
    "/goes/goes16/l2/docs/",
    "https://www.ncei.noaa.gov/data/goes-space-environment-monitor"
    "/access/science/",  # legacy SEM docs land near here / readme pdfs
]
PROV_ATTR_KEYS = ("processing_level", "source", "title", "algorithm",
                  "algorithm_version", "date_created", "time_coverage_resolution",
                  "time_coverage_start", "time_coverage_end", "platform",
                  "instrument", "references", "dataset_name", "id")

RE_SGPS = re.compile(
    r"^sci_sgps-l2-(avg1m|avg5m)_g(\d{2})_d(\d{8})_v(\d+-\d+-\d+)\.nc$", re.I)
# legacy proton-relevant instruments (differential); both cadences kept (monthly)
RE_LEGACY_PROTON = re.compile(
    r"^g(\d{2})_(eps|epead_p17ew|epead_cpflux|hepad_ap|hepad_s\d+|hepad)"
    r"_(\d+m)_(\d{8})_(\d{8})(?:_science)?(?:_v[\d.]+)?\.nc$", re.I)
RE_FLUX = re.compile(r"flux|prot", re.I)


def this_year() -> int:
    return datetime.now(timezone.utc).year


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def discover_sgps(nn, product_key, years, *, timeout, max_retries, wait, log):
    """Crawl sgps product YYYY/MM/ for daily .nc files."""
    product = SGPS_PRODUCTS[product_key]
    base = SGPS_BASE.format(nn=f"{nn:02d}", product=product)
    out = []
    try:
        year_dirs = [h for h in gc.list_links(
            base, timeout=timeout, max_retries=max_retries, log=log)
            if re.match(r"^\d{4}/$", h)]
    except Exception as exc:
        log(f"  SGPS discovery failed for goes{nn:02d}/{product}: {exc!r}")
        return out
    time.sleep(wait)
    for yd in year_dirs:
        yr = int(yd.strip("/"))
        if years and yr not in years:
            continue
        try:
            months = [h for h in gc.list_links(
                base + yd, timeout=timeout, max_retries=max_retries, log=log)
                if re.match(r"^\d{2}/$", h)]
        except Exception as exc:
            log(f"  skip {base+yd}: {exc!r}")
            continue
        time.sleep(wait)
        for md in months:
            url = base + yd + md
            try:
                files = [h for h in gc.list_links(
                    url, timeout=timeout, max_retries=max_retries, log=log)
                    if h.lower().endswith(".nc")]
            except Exception as exc:
                log(f"  skip {url}: {exc!r}")
                continue
            time.sleep(wait)
            best = {}
            for name in files:
                m = RE_SGPS.match(name)
                if not m:
                    continue
                day = m.group(3)
                ver = m.group(4)
                vk = tuple(int(x) for x in ver.split("-"))
                if day not in best or vk > best[day][0]:
                    best[day] = (vk, name, ver)
            for day, (_, name, ver) in best.items():
                out.append({
                    "url": url + name, "filename": name, "satellite": f"goes{nn:02d}",
                    "nn": nn, "year": yr, "product": product, "cadence": product_key,
                    "version": ver, "date": day, "sensor": "sgps",
                    "product_line": "L2 (sci_ prefix; SGPS has no separate _science dir)",
                    "rel_subdir": f"raw/proton/goes{nn:02d}/{product}/{yr}",
                })
    return out


def discover_legacy(nn, years, *, timeout, max_retries, wait, log):
    """Crawl access/avg/YYYY/MM/goesNN/netcdf/ for proton-relevant monthly files."""
    out = []
    yr_range = sorted(years) if years else list(range(1986, this_year() + 1))
    for yr in yr_range:
        for mm in range(1, 13):
            url = AVG_BASE.format(yyyy=yr, mm=mm, nn=nn)
            try:
                files = [h for h in gc.list_links(
                    url, timeout=timeout, max_retries=max_retries, log=log)
                    if h.lower().endswith(".nc")]
            except Exception:
                continue  # month/satellite not present -> skip silently
            time.sleep(wait)
            for name in files:
                m = RE_LEGACY_PROTON.match(name)
                if not m:
                    continue
                instr = m.group(2).split("_")[0]  # eps / epead / hepad
                sensor = ("eps" if instr == "eps"
                          else ("hepad" if instr == "hepad" else "epead"))
                out.append({
                    "url": url + name, "filename": name, "satellite": f"goes{nn:02d}",
                    "nn": nn, "year": yr, "month": mm, "sensor": sensor,
                    "channel_token": m.group(2), "cadence": m.group(3),
                    "product_line": "operational",
                    "rel_subdir": f"raw/proton/goes{nn:02d}/{sensor}",
                })
    return out


# --------------------------------------------------------------------------- #
# Record builders
# --------------------------------------------------------------------------- #

REGIME_NOTE = {
    13: "GOES-13 proton instrument early-mission caveats; cross-check regime (spec §6).",
    19: "GOES-19 recent SGPS may be provisional (some channels recalibrated ~2025-04); confirm (spec §6).",
}


def record_sgps(e, rel, res, meta):
    notes = ("SGPS L2 daily file: differential proton (13ch) + alpha (11ch) + "
             ">500 MeV integral, East+West units. Multi-threshold integral NOT "
             "in this product (see SWPC).")
    if e["nn"] in REGIME_NOTE:
        notes += " " + REGIME_NOTE[e["nn"]]
    return {
        "local_path": rel, "satellite": e["satellite"], "instrument": "proton",
        "sensor": "sgps", "channel": "diff_p1-p13,>500MeV_integral,alpha",
        "product": f"GOES-R SEISS/SGPS L2 1-min/5-min ({e['product']})",
        "product_version": e["version"], "product_line": e["product_line"],
        "validation_maturity": None,
        "calibration_regime": "goes-r SGPS (16-19)",
        "regime_flags": (["maturity_check_goes19"] if e["nn"] == 19 else []),
        "source": "ncei", "source_url": e["url"], "retrieved_utc": gc.now_utc(),
        "sha256": res.sha256, "bytes": res.nbytes,
        "native_cadence": "1min" if e["cadence"] == "avg1m" else "5min",
        "year": e["year"], "date": e["date"],
        "flux_units": meta.get("flux_units"), "time_units_raw": meta.get("time_units_raw"),
        "nc_reader": meta.get("_reader"),
        "nc_inspect": ("ok" if meta.get("_reader") and not meta.get("_error")
                       else ("no_reader" if not meta.get("_reader") else "error")),
        "file_global_attrs": meta.get("global_attrs"), "notes": notes,
    }


def record_legacy(e, rel, res, meta):
    notes = (f"Legacy operational {e['sensor'].upper()} ({e['channel_token']}, "
             f"{e['cadence']}); differential proton, East/West. No science-quality "
             f"reprocessing exists for legacy proton (spec §3.3 fallback).")
    if e["nn"] in REGIME_NOTE:
        notes += " " + REGIME_NOTE[e["nn"]]
    sensor_era = ("eps (GOES 08-12)" if e["sensor"] == "eps"
                  else ("hepad high-energy" if e["sensor"] == "hepad"
                        else "epead (GOES 13-15)"))
    return {
        "local_path": rel, "satellite": e["satellite"], "instrument": "proton",
        "sensor": e["sensor"], "channel": e["channel_token"],
        "product": f"NCEI operational SEM avg ({sensor_era})",
        "product_version": None, "product_line": "operational",
        "validation_maturity": "operational (no science-quality product)",
        "calibration_regime": f"legacy {e['sensor']} (08-15); SGPS<->EPEAD/EPS boundary at GOES-16",
        "regime_flags": [], "source": "ncei", "source_url": e["url"],
        "retrieved_utc": gc.now_utc(), "sha256": res.sha256, "bytes": res.nbytes,
        "native_cadence": e["cadence"], "year": e["year"], "month": e["month"],
        "flux_units": meta.get("flux_units"), "time_units_raw": meta.get("time_units_raw"),
        "nc_reader": meta.get("_reader"),
        "nc_inspect": ("ok" if meta.get("_reader") and not meta.get("_error")
                       else ("no_reader" if not meta.get("_reader") else "error")),
        "file_global_attrs": meta.get("global_attrs"), "notes": notes,
    }


# --------------------------------------------------------------------------- #
# Source runners
# --------------------------------------------------------------------------- #


def run_nc_source(entries, out_root, manifest, stats, *, args, log, kind):
    builder = record_sgps if kind == "sgps" else record_legacy
    for e in entries:
        if args.limit is not None and stats["downloaded"] >= args.limit:
            log(f"  --limit {args.limit} reached; stopping {kind}")
            return
        dest = out_root / e["rel_subdir"] / e["filename"]
        rel = str(dest.relative_to(out_root))
        if manifest.has_verified(rel, out_root):
            log(f"  skip (verified): {e['filename']}")
            stats["skipped"] += 1
            continue
        if args.dry_run:
            log(f"  [dry-run] {e['url']}")
            continue
        res = gc.http_download(e["url"], dest, timeout=args.timeout,
                               max_retries=args.max_retries, log=log)
        if res.status != "downloaded":
            log(f"  FAILED (continuing): {e['filename']} -> {res.detail}")
            stats["failed"] += 1
            stats["failures"].append(f"{rel}: {res.detail}")
            time.sleep(args.wait)
            continue
        meta = ({"_reader": None} if args.no_inspect else
                gc.nc_metadata(dest, flux_regex=RE_FLUX, prov_attr_keys=PROV_ATTR_KEYS))
        manifest.append(builder(e, rel, res, meta))
        stats["downloaded"] += 1
        stats["bytes"] += res.nbytes or 0
        log(f"  ok: {e['filename']} ({res.nbytes} B, {e['sensor']}, "
            f"sha {res.sha256[:10]}.., nc={meta.get('_reader')})")
        time.sleep(args.wait)


def _swpc_one(url, local, out_root, manifest, stats, *, args, log, extra):
    dest = out_root / local
    if manifest.has_verified(local, out_root):
        log(f"  skip (verified): {local}")
        stats["skipped"] += 1
        return
    if args.dry_run:
        log(f"  [dry-run] {url}")
        return
    res = gc.http_download(url, dest, timeout=args.timeout,
                           max_retries=args.max_retries, log=log)
    if res.status != "downloaded":
        log(f"  FAILED (continuing): {url} -> {res.detail}")
        stats["failed"] += 1
        stats["failures"].append(f"{local}: {res.detail}")
        time.sleep(args.wait)
        return
    meta = swpc_summary(dest)
    rec = {"local_path": local, "instrument": "proton", "sensor": "swpc-derived",
           "product_line": "operational", "source": "swpc",
           "source_url": url, "retrieved_utc": gc.now_utc(),
           "sha256": res.sha256, "bytes": res.nbytes}
    rec.update(extra)
    if meta:
        rec["channel"] = meta.get("energies")
        rec["satellites_in_file"] = meta.get("satellites")
        rec["time_span"] = meta.get("span")
    manifest.append(rec)
    log(f"  ok: {local} ({res.nbytes} B) energies={meta.get('energies')}")
    time.sleep(args.wait)


def run_swpc(out_root, manifest, stats, *, args, log):
    log("== SWPC integral proton (recent ~7d, operational) ==")
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    for which in ("primary", "secondary"):
        for fname in SWPC_FILES:
            url = SWPC_BASE.format(which=which) + fname
            local = f"raw/proton/swpc/{fname.replace('.json','')}_{which}_{day}.json"
            _swpc_one(url, local, out_root, manifest, stats, args=args, log=log,
                      extra={"product": f"SWPC GOES {which} {fname}",
                             "validation_maturity": "operational/real-time",
                             "units": "pfu = protons/(cm^2 s sr)",
                             "notes": "Standard multi-threshold integral flux; "
                                      "RECENT ~7-day rolling snapshot only (not a "
                                      "long archive). Source: SWPC."})
    # primary/secondary designation aux live at the json/goes/ ROOT (spec §4):
    root = "https://services.swpc.noaa.gov/json/goes/"
    for fname in SWPC_AUX:
        local = f"raw/proton/swpc/{fname.replace('.json','')}_{day}.json"
        _swpc_one(root + fname, local, out_root, manifest, stats, args=args, log=log,
                  extra={"product": f"SWPC {fname}", "kind": "designation-aux",
                         "notes": "Machine-readable primary/secondary instrument "
                                  "mapping (current snapshot; useful for §4 table)."})


def swpc_summary(path):
    try:
        d = json.loads(path.read_text())
        if isinstance(d, list) and d and isinstance(d[0], dict):
            en = sorted({r.get("energy") for r in d if r.get("energy")})
            sats = sorted({r.get("satellite") for r in d if r.get("satellite") is not None})
            ts = sorted({r.get("time_tag") for r in d if r.get("time_tag")})
            return {"energies": en or None, "satellites": sats or None,
                    "span": [ts[0], ts[-1]] if ts else None}
    except Exception:
        pass
    return {}


def fetch_docs(out_root, manifest, *, args, log):
    log("== proton docs (SEISS/SGPS, EPEAD/EPS readmes) ==")
    found = {}
    for idx in DOC_INDEXES:
        try:
            for h in gc.list_links(idx, timeout=args.timeout,
                                   max_retries=args.max_retries, log=log):
                low = h.lower()
                if low.endswith(".pdf") and any(
                        t in low for t in ("seiss", "sgps", "epead", "eps",
                                           "hepad", "particle", "sem")):
                    found[h.rsplit("/", 1)[-1]] = idx + h
        except Exception as exc:
            log(f"  doc index skip {idx}: {exc!r}")
        time.sleep(args.wait)
    if not found:
        log("  (no proton-specific doc PDFs found in indexes)")
    for name, url in sorted(found.items()):
        dest = out_root / "raw" / "proton" / "docs" / name
        rel = str(dest.relative_to(out_root))
        if manifest.has_verified(rel, out_root):
            log(f"  skip (verified): {rel}")
            continue
        if args.dry_run:
            log(f"  [dry-run] {url}")
            continue
        res = gc.http_download(url, dest, timeout=args.timeout,
                               max_retries=args.max_retries, log=log)
        if res.status == "downloaded":
            manifest.append({"local_path": rel, "kind": "documentation",
                             "instrument": "proton", "source_url": url,
                             "retrieved_utc": gc.now_utc(), "sha256": res.sha256,
                             "bytes": res.nbytes})
            log(f"  doc ok: {rel} ({res.nbytes} B)")
        else:
            log(f"  doc FAILED (continuing): {url} -> {res.detail}")
        time.sleep(args.wait)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = PROJECT_ROOT / "data" / "goes_data"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="GOES proton downloader v02")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--sources", default="sgps,legacy,swpc",
                   help="comma list: sgps,legacy,swpc")
    p.add_argument("--start", type=int, default=8)
    p.add_argument("--end", type=int, default=19)
    p.add_argument("--satellite", type=int, action="append", dest="satellites",
                   metavar="N")
    p.add_argument("--year", type=int, action="append", dest="years", metavar="Y")
    p.add_argument("--recent-years", type=int, default=None,
                   help="only the last N calendar years (staged recent-first)")
    p.add_argument("--cadence", default="avg1m,avg5m",
                   help="SGPS cadences: avg1m,avg5m")
    p.add_argument("--wait", type=float, default=1.0)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-docs", action="store_true")
    p.add_argument("--no-inspect", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--list", action="store_true")
    return p.parse_args(argv)


def resolve_years(args):
    if args.years:
        return set(args.years)
    if args.recent_years:
        ty = this_year()
        return set(range(ty - args.recent_years + 1, ty + 1))
    return None


def main(argv=None):
    args = parse_args(argv)
    out_root = args.output_root.resolve()
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    cadences = [c.strip() for c in args.cadence.split(",") if c.strip()]
    sats = (sorted(set(args.satellites)) if args.satellites
            else list(range(args.start, args.end + 1)))
    years = resolve_years(args)

    ts = gc.now_utc().replace(":", "").replace("-", "")
    log = gc.Logger(None if (args.dry_run or args.list)
                    else out_root / "logs" / f"proton_{ts}.log")
    log(f"output root : {out_root}")
    log(f"sources={sources} sats={sats} years={sorted(years) if years else 'ALL'} "
        f"cadence={cadences} limit={args.limit} dry_run={args.dry_run}")

    if args.list:
        for nn in sats:
            if nn >= 16 and "sgps" in sources:
                for ck in cadences:
                    e = discover_sgps(nn, ck, years, timeout=args.timeout,
                                      max_retries=args.max_retries, wait=args.wait, log=log)
                    log(f"  goes{nn:02d} {SGPS_PRODUCTS[ck]}: {len(e)} daily files")
            if nn <= 15 and "legacy" in sources:
                e = discover_legacy(nn, years, timeout=args.timeout,
                                    max_retries=args.max_retries, wait=args.wait, log=log)
                log(f"  goes{nn:02d} legacy: {len(e)} proton files "
                    f"({sorted({x['sensor'] for x in e})})")
        log.close()
        return 0

    manifest = gc.Manifest(out_root / "manifest" / "provenance.jsonl")
    stats = {"downloaded": 0, "skipped": 0, "failed": 0, "bytes": 0, "failures": []}

    if not args.no_docs:
        fetch_docs(out_root, manifest, args=args, log=log)

    if "swpc" in sources:
        run_swpc(out_root, manifest, stats, args=args, log=log)

    for nn in sats:
        if nn >= 16 and "sgps" in sources:
            for ck in cadences:
                log(f"== goes{nn:02d} SGPS {SGPS_PRODUCTS[ck]} ==")
                entries = discover_sgps(nn, ck, years, timeout=args.timeout,
                                        max_retries=args.max_retries, wait=args.wait, log=log)
                log(f"  discovered {len(entries)} daily files")
                run_nc_source(entries, out_root, manifest, stats,
                              args=args, log=log, kind="sgps")
        if nn <= 15 and "legacy" in sources:
            log(f"== goes{nn:02d} legacy proton (operational avg) ==")
            entries = discover_legacy(nn, years, timeout=args.timeout,
                                      max_retries=args.max_retries, wait=args.wait, log=log)
            log(f"  discovered {len(entries)} proton files")
            run_nc_source(entries, out_root, manifest, stats,
                          args=args, log=log, kind="legacy")

    log("== summary ==")
    log(f"  downloaded: {stats['downloaded']}  skipped: {stats['skipped']}  "
        f"failed: {stats['failed']}  bytes: {stats['bytes']:,}")
    for f in stats["failures"][:50]:
        log(f"    - FAIL {f}")
    log.close()
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
