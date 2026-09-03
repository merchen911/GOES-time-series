#!/usr/bin/env python3
"""
GOES XRS science-quality L2 1-min average downloader -- v02.

Implements the XRS scope of the collection spec in
``workdir/data_download/markdown/GOES_collection_spec.md``.

Highlights (vs v01's bare ``wget`` dump):

* Runtime discovery: the real file list AND the doc/README list are parsed from
  the open Apache directory index at run time -- no guessed leaf paths
  (spec hard-rule #3, §3.4/§3.5). Re-verified every run.
* Downloads the **yearly science aggregates**
  ``sci_xrsf-l2-avg1m_g<NN>_y<YYYY>_v<ver>.nc`` (same science-quality ``v``
  product as the daily files, ~10 requests/sat -> polite, spec #9). The
  per-satellite full-mission aggregate is skipped to avoid duplication and the
  skip is recorded in the manifest (auditable).
* Integrity: each file is streamed to ``<name>.part``; the byte count is checked
  against the server ``Content-Length`` and a zero-byte body is rejected, so a
  truncated/empty response is NEVER promoted to a 'good' file with a misleading
  sha256. Only a size-validated file is atomically renamed into ``raw/``
  (spec #1 immutability, #2/#5 integrity).
* ``http.client.IncompleteRead`` and other transient transport errors are
  retried (exponential backoff, ``Retry-After`` honored but capped), not fatal
  to the whole run (spec #9). Every request is throttled, discovery included.
* Provenance manifest (one JSON line per file, deduped by ``local_path``) with
  sha256, the real per-file version string, regime-boundary metadata
  (GOES-16 calibration transition; GOES-13 anomalous-mode start 2013-06-07;
  GOES-19 possibly-provisional note), and file-derived global attributes
  (processing_level, algorithm, date_created, ...) read from the netCDF itself
  (spec §5, §3.2, §6). The canonical ``units_by_channel`` records only the two
  primary irradiance channels (xrs-a, xrs-b), not every flux-named variable.
* README / User's Guide fetched first into ``raw/xrs/docs/`` (spec §3.4, §8).
* Idempotent / resumable: a file already present whose sha256 matches its
  manifest entry is skipped (spec #5).

The core download path uses only the Python standard library. netCDF metadata
(time:units, channel units, global attrs) is filled when a netCDF reader is
importable; run under an interpreter that has ``netCDF4`` (e.g. ``python3.12``)
so the §5 provenance fields are populated.

Sources (verified reachable; re-verified each run via the index):
  GOES-R 16-19 : https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes<NN>/l2/data/xrsf-l2-avg1m_science/
  Legacy 08-15 : https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs/goes<NN>/xrsf-l2-avg1m_science/
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constants / source layout
# --------------------------------------------------------------------------- #

NGDC_DATA = (
    "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites"
    "/goes/goes{nn}/l2/data/xrsf-l2-avg1m_science/"
)
NCEI_DATA = (
    "https://www.ncei.noaa.gov/data/goes-space-environment-monitor"
    "/access/science/xrs/goes{nn}/xrsf-l2-avg1m_science/"
)
# Documentation index directories crawled at runtime (spec §3.4/§3.5):
DOC_INDEXES = [
    "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites"
    "/goes/goes16/l2/docs/",
    "https://www.ncei.noaa.gov/data/goes-space-environment-monitor"
    "/access/science/xrs/",
]
# Last-resort hint only if the index crawl finds nothing (NOT authoritative):
DOC_FALLBACK = [
    ("https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites"
     "/goes/goes16/l2/docs/GOES-R_XRS_L2_Data_ReadMe.pdf",
     "GOES-R_XRS_L2_Data_ReadMe.pdf"),
    ("https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites"
     "/goes/goes16/l2/docs/GOES-R_XRS_L2_Data_Users_Guide.pdf",
     "GOES-R_XRS_L2_Data_Users_Guide.pdf"),
    ("https://www.ncei.noaa.gov/data/goes-space-environment-monitor"
     "/access/science/xrs/GOES_1-15_XRS_Science-Quality_Data_Readme.pdf",
     "GOES_1-15_XRS_Science-Quality_Data_Readme.pdf"),
]

PRODUCT_DESC = "science-quality L2 1-min avg (xrsf-l2-avg1m_science)"
USER_AGENT = (
    "KASI-GOES-collector/0.2 (XRS science-quality; "
    "contact https://github.com/merchen911/GOES-time-series; polite, low-concurrency)"
)
MAX_BACKOFF = 300.0  # cap any backoff / Retry-After at 5 minutes

RE_YEARLY = re.compile(
    r"^sci_xrsf-l2-avg1m_g(\d{2})_y(\d{4})_v(\d+-\d+-\d+)\.nc$", re.I
)
RE_FULLMISSION = re.compile(
    r"^sci_xrsf-l2-avg1m_g(\d{2})_s(\d{8})_e(\d{8})_v(\d+-\d+-\d+)\.nc$", re.I
)
RE_HREF = re.compile(r'href="([^"]+)"', re.I)
# the two primary irradiance channels only (spec §3.1 / §5), case-insensitive
RE_PRIMARY_CHANNEL = re.compile(r"^xrs([ab])_flux$", re.I)
# global attributes worth keeping in provenance (file-derived; spec §3.4)
PROV_ATTR_KEYS = (
    "processing_level", "processing_level_description", "source", "algorithm",
    "algorithm_version", "algorithm_date", "date_created",
    "time_coverage_resolution", "time_coverage_start", "time_coverage_end",
    "platform", "instrument", "institution", "references", "dataset_name", "id",
)


def sat_data_url(nn: int) -> str:
    return (NGDC_DATA if nn >= 16 else NCEI_DATA).format(nn=f"{nn:02d}")


class IntegrityError(Exception):
    """Raised when a downloaded body fails its size/non-empty check."""


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


class Logger:
    def __init__(self, log_path: Path | None):
        self.log_path = log_path
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = log_path.open("a", encoding="utf-8")
        else:
            self._fh = None

    def __call__(self, msg: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"{stamp} {msg}"
        print(line, flush=True)
        if self._fh is not None:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()


# --------------------------------------------------------------------------- #
# HTTP with polite retry/backoff + integrity validation
# --------------------------------------------------------------------------- #

RETRYABLE_HTTP = (429, 500, 502, 503, 504)
TRANSPORT_ERRORS = (
    urllib.error.URLError, TimeoutError, http.client.HTTPException, OSError,
    IntegrityError,
)


def _request(url: str, method: str = "GET") -> urllib.request.Request:
    return urllib.request.Request(
        url, method=method, headers={"User-Agent": USER_AGENT}
    )


def _retry_delay(exc, attempt: int) -> float:
    """Honor Retry-After (seconds or HTTP-date) but cap it; else backoff."""
    headers = getattr(exc, "headers", None)
    ra = headers.get("Retry-After") if headers else None
    if ra:
        ra = ra.strip()
        if ra.isdigit():
            return float(min(int(ra), MAX_BACKOFF))
        try:
            dt = parsedate_to_datetime(ra)
            if dt is not None:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                delta = (dt - datetime.now(timezone.utc)).total_seconds()
                return float(min(max(delta, 1.0), MAX_BACKOFF))
        except Exception:
            pass
    return float(min(2 ** attempt, MAX_BACKOFF))


def http_text(url: str, *, timeout: int, max_retries: int, log: Logger) -> str:
    attempt = 0
    while True:
        attempt += 1
        try:
            with urllib.request.urlopen(_request(url), timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code in RETRYABLE_HTTP and attempt <= max_retries:
                delay = _retry_delay(exc, attempt)
                log(f"  [retry {attempt}/{max_retries}] HTTP {exc.code} on "
                    f"{url} -> sleeping {delay:.1f}s")
                time.sleep(delay)
                continue
            raise
        except TRANSPORT_ERRORS as exc:
            if attempt <= max_retries:
                delay = min(2 ** attempt, MAX_BACKOFF)
                log(f"  [retry {attempt}/{max_retries}] {exc} on {url} -> "
                    f"sleeping {delay:.1f}s")
                time.sleep(delay)
                continue
            raise


@dataclass
class DownloadResult:
    status: str  # "downloaded" | "failed"
    sha256: str | None = None
    nbytes: int | None = None
    detail: str = ""


def http_download(
    url: str, dest: Path, *, timeout: int, max_retries: int, log: Logger
) -> DownloadResult:
    """Stream ``url`` to ``dest`` via ``.part`` + atomic rename, validating the
    byte count against Content-Length and rejecting empty bodies."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    attempt = 0
    while True:
        attempt += 1
        sha = hashlib.sha256()
        nbytes = 0
        expected: int | None = None
        try:
            try:
                with urllib.request.urlopen(
                    _request(url), timeout=timeout
                ) as resp:
                    cl = resp.headers.get("Content-Length")
                    expected = int(cl) if (cl and cl.strip().isdigit()) else None
                    with part.open("wb") as fh:
                        while True:
                            chunk = resp.read(1 << 16)
                            if not chunk:
                                break
                            fh.write(chunk)
                            sha.update(chunk)
                            nbytes += len(chunk)
                if nbytes == 0:
                    raise IntegrityError("empty response body (0 bytes)")
                if expected is not None and nbytes != expected:
                    raise IntegrityError(
                        f"size mismatch: got {nbytes} != Content-Length {expected}"
                    )
            finally:
                # If we did not (yet) promote .part, ensure it does not linger.
                if part.exists() and not dest.exists():
                    pass  # keep for the success rename below
            part.replace(dest)  # atomic, same-filesystem
            if expected is None:
                log(f"  (note: no Content-Length; size unverified) {dest.name}")
            return DownloadResult("downloaded", sha.hexdigest(), nbytes)
        except urllib.error.HTTPError as exc:
            part.unlink(missing_ok=True)
            if exc.code in RETRYABLE_HTTP and attempt <= max_retries:
                delay = _retry_delay(exc, attempt)
                log(f"  [retry {attempt}/{max_retries}] HTTP {exc.code} on "
                    f"{url} -> sleeping {delay:.1f}s")
                time.sleep(delay)
                continue
            return DownloadResult("failed", detail=f"HTTP {exc.code}")
        except TRANSPORT_ERRORS as exc:
            part.unlink(missing_ok=True)
            if attempt <= max_retries:
                delay = min(2 ** attempt, MAX_BACKOFF)
                log(f"  [retry {attempt}/{max_retries}] {exc!r} on {url} -> "
                    f"sleeping {delay:.1f}s")
                time.sleep(delay)
                continue
            return DownloadResult("failed", detail=repr(exc))
        except Exception as exc:  # truly unexpected -> clean up, do not loop
            part.unlink(missing_ok=True)
            return DownloadResult("failed", detail=f"unexpected {exc!r}")


def sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


@dataclass
class FileEntry:
    satellite: str
    nn: int
    year: int
    version: str
    filename: str
    url: str

    @property
    def version_key(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self.version.split("-"))


def discover_satellite(
    nn: int, *, timeout: int, max_retries: int, log: Logger
) -> tuple[list[FileEntry], list[str]]:
    """(yearly entries [latest version per year], full-mission filenames)."""
    base = sat_data_url(nn)
    html = http_text(base, timeout=timeout, max_retries=max_retries, log=log)
    yearly: dict[int, FileEntry] = {}
    fullmission: list[str] = []
    for href in RE_HREF.findall(html):
        name = href.rsplit("/", 1)[-1]
        m = RE_YEARLY.match(name)
        if m:
            entry = FileEntry(
                satellite=f"goes{nn:02d}", nn=nn, year=int(m.group(2)),
                version=m.group(3), filename=name, url=base + name,
            )
            prev = yearly.get(entry.year)
            if prev is None or entry.version_key > prev.version_key:
                yearly[entry.year] = entry
        elif RE_FULLMISSION.match(name):
            fullmission.append(name)
    return [yearly[y] for y in sorted(yearly)], sorted(set(fullmission))


def discover_docs(*, timeout: int, max_retries: int, log: Logger, wait: float
                  ) -> list[tuple[str, str]]:
    """Crawl the doc index dirs for XRS README / User's-Guide PDFs (spec §3.4)."""
    found: dict[str, str] = {}  # name -> url
    for idx in DOC_INDEXES:
        try:
            html = http_text(idx, timeout=timeout, max_retries=max_retries, log=log)
        except Exception as exc:
            log(f"  doc index unreachable, skipping: {idx} ({exc!r})")
            time.sleep(wait)
            continue
        for href in RE_HREF.findall(html):
            name = href.rsplit("/", 1)[-1]
            low = name.lower()
            if (low.endswith(".pdf") and "xrs" in low
                    and ("readme" in low or "guide" in low or "users" in low)):
                found[name] = idx + name
        time.sleep(wait)
    if not found:
        log("  doc index crawl found nothing; using fallback hints")
        return DOC_FALLBACK
    return [(url, name) for name, url in sorted(found.items())]


# --------------------------------------------------------------------------- #
# Provenance manifest (deduped by local_path: 1 file = 1 line)
# --------------------------------------------------------------------------- #


class Manifest:
    def __init__(self, path: Path):
        self.path = path
        self.by_local_path: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "local_path" in rec:
                    self.by_local_path[rec["local_path"]] = rec

    def append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lp = record["local_path"]
        if lp in self.by_local_path:
            self.by_local_path[lp] = record
            self._rewrite()
        else:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.by_local_path[lp] = record

    def _rewrite(self) -> None:
        tmp = self.path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for rec in self.by_local_path.values():
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp.replace(self.path)


# --------------------------------------------------------------------------- #
# Best-effort netCDF metadata
# --------------------------------------------------------------------------- #


def _coerce(value):
    try:
        import numpy as np  # type: ignore
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def nc_metadata(path: Path) -> dict:
    """Return time_units_raw, canonical units_by_channel (xrs-a/xrs-b only),
    variables, and global attrs. {'_reader': None} if no reader. Never raises."""
    try:
        import netCDF4  # type: ignore
    except Exception:
        return {"_reader": None}
    try:
        ds = netCDF4.Dataset(str(path), "r")
        try:
            variables = list(ds.variables.keys())
            time_units = (getattr(ds.variables["time"], "units", None)
                          if "time" in ds.variables else None)
            units_by_channel = {}
            for v in variables:
                m = RE_PRIMARY_CHANNEL.match(v)
                if m:
                    key = "xrs-a" if m.group(1).lower() == "a" else "xrs-b"
                    u = getattr(ds.variables[v], "units", None)
                    if u:
                        units_by_channel[key] = _coerce(u)
            gattrs = {k: _coerce(getattr(ds, k)) for k in ds.ncattrs()}
            return {
                "_reader": "netCDF4",
                "time_units_raw": _coerce(time_units),
                "units_by_channel": units_by_channel or None,
                "variables": variables,
                "global_attrs": gattrs,
            }
        finally:
            ds.close()
    except Exception as exc:
        return {"_reader": "netCDF4", "_error": repr(exc)}


# --------------------------------------------------------------------------- #
# Regime-boundary metadata (spec §3.2, §6) -- flag only, never modify data
# --------------------------------------------------------------------------- #


def regime_metadata(nn: int, year: int) -> tuple[str, list[str], list[str]]:
    if nn >= 16:
        regime = "goes-r (GOES-16..19); GOES-16 onward = new XRS calibration regime"
    else:
        regime = "pre-goes16 (legacy GOES-08..15, EPS-era reprocessed)"
    flags: list[str] = []
    notes: list[str] = []
    if nn == 16:
        notes.append("GOES-16 marks the GOES-R XRS calibration-regime transition "
                     "(~2017); do not splice across the legacy/GOES-R boundary "
                     "without flagging (spec §6).")
    if nn == 13:
        flags.append("goes13_anomalous_mode")
        notes.append("GOES-13 XRS science-quality begins 2013-06-07; early-mission "
                     "anomalous/dip behavior preceded it -- do not splice pre-science "
                     "data (spec §6).")
    if nn == 19:
        flags.append("maturity_check_goes19")
        notes.append("GOES-19 recent period may be 'provisional' maturity; confirm "
                     "against README/version table (spec §6).")
    return regime, flags, notes


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


@dataclass
class Stats:
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0
    bytes_down: int = 0
    failures: list[str] = field(default_factory=list)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def satellites_from_args(args) -> list[int]:
    if args.satellites:
        return sorted(set(args.satellites))
    if args.start > args.end:
        raise SystemExit(f"--start ({args.start}) must be <= --end ({args.end})")
    return list(range(args.start, args.end + 1))


def fetch_docs(out_root: Path, manifest: Manifest, *, args, log: Logger) -> None:
    docs_dir = out_root / "raw" / "xrs" / "docs"
    log("== Documentation (README / User's Guide) first ==")
    docs = discover_docs(timeout=args.timeout, max_retries=args.max_retries,
                         log=log, wait=args.wait)
    for url, name in docs:
        dest = docs_dir / name
        rel = str(dest.relative_to(out_root))
        if dest.exists() and rel in manifest.by_local_path:
            if sha256_file(dest) == manifest.by_local_path[rel].get("sha256"):
                log(f"  skip (verified): {rel}")
                continue
        if args.dry_run:
            log(f"  [dry-run] would download doc: {url}")
            continue
        res = http_download(url, dest, timeout=args.timeout,
                            max_retries=args.max_retries, log=log)
        if res.status == "downloaded":
            manifest.append({
                "local_path": rel, "kind": "documentation", "instrument": "xrs",
                "source_url": url, "retrieved_utc": _now(),
                "sha256": res.sha256, "bytes": res.nbytes,
            })
            log(f"  doc ok: {rel} ({res.nbytes} bytes)")
        else:
            log(f"  doc FAILED (continuing): {url} -> {res.detail}")
        time.sleep(args.wait)


def build_record(entry: FileEntry, rel: str, res: DownloadResult, meta: dict,
                 fullmission: list[str]) -> dict:
    regime, flags, regime_notes = regime_metadata(entry.nn, entry.year)
    gattrs = meta.get("global_attrs") or {}
    file_attrs = {k: gattrs[k] for k in PROV_ATTR_KEYS if k in gattrs}
    maturity = None
    for k, v in gattrs.items():
        if "matur" in k.lower():
            maturity = _coerce(v)
            break
    notes = "yearly science aggregate (daily files exist; aggregate chosen for politeness)."
    if regime_notes:
        notes += " " + " ".join(regime_notes)
    return {
        "local_path": rel,
        "satellite": entry.satellite,
        "instrument": "xrs",
        "sensor": "xrs",
        "channel": "xrs-a,xrs-b",
        "product": PRODUCT_DESC,
        "product_version": entry.version,
        "product_line": "science-quality",
        "validation_maturity": maturity,  # null if no per-file maturity attr
        "calibration_regime": regime,
        "regime_flags": flags,
        "source_url": entry.url,
        "retrieved_utc": _now(),
        "sha256": res.sha256,
        "bytes": res.nbytes,
        "native_cadence": "1min",
        "year": entry.year,
        "units_by_channel": meta.get("units_by_channel"),
        "time_units_raw": meta.get("time_units_raw"),
        "nc_reader": meta.get("_reader"),
        "nc_inspect": ("ok" if meta.get("_reader") and not meta.get("_error")
                       else ("no_reader" if not meta.get("_reader") else "error")),
        "file_global_attrs": file_attrs or None,
        "fullmission_aggregate_skipped": fullmission or None,
        "notes": notes,
    }


def download_satellite(nn: int, out_root: Path, manifest: Manifest, stats: Stats,
                       *, args, log: Logger) -> None:
    sat = f"goes{nn:02d}"
    base = sat_data_url(nn)
    log(f"== {sat} == ({base})")
    try:
        entries, fullmission = discover_satellite(
            nn, timeout=args.timeout, max_retries=args.max_retries, log=log)
    except Exception as exc:
        log(f"  DISCOVERY FAILED for {sat}: {exc!r} -- skipping (not guessing)")
        stats.failed += 1
        stats.failures.append(f"{sat}: discovery {exc!r}")
        return
    time.sleep(args.wait)  # throttle discovery request too (spec #9)

    if not entries:
        log(f"  no yearly science files found for {sat} -- GAP (no science "
            f"product; not falling back to operational in this run)")
        return
    if fullmission:
        log(f"  full-mission aggregate available, skipped to avoid duplication: "
            f"{fullmission}")

    all_years = sorted({e.year for e in entries})
    if args.years:
        entries = [e for e in entries if e.year in set(args.years)]
    versions = sorted({e.version for e in entries})
    log(f"  years {all_years[0]}..{all_years[-1]} ({len(all_years)}), "
        f"versions={versions}; selected {len(entries)} file(s)")

    dest_dir = out_root / "raw" / "xrs" / sat
    for entry in entries:
        if args.limit is not None and stats.downloaded >= args.limit:
            log(f"  --limit {args.limit} reached; stopping")
            return
        dest = dest_dir / entry.filename
        rel = str(dest.relative_to(out_root))
        if dest.exists() and rel in manifest.by_local_path:
            recorded = manifest.by_local_path[rel].get("sha256")
            if recorded and sha256_file(dest) == recorded:
                log(f"  skip (verified): {entry.filename}")
                stats.skipped += 1
                continue
            log(f"  re-download (sha mismatch/unverified): {entry.filename}")
        if args.dry_run:
            log(f"  [dry-run] {entry.url}")
            continue

        res = http_download(entry.url, dest, timeout=args.timeout,
                            max_retries=args.max_retries, log=log)
        if res.status != "downloaded":
            log(f"  FAILED (continuing): {entry.filename} -> {res.detail}")
            stats.failed += 1
            stats.failures.append(f"{rel}: {res.detail}")
            time.sleep(args.wait)
            continue
        meta = ({"_reader": None} if args.no_inspect else nc_metadata(dest))
        manifest.append(build_record(entry, rel, res, meta, fullmission))
        stats.downloaded += 1
        stats.bytes_down += res.nbytes or 0
        log(f"  ok: {entry.filename} ({res.nbytes} bytes, v{entry.version}, "
            f"sha {res.sha256[:12]}..., nc={meta.get('_reader')})")
        time.sleep(args.wait)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = PROJECT_ROOT / "data" / "goes_data"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download GOES XRS science-quality L2 1-min averages (v02).")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--start", type=int, default=8)
    p.add_argument("--end", type=int, default=19)
    p.add_argument("--satellite", type=int, action="append", dest="satellites",
                   metavar="N", help="only this satellite (repeatable)")
    p.add_argument("--year", type=int, action="append", dest="years",
                   metavar="Y", help="only this year (repeatable)")
    p.add_argument("--limit", type=int, default=None,
                   help="stop after N newly downloaded flux files (testing)")
    p.add_argument("--wait", type=float, default=1.0,
                   help="seconds between requests (politeness; default 1.0)")
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--no-docs", action="store_true")
    p.add_argument("--docs-only", action="store_true")
    p.add_argument("--no-inspect", action="store_true",
                   help="skip netCDF metadata extraction")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--list", action="store_true",
                   help="discover and list files per satellite, then exit")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_root: Path = args.output_root.resolve()
    sats = satellites_from_args(args)

    ts = _now().replace(":", "").replace("-", "")
    log = Logger(None if (args.dry_run or args.list)
                 else out_root / "logs" / f"download_{ts}.log")
    log(f"output root : {out_root}")
    log(f"satellites  : {', '.join(f'GOES-{s:02d}' for s in sats)}")
    log(f"wait={args.wait}s timeout={args.timeout}s retries={args.max_retries} "
        f"limit={args.limit} dry_run={args.dry_run}")

    if args.list:
        if args.years or args.limit is not None:
            log("note: --year/--limit are ignored in --list mode")
        for nn in sats:
            try:
                entries, full = discover_satellite(
                    nn, timeout=args.timeout, max_retries=args.max_retries, log=log)
            except Exception as exc:
                log(f"GOES-{nn:02d}: discovery failed {exc!r}")
                time.sleep(args.wait)
                continue
            vers = sorted({e.version for e in entries})
            yrs = [e.year for e in entries]
            log(f"GOES-{nn:02d}: {len(entries)} yearly files "
                f"{yrs[:1]}..{yrs[-1:]} versions={vers} fullmission={full}")
            time.sleep(args.wait)
        log.close()
        return 0

    manifest = Manifest(out_root / "manifest" / "provenance.jsonl")
    stats = Stats()
    if not args.no_docs:
        fetch_docs(out_root, manifest, args=args, log=log)
    if args.docs_only:
        log("docs-only: done.")
        log.close()
        return 0
    for nn in sats:
        download_satellite(nn, out_root, manifest, stats, args=args, log=log)

    log("== summary ==")
    log(f"  downloaded: {stats.downloaded}  skipped: {stats.skipped}  "
        f"failed: {stats.failed}  bytes: {stats.bytes_down:,}")
    for f in stats.failures:
        log(f"    - FAIL {f}")
    log.close()
    return 1 if stats.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
