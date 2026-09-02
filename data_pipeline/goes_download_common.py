#!/usr/bin/env python3
"""
Shared primitives for GOES collectors (extracted from the hardened XRS v02
downloader): polite HTTP with retry/backoff + integrity validation, a
deduped provenance manifest, sha256, and best-effort netCDF metadata.

Standard library only, except the optional netCDF reader used by nc_metadata.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

USER_AGENT = (
    "KASI-GOES-collector/0.2 (proton+xrs science collection; "
    "contact https://github.com/merchen911/GOES-time-series; polite, low-concurrency)"
)
MAX_BACKOFF = 300.0
RETRYABLE_HTTP = (429, 500, 502, 503, 504)
RE_HREF = re.compile(r'href="([^"]+)"', re.I)


class IntegrityError(Exception):
    """Downloaded body failed its size/non-empty check."""


TRANSPORT_ERRORS = (
    urllib.error.URLError, TimeoutError, http.client.HTTPException, OSError,
    IntegrityError,
)


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Logger:
    def __init__(self, log_path: Path | None):
        self.log_path = log_path
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = log_path.open("a", encoding="utf-8")
        else:
            self._fh = None

    def __call__(self, msg: str) -> None:
        line = f"{now_utc()} {msg}"
        print(line, flush=True)
        if self._fh is not None:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()


def _request(url: str, method: str = "GET") -> urllib.request.Request:
    return urllib.request.Request(
        url, method=method, headers={"User-Agent": USER_AGENT})


def _retry_delay(exc, attempt: int) -> float:
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
                return float(min(max((dt - datetime.now(timezone.utc))
                                     .total_seconds(), 1.0), MAX_BACKOFF))
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
                d = _retry_delay(exc, attempt)
                log(f"  [retry {attempt}/{max_retries}] HTTP {exc.code} {url} "
                    f"-> {d:.1f}s")
                time.sleep(d)
                continue
            raise
        except TRANSPORT_ERRORS as exc:
            if attempt <= max_retries:
                d = min(2 ** attempt, MAX_BACKOFF)
                log(f"  [retry {attempt}/{max_retries}] {exc} {url} -> {d:.1f}s")
                time.sleep(d)
                continue
            raise


def list_links(url: str, *, timeout: int, max_retries: int, log: Logger) -> list[str]:
    """Return child hrefs of an Apache directory index (no boilerplate)."""
    html = http_text(url, timeout=timeout, max_retries=max_retries, log=log)
    out = []
    for href in RE_HREF.findall(html):
        if href.startswith(("?", "/", "http", "mailto:", "#", "javascript")):
            continue
        if href in ("ngdc.html", "privacy.html"):
            continue
        out.append(href)
    return out


@dataclass
class DownloadResult:
    status: str
    sha256: str | None = None
    nbytes: int | None = None
    detail: str = ""


def http_download(url: str, dest: Path, *, timeout: int, max_retries: int,
                  log: Logger) -> DownloadResult:
    """Stream to <name>.part with Content-Length + non-empty validation, then
    atomic rename. Never promotes a truncated/empty body."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    attempt = 0
    while True:
        attempt += 1
        sha = hashlib.sha256()
        nbytes = 0
        expected: int | None = None
        try:
            with urllib.request.urlopen(_request(url), timeout=timeout) as resp:
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
                raise IntegrityError("empty response body")
            if expected is not None and nbytes != expected:
                raise IntegrityError(
                    f"size mismatch: {nbytes} != Content-Length {expected}")
            part.replace(dest)
            return DownloadResult("downloaded", sha.hexdigest(), nbytes)
        except urllib.error.HTTPError as exc:
            part.unlink(missing_ok=True)
            if exc.code in RETRYABLE_HTTP and attempt <= max_retries:
                d = _retry_delay(exc, attempt)
                log(f"  [retry {attempt}/{max_retries}] HTTP {exc.code} {url} "
                    f"-> {d:.1f}s")
                time.sleep(d)
                continue
            return DownloadResult("failed", detail=f"HTTP {exc.code}")
        except TRANSPORT_ERRORS as exc:
            part.unlink(missing_ok=True)
            if attempt <= max_retries:
                d = min(2 ** attempt, MAX_BACKOFF)
                log(f"  [retry {attempt}/{max_retries}] {exc!r} {url} -> {d:.1f}s")
                time.sleep(d)
                continue
            return DownloadResult("failed", detail=repr(exc))
        except Exception as exc:
            part.unlink(missing_ok=True)
            return DownloadResult("failed", detail=f"unexpected {exc!r}")


def sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


class Manifest:
    """JSONL provenance, deduped by local_path (1 file = 1 line)."""

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

    def has_verified(self, rel: str, root: Path) -> bool:
        if rel not in self.by_local_path:
            return False
        fp = root / rel
        rec = self.by_local_path[rel]
        return fp.exists() and rec.get("sha256") and sha256_file(fp) == rec["sha256"]

    def append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lp = record["local_path"]
        if lp in self.by_local_path:
            self.by_local_path[lp] = record
            tmp = self.path.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                for rec in self.by_local_path.values():
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            tmp.replace(self.path)
        else:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.by_local_path[lp] = record


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


def nc_metadata(path: Path, *, flux_regex: re.Pattern | None = None,
                prov_attr_keys: tuple[str, ...] = ()) -> dict:
    """Best-effort: time_units_raw, units of flux-like variables, curated global
    attrs. {'_reader': None} if no reader. Never raises."""
    try:
        import netCDF4  # type: ignore
    except Exception:
        return {"_reader": None}
    try:
        ds = netCDF4.Dataset(str(path), "r")
        try:
            variables = list(ds.variables.keys())
            tname = "time" if "time" in ds.variables else (
                "time_tag" if "time_tag" in ds.variables else None)
            time_units = getattr(ds.variables[tname], "units", None) if tname else None
            units = {}
            if flux_regex is not None:
                for v in variables:
                    if flux_regex.search(v):
                        u = getattr(ds.variables[v], "units", None)
                        if u:
                            units[v] = _coerce(u)
            gattrs = {k: _coerce(getattr(ds, k)) for k in ds.ncattrs()}
            file_attrs = {k: gattrs[k] for k in prov_attr_keys if k in gattrs}
            return {
                "_reader": "netCDF4",
                "time_units_raw": _coerce(time_units),
                "flux_units": units or None,
                "n_variables": len(variables),
                "global_attrs": file_attrs or None,
            }
        finally:
            ds.close()
    except Exception as exc:
        return {"_reader": "netCDF4", "_error": repr(exc)}
