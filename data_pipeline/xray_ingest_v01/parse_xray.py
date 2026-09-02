"""Pure parser for SWPC GOES X-ray (XRS) 1-minute daily text files.

Sibling of particle_ingest_v01/parse_particle.py. The SWPC GOES daily archive
holds one daily
file per satellite series: `YYYYMMDD_<series>xr_1m.txt`, <series> in
{G8,G10,G11,G12 (2002-2009), Gp (primary, 2009+), Gs (secondary, 2015-2020)}.

Two on-disk formats (verified 2026-06-25), exactly mirroring the particle DB:

  Format A  "noaa"  (2002-2019, NOAA/SWPC): 8 data columns
    YR MO DA HHMM  MJD  SOD  Short  Long
  Format B  "kasi"  (2020-2026, KASI):       7 data columns
    YR MO DA HHMM  SatNo  Short  Long

Channels (both formats, units W/m^2):
  Short = 0.05-0.4 nm  (XRS-A)  -> key "xrs_short"
  Long  = 0.1 - 0.8 nm (XRS-B)  -> key "xrs_long"
Missing is -1.00e+05 (declared in "# Missing data:"). Timestamps are UTC.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

MISSING_DEFAULT = -1.0e5
_MISSING_ATOL = 1.0

CHANNEL_KEYS = ["xrs_short", "xrs_long"]  # XRS-A (0.05-0.4nm), XRS-B (0.1-0.8nm)

_RE_SERIES = re.compile(r"^\d{8}_(G\d{1,2}|Gp|Gs)_?xr_1m\.txt$")
_RE_SOURCE = re.compile(r"#\s*Source:\s*GOES[- ]?(\d+)", re.IGNORECASE)
_RE_LOCATION = re.compile(r"#\s*Location:\s*(\S+)")
_RE_MISSING = re.compile(r"#\s*Missing data:\s*([-\d.eE+]+)")


@dataclass
class FileMeta:
    fmt: str                       # "noaa" | "kasi"
    series: str                    # G8/G10/G11/G12/Gp/Gs
    source_sat: int | None
    location: str | None
    prepared_by: str | None
    missing_value: float
    channel_keys: list[str] = field(default_factory=lambda: list(CHANNEL_KEYS))


@dataclass
class ParsedFile:
    meta: FileMeta
    rows: list[dict]


def series_from_name(filename: str) -> str:
    m = _RE_SERIES.match(filename)
    if not m:
        raise ValueError(f"not an X-ray 1m filename: {filename!r}")
    return m.group(1)


def _detect_format(text: str, prepared_by: str | None) -> str:
    """kasi if prepared by KASI (Sat# layout), else noaa (MJD/SOD layout)."""
    if prepared_by and ("KASI" in prepared_by or "Korea Astronomy" in prepared_by):
        return "kasi"
    if "Julian" in text:
        return "noaa"
    # Fall back on the Satellite-column header wording.
    if "Satellite" in text:
        return "kasi"
    raise ValueError("cannot determine X-ray file format from header")


def parse_header(text: str, filename: str) -> FileMeta:
    series = series_from_name(filename)
    prepared_by = source_sat = location = None
    missing = MISSING_DEFAULT
    for raw in text.splitlines():
        s = raw.rstrip("\n")
        if s.startswith("# Prepared by"):
            prepared_by = s[2:].strip()
        m = _RE_SOURCE.search(s)
        if m:
            source_sat = int(m.group(1))
        m = _RE_LOCATION.search(s)
        if m:
            location = m.group(1)
        m = _RE_MISSING.search(s)
        if m:
            try:
                missing = float(m.group(1))
            except ValueError:
                pass
        if not s.startswith(("#", ":")) and s.strip():
            break
    fmt = _detect_format(text, prepared_by)
    return FileMeta(fmt=fmt, series=series, source_sat=source_sat,
                    location=location, prepared_by=prepared_by,
                    missing_value=missing)


def _val(token: str, missing: float) -> float:
    try:
        v = float(token)
    except ValueError:
        return float("nan")
    if abs(v - missing) <= _MISSING_ATOL:
        return float("nan")
    # Exactly 0.0 W/m^2 is non-physical for XRS flux; the KASI-era files write
    # 0.0 as a no-data sentinel (the declared -1e5 marker is used elsewhere).
    # Mask it so it does not distort the log-scale baseline (spec: mask fills).
    if v == 0.0:
        return float("nan")
    return v


def parse_file(path: str, *, text: str | None = None) -> ParsedFile:
    filename = os.path.basename(path)
    if text is None:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    meta = parse_header(text, filename)
    n_index = 5 if meta.fmt == "kasi" else 6   # leading cols before Short/Long
    n_expected = n_index + len(meta.channel_keys)

    rows: list[dict] = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith(("#", ":")):
            continue
        parts = s.split()
        if len(parts) < n_expected:
            continue
        try:
            yr, mo, da, hhmm = parts[0], parts[1], parts[2], parts[3]
            hh = int(hhmm.zfill(4)[:2])
            mm = int(hhmm.zfill(4)[2:])
            ts = datetime(int(yr), int(mo), int(da), hh, mm, tzinfo=timezone.utc)
        except (ValueError, IndexError):
            continue
        chan = parts[n_index:n_index + len(meta.channel_keys)]
        row = {"time_utc": ts}
        for key, tok in zip(meta.channel_keys, chan):
            row[key] = _val(tok, meta.missing_value)
        rows.append(row)
    return ParsedFile(meta=meta, rows=rows)
