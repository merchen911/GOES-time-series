"""Pure parser for SWPC GOES particle 5-minute daily text files.

The SWPC GOES daily archive holds one daily text file per satellite series:
`YYYYMMDD_<series>part_5m.txt` where <series> in {G8,G9,G10,G11,G12,Gp,Gs}.

Two on-disk formats exist (verified 2026-06-25):

  Format A  "noaa"  (1998-2019, "Prepared by ... NOAA/SWPC"): 15 data columns
    YR MO DA HHMM  MJD  SOD  P>1 P>5 P>10 P>30 P>50 P>100  E_lo E>2.0 E>4.0
    6 proton integral thresholds (NO >60, NO >500).
    E_lo label is E>0.6 (<=2009) or E>0.8 (>=2010) -- read from the header.

  Format B  "kasi"  (2020-2026, "Prepared by ... KASI"): 14 data columns
    YR MO DA HHMM  SatNo  P>1 P>5 P>10 P>30 P>50 P>60 P>100 P>500  E>2.0
    8 proton integral thresholds (adds >60 and >500); electron E>2.0 only.

Both encode missing as -1.00e+05 (declared in the "# Missing data:" header line).
Units: protons/(cm^2 s sr) and electrons/(cm^2 s sr). All timestamps are UTC.

The parser is stdlib-only and pure (no I/O side effects beyond reading the file
it is handed). Missing values are mapped to float('nan'). Thresholds absent in a
given format are simply not present in the returned per-row dict.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

MISSING_DEFAULT = -1.0e5
# How close to the declared missing sentinel still counts as "missing".
_MISSING_ATOL = 1.0

# Unified canonical channel keys (a row dict uses a subset of these).
PROTON_KEYS = ["p_gt1", "p_gt5", "p_gt10", "p_gt30", "p_gt50",
               "p_gt60", "p_gt100", "p_gt500"]
ELECTRON_KEYS = ["e_gt0p6", "e_gt0p8", "e_gt2", "e_gt4"]
ALL_CHANNEL_KEYS = PROTON_KEYS + ELECTRON_KEYS

# Column layouts (channel keys in physical order, AFTER the leading time/index
# columns). Format A has two index columns (MJD, SOD); B has one (SatNo).
_LAYOUT_NOAA_06 = ["p_gt1", "p_gt5", "p_gt10", "p_gt30", "p_gt50", "p_gt100",
                   "e_gt0p6", "e_gt2", "e_gt4"]
_LAYOUT_NOAA_08 = ["p_gt1", "p_gt5", "p_gt10", "p_gt30", "p_gt50", "p_gt100",
                   "e_gt0p8", "e_gt2", "e_gt4"]
_LAYOUT_KASI = ["p_gt1", "p_gt5", "p_gt10", "p_gt30", "p_gt50", "p_gt60",
                "p_gt100", "p_gt500", "e_gt2"]


@dataclass
class FileMeta:
    fmt: str                       # "noaa" | "kasi"
    series: str                    # G8/G9/G10/G11/G12/Gp/Gs (from filename)
    source_sat: int | None         # GOES number from "# Source:" line
    location: str | None           # e.g. "W075"
    prepared_by: str | None
    missing_value: float
    channel_keys: list[str] = field(default_factory=list)  # in physical order
    date_label: str = "UTC"        # "UT" or "UTC" header wording


@dataclass
class ParsedFile:
    meta: FileMeta
    rows: list[dict]               # each: {"time_utc": datetime, <channel>: float}


_RE_SERIES = re.compile(r"^\d{8}_(G\d{1,2}|Gp|Gs)_?part_5m\.txt$")
_RE_SOURCE = re.compile(r"#\s*Source:\s*GOES[- ]?(\d+)", re.IGNORECASE)
_RE_LOCATION = re.compile(r"#\s*Location:\s*(\S+)")
_RE_MISSING = re.compile(r"#\s*Missing data:\s*([-\d.eE+]+)")


def series_from_name(filename: str) -> str:
    m = _RE_SERIES.match(filename)
    if not m:
        raise ValueError(f"not a particle 5m filename: {filename!r}")
    return m.group(1)


def _detect_format(header_text: str, col_header: str | None) -> str:
    """kasi if the 8-threshold set is present, else noaa. Fail loud if unclear."""
    hay = (col_header or "") + "\n" + header_text
    has_500 = "P>500" in hay or ">500" in hay
    has_kasi = "KASI" in header_text or "Korea Astronomy" in header_text
    if has_500 or has_kasi:
        return "kasi"
    # NOAA legacy carries the Modified Julian Day index columns.
    if "Julian" in header_text or "P>100" in hay:
        return "noaa"
    raise ValueError("cannot determine particle file format from header")


def parse_header(text: str, filename: str) -> FileMeta:
    """Parse the comment/header block. `text` is the full file content."""
    series = series_from_name(filename)
    prepared_by = None
    source_sat = None
    location = None
    missing = MISSING_DEFAULT
    col_header = None
    e_lo_label = None  # "0p6" or "0p8"
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
        if "E>0.6" in s and s.lstrip().startswith("# Label"):
            e_lo_label = "0p6"
        if "E>0.8" in s and s.lstrip().startswith("# Label"):
            e_lo_label = "0p8"
        # The data-column header line begins with "# YR ... HHMM".
        if col_header is None and s.startswith("#") and "YR" in s and "HHMM" in s:
            col_header = s
        if not s.startswith(("#", ":")) and s.strip():
            break  # reached data; header fully scanned

    fmt = _detect_format(text, col_header)
    if fmt == "kasi":
        channel_keys = list(_LAYOUT_KASI)
    else:
        if e_lo_label == "0p8":
            channel_keys = list(_LAYOUT_NOAA_08)
        else:
            # Default low-band to E>0.6 when unlabeled (pre-2010 GOES 8-12).
            channel_keys = list(_LAYOUT_NOAA_06)
    date_label = "UTC" if "UTC" in (col_header or "") else "UT"
    return FileMeta(fmt=fmt, series=series, source_sat=source_sat,
                    location=location, prepared_by=prepared_by,
                    missing_value=missing, channel_keys=channel_keys,
                    date_label=date_label)


def _val(token: str, missing: float) -> float:
    try:
        v = float(token)
    except ValueError:
        return float("nan")
    if abs(v - missing) <= _MISSING_ATOL:
        return float("nan")
    # Exactly 0.0 is non-physical for integral particle flux; the KASI-era files
    # write 0.0 as a no-data sentinel (alongside the declared -1e5 marker, e.g.
    # a record whose other channels are -1e5 but p>500 is 0.0). Mask it so it
    # does not corrupt the log-scale baseline (spec: must mask fills).
    if v == 0.0:
        return float("nan")
    return v


def parse_file(path: str, *, text: str | None = None) -> ParsedFile:
    """Parse one daily particle file into metadata + 5-minute rows.

    `text` may be supplied to avoid a second read (e.g. when the caller already
    read the bytes to compute a checksum).
    """
    import os
    filename = os.path.basename(path)
    if text is None:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    meta = parse_header(text, filename)
    n_lead = 3 if meta.fmt == "kasi" else 4  # YR MO DA HHMM(+...) leading cols
    # kasi: YR MO DA HHMM SatNo -> 5 leading tokens then channels
    # noaa: YR MO DA HHMM MJD SOD -> 6 leading tokens then channels
    n_index = 5 if meta.fmt == "kasi" else 6
    n_expected = n_index + len(meta.channel_keys)

    rows: list[dict] = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith(("#", ":")):
            continue
        parts = s.split()
        if len(parts) < n_expected:
            # tolerate trailing-truncated lines only if time + all channels read
            if len(parts) < n_index + 1:
                continue
        try:
            yr, mo, da, hhmm = parts[0], parts[1], parts[2], parts[3]
            hh, mm = int(hhmm[:-2]), int(hhmm[-2:]) if len(hhmm) >= 3 else (0, 0)
            hh = int(hhmm.zfill(4)[:2])
            mm = int(hhmm.zfill(4)[2:])
            ts = datetime(int(yr), int(mo), int(da), hh, mm, tzinfo=timezone.utc)
        except (ValueError, IndexError):
            continue
        chan_tokens = parts[n_index:n_index + len(meta.channel_keys)]
        row: dict = {"time_utc": ts}
        for key, tok in zip(meta.channel_keys, chan_tokens):
            row[key] = _val(tok, meta.missing_value)
        rows.append(row)
    return ParsedFile(meta=meta, rows=rows)
