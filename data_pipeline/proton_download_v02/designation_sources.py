"""Designation sources: machine snapshot (SWPC) + curated historical intervals."""
from __future__ import annotations

import glob
import json
import os
from datetime import datetime, timezone

from designation_table import DesigRow, VALID_SATS

CURRENT_SOURCE_URL = "https://services.swpc.noaa.gov/json/goes/instrument-sources.json"
GLOBAL_COVERAGE_START = "1986-01-01T00:00:00Z"


def _mtime_utc(path: str) -> str:
    ts = os.path.getmtime(path)
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _provenance_retrieved_utc(manifest_path: str, target_basename: str) -> str | None:
    """Look up retrieved_utc from provenance manifest by local_path basename.

    Args:
        manifest_path: Path to provenance.jsonl file.
        target_basename: Basename of the file to find (e.g., 'instrument-sources_20260623.json').

    Returns:
        The retrieved_utc string (ISO format with Z) if found, else None.
    """
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            lp = rec.get("local_path", "")
            if os.path.basename(lp) == target_basename and rec.get("retrieved_utc"):
                return rec["retrieved_utc"]
    return None


def read_current_designation(swpc_dir: str) -> DesigRow:
    """Build the current open proton interval from the latest instrument-sources JSON.

    Fail loud (raise) if the directory/file is missing or the schema is not as
    expected — never guess a designation.

    Retrieval time is sourced from the provenance manifest (retrieved_utc field),
    which records the real retrieval time, not the file mtime (which may drift if
    the file is re-copied). Falls back to mtime if provenance is not available.
    """
    pattern = os.path.join(swpc_dir, "instrument-sources_*.json")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no instrument-sources_*.json under {swpc_dir!r}")
    path = matches[-1]  # latest by filename date
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"unexpected instrument-sources structure in {path!r}")
    obj = data[0]
    try:
        protons = obj["protons"]
        primary = str(protons["primary"])
        secondary = str(protons["secondary"])
        time_tag = obj["time_tag"]
    except (KeyError, TypeError) as e:
        raise ValueError(f"missing protons/time_tag keys in {path!r}: {e}")
    iso = time_tag.replace("Z", "+00:00")
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        raise ValueError(f"instrument-sources time_tag missing timezone: {time_tag!r}")
    start_utc = dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Look up retrieved_utc from provenance manifest (authoritative); fall back to mtime.
    manifest_path = os.path.normpath(
        os.path.join(swpc_dir, "..", "..", "..", "manifest", "provenance.jsonl")
    )
    retrieved_utc = _provenance_retrieved_utc(manifest_path, os.path.basename(path))
    if not retrieved_utc:
        retrieved_utc = _mtime_utc(path)

    return DesigRow(
        start_utc=start_utc,
        end_utc="open",
        instrument="proton",
        primary_sat=primary,
        secondary_sat=secondary,
        source="swpc-instrument-sources",
        source_url=CURRENT_SOURCE_URL,
        retrieved_utc=retrieved_utc,
    )


# --- Curated historical intervals (Task 4) -------------------------------------
#
# Source document: GOES X-ray Sensor (XRS) Operational Data readme, v1.5
# (17 June 2022), Janet Machol et al., NOAA/NCEI/SWPC. Section 4, "Table 2.
# Chronology of designation of primary and secondary satellites for XRS
# measurements since 1986. Designations are unknown for period 1974-1986.
# (Table revised 7 June 2016)". Figure 1 corroborates the primary/secondary
# spacecraft chronology for 1975-2021.
#
# IMPORTANT — proton vs XRS:
#   This table records the *spacecraft-level* primary/secondary designation as
#   published for XRS. The readme contains NO proton-specific designation. In the
#   pre-GOES-R era (GOES-8..-15 SEM/EPEAD), the primary/secondary spacecraft was
#   designated at the satellite level and applied to all SEM instruments,
#   including proton. We therefore apply the XRS spacecraft-level primary to
#   proton as the documented pre-GOES-R-era assumption, and record that
#   assumption explicitly in each row's `source` string.
#
# COVERAGE BOUNDS:
#   * Lower: the table schema (VALID_SATS) only represents GOES-8..-19, matching
#     the proton-data archive (GOES-8 I-M series onward; EPS/EPEAD). Table 2's
#     1986-1995 era is GOES-5/6/7, which is NOT representable here, so those
#     intervals are omitted and become `unknown`. The first representable,
#     fully-valid interval is 1998-07-27 (primary 8 / secondary 10). (The
#     1995-03-01 8/7 entry is also omitted because secondary GOES-7 is not a
#     representable sat value.)
#   * Upper: we cite only through 2016-06-09. Table 2's last fully specified
#     entry is 2016-05-16 17:00 UTC (14/15); the 2016-06-09 (15/13) entry is
#     marked provisional ("will be added after June 9") with no exact time, so we
#     end cited coverage at 2016-06-09T00:00:00Z. The span from there to the
#     current SWPC machine snapshot is left UNKNOWN: the readme is XRS-only and
#     stops in 2016, and the GOES-R-era *proton* primary/secondary designation is
#     NOT determinable from operational GOES-East/West status (e.g. GOES-18 is
#     GOES-West yet is the current proton primary), so no proton-specific
#     citation exists for that span. Per the spec, unverifiable spans are omitted
#     and become `unknown` fillers.
#
# Each tuple is (start_utc, primary, secondary) from Table 2, ascending. The end
# of each interval is the start of the next tuple (half-open). Consecutive
# entries with identical primary+secondary are merged into a single interval.
_XRS_README_URL = "https://www.ngdc.noaa.gov/stp/satellite/goes/doc/GOES_XRS_readme.pdf"
_XRS_README_SOURCE = (
    "GOES XRS readme v1.5 (2022-06-17) Table 2 (revised 2016-06-07); "
    "spacecraft-level XRS primary/secondary applied to proton "
    "(documented pre-GOES-R-era assumption)"
)

# (start_utc, primary_sat, secondary_sat) — verbatim from Table 2.
_TABLE2 = [
    ("1986-01-01T00:00:00Z", "6", "5"),
    ("1988-01-26T00:00:00Z", "7", "6"),
    ("1994-12-11T00:00:00Z", "7", "8"),
    ("1995-03-01T00:00:00Z", "8", "7"),
    ("1998-07-27T00:00:00Z", "8", "10"),
    ("2003-04-08T15:00:00Z", "10", "12"),
    ("2003-05-15T15:00:00Z", "12", "10"),
    ("2006-06-28T00:00:00Z", "12", "11"),   # 00:00 and 14:00 entries are identical
    ("2007-01-01T00:00:00Z", "10", "11"),
    ("2007-04-12T00:00:00Z", "11", "10"),
    ("2007-11-21T00:00:00Z", "11", "none"),
    ("2007-12-05T00:00:00Z", "11", "10"),
    ("2007-12-18T00:00:00Z", "11", "none"),
    ("2008-02-10T16:30:00Z", "10", "none"),
    ("2009-12-01T00:00:00Z", "14", "none"),
    ("2010-09-01T00:00:00Z", "14", "15"),
    ("2010-10-28T00:00:00Z", "15", "none"),
    ("2011-09-01T00:00:00Z", "15", "14"),   # secondary listed as "GOES-14"
    ("2012-10-23T16:00:00Z", "14", "15"),
    ("2012-11-19T16:31:00Z", "15", "none"),
    ("2015-01-26T16:01:00Z", "15", "13"),
    ("2015-05-21T18:00:00Z", "14", "13"),
    ("2015-06-09T16:25:00Z", "15", "13"),
    ("2016-05-03T13:00:00Z", "13", "14"),
    ("2016-05-12T17:30:00Z", "14", "13"),
    ("2016-05-16T17:00:00Z", "14", "15"),
]
# End of cited coverage. Table 2's 2016-06-09 entry (15/13) is provisional with
# no exact time; everything after this is left to the `unknown` filler.
_TABLE2_END = "2016-06-09T00:00:00Z"


def historical_cited_intervals() -> list[DesigRow]:
    """Verified proton primary/secondary intervals in [1986, current). Cited per row.

    Encodes the GOES XRS readme Table 2 spacecraft-level primary/secondary
    chronology (1986-01-01 .. 2016-06-09), applied to proton as the documented
    pre-GOES-R-era assumption (see module notes above). The GOES-R-era proton
    designation (2016-06-09 onward, up to the current machine snapshot) is NOT
    citable proton-side and is intentionally omitted, so `assemble_rows` fills it
    with an `unknown` row. No designation is invented.
    """
    retrieved = "2026-06-24T00:00:00Z"
    rows: list[DesigRow] = []
    for i, (start, primary, secondary) in enumerate(_TABLE2):
        end = _TABLE2[i + 1][0] if i + 1 < len(_TABLE2) else _TABLE2_END
        # Drop intervals whose primary/secondary are not representable GOES sats
        # (e.g. GOES-5/6/7 in 1986-1995). These become `unknown` via assemble_rows.
        if primary not in VALID_SATS:
            continue
        if secondary != "none" and secondary not in VALID_SATS:
            continue
        # Merge consecutive *contiguous* duplicates (same primary+secondary).
        if rows and rows[-1].primary_sat == primary and \
                rows[-1].secondary_sat == secondary and rows[-1].end_utc == start:
            prev = rows[-1]
            rows[-1] = DesigRow(prev.start_utc, end, "proton", primary, secondary,
                                _XRS_README_SOURCE, _XRS_README_URL, retrieved)
            continue
        rows.append(DesigRow(
            start_utc=start,
            end_utc=end,
            instrument="proton",
            primary_sat=primary,
            secondary_sat=secondary,
            source=_XRS_README_SOURCE,
            source_url=_XRS_README_URL,
            retrieved_utc=retrieved,
        ))
    return rows
