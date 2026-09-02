"""Core model + validation for the GOES designation table (spec Phase 1)."""
from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

COLUMNS = [
    "start_utc", "end_utc", "instrument", "primary_sat",
    "secondary_sat", "source", "source_url", "retrieved_utc",
]
VALID_SATS = {str(n) for n in range(8, 20)}  # GOES 8..19
OPEN = "open"


@dataclass(frozen=True)
class DesigRow:
    start_utc: str
    end_utc: str
    instrument: str
    primary_sat: str
    secondary_sat: str
    source: str
    source_url: str
    retrieved_utc: str


def parse_utc(s: str) -> datetime:
    """Parse an ISO8601 UTC timestamp ending in 'Z'. Raises ValueError if bad."""
    if not isinstance(s, str) or not s:
        raise ValueError(f"empty/non-str timestamp: {s!r}")
    if not s.endswith("Z"):
        raise ValueError(f"timestamp must end in 'Z' (UTC): {s!r}")
    iso = s.replace("Z", "+00:00")
    dt = datetime.fromisoformat(iso)  # py3.11+ also parses 'Z', but be explicit
    if dt.tzinfo is None:
        raise ValueError(f"timestamp missing timezone: {s!r}")
    return dt.astimezone(timezone.utc)


def _check_sat(value: str, *, allow_none: bool) -> None:
    ok = value in VALID_SATS or value == "unknown" or (allow_none and value == "none")
    if not ok:
        raise ValueError(f"invalid satellite value: {value!r}")


def validate_rows(rows: list[DesigRow]) -> None:
    """Raise ValueError unless rows are a sorted, contiguous, valid proton table."""
    if not rows:
        raise ValueError("no rows")
    for i, r in enumerate(rows):
        if r.instrument != "proton":
            raise ValueError(f"row {i}: instrument must be 'proton', got {r.instrument!r}")
        _check_sat(r.primary_sat, allow_none=False)
        _check_sat(r.secondary_sat, allow_none=True)
        parse_utc(r.start_utc)
        is_last = i == len(rows) - 1
        if r.end_utc == OPEN:
            if not is_last:
                raise ValueError(f"row {i}: 'open' end only allowed on the last row")
        else:
            end = parse_utc(r.end_utc)
            if end <= parse_utc(r.start_utc):
                raise ValueError(f"row {i}: end_utc <= start_utc")
    # contiguity / ordering
    for i in range(len(rows) - 1):
        cur_end = rows[i].end_utc
        nxt_start = rows[i + 1].start_utc
        if cur_end == OPEN:
            raise ValueError(f"row {i}: 'open' end before the last row")
        if parse_utc(cur_end) != parse_utc(nxt_start):
            raise ValueError(
                f"rows {i}->{i+1} not contiguous: {cur_end} != {nxt_start}")


def write_csv(rows: list[DesigRow], path: str) -> None:
    """Write rows to a CSV at `path` using COLUMNS order."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def _unknown_row(start_utc: str, end_utc: str) -> DesigRow:
    return DesigRow(start_utc, end_utc, "proton", "unknown", "unknown",
                    "unresolved", "", "")


def assemble_rows(global_start: str, current: DesigRow,
                  cited: list[DesigRow]) -> list[DesigRow]:
    """Combine cited historical intervals + unknown fillers + the current row.

    Result spans [global_start, 'open') with no gaps/overlaps. Cited intervals
    must lie inside [global_start, current.start_utc) and not overlap.
    """
    g0 = parse_utc(global_start)
    cstart = parse_utc(current.start_utc)
    if cstart <= g0:
        raise ValueError("current.start_utc must be after global_start")
    cited_sorted = sorted(cited, key=lambda r: parse_utc(r.start_utc))
    for r in cited_sorted:
        s, e = parse_utc(r.start_utc), parse_utc(r.end_utc)
        if s < g0 or e > cstart:
            raise ValueError(f"cited interval {r.start_utc}..{r.end_utc} out of bounds")
        if e <= s:
            raise ValueError(f"cited interval end<=start: {r.start_utc}..{r.end_utc}")
    # walk the timeline, inserting unknown fillers between cited intervals
    out: list[DesigRow] = []
    cursor = global_start
    for r in cited_sorted:
        if parse_utc(r.start_utc) > parse_utc(cursor):
            out.append(_unknown_row(cursor, r.start_utc))
        elif parse_utc(r.start_utc) < parse_utc(cursor):
            raise ValueError(f"overlapping cited interval at {r.start_utc}")
        out.append(r)
        cursor = r.end_utc
    if parse_utc(cursor) < cstart:
        out.append(_unknown_row(cursor, current.start_utc))
    out.append(current)
    validate_rows(out)
    return out
