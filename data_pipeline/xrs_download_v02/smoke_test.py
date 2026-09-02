#!/usr/bin/env python3
"""
Acceptance smoke test for the XRS v02 download (spec §8).

For one netCDF per satellite (latest year present), verify:
  * the file opens and its variables can be listed;
  * the two primary XRS channels are present (matched precisely as
    ``xrs[ab]_flux`` -- read from the file, not hard-coded, spec §3.4);
  * the CF time axis decodes to UTC using the file's own ``time:units``
    (spec #7: never assume the epoch), AND the *full* axis is strictly
    monotonic, at the native cadence, with gaps reported (spec #4);
  * masking is applied for ``_FillValue`` AND, when present,
    ``missing_value`` / ``valid_range`` / ``valid_min`` / ``valid_max``
    (spec #8) -- the report states which attributes each channel actually
    carries, so masking coverage is auditable rather than assumed.

The GOES products neglect leap seconds (documented in ``time:long_name`` /
``time:comments``); that note is captured into the report so downstream UTC
alignment across the legacy/GOES-R boundary does not assume leap-second-exact
time.

Requires ``netCDF4``. Run with the interpreter that has it, e.g.
    python3.12 smoke_test.py
Writes ``<root>/manifest/smoke_test_report.md``.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import netCDF4  # type: ignore
import numpy as np  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = PROJECT_ROOT / "data" / "goes_data"

RE_PRIMARY_CHANNEL = re.compile(r"^xrs([ab])_flux$", re.I)


def primary_channels(ds: netCDF4.Dataset) -> list[str]:
    return [v for v in ds.variables if RE_PRIMARY_CHANNEL.match(v)]


def _coerce(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def channel_mask_report(var) -> dict:
    """Explicitly apply _FillValue/missing_value/valid_range/valid_min/max and
    report which attributes were present and the resulting masked fraction."""
    present = [a for a in ("_FillValue", "missing_value", "valid_range",
                           "valid_min", "valid_max")
               if a in var.ncattrs()]
    var.set_auto_mask(False)
    raw = np.asarray(var[:], dtype="float64")
    mask = np.zeros(raw.shape, dtype=bool)
    fv = getattr(var, "_FillValue", None)
    if fv is not None:
        mask |= np.isclose(raw, float(fv))
    mv = getattr(var, "missing_value", None)
    if mv is not None:
        mask |= np.isclose(raw, float(mv))
    vr = getattr(var, "valid_range", None)
    if vr is not None and len(vr) == 2:
        mask |= (raw < float(vr[0])) | (raw > float(vr[1]))
    vmin = getattr(var, "valid_min", None)
    if vmin is not None:
        mask |= raw < float(vmin)
    vmax = getattr(var, "valid_max", None)
    if vmax is not None:
        mask |= raw > float(vmax)
    n = int(raw.size)
    return {
        "units": _coerce(getattr(var, "units", None)),
        "attrs_present": present,
        "_FillValue": _coerce(fv),
        "masked_fraction": round(float(mask.sum()) / n, 4) if n else 0.0,
        "n": n,
    }


def check_file(path: Path) -> dict:
    rep: dict = {"path": str(path), "ok": True, "problems": []}
    ds = netCDF4.Dataset(str(path), "r")
    try:
        rep["n_variables"] = len(ds.variables)
        # ---- time axis (full-array checks) ------------------------------- #
        if "time" not in ds.variables:
            rep["ok"] = False
            rep["problems"].append("no 'time' variable")
        else:
            tvar = ds.variables["time"]
            tunits = getattr(tvar, "units", None)
            tcal = getattr(tvar, "calendar", "standard")
            rep["time_units_raw"] = tunits
            rep["time_long_name"] = _coerce(getattr(tvar, "long_name", None))
            rep["time_comments"] = _coerce(getattr(tvar, "comments", None))
            if not tunits:
                rep["ok"] = False
                rep["problems"].append("time has no 'units' attribute")
            else:
                tvar.set_auto_mask(False)
                tvals = np.asarray(tvar[:], dtype="float64")
                rep["time_n"] = int(tvals.size)
                dts = netCDF4.num2date([tvals[0], tvals[-1]], units=tunits,
                                       calendar=tcal)
                rep["time_start_utc"] = str(dts[0])
                rep["time_end_utc"] = str(dts[-1])
                if not (1990 <= dts[0].year <= 2030):
                    rep["ok"] = False
                    rep["problems"].append(
                        f"decoded start year {dts[0].year} implausible "
                        f"(epoch-assumption bug?)")
                diffs = np.diff(tvals)
                if diffs.size:
                    if not np.all(diffs > 0):
                        rep["ok"] = False
                        ndup = int(np.sum(diffs == 0))
                        nrev = int(np.sum(diffs < 0))
                        rep["problems"].append(
                            f"time not strictly increasing "
                            f"(dups={ndup}, reversals={nrev})")
                    modal = float(np.median(diffs))
                    rep["cadence_seconds_modal"] = modal
                    gaps = diffs[diffs > 1.5 * modal]
                    rep["n_gaps"] = int(gaps.size)
                    rep["max_gap_seconds"] = float(diffs.max())
                    span = tvals[-1] - tvals[0]
                    expected_n = int(round(span / modal)) + 1 if modal else None
                    rep["expected_n_at_modal_cadence"] = expected_n
                    if expected_n:
                        rep["coverage_fraction"] = round(rep["time_n"]
                                                         / expected_n, 4)
        # ---- channels + masking ------------------------------------------ #
        chans = primary_channels(ds)
        rep["channel_vars"] = chans
        if len(chans) < 2:
            rep["ok"] = False
            rep["problems"].append(
                f"expected 2 primary channels (xrsa_flux/xrsb_flux), got {chans}")
        rep["channels"] = {c: channel_mask_report(ds.variables[c]) for c in chans}
    finally:
        ds.close()
    return rep


def latest_file_per_sat(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    base = root / "raw" / "xrs"
    if not base.exists():
        return out
    for satdir in sorted(base.glob("goes*")):
        if satdir.is_dir():
            ncs = sorted(satdir.glob("sci_xrsf-l2-avg1m_*_y*.nc"))
            if ncs:
                out[satdir.name] = ncs[-1]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="XRS v02 acceptance smoke test")
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--file", type=Path, default=None,
                    help="check one explicit file instead of one per satellite")
    args = ap.parse_args(argv)
    root = args.output_root.resolve()
    targets = ({"(explicit)": args.file} if args.file
               else latest_file_per_sat(root))
    if not targets:
        print("no files found to smoke-test", file=sys.stderr)
        return 2

    lines = ["# XRS v02 smoke test (spec §8)", ""]
    all_ok = True
    for sat, path in targets.items():
        print(f"== {sat}: {path.name}")
        try:
            rep = check_file(path)
        except Exception as exc:
            all_ok = False
            print(f"   ERROR: {exc!r}")
            lines += [f"## {sat} — ERROR opening `{path.name}`", "",
                      f"`{exc!r}`", ""]
            continue
        ok = rep["ok"]
        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        print(f"   {status}  time={rep.get('time_start_utc')}.."
              f"{rep.get('time_end_utc')} n={rep.get('time_n')} "
              f"cadence={rep.get('cadence_seconds_modal')}s "
              f"gaps={rep.get('n_gaps')} chans={rep.get('channel_vars')}")
        for c, ci in rep.get("channels", {}).items():
            print(f"     {c}: units={ci['units']} masked={ci['masked_fraction']:.2%} "
                  f"attrs={ci['attrs_present']}")
        if rep["problems"]:
            print(f"   problems: {rep['problems']}")

        lines += [f"## {sat} — {status}", "",
                  f"- file: `{path.name}`",
                  f"- time_units_raw: `{rep.get('time_units_raw')}`",
                  f"- time (UTC): {rep.get('time_start_utc')} .. "
                  f"{rep.get('time_end_utc')}  (n={rep.get('time_n')})",
                  f"- modal cadence: {rep.get('cadence_seconds_modal')} s; "
                  f"gaps>1.5×cadence: {rep.get('n_gaps')}; "
                  f"max gap: {rep.get('max_gap_seconds')} s; "
                  f"coverage vs modal-cadence grid: {rep.get('coverage_fraction')}",
                  f"- leap-second note: {rep.get('time_long_name')}"]
        for c, ci in rep.get("channels", {}).items():
            lines.append(f"  - `{c}`: units={ci['units']} "
                         f"_FillValue={ci['_FillValue']} "
                         f"attrs_present={ci['attrs_present']} "
                         f"masked={ci['masked_fraction']:.2%} n={ci['n']}")
        if rep["problems"]:
            lines.append(f"- **problems:** {rep['problems']}")
        lines.append("")

    report = root / "manifest" / "smoke_test_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    header = "**RESULT: ALL PASS**" if all_ok else "**RESULT: SOME FAIL**"
    report.write_text("\n".join(lines[:2] + [header, ""] + lines[2:]),
                      encoding="utf-8")
    print(f"\nreport -> {report}  ({'ALL PASS' if all_ok else 'SOME FAIL'})")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
