import os
import unittest
from datetime import datetime, timezone

from designation_table import (
    COLUMNS, VALID_SATS, DesigRow, parse_utc, validate_rows, assemble_rows,
)
from designation_sources import (
    CURRENT_SOURCE_URL, GLOBAL_COVERAGE_START, read_current_designation,
    historical_cited_intervals,
)


def row(start, end, primary="unknown", secondary="unknown",
        source="unresolved", url="", retrieved=""):
    return DesigRow(start, end, "proton", primary, secondary, source, url, retrieved)


class TestCoreModel(unittest.TestCase):
    def test_columns_exact_order(self):
        self.assertEqual(COLUMNS, [
            "start_utc", "end_utc", "instrument", "primary_sat",
            "secondary_sat", "source", "source_url", "retrieved_utc",
        ])

    def test_parse_utc_handles_z(self):
        dt = parse_utc("2026-06-03T16:23:07Z")
        self.assertEqual(dt, datetime(2026, 6, 3, 16, 23, 7, tzinfo=timezone.utc))

    def test_parse_utc_rejects_garbage(self):
        with self.assertRaises(ValueError):
            parse_utc("not-a-date")

    def test_validate_accepts_contiguous(self):
        rows = [
            row("1986-01-01T00:00:00Z", "2026-06-03T16:23:07Z"),
            row("2026-06-03T16:23:07Z", "open", "18", "19", "swpc-instrument-sources"),
        ]
        self.assertIsNone(validate_rows(rows))

    def test_validate_rejects_gap(self):
        rows = [
            row("1986-01-01T00:00:00Z", "2000-01-01T00:00:00Z"),
            row("2001-01-01T00:00:00Z", "open", "18", "19"),
        ]
        with self.assertRaises(ValueError):
            validate_rows(rows)

    def test_validate_rejects_overlap(self):
        rows = [
            row("1986-01-01T00:00:00Z", "2010-01-01T00:00:00Z"),
            row("2009-01-01T00:00:00Z", "open", "18", "19"),
        ]
        with self.assertRaises(ValueError):
            validate_rows(rows)

    def test_validate_rejects_bad_sat(self):
        rows = [row("1986-01-01T00:00:00Z", "open", "99", "19")]
        with self.assertRaises(ValueError):
            validate_rows(rows)

    def test_validate_open_only_on_last(self):
        rows = [
            row("1986-01-01T00:00:00Z", "open"),
            row("2026-06-03T16:23:07Z", "open", "18", "19"),
        ]
        with self.assertRaises(ValueError):
            validate_rows(rows)

    def test_validate_rejects_non_proton(self):
        bad = DesigRow("1986-01-01T00:00:00Z", "open", "xrs",
                       "18", "19", "s", "", "")
        with self.assertRaises(ValueError):
            validate_rows([bad])

    def test_parse_utc_rejects_offset_string(self):
        with self.assertRaises(ValueError):
            parse_utc("2026-01-01T00:00:00+05:00")
        with self.assertRaises(ValueError):
            parse_utc("2026-01-01T00:00:00+00:00")


SWPC_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "data", "goes_data", "raw", "proton", "swpc",
)
_HAS_SWPC = os.path.isdir(SWPC_DIR)


class TestCurrentDesignation(unittest.TestCase):
    @unittest.skipUnless(_HAS_SWPC, "real SWPC snapshot not present")
    def test_reads_proton_primary_secondary_from_real_snapshot(self):
        r = read_current_designation(SWPC_DIR)
        self.assertEqual(r.instrument, "proton")
        self.assertEqual(r.primary_sat, "18")
        self.assertEqual(r.secondary_sat, "19")
        self.assertEqual(r.end_utc, "open")
        self.assertEqual(r.source, "swpc-instrument-sources")
        self.assertEqual(r.source_url, CURRENT_SOURCE_URL)
        parse_utc(r.start_utc)      # valid timestamp
        parse_utc(r.retrieved_utc)  # valid timestamp

    def test_fail_loud_when_dir_missing(self):
        with self.assertRaises((FileNotFoundError, ValueError)):
            read_current_designation("/no/such/dir")

    def test_global_coverage_start_is_1986(self):
        self.assertEqual(GLOBAL_COVERAGE_START, "1986-01-01T00:00:00Z")

    @unittest.skipUnless(_HAS_SWPC, "real SWPC snapshot not present")
    def test_current_start_utc_is_canonical_z(self):
        r = read_current_designation(SWPC_DIR)
        self.assertTrue(r.start_utc.endswith("Z"))


class TestAssemble(unittest.TestCase):
    def _current(self):
        return row("2025-04-04T00:00:00Z", "open", "18", "19",
                   "swpc-instrument-sources", "http://x", "2026-06-23T08:35:44Z")

    def test_no_cited_yields_single_unknown_then_current(self):
        out = assemble_rows("1986-01-01T00:00:00Z", self._current(), [])
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].primary_sat, "unknown")
        self.assertEqual(out[0].secondary_sat, "unknown")
        self.assertEqual(out[0].source, "unresolved")
        self.assertEqual(out[0].start_utc, "1986-01-01T00:00:00Z")
        self.assertEqual(out[0].end_utc, "2025-04-04T00:00:00Z")
        self.assertEqual(out[-1].end_utc, "open")
        validate_rows(out)  # must be contiguous + valid

    def test_cited_interval_is_surrounded_by_unknown_fillers(self):
        cited = [row("2017-12-01T00:00:00Z", "2025-04-04T00:00:00Z",
                     "16", "15", "xrs-readme-fig1+sem-docs",
                     "http://doc", "2026-06-24T00:00:00Z")]
        out = assemble_rows("1986-01-01T00:00:00Z", self._current(), cited)
        # unknown [1986..2017-12), cited [2017-12..2025-04), current [2025-04..open)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0].primary_sat, "unknown")
        self.assertEqual(out[1].primary_sat, "16")
        self.assertEqual(out[2].primary_sat, "18")
        validate_rows(out)

    def test_rejects_cited_out_of_bounds(self):
        cited = [row("2030-01-01T00:00:00Z", "2031-01-01T00:00:00Z", "16", "15")]
        with self.assertRaises(ValueError):
            assemble_rows("1986-01-01T00:00:00Z", self._current(), cited)

    def test_rejects_overlapping_cited(self):
        cited = [
            row("2010-01-01T00:00:00Z", "2018-01-01T00:00:00Z", "15", "13"),
            row("2017-01-01T00:00:00Z", "2025-04-04T00:00:00Z", "16", "15"),
        ]
        with self.assertRaises(ValueError):
            assemble_rows("1986-01-01T00:00:00Z", self._current(), cited)


class TestCited(unittest.TestCase):
    def test_every_cited_interval_is_well_formed_and_cited(self):
        for r in historical_cited_intervals():
            self.assertEqual(r.instrument, "proton")
            self.assertIn(r.primary_sat, VALID_SATS)          # never 'unknown'
            self.assertIn(r.secondary_sat, VALID_SATS | {"none"})
            self.assertTrue(r.source.strip(), "cited row needs a source")
            self.assertTrue(r.source_url.strip(), "cited row needs a source_url")
            parse_utc(r.start_utc)
            parse_utc(r.end_utc)  # cited rows are never 'open'
            self.assertNotEqual(r.end_utc, "open")

    def test_cited_intervals_do_not_overlap(self):
        rows = sorted(historical_cited_intervals(), key=lambda r: parse_utc(r.start_utc))
        for a, b in zip(rows, rows[1:]):
            self.assertLessEqual(parse_utc(a.end_utc), parse_utc(b.start_utc))


import csv as _csv
import tempfile, shutil
from build_designation_table import build

DATA_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "goes_data")
_HAS_DATA = os.path.isdir(os.path.join(DATA_ROOT, "raw", "proton", "swpc"))


@unittest.skipUnless(_HAS_DATA, "real proton snapshot not present")
class TestBuildEndToEnd(unittest.TestCase):
    def test_build_against_real_inputs_is_valid_and_current_is_18_19(self):
        tmp = tempfile.mkdtemp()
        try:
            # Mirror swpc inputs and provenance manifest into temp root for isolation
            src_swpc = os.path.join(DATA_ROOT, "raw", "proton", "swpc")
            dst_swpc = os.path.join(tmp, "raw", "proton", "swpc")
            shutil.copytree(src_swpc, dst_swpc)
            os.makedirs(os.path.join(tmp, "manifest"))
            shutil.copy(
                os.path.join(DATA_ROOT, "manifest", "provenance.jsonl"),
                os.path.join(tmp, "manifest", "provenance.jsonl"),
            )
            rows = build(tmp)
            validate_rows(rows)
            self.assertEqual(rows[0].start_utc, GLOBAL_COVERAGE_START)
            self.assertEqual(rows[-1].end_utc, "open")
            self.assertEqual(rows[-1].primary_sat, "18")
            self.assertEqual(rows[-1].secondary_sat, "19")
        finally:
            shutil.rmtree(tmp)

    def test_build_writes_csv_with_exact_header(self):
        tmp = tempfile.mkdtemp()
        try:
            # mirror just the swpc inputs into a temp root
            src = os.path.join(DATA_ROOT, "raw", "proton", "swpc")
            dst = os.path.join(tmp, "raw", "proton", "swpc")
            shutil.copytree(src, dst)
            os.makedirs(os.path.join(tmp, "manifest"))
            build(tmp)
            out = os.path.join(tmp, "manifest", "designation_table.csv")
            self.assertTrue(os.path.exists(out))
            with open(out, encoding="utf-8") as f:
                header = next(_csv.reader(f))
            self.assertEqual(header, COLUMNS)
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
