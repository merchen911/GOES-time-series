"""Tests for parse_xray against real KASI DB files (READ-ONLY).

Run: cd xray_ingest_v01 && python3.12 -m unittest test_parse_xray -v
"""
import math
import os
import unittest

from parse_xray import parse_file, series_from_name

DB = os.environ.get("SW_GOES_ARCHIVE_XRAY", "")
NOAA_GP = os.path.join(DB, "2010", "20100601_Gp_xr_1m.txt")
NOAA_G12 = os.path.join(DB, "2004", "20040101_G12xr_1m.txt")
KASI_GP = os.path.join(DB, "2024", "20240101_Gp_xr_1m.txt")

_HAS_DB = bool(DB) and os.path.isdir(DB)


class TestSeriesName(unittest.TestCase):
    def test_tokens(self):
        self.assertEqual(series_from_name("20100601_Gp_xr_1m.txt"), "Gp")
        self.assertEqual(series_from_name("20150601_Gs_xr_1m.txt"), "Gs")
        self.assertEqual(series_from_name("20040101_G12xr_1m.txt"), "G12")

    def test_rejects_particle_and_5m(self):
        for bad in ("20150101_Gp_part_5m.txt", "20200101_Gp_xr_5m.txt"):
            with self.assertRaises(ValueError):
                series_from_name(bad)


@unittest.skipUnless(_HAS_DB, "set SW_GOES_ARCHIVE_XRAY to run real-data parser tests")
class TestNoaaFormat(unittest.TestCase):
    def test_gp_2010(self):
        p = parse_file(NOAA_GP)
        self.assertEqual(p.meta.fmt, "noaa")
        self.assertEqual(p.meta.series, "Gp")
        self.assertEqual(p.meta.source_sat, 14)
        self.assertEqual(p.meta.channel_keys, ["xrs_short", "xrs_long"])
        self.assertGreater(len(p.rows), 1000)          # ~1440/day at 1-min
        r0 = p.rows[0]
        self.assertTrue(math.isfinite(r0["xrs_long"]))
        self.assertLess(r0["xrs_long"], 1e-3)          # plausible W/m^2 magnitude

    def test_g12_2004_persat(self):
        p = parse_file(NOAA_G12)
        self.assertEqual(p.meta.fmt, "noaa")
        self.assertEqual(p.meta.series, "G12")
        self.assertEqual(p.meta.source_sat, 12)

    def test_missing_mapped_to_nan(self):
        p = parse_file(NOAA_GP)
        vals = [r["xrs_short"] for r in p.rows]
        self.assertFalse(any(v < -1e4 for v in vals if math.isfinite(v)))


@unittest.skipUnless(_HAS_DB, "set SW_GOES_ARCHIVE_XRAY to run real-data parser tests")
class TestKasiFormat(unittest.TestCase):
    def test_gp_2024(self):
        p = parse_file(KASI_GP)
        self.assertEqual(p.meta.fmt, "kasi")
        self.assertEqual(p.meta.source_sat, 16)
        self.assertGreater(len(p.rows), 1000)
        r0 = p.rows[0]
        self.assertTrue(math.isfinite(r0["xrs_short"]))
        self.assertTrue(math.isfinite(r0["xrs_long"]))


if __name__ == "__main__":
    unittest.main()
