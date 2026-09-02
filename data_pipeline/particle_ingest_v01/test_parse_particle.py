"""Tests for parse_particle against real KASI DB files (READ-ONLY).

Run: cd particle_ingest_v01 && python3.12 -m unittest test_parse_particle -v
"""
import math
import os
import unittest

from parse_particle import parse_file, series_from_name

DB = os.environ.get("SW_GOES_ARCHIVE_PARTICLE", "")

NOAA_G9 = os.path.join(DB, "1998", "19980613_G9part_5m.txt")
NOAA_GP = os.path.join(DB, "2010", "20100601_Gp_part_5m.txt")
NOAA_GS = os.path.join(DB, "2010", "20100601_Gs_part_5m.txt")
KASI_GP = os.path.join(DB, "2024", "20240101_Gp_part_5m.txt")

_HAS_DB = bool(DB) and os.path.isdir(DB)


class TestSeriesName(unittest.TestCase):
    def test_tokens(self):
        self.assertEqual(series_from_name("19980613_G9part_5m.txt"), "G9")
        self.assertEqual(series_from_name("20100601_Gp_part_5m.txt"), "Gp")
        self.assertEqual(series_from_name("20100601_Gs_part_5m.txt"), "Gs")
        self.assertEqual(series_from_name("20030101_G12part_5m.txt"), "G12")

    def test_rejects_xray(self):
        with self.assertRaises(ValueError):
            series_from_name("20130101_Gp_xr_1m.txt")


@unittest.skipUnless(_HAS_DB, "set SW_GOES_ARCHIVE_PARTICLE to run real-data parser tests")
class TestNoaaFormat(unittest.TestCase):
    def test_g9_1998(self):
        p = parse_file(NOAA_G9)
        self.assertEqual(p.meta.fmt, "noaa")
        self.assertEqual(p.meta.series, "G9")
        self.assertEqual(p.meta.source_sat, 9)
        self.assertIn("e_gt0p6", p.meta.channel_keys)   # 1998 low band
        self.assertIn("p_gt100", p.meta.channel_keys)
        self.assertNotIn("p_gt500", p.meta.channel_keys)
        self.assertNotIn("p_gt60", p.meta.channel_keys)
        self.assertGreater(len(p.rows), 200)            # ~288/day
        r0 = p.rows[0]
        self.assertEqual(r0["time_utc"].hour, 0)
        self.assertEqual(r0["time_utc"].minute, 0)
        self.assertTrue(math.isfinite(r0["p_gt10"]))

    def test_gp_2010_low_band_is_0p8(self):
        p = parse_file(NOAA_GP)
        self.assertEqual(p.meta.fmt, "noaa")
        self.assertEqual(p.meta.series, "Gp")
        self.assertEqual(p.meta.source_sat, 13)
        self.assertIn("e_gt0p8", p.meta.channel_keys)
        self.assertNotIn("e_gt0p6", p.meta.channel_keys)

    def test_gs_2010_secondary_source(self):
        p = parse_file(NOAA_GS)
        self.assertEqual(p.meta.series, "Gs")
        self.assertEqual(p.meta.source_sat, 11)

    def test_missing_mapped_to_nan(self):
        # E>4.0 column in legacy files is frequently the missing sentinel.
        p = parse_file(NOAA_G9)
        e4 = [r["e_gt4"] for r in p.rows]
        self.assertTrue(any(math.isnan(v) for v in e4))
        # a fill value must never survive as -1e5
        self.assertFalse(any(v < -1e4 for v in e4 if math.isfinite(v)))


@unittest.skipUnless(_HAS_DB, "set SW_GOES_ARCHIVE_PARTICLE to run real-data parser tests")
class TestKasiFormat(unittest.TestCase):
    def test_gp_2024(self):
        p = parse_file(KASI_GP)
        self.assertEqual(p.meta.fmt, "kasi")
        self.assertEqual(p.meta.series, "Gp")
        self.assertEqual(p.meta.source_sat, 16)
        for k in ("p_gt1", "p_gt60", "p_gt100", "p_gt500", "e_gt2"):
            self.assertIn(k, p.meta.channel_keys)
        self.assertNotIn("e_gt4", p.meta.channel_keys)
        self.assertGreater(len(p.rows), 200)
        r0 = p.rows[0]
        self.assertTrue(math.isfinite(r0["p_gt1"]))
        self.assertTrue(math.isfinite(r0["p_gt500"]))


if __name__ == "__main__":
    unittest.main()
