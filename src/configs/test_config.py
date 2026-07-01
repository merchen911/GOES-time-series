import unittest
from configs.config import exp_parser, config_postprocess


class TestConfig(unittest.TestCase):
    def _parse(self, *extra):
        argv = ["--data_path", "x.parquet", "--target_col", "p_gt10", *extra]
        return exp_parser().parse_args(argv)

    def test_new_defaults(self):
        c = self._parse()
        self.assertEqual(c.role, "primary")
        self.assertEqual(c.transform, "log10")
        self.assertEqual(c.cadence_min, 5)

    def test_transform_choices(self):
        c = self._parse("--transform", "none")
        self.assertEqual(c.transform, "none")

    def test_parquet_year_half_without_time_col_ok(self):
        c = self._parse("--split_type", "year_half")  # no --time_col
        self.assertIs(config_postprocess(c), c)  # must not raise

    def test_csv_year_half_without_time_col_still_errors(self):
        argv = ["--data_path", "x.csv", "--target_col", "t", "--split_type", "year_half"]
        c = exp_parser().parse_args(argv)
        with self.assertRaises(ValueError):
            config_postprocess(c)


class TestMultivarFlags(unittest.TestCase):
    def _parse(self, *extra):
        argv = ["--data_path", "x.parquet", "--target_col", "p_gt10", *extra]
        return exp_parser().parse_args(argv)

    def test_defaults(self):
        c = self._parse()
        self.assertIsNone(c.channels)
        self.assertIsNone(c.target_cols)
        self.assertEqual(c.min_bin_count, 1)

    def test_lists_parse(self):
        c = self._parse("--channels", "a.parquet:p_gt10", "b.parquet:xrs_long",
                        "--target_cols", "p_gt10", "--min_bin_count", "3")
        self.assertEqual(c.channels, ["a.parquet:p_gt10", "b.parquet:xrs_long"])
        self.assertEqual(c.target_cols, ["p_gt10"])
        self.assertEqual(c.min_bin_count, 3)
