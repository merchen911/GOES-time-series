import contextlib
import importlib.util
import io
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tslib.benchmark import verify_archive as va
from tslib.benchmark.driver import parse_cell_name
from tslib.exp.metrics import EVENT_SCORES
from tslib.exp.predictions import save_predictions

_REPO = Path(__file__).resolve().parents[2]

# target channels, physical thresholds and their log10 (stored-space) values
_TRACKS = {
    "uni_a": (["p_gt10"], [10.0], [1.0]),
    "uni_b": (["xrs_long"], [1e-5], [-5.0]),
    "multi": (["p_gt10", "xrs_long"], [10.0, 1e-5], [1.0, -5.0]),
}
_V1_COLUMNS = ["track", "seq_len", "pred_len", "fold", "strategy", "model",
               "best_val_loss", "skipped", "est_train_hours", "rmse", "mae",
               "tss_p_gt10", "hss_p_gt10", "pod_p_gt10", "far_p_gt10",
               "tss_xrs_long", "hss_xrs_long", "pod_xrs_long", "far_xrs_long"]
_PAPER_COLUMNS = ["track", "model", "gran", "TSS", "HSS", "CSI", "MCC", "SEDI",
                  "precision", "FB", "POD", "POFD", "FAR", "base", "n_events", "n",
                  "TP", "FP", "FN", "TN", "fold"]


# ---- the v1 metric code (v003 tslib/exp/metrics.py), kept as the reference ----

def _v1_tss(tp, fp, fn, tn):
    pod = tp / (tp + fn) if (tp + fn) else 0.0
    pofd = fp / (fp + tn) if (fp + tn) else 0.0
    return pod - pofd


def _v1_hss(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    if n == 0:
        return 0.0
    exp = ((tp + fn) * (tp + fp) + (tn + fn) * (tn + fp)) / n
    denom = n - exp
    return (tp + tn - exp) / denom if denom else 0.0


def _v1_pod(tp, fp, fn, tn):
    return tp / (tp + fn) if (tp + fn) else 0.0


def _v1_far(tp, fp, fn, tn):
    return fp / (tp + fp) if (tp + fp) else 0.0


def _v1_reference(pred, true, thresholds_stored, target_cols):
    """What a v1 run wrote to results_master.csv for these arrays."""
    out = {"rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
           "mae": float(np.mean(np.abs(pred - true)))}
    for i, ch in enumerate(target_cols):
        p, t = pred[..., i] >= thresholds_stored[i], true[..., i] >= thresholds_stored[i]
        c = (int(np.sum(p & t)), int(np.sum(p & ~t)), int(np.sum(~p & t)),
             int(np.sum(~p & ~t)))
        for name, fn in [("tss", _v1_tss), ("hss", _v1_hss), ("pod", _v1_pod),
                         ("far", _v1_far)]:
            out[f"{name}_{ch}"] = float(fn(*c))
    return out


# ---- synthetic archive trees and v1 tables ----

def _arrays(track, seed, n=60, p=8):
    """float32 (pred, true) around the track's stored thresholds: some events,
    some false alarms, some misses."""
    centres = np.asarray(_TRACKS[track][2], dtype=np.float64)
    rng = np.random.default_rng(seed)
    true = centres + rng.normal(-0.4, 0.6, (n, p, centres.size))
    pred = true + rng.normal(0.0, 0.5, true.shape)
    return pred.astype(np.float32), true.astype(np.float32)


def _write_run(bench, cell_name, model, pred, true):
    cell = parse_cell_name(cell_name)
    cols, thresholds, _ = _TRACKS[cell["track"]]
    meta = {"model": model, "strategy": cell["strategy"], "run_name": f"bench/{cell_name}",
            "seq_len": cell["seq_len"], "pred_len": cell["pred_len"],
            "fold_numb": cell["fold"], "transform": "log10", "cadence_min": 5,
            "event_threshold": thresholds, "target_cols": cols}
    starts = np.arange(pred.shape[0], dtype=np.int64) * 300 * 10**9
    save_predictions(str(Path(bench) / cell_name / "pred" / f"{model}.npz"),
                     pred, true, starts, meta)


def _v1_row(cell_name, model, pred, true):
    cell = parse_cell_name(cell_name)
    cols, _, stored = _TRACKS[cell["track"]]
    return {**cell, "model": model, "best_val_loss": 0.1, "skipped": False,
            "est_train_hours": float("nan"), **_v1_reference(pred, true, stored, cols)}


def _write_v1(path, rows):
    pd.DataFrame(rows, columns=_V1_COLUMNS).to_csv(path, index=False)
    return path


def _paper_csv(path, rows):
    pd.DataFrame(rows, columns=_PAPER_COLUMNS).to_csv(path, index=False)


def _paper_row(track, model, gran, fold, counts, **scores):
    tp, fp, fn, tn = counts
    return {"track": track, "model": model, "gran": gran, "fold": fold,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn, **scores}


def _counts(p, t):
    return (int(np.sum(p & t)), int(np.sum(p & ~t)), int(np.sum(~p & t)),
            int(np.sum(~p & ~t)))


class TestLegacyScores(unittest.TestCase):
    def test_legacy_zero_convention(self):
        # one channel, threshold 0 (stored); the forecast never raises an alarm
        true = np.array([[[1.0], [-1.0], [2.0], [-2.0]]], dtype=np.float32)
        pred = np.array([[[-3.0], [-3.0], [np.nan], [np.nan]]], dtype=np.float32)
        scores, counts = va.legacy_v1_scores(pred, true, [0.0], ["xrs_long"])
        # the NaN predictions are "no event": a miss and a correct negative
        self.assertEqual(counts, {"xrs_long": (0, 0, 2, 2)})
        self.assertEqual(scores["far_xrs_long"], 0.0)   # no alarms: 0.0, not nan
        self.assertEqual(scores["pod_xrs_long"], 0.0)
        self.assertEqual(scores["tss_xrs_long"], 0.0)
        self.assertEqual(scores["hss_xrs_long"], 0.0)
        self.assertTrue(math.isnan(scores["rmse"]))     # the NaN reaches rmse/mae
        self.assertTrue(math.isnan(scores["mae"]))

    def test_no_events_at_all_scores_zero(self):
        true = np.full((3, 2, 1), -1.0, dtype=np.float32)
        scores, counts = va.legacy_v1_scores(true.copy(), true, [0.0], ["p_gt10"])
        self.assertEqual(counts, {"p_gt10": (0, 0, 0, 6)})
        for name in ("tss", "hss", "pod", "far"):
            self.assertEqual(scores[f"{name}_p_gt10"], 0.0, name)

    def test_threshold_is_inclusive(self):
        true = np.array([[[-5.0], [-6.0]]], dtype=np.float32)
        pred = np.array([[[-5.0], [-5.0]]], dtype=np.float32)
        _, counts = va.legacy_v1_scores(pred, true, [-5.0], ["xrs_long"])
        self.assertEqual(counts, {"xrs_long": (1, 1, 0, 0)})

    def test_rmse_and_mae_join_every_channel(self):
        true = np.zeros((2, 3, 2), dtype=np.float32)
        pred = true.copy()
        pred[..., 0] = 1.0     # |delta| 1 on channel 0
        pred[..., 1] = -3.0    # |delta| 3 on channel 1
        scores, _ = va.legacy_v1_scores(pred, true, [9.0, 9.0], ["p_gt10", "xrs_long"])
        self.assertAlmostEqual(scores["rmse"], math.sqrt(5.0), places=12)
        self.assertAlmostEqual(scores["mae"], 2.0, places=12)

    def test_matches_the_v1_code(self):
        pred, true = _arrays("multi", seed=3)
        pred[0, :3, 1] = -5.0          # exactly on the threshold
        pred[1, 2, 0] = np.nan         # a NaN prediction
        cols, _, stored = _TRACKS["multi"]
        scores, counts = va.legacy_v1_scores(pred, true, stored, cols)
        expected = _v1_reference(pred, true, stored, cols)
        self.assertEqual(list(scores), list(expected))
        for name, value in expected.items():
            if name in ("rmse", "mae"):
                self.assertTrue(math.isnan(scores[name]), name)
            else:
                self.assertEqual(scores[name], value, name)
        pred[1, 2, 0] = 0.0
        scores, _ = va.legacy_v1_scores(pred, true, stored, cols)
        expected = _v1_reference(pred, true, stored, cols)
        for name in ("rmse", "mae"):
            self.assertAlmostEqual(scores[name], expected[name], delta=1e-6 * expected[name])
        self.assertEqual(set(counts), {"p_gt10", "xrs_long"})

    def test_unnamed_channels_and_threshold_count(self):
        pred, true = _arrays("multi", seed=4)
        scores, counts = va.legacy_v1_scores(pred, true, [1.0, -5.0], ["p_gt10"])
        self.assertEqual(set(counts), {"p_gt10", "c1"})
        self.assertIn("tss_c1", scores)
        with self.assertRaisesRegex(ValueError, "threshold"):
            va.legacy_v1_scores(pred, true, [1.0], ["p_gt10", "xrs_long"])


class TestFlipBound(unittest.TestCase):
    COUNTS = (1428, 1218, 372, 996982)

    def test_flip_bound_accepts_one_flip(self):
        # v1 saw one more hit (a miss of the dump flipped to a hit)
        v1_far = 1218 / (1429 + 1218)
        self.assertTrue(va.within_flip_bound("far", self.COUNTS, v1_far))

    def test_flip_bound_rejects_large_gap(self):
        far = 1218 / (1428 + 1218)
        self.assertFalse(va.within_flip_bound("far", self.COUNTS, far + 0.05))

    def test_unchanged_counts_pass_for_every_metric(self):
        tp, fp, fn, tn = self.COUNTS
        for name, fn_ in [("tss", _v1_tss), ("hss", _v1_hss), ("pod", _v1_pod),
                          ("far", _v1_far)]:
            self.assertTrue(va.within_flip_bound(name, self.COUNTS, fn_(tp, fp, fn, tn)), name)

    def test_flips_on_both_axes_add_up(self):
        tp, fp, fn, tn = self.COUNTS
        two_and_three = _v1_tss(tp + 2, fp - 3, fn - 2, tn + 3)
        self.assertTrue(va.within_flip_bound("tss", self.COUNTS, two_and_three))
        pod_six = _v1_pod(tp - 6, fp, fn + 6, tn)    # six hits lost: beyond 5
        self.assertFalse(va.within_flip_bound("pod", self.COUNTS, pod_six))
        self.assertTrue(va.within_flip_bound("pod", self.COUNTS, pod_six, k=6))

    def test_false_alarm_flips_count_too(self):
        # few non-events: one FP -> TN flip moves tss far more than 5 hit flips can
        counts = (500, 3, 400, 10)
        self.assertTrue(va.within_flip_bound("tss", counts, _v1_tss(500, 2, 400, 11)))
        self.assertFalse(va.within_flip_bound("tss", counts, _v1_tss(500, 0, 400, 13), k=2))

    def test_k_zero_needs_the_same_value(self):
        tp, fp, fn, tn = self.COUNTS
        self.assertTrue(va.within_flip_bound("pod", self.COUNTS, _v1_pod(tp, fp, fn, tn), k=0))
        self.assertFalse(va.within_flip_bound("pod", self.COUNTS,
                                              _v1_pod(tp + 1, fp, fn - 1, tn), k=0))

    def test_tolerance_is_tiny(self):
        tp, fp, fn, tn = self.COUNTS
        pod = _v1_pod(tp, fp, fn, tn)
        self.assertTrue(va.within_flip_bound("pod", self.COUNTS, pod + 5e-10, k=0))
        self.assertFalse(va.within_flip_bound("pod", self.COUNTS, pod + 5e-9, k=0))

    def test_counts_never_go_negative(self):
        # one hit only: two hits lost would need tp = -1 (pod -0.1)
        self.assertTrue(va.within_flip_bound("pod", (1, 0, 9, 90), 0.0))
        self.assertFalse(va.within_flip_bound("pod", (1, 0, 9, 90), -0.1))

    def test_nan_v1_value_fails(self):
        # the v1 rules never give nan, so a nan in the v1 table is a mismatch
        self.assertFalse(va.within_flip_bound("tss", self.COUNTS, float("nan")))

    def test_unknown_metric(self):
        with self.assertRaises(KeyError):
            va.within_flip_bound("csi", self.COUNTS, 0.1)


class TestVerify(unittest.TestCase):
    UNI = "uni_b_seq288_pred36_f0_direct"
    MULTI = "multi_seq864_pred72_f1_direct"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bench = self.root / "archive" / "bench"
        self.uni = _arrays("uni_b", seed=1)
        self.multi = _arrays("multi", seed=2)
        _write_run(self.bench, self.UNI, "dlinear", *self.uni)
        _write_run(self.bench, self.MULTI, "patchtst", *self.multi)
        self.v1_rows = [_v1_row(self.UNI, "dlinear", *self.uni),
                        _v1_row(self.MULTI, "patchtst", *self.multi)]

    def _verify(self, rows=None):
        v1 = _write_v1(self.root / "results_master.csv",
                       self.v1_rows if rows is None else rows)
        return va.verify(self.bench, v1)

    def _row(self, report, model, metric):
        hit = report[(report.model == model) & (report.metric == metric)]
        self.assertEqual(len(hit), 1, (model, metric))
        return hit.iloc[0]

    def test_a_faithful_dump_passes(self):
        report = self._verify()
        self.assertEqual(list(report.columns),
                         ["track", "seq_len", "pred_len", "fold", "strategy", "model",
                          "metric", "v1", "archive", "ok"])
        self.assertTrue(report.ok.all(), report[~report.ok])
        uni = report[report.model == "dlinear"]
        self.assertEqual(list(uni.metric), ["rmse", "mae", "tss_xrs_long", "hss_xrs_long",
                                            "pod_xrs_long", "far_xrs_long"])
        multi = report[report.model == "patchtst"]
        self.assertEqual(len(multi), 2 + 4 + 4)
        self.assertEqual(multi.iloc[0][["track", "seq_len", "pred_len", "fold", "strategy"]]
                         .tolist(), ["multi", 864, 72, 1, "direct"])
        row = self._row(report, "dlinear", "tss_xrs_long")
        self.assertEqual(row.v1, row.archive)

    def test_a_few_flipped_threshold_points_pass(self):
        pred, true = self.uni
        v1_pred = pred.copy()
        idx = np.argwhere((pred[..., 0] < -5.0) & (true[..., 0] >= -5.0))[:3]
        self.assertEqual(len(idx), 3)
        below = np.nextafter(np.float32(-5.0), np.float32(-np.inf))
        for i, j in idx:          # dump: a hit exactly on the threshold; v1: a miss
            pred[i, j, 0] = -5.0
            v1_pred[i, j, 0] = below
        _write_run(self.bench, self.UNI, "dlinear", pred, true)
        self.v1_rows[0] = _v1_row(self.UNI, "dlinear", v1_pred, true)
        report = self._verify()
        self.assertTrue(report.ok.all(), report[~report.ok])
        row = self._row(report, "dlinear", "pod_xrs_long")
        self.assertNotEqual(row.v1, row.archive)

    def test_many_flipped_points_fail(self):
        pred, true = self.uni
        v1_pred = pred.copy()
        hits = np.argwhere((pred[..., 0] >= -5.0) & (true[..., 0] >= -5.0))
        self.assertGreater(len(hits), 6)
        for i, j in hits:          # v1 missed every one of the dump's hits
            v1_pred[i, j, 0] = -9.0
        row = _v1_row(self.UNI, "dlinear", v1_pred, true)
        faithful = _v1_row(self.UNI, "dlinear", pred, true)
        row["rmse"], row["mae"] = faithful["rmse"], faithful["mae"]
        self.v1_rows[0] = row
        report = self._verify()
        self.assertFalse(self._row(report, "dlinear", "pod_xrs_long").ok)
        self.assertTrue(self._row(report, "dlinear", "rmse").ok)
        self.assertTrue(report[report.model == "patchtst"].ok.all())

    def test_rmse_relative_tolerance(self):
        rmse = self.v1_rows[1]["rmse"]
        self.v1_rows[1]["rmse"] = rmse * (1 + 5e-5)
        self.assertTrue(self._row(self._verify(), "patchtst", "rmse").ok)
        self.v1_rows[1]["rmse"] = rmse * (1 + 2e-4)
        self.assertFalse(self._row(self._verify(), "patchtst", "rmse").ok)
        self.v1_rows[1]["rmse"] = float("nan")
        self.assertFalse(self._row(self._verify(), "patchtst", "rmse").ok)

    def test_a_diverged_run_passes_with_nan_rmse(self):
        pred, true = self.uni
        pred[:] = np.nan
        _write_run(self.bench, self.UNI, "dlinear", pred, true)
        self.v1_rows[0] = _v1_row(self.UNI, "dlinear", pred, true)
        report = self._verify()
        self.assertTrue(report.ok.all(), report[~report.ok])
        row = self._row(report, "dlinear", "rmse")
        self.assertTrue(math.isnan(row.v1) and math.isnan(row.archive))
        self.assertEqual(self._row(report, "dlinear", "far_xrs_long").archive, 0.0)

    def test_a_missing_v1_score_fails(self):
        self.v1_rows[0]["hss_xrs_long"] = float("nan")
        self.assertFalse(self._row(self._verify(), "dlinear", "hss_xrs_long").ok)

    def test_runs_on_one_side_only_fail(self):
        extra = _v1_row("uni_a_seq288_pred36_f0_direct", "tide", *_arrays("uni_a", seed=5))
        _write_run(self.bench, self.UNI, "frets", *self.uni)       # not in v1
        report = self._verify(self.v1_rows + [extra])
        missing = self._row(report, "tide", "run")
        self.assertEqual((missing.v1, missing.archive, missing.ok), (1.0, 0.0, False))
        self.assertEqual(missing.track, "uni_a")
        unknown = self._row(report, "frets", "run")
        self.assertEqual((unknown.v1, unknown.archive, unknown.ok), (0.0, 1.0, False))
        self.assertEqual(len(report[report.model.isin(["tide", "frets"])]), 2)
        self.assertEqual(int((~report.ok).sum()), 2)

    def test_other_files_are_not_runs(self):
        (self.bench / "smoke" / "pred").mkdir(parents=True)
        (self.bench / "smoke" / "pred" / "dlinear.npz").write_bytes(b"")
        (self.bench / self.UNI / "pred" / "tide.npz.tmp").write_bytes(b"")
        report = self._verify()
        self.assertEqual(set(report.model), {"dlinear", "patchtst"})
        self.assertTrue(report.ok.all())

    def test_rows_are_in_run_order(self):
        report = self._verify()
        self.assertEqual(list(report.drop_duplicates("model").track), ["multi", "uni_b"])

    def test_a_repeated_v1_run_is_refused(self):
        with self.assertRaisesRegex(ValueError, "twice"):
            self._verify(self.v1_rows + [self.v1_rows[0]])


class TestPaper(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.bench = root / "bench"
        self.folds = root / "folds"
        self.folds.mkdir()

    def test_compare_paper_counts(self):
        uni = _arrays("uni_b", seed=6)
        seg = _arrays("uni_b", seed=7)
        multi = _arrays("multi", seed=8)
        _write_run(self.bench, "uni_b_seq288_pred144_f0_direct", "dlinear", *uni)
        _write_run(self.bench, "uni_b_seq288_pred144_f0_direct", "segrnn_thuml", *seg)
        _write_run(self.bench, "multi_seq288_pred288_f1_direct", "dlinear", *multi)
        uni_point = _counts(uni[0][..., 0] >= -5.0, uni[1][..., 0] >= -5.0)
        uni_window = _counts(uni[0][..., 0].max(1) >= -5.0, uni[1][..., 0].max(1) >= -5.0)
        seg_point = _counts(seg[0][..., 0] >= -5.0, seg[1][..., 0] >= -5.0)
        p_point = _counts(multi[0][..., 0] >= 1.0, multi[1][..., 0] >= 1.0)
        x_window = _counts(multi[0][..., 1].max(1) >= -5.0, multi[1][..., 1].max(1) >= -5.0)
        tp, fp, fn, tn = uni_point
        _paper_csv(self.folds / "event_skill_fold0.csv", [
            _paper_row("XRS M-class", "naive (persistence)", "per-point", 0, (1, 1, 1, 1)),
            _paper_row("XRS M-class", "dlinear", "per-point", 0,
                       (tp - 2, fp + 1, fn + 2, tn - 1)),
            _paper_row("XRS M-class", "dlinear", "per-window", 0, uni_window),
            _paper_row("XRS M-class", "dlinear", "per-day", 0, (1, 1, 1, 1)),
            _paper_row("XRS X-class", "dlinear", "per-point", 0, (1, 1, 1, 1)),
            _paper_row("XRS M-class", "segrnn", "per-point", 0, seg_point),
            _paper_row("XRS M-class", "tide", "per-point", 0, (1, 1, 1, 1)),
        ])
        _paper_csv(self.folds / "event_skill_p288_fold1.csv", [
            _paper_row("Proton SPE (multi)", "dlinear", "per-point", 1, p_point),
            _paper_row("XRS M-class (multi)", "dlinear", "per-window", 1, x_window),
        ])
        _paper_csv(self.folds / "persistence_panel_fold0.csv", [
            _paper_row("XRS M-class", "dlinear", "per-point", 0, (1, 1, 1, 1))])

        out = va.compare_paper_counts(self.bench, self.folds)
        self.assertEqual(list(out.columns),
                         ["track", "seq_len", "pred_len", "fold", "strategy", "model",
                          "channel", "granularity", "dTP", "dFP", "dFN", "dTN"])
        got = {(r.track, r.pred_len, r.fold, r.model, r.channel, r.granularity):
               (r.dTP, r.dFP, r.dFN, r.dTN) for r in out.itertuples()}
        self.assertEqual(set(got), {
            ("uni_b", 144, 0, "dlinear", "xrs_long", "point"),
            ("uni_b", 144, 0, "dlinear", "xrs_long", "window"),
            ("uni_b", 144, 0, "segrnn_thuml", "xrs_long", "point"),
            ("uni_b", 144, 0, "tide", "xrs_long", "point"),
            ("multi", 288, 1, "dlinear", "p_gt10", "point"),
            ("multi", 288, 1, "dlinear", "xrs_long", "window"),
        })
        self.assertEqual(got[("uni_b", 144, 0, "dlinear", "xrs_long", "point")],
                         (2, -1, -2, 1))
        for key, diff in got.items():
            if key[3] == "tide":          # no prediction file
                self.assertTrue(all(pd.isna(d) for d in diff), diff)
            elif key[5] != "point" or key[3] != "dlinear" or key[0] != "uni_b":
                self.assertEqual(diff, (0, 0, 0, 0), key)
        self.assertTrue((out.seq_len == 288).all() and (out.strategy == "direct").all())

    def test_no_paper_files_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "event_skill"):
            va.check_formulas_against_paper(self.folds)

    def test_check_formulas_against_paper(self):
        def rounded(counts):
            return {m.upper(): round(EVENT_SCORES[m](*counts), 3)
                    for m in ("tss", "hss", "csi", "sedi")}
        good = (120, 45, 30, 9000)
        tie = (1, 1, 14, 808)            # csi 0.0625, which round() made 0.062
        no_hit = (0, 3, 9, 900)          # sedi undefined: the paper left it empty
        rows = [
            _paper_row("XRS M-class", "naive (persistence)", "per-point", 0, good,
                       **rounded(good)),
            _paper_row("XRS X-class", "xpatch", "per-day", 0, tie,
                       **{**rounded(tie), "CSI": 0.062}),
            _paper_row("Proton SPE", "tide", "per-window", 0, no_hit,
                       **{**rounded(no_hit), "SEDI": float("nan")}),
            _paper_row("Proton SPE", "dlinear", "per-point", 0, good,
                       **{**rounded(good),
                          "TSS": round(EVENT_SCORES["tss"](*good), 3) + 0.001}),
            _paper_row("Proton SPE", "rlinear", "per-point", 0, good,
                       **{**rounded(good), "SEDI": float("nan")}),
        ]
        _paper_csv(self.folds / "event_skill_fold0.csv", rows)
        _paper_csv(self.folds / "event_skill_p288_fold0.csv", rows[:1])
        out = va.check_formulas_against_paper(self.folds)
        self.assertEqual(list(out.columns), ["file", "track", "model", "gran", "fold",
                                             "metric", "paper", "ours", "ok"])
        self.assertEqual(len(out), 6 * 4)
        bad = out[~out.ok]
        self.assertEqual(sorted(zip(bad.model, bad.metric)),
                         [("dlinear", "tss"), ("rlinear", "sedi")])
        self.assertEqual(set(out.file),
                         {"event_skill_fold0.csv", "event_skill_p288_fold0.csv"})
        tie_row = out[(out.model == "xpatch") & (out.metric == "csi")].iloc[0]
        self.assertEqual((tie_row.paper, tie_row.ours, tie_row.ok), (0.062, 0.0625, True))
        sedi = out[(out.model == "tide") & (out.metric == "sedi")].iloc[0]
        self.assertTrue(math.isnan(sedi.paper) and math.isnan(sedi.ours) and sedi.ok)


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "verify_archive_script", _REPO / "scripts" / "verify_archive.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestVerifyScript(unittest.TestCase):
    CELL = "uni_b_seq288_pred144_f0_direct"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bench = self.root / "archive" / "bench"
        self.arrays = _arrays("uni_b", seed=9)
        _write_run(self.bench, self.CELL, "dlinear", *self.arrays)
        self.v1_row = _v1_row(self.CELL, "dlinear", *self.arrays)
        self.v1 = _write_v1(self.root / "results_master.csv", [self.v1_row])
        self.out = self.root / "reports" / "verify.csv"
        self.folds = self.root / "folds"
        self.folds.mkdir()
        tp, fp, fn, tn = _counts(self.arrays[0][..., 0] >= -5.0,
                                 self.arrays[1][..., 0] >= -5.0)
        paper = (tp + 1, fp, fn - 1, tn)     # the paper saw one more hit
        scores = {m.upper(): round(EVENT_SCORES[m](*paper), 3)
                  for m in ("tss", "hss", "csi", "sedi")}
        _paper_csv(self.folds / "event_skill_fold0.csv", [
            _paper_row("XRS M-class", "dlinear", "per-point", 0, paper, **scores)])

    def _main(self, *extra):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = _load_script().main(["--archive", str(self.bench), "--v1", str(self.v1),
                                        "--out", str(self.out), *extra])
        return code, out.getvalue(), err.getvalue()

    def test_a_clean_archive_exits_zero(self):
        code, out, err = self._main()
        self.assertEqual(code, 0, out + err)
        report = pd.read_csv(self.out)
        self.assertEqual(len(report), 6)
        self.assertTrue(report.ok.all())
        self.assertIn("runs: 1, rows: 6, failing rows: 0", out)
        self.assertFalse((self.root / "reports" / "verify_paper_formulas.csv").exists())

    def test_a_failing_row_exits_one(self):
        self.v1_row["far_xrs_long"] += 0.2
        _write_v1(self.v1, [self.v1_row])
        code, out, _ = self._main()
        self.assertEqual(code, 1)
        self.assertIn("failing rows: 1", out)
        self.assertIn("far_xrs_long: 1", out)

    def test_paper_checks(self):
        code, out, err = self._main("--paper-folds", str(self.folds))
        self.assertEqual(code, 0, out + err)
        formulas = pd.read_csv(self.root / "reports" / "verify_paper_formulas.csv")
        self.assertEqual(len(formulas), 4)
        self.assertTrue(formulas.ok.all())
        counts = pd.read_csv(self.root / "reports" / "verify_paper_counts.csv")
        self.assertEqual(counts[["dTP", "dFP", "dFN", "dTN"]].iloc[0].tolist(), [-1, 0, 1, 0])
        self.assertIn("reference", out)    # count differences do not fail the check

    def test_a_formula_mismatch_exits_one(self):
        table = pd.read_csv(self.folds / "event_skill_fold0.csv")
        table.loc[0, "HSS"] += 0.01
        table.to_csv(self.folds / "event_skill_fold0.csv", index=False)
        code, out, _ = self._main("--paper-folds", str(self.folds))
        self.assertEqual(code, 1)
        self.assertIn("hss: 1", out)

    def test_bad_inputs_are_errors(self):
        for args, flag in [(["--archive", str(self.root / "nope")], "--archive"),
                           (["--v1", str(self.root / "nope.csv")], "--v1"),
                           (["--paper-folds", str(self.root / "nope")], "--paper-folds")]:
            out, err = io.StringIO(), io.StringIO()
            argv = {"--archive": str(self.bench), "--v1": str(self.v1), "--out": str(self.out)}
            argv.update(dict(zip(args[::2], args[1::2])))
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = _load_script().main([x for kv in argv.items() for x in kv])
            self.assertEqual(code, 1, flag)
            self.assertIn("error:", err.getvalue(), flag)
        empty = self.root / "empty"
        empty.mkdir()
        code, _, err = self._main("--paper-folds", str(empty))
        self.assertEqual(code, 1)
        self.assertIn("event_skill", err)


if __name__ == "__main__":
    unittest.main()
