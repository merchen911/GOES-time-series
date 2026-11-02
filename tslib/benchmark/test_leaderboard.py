import contextlib
import importlib.util
import io
import re
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

from tslib.benchmark.leaderboard import leaderboard_markdown, summary_html
from tslib.exp.records import event_columns, regression_columns

_REPO = Path(__file__).resolve().parents[2]
_NAN = float("nan")
_CHANNEL = {"uni_a": "p_gt10", "uni_b": "xrs_long"}


def _reg(*rows):
    """A regression table; each row is a dict, unnamed columns are NaN."""
    return pd.DataFrame([{"strategy": "direct", "n": 100, **r} for r in rows],
                        columns=regression_columns())


def _evt(*rows):
    return pd.DataFrame([{"strategy": "direct", "granularity": "day", "n": 100, **r}
                         for r in rows], columns=event_columns())


def _run(track="uni_b", model="dlinear", pred_len=36, seq_len=288, fold=0, **kw):
    return {"track": track, "model": model, "pred_len": pred_len, "seq_len": seq_len,
            "fold": fold, **kw}


def _reg_run(rmse=0.5, msa=10.0, logr=0.9, channel=None, **kw):
    run = _run(**kw)
    return {**run, "channel": channel or _CHANNEL[run["track"]],
            "rmse": rmse, "msa": msa, "logr": logr}


def _evt_run(hss=0.4, tss=0.3, channel=None, **kw):
    run = _run(**kw)
    return {**run, "channel": channel or _CHANNEL[run["track"]], "hss": hss, "tss": tss}


def _tables(text):
    """{(horizon heading, track heading): table lines} of a leaderboard."""
    out, horizon, track = {}, None, None
    for line in text.splitlines():
        if line.startswith("### "):
            horizon, track = line[4:], None
        elif line.startswith("**") and line.endswith("**"):
            track = line.strip("*")
            out[(horizon, track)] = []
        elif line.startswith("|") and track is not None:
            out[(horizon, track)].append([c.strip() for c in line.strip("|").split("|")])
    return out


def _table(markdown, track="SXR", horizon="3 h"):
    """The one table of a leaderboard whose track heading and horizon title start
    with the given text, as parsed lines."""
    (table,) = [t for (h, tr), t in _tables(markdown).items()
                if tr.startswith(track) and h.startswith(horizon)]
    return table


def _body(table):
    """Data rows of a parsed table (header and rule line dropped)."""
    return table[2:]


class TestSections(unittest.TestCase):
    def test_four_horizon_sections(self):
        pred_lens = (36, 72, 144, 288)
        reg = _reg(*[_reg_run(track=t, pred_len=p) for p in pred_lens
                     for t in ("uni_b", "uni_a")],
                   *[_reg_run(track="multi", channel=c, pred_len=p) for p in pred_lens
                     for c in ("p_gt10", "xrs_long")])
        text = leaderboard_markdown(reg, _evt())
        horizons = [line[4:] for line in text.splitlines() if line.startswith("### ")]
        self.assertEqual([h.split(" (")[0] for h in horizons],
                         ["3 h", "6 h", "12 h", "24 h"])
        self.assertEqual([re.search(r"pred_len (\d+)", h).group(1) for h in horizons],
                         ["36", "72", "144", "288"])
        tables = _tables(text)
        self.assertEqual(len(tables), 12)   # 4 horizons x 3 tracks
        for h in horizons:
            self.assertEqual([k[1] for k in tables if k[0] == h],
                             ["SXR 1--8 A (univariate)", "Proton >=10 MeV (univariate)",
                              "Multivariate (joint proton + SXR)"])

    def test_a_run_counts_only_in_its_own_horizon(self):
        reg = _reg(_reg_run(pred_len=36, rmse=0.1), _reg_run(pred_len=72, rmse=0.9))
        tables = _tables(leaderboard_markdown(reg, _evt()))
        by_h = {k[0].split(" (")[0]: _body(t) for k, t in tables.items()
                if k[1].startswith("SXR")}
        self.assertEqual(by_h["3 h"][0][1], "0.100")
        self.assertEqual(by_h["6 h"][0][1], "0.900")
        self.assertEqual(set(by_h), {"3 h", "6 h"})   # no runs, no table

    def test_columns_per_channel(self):
        reg = _reg(_reg_run(pred_len=144),
                   *[_reg_run(track="multi", channel=c, pred_len=144)
                     for c in ("p_gt10", "xrs_long")])
        evt = _evt(_evt_run(pred_len=144),
                   *[_evt_run(track="multi", channel=c, pred_len=144)
                     for c in ("p_gt10", "xrs_long")])
        tables = _tables(leaderboard_markdown(reg, evt))
        header = tables[("12 h (pred_len 144)", "Multivariate (joint proton + SXR)")][0]
        self.assertEqual(header, ["Model"] + [f"{ch} {m}" for ch in ("Proton", "SXR")
                                              for m in ("RMSE", "MSA", "logR", "HSS", "TSS")])
        uni = tables[("12 h (pred_len 144)", "SXR 1--8 A (univariate)")][0]
        self.assertEqual(uni, ["Model"] + [f"SXR {m}" for m in
                                           ("RMSE", "MSA", "logR", "HSS", "TSS")])


class TestCells(unittest.TestCase):
    def test_cell_is_the_mean_over_folds_and_input_lengths(self):
        runs = [(0, 288, 0.2), (1, 288, 0.4), (0, 864, 0.6)]
        reg = _reg(*[_reg_run(fold=f, seq_len=s, rmse=v, msa=v * 10, logr=1 - v)
                     for f, s, v in runs])
        evt = _evt(*[_evt_run(fold=f, seq_len=s, hss=v, tss=v / 2) for f, s, v in runs])
        table = _table(leaderboard_markdown(reg, evt))
        self.assertEqual(_body(table), [["DLinear", "0.400", "4.000", "0.600", "0.400", "0.200"]])

    def test_event_cells_read_the_day_granularity(self):
        evt = _evt(_evt_run(hss=0.9, granularity="point"),
                   _evt_run(hss=0.8, granularity="window"),
                   _evt_run(hss=0.1, tss=0.2, granularity="day"))
        table = _table(leaderboard_markdown(_reg(_reg_run()), evt))
        self.assertEqual(_body(table)[0][4:], ["0.100", "0.200"])

    def test_defined_count_annotation(self):
        reg = _reg(_reg_run(fold=0, rmse=0.5, msa=10.0, logr=0.9),
                   _reg_run(fold=1, rmse=_NAN, msa=30.0, logr=0.7))
        table = _table(leaderboard_markdown(reg, _evt()))
        row = _body(table)[0]
        self.assertEqual(row[1], "0.500 (1/2)")          # NaN run left out, count shown
        self.assertEqual(row[2], "20.000")                # fully defined: no annotation
        self.assertEqual(row[3], "0.800")

    def test_count_annotation_applies_to_event_cells_too(self):
        evt = _evt(_evt_run(fold=0, hss=0.5, tss=0.4), _evt_run(fold=1, hss=_NAN, tss=0.2),
                   _evt_run(fold=2, hss=_NAN, tss=0.3))
        table = _table(leaderboard_markdown(_reg(_reg_run()), evt))
        self.assertEqual(_body(table)[0][4:], ["0.500 (1/3)", "0.300"])

    def test_cell_without_any_defined_run_is_a_dash_with_its_count(self):
        reg = _reg(_reg_run(fold=0, rmse=_NAN), _reg_run(fold=1, rmse=_NAN))
        table = _table(leaderboard_markdown(reg, _evt()))
        self.assertEqual(_body(table)[0][1], "-- (0/2)")

    def test_cell_without_rows_is_a_plain_dash(self):
        # the model has regression rows but no event rows at all
        table = _table(leaderboard_markdown(_reg(_reg_run()), _evt()))
        self.assertEqual(_body(table)[0][4:], ["--", "--"])

    def test_channels_and_tracks_do_not_mix(self):
        reg = _reg(_reg_run(track="multi", channel="p_gt10", rmse=0.1),
                   _reg_run(track="multi", channel="xrs_long", rmse=0.9),
                   _reg_run(track="uni_a", rmse=0.5))
        tables = _tables(leaderboard_markdown(reg, _evt()))
        multi = _body(tables[("3 h (pred_len 36)", "Multivariate (joint proton + SXR)")])[0]
        self.assertEqual((multi[1], multi[6]), ("0.100", "0.900"))
        uni = _body(tables[("3 h (pred_len 36)", "Proton >=10 MeV (univariate)")])[0]
        self.assertEqual(uni[1], "0.500")


class TestRowOrder(unittest.TestCase):
    def test_rows_sorted_by_first_channel_rmse(self):
        reg = _reg(*[_reg_run(track="multi", model=m, channel=c, rmse=r)
                     for m, rp, rx in (("dlinear", 0.3, 0.1), ("segrnn_thuml", 0.1, 0.9),
                                       ("tide", 0.2, 0.5))
                     for c, r in (("p_gt10", rp), ("xrs_long", rx))])
        table = _table(leaderboard_markdown(reg, _evt()), "Multivariate")
        # the first channel of the multivariate track is p_gt10 (the driver's order)
        self.assertEqual([r[0] for r in _body(table)], ["SegRNN", "TiDE", "DLinear"])

    def test_undefined_rmse_sinks_to_the_bottom(self):
        reg = _reg(_reg_run(model="dlinear", rmse=_NAN), _reg_run(model="tide", rmse=0.9),
                   _reg_run(model="rlinear", rmse=0.2))
        table = _table(leaderboard_markdown(reg, _evt()))
        self.assertEqual([r[0] for r in _body(table)], ["RLinear", "TiDE", "DLinear"])

    def test_order_follows_the_mean_not_the_annotated_text(self):
        reg = _reg(_reg_run(model="tide", fold=0, rmse=0.30),
                   _reg_run(model="tide", fold=1, rmse=_NAN),
                   _reg_run(model="rlinear", fold=0, rmse=0.9),
                   _reg_run(model="rlinear", fold=1, rmse=0.9))
        table = _table(leaderboard_markdown(reg, _evt()))
        self.assertEqual([(r[0], r[1]) for r in _body(table)],
                         [("TiDE", "0.300 (1/2)"), ("RLinear", "0.900")])


def _notes(text):
    """Footnote paragraphs of a leaderboard (lines in _italics_)."""
    return [line for line in text.splitlines() if line.startswith("_")]


def _section(text, horizon):
    """The text of one horizon section ("3 h", "12 h", ...)."""
    return text.split(f"### {horizon} (")[1].split("### ")[0]


class TestExclusion(unittest.TestCase):
    def _frames(self, pred_len):
        # dlinear ran the whole grid (fold 0, inputs 288 and 2016); xPatch ran
        # 288 and diverged at 2016 (two excluded runs, one of them at fold 1)
        runs = [dict(model="dlinear", seq_len=288, rmse=0.4, hss=0.3),
                dict(model="dlinear", seq_len=2016, rmse=0.4, hss=0.3),
                dict(model="xpatch", seq_len=288, rmse=0.2, hss=0.5),
                dict(model="xpatch", seq_len=2016, rmse=_NAN, hss=_NAN),
                dict(model="xpatch", seq_len=2016, fold=1, rmse=_NAN, hss=_NAN)]
        reg = _reg(*[_reg_run(pred_len=pred_len, **{k: v for k, v in r.items() if k != "hss"})
                     for r in runs])
        evt = _evt(*[_evt_run(pred_len=pred_len, **{k: v for k, v in r.items() if k != "rmse"})
                     for r in runs])
        return reg, evt

    def test_xpatch_2016_excluded_marked_and_footnoted(self):
        text = leaderboard_markdown(*self._frames(144))
        rows = {r[0]: r for r in _body(_table(text, horizon="12 h"))}
        # the excluded runs are not in the mean, and the cell shows that it
        # holds 1 of the 2 runs of this forecast length's grid
        self.assertEqual((rows["xPatch\u2020"][1], rows["xPatch\u2020"][4]),
                         ("0.200 (1/2)", "0.500 (1/2)"))
        self.assertEqual(rows["DLinear"][1], "0.400")          # full grid: no mark, no count
        (note,) = _notes(text)
        for part in ("\u2020 xPatch", "7-day input", "seq_len 2016", "diverges",
                     "1 run", "2 runs", "diverged"):
            self.assertIn(part, note)
        # the note sits below the tables of its horizon
        lines = text.splitlines()
        self.assertGreater(lines.index(note), max(i for i, l in enumerate(lines)
                                                  if l.startswith("|")))

    def test_xpatch_without_7_day_runs_is_footnoted_where_nothing_was_excluded(self):
        # v004 never ran xPatch at 2016 (3 h / 6 h): its cells still average
        # fewer runs, which the table must say, but no run was left out
        reg, evt = self._frames(36)
        reg = reg[~((reg.model == "xpatch") & (reg.seq_len == 2016))]
        evt = evt[~((evt.model == "xpatch") & (evt.seq_len == 2016))]
        text = leaderboard_markdown(reg, evt)
        rows = {r[0]: r for r in _body(_table(text))}
        self.assertEqual(rows["xPatch\u2020"][1], "0.200 (1/2)")
        (note,) = _notes(text)
        self.assertIn("xPatch is left out at the 7-day input", note)
        self.assertNotIn("diverged", note)

    def test_diverged_runs_are_counted_only_where_they_were_left_out(self):
        reg36, evt36 = self._frames(36)
        keep = ~((reg36.model == "xpatch") & (reg36.seq_len == 2016))
        reg36, evt36 = reg36[keep], evt36[~((evt36.model == "xpatch") & (evt36.seq_len == 2016))]
        reg144, evt144 = self._frames(144)
        text = leaderboard_markdown(pd.concat([reg36, reg144]), pd.concat([evt36, evt144]))
        self.assertIn("xPatch is left out at the 7-day input", _section(text, "3 h"))
        self.assertNotIn("diverged", _section(text, "3 h"))
        self.assertIn("2 diverged 7-day runs", _section(text, "12 h"))

    def test_no_mark_and_no_note_when_every_model_has_the_whole_grid(self):
        reg, evt = self._frames(144)
        reg, evt = reg[reg.seq_len != 2016], evt[evt.seq_len != 2016]
        text = leaderboard_markdown(reg, evt)
        self.assertEqual(_notes(text), [])
        self.assertNotIn("\u2020", text)

    def test_a_model_left_with_no_runs_is_not_listed(self):
        reg = _reg(_reg_run(model="xpatch", seq_len=2016, rmse=_NAN),
                   _reg_run(model="dlinear", rmse=0.3))
        text = leaderboard_markdown(reg, _evt())
        self.assertEqual([r[0] for r in _body(_table(text))], ["DLinear"])
        (note,) = _notes(text)
        self.assertIn("left out", note)
        self.assertIn("(1 run)", note)


class TestRunCountsAgainstTheGrid(unittest.TestCase):
    """A cell is comparable only if it averages the same runs as the others: a
    model with fewer runs than the forecast length's grid (folds x input
    lengths) is marked, counted and footnoted at every horizon."""

    def _grid(self, pred_len=36, tide_runs=((0, 288), (1, 288))):
        runs = [dict(model="dlinear", fold=f, seq_len=s, rmse=0.4)
                for f in (0, 1) for s in (288, 864)]
        runs += [dict(model="tide", fold=f, seq_len=s, rmse=0.3) for f, s in tide_runs]
        return _reg(*[_reg_run(pred_len=pred_len, **r) for r in runs])

    def test_a_model_without_a_whole_input_length(self):
        text = leaderboard_markdown(self._grid(), _evt())
        rows = {r[0]: r for r in _body(_table(text))}
        self.assertEqual(rows["TiDE\u2020"][1], "0.300 (2/4)")
        self.assertEqual(rows["DLinear"][1], "0.400")
        (note,) = _notes(text)
        for part in ("\u2020 TiDE has no runs at the 3-day input (seq_len 864)",
                     "2 runs (1-day input)", "all 4 runs"):
            self.assertIn(part, note)
        self.assertNotIn("diverg", note)

    def test_a_model_missing_single_runs(self):
        text = leaderboard_markdown(self._grid(tide_runs=((0, 288), (1, 288), (0, 864))), _evt())
        rows = {r[0]: r for r in _body(_table(text))}
        self.assertEqual(rows["TiDE\u2020"][1], "0.300 (3/4)")
        (note,) = _notes(text)
        self.assertIn("fold 1 of the 3-day input", note)

    def test_the_note_names_the_tables_when_tracks_differ(self):
        reg = pd.concat([self._grid(),
                         _reg(*[_reg_run(track="uni_a", model=m, fold=f, seq_len=s, rmse=0.5)
                                for m in ("dlinear", "tide") for f in (0, 1)
                                for s in (288, 864)])])
        text = leaderboard_markdown(reg, _evt())
        proton = {r[0] for r in _body(_table(text, track="Proton"))}
        self.assertEqual(proton, {"DLinear", "TiDE"})            # full grid there
        (note,) = _notes(text)
        self.assertIn("SXR 1--8 A (univariate)", note)

    def test_undefined_values_still_count_against_the_grid(self):
        reg = self._grid(tide_runs=((0, 288), (1, 288), (0, 864), (1, 864)))
        reg.loc[(reg.model == "tide") & (reg.fold == 1) & (reg.seq_len == 864), "rmse"] = _NAN
        text = leaderboard_markdown(reg, _evt())
        rows = {r[0]: r for r in _body(_table(text))}
        self.assertEqual(rows["TiDE"][1], "0.300 (3/4)")         # all runs ran: no mark
        self.assertEqual(_notes(text), [])



class _Cells(HTMLParser):
    """Rows of every <table> in an HTML fragment as [[(tag, attrs, text)]]."""
    def __init__(self):
        super().__init__()
        self.tables, self._cell = [], None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.tables.append([])
        elif tag == "tr":
            self.tables[-1].append([])
        elif tag in ("td", "th"):
            self._cell = [tag, dict(attrs), "", False]
        elif tag == "sub" and self._cell is not None:
            self._cell[3] = True

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            self.tables[-1][-1].append(tuple(self._cell))
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell[2] += data


def _html_tables(html):
    parser = _Cells()
    parser.feed(html)
    return parser.tables


def _row(table, name):
    """Cell texts of the body row whose first cell starts with `name`."""
    (row,) = [r for r in table[2:] if r[0][2].startswith(name)]
    return [c[2] for c in row[1:]]


def _summary_runs(metric_rows, pred_lens=(36, 72, 144, 288)):
    """One run per (track, model, pred_len) from {model: value-or-callable}."""
    reg, evt = [], []
    for track in ("uni_b", "uni_a"):
        for model, value in metric_rows.items():
            for p in pred_lens:
                v = value(p) if callable(value) else value
                reg.append(_reg_run(track=track, model=model, pred_len=p, rmse=v))
                evt.append(_evt_run(track=track, model=model, pred_len=p, hss=v))
    return _reg(*reg), _evt(*evt)


class TestSummary(unittest.TestCase):
    def test_two_tables_with_grouped_two_row_headers(self):
        html = summary_html(*_summary_runs({"dlinear": 0.3}))
        tables = _html_tables(html)
        self.assertEqual(len(tables), 2)
        for table in tables:
            top, sub = table[0], table[1]
            self.assertEqual(top[0][2], "Model")
            self.assertEqual(top[0][1].get("rowspan"), "2")
            self.assertEqual([c[1].get("colspan") for c in top[1:]], ["4", "4"])
            self.assertTrue(top[1][2].startswith("SXR"))
            self.assertTrue(top[2][2].startswith("Proton"))
            self.assertEqual([c[2] for c in sub], ["3 h", "6 h", "12 h", "24 h"] * 2)

    def test_every_cell_is_small_text(self):
        for table in _html_tables(summary_html(*_summary_runs({"dlinear": 0.3}))):
            for row in table:
                self.assertTrue(all(c[3] for c in row), row)

    def test_forecast_lengths_are_not_averaged_together(self):
        values = {36: 0.1, 72: 0.2, 144: 0.3, 288: 0.4}
        reg_table, evt_table = _html_tables(summary_html(*_summary_runs({"dlinear": values.get})))
        self.assertEqual(_row(reg_table, "DLinear"), ["0.100", "0.200", "0.300", "0.400"] * 2)
        self.assertEqual(_row(evt_table, "DLinear"), ["0.100", "0.200", "0.300", "0.400"] * 2)

    def test_cell_is_the_mean_over_folds_and_input_lengths(self):
        reg = _reg(_reg_run(rmse=0.2, fold=0, seq_len=288), _reg_run(rmse=0.4, fold=1, seq_len=864))
        reg_table, _ = _html_tables(summary_html(reg, _evt()))
        self.assertEqual(_row(reg_table, "DLinear")[0], "0.300")

    def test_regression_is_rmse_and_classification_is_day_hss(self):
        reg = _reg(_reg_run(rmse=0.25, msa=99.0))
        evt = _evt(_evt_run(hss=0.6, tss=0.1),
                   {**_evt_run(hss=0.9), "granularity": "point"})
        reg_table, evt_table = _html_tables(summary_html(reg, evt))
        self.assertEqual(_row(reg_table, "DLinear")[0], "0.250")
        self.assertEqual(_row(evt_table, "DLinear")[0], "0.600")

    def test_best_per_column_is_bold_low_rmse_high_hss(self):
        reg, evt = _summary_runs({"dlinear": 0.2, "tide": 0.5})
        html = summary_html(reg, evt)
        reg_html, evt_html = html.split("</table>")[:2]
        self.assertIn("<b>0.200</b>", reg_html)
        self.assertNotIn("<b>0.500</b>", reg_html)
        self.assertIn("<b>0.500</b>", evt_html)
        self.assertNotIn("<b>0.200</b>", evt_html)

    def test_best_is_chosen_per_column(self):
        # dlinear wins every 3 h column, tide every other one
        reg, evt = _summary_runs({"dlinear": lambda p: 0.1 if p == 36 else 0.9,
                                  "tide": lambda p: 0.9 if p == 36 else 0.1})
        reg_html = summary_html(reg, evt).split("</table>")[0]
        rows = {r.split("</sub></td>", 1)[0].rsplit(">", 1)[-1]: r
                for r in reg_html.splitlines() if r.startswith("<tr><td>")}
        bold = lambda row: [("<b>" in c) for c in row.split("<td align=\"right\">")[1:]]
        self.assertEqual(bold(rows["DLinear"]), [True, False, False, False] * 2)
        self.assertEqual(bold(rows["TiDE"]), [False, True, True, True] * 2)

    def test_rows_in_alphabetical_order_of_the_shown_names(self):
        reg, evt = _summary_runs({"tide": 0.1, "dlinear": 0.9, "itransformer": 0.5})
        reg_table, _ = _html_tables(summary_html(reg, evt))
        self.assertEqual([r[0][2] for r in reg_table[2:]], ["DLinear", "iTransformer", "TiDE"])

    def test_only_the_univariate_tracks(self):
        reg = _reg(_reg_run(), _reg_run(track="multi", channel="xrs_long", model="tide"))
        reg_table, _ = _html_tables(summary_html(reg, _evt()))
        self.assertEqual([r[0][2] for r in reg_table[2:]], ["DLinear"])

    def test_missing_horizon_is_a_dash(self):
        reg_table, _ = _html_tables(summary_html(_reg(_reg_run(pred_len=36)), _evt()))
        self.assertEqual(_row(reg_table, "DLinear")[:4], ["0.500", "–", "–", "–"])

    def test_undefined_runs_are_counted_in_the_cell(self):
        reg = _reg(_reg_run(rmse=0.2, fold=0), _reg_run(rmse=_NAN, fold=1))
        reg_table, _ = _html_tables(summary_html(reg, _evt()))
        self.assertEqual(_row(reg_table, "DLinear")[0], "0.200 (1/2)")

    def test_short_model_is_marked_and_footnoted_and_2016_xpatch_left_out(self):
        runs = [_reg_run(model=m, seq_len=s, fold=f,
                         rmse=_NAN if (m, s) == ("xpatch", 2016) else 0.3)
                for m in ("dlinear", "xpatch") for s in (288, 2016) for f in (0, 1)]
        reg = _reg(*[r for r in runs if not (r["model"] == "xpatch" and r["seq_len"] == 2016
                                             and r["fold"] == 1)])
        html = summary_html(reg, _evt())
        reg_table, _ = _html_tables(html)
        self.assertEqual([r[0][2] for r in reg_table[2:]], ["DLinear", "xPatch†"])
        self.assertIn("† xPatch is left out at the 7-day input", html)
        self.assertNotIn("(2/3)", html)   # a leaked NaN 2016 run would show as 2 of 3
        self.assertEqual(_row(reg_table, "xPatch")[0], "0.300")


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "make_leaderboard_script", _REPO / "scripts" / "make_leaderboard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestScript(unittest.TestCase):
    README = ("# Title\n\nintro\n\n<!-- LEADERBOARD:START -->\nOLD TABLES\n"
              "<!-- LEADERBOARD:END -->\n\nRegenerate with the script.\n")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.reg, self.evt, self.readme = (root / "reg.csv", root / "evt.csv", root / "README.md")
        self.full = root / "results" / "LEADERBOARD.md"
        _reg(_reg_run(rmse=0.25)).to_csv(self.reg, index=False)
        _evt(_evt_run(hss=0.5)).to_csv(self.evt, index=False)
        self.readme.write_text(self.README)
        self.module = _load_script()

    def _main(self):
        argv = ["--regression", str(self.reg), "--event", str(self.evt),
                "--readme", str(self.readme), "--full", str(self.full)]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return self.module.main(argv)

    def test_replaces_only_the_marked_block(self):
        self.assertEqual(self._main(), 0)
        text = self.readme.read_text()
        head, tail = self.README.split("OLD TABLES\n")
        self.assertTrue(text.startswith(head))
        self.assertTrue(text.endswith(tail))
        self.assertNotIn("OLD TABLES", text)
        self.assertIn("<table>", text)                       # the summary tables
        self.assertIn("0.250", text)                         # CSV round trip keeps the values
        self.assertNotIn("### 3 h (pred_len 36)", text)      # full tables live elsewhere
        self.assertIn("results/LEADERBOARD.md", text)        # ... and the README says where

    def test_writes_the_full_tables(self):
        self.assertEqual(self._main(), 0)
        full = self.full.read_text(encoding="utf-8")
        self.assertIn("### 3 h (pred_len 36)", full)
        self.assertIn("| DLinear | 0.250 |", full)
        self.assertIn("input lengths", full)

    def test_rerun_is_idempotent(self):
        self._main()
        once = (self.readme.read_text(), self.full.read_text())
        self._main()
        self.assertEqual((self.readme.read_text(), self.full.read_text()), once)

    def test_missing_markers_fail_without_touching_the_readme(self):
        self.readme.write_text("# no markers here\n")
        self.assertEqual(self._main(), 1)
        self.assertEqual(self.readme.read_text(), "# no markers here\n")
        self.assertFalse(self.full.exists())

    def test_utf8_readme_under_an_ascii_locale(self):
        # REPRODUCE step 5 runs the script on README.md (em dashes, and the
        # leaderboard's dagger); a non-UTF-8 locale must not break the rewrite
        import os
        import subprocess
        import sys
        self.readme.write_text("# GOES \u2014 time series\n\n" + self.README, encoding="utf-8")
        env = {**os.environ, "LC_ALL": "C", "LANG": "C", "PYTHONCOERCECLOCALE": "0",
               "PYTHONUTF8": "0"}
        run = subprocess.run([sys.executable, "-X", "utf8=0",
                              str(_REPO / "scripts" / "make_leaderboard.py"),
                              "--regression", str(self.reg), "--event", str(self.evt),
                              "--readme", str(self.readme), "--full", str(self.full)],
                             capture_output=True, text=True, env=env, timeout=120)
        self.assertEqual(run.returncode, 0, run.stderr[-500:])
        text = self.readme.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("# GOES \u2014 time series"))
        self.assertIn("0.250", text)
        self.assertIn("\u2020", self.full.read_text(encoding="utf-8"))   # non-ASCII written

    def test_missing_table_file_fails(self):
        self.reg.unlink()
        self.assertEqual(self._main(), 1)
        self.assertEqual(self.readme.read_text(), self.README)


if __name__ == "__main__":
    unittest.main()
