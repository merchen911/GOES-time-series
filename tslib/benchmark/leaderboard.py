"""The leaderboard, built from the two v2 result tables, in two forms.

`summary_html` is the README's summary: one regression table (RMSE) and one
classification table (per-day HSS) for the univariate tracks, with a two-row
header that groups the four forecast lengths under each channel. Forecast
lengths are never averaged together.

`leaderboard_markdown` is the full leaderboard (spec 4.9, D9): one section per
forecast length, one table per track, every shown metric.

A cell is the mean, over every fold and input length that ran at that forecast
length, of the run's own value. Runs whose value is NaN are left out of the
mean. The grid of a forecast length is every (fold, input length) pair that any
model of the track ran there; a cell shows `k/n` when only k of those n runs
have a value (`0.512 (14/15)`), whether the others are undefined or were never
run. A model with fewer runs than the grid is marked with a dagger and
footnoted under the section, because its means are not over the same inputs as
the other rows. The columns are the regression metrics of each channel of the
track and, per channel, the event scores at the `day` granularity; rows are
ordered by the first channel's rmse."""
from __future__ import annotations

from html import escape
from typing import Dict, FrozenSet, List, Tuple

import pandas as pd

from tslib.exp.records import RUN_KEYS

# pred_len -> section title (5-minute steps)
HORIZONS = {36: "3 h", 72: "6 h", 144: "12 h", 288: "24 h"}

# track -> (table heading, its channels in the order the driver forecasts them;
# the first one orders the rows)
TRACKS = {
    "uni_b": ("SXR 1--8 A (univariate)", ("xrs_long",)),
    "uni_a": ("Proton >=10 MeV (univariate)", ("p_gt10",)),
    "multi": ("Multivariate (joint proton + SXR)", ("p_gt10", "xrs_long")),
}
CHANNEL_LABEL = {"xrs_long": "SXR", "p_gt10": "Proton"}

# metric column -> header label; regression metrics first, then the day scores
REGRESSION_SHOWN = {"rmse": "RMSE", "msa": "MSA", "logr": "logR"}
EVENT_SHOWN = {"hss": "HSS", "tss": "TSS"}
EVENT_GRANULARITY = "day"

# (model, seq_len) runs left out of every mean: xPatch diverges at the 7-day
# input (v003, fold 0); its NaN rows stay in the result tables.
EXCLUDED = ("xpatch", 2016)

DISP = {"segrnn_thuml": "SegRNN", "xpatch": "xPatch", "itransformer": "iTransformer",
        "patchtst": "PatchTST", "tide": "TiDE", "rlinear": "RLinear",
        "patchmixer": "PatchMixer", "dlinear": "DLinear", "tsmixer": "TSMixer",
        "frets": "FReTS"}

# minutes per input step (the 5-minute grid); names the input lengths in notes
CADENCE_MIN = 5
SHORT_MARK = "\u2020"   # dagger: a model with fewer runs than its table's grid

# The README summary: the univariate tracks as (track, channel), and per table
# (title, which result table, metric, lower is better, channel group labels).
SUMMARY_TRACKS = (("uni_b", "xrs_long"), ("uni_a", "p_gt10"))
SUMMARY_TABLES = (
    ("Regression — RMSE of log10 flux (lower is better)", "regression", "rmse", True,
     ("SXR 1–8 Å", "Proton ≥10 MeV")),
    ("Classification — HSS, one forecast per day (higher is better)", "event", "hss",
     False, ("SXR ≥ M-class", "Proton ≥ 10 pfu")),
)
NO_VALUE = "–"     # en dash: no run at this forecast length

# header, per-model mean, per-model defined values, per-model rows
_Column = Tuple[str, pd.Series, pd.Series, pd.Series]
# (table heading, model, its (fold, seq_len) runs, the table's grid)
_Shortfall = Tuple[str, str, FrozenSet[Tuple[int, int]], FrozenSet[Tuple[int, int]]]


def leaderboard_markdown(reg: pd.DataFrame, evt: pd.DataFrame) -> str:
    """The leaderboard as Markdown: for each forecast length a `###` section
    that holds one table per track (tracks without runs are left out), then a
    footnote for every model that has fewer runs than its table's grid and for
    the runs of `EXCLUDED` that were left out.

    `reg` and `evt` are the regression and event tables (columns
    `records.regression_columns()` and `records.event_columns()`)."""
    model, seq_len = EXCLUDED
    dropped = reg[(reg["model"] == model) & (reg["seq_len"] == seq_len)]
    reg, evt = _without_excluded(reg), _without_excluded(evt)
    evt = evt[evt["granularity"] == EVENT_GRANULARITY]

    sections = []
    for pred_len, title in HORIZONS.items():
        parts = [f"### {title} (pred_len {pred_len})"]
        shortfalls: List[_Shortfall] = []
        listed: Dict[str, List[str]] = {}
        for track, (heading, channels) in TRACKS.items():
            built = _track_table(reg[reg["pred_len"] == pred_len],
                                 evt[evt["pred_len"] == pred_len], track, channels)
            if built is None:
                continue
            table, models, short = built
            parts.append(f"**{heading}**\n\n{table}")
            for m in models:
                listed.setdefault(m, []).append(heading)
            shortfalls += [(heading, m, runs, grid) for m, (runs, grid) in short.items()]
        n_dropped = len(dropped.loc[dropped["pred_len"] == pred_len, RUN_KEYS]
                        .drop_duplicates())
        parts += _footnotes(shortfalls, listed, n_dropped)
        sections.append("\n\n".join(parts))
    return "\n\n".join(sections)


def summary_html(reg: pd.DataFrame, evt: pd.DataFrame) -> str:
    """The README summary: for each of `SUMMARY_TABLES` a bold title and an HTML
    table of the univariate tracks, one column per channel and forecast length
    under a two-row header, all text small; then one footnote per model that
    has fewer runs than its column's grid.

    Rows are the models in alphabetical order of their shown names, and the
    best value of each column is bold. A cell is the mean, over the folds and
    input lengths at that forecast length, of the model's values, with `(k/n)`
    when only k of its n runs have one; `EXCLUDED` runs are left out."""
    reg, evt = _without_excluded(reg), _without_excluded(evt)
    frames = {"regression": reg,
              "event": evt[evt["granularity"] == EVENT_GRANULARITY]}
    parts, notes = [], []
    for title, which, metric, lower, groups in SUMMARY_TABLES:
        frame = frames[which]
        columns, short = [], set()
        for track, channel in SUMMARY_TRACKS:
            for pred_len in HORIZONS:
                sub = frame[(frame["track"] == track) & (frame["channel"] == channel)
                            & (frame["pred_len"] == pred_len)]
                values = sub[["model", metric]].astype({metric: float}).groupby("model")[metric]
                columns.append((values.mean(), values.count(), values.size()))
                runs = _runs(sub)
                grid = frozenset().union(*runs.values()) if runs else frozenset()
                for model, model_runs in runs.items():
                    if len(model_runs) < len(grid):
                        short.add(model)
                        note = _shortfall_note(model, model_runs, grid)
                        if note not in notes:
                            notes.append(note)
        models = sorted(set().union(*(size.index for _, _, size in columns)),
                        key=lambda m: DISP.get(m, m).lower())
        best = [mean.min() if lower else mean.max() for mean, _, _ in columns]
        rows = [(DISP.get(m, m) + (SHORT_MARK if m in short else ""),
                 [_summary_cell(m, column, b) for column, b in zip(columns, best)])
                for m in models]
        parts.append(f"**{title}**\n\n{_html_table(groups, rows)}")
    return "\n\n".join(parts + [f"_{SHORT_MARK} {note}._" for note in notes])


def _summary_cell(model: str, column, best: float) -> str:
    """`0.512`, `<b>0.512</b>` when it is the column's best, `0.512 (1/2)` when
    only 1 of the model's 2 runs has a value, `NO_VALUE` when it has no run."""
    mean, defined, rows = (s.get(model) for s in column)
    if not rows:
        return NO_VALUE
    if not defined:
        return f"{NO_VALUE} (0/{rows})"
    text = f"{mean:.3f}"
    if abs(mean - best) <= 1e-12:
        text = f"<b>{text}</b>"
    return text if defined == rows else f"{text} ({defined}/{rows})"


def _html_table(groups: Tuple[str, ...], rows: List[Tuple[str, List[str]]]) -> str:
    """An HTML table with a two-row header: `Model`, then each group spanning
    the forecast lengths. Every cell is wrapped in <sub> for small text (GitHub
    drops style attributes); `rows` holds a name and already formatted cells."""
    def small(text: str) -> str:
        return f"<sub>{text}</sub>"
    top = [f'<th rowspan="2" align="left">{small("Model")}</th>']
    top += [f'<th colspan="{len(HORIZONS)}">{small(escape(g))}</th>' for g in groups]
    second = [f"<th>{small(h)}</th>" for _ in groups for h in HORIZONS.values()]
    lines = ["<table>", "<thead>", "<tr>" + "".join(top) + "</tr>",
             "<tr>" + "".join(second) + "</tr>", "</thead>", "<tbody>"]
    for name, cells in rows:
        lines.append(f"<tr><td>{small(escape(name))}</td>"
                     + "".join(f'<td align="right">{small(c)}</td>' for c in cells)
                     + "</tr>")
    lines += ["</tbody>", "</table>"]
    return "\n".join(lines)


def _without_excluded(frame: pd.DataFrame) -> pd.DataFrame:
    model, seq_len = EXCLUDED
    return frame[~((frame["model"] == model) & (frame["seq_len"] == seq_len))]


def _runs(frame: pd.DataFrame) -> Dict[str, FrozenSet[Tuple[int, int]]]:
    """model -> its (fold, seq_len) runs in `frame`."""
    return {m: frozenset(zip(g["fold"].astype(int), g["seq_len"].astype(int)))
            for m, g in frame.groupby("model")}


def _track_table(reg: pd.DataFrame, evt: pd.DataFrame, track: str, channels: Tuple[str, ...]):
    """One track's table at one forecast length as `(markdown, models, short)`,
    where `short` maps each model with fewer runs than the grid to `(its runs,
    the grid)`; None when the track has no runs."""
    reg, evt = reg[reg["track"] == track], evt[evt["track"] == track]
    keys = ["model", "fold", "seq_len"]
    runs = _runs(pd.concat([reg[keys], evt[keys]]))
    models = sorted(runs)
    if not models:
        return None
    grid = frozenset().union(*runs.values())
    short = {m: (r, grid) for m, r in runs.items() if len(r) < len(grid)}
    columns: List[_Column] = []
    for channel in channels:
        label = CHANNEL_LABEL[channel]
        for frame, shown in ((reg, REGRESSION_SHOWN), (evt, EVENT_SHOWN)):
            sub = frame[frame["channel"] == channel]
            grouped = sub[["model", *shown]].astype({m: float for m in shown}).groupby("model")
            mean, defined, rows = grouped.mean(), grouped.count(), grouped.size()
            for metric, name in shown.items():
                columns.append((f"{label} {name}",
                                mean[metric].reindex(models),
                                defined[metric].reindex(models, fill_value=0),
                                rows.reindex(models, fill_value=0)))

    rmse = columns[0][1]   # the first channel's; an undefined mean sorts last
    models.sort(key=lambda m: (pd.isna(rmse[m]), 0.0 if pd.isna(rmse[m]) else rmse[m], m))
    lines = ["| " + " | ".join(["Model"] + [c[0] for c in columns]) + " |",
             "|" + "|".join(["---"] * (len(columns) + 1)) + "|"]
    for model in models:
        cells = [_cell(mean[model], defined[model], rows[model], len(grid))
                 for _, mean, defined, rows in columns]
        name = DISP.get(model, model) + (SHORT_MARK if model in short else "")
        lines.append("| " + " | ".join([name, *cells]) + " |")
    return "\n".join(lines), models, short


def _cell(mean: float, defined: int, rows: int, grid: int) -> str:
    """`0.512`, or `0.512 (14/15)` when only 14 of the grid's 15 runs have a
    value; `--` when no run does (and no count when the model has no row for
    this column at all)."""
    if rows == 0:
        return "--"
    text = "--" if pd.isna(mean) else f"{mean:.3f}"
    return text if defined == grid else f"{text} ({defined}/{grid})"


def _input(seq_len: int) -> str:
    """`1-day`, `7-day` or `6-hour`: an input length in words."""
    minutes = int(seq_len) * CADENCE_MIN
    if minutes % 1440 == 0:
        return f"{minutes // 1440}-day"
    return f"{minutes / 60:g}-hour"


def _count(n: int, noun: str = "run") -> str:
    return f"{n} {noun}" + ("" if n == 1 else "s")


def _and(items: List[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _shortfall_note(model: str, runs, grid) -> str:
    """Why `model` has fewer runs than the grid, and what its cells average."""
    name = DISP.get(model, model)
    have = sorted({s for _, s in runs})
    gaps: Dict[int, List[int]] = {}
    for fold, s in sorted(grid - runs):
        gaps.setdefault(s, []).append(fold)
    whole = [s for s in gaps if s not in have]
    if model == EXCLUDED[0] and whole == [EXCLUDED[1]] and len(gaps) == 1:
        lead = (f"{name} is left out at the {_input(EXCLUDED[1])} input (seq_len "
                f"{EXCLUDED[1]}) because its training diverges there")
    else:
        bits = [f"the {_input(s)} input (seq_len {s})" for s in whole]
        bits += [f"fold{'s' if len(f) > 1 else ''} {_and([str(x) for x in f])} of the "
                 f"{_input(s)} input" for s, f in gaps.items() if s not in whole]
        lead = f"{name} has no runs at {_and(bits)}"
    inputs = _and([_input(s) for s in have])
    return (f"{lead}, so its cells average {_count(len(runs))} ({inputs} "
            f"input{'s' if len(have) > 1 else ''}) instead of all {len(grid)} runs "
            f"of this forecast length and are not directly comparable with the "
            f"other rows")


def _footnotes(shortfalls: List[_Shortfall], listed: Dict[str, List[str]],
               n_dropped: int) -> List[str]:
    """One footnote per marked model (naming the tables when the model is short
    in only some of them, or short differently), plus the left-out runs of
    `EXCLUDED`, which join that model's note when it has one."""
    model_ex, seq_ex = EXCLUDED
    notes: Dict[str, Dict[str, List[str]]] = {}
    for heading, model, runs, grid in shortfalls:
        notes.setdefault(model, {}).setdefault(_shortfall_note(model, runs, grid),
                                               []).append(heading)
    dropped = (f"The {_count(n_dropped, f'diverged {_input(seq_ex)} run')} at this "
               f"forecast length {'stays' if n_dropped == 1 else 'stay'} in the result "
               f"tables as NaN rows")
    out = []
    for model in sorted(notes, key=lambda m: DISP.get(m, m)):
        texts = notes[model]
        for text, headings in texts.items():
            if len(texts) > 1 or headings != listed.get(model, []):
                text += f" (table{'s' if len(headings) > 1 else ''} {_and(headings)})"
            if model == model_ex and n_dropped:
                text += f". {dropped}"
            out.append(f"_{SHORT_MARK} {text}._")
    if n_dropped and model_ex not in notes:
        out.append(f"_{DISP[model_ex]} runs with input length {seq_ex} ({_input(seq_ex)}) "
                   f"are left out of the means above, because its training diverges "
                   f"there ({_count(n_dropped)})._")
    return out
