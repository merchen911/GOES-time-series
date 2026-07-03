"""Generate a LaTeX (+ later PDF) report of the trainable-sample tables.

Numbers are recomputed live via count_fold_samples so the document never drifts
from the data. Output path is set by SW_TABLES_OUT (default: ./fold_sample_tables.tex).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from tslib.preprocessing import count_fold_samples as cfs
from tslib.preprocessing import term_split as ts

DAY = cfs.DAY
OUT = os.environ.get("SW_TABLES_OUT", "fold_sample_tables.tex")
HIST_DAYS = [7, 5, 3, 2, 1]
PRED_DAYS = [1, 3]
FOCUS_HIST, FOCUS_PRED = 3, 1  # recommended config for the per-fold table


def fnum(x: int) -> str:
    return f"{int(x):,}"


def run_stats(valid: np.ndarray) -> dict:
    padded = np.concatenate([[0], valid.astype(np.int8), [0]])
    d = np.diff(padded)
    rl = np.where(d == -1)[0] - np.where(d == 1)[0]
    return {"valid_pct": valid.mean() * 100,
            "median_d": np.median(rl) / DAY, "p90_d": np.percentile(rl, 90) / DAY,
            "max_d": rl.max() / DAY, "ge8d": int((rl >= 8 * DAY).sum())}


def build():
    # load validity once per config
    tv = {name: cfs.load_validity(ch) for name, ch in cfs.CONFIGS.items()}
    names = list(cfs.CONFIGS)  # particle univar / xray univar / multivar

    # --- per-config concatenated validity + run stats (univariate rows only) ---
    stats = {n: run_stats(np.concatenate(list(tv[n].values()))) for n in names}

    # --- sweep totals: name -> {(h,p): total} ---
    sweep = {n: {} for n in names}
    for n in names:
        for h in HIST_DAYS:
            for p in PRED_DAYS:
                L = int(round((h + p) * DAY))
                sweep[n][(h, p)] = sum(cfs.windows_in_run(v, L)
                                       for v in tv[n].values())

    # --- per-fold table at focus config ---
    Lf = int(round((FOCUS_HIST + FOCUS_PRED) * DAY))
    perfold = {n: cfs.fold_split_counts(tv[n], Lf) for n in names}

    L = []
    A = L.append
    A(r"\documentclass[11pt]{article}")
    A(r"\usepackage[margin=2.2cm,a4paper]{geometry}")
    A(r"\usepackage{booktabs}\usepackage{multirow}\usepackage{array}")
    A(r"\usepackage{caption}\captionsetup{font=small}")
    A(r"\title{KASI SWPC GOES: Strict Time-Series Splitting and Trainable-Sample Counts}")
    A(r"\author{Space Weather AI Framework -- preprocessing}")
    A(r"\date{Generated automatically; data v02 (regular 5-min grid)}")
    A(r"\begin{document}\maketitle")

    A(r"\section*{Setup}")
    A(r"Each timestamp is tagged with a half-year \emph{term} "
      r"(\texttt{year\_half}: H1=Jan--Jun, H2=Jul--Dec) derived in memory from "
      r"the parquet \texttt{time\_utc}. Terms are split into train/val/test by a "
      r"rotating 5-fold (val=$\mathrm{terms}[(n_f{-}2{+}i)\bmod n_f::n_f]$, "
      r"test=$[(n_f{-}1{+}i)\bmod n_f::n_f]$, train=rest), so a whole half-year "
      r"goes entirely to one split (leakage-free at the boundary). "
      r"The \texttt{primary}-role series is used. "
      r"Cadence is 5-min, i.e.\ $288$ steps/day $=24\times12$ "
      r"(legacy hourly $\times12$). A \emph{trainable sample} is one sliding "
      r"window of length $L=(\mathrm{hist}+\mathrm{pred})\times288$ that lies "
      r"within a single term and has \emph{no} missing value across the whole "
      r"window (legacy no-gap rule). For the multivariate config the two channels "
      r"are aligned on a common 5-min grid and \emph{all} channels must be present.")

    # Table 1: conversion
    A(r"\begin{table}[h]\centering\caption{Cadence and window-length conversion.}")
    A(r"\begin{tabular}{lrr}\toprule")
    A(r"Span & Hourly (legacy) & 5-min (current, $\times12$)\\\midrule")
    A(r"1 day & 24 & 288\\")
    A(r"history 7\,d & 168 & 2016\\")
    A(r"history 3\,d & 72 & 864\\")
    A(r"forecast 1\,d & 24 & 288\\")
    A(r"forecast 3\,d & 72 & 864\\\bottomrule")
    A(r"\end{tabular}\end{table}")

    # Table 2: availability (univariate series + multivar valid frac)
    A(r"\begin{table}[h]\centering\caption{Data availability of the "
      r"\texttt{primary} series on the regular 5-min grid (term-wise). "
      r"Contiguous-run lengths in days.}")
    A(r"\begin{tabular}{lrrrrr}\toprule")
    A(r"Series & valid \% & median run & p90 run & max run & runs $\ge$8\,d\\\midrule")
    for n in names:
        s = stats[n]
        A(f"{tex_name(n)} & {s['valid_pct']:.1f} & {s['median_d']:.2f} & "
          f"{s['p90_d']:.1f} & {s['max_d']:.1f} & {s['ge8d']} \\\\")
    A(r"\bottomrule\end{tabular}\end{table}")

    # Table 3: sweep totals
    A(r"\begin{table}[h]\centering\caption{Total trainable windows by history "
      r"and forecast length (whole dataset, summed over terms). Sample count "
      r"depends only on $L=\mathrm{hist}+\mathrm{pred}$.}")
    A(r"\small\begin{tabular}{l" + "rr" * len(names) + r"}\toprule")
    A(" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{tex_name(n)}}}" for n in names)
      + r"\\")
    A(" hist & " + " & ".join("pred\\,1d & pred\\,3d" for _ in names) + r"\\\midrule")
    for h in HIST_DAYS:
        cells = []
        for n in names:
            for p in PRED_DAYS:
                cells.append(fnum(sweep[n][(h, p)]))
        A(f"{h}\\,d & " + " & ".join(cells) + r"\\")
    A(r"\bottomrule\end{tabular}\end{table}")

    # Table 4: per-fold at focus
    A(rf"\begin{{table}}[h]\centering\caption{{Per-fold train/val/test trainable "
      rf"windows at history {FOCUS_HIST}\,d + forecast {FOCUS_PRED}\,d "
      rf"($L={Lf}$, {Lf//DAY}\,d no-gap). Rotating 5-fold; totals per fold are "
      rf"constant by construction.}}")
    A(r"\small\begin{tabular}{l" + "rrr" * len(names) + r"}\toprule")
    A(" & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{tex_name(n)}}}" for n in names)
      + r"\\")
    A(" fold & " + " & ".join("train & val & test" for _ in names) + r"\\\midrule")
    for k in range(5):
        cells = []
        for n in names:
            r = perfold[n].iloc[k]
            cells += [fnum(r["train"]), fnum(r["val"]), fnum(r["test"])]
        A(f"{k} & " + " & ".join(cells) + r"\\")
    A(r"\midrule total & "
      + " & ".join(f"\\multicolumn{{3}}{{c}}{{{fnum(perfold[n]['total'].iloc[0])}}}"
                   for n in names) + r"\\")
    A(r"\bottomrule\end{tabular}\end{table}")

    A(r"\end{document}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {OUT} ({len(L)} lines)")


def tex_name(n: str) -> str:
    return (n.replace("_", r"\_").replace(">=", r"$\ge$")
            .replace("(", r"(\,").replace(")", r"\,)"))


if __name__ == "__main__":
    build()
