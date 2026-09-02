#!/usr/bin/env python3.12
"""Generate the static benchmark leaderboard from results/results_master.csv and
inject it into README.md between the LEADERBOARD markers.

Leaderboard = mean over the 5-fold rotating split and all input-length/horizon
cells, per model and track; ranked by mean log-flux RMSE (lower is better).
Event skill is mean HSS / TSS at the operational thresholds.

Run from the repo root:  python3.12 scripts/make_leaderboard.py
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CSV = os.path.join(REPO, "results", "results_master.csv")
README = os.path.join(REPO, "README.md")
START, END = "<!-- LEADERBOARD:START -->", "<!-- LEADERBOARD:END -->"

DISP = {"segrnn_thuml": "SegRNN", "xpatch": "xPatch", "itransformer": "iTransformer",
        "patchtst": "PatchTST", "tide": "TiDE", "rlinear": "RLinear",
        "patchmixer": "PatchMixer", "dlinear": "DLinear", "tsmixer": "TSMixer",
        "frets": "FReTS"}
# track -> (heading, event-skill channels shown as [(label, hss_col, tss_col), ...])
TRACKS = {
    "uni_b": ("SXR 1--8 A (univariate)", [("SXR", "hss_xrs_long", "tss_xrs_long")]),
    "uni_a": ("Proton >=10 MeV (univariate)", [("Proton", "hss_p_gt10", "tss_p_gt10")]),
    "multi": ("Multivariate (joint proton + SXR)",
              [("SXR", "hss_xrs_long", "tss_xrs_long"), ("Proton", "hss_p_gt10", "tss_p_gt10")]),
}


def fmt(x):
    return "--" if pd.isna(x) else f"{x:.3f}"


def table(df, track, heading, channels):
    sub = df[df.track == track]
    hdr = ["Model", "RMSE (log-flux)"]
    for lab, _, _ in channels:
        hdr.append(f"{lab} HSS / TSS")
    lines = [f"**{heading}**", "", "| " + " | ".join(hdr) + " |",
             "|" + "|".join(["---"] * len(hdr)) + "|"]
    agg = sub.groupby("model").mean(numeric_only=True)
    agg = agg.sort_values("rmse")  # NaN RMSE sinks to the bottom
    for model, row in agg.iterrows():
        cells = [DISP.get(model, model), fmt(row["rmse"])]
        for _, hcol, tcol in channels:
            cells.append(f"{fmt(row.get(hcol))} / {fmt(row.get(tcol))}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    df = pd.read_csv(CSV)
    df = df[df["skipped"] != True]  # noqa: E712 -- keep completed runs only
    blocks = [table(df, t, h, ch) for t, (h, ch) in TRACKS.items()]
    body = ("\n_Mean over the 5-fold rotating split and all input-length/horizon "
            "cells; lower RMSE is better. Full per-configuration results in "
            "[`results/results_master.csv`](results/results_master.csv)._\n\n"
            + "\n\n".join(blocks) + "\n")
    text = open(README).read()
    assert START in text and END in text, "LEADERBOARD markers missing from README.md"
    pre, rest = text.split(START, 1)
    _, post = rest.split(END, 1)
    open(README, "w").write(f"{pre}{START}\n{body}\n{END}{post}")
    print("leaderboard written to README.md")


if __name__ == "__main__":
    main()
