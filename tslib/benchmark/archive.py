"""Dump test predictions from the checkpoints of an existing run tree (spec 4.7).

The v003/v004 sweeps kept every trained model as ``<bench>/<cell>/ckpt/<model>.ckpt``
but no prediction files. This module rebuilds each cell's run settings from the
driver's cell definition, runs the same test-only inference pass the resume path
uses (Lightning ``trainer.test``, batch 64, one device) and writes
``<out_bench>/<cell>/pred/<model>.npz`` plus ``score/rows/<model>.json``, so the
old runs can be re-scored without a GPU from then on. The source tree is only
read. The work list comes from the checkpoints on disk, not from the driver
grid (v003's grid has 900 runs, its tree 876 checkpoints)."""
from __future__ import annotations

import argparse
import os
import re
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

from tslib.configs.config import config_postprocess, exp_parser
from tslib.data.loader import DataModule
from tslib.exp.exp import _row_path, _write_row, save_dir_name
from tslib.exp.predictions import prediction_path
from tslib.exp.records import result_record
from tslib.exp.strategy import test_only_neural

from .driver import cell_argv, parse_cell_name, run_name_for

# Lightning's versioned siblings of a checkpoint ("<model>-v1.ckpt"): written
# when the canonical name was taken, never the model the run recorded.
_VERSIONED = re.compile(r".+-v\d+")

# the old runs' loader count was 12 (v003) or 8 (v004); results do not depend on it
_NUM_WORKERS = 8


def cell_name(cell: dict) -> str:
    """The cell's run directory name (``run_name_for`` without ``bench/``)."""
    return run_name_for(cell["track"], cell["seq_len"], cell["pred_len"],
                        cell["fold"], cell["strategy"]).removeprefix("bench/")


def _sort_key(cell: dict, model: str):
    return (cell["fold"], cell["seq_len"], cell["pred_len"], cell["track"],
            cell["strategy"], model)


def discover_checkpoints(src_bench: Path) -> List[Tuple[dict, str, Path]]:
    """Every ``src_bench/<cell>/ckpt/<model>.ckpt`` as ``(cell, model, file)``,
    ordered by fold, seq_len, pred_len, track, strategy and model. A folder is a
    cell when ``parse_cell_name`` accepts its name; versioned ``-vN`` siblings
    are not checkpoints of a run."""
    found = []
    for ckpt in Path(src_bench).glob("*/ckpt/*.ckpt"):
        if _VERSIONED.fullmatch(ckpt.stem):
            continue
        try:
            cell = parse_cell_name(ckpt.parents[1].name)
        except ValueError:
            continue
        found.append((cell, ckpt.stem, ckpt))
    return sorted(found, key=lambda f: _sort_key(f[0], f[1]))


def group_cells(found: List[Tuple[dict, str, Path]]
                ) -> List[Tuple[dict, List[Tuple[str, Path]]]]:
    """``discover_checkpoints`` output as ``(cell, [(model, file), ...])`` per
    cell, in the order the cells first appear."""
    cells: Dict[str, Tuple[dict, list]] = {}
    for cell, model, ckpt in found:
        cells.setdefault(cell_name(cell), (cell, []))[1].append((model, ckpt))
    return list(cells.values())


def parse_shard(text: str) -> Tuple[int, int]:
    """``"i/n"`` -> ``(i, n)`` with ``0 <= i < n``; ValueError otherwise."""
    parts = text.split("/")
    try:
        index, count = (int(p) for p in parts)
    except ValueError:
        raise ValueError(f"shard must be i/n, e.g. 0/4; got {text!r}") from None
    if not 0 <= index < count:
        raise ValueError(f"shard {text!r} needs 0 <= i < n")
    return index, count


def select_shard(cells: list, index: int, count: int) -> list:
    """Shard ``index`` of ``count``: the ``k``-th cell when ``k % count == index``.
    Run one shard per GPU and every cell is dumped exactly once."""
    return [c for k, c in enumerate(cells) if k % count == index]


def check_out_bench(out_bench: Path) -> None:
    """The output tree must end in a folder named ``bench``: the run name of a
    cell is ``bench/<cell>`` under ``runs_root = out_bench.parent``, so any
    other name would write the cells beside ``out_bench``, not inside it."""
    if Path(out_bench).name != "bench":
        raise ValueError(f"the output folder must be named 'bench' (cells are "
                         f"written to <runs_root>/bench/<cell>); got {out_bench}")


def archive_config(cell: dict, models: List[str], out_bench: Path) -> argparse.Namespace:
    """The run settings of ``cell`` as the sweep driver passed them to
    ``main.py``, with the output under ``out_bench/<cell>/``."""
    out_bench = Path(out_bench)
    check_out_bench(out_bench)
    argv = cell_argv(cell, epochs=1, models=list(models), runs_root=out_bench.parent)
    config = config_postprocess(exp_parser().parse_args(argv))
    config.num_workers = _NUM_WORKERS
    config.save_predictions = True
    return config


def dump_cell(cell: dict, items: List[Tuple[str, Path]], out_bench: Path) -> List[str]:
    """Test every ``(model, checkpoint file)`` of ``cell`` once and write its
    prediction file and result row under ``out_bench/<cell>/``. The data is
    loaded once per cell, and not at all when every model is already dumped.
    A model with both files on disk is skipped, so a dump resumes where it
    stopped. Returns the models dumped now.

    A model that fails (or whose prediction file was not written) does not stop
    the others; once all were tried, a RuntimeError names the failed ones and
    no row is written for them."""
    config = archive_config(cell, [m for m, _ in items], out_bench)
    dirs = save_dir_name(config)
    name = cell_name(cell)
    pending = [(m, f) for m, f in items
               if not (os.path.exists(prediction_path(dirs["root"], m))
                       and os.path.exists(_row_path(dirs, m)))]
    if not pending:
        return []
    data_bundle = DataModule(config).setup()
    done, failed = [], {}
    for model, ckpt_file in pending:
        pred_path = prediction_path(dirs["root"], model)
        try:
            result = test_only_neural(cell["strategy"], model, data_bundle, config,
                                      str(ckpt_file), str(ckpt_file),
                                      pred_path=pred_path)
            if not os.path.exists(pred_path):
                raise RuntimeError(f"prediction file {pred_path} was not written")
        except Exception as e:  # noqa: BLE001 - one bad checkpoint must not stop the cell
            traceback.print_exc()
            failed[model] = repr(e)
            continue
        _write_row(_row_path(dirs, model), result_record(result))
        done.append(model)
        print(f"[dump] {name} {model}: written to {pred_path}", flush=True)
    if failed:
        raise RuntimeError(f"{name}: {len(failed)} of {len(pending)} models failed "
                           f"({len(done)} dumped): {failed}")
    return done
