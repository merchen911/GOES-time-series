from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd
import torch

from tslib.data.loader import DataModule
from .strategy import run_strategy


def build_comparison(results, sort_metric):
    rows = [{"strategy": r.strategy, "model": r.model_name,
             "best_val_loss": r.best_val_loss,
             "skipped": r.skipped, "est_train_hours": r.est_train_hours,
             **r.metrics}
            for r in results]
    df = pd.DataFrame(rows)
    key = sort_metric if sort_metric in df.columns else "best_val_loss"
    return df.sort_values(by=key, na_position="last").reset_index(drop=True)


def merge_comparisons(frames, sort_metric="best_val_loss"):
    df = pd.concat(list(frames), ignore_index=True)
    key = sort_metric if sort_metric in df.columns else "best_val_loss"
    return df.sort_values(by=key, na_position="last").reset_index(drop=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_dir_name(config, makedir: bool = False):
    run_root = os.path.join("runs", config.run_name)
    dir_types = {
        "root": run_root,
        "ckpt": os.path.join(run_root, "ckpt"),
        "score": os.path.join(run_root, "score"),
    }
    if makedir:
        for p in dir_types.values():
            os.makedirs(p, exist_ok=True)
    return dir_types


def run_experiment(config):
    _set_seed(config.seed)
    dirs = save_dir_name(config, makedir=True)

    data_module = DataModule(config)
    data_bundle = data_module.setup()

    strategy = getattr(config, "forecast_strategy", "direct")
    results = []
    for model_name in config.models:
        ckpt_path = os.path.join(dirs["ckpt"], f"{model_name}.pt")
        results.append(run_strategy(strategy, model_name, data_bundle,
                                    config, ckpt_path))

    comparison = build_comparison(results, config.sort_metric)
    for r in results:
        if getattr(r, "skipped", False):
            print(f"SKIPPED (too slow): {r.model_name} "
                  f"~{r.est_train_hours:.2f}h — re-include with --on_slow proceed")
    comparison_path = os.path.join(dirs["score"], "comparison.csv")
    comparison.to_csv(comparison_path, index=False)
    return comparison, comparison_path
