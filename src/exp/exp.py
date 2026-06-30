from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd
import torch

from data.loader import DataModule
from model import build_model
from .lightning_model import pl_model


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

    rows = []
    for model_name in config.models:
        model = build_model(
            model_name=model_name,
            config=config,
            input_size=data_bundle.input_size,
            target_index=data_bundle.target_index,
        )
        runner = pl_model(model, config)

        ckpt_path = os.path.join(dirs["ckpt"], f"{model_name}.pt")
        result = runner.fit_and_test(data_bundle, model_name=model_name, ckpt_path=ckpt_path)
        rows.append(
            {
                "model": result.model_name,
                "best_val_loss": result.best_val_loss,
                "test_mse": result.test_mse,
                "test_mae": result.test_mae,
                "ckpt_path": result.ckpt_path,
            }
        )

    comparison = pd.DataFrame(rows).sort_values(by=["test_mse", "test_mae"]).reset_index(drop=True)
    comparison_path = os.path.join(dirs["score"], "comparison.csv")
    comparison.to_csv(comparison_path, index=False)
    return comparison, comparison_path
