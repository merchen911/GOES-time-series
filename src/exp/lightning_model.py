from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch import nn


@dataclass
class TrainResult:
    model_name: str
    best_val_loss: float
    test_mse: float
    test_mae: float
    ckpt_path: str


class pl_model:
    """
    legacy/2026의 lightning_model 명명/역할을 유지하되,
    최소 실행 프레임워크를 위해 순수 PyTorch 학습 루프를 제공.
    """

    def __init__(self, model: nn.Module, config) -> None:
        self.model = model
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )

    def _run_epoch(self, loader, train: bool) -> float:
        self.model.train(train)
        losses: List[float] = []
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            pred = self.model(x)
            loss = self.criterion(pred, y)
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        return float(np.mean(losses)) if losses else float("inf")

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        self.model.eval()
        all_pred, all_true = [], []
        for x, y in loader:
            pred = self.model(x.to(self.device)).cpu().numpy()
            all_pred.append(pred)
            all_true.append(y.numpy())
        pred = np.concatenate(all_pred, axis=0)
        true = np.concatenate(all_true, axis=0)
        return {
            "mse": float(np.mean((pred - true) ** 2)),
            "mae": float(np.mean(np.abs(pred - true))),
        }

    def fit_and_test(self, datamodule, model_name: str, ckpt_path: str) -> TrainResult:
        best_val = float("inf")
        for _ in range(self.config.epochs):
            self._run_epoch(datamodule.train_loader, train=True)
            val_loss = self._run_epoch(datamodule.val_loader, train=False)
            if val_loss < best_val:
                best_val = val_loss
                torch.save(self.model.state_dict(), ckpt_path)

        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        metrics = self.evaluate(datamodule.test_loader)
        return TrainResult(
            model_name=model_name,
            best_val_loss=best_val,
            test_mse=metrics["mse"],
            test_mae=metrics["mae"],
            ckpt_path=ckpt_path,
        )
