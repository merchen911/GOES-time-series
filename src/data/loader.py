from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def _read_table(path, columns=None):
    if str(path).endswith(".parquet"):
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path)


def _valid_starts(valid, L):
    """int64 positions i where valid[i:i+L] is entirely True."""
    n = len(valid) - L + 1
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    invalid = (~np.asarray(valid)).astype(np.int64)
    csum = np.concatenate([[0], np.cumsum(invalid)])
    cnt = csum[L:L + n] - csum[:n]
    return np.nonzero(cnt == 0)[0].astype(np.int64)


def _term_labels(index, split_type):
    year = index.year.to_numpy().astype("U4")
    if split_type == "year":
        return year
    if split_type == "year_half":
        half = np.where(index.month.to_numpy() <= 6, "H1", "H2")
        return np.char.add(np.char.add(year, "-"), half)
    raise ValueError(f"Unsupported split_type: {split_type}")


def _prepare_series(df, time_col, target_col, role, transform, role_col="role"):
    sub = df
    if role is not None and role_col in df.columns:
        sub = df[df[role_col] == role]
    sub = sub[[time_col, target_col]].copy()
    sub[time_col] = pd.to_datetime(sub[time_col])
    sub = (sub.dropna(subset=[time_col])
              .drop_duplicates(time_col, keep="last")
              .sort_values(time_col))
    s = sub.set_index(time_col)[target_col].astype("float64")
    if transform == "log10":
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.log10(s.where(s > 0))
    elif transform != "none":
        raise ValueError(f"unknown transform: {transform}")
    return s


def _grid_and_starts(series, terms, cadence_min, seq_len, pred_len, split_type):
    L = seq_len + pred_len
    step = pd.Timedelta(minutes=cadence_min)
    labels = _term_labels(series.index, split_type)
    all_vals, all_starts, offset = [], [], 0
    for term in sorted(terms):
        sub = series[labels == term]
        if sub.empty:
            continue
        grid = pd.date_range(sub.index.min(), sub.index.max(), freq=step)
        g = sub.reindex(grid).to_numpy(dtype="float64")
        starts = _valid_starts(~np.isnan(g), L)
        all_vals.append(g)
        if len(starts):
            all_starts.append(starts + offset)
        offset += len(g)
    values = np.concatenate(all_vals) if all_vals else np.empty(0, dtype="float64")
    starts = np.concatenate(all_starts) if all_starts else np.empty(0, dtype=np.int64)
    return values, starts


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    input_size: int
    target_index: int


def _build_windows(values: np.ndarray, seq_len: int, pred_len: int) -> Tuple[np.ndarray, np.ndarray]:
    total = seq_len + pred_len
    if len(values) < total:
        raise ValueError(f"rows={len(values)} is too short for seq_len+pred_len={total}")
    xs, ys = [], []
    for i in range(len(values) - total + 1):
        window = values[i : i + total]
        xs.append(window[:seq_len, :])
        ys.append(window[-pred_len:, :])
    return np.stack(xs), np.stack(ys)


def _split_indices(n: int, train_ratio: float, val_ratio: float) -> Dict[str, slice]:
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return {
        "train": slice(0, train_end),
        "val": slice(train_end, val_end),
        "test": slice(val_end, n),
    }


def _add_term_column(df: pd.DataFrame, time_col: str, split_type: str) -> pd.DataFrame:
    copied = df.copy()
    copied[time_col] = pd.to_datetime(copied[time_col])
    if split_type == "year":
        copied["_term"] = copied[time_col].dt.year.astype(str)
    elif split_type == "year_half":
        year = copied[time_col].dt.year.astype(str)
        half = np.where(copied[time_col].dt.month <= 6, "H1", "H2")
        copied["_term"] = year + "-" + half
    else:
        raise ValueError(f"Unsupported split_type for k-fold: {split_type}")
    return copied


def _fold_indices(n_term: int, n_fold: int, fold_numb: int) -> Dict[str, np.ndarray]:
    term_range = np.arange(n_term)
    vl_idx = term_range[(n_fold - 2 + fold_numb) % n_fold :: n_fold]
    ts_idx = term_range[(n_fold - 1 + fold_numb) % n_fold :: n_fold]
    tr_idx = np.setdiff1d(term_range, np.concatenate([vl_idx, ts_idx]))
    return {"train": tr_idx, "val": vl_idx, "test": ts_idx}


class DataModule:
    """legacy/2026의 DataModule 흐름을 단순화해 반영한 모듈."""

    def __init__(self, config) -> None:
        self.config = config

    def setup(self) -> DataBundle:
        df = pd.read_csv(self.config.data_path)
        if self.config.target_col not in df.columns:
            raise ValueError(f"target_col '{self.config.target_col}' not found in data.")

        if self.config.time_col:
            if self.config.time_col not in df.columns:
                raise ValueError(f"time_col '{self.config.time_col}' not found in data.")
            df[self.config.time_col] = pd.to_datetime(df[self.config.time_col])
            df = df.sort_values(self.config.time_col).reset_index(drop=True)

        if self.config.feature_cols is None:
            exclude = {self.config.target_col}
            if self.config.time_col:
                exclude.add(self.config.time_col)
            feature_cols = [c for c in df.columns if c not in exclude]
        else:
            feature_cols = self.config.feature_cols

        use_cols = feature_cols + [self.config.target_col]
        if self.config.split_type == "ratio":
            values = df[use_cols].to_numpy(dtype=np.float32)
            x, y = _build_windows(values, self.config.seq_len, self.config.pred_len)
            y = y[:, :, -1:]  # target만 사용

            splits = _split_indices(len(x), self.config.train_ratio, self.config.val_ratio)
            tr = SequenceDataset(x[splits["train"]], y[splits["train"]])
            va = SequenceDataset(x[splits["val"]], y[splits["val"]])
            te = SequenceDataset(x[splits["test"]], y[splits["test"]])
        else:
            # legacy/2026 방식: term(year/year_half) 단위 회전식 k-fold
            term_df = _add_term_column(df, self.config.time_col, self.config.split_type)
            terms = sorted(term_df["_term"].dropna().unique().tolist())
            if len(terms) < self.config.n_fold:
                raise ValueError(
                    f"Not enough terms for n_fold={self.config.n_fold}. "
                    f"Current term count={len(terms)}"
                )
            fold_map = _fold_indices(len(terms), self.config.n_fold, self.config.fold_numb)

            x_parts = {"train": [], "val": [], "test": []}
            y_parts = {"train": [], "val": [], "test": []}
            split_terms = {
                k: [terms[i] for i in idxs.tolist()] for k, idxs in fold_map.items()
            }

            for split_name, split_term_values in split_terms.items():
                for term_v in split_term_values:
                    sub = term_df.loc[term_df["_term"] == term_v]
                    values = sub[use_cols].to_numpy(dtype=np.float32)
                    total = self.config.seq_len + self.config.pred_len
                    if len(values) < total:
                        continue
                    x_sub, y_sub = _build_windows(values, self.config.seq_len, self.config.pred_len)
                    y_sub = y_sub[:, :, -1:]
                    x_parts[split_name].append(x_sub)
                    y_parts[split_name].append(y_sub)

            def _concat_or_empty(parts):
                if parts:
                    return np.concatenate(parts, axis=0)
                return np.empty((0, self.config.seq_len, len(use_cols)), dtype=np.float32)

            x_tr = _concat_or_empty(x_parts["train"])
            x_va = _concat_or_empty(x_parts["val"])
            x_te = _concat_or_empty(x_parts["test"])

            def _concat_y_or_empty(parts):
                if parts:
                    return np.concatenate(parts, axis=0)
                return np.empty((0, self.config.pred_len, 1), dtype=np.float32)

            y_tr = _concat_y_or_empty(y_parts["train"])
            y_va = _concat_y_or_empty(y_parts["val"])
            y_te = _concat_y_or_empty(y_parts["test"])

            tr = SequenceDataset(x_tr, y_tr)
            va = SequenceDataset(x_va, y_va)
            te = SequenceDataset(x_te, y_te)

        return DataBundle(
            train_loader=DataLoader(
                tr,
                batch_size=self.config.batch_size,
                shuffle=self.config.shuffle_train,
                num_workers=self.config.num_workers,
            ),
            val_loader=DataLoader(
                va, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers
            ),
            test_loader=DataLoader(
                te, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers
            ),
            input_size=len(use_cols),       
            target_index=len(use_cols) - 1,
        )
