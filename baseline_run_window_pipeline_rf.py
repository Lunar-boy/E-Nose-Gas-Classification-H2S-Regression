#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline pipeline for all_gases_framewise.csv

Key design choices (leakage-aware):
- Split is done at RUN level (no frame leakage across train/test).
- Window augmentation is done *within each run* and (by default) within 3 phases:
    exposure:      [t_s, t_e]           (60s)
    desorb_early:  [t_e, t_e + 60s]
    desorb_mid:    [t_e + 60s, t_e + 120s]
  with sliding windows: 30s window, 10s stride (=> 4 windows per phase, 12 per run).

Models:
- Classical: SVM (RBF), KNN, Random Forest
- Deep: 1D-CNN, LSTM (PyTorch)

Outputs:
- Prints mean±std of (macro) precision/recall/F1 and accuracy over repeated run-level splits
  for both window-level and run-level (majority-vote) evaluation.
- Saves aggregated confusion matrices to PNG under --outdir.

Usage:
  python baseline_run_window_pipeline.py --csv all_gases_framewise.csv

Tip (HPC):
  Use --no_deep if you only want SVM/KNN quickly.
"""
from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except Exception as e:
    torch = None  # type: ignore
    nn = None  # type: ignore
    Dataset = object  # type: ignore
    DataLoader = object  # type: ignore


# -----------------------------
# Configuration
# -----------------------------
@dataclass
class Config:
    window_sec: float = 30.0
    stride_sec: float = 10.0
    phase_mode: str = "3phase"  # "3phase" or "fullrun"
    n_splits: int = 5
    test_size: float = 0.20
    seed: int = 42

    # Resampling
    n_steps: Optional[int] = None  # if None, infer from median dt
    min_points_in_raw_window: int = 10  # skip windows with fewer raw points

    # Classical models
    knn_k: int = 7
    svm_c: float = 10.0
    svm_gamma: str = "scale"

    rf_estimators: int = 500
    rf_max_depth: Optional[int] = None
    rf_min_samples_leaf: int = 1
    rf_max_features: str = "sqrt"
    rf_n_jobs: int = -1

    # Deep models
    use_deep: bool = True
    batch_size: int = 64
    max_epochs: int = 40
    patience: int = 7
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "auto"  # "auto", "cpu", "cuda"


# -----------------------------
# Utilities
# -----------------------------
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = False  # type: ignore


def ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def infer_device(cfg: Config) -> str:
    if cfg.device in {"cpu", "cuda"}:
        return cfg.device
    if torch is None:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def detect_columns(df: pd.DataFrame) -> Tuple[str, str, str, List[str]]:
    """
    Returns (time_col, run_col, label_col, sensor_cols)
    Expected columns in provided file:
      time, s01..s16, gas_name, run_id, t_exposure_start_ms, t_exposure_end_ms
    """
    # time
    time_candidates = [c for c in df.columns if c.lower() in {"time", "time_ms", "t", "t_ms"}]
    time_col = time_candidates[0] if time_candidates else df.columns[0]

    # run
    run_candidates = [c for c in df.columns if c.lower() in {"run_id", "run"}]
    if not run_candidates:
        raise ValueError("Could not find run id column (expected 'run_id' or 'run').")
    run_col = run_candidates[0]

    # label
    label_candidates = [c for c in df.columns if c.lower() in {"gas_name", "label", "y", "gas"}]
    if not label_candidates:
        # heuristic: find object column with small number of unique values
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        if not obj_cols:
            raise ValueError("Could not infer label column.")
        label_col = min(obj_cols, key=lambda c: df[c].nunique())
    else:
        label_col = label_candidates[0]

    # sensors: numeric cols except metadata
    meta = {time_col, run_col, label_col, "gas_id", "t_exposure_start_ms", "t_exposure_end_ms"}
    sensor_cols = [c for c in df.columns if c not in meta and np.issubdtype(df[c].dtype, np.number)]

    if len(sensor_cols) < 2:
        raise ValueError("Could not detect sensor columns.")
    return time_col, run_col, label_col, sensor_cols


def infer_n_steps(df: pd.DataFrame, time_col: str, run_col: str, window_ms: int) -> int:
    # Use median dt within runs
    dts = []
    for _, g in df.groupby(run_col):
        t = np.asarray(g[time_col].sort_values().values, dtype=np.float64)
        if len(t) < 5:
            continue
        dt = np.median(np.diff(t))
        if np.isfinite(dt) and dt > 0:
            dts.append(dt)
    if not dts:
        # fallback: assume 10 Hz => 100ms
        dt_med = 100.0
    else:
        dt_med = float(np.median(dts))
    n_steps = int(round(window_ms / dt_med)) + 1
    # Clamp to avoid pathological sizes
    n_steps = int(np.clip(n_steps, 64, 512))
    return n_steps


def get_phase_intervals(run_row: pd.Series, cfg: Config) -> List[Tuple[str, int, int]]:
    """
    Returns list of (phase_name, start_ms, end_ms) in ABSOLUTE ms.
    """
    if cfg.phase_mode == "fullrun":
        # Use the run's min/max time indirectly:
        # Caller should override with actual min/max time for the run.
        return [("fullrun", -1, -1)]

    # 3-phase mode (requires exposure markers)
    if not {"t_exposure_start_ms", "t_exposure_end_ms"}.issubset(run_row.index):
        raise ValueError("phase_mode=3phase requires t_exposure_start_ms and t_exposure_end_ms columns.")

    t_s = int(run_row["t_exposure_start_ms"])
    t_e = int(run_row["t_exposure_end_ms"])
    return [
        ("exposure", t_s, t_e),
        ("desorb_early", t_e, t_e + 60_000),
        ("desorb_mid", t_e + 60_000, t_e + 120_000),
    ]


def resample_to_fixed_grid(times_abs: np.ndarray, values: np.ndarray, start_abs: int, window_ms: int, n_steps: int) -> np.ndarray:
    """
    times_abs: (m,)
    values: (m, n_channels)
    returns: (n_steps, n_channels)
    """
    # relative time in [0, window_ms]
    t_rel = times_abs.astype(np.float64) - float(start_abs)
    # Guard: need increasing times for interp
    order = np.argsort(t_rel)
    t_rel = t_rel[order]
    v = values[order]
    # Remove duplicates in time (interp requires increasing x)
    uniq_t, uniq_idx = np.unique(t_rel, return_index=True)
    v = v[uniq_idx]
    t_rel = uniq_t
    if len(t_rel) < 2:
        raise ValueError("Not enough points for interpolation.")
    new_t = np.linspace(0.0, float(window_ms), n_steps, dtype=np.float64)
    out = np.empty((n_steps, v.shape[1]), dtype=np.float32)
    for c in range(v.shape[1]):
        out[:, c] = np.interp(new_t, t_rel, v[:, c]).astype(np.float32)
    return out


def make_windows_for_run(
    g: pd.DataFrame,
    time_col: str,
    sensor_cols: List[str],
    label_col: str,
    run_col: str,
    cfg: Config,
    n_steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X: (n_windows, n_steps, n_channels)
      y: (n_windows,)
      run_ids: (n_windows,)
    """
    window_ms = int(cfg.window_sec * 1000)
    stride_ms = int(cfg.stride_sec * 1000)

    g = g.sort_values(time_col).reset_index(drop=True)
    run_id = g[run_col].iloc[0]
    y_label = g[label_col].iloc[0]

    # run-level metadata row
    run_meta = g.iloc[0]

    # Determine phase intervals
    if cfg.phase_mode == "fullrun":
        run_start = int(g[time_col].iloc[0])
        run_end = int(g[time_col].iloc[-1])
        phases = [("fullrun", run_start, run_end)]
    else:
        phases = get_phase_intervals(run_meta, cfg)

    X_list, y_list, r_list = [], [], []

    for phase_name, phase_start, phase_end in phases:
        # Make sure phase bounds are within this run
        phase_start = max(phase_start, int(g[time_col].iloc[0]))
        phase_end = min(phase_end, int(g[time_col].iloc[-1]))

        last_start = phase_end - window_ms
        if last_start < phase_start:
            continue

        starts = np.arange(phase_start, last_start + 1, stride_ms, dtype=np.int64)
        for s in starts:
            e = s + window_ms
            sub = g[(g[time_col] >= s) & (g[time_col] <= e)]
            if len(sub) < cfg.min_points_in_raw_window:
                continue
            try:
                Xw = resample_to_fixed_grid(
                    times_abs=sub[time_col].values,
                    values=sub[sensor_cols].values.astype(np.float64),
                    start_abs=int(s),
                    window_ms=window_ms,
                    n_steps=n_steps,
                )
            except Exception:
                continue
            X_list.append(Xw)
            y_list.append(y_label)
            r_list.append(run_id)

    if not X_list:
        return np.empty((0, n_steps, len(sensor_cols)), dtype=np.float32), np.empty((0,), dtype=object), np.empty((0,), dtype=object)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list)
    r = np.asarray(r_list)
    return X, y, r


def windows_from_runs(
    df: pd.DataFrame,
    run_ids: List,
    time_col: str,
    sensor_cols: List[str],
    label_col: str,
    run_col: str,
    cfg: Config,
    n_steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, rs = [], [], []
    for rid in run_ids:
        g = df[df[run_col] == rid]
        X, y, r = make_windows_for_run(g, time_col, sensor_cols, label_col, run_col, cfg, n_steps)
        if len(y) == 0:
            continue
        xs.append(X); ys.append(y); rs.append(r)
    if not xs:
        return np.empty((0, n_steps, len(sensor_cols)), dtype=np.float32), np.empty((0,), dtype=object), np.empty((0,), dtype=object)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), np.concatenate(rs, axis=0)


def ts_stats_features(X: np.ndarray) -> np.ndarray:
    """
    X: (n, steps, channels)
    returns: (n, 6*channels) features (mean, std, min, max, delta, slope)
    """
    n, steps, ch = X.shape
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    vmin = X.min(axis=1)
    vmax = X.max(axis=1)
    delta = X[:, -1, :] - X[:, 0, :]

    t = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    t0 = t - t.mean()
    den = float((t0 ** 2).sum()) + 1e-12
    X0 = X - mean[:, None, :]
    slope = (t0[None, :, None] * X0).sum(axis=1) / den

    feats = np.concatenate([mean, std, vmin, vmax, delta, slope], axis=1)
    return feats.astype(np.float32)


def majority_vote_by_run(y_pred: np.ndarray, run_ids: np.ndarray) -> Dict:
    out = {}
    for rid in np.unique(run_ids):
        preds = y_pred[run_ids == rid]
        # majority vote (ties broken deterministically by sorted label)
        labels, counts = np.unique(preds, return_counts=True)
        best = labels[np.argmax(counts)]
        out[rid] = best
    return out


def proba_vote_by_run(proba: np.ndarray, run_ids: np.ndarray, classes: List[str]) -> Dict:
    # proba: (n_windows, n_classes)
    out = {}
    for rid in np.unique(run_ids):
        p = proba[run_ids == rid].mean(axis=0)
        out[rid] = classes[int(np.argmax(p))]
    return out


def metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {"accuracy": float(acc), "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f1)}


def plot_confmat(cm: np.ndarray, labels: List[str], title: str, outpath: str, normalize: bool = True) -> None:
    if normalize:
        cmn = cm.astype(np.float64)
        row = cmn.sum(axis=1, keepdims=True)
        cmn = np.divide(cmn, np.maximum(row, 1.0))
        mat = cmn
    else:
        mat = cm

    fig = plt.figure(figsize=(7.2, 6.2))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def save_confusion_artifacts(
    cm: np.ndarray,
    labels: List[str],
    out_prefix: str,
    top_k: int = 30,
) -> None:
    """Save confusion matrix as CSVs and export top confusion pairs.

    Files written:
      - {out_prefix}_counts.csv
      - {out_prefix}_row_normalized.csv
      - {out_prefix}_top_pairs.csv
    """
    # Counts CSV
    df_counts = pd.DataFrame(cm, index=labels, columns=labels)
    df_counts.to_csv(out_prefix + "_counts.csv")

    # Row-normalized CSV (per-true-class error profile)
    row = cm.sum(axis=1, keepdims=True).astype(np.float64)
    row = np.maximum(row, 1.0)
    cmn = cm.astype(np.float64) / row
    df_norm = pd.DataFrame(cmn, index=labels, columns=labels)
    df_norm.to_csv(out_prefix + "_row_normalized.csv")

    # Top confusion pairs (off-diagonal)
    pairs = []
    for i, t_lab in enumerate(labels):
        denom = float(cm[i, :].sum())
        denom = denom if denom > 0 else 1.0
        for j, p_lab in enumerate(labels):
            if i == j:
                continue
            cnt = int(cm[i, j])
            if cnt <= 0:
                continue
            pairs.append(
                {
                    "true": t_lab,
                    "pred": p_lab,
                    "count": cnt,
                    "rate_in_true": float(cnt / denom),
                }
            )
    df_pairs = pd.DataFrame(pairs)
    if len(df_pairs) == 0:
        df_pairs = pd.DataFrame(columns=["true", "pred", "count", "rate_in_true"])
    else:
        df_pairs = df_pairs.sort_values(["count", "rate_in_true"], ascending=[False, False]).head(top_k)
    df_pairs.to_csv(out_prefix + "_top_pairs.csv", index=False)


# -----------------------------
# Deep learning (PyTorch)
# -----------------------------
class WindowDatasetTorch(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, label_to_idx: Dict[str, int], mode: str):
        self.X = X.astype(np.float32)
        self.y = np.asarray([label_to_idx[v] for v in y], dtype=np.int64)
        self.mode = mode  # "cnn" or "lstm"

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        # x: (steps, ch)
        if self.mode == "cnn":
            x = np.transpose(x, (1, 0))  # (ch, steps)
        # lstm keeps (steps, ch)
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)


class CNN1D(nn.Module):
    def __init__(self, n_channels: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        # x: (B, C, T)
        z = self.net(x)
        return self.head(z)


class LSTMNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, hidden: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        # x: (B, T, C)
        out, (h, _) = self.lstm(x)
        h_last = h[-1]  # (B, hidden)
        return self.head(h_last)


@torch.no_grad()
def torch_predict(model: nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_y = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        all_logits.append(logits.cpu().numpy())
        all_y.append(yb.numpy())
    logits = np.concatenate(all_logits, axis=0)
    y = np.concatenate(all_y, axis=0)
    proba = np.exp(logits - logits.max(axis=1, keepdims=True))
    proba = proba / np.maximum(proba.sum(axis=1, keepdims=True), 1e-12)
    pred = proba.argmax(axis=1)
    return pred, proba


def train_torch(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: np.ndarray,
    cfg: Config,
    device: str,
) -> nn.Module:
    model = model.to(device)
    w = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_state = None
    best_val = -1.0
    bad = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

        # validate on run-level (more stable) using window-level for early stop
        vp, _ = torch_predict(model, val_loader, device)
        vy = np.concatenate([yb.numpy() for _, yb in val_loader], axis=0)
        val_acc = float((vp == vy).mean())

        if val_acc > best_val + 1e-5:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def standardize_ts_train_only(X_train: np.ndarray, X_other: np.ndarray) -> np.ndarray:
    """
    Channel-wise standardization using train statistics only.
    X: (n, steps, ch)
    """
    mu = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    sd = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X_other - mu[None, None, :]) / sd[None, None, :]


# -----------------------------
# Splitting (run-level stratified shuffle)
# -----------------------------
def stratified_run_splits(runs_df: pd.DataFrame, label_col: str, run_col: str, cfg: Config) -> List[Tuple[List, List]]:
    """
    runs_df: unique rows of (run_id, label)
    returns list of (train_run_ids, test_run_ids)
    """
    set_all_seeds(cfg.seed)
    # group runs by label
    by_label = {lab: runs_df[runs_df[label_col] == lab][run_col].tolist() for lab in sorted(runs_df[label_col].unique())}

    splits = []
    for s in range(cfg.n_splits):
        rng = np.random.RandomState(cfg.seed + 1000 + s)
        train_ids, test_ids = [], []
        for lab, ids in by_label.items():
            ids = ids.copy()
            rng.shuffle(ids)
            n_test = max(1, int(round(len(ids) * cfg.test_size)))
            test = ids[:n_test]
            train = ids[n_test:]
            test_ids.extend(test)
            train_ids.extend(train)
        rng.shuffle(train_ids); rng.shuffle(test_ids)
        splits.append((train_ids, test_ids))
    return splits


def pick_val_runs(train_run_ids: List, runs_df: pd.DataFrame, label_col: str, run_col: str, seed: int, val_per_class: int = 1) -> Tuple[List, List]:
    """
    Choose validation runs from train runs, stratified by class.
    """
    rng = np.random.RandomState(seed)
    df_tr = runs_df[runs_df[run_col].isin(train_run_ids)]
    val, tr = [], []
    for lab in sorted(df_tr[label_col].unique()):
        ids = df_tr[df_tr[label_col] == lab][run_col].tolist()
        rng.shuffle(ids)
        take = min(val_per_class, max(0, len(ids) - 1))
        val.extend(ids[:take])
        tr.extend(ids[take:])
    rng.shuffle(val); rng.shuffle(tr)
    return tr, val


# -----------------------------
# Main evaluation
# -----------------------------
def run_pipeline(csv_path: str, outdir: str, cfg: Config) -> None:
    df = pd.read_csv(csv_path)
    time_col, run_col, label_col, sensor_cols = detect_columns(df)

    # Consistent class order (use gas_id if available)
    if "gas_id" in df.columns and label_col in df.columns:
        class_map = df[["gas_id", label_col]].drop_duplicates().sort_values("gas_id")
        classes = class_map[label_col].tolist()
    else:
        classes = sorted(df[label_col].unique().tolist())

    runs_df = df[[run_col, label_col, "t_exposure_start_ms", "t_exposure_end_ms"]].drop_duplicates(subset=[run_col])
    runs_df = runs_df[[run_col, label_col, "t_exposure_start_ms", "t_exposure_end_ms"]]

    window_ms = int(cfg.window_sec * 1000)
    n_steps = cfg.n_steps if cfg.n_steps is not None else infer_n_steps(df, time_col, run_col, window_ms)

    print(f"[INFO] CSV: {csv_path}")
    print(f"[INFO] Detected columns: time={time_col}, run={run_col}, label={label_col}, sensors={len(sensor_cols)}")
    print(f"[INFO] Runs: {runs_df.shape[0]}, Classes: {len(classes)} => {classes}")
    print(f"[INFO] Window: {cfg.window_sec:.1f}s, Stride: {cfg.stride_sec:.1f}s, phase_mode={cfg.phase_mode}, n_steps={n_steps}")

    ensure_outdir(outdir)

    splits = stratified_run_splits(runs_df[[run_col, label_col]], label_col, run_col, cfg)

    # Storage
    models = ["SVM", "KNN", "RF"] + (["CNN", "LSTM"] if cfg.use_deep else [])
    metrics_window = {m: [] for m in models}
    metrics_run = {m: [] for m in models}
    cm_window_sum = {m: np.zeros((len(classes), len(classes)), dtype=np.int64) for m in models}
    cm_run_sum = {m: np.zeros((len(classes), len(classes)), dtype=np.int64) for m in models}

    label_to_idx = {lab: i for i, lab in enumerate(classes)}
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}

    device = infer_device(cfg)

    for split_i, (train_runs, test_runs) in enumerate(splits, start=1):
        # Add a small validation set for deep models (run-level to avoid leakage)
        tr_runs, val_runs = pick_val_runs(train_runs, runs_df[[run_col, label_col]], label_col, run_col, seed=cfg.seed + split_i, val_per_class=1)

        Xtr, ytr, rtr = windows_from_runs(df, tr_runs, time_col, sensor_cols, label_col, run_col, cfg, n_steps)
        Xva, yva, rva = windows_from_runs(df, val_runs, time_col, sensor_cols, label_col, run_col, cfg, n_steps)
        Xte, yte, rte = windows_from_runs(df, test_runs, time_col, sensor_cols, label_col, run_col, cfg, n_steps)

        if len(yte) == 0 or len(ytr) == 0:
            print(f"[WARN] Split {split_i}: empty windows (train={len(ytr)}, test={len(yte)}). Skipping.")
            continue

        # -----------------
        # Classical models
        # -----------------
        Ftr = ts_stats_features(Xtr)
        Fte = ts_stats_features(Xte)

        test_run_ids = np.unique(rte)

        # KNN
        knn = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=cfg.knn_k)),
        ])
        knn.fit(Ftr, ytr)
        ypred_w = knn.predict(Fte)

        m_w = metrics_from_preds(yte, ypred_w, labels=classes)
        metrics_window["KNN"].append(m_w)
        cm_window_sum["KNN"] += confusion_matrix(yte, ypred_w, labels=classes)

        # run-level (majority vote)
        y_true_run = np.asarray([df[df[run_col] == rid][label_col].iloc[0] for rid in test_run_ids])
        vote = majority_vote_by_run(ypred_w, rte)
        y_pred_run = np.asarray([vote[rid] for rid in test_run_ids])
        m_r = metrics_from_preds(y_true_run, y_pred_run, labels=classes)
        metrics_run["KNN"].append(m_r)
        cm_run_sum["KNN"] += confusion_matrix(y_true_run, y_pred_run, labels=classes)

        # SVM
        svm = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(C=cfg.svm_c, gamma=cfg.svm_gamma, kernel="rbf", class_weight="balanced")),
        ])
        svm.fit(Ftr, ytr)
        ypred_w = svm.predict(Fte)

        m_w = metrics_from_preds(yte, ypred_w, labels=classes)
        metrics_window["SVM"].append(m_w)
        cm_window_sum["SVM"] += confusion_matrix(yte, ypred_w, labels=classes)

        vote = majority_vote_by_run(ypred_w, rte)
        y_pred_run = np.asarray([vote[rid] for rid in test_run_ids])
        y_true_run = np.asarray([df[df[run_col] == rid][label_col].iloc[0] for rid in test_run_ids])
        m_r = metrics_from_preds(y_true_run, y_pred_run, labels=classes)
        metrics_run["SVM"].append(m_r)
        cm_run_sum["SVM"] += confusion_matrix(y_true_run, y_pred_run, labels=classes)

        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=cfg.rf_estimators,
            max_depth=cfg.rf_max_depth,
            min_samples_leaf=cfg.rf_min_samples_leaf,
            max_features=cfg.rf_max_features,
            n_jobs=cfg.rf_n_jobs,
            class_weight="balanced_subsample",
            random_state=cfg.seed + 50_000 + split_i,
        )
        rf.fit(Ftr, ytr)
        ypred_w = rf.predict(Fte)

        m_w = metrics_from_preds(yte, ypred_w, labels=classes)
        metrics_window["RF"].append(m_w)
        cm_window_sum["RF"] += confusion_matrix(yte, ypred_w, labels=classes)

        # run-level (probability vote when available)
        if hasattr(rf, "predict_proba"):
            proba = rf.predict_proba(Fte)
            # Map RF's internal class order to the global 'classes' order
            proba_full = np.zeros((proba.shape[0], len(classes)), dtype=np.float64)
            for j, cls in enumerate(rf.classes_):
                if cls in label_to_idx:
                    proba_full[:, label_to_idx[cls]] = proba[:, j]
            vote = proba_vote_by_run(proba_full, rte, classes=classes)
        else:
            vote = majority_vote_by_run(ypred_w, rte)

        y_pred_run = np.asarray([vote[rid] for rid in test_run_ids])
        y_true_run = np.asarray([df[df[run_col] == rid][label_col].iloc[0] for rid in test_run_ids])
        m_r = metrics_from_preds(y_true_run, y_pred_run, labels=classes)
        metrics_run["RF"].append(m_r)
        cm_run_sum["RF"] += confusion_matrix(y_true_run, y_pred_run, labels=classes)

        # -----------------
        # Deep models
        # -----------------
        if cfg.use_deep:
            if torch is None:
                raise RuntimeError("PyTorch is not available but --use_deep was requested.")

            # Standardize sequences using train-only stats
            Xtr_s = standardize_ts_train_only(Xtr, Xtr)
            Xva_s = standardize_ts_train_only(Xtr, Xva)
            Xte_s = standardize_ts_train_only(Xtr, Xte)

            # Class weights (train windows)
            ytr_idx = np.asarray([label_to_idx[v] for v in ytr], dtype=np.int64)
            counts = np.bincount(ytr_idx, minlength=len(classes)).astype(np.float64)
            w = counts.sum() / np.maximum(counts, 1.0)
            w = w / w.mean()

            # ---- CNN
            set_all_seeds(cfg.seed + 10_000 + split_i)
            tr_ds = WindowDatasetTorch(Xtr_s, ytr, label_to_idx, mode="cnn")
            va_ds = WindowDatasetTorch(Xva_s, yva, label_to_idx, mode="cnn")
            te_ds = WindowDatasetTorch(Xte_s, yte, label_to_idx, mode="cnn")

            tr_ld = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
            va_ld = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False)
            te_ld = DataLoader(te_ds, batch_size=cfg.batch_size, shuffle=False)

            cnn = CNN1D(n_channels=len(sensor_cols), n_classes=len(classes))
            cnn = train_torch(cnn, tr_ld, va_ld, class_weights=w, cfg=cfg, device=device)

            pred_idx, proba = torch_predict(cnn, te_ld, device=device)
            ypred_w = np.asarray([idx_to_label[i] for i in pred_idx])
            m_w = metrics_from_preds(yte, ypred_w, labels=classes)
            metrics_window["CNN"].append(m_w)
            cm_window_sum["CNN"] += confusion_matrix(yte, ypred_w, labels=classes)

            vote = proba_vote_by_run(proba, rte, classes=classes)
            y_pred_run = np.asarray([vote[rid] for rid in np.unique(rte)])
            y_true_run = np.asarray([df[df[run_col] == rid][label_col].iloc[0] for rid in np.unique(rte)])
            m_r = metrics_from_preds(y_true_run, y_pred_run, labels=classes)
            metrics_run["CNN"].append(m_r)
            cm_run_sum["CNN"] += confusion_matrix(y_true_run, y_pred_run, labels=classes)

            # ---- LSTM
            set_all_seeds(cfg.seed + 20_000 + split_i)
            tr_ds = WindowDatasetTorch(Xtr_s, ytr, label_to_idx, mode="lstm")
            va_ds = WindowDatasetTorch(Xva_s, yva, label_to_idx, mode="lstm")
            te_ds = WindowDatasetTorch(Xte_s, yte, label_to_idx, mode="lstm")

            tr_ld = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
            va_ld = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False)
            te_ld = DataLoader(te_ds, batch_size=cfg.batch_size, shuffle=False)

            lstm = LSTMNet(n_channels=len(sensor_cols), n_classes=len(classes), hidden=128)
            lstm = train_torch(lstm, tr_ld, va_ld, class_weights=w, cfg=cfg, device=device)

            pred_idx, proba = torch_predict(lstm, te_ld, device=device)
            ypred_w = np.asarray([idx_to_label[i] for i in pred_idx])
            m_w = metrics_from_preds(yte, ypred_w, labels=classes)
            metrics_window["LSTM"].append(m_w)
            cm_window_sum["LSTM"] += confusion_matrix(yte, ypred_w, labels=classes)

            vote = proba_vote_by_run(proba, rte, classes=classes)
            y_pred_run = np.asarray([vote[rid] for rid in np.unique(rte)])
            y_true_run = np.asarray([df[df[run_col] == rid][label_col].iloc[0] for rid in np.unique(rte)])
            m_r = metrics_from_preds(y_true_run, y_pred_run, labels=classes)
            metrics_run["LSTM"].append(m_r)
            cm_run_sum["LSTM"] += confusion_matrix(y_true_run, y_pred_run, labels=classes)

        print(f"[INFO] Finished split {split_i}/{cfg.n_splits}: test_runs={len(test_runs)} | test_windows={len(yte)}")

    # -----------------
    # Summary + save
    # -----------------
    def summarize(metrics_list: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
        keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
        out = {}
        for k in keys:
            vals = [d[k] for d in metrics_list] if metrics_list else []
            if not vals:
                out[k] = (float("nan"), float("nan"))
            else:
                out[k] = (float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0))
        return out

    rows = []
    print("\n==================== RESULTS (mean ± std over splits) ====================")
    for m in models:
        sw = summarize(metrics_window[m])
        sr = summarize(metrics_run[m])
        rows.append({"model": m, "level": "window", **{k: sw[k][0] for k in sw}, **{k+"_std": sw[k][1] for k in sw}})
        rows.append({"model": m, "level": "run", **{k: sr[k][0] for k in sr}, **{k+"_std": sr[k][1] for k in sr}})

        print(f"\n[{m}] WINDOW-level:")
        print(f"  accuracy        = {sw['accuracy'][0]:.4f} ± {sw['accuracy'][1]:.4f}")
        print(f"  precision(macro)= {sw['precision_macro'][0]:.4f} ± {sw['precision_macro'][1]:.4f}")
        print(f"  recall(macro)   = {sw['recall_macro'][0]:.4f} ± {sw['recall_macro'][1]:.4f}")
        print(f"  f1(macro)       = {sw['f1_macro'][0]:.4f} ± {sw['f1_macro'][1]:.4f}")

        print(f"[{m}] RUN-level (vote over windows per run):")
        print(f"  accuracy        = {sr['accuracy'][0]:.4f} ± {sr['accuracy'][1]:.4f}")
        print(f"  precision(macro)= {sr['precision_macro'][0]:.4f} ± {sr['precision_macro'][1]:.4f}")
        print(f"  recall(macro)   = {sr['recall_macro'][0]:.4f} ± {sr['recall_macro'][1]:.4f}")
        print(f"  f1(macro)       = {sr['f1_macro'][0]:.4f} ± {sr['f1_macro'][1]:.4f}")

    summary_df = pd.DataFrame(rows)
    summary_csv = os.path.join(outdir, "metrics_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    # Confusion matrices
    for m in models:
        win_prefix = os.path.join(outdir, f"confmat_{m}_window")
        run_prefix = os.path.join(outdir, f"confmat_{m}_run")

        plot_confmat(cm_window_sum[m], classes, f"{m} confusion matrix (WINDOW, normalized)", win_prefix + ".png", normalize=True)
        plot_confmat(cm_run_sum[m], classes, f"{m} confusion matrix (RUN vote, normalized)", run_prefix + ".png", normalize=True)

        # CSV + top confusion pairs for mechanistic discussion
        save_confusion_artifacts(cm_window_sum[m], classes, win_prefix)
        save_confusion_artifacts(cm_run_sum[m], classes, run_prefix)

    print(f"\n[INFO] Saved summary: {summary_csv}")
    print(f"[INFO] Saved confusion matrices to: {outdir}/confmat_*")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="Path to all_gases_framewise.csv")
    p.add_argument("--outdir", type=str, default="outputs_baseline", help="Output directory")
    p.add_argument("--window_sec", type=float, default=30.0)
    p.add_argument("--stride_sec", type=float, default=10.0)
    p.add_argument("--phase_mode", type=str, default="3phase", choices=["3phase", "fullrun"])
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--test_size", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--n_steps", type=int, default=None, help="Resample length per window (override auto-infer)")
    p.add_argument("--no_deep", action="store_true", help="Disable CNN/LSTM (only SVM/KNN/RF)")

    # classical
    p.add_argument("--knn_k", type=int, default=7)
    p.add_argument("--svm_c", type=float, default=10.0)
    p.add_argument("--svm_gamma", type=str, default="scale")

    p.add_argument("--rf_estimators", type=int, default=500)
    p.add_argument("--rf_max_depth", type=int, default=None)
    p.add_argument("--rf_min_samples_leaf", type=int, default=1)
    p.add_argument("--rf_max_features", type=str, default="sqrt")
    p.add_argument("--rf_n_jobs", type=int, default=-1)

    # deep
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return p


def main():
    args = build_argparser().parse_args()
    cfg = Config(
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        phase_mode=args.phase_mode,
        n_splits=args.n_splits,
        test_size=args.test_size,
        seed=args.seed,
        n_steps=args.n_steps,
        use_deep=(not args.no_deep),
        knn_k=args.knn_k,
        svm_c=args.svm_c,
        svm_gamma=args.svm_gamma,
        rf_estimators=args.rf_estimators,
        rf_max_depth=args.rf_max_depth,
        rf_min_samples_leaf=args.rf_min_samples_leaf,
        rf_max_features=args.rf_max_features,
        rf_n_jobs=args.rf_n_jobs,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
    )
    run_pipeline(args.csv, args.outdir, cfg)


if __name__ == "__main__":
    main()
