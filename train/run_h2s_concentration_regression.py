#!/usr/bin/env python3
"""
Diagnostic-first pipeline for H2S concentration regression from multi-channel resistance recordings.
"""

import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.signal import savgol_filter as _savgol_filter  # type: ignore
    _HAS_SAVGOL = True
except Exception:
    _HAS_SAVGOL = False

_SAVGOL_WARNED = False


@dataclass
class CycleData:
    file_id: str
    concentration_ppm: float
    cycle_id: int
    sensor_count: int
    dt: float
    exposure_len: int
    data: np.ndarray  # shape (cycle_len, K)


def parse_concentration_from_filename(filename: str) -> int:
    match = re.search(r"(\d+)ppm", filename)
    if not match:
        raise ValueError(f"Could not parse concentration from filename: {filename}")
    return int(match.group(1))


def estimate_dt_seconds(time_col: np.ndarray) -> float:
    diffs = np.diff(time_col.astype(float))
    if len(diffs) == 0:
        raise ValueError("Not enough timestamps to estimate dt.")
    dt_raw = float(np.median(diffs))
    if dt_raw > 5.0:
        return dt_raw / 1000.0
    return dt_raw


def load_data(data_dir: str) -> List[Tuple[str, pd.DataFrame, int, float]]:
    files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith("ppm.txt") or f.endswith("ppm.TXT")]
    )
    if not files:
        raise FileNotFoundError(f"No *ppm.txt files found in {data_dir}")

    datasets = []
    for fname in files:
        fpath = os.path.join(data_dir, fname)
        df = pd.read_csv(fpath, sep=r"\s+", header=None)
        if df.shape[1] < 2:
            raise ValueError(f"Expected at least 2 columns in {fpath}, found {df.shape[1]}")
        concentration = parse_concentration_from_filename(fname)
        dt = estimate_dt_seconds(df.iloc[:, 0].values)
        datasets.append((fname, df, concentration, dt))
    return datasets


def smooth_signal(values: np.ndarray, dt: float, method: str) -> np.ndarray:
    if method == "none":
        return values

    window_len = max(5, int(round(1.0 / dt)))
    if window_len % 2 == 0:
        window_len += 1
    if window_len > values.shape[0]:
        window_len = values.shape[0] if values.shape[0] % 2 == 1 else values.shape[0] - 1
        window_len = max(window_len, 3)

    if method == "savgol":
        if _HAS_SAVGOL:
            return _savgol_filter(values, window_length=window_len, polyorder=2, axis=0, mode="interp")
        global _SAVGOL_WARNED
        if not _SAVGOL_WARNED:
            print("Warning: scipy not available, falling back to median smoothing.")
            _SAVGOL_WARNED = True
        method = "median"

    if method == "median":
        smoothed = np.zeros_like(values)
        for i in range(values.shape[1]):
            series = pd.Series(values[:, i])
            smoothed[:, i] = (
                series.rolling(window=window_len, center=True, min_periods=1)
                .median()
                .to_numpy()
            )
        return smoothed

    return values


def segment_cycles(
    df: pd.DataFrame,
    dt: float,
    exposure_s: float,
    purge_s: float,
    n_cycles: int,
    file_id: str,
    concentration_ppm: int,
) -> List[CycleData]:
    values = df.iloc[:, 1:].to_numpy(dtype=float)
    sensor_count = values.shape[1]
    exposure_len = int(round(exposure_s / dt))
    purge_len = int(round(purge_s / dt))
    if exposure_len < 1 or purge_len < 1:
        raise ValueError(
            f"Exposure or purge length too short with dt={dt:.6f}s: exposure_len={exposure_len}, purge_len={purge_len}"
        )
    cycle_len = exposure_len + purge_len

    expected_len = n_cycles * cycle_len
    if values.shape[0] < expected_len:
        raise ValueError(
            f"File {file_id} too short: expected >= {expected_len} rows, got {values.shape[0]} rows."
        )
    if values.shape[0] > expected_len:
        values = values[:expected_len, :]

    cycles = []
    for cycle_id in range(n_cycles):
        start = cycle_id * cycle_len
        end = start + cycle_len
        cycle_values = values[start:end, :]
        cycles.append(
            CycleData(
                file_id=file_id,
                concentration_ppm=concentration_ppm,
                cycle_id=cycle_id,
                sensor_count=sensor_count,
                dt=dt,
                exposure_len=exposure_len,
                data=cycle_values,
            )
        )
    return cycles


def clip_outliers(values: np.ndarray, low_pct: float = 0.5, high_pct: float = 99.5) -> np.ndarray:
    clipped = values.copy()
    for i in range(values.shape[1]):
        low = np.percentile(values[:, i], low_pct)
        high = np.percentile(values[:, i], high_pct)
        clipped[:, i] = np.clip(values[:, i], low, high)
    return clipped


def linear_slope(x: np.ndarray, dt: float) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    t = np.arange(n, dtype=float) * dt
    t_mean = t.mean()
    x_mean = x.mean()
    denom = np.sum((t - t_mean) ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum((t - t_mean) * (x - x_mean)) / denom)


def extract_features(
    cycle: CycleData,
    baseline_tail_s: float,
    smooth: str,
    prev_baseline: Optional[np.ndarray],
    eps: float = 1e-8,
) -> Tuple[Dict[str, float], np.ndarray, float]:
    data = cycle.data
    data = smooth_signal(data, cycle.dt, smooth)
    data = clip_outliers(data)

    exposure = data[: cycle.exposure_len, :]
    purge = data[cycle.exposure_len :, :]

    baseline_n = int(round(baseline_tail_s / cycle.dt))
    baseline_n = max(1, min(baseline_n, purge.shape[0]))
    baseline = purge[-baseline_n:, :].mean(axis=0)

    exp_mean = exposure.mean(axis=0)
    exp_min = exposure.min(axis=0)
    exp_max = exposure.max(axis=0)

    rel_mean = exp_mean - baseline
    rel_min = exp_min - baseline
    rel_max = exp_max - baseline

    norm_mean = rel_mean / (baseline + eps)
    norm_min = rel_min / (baseline + eps)
    norm_max = rel_max / (baseline + eps)

    log_mean = np.log((exp_mean + eps) / (baseline + eps))
    log_min = np.log((exp_min + eps) / (baseline + eps))
    log_max = np.log((exp_max + eps) / (baseline + eps))

    onset_n = max(2, int(round(10.0 / cycle.dt)))
    onset_n = min(onset_n, exposure.shape[0])
    recovery_n = max(2, int(round(10.0 / cycle.dt)))
    recovery_n = min(recovery_n, purge.shape[0])

    onset_slope = np.array([
        linear_slope(exposure[:onset_n, i], cycle.dt) for i in range(cycle.sensor_count)
    ])
    recovery_slope = np.array([
        linear_slope(purge[:recovery_n, i], cycle.dt) for i in range(cycle.sensor_count)
    ])

    end_n = max(2, int(round(10.0 / cycle.dt)))
    end_n = min(end_n, exposure.shape[0])
    end_slope = np.array([
        linear_slope(exposure[-end_n:, i], cycle.dt) for i in range(cycle.sensor_count)
    ])

    auc_exposure = np.trapz(exposure - baseline, dx=cycle.dt, axis=0)
    auc_purge = np.trapz(purge - baseline, dx=cycle.dt, axis=0)

    purge_start_n = max(2, int(round(10.0 / cycle.dt)))
    purge_start_n = min(purge_start_n, purge.shape[0])
    purge_start_mean = purge[:purge_start_n, :].mean(axis=0)
    purge_end_mean = purge[-purge_start_n:, :].mean(axis=0)
    incomplete_recovery = purge_end_mean - purge_start_mean

    hysteresis_ratio = auc_purge / (auc_exposure + eps)
    saturation_ratio = (exp_max - baseline) / (exp_mean - baseline + eps)

    scalar_response = np.median(norm_max)

    drift = np.zeros_like(baseline)
    if prev_baseline is not None and prev_baseline.shape == baseline.shape:
        drift = baseline - prev_baseline

    feature_names = [
        "R0",
        "exp_mean",
        "exp_min",
        "exp_max",
        "rel_mean",
        "rel_min",
        "rel_max",
        "norm_mean",
        "norm_min",
        "norm_max",
        "log_mean",
        "log_min",
        "log_max",
        "onset_slope",
        "recovery_slope",
        "end_slope",
        "auc_exposure",
        "auc_purge",
        "incomplete_recovery",
        "hysteresis_ratio",
        "saturation_ratio",
        "baseline_drift",
    ]

    feature_arrays = {
        "R0": baseline,
        "exp_mean": exp_mean,
        "exp_min": exp_min,
        "exp_max": exp_max,
        "rel_mean": rel_mean,
        "rel_min": rel_min,
        "rel_max": rel_max,
        "norm_mean": norm_mean,
        "norm_min": norm_min,
        "norm_max": norm_max,
        "log_mean": log_mean,
        "log_min": log_min,
        "log_max": log_max,
        "onset_slope": onset_slope,
        "recovery_slope": recovery_slope,
        "end_slope": end_slope,
        "auc_exposure": auc_exposure,
        "auc_purge": auc_purge,
        "incomplete_recovery": incomplete_recovery,
        "hysteresis_ratio": hysteresis_ratio,
        "saturation_ratio": saturation_ratio,
        "baseline_drift": drift,
    }

    features: Dict[str, float] = {}
    for sensor_idx in range(cycle.sensor_count):
        sensor_prefix = f"s{sensor_idx + 1}"
        for name in feature_names:
            features[f"{sensor_prefix}_{name}"] = float(feature_arrays[name][sensor_idx])

    for name in feature_names:
        arr = feature_arrays[name]
        features[f"agg_{name}_mean"] = float(np.mean(arr))
        features[f"agg_{name}_median"] = float(np.median(arr))
        features[f"agg_{name}_std"] = float(np.std(arr))

    features["scalar_response_median"] = float(scalar_response)

    return features, baseline, float(scalar_response)


def build_models(seed: int) -> Dict[str, Tuple[Pipeline, Dict[str, List]]]:
    models = {}

    ridge = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(random_state=seed)),
    ])
    ridge_params = {"model__alpha": list(np.logspace(-3, 3, 20))}
    models["Ridge"] = (ridge, ridge_params)

    enet = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", ElasticNet(random_state=seed, max_iter=10000)),
    ])
    enet_params = {
        "model__alpha": list(np.logspace(-4, 1, 15)),
        "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    }
    models["ElasticNet"] = (enet, enet_params)

    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(random_state=seed)),
    ])
    rf_params = {
        "model__n_estimators": [200, 400, 800],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": [1.0, "sqrt", 0.5],
    }
    models["RandomForest"] = (rf, rf_params)

    hgbr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(random_state=seed)),
    ])
    hgbr_params = {
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [2, 3, 4, None],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__min_samples_leaf": [5, 10, 20],
    }
    models["HistGradientBoosting"] = (hgbr, hgbr_params)

    svr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVR(kernel="rbf")),
    ])
    svr_params = {
        "model__C": [1.0, 10.0, 100.0],
        "model__gamma": ["scale", "auto", 0.1, 0.01],
        "model__epsilon": [0.01, 0.1, 0.2],
    }
    models["SVR"] = (svr, svr_params)

    return models


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    if np.std(xr) == 0 or np.std(yr) == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def compute_global_metrics(y_true_ppm: np.ndarray, y_pred_ppm: np.ndarray) -> Dict[str, float]:
    eps = 1e-8
    y_pred_ppm = np.maximum(y_pred_ppm, eps)
    y_true_ppm = np.maximum(y_true_ppm, eps)

    y_true_log = np.log10(y_true_ppm)
    y_pred_log = np.log10(y_pred_ppm)

    mae_ppm = float(mean_absolute_error(y_true_ppm, y_pred_ppm))
    rmse_ppm = float(math.sqrt(mean_squared_error(y_true_ppm, y_pred_ppm)))
    mape_ppm = float(np.mean(np.abs((y_true_ppm - y_pred_ppm) / np.maximum(y_true_ppm, 1e-6))))
    mae_log = float(mean_absolute_error(y_true_log, y_pred_log))
    rmse_log = float(math.sqrt(mean_squared_error(y_true_log, y_pred_log)))
    r2_ppm = float(r2_score(y_true_ppm, y_pred_ppm))
    rho = spearman_rho(y_true_ppm, y_pred_ppm)

    return {
        "global_mae_ppm": mae_ppm,
        "global_rmse_ppm": rmse_ppm,
        "global_mape_ppm": mape_ppm,
        "global_mae_log": mae_log,
        "global_rmse_log": rmse_log,
        "global_r2": r2_ppm,
        "spearman_rho": rho,
    }


def per_concentration_errors(y_true_ppm: np.ndarray, y_pred_ppm: np.ndarray) -> Dict[float, Dict[str, float]]:
    eps = 1e-8
    results: Dict[float, Dict[str, float]] = {}
    for conc in sorted(np.unique(y_true_ppm)):
        mask = y_true_ppm == conc
        if mask.sum() == 0:
            continue
        y_true = y_true_ppm[mask]
        y_pred = y_pred_ppm[mask]
        y_true_log = np.log10(np.maximum(y_true, eps))
        y_pred_log = np.log10(np.maximum(y_pred, eps))
        results[float(conc)] = {
            "mae_ppm": float(mean_absolute_error(y_true, y_pred)),
            "mae_log": float(mean_absolute_error(y_true_log, y_pred_log)),
            "count": int(mask.sum()),
        }
    return results


def fit_calibration_baseline(logS: np.ndarray, logY: np.ndarray) -> Tuple[float, float, bool]:
    if len(logS) < 2 or np.std(logY) == 0 or np.std(logS) == 0:
        return 0.0, 0.0, False
    b, a = np.polyfit(logY, logS, 1)
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-6:
        return 0.0, 0.0, False
    return a, b, True


def predict_calibration(logS: np.ndarray, a: float, b: float) -> np.ndarray:
    return (logS - a) / b


def get_inner_cv(groups_train: np.ndarray) -> Optional[GroupKFold]:
    unique_groups = np.unique(groups_train)
    if len(unique_groups) < 2:
        return None
    n_splits = min(4, len(unique_groups))
    return GroupKFold(n_splits=n_splits)


def run_ml_fold_search(
    estimator: Pipeline,
    params: Dict[str, List],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    seed: int,
) -> Pipeline:
    inner_cv = get_inner_cv(groups_train)
    if inner_cv is None:
        model = clone(estimator)
        model.fit(X_train, y_train)
        return model

    n_iter = min(10, max(1, sum(len(v) for v in params.values())))
    search = RandomizedSearchCV(
        estimator,
        param_distributions=params,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=inner_cv,
        random_state=seed,
        n_jobs=-1,
    )
    search.fit(X_train, y_train, groups=groups_train)
    return search.best_estimator_


def run_scheme(
    scheme: str,
    groups: np.ndarray,
    dataset: pd.DataFrame,
    X: pd.DataFrame,
    y_log: np.ndarray,
    y_ppm: np.ndarray,
    scalar_response: np.ndarray,
    models: Dict[str, Tuple[Pipeline, Dict[str, List]]],
    seed: int,
    use_log_target: bool,
) -> Tuple[List[Dict[str, float]], List[Dict[str, str]], Dict[str, np.ndarray]]:
    cv = GroupKFold(n_splits=len(np.unique(groups)))
    metrics_rows: List[Dict[str, float]] = []
    fold_predictions: List[Dict[str, str]] = []
    predictions_by_model: Dict[str, np.ndarray] = {}

    def add_fold_predictions(
        model_name: str,
        fold_id: int,
        heldout_conc: Optional[float],
        test_idx: np.ndarray,
        y_pred_log: np.ndarray,
    ) -> None:
        for idx, pred_log in zip(test_idx, y_pred_log):
            y_true_ppm_val = y_ppm[idx]
            y_pred_ppm_val = float(np.power(10.0, pred_log))
            fold_predictions.append({
                "scheme": scheme,
                "fold_id": str(fold_id),
                "heldout_concentration": "" if heldout_conc is None else str(heldout_conc),
                "sample_id": dataset.iloc[idx]["sample_id"],
                "y_true_ppm": f"{y_true_ppm_val:.6f}",
                "y_pred_ppm": f"{y_pred_ppm_val:.6f}",
                "y_true_log": f"{y_log[idx]:.6f}",
                "y_pred_log": f"{pred_log:.6f}",
                "model_name": model_name,
            })

    def finalize_model(model_name: str, preds_log: np.ndarray) -> None:
        preds_ppm = np.power(10.0, preds_log)
        metrics = compute_global_metrics(y_ppm, preds_ppm)
        per_conc = per_concentration_errors(y_ppm, preds_ppm)
        row = {
            "scheme": scheme,
            "model_name": model_name,
            **metrics,
        }
        for conc, stats in per_conc.items():
            row[f"mae_log_{int(conc)}ppm"] = stats["mae_log"]
        metrics_rows.append(row)
        predictions_by_model[model_name] = preds_ppm

    # Baseline 1: constant-per-fold
    preds_log = np.zeros_like(y_log)
    for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y_log, groups)):
        heldout = None
        if scheme == "loco":
            heldout = float(groups[test_idx][0])
        mean_log = float(np.mean(y_log[train_idx]))
        preds_log[test_idx] = mean_log
        add_fold_predictions("Baseline_Constant", fold_id, heldout, test_idx, preds_log[test_idx])
    finalize_model("Baseline_Constant", preds_log)

    # Baseline 2: calibration log-log
    preds_log = np.zeros_like(y_log)
    eps = 1e-8
    for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y_log, groups)):
        heldout = None
        if scheme == "loco":
            heldout = float(groups[test_idx][0])
        logS_train = np.log10(np.maximum(scalar_response[train_idx], eps))
        logY_train = y_log[train_idx]
        a, b, ok = fit_calibration_baseline(logS_train, logY_train)
        if not ok:
            mean_log = float(np.mean(logY_train))
            preds_log[test_idx] = mean_log
        else:
            logS_test = np.log10(np.maximum(scalar_response[test_idx], eps))
            preds_log[test_idx] = predict_calibration(logS_test, a, b)
        add_fold_predictions("Baseline_Calibration", fold_id, heldout, test_idx, preds_log[test_idx])
    finalize_model("Baseline_Calibration", preds_log)

    # Baseline 3: isotonic
    preds_log = np.zeros_like(y_log)
    for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y_log, groups)):
        heldout = None
        if scheme == "loco":
            heldout = float(groups[test_idx][0])
        x_train = scalar_response[train_idx]
        y_train = y_log[train_idx]
        if np.std(x_train) < 1e-8:
            mean_log = float(np.mean(y_train))
            preds_log[test_idx] = mean_log
        else:
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            iso.fit(x_train, y_train)
            preds_log[test_idx] = iso.predict(scalar_response[test_idx])
        add_fold_predictions("Baseline_Isotonic", fold_id, heldout, test_idx, preds_log[test_idx])
    finalize_model("Baseline_Isotonic", preds_log)

    # ML models
    for model_name, (estimator, params) in models.items():
        preds_log = np.zeros_like(y_log)
        for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y_log, groups)):
            heldout = None
            if scheme == "loco":
                heldout = float(groups[test_idx][0])
            train_target = y_log[train_idx] if use_log_target else y_ppm[train_idx]
            model = run_ml_fold_search(
                estimator,
                params,
                X.iloc[train_idx],
                train_target,
                groups[train_idx],
                seed,
            )
            preds = model.predict(X.iloc[test_idx])
            if use_log_target:
                preds_log[test_idx] = preds
            else:
                preds_log[test_idx] = np.log10(np.maximum(preds, 1e-8))
            add_fold_predictions(model_name, fold_id, heldout, test_idx, preds_log[test_idx])
        finalize_model(model_name, preds_log)

    # Calibration + residual ML (Ridge)
    residual_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(random_state=seed)),
    ])
    residual_params = {"model__alpha": list(np.logspace(-3, 2, 10))}

    preds_log = np.zeros_like(y_log)
    for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y_log, groups)):
        heldout = None
        if scheme == "loco":
            heldout = float(groups[test_idx][0])
        logS_train = np.log10(np.maximum(scalar_response[train_idx], eps))
        logY_train = y_log[train_idx]
        a, b, ok = fit_calibration_baseline(logS_train, logY_train)
        if not ok:
            mean_log = float(np.mean(logY_train))
            calib_train = np.full_like(logY_train, mean_log)
            calib_test = np.full(len(test_idx), mean_log)
        else:
            calib_train = predict_calibration(logS_train, a, b)
            logS_test = np.log10(np.maximum(scalar_response[test_idx], eps))
            calib_test = predict_calibration(logS_test, a, b)

        residuals_train = logY_train - calib_train
        model = run_ml_fold_search(
            residual_model,
            residual_params,
            X.iloc[train_idx],
            residuals_train,
            groups[train_idx],
            seed,
        )
        residuals_pred = model.predict(X.iloc[test_idx])
        preds_log[test_idx] = calib_test + residuals_pred
        add_fold_predictions("CalibResidual_Ridge", fold_id, heldout, test_idx, preds_log[test_idx])
    finalize_model("CalibResidual_Ridge", preds_log)

    return metrics_rows, fold_predictions, predictions_by_model


def plot_top_features_dose_response(
    dataset: pd.DataFrame,
    feature_cols: List[str],
    out_path: str,
    top_k: int = 8,
) -> None:
    log_y = dataset["concentration_log10"].to_numpy()
    corrs = []
    for col in feature_cols:
        rho = spearman_rho(dataset[col].to_numpy(), log_y)
        if np.isnan(rho):
            continue
        corrs.append((col, abs(rho)))
    corrs.sort(key=lambda x: x[1], reverse=True)
    top_features = [c for c, _ in corrs[:top_k]]
    if not top_features:
        return

    n_rows = int(math.ceil(len(top_features) / 2))
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, feat in zip(axes, top_features):
        df = dataset[["concentration_ppm", feat]]
        ax.scatter(df["concentration_ppm"], df[feat], alpha=0.6)
        stats = df.groupby("concentration_ppm")[feat].agg(["mean", "std"]).reset_index()
        ax.errorbar(stats["concentration_ppm"], stats["mean"], yerr=stats["std"], fmt="o", color="red")
        ax.set_xscale("log")
        ax.set_xlabel("Concentration (ppm)")
        ax.set_ylabel(feat)
    for ax in axes[len(top_features):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_parity_log(y_true_ppm: np.ndarray, y_pred_ppm: np.ndarray, conc: np.ndarray, out_path: str) -> None:
    eps = 1e-8
    plt.figure(figsize=(6, 6))
    plt.scatter(
        np.log10(np.maximum(y_true_ppm, eps)),
        np.log10(np.maximum(y_pred_ppm, eps)),
        c=conc,
        cmap="viridis",
        alpha=0.8,
    )
    min_val = min(np.log10(np.maximum(y_true_ppm, eps)).min(), np.log10(np.maximum(y_pred_ppm, eps)).min())
    max_val = max(np.log10(np.maximum(y_true_ppm, eps)).max(), np.log10(np.maximum(y_pred_ppm, eps)).max())
    plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=1)
    plt.xlabel("log10(True ppm)")
    plt.ylabel("log10(Predicted ppm)")
    plt.colorbar(label="Concentration (ppm)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_residuals_by_conc(y_true_ppm: np.ndarray, y_pred_ppm: np.ndarray, out_path: str) -> None:
    eps = 1e-8
    concs = sorted(np.unique(y_true_ppm))
    residuals = []
    for conc in concs:
        mask = y_true_ppm == conc
        y_true_log = np.log10(np.maximum(y_true_ppm[mask], eps))
        y_pred_log = np.log10(np.maximum(y_pred_ppm[mask], eps))
        residuals.append(y_true_log - y_pred_log)

    plt.figure(figsize=(8, 5))
    plt.boxplot(residuals, labels=[str(int(c)) for c in concs])
    plt.xlabel("True concentration (ppm)")
    plt.ylabel("Log residual (true - pred)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_readme(out_dir: str):
    readme_path = os.path.join(out_dir, "README.md")
    content = """# H2S Concentration Regression Results (Diagnostic-First)

This folder contains outputs from `run_h2s_concentration_regression.py`.

## Why fold-wise R² is invalid under LOCO

Under leave-one-concentration-out (LOCO), each test fold contains a single concentration, so the
true targets are constant within that fold. R² is undefined in that case and will be 0 or NaN.
Instead, this pipeline concatenates predictions from all LOCO folds and computes global metrics
on the aggregated predictions.

## Baseline diagnostics

- **Baseline_Constant**: predicts the train-fold mean of log10(ppm). Tests whether any model beats
  a trivial constant predictor.
- **Baseline_Calibration**: fits a log-log calibration between a scalar response and ppm on TRAIN
  only. Diagnoses whether a simple physics-inspired scalar mapping exists.
- **Baseline_Isotonic**: learns a monotonic mapping from scalar response to log10(ppm). Diagnoses
  whether a monotone dose-response signal exists without assuming a parametric form.

## Failure modes and interpretation

- **No dose information**: baselines and ML models all perform near the constant baseline.
- **Saturation**: features show non-monotonicity or flattening at high ppm.
- **Drift**: strong baseline drift or incomplete recovery features correlate with errors.
- **Normalization washout**: per-cycle normalization can remove amplitude; this pipeline keeps
  absolute and relative features simultaneously.

## Outputs

- `dataset_cycles.csv`: cycle-level features and targets (50 samples = 5 concentrations x 10 cycles)
- `metrics.csv`: global metrics for LOCO and within-concentration schemes
- `fold_predictions.csv`: per-sample predictions for all models and schemes
- `per_concentration_errors.csv`: MAE by true concentration
- `baseline_comparison.csv`: baseline vs ML comparisons
- `feature_monotonicity.csv`: Spearman monotonicity scores per feature
- `diagnostics_dose_response.png`: top features vs concentration
- `diagnostics_parity_log.png`: log-space parity plot
- `diagnostics_residuals_by_conc.png`: residuals by concentration
- `best_model.joblib`: best ML model artifact
- `feature_schema.json`: feature list and metadata
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(description="H2S concentration regression pipeline")
    parser.add_argument("--data_dir", default="/home/chunzhi/Desktop/nose/Network/data")
    parser.add_argument("--out_dir", default="./results_h2s_regression")
    parser.add_argument("--exposure_s", type=float, default=60)
    parser.add_argument("--purge_s", type=float, default=120)
    parser.add_argument("--n_cycles", type=int, default=10)
    parser.add_argument("--baseline_tail_s", type=float, default=10)
    parser.add_argument("--smooth", choices=["none", "savgol", "median"], default="savgol")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--use_log_target",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Train on log10(ppm) targets for ML models",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    datasets = load_data(args.data_dir)

    all_rows = []
    for fname, df, concentration, dt in datasets:
        cycles = segment_cycles(
            df,
            dt,
            args.exposure_s,
            args.purge_s,
            args.n_cycles,
            file_id=fname,
            concentration_ppm=concentration,
        )

        prev_baseline = None
        for cycle in cycles:
            features, baseline, scalar_response = extract_features(
                cycle, args.baseline_tail_s, args.smooth, prev_baseline
            )
            row = {
                **features,
                "concentration_ppm": cycle.concentration_ppm,
                "concentration_log10": math.log10(cycle.concentration_ppm),
                "file_id": cycle.file_id,
                "cycle_id": cycle.cycle_id,
                "sample_id": f"{cycle.file_id}_c{cycle.cycle_id}",
                "sensor_count": cycle.sensor_count,
                "dt": cycle.dt,
                "scalar_response": scalar_response,
            }
            all_rows.append(row)
            prev_baseline = baseline

    dataset = pd.DataFrame(all_rows)
    expected_samples = args.n_cycles * dataset["concentration_ppm"].nunique()
    if dataset.shape[0] != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} samples, got {dataset.shape[0]} samples."
        )

    dataset_path = os.path.join(args.out_dir, "dataset_cycles.csv")
    dataset.to_csv(dataset_path, index=False)

    feature_cols = [c for c in dataset.columns if c not in [
        "concentration_ppm",
        "concentration_log10",
        "file_id",
        "cycle_id",
        "sample_id",
        "sensor_count",
        "dt",
        "scalar_response",
    ]]

    X = dataset[feature_cols]
    y_ppm = dataset["concentration_ppm"].to_numpy(dtype=float)
    y_log = dataset["concentration_log10"].to_numpy(dtype=float)
    scalar_response = dataset["scalar_response"].to_numpy(dtype=float)

    models = build_models(args.seed)

    metrics_rows = []
    fold_predictions_rows = []
    predictions_cache = {}

    # Scheme 1: LOCO
    groups_loco = dataset["concentration_ppm"].to_numpy()
    loco_metrics, loco_folds, loco_preds = run_scheme(
        "loco",
        groups_loco,
        dataset,
        X,
        y_log,
        y_ppm,
        scalar_response,
        models,
        args.seed,
        args.use_log_target,
    )
    metrics_rows.extend(loco_metrics)
    fold_predictions_rows.extend(loco_folds)
    predictions_cache["loco"] = loco_preds

    # Scheme 2: within-concentration cycle holdout
    groups_cycle = dataset["cycle_id"].to_numpy()
    within_metrics, within_folds, within_preds = run_scheme(
        "within_conc",
        groups_cycle,
        dataset,
        X,
        y_log,
        y_ppm,
        scalar_response,
        models,
        args.seed,
        args.use_log_target,
    )
    metrics_rows.extend(within_metrics)
    fold_predictions_rows.extend(within_folds)
    predictions_cache["within_conc"] = within_preds

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv_path = os.path.join(args.out_dir, "metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)

    metrics_json_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_rows, f, indent=2)

    fold_predictions_df = pd.DataFrame(fold_predictions_rows)
    fold_predictions_path = os.path.join(args.out_dir, "fold_predictions.csv")
    fold_predictions_df.to_csv(fold_predictions_path, index=False)

    # Per-concentration errors (LOCO)
    per_conc_rows = []
    for model_name, preds in predictions_cache["loco"].items():
        per_conc = per_concentration_errors(y_ppm, preds)
        for conc, stats in per_conc.items():
            per_conc_rows.append({
                "scheme": "loco",
                "model_name": model_name,
                "concentration_ppm": conc,
                "mae_ppm": stats["mae_ppm"],
                "mae_log": stats["mae_log"],
                "count": stats["count"],
            })
    per_conc_df = pd.DataFrame(per_conc_rows)
    per_conc_df.to_csv(os.path.join(args.out_dir, "per_concentration_errors.csv"), index=False)

    # Baseline comparison
    baseline_models = ["Baseline_Constant", "Baseline_Calibration", "Baseline_Isotonic"]
    baseline_rows = metrics_df[metrics_df["model_name"].isin(baseline_models + list(models.keys()) + ["CalibResidual_Ridge"])].copy()
    baseline_rows.to_csv(os.path.join(args.out_dir, "baseline_comparison.csv"), index=False)

    # Feature monotonicity
    mono_rows = []
    for col in feature_cols:
        rho = spearman_rho(dataset[col].to_numpy(), y_log)
        mono_rows.append({"feature": col, "spearman_rho": rho})
    mono_df = pd.DataFrame(mono_rows).sort_values(by="spearman_rho", key=lambda s: s.abs(), ascending=False)
    mono_df.to_csv(os.path.join(args.out_dir, "feature_monotonicity.csv"), index=False)

    # Diagnostics plots
    plot_top_features_dose_response(
        dataset,
        feature_cols,
        os.path.join(args.out_dir, "diagnostics_dose_response.png"),
        top_k=8,
    )

    # Best LOCO model selection by MAE
    loco_metrics_df = metrics_df[metrics_df["scheme"] == "loco"].copy()
    best_row = loco_metrics_df.sort_values(by="global_mae_ppm").iloc[0]
    best_model_name = best_row["model_name"]
    best_preds = predictions_cache["loco"][best_model_name]

    ml_model_names = list(models.keys()) + ["CalibResidual_Ridge"]
    ml_metrics_df = loco_metrics_df[loco_metrics_df["model_name"].isin(ml_model_names)].copy()
    best_ml_row = ml_metrics_df.sort_values(by="global_mae_ppm").iloc[0]
    best_ml_name = best_ml_row["model_name"]

    best_ml_preds = predictions_cache["loco"][best_ml_name]
    plot_parity_log(
        y_ppm,
        best_ml_preds,
        y_ppm,
        os.path.join(args.out_dir, "diagnostics_parity_log.png"),
    )
    plot_residuals_by_conc(
        y_ppm,
        best_ml_preds,
        os.path.join(args.out_dir, "diagnostics_residuals_by_conc.png"),
    )

    # Save best model artifact (retrain on full data)
    best_estimator = None
    if best_ml_name in models:
        estimator, params = models[best_ml_name]
        target = y_log if args.use_log_target else y_ppm
        best_estimator = run_ml_fold_search(
            estimator,
            params,
            X,
            target,
            groups_loco,
            args.seed,
        )
    elif best_ml_name == "CalibResidual_Ridge":
        residual_estimator = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(random_state=args.seed)),
        ])
        residual_estimator.fit(X, y_log)
        logS_full = np.log10(np.maximum(scalar_response, 1e-8))
        a_full, b_full, ok = fit_calibration_baseline(logS_full, y_log)
        best_estimator = {
            "type": "calib_residual_ridge",
            "calibration": {"a": float(a_full), "b": float(b_full), "ok": bool(ok)},
            "residual_model": residual_estimator,
        }

    if best_estimator is not None:
        joblib.dump(best_estimator, os.path.join(args.out_dir, "best_model.joblib"))

    schema_path = os.path.join(args.out_dir, "feature_schema.json")
    schema = {
        "feature_columns": feature_cols,
        "target": "log10(ppm)" if args.use_log_target else "ppm",
        "best_model": best_ml_name,
        "sensor_count": int(dataset["sensor_count"].iloc[0]),
        "dt_seconds_unique": sorted({float(x) for x in dataset["dt"].unique()}),
    }
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    write_readme(args.out_dir)

    print("Run complete")
    print(f"Best LOCO model (overall): {best_model_name}")
    print(f"Best LOCO model (ML): {best_ml_name}")
    print(f"Best LOCO MAE (ppm): {best_row['global_mae_ppm']:.4f}")
    print(f"Outputs saved to: {args.out_dir}")

    # Baseline comparison check
    calib_mae = loco_metrics_df[loco_metrics_df["model_name"] == "Baseline_Calibration"]["global_mae_ppm"].min()
    if not math.isnan(calib_mae) and best_ml_row["global_mae_ppm"] >= calib_mae:
        print("Warning: ML does NOT beat calibration baseline under LOCO.")


if __name__ == "__main__":
    main()
