#!/usr/bin/env python3
"""
Generate fingerprint heatmap + lollipop strip and PCA scatter plot from run-level features.
"""
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

try:
    from sklearn.decomposition import PCA  # type: ignore
    from sklearn.metrics import silhouette_score  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


DATA_PATH = "/home/chunzhi/Desktop/nose/Network/all_gases_framewise.csv"
OUTPUT_PNG = "figure_fingerprint_pca.png"
OUTPUT_PDF = "figure_fingerprint_pca.pdf"

BASE_GAS_COLORS = {
    "NO": "#0000F0",
    "NH3": "#700070",
    "SO2": "#F0A000",
    "H2S": "#F0F000",
    # additional vivid, harmonized colors
    "C2H6O": "#00B0F0",
    "H2": "#00C000",
    "NO2": "#F00000",
    "PH3": "#C000F0",
}

FALLBACK_COLORS = [
    "#00A0A0",  # teal
    "#A000A0",  # magenta
    "#A0A000",  # olive
    "#00A000",  # green
    "#A00000",  # dark red
    "#000000",  # black
]


@dataclass
class VariantResult:
    name: str
    score: float
    evr_sum: float
    silhouette: float
    pca: object
    X_pca: np.ndarray
    scaler: object
    features: np.ndarray
    feature_names: List[str]


def build_colormap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "red_white_green",
        ["#D00000", "#FFFFFF", "#00A000"],
        N=256,
    )


def parse_ppm(run_id: str) -> float:
    match = re.search(r"ppm(\d+)", str(run_id))
    return float(match.group(1)) if match else np.nan


class SimpleScaler:
    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "SimpleScaler":
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.scale_ is not None
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class SimplePCA:
    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "SimplePCA":
        self.mean_ = X.mean(axis=0, keepdims=True)
        X_centered = X - self.mean_
        _, s, vt = np.linalg.svd(X_centered, full_matrices=False)
        components = vt[: self.n_components]
        n_samples = X.shape[0]
        explained_variance = (s**2) / max(n_samples - 1, 1)
        total_variance = explained_variance.sum()
        if total_variance == 0:
            total_variance = 1.0
        evr = explained_variance[: self.n_components] / total_variance
        self.components_ = components
        self.explained_variance_ratio_ = evr
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.components_ is not None
        assert self.mean_ is not None
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def silhouette_score_simple(X: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or X.shape[0] < 3:
        return 0.0
    diffs = X[:, None, :] - X[None, :, :]
    dist = np.sqrt((diffs**2).sum(axis=2))
    scores = []
    for i in range(X.shape[0]):
        same = labels == labels[i]
        same[i] = False
        if same.any():
            a = dist[i, same].mean()
        else:
            a = 0.0
        b = np.inf
        for lbl in unique_labels:
            if lbl == labels[i]:
                continue
            mask = labels == lbl
            if mask.any():
                b = min(b, dist[i, mask].mean())
        denom = max(a, b)
        if denom == 0 or np.isinf(b):
            scores.append(0.0)
        else:
            scores.append((b - a) / denom)
    return float(np.mean(scores))


def compute_run_features(df: pd.DataFrame, sensors: List[str]) -> pd.DataFrame:
    rows = []
    for run_id, g in df.groupby("run_id", sort=False):
        g = g.sort_values("t_rel")
        t_rel = g["t_rel"].to_numpy()
        t_sec = t_rel / 1000.0
        gas = g["gas_name"].iloc[0]
        ppm = g["ppm"].iloc[0]

        mask_baseline = (t_rel >= 0) & (t_rel <= 10_000)
        mask_sat = (t_rel >= 50_000) & (t_rel <= 60_000)
        mask_auc = (t_rel >= 0) & (t_rel <= 60_000)

        row: Dict[str, float] = {
            "run_id": run_id,
            "gas_name": gas,
            "ppm": ppm,
        }

        for s in sensors:
            y = g[s].to_numpy()
            if mask_baseline.any():
                baseline = y[mask_baseline].mean()
            else:
                baseline = np.nan
            y_diff = y - baseline

            sat = y_diff[mask_sat].mean() if mask_sat.any() else np.nan
            if mask_auc.any():
                auc = np.trapz(y_diff[mask_auc], t_sec[mask_auc])
                dmax = np.max(y_diff[mask_auc])
            else:
                auc = np.nan
                dmax = np.nan

            if mask_baseline.sum() >= 2:
                slope = np.polyfit(t_sec[mask_baseline], y_diff[mask_baseline], 1)[0]
            else:
                slope = np.nan

            row[f"sat_{s}"] = sat
            row[f"auc_{s}"] = auc
            row[f"slope_{s}"] = slope
            row[f"dmax_{s}"] = dmax

        rows.append(row)

    run_df = pd.DataFrame(rows)
    run_df = run_df.dropna(axis=0)
    return run_df


def get_gas_order(unique_gases: List[str]) -> List[str]:
    base_order = ["NO", "NH3", "SO2", "H2S"]
    ordered = [g for g in base_order if g in unique_gases]
    remaining = sorted([g for g in unique_gases if g not in ordered])
    return ordered + remaining


def get_gas_colors(unique_gases: List[str]) -> Dict[str, str]:
    colors = dict(BASE_GAS_COLORS)
    fallback_iter = iter(FALLBACK_COLORS)
    for gas in unique_gases:
        if gas not in colors:
            try:
                colors[gas] = next(fallback_iter)
            except StopIteration:
                colors[gas] = "#333333"
    return colors


def select_variant(
    run_df: pd.DataFrame,
    sensors: List[str],
    gas_labels: np.ndarray,
    lambda_sil: float = 0.25,
) -> VariantResult:
    variants = {
        "sat+auc+slope": (
            [f"sat_{s}" for s in sensors]
            + [f"auc_{s}" for s in sensors]
            + [f"slope_{s}" for s in sensors]
        ),
        "dmax+auc+slope": (
            [f"dmax_{s}" for s in sensors]
            + [f"auc_{s}" for s in sensors]
            + [f"slope_{s}" for s in sensors]
        ),
        "sat+auc": (
            [f"sat_{s}" for s in sensors]
            + [f"auc_{s}" for s in sensors]
        ),
    }

    best: Optional[VariantResult] = None

    for name, cols in variants.items():
        X = run_df[cols].to_numpy(dtype=float)
        if SKLEARN_AVAILABLE:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=2, random_state=0)
            X_pca = pca.fit_transform(X_scaled)
            evr_sum = float(pca.explained_variance_ratio_.sum())
        else:
            scaler = SimpleScaler()
            X_scaled = scaler.fit_transform(X)
            pca = SimplePCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            evr_sum = float(np.sum(pca.explained_variance_ratio_))

        sil = 0.0
        if len(np.unique(gas_labels)) > 1 and X_pca.shape[0] >= 3:
            try:
                if SKLEARN_AVAILABLE:
                    sil = float(silhouette_score(X_pca, gas_labels))
                else:
                    sil = float(silhouette_score_simple(X_pca, gas_labels))
            except Exception:
                sil = 0.0

        score = evr_sum + lambda_sil * sil
        result = VariantResult(
            name=name,
            score=score,
            evr_sum=evr_sum,
            silhouette=sil,
            pca=pca,
            X_pca=X_pca,
            scaler=scaler,
            features=X_scaled,
            feature_names=cols,
        )

        if best is None or result.score > best.score:
            best = result

    assert best is not None
    print(
        f"Selected variant: {best.name} | EVR sum={best.evr_sum:.3f} "
        f"| silhouette={best.silhouette:.3f} | score={best.score:.3f}"
    )
    return best


def build_heatmap_matrix(
    run_df: pd.DataFrame,
    sensors: List[str],
    ordered_run_ids: List[str],
) -> Tuple[np.ndarray, List[str]]:
    feature_cols = (
        [f"auc_{s}" for s in sensors]
        + [f"slope_{s}" for s in sensors]
        + [f"sat_{s}" for s in sensors]
    )
    heat_df = run_df.set_index("run_id").loc[ordered_run_ids, feature_cols]
    matrix = heat_df.to_numpy().T

    row_mean = matrix.mean(axis=1, keepdims=True)
    row_std = matrix.std(axis=1, keepdims=True)
    row_std[row_std == 0] = 1.0
    z = (matrix - row_mean) / row_std
    z = np.clip(z, -2, 2)
    return z, feature_cols


def size_for_ppm(ppm: float) -> float:
    return 20 + 5 * ppm


def main() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
        }
    )

    df = pd.read_csv(DATA_PATH)
    sensors = [c for c in df.columns if re.match(r"s\d{2}$", c)]
    sensors = sorted(sensors)

    df["ppm"] = df["run_id"].apply(parse_ppm)
    df["t_rel"] = df["time"] - df["t_exposure_start_ms"]

    run_df = compute_run_features(df, sensors)
    run_df = run_df.sort_values(["gas_name", "ppm", "run_id"]).reset_index(drop=True)

    unique_gases = sorted(run_df["gas_name"].unique())
    gas_order = get_gas_order(unique_gases)
    gas_colors = get_gas_colors(unique_gases)

    ordered_run_ids: List[str] = []
    gas_group_bounds: List[Tuple[str, int, int]] = []
    for gas in gas_order:
        subset = run_df[run_df["gas_name"] == gas].sort_values(["ppm", "run_id"])
        start = len(ordered_run_ids)
        ordered_run_ids.extend(subset["run_id"].tolist())
        end = len(ordered_run_ids)
        if end > start:
            gas_group_bounds.append((gas, start, end))

    heat_z, _ = build_heatmap_matrix(run_df, sensors, ordered_run_ids)

    # Prepare PCA selection
    ordered_df = run_df.set_index("run_id").loc[ordered_run_ids].reset_index()
    gas_labels = ordered_df["gas_name"].to_numpy()
    ppm_values = ordered_df["ppm"].to_numpy()

    best = select_variant(run_df, sensors, run_df["gas_name"].to_numpy())
    pca_df = run_df.copy()
    X_scaled = best.scaler.transform(pca_df[best.feature_names].to_numpy(dtype=float))
    X_pca = best.pca.transform(X_scaled)
    pca_df["PC1"] = X_pca[:, 0]
    pca_df["PC2"] = X_pca[:, 1]

    # Use the same ordering for plotting
    pca_df = pca_df.set_index("run_id").loc[ordered_run_ids].reset_index()

    # Figure layout
    fig = plt.figure(figsize=(12.5, 8.5), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.15, 1.0],
        left=0.08,
        right=0.98,
        top=0.98,
        bottom=0.08,
        hspace=0.10,
    )
    top = mpl.gridspec.GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=outer[0],
        height_ratios=[6, 2],
        width_ratios=[1.0, 0.025],
        hspace=0.12,
        wspace=0.02,
    )

    ax_heat = fig.add_subplot(top[0, 0])
    ax_cbar = fig.add_subplot(top[0, 1])
    ax_lolli = fig.add_subplot(top[1, 0], sharex=ax_heat)
    ax_lolli.set_xlim(ax_heat.get_xlim())
    ax_pca = fig.add_subplot(outer[1])

    pos_a = ax_heat.get_position()
    pos_b = ax_pca.get_position()
    x_label = pos_a.x0 - 0.02
    fig.text(x_label, pos_a.y1 + 0.005, "(a)", ha="right", va="bottom", fontsize=11, fontweight="bold")
    fig.text(x_label, pos_b.y1 + 0.005, "(b)", ha="right", va="bottom", fontsize=11, fontweight="bold")

    cmap = build_colormap()
    im = ax_heat.imshow(
        heat_z,
        aspect="auto",
        cmap=cmap,
        vmin=-2,
        vmax=2,
        interpolation="nearest",
    )

    ax_heat.set_yticks([])
    ax_heat.set_ylabel("response feature", labelpad=10)
    ax_heat.set_xticks([])

    # Gas group separators and labels
    for gas, start, end in gas_group_bounds:
        if start > 0:
            ax_heat.axvline(start - 0.5, color="0.7", lw=0.8)
        center = (start + end - 1) / 2
        ax_heat.text(
            center,
            1.02,
            gas,
            transform=ax_heat.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Colorbar
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label("transformed response")
    cbar.set_ticks([-2, -1, 0, 1, 2])
    ax_cbar.yaxis.tick_right()
    ax_cbar.yaxis.set_label_position("right")

    # Lollipop strip
    x = np.arange(len(ordered_run_ids))
    y_max = max(55, np.nanmax(ppm_values) * 1.1)
    ax_lolli.set_ylim(0, y_max)
    ax_lolli.set_xlim(-0.5, len(x) - 0.5)
    ax_heat.set_xlim(-0.5, len(x) - 0.5)

    # Vertical guide lines to align dots with heatmap columns
    ax_lolli.vlines(x, 0, y_max, color="0.9", lw=0.4, zorder=0)
    ax_lolli.vlines(x, 0, ppm_values, color="0.6", lw=0.8, zorder=1)
    for xi, ppm, gas in zip(x, ppm_values, gas_labels):
        ax_lolli.scatter(
            xi,
            ppm,
            s=30,
            color=gas_colors.get(gas, "#333333"),
            edgecolor="none",
            zorder=3,
        )
    ax_lolli.set_ylabel("ppm")
    ax_lolli.set_yticks([0, 10, 20, 50])
    ax_lolli.set_xticks([])

    # PCA scatter
    for gas in gas_order:
        subset = pca_df[pca_df["gas_name"] == gas]
        ax_pca.scatter(
            subset["PC1"],
            subset["PC2"],
            s=[size_for_ppm(v) for v in subset["ppm"]],
            c=gas_colors.get(gas, "#333333"),
            alpha=0.75,
            edgecolor="none",
            label=gas,
        )

    ax_pca.axhline(0, color="0.7", lw=0.8, zorder=0)
    ax_pca.axvline(0, color="0.7", lw=0.8, zorder=0)

    evr = best.pca.explained_variance_ratio_ * 100
    ax_pca.set_xlabel(f"PC1 ({evr[0]:.1f}%)")
    ax_pca.set_ylabel(f"PC2 ({evr[1]:.1f}%)")

    # Legends
    gas_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=gas_colors[g], label=g, markersize=6)
        for g in gas_order
    ]
    gas_legend = ax_pca.legend(
        handles=gas_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=min(8, max(1, len(gas_handles))),
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    ax_pca.add_artist(gas_legend)

    size_levels = [1, 5, 10, 20, 50]
    size_handles = [
        plt.scatter([], [], s=size_for_ppm(v), color="0.5", alpha=0.7, label=str(v))
        for v in size_levels
    ]
    ax_pca.legend(
        handles=size_handles,
       
        loc="lower center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=len(size_levels),
        frameon=False,
        handletextpad=0.6,
        columnspacing=1.2,
    )

    # Style spines
    for ax in [ax_heat, ax_lolli, ax_pca]:
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("0.2")

    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight")
    print(f"Saved {OUTPUT_PNG} and {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
