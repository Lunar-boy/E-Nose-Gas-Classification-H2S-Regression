#!/usr/bin/env python3
"""Standalone Global performance dumbbell plot, styled to match the standalone PCA figure."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

from mpl_nimbus_roman import set_global_font


CSV_PATH = "/home/nate/Desktop/E-Nose-Gas-Classification-H2S-Regression/data/baseline_comparison.csv"
OUTPUT_PNG = "figure_global_performance_only.png"
OUTPUT_PDF = "figure_global_performance_only.pdf"

NAME_ABBREV = {
    "HistGradientBoosting": "HGB",
    "RandomForest": "RF",
    "SVR": "SVR",
    "ElasticNet": "EN",
    "Ridge": "Ridge",
    "CalibResidual_Ridge": "CR-Ridge",
    "Baseline_Isotonic": "Iso",
    "Baseline_Calibration": "Calib",
    "Baseline_Constant": "Const",
}


def warn(msg: str) -> None:
    print(f"Warning: {msg}", file=sys.stderr)


def format_val(val: float, decimals: int = 3, sci_threshold: float = 1e4) -> str:
    if pd.isna(val):
        return "–"
    if abs(val) >= sci_threshold:
        return f"{val:.2e}"
    return f"{val:.{decimals}f}"


def abbreviate_model(name: str, auto_log: set[str]) -> str:
    if name in NAME_ABBREV:
        return NAME_ABBREV[name]
    tokens = [t for t in "".join([c if c.isalnum() else " " for c in name]).split() if t]
    letters = "".join([t[0].upper() for t in tokens if t[0].isalpha()])
    abbrev = letters if letters else name[:10]
    abbrev = abbrev[:10]
    auto_log.add(name)
    return abbrev


def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "model_name" not in df.columns:
        raise ValueError("Missing required column: model_name")

    if "scheme" in df.columns:
        schemes = df["scheme"].dropna().astype(str).unique().tolist()
        if "within_conc" in schemes:
            df = df[df["scheme"] == "within_conc"].copy()
        elif schemes:
            warn(f"'within_conc' scheme not found; using scheme='{schemes[0]}'.")
            df = df[df["scheme"] == schemes[0]].copy()
        else:
            warn("Scheme column is empty; using full table.")
    else:
        warn("No 'scheme' column found; using full table.")

    return df


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def pick_best_mae(df: pd.DataFrame, used: set[str]) -> tuple[str | None, pd.Series | None]:
    sorted_df = df.sort_values("global_mae_log", ascending=True, na_position="last")
    for _, row in sorted_df.iterrows():
        name = str(row["model_name"])
        if name not in used:
            return name, row
    return None, None


def pick_best_r2(df: pd.DataFrame, used: set[str]) -> tuple[str | None, pd.Series | None]:
    sorted_df = df.sort_values(
        ["global_r2", "global_rmse_log", "global_mae_log"],
        ascending=[False, True, True],
        na_position="last",
    )
    for _, row in sorted_df.iterrows():
        name = str(row["model_name"])
        if name not in used:
            return name, row
    return None, None


def pick_best_linear(
    df: pd.DataFrame,
    used: set[str],
) -> tuple[str | None, pd.Series | None, str | None]:
    candidates = df[df["model_name"].isin(["Ridge", "ElasticNet"])]
    if candidates.empty:
        candidates = df[df["model_name"].str.contains("ridge|elastic", case=False, na=False)]
        if candidates.empty:
            return None, None, "No Ridge/ElasticNet found; falling back."

    candidates = candidates.sort_values("global_mae_log", ascending=True, na_position="last")
    for _, row in candidates.iterrows():
        name = str(row["model_name"])
        if name not in used:
            return name, row, None
    return None, None, "Ridge/ElasticNet already selected; falling back."


def pick_constant(
    df: pd.DataFrame,
    used: set[str],
) -> tuple[str | None, pd.Series | None, str | None]:
    candidates = df[df["model_name"] == "Baseline_Constant"]
    if candidates.empty:
        candidates = df[df["model_name"].str.contains("constant", case=False, na=False)]
    if candidates.empty:
        candidates = df[df["model_name"].str.contains("baseline", case=False, na=False)]
    if candidates.empty:
        return None, None, "Baseline_Constant not found; falling back to worst model."

    candidates = candidates.sort_values("global_mae_log", ascending=False, na_position="last")
    for _, row in candidates.iterrows():
        name = str(row["model_name"])
        if name not in used:
            return name, row, None
    return None, None, "Baseline_Constant already selected; falling back."


def select_representative_models(df_sorted: pd.DataFrame) -> list[str]:
    """Keep the representative-model selection logic from the original main figure."""
    used: set[str] = set()
    selections: dict[str, str | None] = {}

    best_mae_name, _ = pick_best_mae(df_sorted, used)
    if best_mae_name is None:
        raise ValueError("Failed to select best MAE model.")
    selections["Best MAE (lowest global_mae_log)"] = best_mae_name
    used.add(best_mae_name)

    best_r2_name, _ = pick_best_r2(df_sorted, used)
    if best_r2_name is None:
        warn("No distinct best R2 found; falling back to next-best MAE.")
        best_r2_name, _ = pick_best_mae(df_sorted, used)
    selections["Best R2 (highest global_r2)"] = best_r2_name
    if best_r2_name:
        used.add(best_r2_name)

    best_linear_name, _, msg = pick_best_linear(df_sorted, used)
    if msg:
        warn(msg)
    if best_linear_name is None:
        warn("Falling back to next-best MAE for linear baseline.")
        best_linear_name, _ = pick_best_mae(df_sorted, used)
    selections["Best linear baseline (Ridge/ElasticNet)"] = best_linear_name
    if best_linear_name:
        used.add(best_linear_name)

    constant_name, _, msg = pick_constant(df_sorted, used)
    if msg:
        warn(msg)
    if constant_name is None:
        warn("Falling back to worst MAE for trivial baseline.")
        fallback = df_sorted.sort_values("global_mae_log", ascending=False, na_position="last")
        for _, row in fallback.iterrows():
            name = str(row["model_name"])
            if name not in used:
                constant_name = name
                break
    selections["Trivial baseline (Baseline_Constant)"] = constant_name
    if constant_name:
        used.add(constant_name)

    highlight_names = [name for name in selections.values() if name is not None]

    if len(set(highlight_names)) < 4:
        warn("Representative models are not unique; filling with next-best MAE models.")
        for _, row in df_sorted.iterrows():
            name = str(row["model_name"])
            if name not in used:
                highlight_names.append(name)
                used.add(name)
            if len(set(highlight_names)) >= 4:
                break
        highlight_names = list(dict.fromkeys(highlight_names))[:4]

    print("Selected representative models:")
    for role, name in selections.items():
        if name is None:
            print(f" - {role}: (missing)")
            continue
        row = df_sorted[df_sorted["model_name"] == name].iloc[0]
        print(
            f" - {role}: {name} "
            f"(global_mae_log={format_val(row.get('global_mae_log'))}, "
            f"global_r2={format_val(row.get('global_r2'))})"
        )

    return highlight_names


def main() -> None:
    df = load_data(CSV_PATH)
    coerce_numeric(
        df,
        ["global_mae_log", "global_rmse_log", "global_r2"],
    )

    required = ["global_mae_log", "global_rmse_log", "global_r2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        raise ValueError("No rows available after filtering; cannot plot.")

    df_sorted = df.sort_values("global_mae_log", ascending=True, na_position="last").reset_index(drop=True)
    highlight_names = select_representative_models(df_sorted)
    highlight_set = set(highlight_names)

    # Match the standalone PCA figure's overall typography and scale.
    matplotlib.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 0.8,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    set_global_font()

    # Same canvas size as the standalone PCA figure.
    fig, ax = plt.subplots(figsize=(22, 13))

    mae_color = "#4C72B0"
    rmse_color = "#DD8452"
    context_color = "#9A9A9A"
    highlight_color = "#333333"

    y = np.arange(len(df_sorted))
    mae = df_sorted["global_mae_log"].to_numpy(dtype=float)
    rmse = df_sorted["global_rmse_log"].to_numpy(dtype=float)
    model_names = df_sorted["model_name"].astype(str).tolist()

    auto_abbrev: set[str] = set()
    display_names = [abbreviate_model(name, auto_abbrev) for name in model_names]
    if auto_abbrev:
        warn(f"Auto-abbreviated model names: {', '.join(sorted(auto_abbrev))}")

    # Context models: larger than in the composite figure, but still visually de-emphasized.
    for idx, name in enumerate(model_names):
        if name in highlight_set:
            continue
        ax.plot(
            [mae[idx], rmse[idx]],
            [y[idx], y[idx]],
            color=context_color,
            alpha=0.18,
            lw=3.0,
            zorder=1,
        )
        ax.scatter([mae[idx]], [y[idx]], color=context_color, alpha=0.18, s=180, zorder=2)
        ax.scatter([rmse[idx]], [y[idx]], color=context_color, alpha=0.18, s=180, zorder=2)

    # Highlight models.
    for idx, name in enumerate(model_names):
        if name not in highlight_set:
            continue
        ax.plot(
            [mae[idx], rmse[idx]],
            [y[idx], y[idx]],
            color=highlight_color,
            alpha=1.0,
            lw=5.0,
            zorder=3,
        )
        ax.scatter(
            [mae[idx]],
            [y[idx]],
            color=mae_color,
            edgecolor="white",
            linewidth=2.0,
            s=420,
            zorder=4,
        )
        ax.scatter(
            [rmse[idx]],
            [y[idx]],
            color=rmse_color,
            edgecolor="white",
            linewidth=2.0,
            s=420,
            zorder=4,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(display_names)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    # PCA-style thick dark frame.
    for spine in ax.spines.values():
        spine.set_linewidth(5)
        spine.set_color("0.2")

    # PCA-style ticks.
    ax.tick_params(
        axis="both",
        which="major",
        width=5,
        length=10,
        labelsize=36,
        pad=11,
    )
    ax.tick_params(axis="both", which="minor", width=5)

    for label, full_name in zip(ax.get_yticklabels(), model_names):
        label.set_fontweight("bold")
        if full_name in highlight_set:
            label.set_color("#000000")
        else:
            label.set_color("#7A7A7A")

    for label in ax.get_xticklabels():
        label.set_fontweight("bold")

    # Leave room on the right for R^2 annotations.
    x_min = np.nanmin([np.nanmin(mae), np.nanmin(rmse)])
    x_max = np.nanmax([np.nanmax(mae), np.nanmax(rmse)])
    span = x_max - x_min
    x_pad = 0.08 * span if np.isfinite(span) and span > 0 else 0.1
    ann_x = x_max + x_pad * 0.65
    ax.set_xlim(x_min - x_pad * 0.35, x_max + x_pad * 3.3)

    for name in highlight_names:
        row = df_sorted[df_sorted["model_name"] == name].iloc[0]
        yy = model_names.index(name)
        ax.text(
            ann_x,
            yy,
            fr"$\mathbf{{R^2}}$ {format_val(row.get('global_r2'))}",
            ha="left",
            va="center",
            fontsize=28,
            color="#333333",
            fontweight="bold",
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=mae_color,
            markeredgecolor=mae_color,
            markersize=14,
            label="MAE (typical)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=rmse_color,
            markeredgecolor=rmse_color,
            markersize=14,
            label="RMSE (tail-sensitive)",
        ),
    ]

    legend = ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=30,
        handletextpad=0.5,
        columnspacing=1.8,
    )
    legend.set_in_layout(True)
    for text in legend.get_texts():
        text.set_fontweight("bold")

    ax.set_title(
        "Global performance",
        fontsize=40,
        fontweight="bold",
        pad=72,
    )

    # Same generous standalone margins as the PCA figure, adjusted for long y labels.
    fig.subplots_adjust(
        left=0.18,
        right=0.97,
        top=0.80,
        bottom=0.14,
    )

    fig.savefig(
        OUTPUT_PNG,
        dpi=300,
        bbox_inches="tight",
        bbox_extra_artists=(legend,),
    )
    fig.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
        bbox_extra_artists=(legend,),
    )
    plt.close(fig)

    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
