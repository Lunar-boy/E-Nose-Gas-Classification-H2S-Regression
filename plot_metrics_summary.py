#!/usr/bin/env python3
"""Publication-grade visualization for metrics summary."""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image


PREFERRED_METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
ABBREV_MAP = {
    "accuracy": "Acc",
    "precision_macro": "Prec",
    "recall_macro": "Rec",
    "f1_macro": "F1",
}


def load_and_validate(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"model", "level"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Basic sanity check on metric ranges if present
    metric_cols = [c for c in df.columns if c not in ("model", "level")]
    for col in metric_cols:
        if df[col].dtype.kind in "biufc":
            if ((df[col] < 0) | (df[col] > 1)).any():
                print(f"Warning: values outside [0,1] detected in {col}")
                break
    return df


def detect_metrics(df: pd.DataFrame):
    base_metrics = [
        c for c in df.columns
        if c not in ("model", "level") and not c.endswith("_std")
    ]
    ordered = [m for m in PREFERRED_METRICS if m in base_metrics]
    extras = sorted([m for m in base_metrics if m not in ordered])
    metrics = ordered + extras
    std_map = {m: f"{m}_std" if f"{m}_std" in df.columns else None for m in metrics}
    return metrics, std_map


def abbreviate_metric(metric: str) -> str:
    if metric in ABBREV_MAP:
        return ABBREV_MAP[metric]
    parts = metric.split("_")
    if len(parts) > 1:
        return "".join(p[:1].upper() for p in parts if p)
    return metric[:4].capitalize()


def _aggregate_by_model(df_level: pd.DataFrame) -> pd.DataFrame:
    # If duplicates exist, average numeric values.
    grouped = df_level.groupby("model", as_index=False).mean(numeric_only=True)
    return grouped.set_index("model")


def sort_models(df: pd.DataFrame, metrics: list[str]):
    run_df = _aggregate_by_model(df[df["level"] == "run"])
    win_df = _aggregate_by_model(df[df["level"] == "window"])

    models = sorted(set(df["model"].dropna().unique()))
    rows = []
    for model in models:
        run_f1 = run_df.loc[model, "f1_macro"] if "f1_macro" in run_df.columns and model in run_df.index else np.nan
        run_acc = run_df.loc[model, "accuracy"] if "accuracy" in run_df.columns and model in run_df.index else np.nan
        win_f1 = win_df.loc[model, "f1_macro"] if "f1_macro" in win_df.columns and model in win_df.index else np.nan
        win_acc = win_df.loc[model, "accuracy"] if "accuracy" in win_df.columns and model in win_df.index else np.nan
        sort_f1 = run_f1 if not np.isnan(run_f1) else win_f1
        sort_acc = run_acc if not np.isnan(run_acc) else win_acc
        rows.append({
            "model": model,
            "run_f1_macro": run_f1,
            "run_accuracy": run_acc,
            "sort_f1": sort_f1,
            "sort_acc": sort_acc,
        })

    sort_df = pd.DataFrame(rows)
    # Missing values should sort last.
    sort_df["sort_f1"] = sort_df["sort_f1"].fillna(-1.0)
    sort_df["sort_acc"] = sort_df["sort_acc"].fillna(-1.0)
    sort_df = sort_df.sort_values(["sort_f1", "sort_acc"], ascending=[False, False])
    model_order = sort_df["model"].tolist()

    print("Detected models:", ", ".join(model_order))
    print("Detected levels:", ", ".join(sorted(df["level"].dropna().unique())))
    print("Sorting table (run-level metrics):")
    display_df = sort_df[["model", "run_f1_macro", "run_accuracy"]].copy()
    with pd.option_context("display.max_rows", None, "display.float_format", "{:.6f}".format):
        print(display_df.to_string(index=False))

    return model_order


def prepare_matrices(df: pd.DataFrame, metrics: list[str], std_map: dict, model_order: list[str], levels: list[str]):
    means = {}
    stds = {}
    for level in levels:
        level_df = _aggregate_by_model(df[df["level"] == level])
        mean_mat = np.full((len(model_order), len(metrics)), np.nan, dtype=float)
        std_mat = np.full((len(model_order), len(metrics)), np.nan, dtype=float)
        for i, model in enumerate(model_order):
            if model not in level_df.index:
                continue
            row = level_df.loc[model]
            for j, metric in enumerate(metrics):
                if metric in row.index:
                    mean_mat[i, j] = row[metric]
                std_col = std_map.get(metric)
                if std_col and std_col in row.index:
                    std_mat[i, j] = row[std_col]
        means[level] = mean_mat
        stds[level] = std_mat
    return means, stds


def _apply_heatmap_grid(ax, n_rows: int, n_cols: int):
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="#FFFFFF", linestyle="-", linewidth=0.4, alpha=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)


def plot_top_heatmaps(fig, gs, metrics, models, means, stds, cmap="viridis"):
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    axes = [("window", ax_left, "Window-level (mean ± std)"),
            ("run", ax_right, "Run-level (mean ± std)")]

    shared_im = None
    for level, ax, title in axes:
        if level not in means:
            ax.text(0.5, 0.5, f"{level} level missing", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        data = means[level]
        std_data = stds.get(level)
        im = ax.imshow(
            data,
            vmin=0,
            vmax=1,
            cmap=cmap,
            interpolation="nearest",
            aspect="auto",
            origin="upper",
        )
        if shared_im is None:
            shared_im = im

        _apply_heatmap_grid(ax, len(models), len(metrics))
        metric_labels = [abbreviate_metric(m) for m in metrics]
        ax.set_xticklabels(metric_labels, rotation=0, ha="center")
        ax.set_yticklabels(models)
        ax.set_title(title)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                mean_val = data[i, j]
                std_val = std_data[i, j] if std_data is not None else np.nan
                if np.isnan(mean_val):
                    ax.text(j, i, "NA", ha="center", va="center", fontsize=8, color="black")
                    continue
                color = "white" if mean_val >= 0.6 else "black"
                if level == "run" and models[i] == "LSTM" and metrics[j] == "precision_macro":
                    color = "white"
                if not np.isnan(std_val):
                    ax.text(j, i - 0.12, f"{mean_val:.3f}", ha="center", va="center", fontsize=8, color=color)
                    ax.text(j, i + 0.18, f"±{std_val:.3f}", ha="center", va="center", fontsize=6, color=color)
                else:
                    ax.text(j, i, f"{mean_val:.3f}", ha="center", va="center", fontsize=8, color=color)

    if shared_im is not None:
        cbar = fig.colorbar(shared_im, ax=[ax_left, ax_right], fraction=0.046, pad=0.02)
        cbar.set_label("Mean score")
        if getattr(cbar, "solids", None) is not None:
            cbar.solids.set_rasterized(True)

    return ax_left, ax_right


def plot_delta_panel(fig, gs, metrics, models, means, stds, delta_style="heatmap"):
    if "window" not in means or "run" not in means:
        ax = fig.add_subplot(gs[1, :])
        ax.text(0.5, 0.5, "Delta panel skipped (missing window or run level)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    window_mean = means["window"]
    run_mean = means["run"]
    window_std = stds.get("window")
    run_std = stds.get("run")

    delta_mean = run_mean - window_mean
    if window_std is not None and run_std is not None:
        delta_std = np.sqrt(np.square(run_std) + np.square(window_std))
    else:
        delta_std = None

    # Print delta ranges per metric
    for idx, metric in enumerate(metrics):
        col = delta_mean[:, idx]
        if np.all(np.isnan(col)):
            print(f"Delta range for {metric}: all NaN")
            continue
        print(f"Delta range for {metric}: min={np.nanmin(col):.6f}, max={np.nanmax(col):.6f}")

    # Default: heatmap delta
    ax = fig.add_subplot(gs[1, :])
    vmax = np.nanmax(np.abs(delta_mean))
    if vmax == 0 or np.isnan(vmax):
        vmax = 1e-6
    norm = Normalize(vmin=-vmax, vmax=vmax)
    im = ax.imshow(
        delta_mean,
        cmap="RdBu_r",
        norm=norm,
        interpolation="nearest",
        aspect="auto",
        origin="upper",
    )

    _apply_heatmap_grid(ax, len(models), len(metrics))
    metric_labels = [abbreviate_metric(m) for m in metrics]
    ax.set_xticklabels(metric_labels, rotation=0, ha="center")
    ax.set_yticklabels(models)
    ax.set_title("Delta (run - window)")

    for i in range(delta_mean.shape[0]):
        for j in range(delta_mean.shape[1]):
            mean_val = delta_mean[i, j]
            if np.isnan(mean_val):
                ax.text(j, i, "NA", ha="center", va="center", fontsize=8)
                continue
            primary = f"{mean_val:+.3f}"
            if delta_std is not None and not np.isnan(delta_std[i, j]):
                secondary = f"±{delta_std[i, j]:.3f}"
            else:
                secondary = None

            if secondary is not None:
                color = "white" if (models[i] == "SVM" and metrics[j] == "precision_macro") else "black"
                ax.text(j, i - 0.12, primary, ha="center", va="center", fontsize=8, color=color)
                ax.text(j, i + 0.18, secondary, ha="center", va="center", fontsize=6, color=color)
            else:
                color = "white" if (models[i] == "SVM" and metrics[j] == "precision_macro") else "black"
                ax.text(j, i, primary, ha="center", va="center", fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Δ (run - window)")
    cbar.set_ticks([-vmax, 0, vmax])
    cbar.ax.set_yticklabels([f"{-vmax:.3f}", "0.000", f"{vmax:.3f}"])
    if getattr(cbar, "solids", None) is not None:
        cbar.solids.set_rasterized(True)


def save_outputs(fig, out_dir: str, export_mode: str):
    out_png = os.path.join(out_dir, "metrics_summary_viz.png")
    out_pdf_safe = os.path.join(out_dir, "metrics_summary_viz_SAFE.pdf")
    out_pdf_mixed = os.path.join(out_dir, "metrics_summary_viz_MIXED.pdf")

    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white", transparent=False)
    print(f"Saved PNG: {out_png}")

    if export_mode in ("safe", "both"):
        with Image.open(out_png) as img:
            img.convert("RGB").save(out_pdf_safe, "PDF", resolution=600.0)
        print(f"Saved SAFE PDF: {out_pdf_safe}")

    if export_mode in ("mixed", "both"):
        fig.savefig(out_pdf_mixed, dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
        print(f"Saved MIXED PDF: {out_pdf_mixed}")


def write_latex_test(out_dir: str):
    tex_path = os.path.join(out_dir, "test_include.tex")
    content = "\\\\documentclass{article}\\n\\\\usepackage{graphicx}\\n\\\\begin{document}\\n\\\\includegraphics[width=\\\\linewidth]{metrics_summary_viz_SAFE.pdf}\\n\\\\end{document}\\n"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Wrote LaTeX test file: {tex_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot metrics summary heatmaps and delta panel.")
    parser.add_argument("--csv", default="/home/chunzhi/Desktop/nose/Network/outputs_baseline_rf/metrics_summary.csv")
    parser.add_argument("--out_dir", default="/home/chunzhi/Desktop/nose/Network/outputs_baseline/")
    parser.add_argument("--delta_style", choices=["heatmap"], default="heatmap")
    parser.add_argument("--export", choices=["safe", "mixed", "both"], default="both")
    args = parser.parse_args()

    df = load_and_validate(args.csv)
    metrics, std_map = detect_metrics(df)
    model_order = sort_models(df, metrics)

    levels = sorted(df["level"].dropna().unique())
    # Only care about window/run for plots
    levels = [lvl for lvl in levels if lvl in ("window", "run")]
    if not levels:
        raise ValueError("No recognized levels found (expected 'window' and/or 'run').")

    means, stds = prepare_matrices(df, metrics, std_map, model_order, levels)

    # Figure sizing
    n_models = len(model_order)
    fig_width = max(9.5, 0.9 * len(metrics) * 2)
    fig_height = max(6.0, 3.5 + 0.4 * n_models)

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1], wspace=0.25, hspace=0.35)

    plot_top_heatmaps(fig, gs, metrics, model_order, means, stds)
    plot_delta_panel(fig, gs, metrics, model_order, means, stds, delta_style=args.delta_style)

    os.makedirs(args.out_dir, exist_ok=True)
    for ax in fig.axes:
        ax.set_facecolor("white")

    # Rasterize heatmaps for mixed export only.
    if args.export in ("mixed", "both"):
        for ax in fig.axes:
            for artist in ax.get_images():
                artist.set_rasterized(True)

    save_outputs(fig, args.out_dir, args.export)
    write_latex_test(args.out_dir)
    plt.close(fig)


if __name__ == "__main__":
    main()
