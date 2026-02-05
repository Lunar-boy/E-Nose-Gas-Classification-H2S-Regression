#!/usr/bin/env python3
"""Generate Nature-style main figure for H2S regression (4 representative models)."""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
from PIL import Image


CSV_PATH = "/home/chunzhi/Desktop/nose/Network/results_h2s_regression/baseline_comparison.csv"
OUT_PDF = "/home/chunzhi/Desktop/nose/Network/results_h2s_regression/fig_h2s_regression_main4_v3.pdf"
OUT_PNG = "/home/chunzhi/Desktop/nose/Network/results_h2s_regression/fig_h2s_regression_main4_v3.png"
OUT_TEX = "/home/chunzhi/Desktop/nose/Network/results_h2s_regression/tab_h2s_main4.tex"
OUT_SUMMARY = "/home/chunzhi/Desktop/nose/Network/results_h2s_regression/summary_h2s_ranked.csv"

DOSE_COLS = [
    "mae_log_1ppm",
    "mae_log_5ppm",
    "mae_log_10ppm",
    "mae_log_20ppm",
    "mae_log_50ppm",
]
DOSE_LABELS = ["1", "5", "10", "20", "50"]

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
    if letters:
        abbrev = letters
    else:
        abbrev = name[:10]
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


def pick_best_linear(df: pd.DataFrame, used: set[str]) -> tuple[str | None, pd.Series | None, str | None]:
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


def pick_constant(df: pd.DataFrame, used: set[str]) -> tuple[str | None, pd.Series | None, str | None]:
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


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.07,
        1.02,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="right",
        va="bottom",
    )


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color("#333333")
    ax.tick_params(axis="both", which="major", width=0.6, length=3, color="#333333")


def main() -> None:
    df = load_data(CSV_PATH)
    numeric_cols = [
        "global_mae_log",
        "global_rmse_log",
        "global_r2",
        "spearman_rho",
        "global_mae_ppm",
        "global_rmse_ppm",
    ] + DOSE_COLS
    coerce_numeric(df, numeric_cols)

    if df.empty:
        raise ValueError("No rows available after filtering; cannot plot.")

    df_sorted = df.sort_values("global_mae_log", ascending=True, na_position="last").reset_index(drop=True)

    used: set[str] = set()
    selections: dict[str, str] = {}

    best_mae_name, best_mae_row = pick_best_mae(df_sorted, used)
    if best_mae_name is None:
        raise ValueError("Failed to select best MAE model.")
    selections["Best MAE (lowest global_mae_log)"] = best_mae_name
    used.add(best_mae_name)

    best_r2_name, best_r2_row = pick_best_r2(df_sorted, used)
    if best_r2_name is None:
        warn("No distinct best R2 found; falling back to next-best MAE.")
        best_r2_name, best_r2_row = pick_best_mae(df_sorted, used)
    selections["Best R2 (highest global_r2)"] = best_r2_name
    if best_r2_name:
        used.add(best_r2_name)

    best_linear_name, best_linear_row, msg = pick_best_linear(df_sorted, used)
    if msg:
        warn(msg)
    if best_linear_name is None:
        warn("Falling back to next-best MAE for linear baseline.")
        best_linear_name, best_linear_row = pick_best_mae(df_sorted, used)
    selections["Best linear baseline (Ridge/ElasticNet)"] = best_linear_name
    if best_linear_name:
        used.add(best_linear_name)

    constant_name, constant_row, msg = pick_constant(df_sorted, used)
    if msg:
        warn(msg)
    if constant_name is None:
        warn("Falling back to worst MAE for trivial baseline.")
        fallback = df_sorted.sort_values("global_mae_log", ascending=False, na_position="last")
        for _, row in fallback.iterrows():
            name = str(row["model_name"])
            if name not in used:
                constant_name = name
                constant_row = row
                break
    selections["Trivial baseline (Baseline_Constant)"] = constant_name
    if constant_name:
        used.add(constant_name)

    roles = list(selections.keys())
    highlight_names = [selections[r] for r in roles]
    highlight_names = [n for n in highlight_names if n is not None]

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
    for role in roles:
        name = selections.get(role)
        if name is None:
            print(f" - {role}: (missing)")
            continue
        row = df_sorted[df_sorted["model_name"] == name].iloc[0]
        print(
            f" - {role}: {name} "
            f"(global_mae_log={format_val(row.get('global_mae_log'))}, "
            f"global_r2={format_val(row.get('global_r2'))})"
        )

    print("\nTop models by global_mae_log:")
    top10 = df_sorted[["model_name", "global_mae_log", "global_rmse_log", "global_r2"]].head(10)
    with pd.option_context("display.max_rows", None, "display.float_format", "{:.6f}".format):
        print(top10.to_string(index=False))

    df_sorted.to_csv(OUT_SUMMARY, index=False)

    highlight_df = df_sorted.set_index("model_name").reindex(highlight_names).reset_index()
    highlight_df["factor"] = np.power(10.0, highlight_df["global_mae_log"])
    table_cols = ["model_name", "global_mae_log", "global_rmse_log", "global_r2", "spearman_rho", "factor"]
    fmt3 = lambda x: "--" if pd.isna(x) else f"{x:.3f}"
    formatters = {col: fmt3 for col in table_cols if col != "model_name"}
    latex = highlight_df[table_cols].to_latex(
        index=False,
        escape=True,
        formatters=formatters,
        column_format="lrrrrr",
    )
    with open(OUT_TEX, "w", encoding="utf-8") as f:
        f.write(latex)

    matplotlib.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
    })

    fig = plt.figure(figsize=(7.4, 5.2), dpi=150)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.2, 1.0],
        width_ratios=[1.1, 1.0],
        hspace=0.35,
        wspace=0.32,
    )

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    # Panel (a): Dumbbell plot
    mae_color = "#4C72B0"
    rmse_color = "#DD8452"
    context_color = "#9A9A9A"
    highlight_color = "#333333"

    y = np.arange(len(df_sorted))
    mae = df_sorted["global_mae_log"].to_numpy()
    rmse = df_sorted["global_rmse_log"].to_numpy()
    model_names = df_sorted["model_name"].astype(str).tolist()
    highlight_set = set(highlight_names)
    auto_abbrev: set[str] = set()
    display_names = [abbreviate_model(n, auto_abbrev) for n in model_names]
    display_map = dict(zip(model_names, display_names))
    if auto_abbrev:
        warn(f"Auto-abbreviated model names: {', '.join(sorted(auto_abbrev))}")

    for idx, name in enumerate(model_names):
        if name in highlight_set:
            continue
        ax_a.plot([mae[idx], rmse[idx]], [y[idx], y[idx]], color=context_color, alpha=0.15, lw=0.8, zorder=1)
        ax_a.scatter([mae[idx]], [y[idx]], color=context_color, alpha=0.15, s=18, zorder=2)
        ax_a.scatter([rmse[idx]], [y[idx]], color=context_color, alpha=0.15, s=18, zorder=2)

    for idx, name in enumerate(model_names):
        if name not in highlight_set:
            continue
        ax_a.plot([mae[idx], rmse[idx]], [y[idx], y[idx]], color=highlight_color, alpha=1.0, lw=1.6, zorder=3)
        ax_a.scatter([mae[idx]], [y[idx]], color=mae_color, edgecolor="white", linewidth=0.4, s=40, zorder=4)
        ax_a.scatter([rmse[idx]], [y[idx]], color=rmse_color, edgecolor="white", linewidth=0.4, s=40, zorder=4)

    ax_a.set_yticks(y)
    ax_a.set_yticklabels(display_names)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("log err")
    ax_a.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    style_axes(ax_a)

    for label, full_name in zip(ax_a.get_yticklabels(), model_names):
        if full_name in highlight_set:
            label.set_fontweight("bold")
            label.set_color("#000000")
        else:
            label.set_color("#7A7A7A")

    x_min = np.nanmin([np.nanmin(mae), np.nanmin(rmse)])
    x_max = np.nanmax([np.nanmax(mae), np.nanmax(rmse)])
    x_pad = 0.08 * (x_max - x_min) if np.isfinite(x_max - x_min) else 0.1
    ann_x = x_max + x_pad * 0.8
    ax_a.set_xlim(x_min - x_pad * 0.3, x_max + x_pad * 2.2)

    for name in highlight_names:
        row = df_sorted[df_sorted["model_name"] == name].iloc[0]
        yy = model_names.index(name)
        r2 = format_val(row.get("global_r2"))
        rho = format_val(row.get("spearman_rho"))
        ax_a.text(
            ann_x,
            yy,
            f"R2 {r2} | rho {rho}",
            ha="left",
            va="center",
            fontsize=7,
            color="#333333",
        )

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=mae_color, markeredgecolor=mae_color,
               markersize=5, label="MAE (typical)"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=rmse_color, markeredgecolor=rmse_color,
               markersize=5, label="RMSE (tail-sensitive)"),
    ]
    ax_a.legend(handles=legend_handles, frameon=False, loc="lower left", bbox_to_anchor=(0.01, 0.02))
    ax_a.set_title("Global performance")
    add_panel_label(ax_a, "(a)")

    # Panel (b): Heatmap
    missing_dose_cols = [c for c in DOSE_COLS if c not in df_sorted.columns]
    if missing_dose_cols:
        warn(f"Missing per-dose columns: {missing_dose_cols}; skipping heatmap panel.")
        ax_b.text(0.5, 0.5, "Per-dose MAE columns missing", ha="center", va="center", transform=ax_b.transAxes)
        ax_b.set_axis_off()
        add_panel_label(ax_b, "(b)")
    else:
        heatmap_data = df_sorted[DOSE_COLS].to_numpy(dtype=float)
        vmin = np.nanmin(heatmap_data)
        vmax = np.nanmax(heatmap_data)
        cmap = cm.get_cmap("Blues")
        norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        rgba = cmap(norm(heatmap_data))
        img = (rgba * 255).astype(np.uint8)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            Image.fromarray(img).save(tmp.name)
            raster = Image.open(tmp.name)
            ax_b.imshow(
                raster,
                origin="upper",
                interpolation="nearest",
                aspect="auto",
                extent=(-0.5, len(DOSE_COLS) - 0.5, len(model_names) - 0.5, -0.5),
            )
        finally:
            tmp.close()
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

        ax_b.set_xticks(np.arange(len(DOSE_COLS)))
        ax_b.set_xticklabels(DOSE_LABELS)
        ax_b.set_yticks(np.arange(len(model_names)))
        ax_b.set_yticklabels(display_names)
        ax_b.set_xlabel("dose")
        ax_b.set_title("Per-dose MAE")
        style_axes(ax_b)

        ax_b.set_xticks(np.arange(-0.5, len(DOSE_COLS), 1), minor=True)
        ax_b.set_yticks(np.arange(-0.5, len(model_names), 1), minor=True)
        ax_b.grid(which="minor", color="white", linestyle="-", linewidth=0.4, alpha=0.6)
        ax_b.tick_params(which="minor", bottom=False, left=False)

        for name in highlight_names:
            row_idx = model_names.index(name)
            rect = Rectangle(
                (-0.5, row_idx - 0.5),
                len(DOSE_COLS),
                1.0,
                fill=False,
                edgecolor="#333333",
                linewidth=0.7,
                alpha=0.6,
            )
            ax_b.add_patch(rect)

            for col_idx, col in enumerate(DOSE_COLS):
                val = heatmap_data[row_idx, col_idx]
                if np.isnan(val):
                    text = "–"
                    text_color = "black"
                else:
                    text = f"{val:.3f}"
                    text_color = "white" if val >= (vmin + vmax) / 2 else "black"
                ax_b.text(col_idx, row_idx, text, ha="center", va="center", fontsize=7, color=text_color)

        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=ax_b, fraction=0.046, pad=0.04)
        add_panel_label(ax_b, "(b)")

        for label, full_name in zip(ax_b.get_yticklabels(), model_names):
            if full_name in highlight_set:
                label.set_fontweight("bold")
                label.set_color("#000000")
            else:
                label.set_color("#7A7A7A")

    # Panel (c): Scatter decision plot
    style_axes(ax_c)
    ax_c.set_xlabel("MAE")
    ax_c.set_ylabel("")
    ax_c.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax_c.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    is_highlight = df_sorted["model_name"].isin(highlight_names).to_numpy()
    ax_c.scatter(
        df_sorted.loc[~is_highlight, "global_mae_log"],
        df_sorted.loc[~is_highlight, "global_rmse_log"],
        s=25,
        color=context_color,
        alpha=0.15,
        edgecolors="none",
        zorder=1,
    )

    highlight_df = df_sorted[df_sorted["model_name"].isin(highlight_names)].copy()
    r2_vals = highlight_df["global_r2"].to_numpy(dtype=float)
    if np.all(np.isnan(r2_vals)):
        highlight_colors = [mae_color] * len(highlight_df)
        r2_norm = None
    else:
        r2_min = np.nanmin(r2_vals)
        r2_max = np.nanmax(r2_vals)
        if r2_min == r2_max:
            r2_min -= 1e-6
            r2_max += 1e-6
        r2_norm = colors.Normalize(vmin=r2_min, vmax=r2_max)
        cmap_r2 = cm.get_cmap("viridis")
        highlight_colors = cmap_r2(r2_norm(r2_vals))

    ax_c.scatter(
        highlight_df["global_mae_log"],
        highlight_df["global_rmse_log"],
        s=60,
        color=highlight_colors,
        edgecolor="#333333",
        linewidth=0.6,
        zorder=3,
    )

    xy_min = np.nanmin([np.nanmin(mae), np.nanmin(rmse)])
    xy_max = np.nanmax([np.nanmax(mae), np.nanmax(rmse)])
    xy_pad = 0.08 * (xy_max - xy_min)
    ax_c.set_xlim(xy_min - xy_pad, xy_max + xy_pad)
    ax_c.set_ylim(xy_min - xy_pad, xy_max + xy_pad)
    line = np.linspace(xy_min - xy_pad, xy_max + xy_pad, 100)
    ax_c.plot(line, line, linestyle="--", color="#666666", lw=0.8, alpha=0.6, zorder=2)

    def place_labels(ax: plt.Axes, points: list[tuple[float, float, str]], offset_factor: float) -> tuple[float, float]:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        xspan = xmax - xmin
        yspan = ymax - ymin
        base_dx = 0.015 * xspan * offset_factor
        base_dy = 0.02 * yspan * offset_factor
        placed: list[tuple[float, float]] = []
        for x, y, label in points:
            dx, dy = base_dx, base_dy
            for k in range(12):
                collision = False
                for px, py in placed:
                    if abs((x + dx) - px) < 0.04 * xspan and abs((y + dy) - py) < 0.04 * yspan:
                        collision = True
                        dy += (0.03 * yspan) * (1 if k % 2 == 0 else -1)
                        dx += (0.01 * xspan) * (-1 if k % 2 == 0 else 1)
                        break
                if not collision:
                    break
            placed.append((x + dx, y + dy))
            ax.annotate(
                label,
                (x, y),
                xytext=(x + dx, y + dy),
                textcoords="data",
                ha="left",
                va="center",
                fontsize=7,
                arrowprops=dict(arrowstyle="-", color="#444444", lw=0.6, shrinkA=0, shrinkB=2),
            )
        return base_dx, base_dy

    label_points = [
        (row["global_mae_log"], row["global_rmse_log"], display_map.get(row["model_name"], row["model_name"]))
        for _, row in highlight_df.iterrows()
    ]
    offset_factor = 1.5
    dx, dy = place_labels(ax_c, label_points, offset_factor)

    if r2_norm is not None:
        mappable = cm.ScalarMappable(norm=r2_norm, cmap=cm.get_cmap("viridis"))
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=ax_c, fraction=0.046, pad=0.04)
        cbar.set_label("R2")

    ax_c.set_title("Decision plot")
    add_panel_label(ax_c, "(c)")

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=650, bbox_inches="tight")
    print(f"Panel (c) label offsets increased by {offset_factor:.2f} (dx,dy={dx:.4f},{dy:.4f}).")
    print("\nApplied changes:")
    print(" - Heatmap colormap set to Blues (white-to-blue).")
    print(" - Axis labels abbreviated; global font sizes reduced.")
    print(" - Panel (b) y-axis shows model abbreviations with highlights.")
    print(" - Panel (a) legend moved to lower-left.")
    print(" - All displayed model names use abbreviations.")
    plt.close(fig)


if __name__ == "__main__":
    main()
