#!/usr/bin/env python3
"""
Render confusion-matrix heatmaps as truecolor raster panels and compose a vector-text PDF.
LaTeX note: prefer pdflatex or lualatex (avoid latex->dvips), and include graphicx/xcolor.
"""
import argparse
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import cm as mcm
from PIL import Image


MODELS = ["KNN", "SVM", "RF", "CNN", "LSTM"]


def _is_non_numeric_series(s: pd.Series) -> bool:
    try:
        pd.to_numeric(s)
        return False
    except Exception:
        return True


def _are_numeric_df(df: pd.DataFrame) -> bool:
    try:
        pd.to_numeric(df.stack(), errors="raise")
        return True
    except Exception:
        return False


def find_confmat_file(input_dir: Path, model: str) -> Path:
    expected = f"confmat_{model}_run_row_normalized.csv"
    exact = list(input_dir.rglob(expected))
    if exact:
        return exact[0]

    candidates = []
    for path in input_dir.rglob("*.csv"):
        name = path.name.lower()
        if model.lower() in name and ("confmat" in name or "confusion" in name) and "run" in name and "row" in name and "normalized" in name:
            candidates.append(path)

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        msg = [f"Ambiguous match for {model}. Found multiple candidates:"]
        msg += [f"  - {p}" for p in candidates]
        raise FileNotFoundError("\n".join(msg))

    msg = [f"Missing confusion-matrix CSV for {model}."]
    if candidates:
        msg.append("Candidates:")
        msg += [f"  - {p}" for p in candidates]
    raise FileNotFoundError("\n".join(msg))


def load_confmat(path: Path):
    df = pd.read_csv(path)

    # Handle common index-column patterns
    if df.columns[0].lower().startswith("unnamed"):
        df = df.set_index(df.columns[0])
    elif df.shape[1] == df.shape[0] + 1:
        first_col = df.iloc[:, 0]
        rest = df.iloc[:, 1:]
        if _is_non_numeric_series(first_col) and _are_numeric_df(rest):
            df = df.set_index(df.columns[0])
    elif df.columns[0].lower() in ("class", "label", "labels", "true", "true_class"):
        df = df.set_index(df.columns[0])

    # Extract labels
    x_labels = list(df.columns)
    if isinstance(df.index, pd.RangeIndex) and df.index.name is None:
        if df.shape[0] == df.shape[1]:
            y_labels = list(df.columns)
        else:
            y_labels = list(df.index)
    else:
        y_labels = list(df.index)

    # Numeric matrix
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    if numeric_df.isna().any().any():
        numeric_df = numeric_df.fillna(0.0)

    M = numeric_df.to_numpy(dtype=float)
    return M, x_labels, y_labels


def normalize_rows_to_percent(M: np.ndarray, label: str) -> np.ndarray:
    M = M.astype(float)
    n_classes = M.shape[0]

    max_val = np.nanmax(M)
    if max_val <= 1.0 + 1e-6:
        M = M * 100.0
    else:
        row_sums = M.sum(axis=1)
        tol = 1e-2 * n_classes
        if max_val <= 100.0 + 1e-6 and np.all(np.abs(row_sums - 100.0) <= tol):
            pass
        else:
            row_sum = M.sum(axis=1, keepdims=True)
            M = np.divide(M, row_sum, out=np.zeros_like(M), where=row_sum != 0) * 100.0

    row_sums_after = M.sum(axis=1)
    print(f"[{label}] value range after normalization: min={np.min(M):.2f}, max={np.max(M):.2f}")
    print(
        f"[{label}] row-sum stats after normalization: min={row_sums_after.min():.2f}, "
        f"mean={row_sums_after.mean():.2f}, max={row_sums_after.max():.2f}"
    )

    return M


def matrix_to_rgb_png(M: np.ndarray, norm, cmap, out_png: Path) -> None:
    rgba = cmap(norm(M), bytes=True)  # uint8 RGBA
    rgb = rgba[..., :3].copy()        # uint8 RGB
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(out_png, format="PNG", compress_level=0)


def plot_confmats_raster_panels(mats, labels_x, labels_y, tmp_pngs, out_pdf: Path, out_png: Path, norm, cmap):
    n_classes = max(m.shape[0] for m in mats)
    w_each = max(2.8, 0.35 * n_classes + 1.0)
    h = max(2.6, 0.35 * n_classes + 1.0)
    figsize = (len(mats) * w_each, h)

    fig, axes = plt.subplots(1, len(mats), figsize=figsize, constrained_layout=True)
    fig.patch.set_facecolor("white")
    if len(mats) == 1:
        axes = [axes]

    for idx, (ax, M, title, xlab, ylab, png_path) in enumerate(zip(axes, mats, MODELS, labels_x, labels_y, tmp_pngs)):
        ax.set_facecolor("white")
        panel = np.asarray(Image.open(png_path))
        ax.imshow(panel, interpolation="nearest", resample=False, origin="upper", aspect="equal")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(xlab)))
        ax.set_yticks(np.arange(len(ylab)))
        ax.set_xticklabels(xlab, rotation=45, ha="right")
        ax.set_yticklabels(ylab)
        if idx > 0:
            ax.tick_params(labelleft=False)

        if M.shape[0] <= 15:
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    val = M[i, j]
                    color = "white" if val > 50.0 else "black"
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center", color=color, fontsize=8)

    sm = mcm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location="right", shrink=0.9, pad=0.02)
    cbar.set_label("Run % (row-normalized)")
    cbar.ax.set_facecolor("lightblue")

    fig.supxlabel("Predicted class")
    fig.supylabel("True class")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", dpi=600, bbox_inches="tight", transparent=False, facecolor="white")
    if out_png is not None:
        fig.savefig(out_png, format="png", dpi=600, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)


def run_pdfimages(out_pdf: Path) -> None:
    try:
        result = subprocess.run(
            ["pdfimages", "-list", str(out_pdf)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print("pdfimages -list output:")
        print(result.stdout.strip())
    except FileNotFoundError:
        print("pdfimages not found; skipping embedded image report.")


def main():
    parser = argparse.ArgumentParser(description="Render raster-panel confusion matrices with vector text.")
    parser.add_argument("--input_dir", required=True, help="Root directory to search for confusion-matrix CSVs.")
    parser.add_argument("--out_pdf", required=True, help="Output PDF path.")
    parser.add_argument("--out_png", default=None, help="Optional output PNG path (600 dpi).")
    parser.add_argument("--tmp_dir", default=None, help="Temporary directory for raster panel PNGs.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    out_pdf = Path(args.out_pdf)
    out_png = Path(args.out_png) if args.out_png else out_pdf.with_suffix(".png")
    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else out_pdf.parent / "tmp_confmat_panels"

    mats = []
    labels_x = []
    labels_y = []
    tmp_pngs = []

    norm = mcolors.Normalize(vmin=0, vmax=100)
    cmap = mcolors.LinearSegmentedColormap.from_list("white_blue", ["#ffffff", "#0000ff"], N=256)

    for model in MODELS:
        try:
            path = find_confmat_file(input_dir, model)
        except FileNotFoundError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)

        print(f"Using {model} CSV: {path}")
        M, xlab, ylab = load_confmat(path)
        M = normalize_rows_to_percent(M, model)

        png_path = tmp_dir / f"panel_{model}.png"
        matrix_to_rgb_png(M, norm, cmap, png_path)

        mats.append(M)
        labels_x.append(xlab)
        labels_y.append(ylab)
        tmp_pngs.append(png_path)

    plot_confmats_raster_panels(mats, labels_x, labels_y, tmp_pngs, out_pdf, out_png, norm, cmap)

    if out_pdf.exists():
        print(f"Saved PDF: {out_pdf} ({out_pdf.stat().st_size} bytes)")
    else:
        print(f"Saved PDF: {out_pdf}")
    if out_png is not None:
        if out_png.exists():
            print(f"Saved PNG: {out_png} ({out_png.stat().st_size} bytes)")
        else:
            print(f"Saved PNG: {out_png}")

    run_pdfimages(out_pdf)


if __name__ == "__main__":
    mpl.rcParams.update({
        "font.size": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.compression": 0,
    })
    main()
