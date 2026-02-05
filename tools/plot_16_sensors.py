#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # HPC/no-GUI safe
import matplotlib.pyplot as plt
import pandas as pd


def guess_sep(path: Path) -> str:
    """Guess delimiter: comma, tab, or whitespace."""
    with open(path, "rb") as f:
        sample = f.read(8192)
    if b"," in sample:
        return ","
    if b"\t" in sample:
        return "\t"
    return "whitespace"


def read_table(path: Path, sep: str, header: bool):
    # We only need time + 16 sensors => 17 columns
    usecols = list(range(17))

    if sep == "whitespace":
        return pd.read_csv(
            path,
            delim_whitespace=True,
            header=0 if header else None,
            usecols=usecols,
            engine="python",
        )
    else:
        return pd.read_csv(
            path,
            sep=sep,
            header=0 if header else None,
            usecols=usecols,
            engine="python",
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="Input file: col0=time, col1..col16=sensors")
    ap.add_argument("-o", "--output", type=Path, default=None, help="Output image path (png/pdf/svg)")
    ap.add_argument("--sep", type=str, default=None, help="Delimiter: ',', '\\t', or 'whitespace' (auto if omitted)")
    ap.add_argument("--header", action="store_true", help="Set if the first row is a header")
    ap.add_argument("--stride", type=int, default=1, help="Downsample for plotting: keep every N-th row (default 1)")
    ap.add_argument("--logy", action="store_true", help="Use log scale on y-axis")
    ap.add_argument("--title", type=str, default=None, help="Figure title (optional)")
    ap.add_argument("--dpi", type=int, default=200, help="Output DPI (default 200)")
    args = ap.parse_args()

    in_path = args.input
    sep = args.sep or guess_sep(in_path)

    df = read_table(in_path, sep=sep, header=args.header)

    # Basic sanity
    if df.shape[1] < 17:
        raise ValueError(f"Expected at least 17 columns (time + 16 sensors), but got {df.shape[1]}.")

    # Downsample for plotting if needed
    if args.stride > 1:
        df = df.iloc[:: args.stride, :].reset_index(drop=True)

    # Column names
    if args.header:
        time_name = str(df.columns[0])
        sensor_names = [str(c) for c in df.columns[1:17]]
    else:
        time_name = "time"
        sensor_names = [f"S{i}" for i in range(1, 17)]
        df.columns = [time_name] + sensor_names

    t = df.iloc[:, 0].to_numpy()

    # Create 4x4 subplots
    fig, axes = plt.subplots(4, 4, figsize=(18, 12), sharex=True, constrained_layout=True)
    axes = axes.ravel()

    for i in range(16):
        ax = axes[i]
        y = df.iloc[:, i + 1].to_numpy()
        ax.plot(t, y, linewidth=1.0)
        ax.set_title(sensor_names[i], fontsize=10)
        ax.grid(True, alpha=0.3)

        if args.logy:
            ax.set_yscale("log")

        # Labeling
        if i // 4 == 3:  # bottom row
            ax.set_xlabel(time_name)
        ax.set_ylabel("Resistance")

    fig.suptitle(args.title or f"16 Sensor Responses vs Time ({in_path.name})", fontsize=14)

    out_path = args.output or in_path.with_suffix(".16sensors.png")
    fig.savefig(out_path, dpi=args.dpi)
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()

