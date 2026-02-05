#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # HPC/no-GUI safe
import matplotlib.pyplot as plt
import pandas as pd


def guess_sep(path: Path) -> str:
    with open(path, "rb") as f:
        sample = f.read(8192)
    if b"," in sample:
        return ","
    if b"\t" in sample:
        return "\t"
    return "whitespace"


def read_table(path: Path, sep: str, header: bool):
    usecols = list(range(17))  # time + 16 sensors
    if sep == "whitespace":
        return pd.read_csv(
            path, delim_whitespace=True, header=0 if header else None,
            usecols=usecols, engine="python"
        )
    return pd.read_csv(
        path, sep=sep, header=0 if header else None,
        usecols=usecols, engine="python"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="col0=time(ms), col1..col16=sensors")
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--sep", type=str, default=None, help="Delimiter: ',', '\\t', or 'whitespace' (auto if omitted)")
    ap.add_argument("--header", action="store_true", help="Set if first row is header")
    ap.add_argument("--stride", type=int, default=1, help="Downsample: keep every Nth row (default 1)")
    ap.add_argument("--logy", action="store_true", help="Log scale y-axis")
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--dpi", type=int, default=200)

    # New: skip initial flush
    ap.add_argument("--flush_seconds", type=float, default=600.0, help="Seconds to drop from start (default 600)")
    ap.add_argument("--time_unit", choices=["ms", "s"], default="ms", help="Time unit in file (default ms)")
    ap.add_argument("--relative_time", action="store_true",
                    help="Plot time relative to flush end (i.e., t'=t - t_flush_end). Recommended.")
    args = ap.parse_args()

    in_path = args.input
    sep = args.sep or guess_sep(in_path)
    df = read_table(in_path, sep=sep, header=args.header)

    if df.shape[1] < 17:
        raise ValueError(f"Expected >=17 columns (time + 16 sensors), got {df.shape[1]}.")

    # Name columns
    if args.header:
        time_name = str(df.columns[0])
        sensor_names = [str(c) for c in df.columns[1:17]]
    else:
        time_name = "time"
        sensor_names = [f"S{i}" for i in range(1, 17)]
        df.columns = [time_name] + sensor_names

    # Ensure numeric time
    t = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    df = df.loc[t.notna()].copy()
    t = pd.to_numeric(df.iloc[:, 0], errors="coerce").astype(float)

    # Compute flush cutoff in same unit as file
    if args.time_unit == "ms":
        cutoff = args.flush_seconds * 1000.0
        x_label = "time (ms)"
    else:
        cutoff = args.flush_seconds
        x_label = "time (s)"

    # Drop initial flush segment
    df = df.loc[t >= cutoff].copy()
    if df.empty:
        raise ValueError(
            f"After dropping first {args.flush_seconds}s (cutoff={cutoff} {args.time_unit}), no data remains."
        )

    # Downsample after filtering
    if args.stride > 1:
        df = df.iloc[:: args.stride, :].reset_index(drop=True)

    # X axis
    x = df.iloc[:, 0].astype(float).to_numpy()
    if args.relative_time:
        x = x - cutoff
        x_label = f"time since flush end ({args.time_unit})"

    # Plot 4x4
    fig, axes = plt.subplots(4, 4, figsize=(18, 12), sharex=True, constrained_layout=True)
    axes = axes.ravel()

    for i in range(16):
        ax = axes[i]
        y = pd.to_numeric(df.iloc[:, i + 1], errors="coerce").to_numpy()
        ax.plot(x, y, linewidth=1.0)
        ax.set_title(sensor_names[i], fontsize=10)
        ax.grid(True, alpha=0.3)
        if args.logy:
            ax.set_yscale("log")
        if i // 4 == 3:
            ax.set_xlabel(x_label)
        ax.set_ylabel("Resistance")

    fig.suptitle(args.title or f"16 Sensor Responses (drop first {args.flush_seconds:.0f}s flush)", fontsize=14)

    out_path = args.output or in_path.with_suffix(".16sensors.skip600s.png")
    fig.savefig(out_path, dpi=args.dpi)
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()

