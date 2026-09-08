#!/usr/bin/env python3
"""
Batch-add metadata columns to e-nose gas files.

Input expectations:
- One file per gas (filename stem is gas name, e.g. "NO2", "PH3", "C2H6O.csv")
- Columns: time + 16 sensor resistance columns
- Files may be whitespace-delimited (common) or CSV.

Adds columns (per row, repeated within the same cycle):
- gas_id
- run_id
- t_exposure_start_ms
- t_exposure_end_ms
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd


GAS_ORDER = ["C2H6O", "H2", "H2S", "NH3", "NO", "NO2", "PH3", "SO2"]
GAS_TO_ID: Dict[str, int] = {g: i for i, g in enumerate(GAS_ORDER)}


def is_number_token(tok: str) -> bool:
    try:
        float(tok)
        return True
    except Exception:
        return False


def detect_header_and_sep(path: Path) -> Tuple[bool, str]:
    """Return (has_header, sep_regex_or_comma)."""
    first = path.read_text(errors="ignore").splitlines()[0].strip()
    if "," in first:
        toks = [t.strip() for t in first.split(",") if t.strip() != ""]
        has_header = not all(is_number_token(t) for t in toks)
        return has_header, ","
    toks = [t for t in re.split(r"\s+", first) if t != ""]
    has_header = not all(is_number_token(t) for t in toks)
    return has_header, r"\s+"


def standardize_columns(df: pd.DataFrame, force_time_col: Optional[str] = None) -> pd.DataFrame:
    """Ensure we have a 'time' column; for no-header files, name sensors s01..s16."""
    if force_time_col is not None:
        time_col = force_time_col
    else:
        time_candidates = [c for c in df.columns if isinstance(c, str) and "time" in c.lower()]
        time_col = time_candidates[0] if time_candidates else df.columns[0]

    cols = list(df.columns)
    if all(isinstance(c, (int, np.integer)) for c in cols):
        new_cols = ["time"] + [f"s{j:02d}" for j in range(1, len(cols))]
        df.columns = new_cols
    else:
        if time_col != "time":
            df = df.rename(columns={time_col: "time"})
    return df


def to_time_ms(time_series: pd.Series, expected_total_s: float) -> pd.Series:
    """Convert time to ms if it looks like seconds."""
    t = pd.to_numeric(time_series, errors="coerce")
    max_t = float(np.nanmax(t.values))
    # If the max time is only a few thousand, it's probably seconds.
    if max_t <= expected_total_s * 5.0:
        return t * 1000.0
    return t.astype(float)


def infer_start_ms(min_time_ms: float, max_time_ms: float, total_span_ms: float) -> float:
    """Infer first exposure start as nearest second to min_time_ms, with a simple sanity check."""
    cand = round(min_time_ms / 1000.0) * 1000.0
    if (max_time_ms - cand) < total_span_ms * 0.5:
        return min_time_ms
    return cand


def add_metadata(
    df: pd.DataFrame,
    gas_name: str,
    exposure_s: float,
    flush_s: float,
    n_cycles: int,
    ppm_list: List[float],
    start_ms: Optional[float],
    drop_outside: bool,
) -> pd.DataFrame:
    cycle_s = exposure_s + flush_s
    cycle_ms = cycle_s * 1000.0
    exposure_ms = exposure_s * 1000.0

    expected_total_s = n_cycles * cycle_s
    df = df.copy()
    time_ms = to_time_ms(df["time"], expected_total_s)

    tmin = float(np.nanmin(time_ms.values))
    tmax = float(np.nanmax(time_ms.values))

    start0 = float(start_ms) if start_ms is not None else infer_start_ms(tmin, tmax, total_span_ms=n_cycles * cycle_ms)

    # cycle index
    rel = time_ms - start0
    cycle_idx = np.floor(rel / cycle_ms).astype("Int64")
    valid = ((cycle_idx >= 0) & (cycle_idx < n_cycles)).fillna(False).to_numpy(dtype=bool)
    idx_np = cycle_idx.to_numpy()
    idx_valid = idx_np[valid].astype(int)

    starts = np.array([start0 + i * cycle_ms for i in range(n_cycles)], dtype=float)
    ends = starts + exposure_ms

    if len(ppm_list) < n_cycles:
        ppm_list = ppm_list + [np.nan] * (n_cycles - len(ppm_list))
    ppm_arr = np.array(ppm_list[:n_cycles], dtype=float)
    run_ids = [f"{gas_name}_cycle{i+1}_ppm{int(ppm_arr[i]) if np.isfinite(ppm_arr[i]) else 'NA'}" for i in range(n_cycles)]

    # add columns
    df["gas_id"] = GAS_TO_ID.get(gas_name, -1)
    df["run_id"] = ""
    df["t_exposure_start_ms"] = np.nan
    df["t_exposure_end_ms"] = np.nan

    df.loc[valid, "t_exposure_start_ms"] = starts[idx_valid]
    df.loc[valid, "t_exposure_end_ms"] = ends[idx_valid]
    df.loc[valid, "run_id"] = [run_ids[i] for i in idx_valid]

    if drop_outside:
        df = df.loc[valid].reset_index(drop=True)

    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=str, help="Directory containing gas files (e.g. NO2, PH3, H2, ...)")
    ap.add_argument("--output_dir", type=str, default=None, help="Default: <input_dir>/with_meta")
    ap.add_argument("--pattern", type=str, default="*", help="Glob pattern (default: *)")
    ap.add_argument("--exposure_s", type=float, default=60.0)
    ap.add_argument("--flush_s", type=float, default=120.0)
    ap.add_argument("--n_cycles", type=int, default=5)
    ap.add_argument("--ppm", type=str, default="1,5,10,20,50")
    ap.add_argument("--start_ms", type=float, default=None, help="Override first exposure start (ms)")
    ap.add_argument("--time_col", type=str, default=None, help="If time column has a header name, specify it")
    ap.add_argument("--drop_outside", action="store_true", help="Drop rows outside the inferred cycles")
    ap.add_argument("--out_ext", type=str, default=".csv", choices=[".csv", ".parquet"])
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir) if args.output_dir else in_dir / "with_meta"
    out_dir.mkdir(parents=True, exist_ok=True)

    ppm_list = [float(x) for x in args.ppm.split(",") if x.strip() != ""]

    for path in sorted(in_dir.glob(args.pattern)):
        if path.is_dir():
            continue
        gas_name = path.stem  # works for "NO2" and "NO2.csv"
        if gas_name not in GAS_TO_ID:
            continue

        has_header, sep = detect_header_and_sep(path)
        if sep == ",":
            df = pd.read_csv(path, header=0 if has_header else None)
        else:
            df = pd.read_csv(path, sep=sep, header=0 if has_header else None, engine="python")

        df = standardize_columns(df, force_time_col=args.time_col)

        df2 = add_metadata(
            df=df,
            gas_name=gas_name,
            exposure_s=args.exposure_s,
            flush_s=args.flush_s,
            n_cycles=args.n_cycles,
            ppm_list=ppm_list,
            start_ms=args.start_ms,
            drop_outside=args.drop_outside,
        )

        out_path = out_dir / f"{gas_name}_with_meta{args.out_ext}"
        if args.out_ext == ".csv":
            df2.to_csv(out_path, index=False)
        else:
            df2.to_parquet(out_path, index=False)

        print(f"Wrote {out_path} (rows={len(df2)})")


if __name__ == "__main__":
    main()

