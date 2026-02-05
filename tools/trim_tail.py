#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

def guess_sep(sample: bytes):
    if b"," in sample:
        return b","
    if b"\t" in sample:
        return b"\t"
    return None  # whitespace

def first_token(line: bytes, sep: bytes | None) -> bytes:
    s = line.lstrip()
    if not s:
        return b""
    if sep is None:
        return s.split(None, 1)[0]
    return s.split(sep, 1)[0]

def to_float(tok: bytes):
    try:
        return float(tok.decode("utf-8", errors="ignore"))
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--target_seconds", type=float, default=900.0, help="Target duration in seconds (default 900)")
    ap.add_argument("--time_unit", choices=["ms", "s"], default="ms", help="Time unit in file (default ms)")
    ap.add_argument("--header", action="store_true", help="Keep first line as header")
    args = ap.parse_args()

    in_path = args.input
    out_path = args.output or in_path.with_name(in_path.name + f"_trim_{int(args.target_seconds)}s")

    scale = 1000.0 if args.time_unit == "ms" else 1.0  # convert seconds -> file unit

    with open(in_path, "rb") as fin:
        sample = fin.read(8192)
        sep = guess_sep(sample)
        fin.seek(0)

        with open(out_path, "wb") as fout:
            # optional header
            if args.header:
                hdr = fin.readline()
                if hdr:
                    fout.write(hdr)

            t0 = None
            t_last_all = None
            cutoff = None

            kept = 0
            trimmed = 0
            bad_lines = 0

            for line in fin:
                if not line.strip():
                    continue

                tok = first_token(line, sep)
                t = to_float(tok)

                # 非数字行：保守处理（原样保留），但不用于计时
                if t is None:
                    bad_lines += 1
                    fout.write(line)
                    continue

                if t0 is None:
                    t0 = t
                    cutoff = t0 + args.target_seconds * scale

                t_last_all = t

                if t <= cutoff:
                    fout.write(line)
                    kept += 1
                else:
                    trimmed += 1
                    # 继续读完以获取真实原始结束时间 t_last_all

    if t0 is None or t_last_all is None:
        raise RuntimeError("No numeric time values found in the first column.")

    duration_s = (t_last_all - t0) / scale
    print(f"Start time: {t0:g} {args.time_unit}")
    print(f"End time  : {t_last_all:g} {args.time_unit}")
    print(f"Duration  : {duration_s:.6g} s")

    if duration_s > args.target_seconds:
        kept_end_s = args.target_seconds
        print(f"Trimmed tail to {kept_end_s:g} s (cutoff = {t0 + args.target_seconds * scale:g} {args.time_unit}).")
    else:
        print(f"No tail trimmed (duration <= {args.target_seconds:g}s).")

    print(f"Rows kept={kept}, rows trimmed={trimmed}, non-numeric lines kept={bad_lines}")
    print(f"Written: {out_path}")

if __name__ == "__main__":
    main()

