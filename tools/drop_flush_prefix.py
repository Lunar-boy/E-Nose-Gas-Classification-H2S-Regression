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
        return s.split(None, 1)[0]  # whitespace
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
    ap.add_argument("--flush_seconds", type=float, required=True, help="Seconds to drop from start, e.g. 600")
    ap.add_argument("--time_unit", choices=["ms", "s"], default="ms", help="Time unit in file (default ms)")
    ap.add_argument("--header", action="store_true", help="If first line is header, keep it")
    args = ap.parse_args()

    in_path = args.input
    out_path = args.output or in_path.with_name(in_path.name + f"_drop_first_{int(args.flush_seconds)}s")

    cutoff = args.flush_seconds * (1000.0 if args.time_unit == "ms" else 1.0)

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

            kept = 0
            dropped = 0

            for line in fin:
                if not line.strip():  # skip empty lines (or keep them if you prefer)
                    continue

                tok = first_token(line, sep)
                t = to_float(tok)

                # 如果遇到非数字行（比如中间插入的说明/坏行），保守起见原样保留
                if t is None:
                    fout.write(line)
                    continue

                if t >= cutoff:
                    fout.write(line)
                    kept += 1
                else:
                    dropped += 1

    print(f"Cutoff: {cutoff:g} ({args.time_unit})  | dropped={dropped} kept={kept}")
    print(f"Written: {out_path}")

if __name__ == "__main__":
    main()

