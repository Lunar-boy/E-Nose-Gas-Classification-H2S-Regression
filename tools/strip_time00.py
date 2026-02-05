#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

LINE_RE = re.compile(rb'^(\s*)(\S+)(\s+)(.*?)(\r?\n)?$')

def fix_first_token(tok: bytes) -> bytes:
    # 只去掉前两个 0： "0047400" -> "47400"
    return tok[2:] if tok.startswith(b"00") else tok

def fix_first_token1(tok: bytes) -> bytes:

    return tok[1:] if tok.startswith(b"0") else tok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()

    in_path = args.input
    out_path = args.output or in_path.with_name(in_path.name + "_time_no00")

    with open(in_path, "rb") as fin, open(out_path, "wb") as fout:
        for line in fin:
            m = LINE_RE.match(line)
            if not m:
                # 极少数异常行：原样写出
                fout.write(line)
                continue

            lead, tok, sep, rest, nl = m.groups()
            tok2 = fix_first_token(tok)
            tok3 = fix_first_token1(tok2)
            fout.write(lead + tok3 + sep + rest + (nl or b""))

    print(f"Written: {out_path}")

if __name__ == "__main__":
    main()

