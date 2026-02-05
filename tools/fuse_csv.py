#!/usr/bin/env python3
"""
Fuse 8 gas datasets (already with meta columns) into one big frame-wise dataset.
"""

from pathlib import Path
import pandas as pd

GAS_ORDER = ["C2H6O", "H2", "H2S", "NH3", "NO", "NO2", "PH3", "SO2"]

in_dir = Path("./data/with_meta")
out_path = Path("./data/all_gases_framewise.csv")

dfs = []
for gas in GAS_ORDER:
    path = in_dir / f"{gas}_with_meta.csv"
    if not path.exists():
        print(f"WARNING: {path} not found, skip")
        continue
    df = pd.read_csv(path)
    # 加一列方便肉眼识别
    df["gas_name"] = gas
    dfs.append(df)

big_df = pd.concat(dfs, ignore_index=True)

# 可选：检查几个关键列是不是都有
required_cols = [
    "time", "gas_id", "run_id",
    "t_exposure_start_ms", "t_exposure_end_ms"
]
missing = [c for c in required_cols if c not in big_df.columns]
if missing:
    print("WARNING: missing columns:", missing)

big_df.to_csv(out_path, index=False)
print("Saved fused dataset to", out_path, "rows =", len(big_df))

