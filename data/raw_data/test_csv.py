import pandas as pd
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv("all_gases_framewise.csv")

run_meta = (df.groupby("run_id")
              .agg(gas_name=("gas_name","first"))
              .reset_index())

print("n_runs =", len(run_meta))
print(run_meta["gas_name"].value_counts())

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (tr_idx, te_idx) in enumerate(skf.split(run_meta["run_id"], run_meta["gas_name"])):
    te = run_meta.iloc[te_idx]
    print(f"\nFold {fold} test runs =", len(te))
    print(te["gas_name"].value_counts().sort_index())

