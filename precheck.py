import json, os
import numpy as np
import pandas as pd
from pathlib import Path

PADS = Path('pads_data')

# file_list.csv
df = pd.read_csv(PADS / 'preprocessed' / 'file_list.csv')
print("=== file_list.csv ===")
print(df.columns.tolist())
print(df.head(5).to_string())
print(f"Shape: {df.shape}")

# One patient JSON
with open(PADS / 'patients' / 'patient_001.json') as f:
    print("\n=== patient_001.json ===")
    print(json.dumps(json.load(f), indent=2))

# One movement file
movement_dir = PADS / 'preprocessed' / 'movement'
files = list(movement_dir.iterdir())[:3]
for f in files:
    print(f"\n=== {f.name} ===")
    if f.suffix == '.bin':
        raw = np.fromfile(f, dtype=np.float32)
        print(f"float32 count: {len(raw)}")
        print(f"sqrt guess: {len(raw)**0.5:.1f}")
        for n_cols in [264, 132, 66, 528]:
            if len(raw) % n_cols == 0:
                print(f"  divides by {n_cols} → shape ({len(raw)//n_cols}, {n_cols})")
    elif f.suffix == '.npy':
        arr = np.load(f)
        print(f"shape: {arr.shape}")