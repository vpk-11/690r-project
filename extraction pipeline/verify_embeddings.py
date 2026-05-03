#!/usr/bin/env python3
"""
verify_embeddings.py -- Sanity check Bio-PM features.
Usage: python verify_embeddings.py --features features/biopm_features.npz
"""
import argparse
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    args = p.parse_args()

    d    = np.load(args.features, allow_pickle=True)
    X    = d["features"]
    y    = d["labels"]
    subj = d["subj"]
    week = d["week"]
    ARAT = d["ARAT"]
    FMA  = d["FMA"]

    print("=" * 56)
    print(f"Shape: {X.shape}  (N, 1028)")
    assert X.shape[1] == 1028

    print(f"NaN: {np.isnan(X).sum()}   Inf: {np.isinf(X).sum()}")

    xfm  = X[:, :128]
    grav = X[:, 128:]
    print(f"\nTransformer [0:128]:  mean_abs={np.abs(xfm).mean():.4f}  std={xfm.std():.4f}")
    print(f"Gravity     [128:]:  mean_abs={np.abs(grav).mean():.4f}  std={grav.std():.4f}")

    n_zero = (np.abs(X) < 1e-6).all(axis=1).sum()
    print(f"All-zero rows: {n_zero}/{len(X)}")
    if n_zero > len(X) * 0.3:
        print("  WARNING: many zero rows. Use features[:, 128:] for downstream analysis.")

    print(f"\nLabel split:")
    for lbl, name in [(1, "healthy"), (0, "stroke")]:
        n = (y.astype(int) == lbl).sum()
        print(f"  {lbl} ({name:7s}): {n:5d} windows ({100*n/len(y):.1f}%)")

    print(f"\nPer-subject:")
    print(f"  {'Subj':>5}  {'Wks':>4}  {'#Win':>6}  {'ARAT':>6}  {'FMA':>5}  Group")
    for sid in sorted(np.unique(subj)):
        m = subj == sid
        wks  = len(np.unique(week[m]))
        n    = m.sum()
        a    = ARAT[m][0]
        f    = FMA[m][0]
        grp  = "healthy" if y[m][0] == 1 else "stroke"
        print(f"  {int(sid):>5}  {wks:>4}  {int(n):>6}  {a:>6.0f}  {f:>5.0f}  {grp}")

    print(f"\nTotal: {len(X)} windows, {len(np.unique(subj))} subjects")
    print("=" * 56)

if __name__ == "__main__":
    main()
