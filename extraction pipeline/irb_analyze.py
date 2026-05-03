#!/usr/bin/env python3
"""
irb_analyze.py -- Even/odd split, UMAP visualization, LOSO LR baseline.

Tasks:
  1. Even/odd window split within each (subj, week) group
  2. UMAP visualization -- visit-level 1028-d features, colored healthy/stroke
  3. LOSO logistic regression -- visit-level, report AUC

Outputs:
  results/splits/split_indices.npz
  results/figures/umap_healthy_vs_impaired.png
  results/lr_loso_results.txt

Usage:
  python irb_analyze.py --features features/biopm_features.npz
"""

import os
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import umap
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)


def load_features(path):
    d = np.load(path, allow_pickle=True)
    return (d["features"],
            d["labels"].astype(int),
            d["subj"].astype(int),
            d["week"].astype(int),
            d["ARAT"].astype(float),
            d["FMA"].astype(float))


# -----------------------------------------------------------------------
# 1. Even/Odd split
# -----------------------------------------------------------------------

def even_odd_split(subj_arr, week_arr):
    """
    Within each (subj, week) group, assign even-position windows to train
    and odd-position windows to test (positions are 0-based in load order).

    Returns bool arrays: train_mask, test_mask  (length N each).
    """
    groups = defaultdict(list)
    for i, (s, w) in enumerate(zip(subj_arr, week_arr)):
        groups[(s, w)].append(i)

    within_idx = np.zeros(len(subj_arr), dtype=np.int64)
    for indices in groups.values():
        for pos, global_i in enumerate(indices):
            within_idx[global_i] = pos

    train_mask = (within_idx % 2) == 0
    test_mask  = (within_idx % 2) == 1
    return train_mask, test_mask


# -----------------------------------------------------------------------
# 2. Visit-level aggregation
# -----------------------------------------------------------------------

def aggregate_visits(features, labels, subj_arr, week_arr, arat_arr, fma_arr):
    """Mean-pool windows within each (subj, week) to one visit-level vector."""
    groups = defaultdict(list)
    for i, (s, w) in enumerate(zip(subj_arr, week_arr)):
        groups[(s, w)].append(i)

    vf, vl, vs, vw, va, vfma = [], [], [], [], [], []
    for (s, w), indices in sorted(groups.items()):
        idx = np.array(indices)
        vf.append(features[idx].mean(axis=0))
        vl.append(int(labels[idx[0]]))
        vs.append(int(s))
        vw.append(int(w))
        va.append(float(arat_arr[idx[0]]))
        vfma.append(float(fma_arr[idx[0]]))

    return (np.array(vf), np.array(vl), np.array(vs),
            np.array(vw), np.array(va), np.array(vfma))


# -----------------------------------------------------------------------
# 3. LOSO logistic regression
# -----------------------------------------------------------------------

def loso_lr(visit_features, visit_labels, visit_subj, results_path):
    """
    Leave-One-Subject-Out LR on visit-level features.
    Uses class_weight='balanced' for the 4:194 healthy:stroke imbalance.
    Returns overall AUC and per-visit predicted probabilities.
    """
    unique_subjects = np.unique(visit_subj)
    all_probs = np.zeros(len(visit_labels), dtype=float)
    per_subj_aucs = []

    for subj in tqdm(unique_subjects, desc="LOSO LR", unit="subj", dynamic_ncols=True):
        test_mask  = visit_subj == subj
        train_mask = ~test_mask

        X_tr, y_tr = visit_features[train_mask], visit_labels[train_mask]
        X_te, y_te = visit_features[test_mask],  visit_labels[test_mask]

        if len(np.unique(y_tr)) < 2:
            tqdm.write(f"  s{subj}: skip -- training set has only one class")
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        lr = LogisticRegression(max_iter=2000, class_weight="balanced",
                                C=1.0, solver="lbfgs")
        lr.fit(X_tr_s, y_tr)
        probs = lr.predict_proba(X_te_s)[:, 1]
        all_probs[test_mask] = probs

        if len(np.unique(y_te)) > 1:
            auc = roc_auc_score(y_te, probs)
            per_subj_aucs.append((subj, auc, int(y_te.sum())))
            tqdm.write(f"  s{subj}: AUC={auc:.3f}  healthy_visits={int(y_te.sum())}")

    overall_auc = roc_auc_score(visit_labels, all_probs)

    lines = [
        "=" * 56,
        "LOSO Logistic Regression Results (visit-level)",
        "=" * 56,
        f"Total visits   : {len(visit_labels)}",
        f"Healthy visits : {int(visit_labels.sum())}",
        f"Stroke visits  : {int((visit_labels == 0).sum())}",
        "",
        "Subjects with mixed labels (AUC computable):",
    ]
    for s, a, n in per_subj_aucs:
        lines.append(f"  Subject {s:3d}: AUC={a:.3f}  healthy_visits={n}")
    lines += [
        "",
        f"Overall LOSO AUC : {overall_auc:.4f}",
        f"Threshold check  : {'PASS (>0.6)' if overall_auc > 0.6 else 'FAIL (<0.6) -- investigate labels'}",
        "=" * 56,
    ]

    report = "\n".join(lines)
    print(report)

    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    with open(results_path, "w") as f:
        f.write(report + "\n")

    return overall_auc, all_probs


# -----------------------------------------------------------------------
# 4. UMAP visualization
# -----------------------------------------------------------------------

def run_umap(visit_features, visit_labels, out_path):
    """2D UMAP of visit-level 1028-d features, colored healthy vs stroke."""
    print(f"\nComputing UMAP on {len(visit_features)} visits ...")

    scaler = StandardScaler()
    X_s = scaler.fit_transform(visit_features)

    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                        random_state=42, verbose=False)
    emb = reducer.fit_transform(X_s)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    palette = {0: "#E05C5C", 1: "#4CAF50"}
    legend_labels = {
        0: f"Stroke  (n={int((visit_labels == 0).sum())})",
        1: f"Healthy (n={int((visit_labels == 1).sum())})",
    }

    for lbl in [0, 1]:
        mask = visit_labels == lbl
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=palette[lbl], label=legend_labels[lbl],
                   s=70, alpha=0.85, edgecolors="white", linewidths=0.4)

    ax.set_title("UMAP: Bio-PM Embeddings (visit-level)\nHealthy vs Stroke", fontsize=13)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(frameon=True, fontsize=10)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    return emb


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    os.chdir(REPO_ROOT)
    p = argparse.ArgumentParser(description="Bio-PM IRB: split, UMAP, LOSO LR")
    p.add_argument("--features", default="features/biopm_features.npz")
    p.add_argument("--results",  default="results")
    args = p.parse_args()

    print("=" * 56)
    print("Bio-PM IRB Analysis")
    print("=" * 56)

    print(f"\nLoading {args.features} ...")
    features, labels, subj, week, arat, fma = load_features(args.features)
    N = len(labels)
    print(f"  Windows  : {N}")
    print(f"  Healthy  : {labels.sum()} windows")
    print(f"  Stroke   : {(labels == 0).sum()} windows")
    print(f"  Shape    : {features.shape}")

    # --- 1. Even/odd split ---
    print("\n[1/3] Even/odd window split ...")
    train_mask, test_mask = even_odd_split(subj, week)
    split_path = os.path.join(args.results, "splits", "split_indices.npz")
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    np.savez(split_path, train_mask=train_mask, test_mask=test_mask)
    print(f"  Train windows : {train_mask.sum()}")
    print(f"  Test  windows : {test_mask.sum()}")
    print(f"  Train healthy : {labels[train_mask].sum()}")
    print(f"  Test  healthy : {labels[test_mask].sum()}")
    print(f"  Saved: {split_path}")

    # --- Visit-level aggregation ---
    print("\nAggregating to visit level ...")
    vf, vl, vs, vw, va, vfma = aggregate_visits(features, labels, subj, week, arat, fma)
    n_healthy = int(vl.sum())
    n_stroke  = int((vl == 0).sum())
    print(f"  Total visits   : {len(vl)}")
    print(f"  Healthy visits : {n_healthy}")
    print(f"  Stroke  visits : {n_stroke}")

    if n_healthy != 4:
        print(f"  WARNING: expected 4 healthy visits, got {n_healthy}")
    if n_stroke != 194:
        print(f"  WARNING: expected 194 stroke visits, got {n_stroke}")

    # --- 2. UMAP ---
    print("\n[2/3] UMAP visualization ...")
    umap_path = os.path.join(args.results, "figures", "umap_healthy_vs_impaired.png")
    run_umap(vf, vl, umap_path)

    # --- 3. LOSO LR ---
    print("\n[3/3] LOSO logistic regression ...")
    lr_path = os.path.join(args.results, "lr_loso_results.txt")
    overall_auc, _ = loso_lr(vf, vl, vs, lr_path)

    print()
    print("=" * 56)
    print(f"LOSO AUC : {overall_auc:.4f}  "
          f"{'PASS' if overall_auc > 0.6 else 'FAIL -- investigate'}")
    print(f"Outputs  : {args.results}/")
    print("=" * 56)


if __name__ == "__main__":
    main()
