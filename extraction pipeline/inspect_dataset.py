#!/usr/bin/env python3
"""
inspect_dataset.py -- Characterize clinical_scores.npz and windows.npz.

Usage:
    python inspect_dataset.py
    python inspect_dataset.py --data_dir /path/to/data
"""
import argparse
import numpy as np
from collections import Counter

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    args = p.parse_args()

    clin_path = f"{args.data_dir}/clinical_scores.npz"
    wins_path = f"{args.data_dir}/windows.npz"

    def hr(label=""):
        w = 60
        if label:
            pad = (w - len(label) - 2) // 2
            print("=" * pad + f" {label} " + "=" * (w - pad - len(label) - 2))
        else:
            print("=" * w)

    # ---- clinical_scores ----
    hr("clinical_scores.npz")
    raw = np.load(clin_path, allow_pickle=True)
    clin = raw['clinical_scores'].item()
    keys = list(clin.keys())
    sample = clin[keys[0]]

    subjects = sorted(set(k[0] for k in keys))
    weeks    = sorted(set(k[1] for k in keys))
    arats    = np.array([clin[k].ARAT for k in keys])
    fmas     = np.array([clin[k].FMA  for k in keys])
    healthy  = {k[0] for k in keys if clin[k].ARAT == 57 and clin[k].FMA == 66}
    stroke   = {k[0] for k in keys} - healthy

    print(f"Total (subj, week) entries : {len(keys)}")
    print(f"Unique subjects            : {len(subjects)}  {subjects[:10]}...")
    print(f"Weeks                      : {weeks}")
    print(f"ARAT range                 : [{arats.min():.0f}, {arats.max():.0f}]")
    print(f"FMA  range                 : [{fmas.min():.0f}, {fmas.max():.0f}]")
    print(f"Healthy subjects (max score): {len(healthy)}  {sorted(healthy)}")
    print(f"Stroke  subjects            : {len(stroke)}   (sample: {sorted(stroke)[:5]}...)")

    # ---- windows ----
    hr("windows.npz")
    raw = np.load(wins_path, allow_pickle=True)
    windows = raw['windows']
    w0 = windows[0]

    print(f"Total windows     : {len(windows)}")
    print(f"window attributes : {list(vars(w0).keys())}")
    print(f"wnd.acc shape     : {w0.acc.shape}  (T=90 samples, 3 axes)")
    print(f"wnd.sample_rate   : {w0.sample_rate} Hz")
    win_sec = w0.acc.shape[0] / w0.sample_rate
    print(f"Window duration   : {win_sec:.1f} seconds")

    win_subs = sorted(set(w.subject for w in windows))
    win_wks  = sorted(set(w.week for w in windows))
    print(f"Subjects in windows: {len(win_subs)}  {win_subs[:10]}...")
    print(f"Weeks in windows   : {win_wks}")

    counts = Counter((w.subject, w.week) for w in windows)
    vals = list(counts.values())
    print(f"Windows per (subj,week): min={min(vals)}  max={max(vals)}  median={np.median(vals):.0f}")

    # acc value range
    acc_sample = np.stack([w.acc for w in windows[:100]])
    print(f"\nAcc value range (first 100 windows):")
    print(f"  min={acc_sample.min():.4f}  max={acc_sample.max():.4f}  mean={acc_sample.mean():.4f}")

    # Cross-reference
    hr("Cross-reference")
    clin_pairs = set(keys)
    win_pairs  = set((w.subject, w.week) for w in windows)
    both = clin_pairs & win_pairs
    only_clin = clin_pairs - win_pairs
    only_win  = win_pairs - clin_pairs
    print(f"(subj,week) pairs in both         : {len(both)}")
    print(f"In clinical only (no windows)     : {len(only_clin)}")
    print(f"In windows only  (no clinical)    : {len(only_win)}")

    hr("Summary")
    print(f"Bio-PM config to use:")
    print(f"  WS        = {win_sec:.0f}   (window size in seconds)")
    print(f"  target_FS = {w0.sample_rate}  (already correct rate, no resampling)")
    print(f"  pad_size  = {int(win_sec * 192 / 10)}   (int({win_sec:.0f} * 192 / 10))")
    hr()

if __name__ == "__main__":
    main()
