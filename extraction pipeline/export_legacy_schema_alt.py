#!/usr/bin/env python3
"""
export_legacy_schema_alt.py -- Build old-style visit-level schema from BIOPM ALT window-level output.

Input (from irb_extract_alt.py):
  features/biopm_features_alt.npz with keys:
    features, labels, subj, week, ARAT, FMA

Output (legacy-compatible):
  features/biopm_features_legacy_schema_alt.npz with keys:
    features, features_even, features_odd, feature_names,
    labels, pids, arat, fma, subjects, weeks
"""

import os
import argparse
from collections import defaultdict

import numpy as np


def even_odd_split(subj_arr, week_arr):
    groups = defaultdict(list)
    for i, (s, w) in enumerate(zip(subj_arr, week_arr)):
        groups[(int(s), int(w))].append(i)

    within_idx = np.zeros(len(subj_arr), dtype=np.int64)
    for indices in groups.values():
        for pos, global_i in enumerate(indices):
            within_idx[global_i] = pos

    even_mask = (within_idx % 2) == 0
    odd_mask = ~even_mask
    return even_mask, odd_mask


def aggregate_visits(features, labels, subj_arr, week_arr, arat_arr, fma_arr):
    groups = defaultdict(list)
    for i, (s, w) in enumerate(zip(subj_arr, week_arr)):
        groups[(int(s), int(w))].append(i)

    vf, vl, vs, vw, va, vfma = [], [], [], [], [], []
    for (s, w), indices in sorted(groups.items()):
        idx = np.array(indices)
        vf.append(features[idx].mean(axis=0))
        vl.append(int(labels[idx[0]]))
        vs.append(int(s))
        vw.append(int(w))
        va.append(float(arat_arr[idx[0]]))
        vfma.append(float(fma_arr[idx[0]]))

    return (
        np.array(vf, dtype=np.float32),
        np.array(vl, dtype=np.int32),
        np.array(vs, dtype=np.int32),
        np.array(vw, dtype=np.int32),
        np.array(va, dtype=np.float32),
        np.array(vfma, dtype=np.float32),
    )


def build_legacy_schema(source_npz, out_npz):
    d = np.load(source_npz, allow_pickle=True)

    X_win = np.ascontiguousarray(d["features"], dtype=np.float32)
    y_win = d["labels"].astype(np.int32)
    subj_win = d["subj"].astype(np.int32)
    week_win = d["week"].astype(np.int32)
    arat_win = d["ARAT"].astype(np.float32)
    fma_win = d["FMA"].astype(np.float32)

    features, labels, subjects, weeks, arat, fma = aggregate_visits(
        X_win, y_win, subj_win, week_win, arat_win, fma_win
    )

    even_mask, odd_mask = even_odd_split(subj_win, week_win)
    features_even, _, se, we, _, _ = aggregate_visits(
        X_win[even_mask], y_win[even_mask], subj_win[even_mask], week_win[even_mask], arat_win[even_mask], fma_win[even_mask]
    )
    features_odd, _, so, wo, _, _ = aggregate_visits(
        X_win[odd_mask], y_win[odd_mask], subj_win[odd_mask], week_win[odd_mask], arat_win[odd_mask], fma_win[odd_mask]
    )

    if not (np.array_equal(subjects, se) and np.array_equal(weeks, we) and np.array_equal(subjects, so) and np.array_equal(weeks, wo)):
        raise RuntimeError("Visit alignment mismatch between full/even/odd aggregates.")

    unique_subjects = sorted(set(subjects.tolist()))
    subj_to_int = {s: i for i, s in enumerate(unique_subjects)}
    pids = np.array([subj_to_int[s] for s in subjects], dtype=np.int32)

    feature_names = np.array([f"biopm_{i:04d}" for i in range(features.shape[1])], dtype=object)

    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez(
        out_npz,
        features=features,
        features_even=features_even,
        features_odd=features_odd,
        feature_names=feature_names,
        labels=labels,
        pids=pids,
        arat=arat,
        fma=fma,
        subjects=subjects,
        weeks=weeks,
    )

    print("=" * 64)
    print("Legacy Schema Export ALT")
    print("=" * 64)
    print(f"Source: {source_npz}")
    print(f"Output: {out_npz}")
    print(f"  features      : {features.shape}")
    print(f"  features_even : {features_even.shape}")
    print(f"  features_odd  : {features_odd.shape}")
    print(f"  labels        : {labels.shape}  (1=healthy, 0=stroke)")
    print("=" * 64)


def main():
    p = argparse.ArgumentParser(description="Export BIOPM legacy-compatible visit-level schema (ALT)")
    p.add_argument("--source", default="features/biopm_features_alt.npz")
    p.add_argument("--output", default="features/biopm_features_legacy_schema_alt.npz")
    args = p.parse_args()

    if not os.path.isfile(args.source):
        raise FileNotFoundError(f"Source features not found: {args.source}")

    build_legacy_schema(args.source, args.output)


if __name__ == "__main__":
    main()
