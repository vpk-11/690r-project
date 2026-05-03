#!/usr/bin/env python3
"""
irb_preprocess.py -- Preprocess IRB stroke .npz dataset for Bio-PM.

Groups windows by (subject, week), runs Bio-PM preprocessing
on each wnd.acc, and saves HDF5 files for irb_extract.py.

Usage:
    export BIOPM_ROOT=/path/to/CS690TR
    python irb_preprocess.py --output preprocessed/
    python irb_preprocess.py --output preprocessed/ --subject 549  # single subject test
"""

import os
import sys
import argparse
from collections import defaultdict

import h5py
import numpy as np
from tqdm import tqdm

BIOPM_ROOT = os.environ.get("BIOPM_ROOT", "")
if not BIOPM_ROOT or not os.path.isdir(BIOPM_ROOT):
    print("ERROR: set BIOPM_ROOT env variable to the CS690TR directory.")
    sys.exit(1)
sys.path.insert(0, BIOPM_ROOT)

from src.data.preprocessing import (
    bandpass_filter, lowpass_filter, detect_zero_crossings,
)

CONFIG = {
    "HighF1": 12, "LowF1": 0.5, "Order1": 6,
    "target_FS": 30,
    "WS": 3,        # 3-second windows
    "pad_size": 57, # int(3 * 192 / 10)
    "normalize_size_target": 32,
    "normalize_size_assign": 32,
}

HEALTHY_ARAT = 57
HEALTHY_FMA  = 66


def process_window(raw_acc, config):
    """
    Run Bio-PM preprocessing on one (90, 3) window.
    Returns (x_acc_filt, x_gravity) or (None, None) if no MEs found.

    x_acc_filt : (pad_size, 38) float32, NaN-padded
    x_gravity  : (T, 3)         float32
    """
    T = raw_acc.shape[0]
    t = np.arange(T) / config["target_FS"]

    try:
        acc_filt = bandpass_filter(raw_acc, config["LowF1"], config["HighF1"],
                                   config["target_FS"], order=config["Order1"])
        acc_grav = lowpass_filter(raw_acc, config["LowF1"],
                                  config["target_FS"], order=config["Order1"])
    except Exception:
        return None, None

    try:
        (_, _, me_list, me_norm, me_info, _, _, pos_info,
         zc_list, _) = detect_zero_crossings(acc_filt, t, config)
    except Exception:
        return None, None

    n_me = len(me_list)
    if n_me == 0:
        # No movement elements: write all-NaN x_acc_filt.
        # The model's padding mask will correctly exclude all tokens.
        # The gravity stream is still valid and non-zero.
        x_acc = np.full((config["pad_size"], 38), np.nan, dtype=np.float32)
        return x_acc, acc_grav.astype(np.float32)

    # Pack: [ME_norm(32) | pos(1) | axis,len,min,max,dirct(5)] = 38 cols
    # Axis is 0-indexed (0,1,2) — correct for model's Embedding(4) where 3=padding
    x_acc_valid = np.concatenate([
        me_norm,
        np.array(pos_info).reshape(-1, 1),
        me_info[['axis', 'len', 'min', 'max', 'dirct']].values,
    ], axis=1)

    # Pad to pad_size with NaN (model uses isnan() to detect padding)
    if n_me < config["pad_size"]:
        pad = np.full((config["pad_size"] - n_me, 38), np.nan)
        x_acc = np.vstack([x_acc_valid, pad]).astype(np.float32)
    else:
        x_acc = x_acc_valid[: config["pad_size"]].astype(np.float32)

    return x_acc, acc_grav.astype(np.float32)


def main():
    p = argparse.ArgumentParser(description="Preprocess IRB .npz for Bio-PM")
    p.add_argument("--output",   required=True, help="Output directory for HDF5 files")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--subject",  type=int, default=None,
                   help="Process only this subject (for testing)")
    args = p.parse_args()

    print("=" * 64)
    print("Bio-PM IRB Preprocessing")
    print("=" * 64)
    print(f"  data_dir : {args.data_dir}")
    print(f"  output   : {args.output}")
    print(f"  WS={CONFIG['WS']}s  pad_size={CONFIG['pad_size']}  30 Hz")
    print()

    print("Loading clinical_scores.npz ...", flush=True)
    clin = np.load(f"{args.data_dir}/clinical_scores.npz",
                   allow_pickle=True)['clinical_scores'].item()
    print(f"  {len(clin)} (subj, week) entries loaded.")

    print("Loading windows.npz (5 GB -- this takes ~1-2 min) ...", flush=True)
    windows_all = np.load(f"{args.data_dir}/windows.npz",
                          allow_pickle=True)['windows']
    print(f"  {len(windows_all)} windows loaded.")

    # Group by (subj, week)
    print("Grouping windows by (subject, week) ...", flush=True)
    groups = defaultdict(list)
    for wnd in tqdm(windows_all, desc="Grouping", unit="win", dynamic_ncols=True):
        if args.subject is not None and wnd.subject != args.subject:
            continue
        groups[(wnd.subject, wnd.week)].append(wnd)

    print(f"(subj, week) groups to process: {len(groups)}\n")
    os.makedirs(args.output, exist_ok=True)

    n_saved = n_failed = 0

    sorted_keys = sorted(groups.keys())
    outer_bar = tqdm(sorted_keys, desc="Groups", unit="group", dynamic_ncols=True)

    for (subj, week) in outer_bar:
        outer_bar.set_postfix(subj=subj, week=week)
        key = (subj, week)
        if key not in clin:
            tqdm.write(f"  ({subj}, wk{week}): not in clinical -- skip")
            n_failed += 1
            continue

        arat  = clin[key].ARAT
        fma   = clin[key].FMA
        label = 1 if (arat == HEALTHY_ARAT and fma == HEALTHY_FMA) else 0
        group = "healthy" if label == 1 else "stroke"
        wlist = groups[key]

        raw_list, x_acc_list, x_grav_list, lbl_list = [], [], [], []
        n_skip = 0

        for wnd in tqdm(wlist, desc=f"  s{subj:03d}/wk{week:02d}", unit="win",
                        leave=False, dynamic_ncols=True):
            x_acc, x_grav = process_window(wnd.acc.astype(np.float32), CONFIG)
            if x_acc is None:
                n_skip += 1
                continue
            raw_list.append(wnd.acc.astype(np.float32))
            x_acc_list.append(x_acc)
            x_grav_list.append(x_grav)
            lbl_list.append(float(label))

        n_valid = len(lbl_list)
        tqdm.write(f"  Subject {subj:3d}  wk{week:02d}  [{group:7s}]  "
                   f"ARAT={arat:.0f}  FMA={fma:.0f}  "
                   f"valid={n_valid}  skip={n_skip}")

        if n_valid == 0:
            n_failed += 1
            continue

        # Log what % of tokens are valid (not NaN)
        x_acc_arr = np.array(x_acc_list)   # (W, pad_size, 38)
        valid_tok = ~np.isnan(x_acc_arr[:, :, 0])   # (W, pad_size)
        pct_valid = 100 * valid_tok.mean()
        if pct_valid < 10:
            tqdm.write(f"    WARNING: only {pct_valid:.1f}% tokens valid "
                       f"(NaN-padded). Transformer stream unreliable.")

        h5_path = os.path.join(args.output, f"Data_MeLabel_{subj}_{week}.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("window_acc_raw", data=np.array(raw_list,   dtype=np.float32))
            f.create_dataset("x_acc_filt",     data=np.array(x_acc_list, dtype=np.float32))
            f.create_dataset("x_gravity",      data=np.array(x_grav_list, dtype=np.float32))
            f.create_dataset("window_label",   data=np.array(lbl_list,   dtype=np.float32))
            f.attrs["subject"] = subj
            f.attrs["week"]    = week
            f.attrs["ARAT"]    = float(arat)
            f.attrs["FMA"]     = float(fma)
            f.attrs["group"]   = group
        n_saved += 1

    print()
    print("=" * 64)
    print(f"Done.  {n_saved} files saved,  {n_failed} empty/failed.")
    print(f"\nNext: python irb_extract.py --preprocessed {args.output} ...")
    print("=" * 64)


if __name__ == "__main__":
    main()
