#!/usr/bin/env python3
"""
irb_preprocess_alt.py -- ALT preprocessing: group 3 consecutive 3-second windows
into 9-second blocks, with pad_size=192.

ALT matches ADV preprocessing/padding semantics and differs only in temporal windowing:
  - ADV: 3s windows
  - ALT: 9s grouped blocks
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
    print("ERROR: set BIOPM_ROOT to the CS690TR directory.")
    print("  export BIOPM_ROOT=CS690TR")
    sys.exit(1)
sys.path.insert(0, BIOPM_ROOT)

from src.data.preprocessing import (
    bandpass_filter, lowpass_filter, detect_zero_crossings,
)

GROUP_SIZE = 3

CONFIG = {
    "HighF1": 12, "LowF1": 0.5, "Order1": 6,
    "target_FS": 30,
    "WS": 9,
    "pad_size": 192,
    "normalize_size_target": 32,
    "normalize_size_assign": 32,
}

HEALTHY_ARAT = 57
HEALTHY_FMA = 66


def process_block(raw_acc, config):
    """
    raw_acc: (GROUP_SIZE*90, 3) float32 -- default 270x3 (9 seconds at 30 Hz)

    Returns:
      x_acc_filt: (pad_size, 38) NaN-padded
      x_gravity : (T, 3)
      n_me_raw  : raw ME count before clipping/padding
    """
    T = raw_acc.shape[0]
    t = np.arange(T) / config["target_FS"]

    try:
        acc_filt = bandpass_filter(
            raw_acc, config["LowF1"], config["HighF1"],
            config["target_FS"], order=config["Order1"]
        )
        acc_grav = lowpass_filter(
            raw_acc, config["LowF1"],
            config["target_FS"], order=config["Order1"]
        )
    except Exception:
        return None, None, None

    try:
        (_, _, me_list, me_norm, me_info, _, _, pos_info,
         _, _) = detect_zero_crossings(acc_filt, t, config)
    except Exception:
        return None, None, None

    n_me = len(me_list)
    if n_me == 0:
        x_acc = np.full((config["pad_size"], 38), np.nan, dtype=np.float32)
        return x_acc, acc_grav.astype(np.float32), n_me

    x_acc_valid = np.concatenate([
        me_norm,
        np.array(pos_info).reshape(-1, 1),
        me_info[["axis", "len", "min", "max", "dirct"]].values,
    ], axis=1)

    if n_me < config["pad_size"]:
        pad = np.full((config["pad_size"] - n_me, 38), np.nan)
        x_acc = np.vstack([x_acc_valid, pad]).astype(np.float32)
    else:
        x_acc = x_acc_valid[: config["pad_size"]].astype(np.float32)

    return x_acc, acc_grav.astype(np.float32), n_me


def main():
    p = argparse.ArgumentParser(description="Bio-PM preprocessing ALT: 9-second grouped blocks, pad_size=192")
    p.add_argument("--output", default="preprocessed_alt", help="Output dir for HDF5 files")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--subject", type=int, default=None,
                   help="Process only this subject (testing)")
    p.add_argument("--group_size", type=int, default=GROUP_SIZE,
                   help=f"Windows per block (default: {GROUP_SIZE})")
    p.add_argument("--pad_size", type=int, default=192,
                   help="Pad/truncate x_acc_filt token length (default: 192)")
    args = p.parse_args()

    group_size = int(args.group_size)
    block_samples = group_size * 90
    block_seconds = block_samples / CONFIG["target_FS"]

    cfg = dict(CONFIG)
    cfg["WS"] = int(block_seconds)
    cfg["pad_size"] = int(args.pad_size)

    print("=" * 64)
    print("Bio-PM IRB Preprocessing ALT")
    print("=" * 64)
    print(f"  data_dir    : {args.data_dir}")
    print(f"  output      : {args.output}")
    print(f"  group_size  : {group_size} windows ({block_seconds:.0f}s per block)")
    print(f"  block_len   : {block_samples} samples")
    print(f"  pad_size    : {cfg['pad_size']}")
    print()

    print("Loading clinical_scores.npz ...", flush=True)
    clin = np.load(f"{args.data_dir}/clinical_scores.npz", allow_pickle=True)["clinical_scores"].item()
    print(f"  {len(clin)} (subj, week) entries loaded.")

    print("Loading windows.npz (this takes 1-2 min) ...", flush=True)
    windows_all = np.load(f"{args.data_dir}/windows.npz", allow_pickle=True)["windows"]
    print(f"  {len(windows_all):,} windows loaded.")

    print("Grouping windows by (subject, week) ...", flush=True)
    groups = defaultdict(list)
    for wnd in tqdm(windows_all, desc="Grouping", unit="win", dynamic_ncols=True):
        if args.subject is not None and wnd.subject != args.subject:
            continue
        groups[(wnd.subject, wnd.week)].append(wnd)

    print(f"(subj, week) groups to process: {len(groups)}\n")
    os.makedirs(args.output, exist_ok=True)

    n_saved = n_failed = 0
    total_blocks = total_me_sum = total_fill_sum = 0

    sorted_keys = sorted(groups.keys())
    outer_bar = tqdm(sorted_keys, desc="Groups", unit="group", dynamic_ncols=True)

    for (subj, week) in outer_bar:
        outer_bar.set_postfix(subj=subj, week=week)
        key = (subj, week)

        if key not in clin:
            tqdm.write(f"  ({subj}, wk{week}): not in clinical -- skip")
            n_failed += 1
            continue

        arat = clin[key].ARAT
        fma = clin[key].FMA
        label = 1 if (arat == HEALTHY_ARAT and fma == HEALTHY_FMA) else 0
        group = "healthy" if label == 1 else "stroke"
        wlist = groups[key]

        raw_list, x_acc_list, x_grav_list, lbl_list = [], [], [], []
        me_raw_list = []
        n_skip = 0

        for i in range(0, len(wlist) - group_size + 1, group_size):
            block_wnds = wlist[i:i + group_size]
            combined = np.concatenate([w.acc.astype(np.float32) for w in block_wnds], axis=0)

            x_acc, x_grav, n_me_raw = process_block(combined, cfg)
            if x_acc is None:
                n_skip += 1
                continue

            raw_list.append(combined)
            x_acc_list.append(x_acc)
            x_grav_list.append(x_grav)
            lbl_list.append(float(label))
            me_raw_list.append(int(n_me_raw))

        n_valid = len(lbl_list)
        me_mean = float(np.mean(me_raw_list)) if me_raw_list else 0.0
        me_p95 = float(np.percentile(me_raw_list, 95)) if me_raw_list else 0.0
        tqdm.write(
            f"  Subject {subj:3d}  wk{week:02d}  [{group:7s}]  "
            f"ARAT={arat:.0f}  FMA={fma:.0f}  "
            f"blocks={n_valid}  skip={n_skip}  "
            f"ME(mean/p95)={me_mean:.1f}/{me_p95:.1f}"
        )

        if n_valid == 0:
            n_failed += 1
            continue

        x_acc_arr = np.array(x_acc_list)
        valid_tok = ~np.isnan(x_acc_arr[:, :, 0])
        fill_pct = 100 * valid_tok.mean()
        me_per_blk = valid_tok.sum(axis=1).mean()

        total_blocks += n_valid
        total_me_sum += me_per_blk * n_valid
        total_fill_sum += fill_pct * n_valid

        h5_path = os.path.join(args.output, f"Data_MeLabel_{subj}_{week}.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("window_acc_raw", data=np.array(raw_list, dtype=np.float32))
            f.create_dataset("x_acc_filt", data=np.array(x_acc_list, dtype=np.float32))
            f.create_dataset("x_gravity", data=np.array(x_grav_list, dtype=np.float32))
            f.create_dataset("window_label", data=np.array(lbl_list, dtype=np.float32))
            f.attrs["subject"] = subj
            f.attrs["week"] = week
            f.attrs["ARAT"] = float(arat)
            f.attrs["FMA"] = float(fma)
            f.attrs["group"] = group
            f.attrs["group_size"] = group_size
            f.attrs["pad_size"] = int(cfg["pad_size"])
            f.attrs["pipeline"] = "alt"
        n_saved += 1

    avg_me = total_me_sum / max(total_blocks, 1)
    avg_fill = total_fill_sum / max(total_blocks, 1)

    print()
    print("=" * 64)
    print(f"Done.  {n_saved} files saved,  {n_failed} empty/failed.")
    print(f"Total blocks processed : {total_blocks:,}")
    print(f"Avg valid tokens/block : {avg_me:.1f}")
    print(f"Avg fill rate          : {avg_fill:.1f}%")
    print()
    print(f"Next: python irb_extract_alt.py --preprocessed {args.output} \\")
    print("        --output features/biopm_features_alt.npz")
    print("=" * 64)


if __name__ == "__main__":
    main()
