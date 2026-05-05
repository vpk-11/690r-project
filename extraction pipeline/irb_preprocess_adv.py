#!/usr/bin/env python3
"""
irb_preprocess_adv.py -- Adv preprocessing: groups 3 consecutive windows into
9-second blocks before Bio-PM feature extraction.

WHY THIS EXISTS:
  The standard irb_preprocess.py processes each 3-second window independently,
  producing ~18.6 MEs/window and a 33% fill rate (18.6/57 slots used).

  This script concatenates 3 consecutive windows (90+90+90 = 270 samples = 9s)
  before detection, yielding ~55-60 MEs/block and a ~96% fill rate (55/57).

  Higher fill rate means the std-pool [64:128] of the Bio-PM transformer stream
  becomes a reliable variance estimate instead of collapsing toward zero. This
  directly improves embedding quality for downstream classification.

TRADEOFF:
  You get 1/3 as many embeddings per visit (one per 9s block vs one per 3s window).
  A 24-hour visit with ~28,800 windows becomes ~9,600 blocks. Still more than
  enough for visit-level mean-pooling.

OUTPUT:
  preprocessed_adv/  -- same HDF5 schema as preprocessed/, fixed pad_size=57
  Gravity shape stored: (270, 3) instead of (90, 3). irb_extract_adv.py handles both
  because it interpolates x_gravity to TARGET_GRAV_LEN=300 regardless of T.

COMPATIBILITY:
  Output HDF5 schema is identical to irb_preprocess.py. irb_extract_adv.py,
  verify_embeddings.py, and export_legacy_schema_adv.py work with adv outputs.

Usage:
    export BIOPM_ROOT=CS690TR
    python irb_preprocess_adv.py --output preprocessed_adv/
    python irb_preprocess_adv.py --output preprocessed_adv/ --subject 549
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
    bandpass_filter, detect_zero_crossings,
)

# Number of consecutive windows to group into one block.
# 3 windows x 90 samples = 270 samples = 9 seconds at 30 Hz.
GROUP_SIZE = 3

# Base config values. In Adv, effective pad_size is fixed to 57 in main().
CONFIG = {
    "HighF1": 12, "LowF1": 0.5, "Order1": 6,
    "target_FS": 30,
    "WS": 9,
    "pad_size": 57,
    "normalize_size_target": 32,
    "normalize_size_assign": 32,
}

HEALTHY_ARAT = 57
HEALTHY_FMA  = 66


def process_block(raw_acc, config):
    """
    Run Bio-PM preprocessing on one concatenated block.

    raw_acc    : (GROUP_SIZE * 90, 3) float32  -- 9 seconds at 30 Hz
    config     : Bio-PM config dict

    Returns (x_acc_filt, x_gravity) tuple, same schema as irb_preprocess.py.
    x_acc_filt : (pad_size, 38) float32, NaN-padded
    x_gravity  : (T, 3)         float32   -- T = GROUP_SIZE * 90 = 270
    """
    T = raw_acc.shape[0]
    t = np.arange(T) / config["target_FS"]

    try:
        acc_filt = bandpass_filter(
            raw_acc, config["LowF1"], config["HighF1"],
            config["target_FS"], order=config["Order1"]
        )
        acc_grav = raw_acc.astype(np.float32)   # raw acc: best gravity proxy when DC is removed
    except Exception:
        return None, None

    try:
        (_, _, me_list, me_norm, me_info, _, _, pos_info,
         zc_list, _) = detect_zero_crossings(acc_filt, t, config)
    except Exception:
        return None, None

    n_me = len(me_list)

    if n_me == 0:
        # No MEs: gravity stream is still valid, transformer gets all-NaN tokens.
        x_acc = np.full((config["pad_size"], 38), np.nan, dtype=np.float32)
        return x_acc, acc_grav.astype(np.float32)

    x_acc_valid = np.concatenate([
        me_norm,
        np.array(pos_info).reshape(-1, 1),
        me_info[["axis", "len", "min", "max", "dirct"]].values,
    ], axis=1)

    if n_me < config["pad_size"]:
        pad   = np.full((config["pad_size"] - n_me, 38), np.nan)
        x_acc = np.vstack([x_acc_valid, pad]).astype(np.float32)
    else:
        x_acc = x_acc_valid[: config["pad_size"]].astype(np.float32)

    return x_acc, acc_grav.astype(np.float32)


def main():
    p = argparse.ArgumentParser(
        description="Bio-PM preprocessing Adv: 9-second grouped blocks, pad_size=57, raw gravity"
    )
    p.add_argument("--output",    default="preprocessed_adv", help="Output dir for HDF5 files")
    p.add_argument("--data_dir",  default="data")
    p.add_argument("--subject",   type=int, default=None,
                   help="Process only this subject (testing)")
    p.add_argument("--group_size", type=int, default=GROUP_SIZE,
                   help=f"Windows per block (default: {GROUP_SIZE})")
    args = p.parse_args()

    group_size = args.group_size
    block_samples = group_size * 90
    block_seconds = block_samples / CONFIG["target_FS"]
    pad_size = 57   # DO NOT scale — keep fixed to get ~100% fill from 9s blocks

    # Update config for the requested group size
    cfg = dict(CONFIG)
    cfg["WS"]       = int(block_seconds)
    cfg["pad_size"] = pad_size

    print("=" * 64)
    print("Bio-PM IRB Preprocessing Adv (9s grouped, pad_size=57, raw gravity)")
    print("=" * 64)
    print(f"  data_dir    : {args.data_dir}")
    print(f"  output      : {args.output}")
    print(f"  group_size  : {group_size} windows ({block_seconds:.0f}s per block)")
    print(f"  block_len   : {block_samples} samples")
    print(f"  pad_size    : {pad_size}  (vs 57 in standard pipeline)")
    print(f"  expected ME : ~{int(18.6 * group_size)} per block "
          f"(vs ~18 in standard pipeline)")
    print(f"  expected fill: ~{min(100, int(18.6 * group_size * 100 / pad_size))}% "
          f"(vs ~33% in standard pipeline)")
    print()

    print("Loading clinical_scores.npz ...", flush=True)
    clin = np.load(
        f"{args.data_dir}/clinical_scores.npz", allow_pickle=True
    )["clinical_scores"].item()
    print(f"  {len(clin)} (subj, week) entries loaded.")

    print("Loading windows.npz (this takes 1-2 min) ...", flush=True)
    windows_all = np.load(
        f"{args.data_dir}/windows.npz", allow_pickle=True
    )["windows"]
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

        arat  = clin[key].ARAT
        fma   = clin[key].FMA
        label = 1 if (arat == HEALTHY_ARAT and fma == HEALTHY_FMA) else 0
        group = "healthy" if label == 1 else "stroke"
        wlist = groups[key]

        raw_list, x_acc_list, x_grav_list, lbl_list = [], [], [], []
        n_skip = 0

        # Iterate in steps of group_size, drop the leftover tail.
        # Example: 28,800 windows with group_size=3 -> 9,600 blocks, last 0 dropped.
        for i in range(0, len(wlist) - group_size + 1, group_size):
            block_wnds = wlist[i : i + group_size]
            combined   = np.concatenate(
                [w.acc.astype(np.float32) for w in block_wnds], axis=0
            )  # (270, 3)

            x_acc, x_grav = process_block(combined, cfg)
            if x_acc is None:
                n_skip += 1
                continue

            raw_list.append(combined)
            x_acc_list.append(x_acc)
            x_grav_list.append(x_grav)
            lbl_list.append(float(label))

        n_valid = len(lbl_list)
        tqdm.write(
            f"  Subject {subj:3d}  wk{week:02d}  [{group:7s}]  "
            f"ARAT={arat:.0f}  FMA={fma:.0f}  "
            f"blocks={n_valid}  skip={n_skip}"
        )

        if n_valid == 0:
            n_failed += 1
            continue

        # Fill rate diagnostics
        x_acc_arr  = np.array(x_acc_list)          # (n_blocks, pad_size, 38)
        valid_tok  = ~np.isnan(x_acc_arr[:, :, 0]) # (n_blocks, pad_size)
        fill_pct   = 100 * valid_tok.mean()
        me_per_blk = valid_tok.sum(axis=1).mean()

        total_blocks    += n_valid
        total_me_sum    += me_per_blk * n_valid
        total_fill_sum  += fill_pct * n_valid

        if fill_pct < 50:
            tqdm.write(
                f"    NOTE: {fill_pct:.1f}% fill rate, "
                f"{me_per_blk:.1f} MEs/block -- "
                "lower than expected for grouped windows"
            )

        h5_path = os.path.join(args.output, f"Data_MeLabel_{subj}_{week}.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("window_acc_raw", data=np.array(raw_list,    dtype=np.float32))
            f.create_dataset("x_acc_filt",     data=np.array(x_acc_list,  dtype=np.float32))
            f.create_dataset("x_gravity",      data=np.array(x_grav_list, dtype=np.float32))
            f.create_dataset("window_label",   data=np.array(lbl_list,    dtype=np.float32))
            f.attrs["subject"]    = subj
            f.attrs["week"]       = week
            f.attrs["ARAT"]       = float(arat)
            f.attrs["FMA"]        = float(fma)
            f.attrs["group"]      = group
            f.attrs["group_size"] = group_size   # record alt config in file
            f.attrs["pad_size"]   = pad_size
        n_saved += 1

    # Summary statistics
    avg_me   = total_me_sum   / max(total_blocks, 1)
    avg_fill = total_fill_sum / max(total_blocks, 1)

    print()
    print("=" * 64)
    print(f"Done.  {n_saved} files saved,  {n_failed} empty/failed.")
    print(f"Total blocks processed : {total_blocks:,}")
    print(f"Avg MEs per block      : {avg_me:.1f}  (standard pipeline: ~18.6)")
    print(f"Avg fill rate          : {avg_fill:.1f}%  (standard pipeline: ~33%)")
    print()
    print(f"Next: python irb_extract_adv.py --preprocessed {args.output} \\")
    print(f"        --output features/biopm_features_adv.npz")
    print("=" * 64)


if __name__ == "__main__":
    main()
