#!/usr/bin/env python3
"""
debug_pipeline.py -- Step-by-step diagnostic to find why embeddings are all zeros.

Run BEFORE irb_preprocess.py and irb_extract.py.
Traces through the entire pipeline on a single subject and prints
value statistics at every stage. Read every line of output.

Usage:
    export BIOPM_ROOT=/path/to/CS690TR
    python debug_pipeline.py --subject 549
    python debug_pipeline.py --subject 549 --n_windows 10
"""

import os
import sys
import argparse
import numpy as np

BIOPM_ROOT = os.environ.get("BIOPM_ROOT", "")
if not BIOPM_ROOT or not os.path.isdir(BIOPM_ROOT):
    print("ERROR: set BIOPM_ROOT env variable to the CS690TR directory.")
    print("  export BIOPM_ROOT=/path/to/CS690TR")
    sys.exit(1)
sys.path.insert(0, BIOPM_ROOT)

from src.data.preprocessing import bandpass_filter, lowpass_filter, detect_zero_crossings

def hr(label=""):
    w = 64
    if label:
        pad = (w - len(label) - 2) // 2
        print("=" * pad + f" {label} " + "=" * (w - pad - len(label) - 2))
    else:
        print("=" * w)

def check_array(name, arr, indent=4):
    sp = " " * indent
    if arr is None:
        print(f"{sp}{name}: None")
        return
    flat = arr.flatten()
    n_nan = int(np.isnan(flat).sum())
    n_inf = int(np.isinf(flat).sum())
    finite = flat[np.isfinite(flat)]
    if len(finite) > 0:
        print(f"{sp}{name}: shape={arr.shape}  dtype={arr.dtype}")
        print(f"{sp}  range=[{finite.min():.4f}, {finite.max():.4f}]"
              f"  mean={finite.mean():.4f}  std={finite.std():.4f}")
        print(f"{sp}  NaN={n_nan}/{len(flat)}  Inf={n_inf}")
        if n_nan == len(flat):
            print(f"{sp}  *** ALL NaN — this will produce zero/garbage embeddings ***")
        elif n_nan > len(flat) * 0.9:
            print(f"{sp}  *** >90% NaN — very sparse, embeddings will be unreliable ***")
    else:
        print(f"{sp}{name}: ALL NaN or Inf — {arr.shape}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject",   type=int, required=True, help="Subject ID to test")
    p.add_argument("--data_dir",  default="data")
    p.add_argument("--n_windows", type=int, default=20, help="How many windows to test")
    p.add_argument("--checkpoint", default=None,
                   help="Path to checkpoint.pt (skip model test if not provided)")
    args = p.parse_args()

    CONFIG = {
        "HighF1": 12, "LowF1": 0.5, "Order1": 6,
        "target_FS": 30, "WS": 3, "pad_size": 57,
        "normalize_size_target": 32, "normalize_size_assign": 32,
    }

    # ----------------------------------------------------------------
    hr("Stage 1: Load data")
    # ----------------------------------------------------------------
    clin_path = f"{args.data_dir}/clinical_scores.npz"
    wins_path = f"{args.data_dir}/windows.npz"

    clin = np.load(clin_path, allow_pickle=True)['clinical_scores'].item()
    windows_all = np.load(wins_path, allow_pickle=True)['windows']

    subj_windows = [w for w in windows_all if w.subject == args.subject]
    if not subj_windows:
        print(f"ERROR: subject {args.subject} not found. Available: "
              f"{sorted(set(w.subject for w in windows_all))[:20]}")
        sys.exit(1)

    test_windows = subj_windows[:args.n_windows]
    clin_key = (args.subject, test_windows[0].week)
    if clin_key in clin:
        arat = clin[clin_key].ARAT
        fma  = clin[clin_key].FMA
        print(f"Subject {args.subject}: ARAT={arat:.0f}  FMA={fma:.0f}  "
              f"({'healthy' if arat==57 and fma==66 else 'stroke'})")
    print(f"Testing {len(test_windows)} windows")

    # ----------------------------------------------------------------
    hr("Stage 2: Raw acceleration")
    # ----------------------------------------------------------------
    w0 = test_windows[0]
    print(f"wnd.acc shape: {w0.acc.shape}  sample_rate: {w0.sample_rate}")
    check_array("wnd.acc", w0.acc)

    # ----------------------------------------------------------------
    hr("Stage 3: Bandpass filter (0.5-12 Hz)")
    # ----------------------------------------------------------------
    t = np.arange(w0.acc.shape[0]) / CONFIG["target_FS"]
    try:
        acc_filt = bandpass_filter(w0.acc, CONFIG["LowF1"], CONFIG["HighF1"],
                                   CONFIG["target_FS"], order=CONFIG["Order1"])
        check_array("acc_filt", acc_filt)
    except Exception as e:
        print(f"BANDPASS FAILED: {e}")
        print("  This is a problem — check window length vs filter order")
        sys.exit(1)

    # Check if signal is flat (too-aggressively pre-filtered)
    signal_power = np.mean(acc_filt ** 2)
    print(f"  Signal power after bandpass: {signal_power:.6f}")
    if signal_power < 1e-6:
        print("  *** NEAR-ZERO POWER — dataset's LP filter may have removed "
              "frequencies Bio-PM needs. Zero-crossing detection will likely fail. ***")

    # ----------------------------------------------------------------
    hr("Stage 4: Zero-crossing movement element detection")
    # ----------------------------------------------------------------
    me_counts = []
    fail_count = 0

    for i, wnd in enumerate(test_windows):
        t_w = np.arange(wnd.acc.shape[0]) / CONFIG["target_FS"]
        try:
            acc_f = bandpass_filter(wnd.acc, CONFIG["LowF1"], CONFIG["HighF1"],
                                    CONFIG["target_FS"], order=CONFIG["Order1"])
            (_, _, me_list, me_norm, me_info, _, _, pos_info,
             zc_list, _) = detect_zero_crossings(acc_f, t_w, CONFIG)
            me_counts.append(len(me_list))
        except Exception as e:
            fail_count += 1
            me_counts.append(0)

    me_counts = np.array(me_counts)
    print(f"Windows tested : {len(me_counts)}")
    print(f"Failures       : {fail_count}")
    print(f"MEs per window : min={me_counts.min()}  max={me_counts.max()}  "
          f"mean={me_counts.mean():.1f}  median={np.median(me_counts):.1f}")
    print(f"Windows with 0 MEs: {(me_counts == 0).sum()}/{len(me_counts)} "
          f"({100*(me_counts==0).mean():.1f}%)")

    if me_counts.mean() < 3:
        print()
        print("*** WARNING: Very few movement elements detected on average.")
        print("    Bio-PM transformer stream [0:128] will produce near-zero embeddings.")
        print("    The gravity stream [128:1028] will still work correctly.")
        print("    Consider using gravity-only features for downstream analysis.")
    elif me_counts.mean() < 10:
        print()
        print("  NOTE: Low ME count. Std-pool [64:128] may be near-zero (few unique tokens).")
    else:
        print()
        print("  ME count looks healthy. Transformer stream should work.")

    # Show one example in detail
    hr("Stage 5: Inspect one ME extraction in detail")
    t_w = np.arange(test_windows[0].acc.shape[0]) / CONFIG["target_FS"]
    acc_f = bandpass_filter(test_windows[0].acc, CONFIG["LowF1"], CONFIG["HighF1"],
                            CONFIG["target_FS"], order=CONFIG["Order1"])
    (_, _, me_list, me_norm, me_info, _, _, pos_info,
     zc_list, _) = detect_zero_crossings(acc_f, t_w, CONFIG)

    n_me = len(me_list)
    print(f"MEs found in window 0: {n_me}")
    if n_me > 0:
        check_array("me_norm (ME patches)", me_norm)
        print(f"  me_info columns: {list(me_info.columns)}")
        print(f"  me_info sample:\n{me_info.head(3).to_string()}")
        check_array("pos_info", np.array(pos_info))
    else:
        print("  No MEs found — entire x_acc_filt will be NaN-padded for this window.")

    # Gravity
    hr("Stage 6: Gravity stream")
    acc_grav = lowpass_filter(test_windows[0].acc, CONFIG["LowF1"],
                              CONFIG["target_FS"], order=CONFIG["Order1"])
    check_array("gravity signal", acc_grav)

    # ----------------------------------------------------------------
    hr("Stage 7: Build x_acc_filt (what gets written to HDF5)")
    # ----------------------------------------------------------------
    if n_me > 0:
        x_acc = np.concatenate([
            me_norm,
            np.array(pos_info).reshape(-1, 1),
            me_info[['axis', 'len', 'min', 'max', 'dirct']].values,
        ], axis=1)
        pad = np.full((CONFIG["pad_size"] - n_me, x_acc.shape[1]), np.nan)
        x_acc_padded = np.vstack([x_acc, pad]) if n_me < CONFIG["pad_size"] else x_acc[:CONFIG["pad_size"]]
        check_array("x_acc_filt (padded)", x_acc_padded)
        n_nan_rows = np.isnan(x_acc_padded).all(axis=1).sum()
        n_valid_rows = CONFIG["pad_size"] - n_nan_rows
        print(f"  Valid token rows : {n_valid_rows}/{CONFIG['pad_size']}")
        print(f"  NaN-padded rows  : {n_nan_rows}/{CONFIG['pad_size']}")
        if n_nan_rows == CONFIG["pad_size"]:
            print("  *** ALL ROWS NaN — this window produces no signal to Bio-PM ***")
    else:
        print("  x_acc_filt: all NaN (no MEs detected)")
        print("  *** THIS IS THE ZERO-EMBEDDING BUG — no valid tokens to encode ***")

    # ----------------------------------------------------------------
    hr("Stage 8: Model checkpoint (if provided)")
    # ----------------------------------------------------------------
    if args.checkpoint and os.path.isfile(args.checkpoint):
        import torch
        from src.models.biopm import BioPMModel, masked_mean_std
        try:
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            print(f"Checkpoint type: {type(ckpt)}")
            if isinstance(ckpt, dict):
                print(f"Checkpoint keys: {list(ckpt.keys())[:5]}")

            model = BioPMModel(n_classes=2)
            result = model.encoder_acc.load_state_dict(ckpt, strict=False)
            print(f"Missing keys   : {len(result.missing_keys)}")
            print(f"Unexpected keys: {len(result.unexpected_keys)}")
            if result.missing_keys:
                print(f"  Sample missing: {result.missing_keys[:5]}")
            if result.unexpected_keys:
                print(f"  Sample unexpected: {result.unexpected_keys[:5]}")
            if len(result.missing_keys) > 5:
                print("  *** MANY MISSING KEYS — model may be running with random weights ***")
                print("  *** This would cause inconsistent (but not necessarily zero) embeddings ***")
            else:
                print("  Checkpoint loaded OK.")

            # Quick forward test with the one window
            if n_me > 0:
                model.eval()
                with torch.no_grad():
                    patches = torch.from_numpy(x_acc_padded[:, :32]).float().unsqueeze(0)
                    pos_t   = torch.from_numpy(x_acc_padded[:, 32]).float().unsqueeze(0)
                    add_t   = torch.from_numpy(x_acc_padded[:, 33:]).float().unsqueeze(0)
                    mask_t  = torch.zeros(1, CONFIG["pad_size"])

                    tokens = model.encoder_acc(patches, pos_t, mask_t, add_t)
                    pooled = masked_mean_std(tokens)

                    print(f"\nForward pass output:")
                    check_array("acc_tokens (B,L,64)", tokens.numpy()[0])
                    check_array("pooled (B,128)", pooled.numpy())

                    n_zero = (pooled.abs() < 1e-6).sum().item()
                    print(f"  Near-zero values in pooled: {n_zero}/128")
                    if n_zero > 100:
                        print("  *** POOLED IS NEARLY ALL ZERO — confirms the zero-embedding bug ***")
                    else:
                        print("  Pooled looks non-zero. Model is working.")
        except Exception as e:
            print(f"ERROR loading checkpoint: {e}")
    else:
        print("No checkpoint provided — skipping model test.")
        print("Pass --checkpoint $BIOPM_ROOT/checkpoints/checkpoint.pt to test.")

    # ----------------------------------------------------------------
    hr("Summary & Recommendations")
    # ----------------------------------------------------------------
    avg_me = me_counts.mean()
    pct_zero_me = 100 * (me_counts == 0).mean()

    print(f"Mean MEs per window : {avg_me:.1f}")
    print(f"% windows with 0 MEs: {pct_zero_me:.1f}%")
    print()

    if avg_me < 2:
        print("DIAGNOSIS: Zero-embedding bug confirmed.")
        print("  3-second windows have too few zero-crossings after bandpass filtering")
        print("  the already-smoothed dataset acceleration signal.")
        print()
        print("RECOMMENDATIONS:")
        print("  1. Use gravity-only features: npz['features'][:, 128:] — always non-zero")
        print("  2. OR: group windows into longer blocks before running Bio-PM")
        print("  3. OR: try without the bandpass pre-filter (comment it out in irb_preprocess.py)")
        print("     since the data is already smoothed")
    elif avg_me < 5:
        print("DIAGNOSIS: Low ME count — transformer stream likely unreliable.")
        print("  Use full 1028-d features but treat [0:128] with caution.")
        print("  Gravity [128:1028] is the reliable part.")
    else:
        print("DIAGNOSIS: ME count looks OK.")
        print("  If embeddings are still zero, check Stage 8 checkpoint loading output.")

    hr()

if __name__ == "__main__":
    main()
