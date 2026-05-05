#!/usr/bin/env python3
"""
irb_extract_v3.py -- Extract Bio-PM embeddings from preprocessed IRB HDF5 files (V3).

Custom extractor (not the generic extract_features.py) because:
  - Filenames are Data_MeLabel_{subj}_{week}.h5 (two IDs)
  - We want ARAT, FMA, week stored alongside features

Output .npz:
  features : (N, 1028)  Bio-PM embeddings
  labels   : (N,)       1=healthy, 0=stroke
  subj     : (N,)       subject ID
  week     : (N,)       assessment week
  ARAT     : (N,)       continuous clinical score
  FMA      : (N,)       continuous clinical score

Usage:
    export BIOPM_ROOT=/path/to/CS690TR
    python irb_extract_v3.py \
        --preprocessed preprocessed_v3/ \
        --checkpoint   $BIOPM_ROOT/checkpoints/checkpoint.pt \
        --output       features/biopm_features_v3.npz \
        --device       cpu
"""

import os
import sys
import re
import argparse

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

BIOPM_ROOT = os.environ.get("BIOPM_ROOT", "")
if not BIOPM_ROOT or not os.path.isdir(BIOPM_ROOT):
    print("ERROR: set BIOPM_ROOT to the CS690TR directory.")
    sys.exit(1)
sys.path.insert(0, BIOPM_ROOT)

from src.models.biopm import BioPMModel

TARGET_GRAV_LEN = 300   # gravity interpolated to this length
H5_PAT = re.compile(r'Data_MeLabel_(\d+)_(\d+)\.h5')
NORM_SIZE = 32


def masked_mean_std_valid(tokens):
    """
    Correct implementation of masked mean+std pooling.

    The original masked_mean_std in biopm.py ignores the mask argument and
    pools over all pad_size token slots, including NaN-padded ones. This
    function detects real vs padding tokens by their output magnitude and
    pools only over the real ones.

    Padding detection: the transformer outputs near-zero vectors (magnitude < 1e-4)
    for input rows that were NaN-padded in x_acc_filt. Real ME tokens produce
    outputs with substantially higher magnitude.

    tokens: (B, L, D) — transformer output, D=64
    returns: (B, 128) — [mean_over_real | std_over_real]
    """
    mag   = tokens.abs().mean(dim=-1)     # (B, L) — per-token magnitude
    valid = (mag > 1e-4)                   # (B, L) — True = real token

    B, L, D = tokens.shape
    out_mean = torch.zeros(B, D, device=tokens.device, dtype=tokens.dtype)
    out_std  = torch.zeros(B, D, device=tokens.device, dtype=tokens.dtype)

    for b in range(B):
        real_tokens = tokens[b, valid[b]]    # (n_real, D)
        n_real = real_tokens.shape[0]
        if n_real == 0:
            continue
        out_mean[b] = real_tokens.mean(dim=0)
        if n_real > 1:
            out_std[b] = real_tokens.std(dim=0)
        # if n_real == 1: std stays zero (single token has no variance)

    return torch.cat([out_mean, out_std], dim=-1)   # (B, 128)


def load_all_h5(data_root):
    """Load all Data_MeLabel_{s}_{w}.h5 files from a directory."""
    files = []
    for root, _, fnames in os.walk(data_root):
        for f in sorted(fnames):
            m = H5_PAT.match(f)
            if m:
                files.append((os.path.join(root, f), int(m.group(1)), int(m.group(2))))

    if not files:
        raise FileNotFoundError(
            f"No Data_MeLabel_*.h5 files in {data_root}.\n"
            "Run irb_preprocess_v3.py first."
        )

    x_acc_list, x_grav_list, raw_list = [], [], []
    lbl_list, subj_list, week_list, arat_list, fma_list = [], [], [], [], []

    for path, subj, week in tqdm(sorted(files), desc="Loading HDF5", unit="file", dynamic_ncols=True):
        with h5py.File(path, "r") as hf:
            xa   = np.array(hf["x_acc_filt"])       # (W, pad, 38)
            xg   = np.array(hf["x_gravity"])         # (W, T, 3)
            raw  = np.array(hf["window_acc_raw"])    # (W, T, 3)
            lbl  = np.array(hf["window_label"])      # (W,)
            arat = float(hf.attrs.get("ARAT", -1))
            fma  = float(hf.attrs.get("FMA",  -1))

        n = len(lbl)
        x_acc_list.append(xa)
        x_grav_list.append(xg)
        raw_list.append(raw)
        lbl_list.append(lbl)
        subj_list.append(np.full(n, subj))
        week_list.append(np.full(n, week))
        arat_list.append(np.full(n, arat))
        fma_list.append(np.full(n, fma))

    X_acc  = np.concatenate(x_acc_list)    # (N, pad, 38)
    X_grav = np.concatenate(x_grav_list)   # (N, T, 3)
    X_raw  = np.concatenate(raw_list)      # (N, T, 3)
    labels = np.concatenate(lbl_list)      # (N,)
    subj   = np.concatenate(subj_list)
    week   = np.concatenate(week_list)
    ARAT   = np.concatenate(arat_list)
    FMA    = np.concatenate(fma_list)

    # Split x_acc_filt into model inputs (matching load_preprocessed_h5)
    acc_patches   = X_acc[:, :, :NORM_SIZE]         # (N, pad, 32)
    pos_info      = X_acc[:, :, NORM_SIZE]           # (N, pad)
    add_embedding = X_acc[:, :, NORM_SIZE + 1:]      # (N, pad, 5)

    return (acc_patches, pos_info, add_embedding,
            X_grav, X_raw, labels, subj, week, ARAT, FMA)


def run_extraction(preprocessed_dir, checkpoint_path,
                   batch_size=32, device="cpu"):
    """End-to-end Bio-PM feature extraction for the IRB dataset."""
    print("Loading HDF5 files ...")
    (acc_patches, pos_info, add_embedding,
     X_grav, X_raw, labels, subj, week, ARAT, FMA) = load_all_h5(preprocessed_dir)

    N = len(labels)
    print(f"  Total windows : {N}")
    print(f"  acc_patches   : {acc_patches.shape}")
    print(f"  X_grav        : {X_grav.shape}")

    # Check for NaN-only windows (all-NaN tokens → gravity-only signal)
    valid_tok = ~np.isnan(acc_patches[:, :, 0])   # (N, pad_size)
    pct_any_valid = 100 * (valid_tok.any(axis=1)).mean()
    print(f"  Windows with >=1 valid ME token: {pct_any_valid:.1f}%")
    if pct_any_valid < 20:
        print("  WARNING: <20% windows have valid tokens.")
        print("  Transformer stream [0:128] will be near-zero.")
        print("  Gravity stream [128:1028] is the reliable signal.")

    print(f"\nLoading checkpoint from {checkpoint_path} ...")
    model = BioPMModel(n_classes=2)
    model.to(device, dtype=torch.float)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    result = model.encoder_acc.load_state_dict(ckpt, strict=False)
    print(f"  Missing keys: {len(result.missing_keys)}  "
          f"Unexpected: {len(result.unexpected_keys)}")
    if result.missing_keys:
        print(f"  WARNING: {len(result.missing_keys)} missing keys — "
              f"some weights not loaded. Sample: {result.missing_keys[:3]}")
    model.eval()

    # Build tensors
    acc_t   = torch.from_numpy(acc_patches).float()
    pos_t   = torch.from_numpy(pos_info).float()
    add_t   = torch.from_numpy(add_embedding).float()
    grav_t  = torch.from_numpy(X_grav).float()
    lbl_t   = torch.from_numpy(labels).float()

    dataset = TensorDataset(acc_t, pos_t, add_t, lbl_t)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_features = []
    global_idx = 0

    n_batches = len(loader)
    print(f"\nRunning encoder (device={device}, {N} windows, {n_batches} batches) ...")
    with torch.no_grad():
        for batch_acc, batch_pos, batch_add, _ in tqdm(loader, desc="Extracting", unit="batch", dynamic_ncols=True):
            bs = batch_acc.shape[0]

            batch_acc = batch_acc.to(device)
            batch_pos = batch_pos.to(device)
            batch_add = batch_add.to(device)

            # Zero mask at inference (no masking)
            mask = torch.zeros(bs, batch_acc.shape[1], device=device)

            # Transformer stream → (B, L, 64)
            tokens = model.encoder_acc(batch_acc, batch_pos, mask, batch_add)

            # Pool → (B, 128)
            pooled = masked_mean_std_valid(tokens)

            # Gravity stream
            g = grav_t[global_idx : global_idx + bs].to(device)
            g = g.transpose(1, 2)   # (B, 3, T)
            g = torch.nan_to_num(g, nan=0.0)
            if g.shape[-1] != TARGET_GRAV_LEN:
                g = F.interpolate(g, size=TARGET_GRAV_LEN,
                                  mode="linear", align_corners=False)
            g_flat = g.reshape(bs, -1)   # (B, 900)

            # Fused (B, 1028)
            fused = torch.cat([pooled, g_flat], dim=-1)
            all_features.append(fused.cpu().numpy())
            global_idx += bs

    features = np.concatenate(all_features)   # (N, 1028)

    # Report on embedding quality
    xfm_stream = features[:, :128]
    grav_stream = features[:, 128:]
    print(f"\nEmbedding quality check:")
    print(f"  Transformer [0:128]  — mean_abs={np.abs(xfm_stream).mean():.4f}  "
          f"std={xfm_stream.std():.4f}")
    print(f"  Gravity     [128:]   — mean_abs={np.abs(grav_stream).mean():.4f}  "
          f"std={grav_stream.std():.4f}")
    n_zero_rows = (np.abs(features) < 1e-6).all(axis=1).sum()
    if n_zero_rows > 0:
        print(f"  WARNING: {n_zero_rows}/{N} rows are all-zero.")
        if n_zero_rows > N * 0.5:
            print("  This confirms the zero-embedding bug.")
            print("  Use gravity-only features: features[:, 128:]")

    return features, labels, subj, week, ARAT, FMA


def main():
    os.chdir(REPO_ROOT)
    p = argparse.ArgumentParser(description="Extract Bio-PM V3 features from IRB data")
    p.add_argument("--preprocessed", required=True,
                   help="Directory with Data_MeLabel_{s}_{w}.h5 files")
    p.add_argument("--checkpoint",   required=True,
                   help="$BIOPM_ROOT/checkpoints/checkpoint.pt")
    p.add_argument("--output",       default="features/biopm_features_v3.npz")
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--device",       default="cpu", choices=["cpu", "cuda", "mps"])
    args = p.parse_args()

    print("=" * 64)
    print("Bio-PM IRB Feature Extraction V3")
    print("=" * 64)

    if not os.path.isfile(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    features, labels, subj, week, ARAT, FMA = run_extraction(
        args.preprocessed, args.checkpoint, args.batch_size, args.device
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(args.output,
             features=features, labels=labels,
             subj=subj, week=week, ARAT=ARAT, FMA=FMA)

    print()
    print("=" * 64)
    print("Bio-PM IRB Feature Extraction V3")
    print("  Pooling: valid-token-only (masked_mean_std_valid)")
    print("  Gravity: flat900 with raw acc input")
    print(f"  Saved: {args.output}")
    print(f"  features : {features.shape}  (N, 1028)")
    print("=" * 64)


if __name__ == "__main__":
    main()
