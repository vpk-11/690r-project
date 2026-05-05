# CLAUDE.md — Bio-PM IRB Pipeline Full Autopsy

You are Claude Code running in the `690r-project` directory on an Apple Silicon Mac.
Your job is a complete forensic analysis of the Bio-PM feature extraction pipeline.

Read every instruction in this file before touching anything.
Then execute systematically, section by section.
At the end, write `AUTOPSY_REPORT.md` with every finding.

---

## Your Environment

```
Device:     Apple Silicon Mac — always use MPS
Conda env:  biopm-690r  (already activated before you run)
BIOPM_ROOT: CS690TR     (relative to project root)
```

**First thing you do:**
```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```
If MPS is False, note it and use CPU. All inference must go through MPS when available.

---

## Actual Repository Structure

```
690r-project/                          <- you are here
├── CLAUDE.md                          <- this file
├── CS690TR/                           <- Bio-PM model source (BIOPM_ROOT)
│   ├── checkpoints/checkpoint.pt     <- pretrained weights, 5.6 MB
│   └── src/
│       ├── models/biopm.py           <- READ THIS FIRST
│       └── data/preprocessing.py    <- READ THIS SECOND
├── data/
│   ├── windows.npz                   <- 587,046 window objects
│   └── clinical_scores.npz          <- (subject_id, week) -> ARAT, FMA scores
├── extraction pipeline/              <- NOTE: space in folder name
│   ├── irb_preprocess.py            <- standard: pad_size=57
│   ├── irb_preprocess_alt.py        <- alt: pad_size=171, 9s grouped
│   ├── irb_extract.py
│   ├── irb_analyze.py
│   ├── debug_pipeline.py
│   ├── inspect_dataset.py
│   ├── verify_embeddings.py
│   └── export_legacy_schema.py
├── preprocessed/                     <- HDF5 files from standard pipeline
├── preprocessed_alt/                 <- HDF5 files from alt pipeline
├── features/
│   ├── biopm_features.npz           <- standard pipeline output
│   └── biopm_features_alt.npz      <- alt pipeline output
├── biopm_irb_pipeline.ipynb
└── biopm_irb_pipeline_alt.ipynb
```

---

## Step 0: Read the Model Source Code

Before writing a single line of analysis, read these two files completely:

```bash
cat CS690TR/src/models/biopm.py
cat CS690TR/src/data/preprocessing.py
```

From `biopm.py`, answer these questions in your report:
- What is `BioPMModel`? What are its components?
- What is `masked_mean_std`? What exactly does it do with NaN tokens?
- What is `load_pretrained_encoder`? What keys does it load vs skip?
- Where exactly is the gravity CNN defined? What layers does it have?
- Where exactly is the acc transformer defined? What is its input shape?
- Does the model ever call `lowpass_filter` or `bandpass_filter` internally, or are
  those called externally before the model?

From `preprocessing.py`, answer:
- What does `detect_zero_crossings` return? What is each return value?
- What does `assign_zero_crossings` do? What does it add to the ME data?
- What does `bandpass_filter` do to the signal mathematically?
- What does `lowpass_filter` do? What frequency does it preserve?
- What config keys does `detect_zero_crossings` use from CONFIG?

---

## Step 1: Read the Existing Pipeline Scripts

Read every script in `extraction pipeline/`. For each one, in your report write:
- What it does in one sentence
- What its inputs and outputs are
- Any hardcoded values that look suspicious (especially CONFIG dict values)

Pay special attention to:
1. The CONFIG dict in `irb_preprocess.py` — note every key and value
2. The CONFIG dict in `irb_preprocess_alt.py` — what changed vs standard
3. How `x_acc_filt` is constructed — what 38 columns does it have?
4. How `x_gravity` is constructed — is it filtered? what filter?
5. How `irb_extract.py` splits `x_acc_filt` into patches/pos/add inputs

---

## Step 2: Load Data and Pick 4 Test Subjects

```python
import sys, os, numpy as np
sys.path.insert(0, 'CS690TR')

clin    = np.load('data/clinical_scores.npz', allow_pickle=True)['clinical_scores'].item()
windows = np.load('data/windows.npz', allow_pickle=True)['windows']
```

Find and use these 4 subjects:
- 2 stroke subjects with many windows (pick subjects with >3000 windows)
- 1 stroke subject with few windows (pick subject with <500 windows)
- 1 healthy subject (ARAT==57 AND FMA==66)

For each subject print: subject_id, group, window count, ARAT, FMA, sample_rate, window shape.

Take 30 windows from each subject for all subsequent tests.

---

## Step 3: The Gravity Stream — Full Dissection

The gravity stream output (`dims [128:1028]` of the final embedding) is supposed to
capture wrist orientation relative to Earth's gravity. In the alt pipeline output, it
showed `mean_abs = 0.0028`. This is essentially zero. Investigate why.

### 3a. Check if gravity exists in the raw data

For each test subject, compute:
```python
# Per-window mean acceleration magnitude
window_means = np.stack([w.acc.mean(axis=0) for w in test_windows])  # (30, 3)
magnitudes   = np.linalg.norm(window_means, axis=1)                  # (30,)
print(f"Mean gravity magnitude: {magnitudes.mean():.4f}g (expected ~1.0g)")
```

If `magnitudes.mean() < 0.1g`, gravity has been removed from the data before delivery.
Document this finding. It explains why the gravity stream is dead.

### 3b. Test the lowpass filter behavior

For each test subject, on 10 windows:
```python
from src.data.preprocessing import lowpass_filter

# Standard: 0.5 Hz cutoff
for cutoff in [0.1, 0.5, 1.0, 2.0, 5.0]:
    grav = lowpass_filter(w.acc.astype(np.float64), cutoff, 30, order=6)
    print(f"Lowpass {cutoff}Hz: mean_abs={np.abs(grav).mean():.5f}  max_abs={np.abs(grav).max():.5f}")

# No filter (raw acc)
print(f"Raw (no filter): mean_abs={np.abs(w.acc).mean():.5f}")
```

Document which cutoffs produce non-zero output. If all are near-zero,
the data has gravity removed. If higher cutoffs produce signal, the
0.5 Hz cutoff is too aggressive for these 3-second windows.

### 3c. Test alternative gravity inputs for the CNN

The gravity CNN takes a (T, 3) signal, interpolates to 300 samples, and flattens to 900.
Test what happens when you feed it:
- Standard: `lowpass_filter(acc, 0.5, 30, order=6)`
- Raw: `acc` unchanged
- Window mean: `np.tile(acc.mean(axis=0), (len(acc), 1))`  — constant gravity vector
- Demeaned: `acc - acc.mean(axis=0)` then `lowpass_filter`

For each, run it through the gravity CNN path and report the `mean_abs` of the 900-d output.

```python
import torch, torch.nn.functional as F

def run_gravity_cnn(model, acc_signal):
    g = torch.from_numpy(acc_signal.astype(np.float32)).unsqueeze(0)  # (1, T, 3)
    g = torch.nan_to_num(g.permute(0, 2, 1))                          # (1, 3, T)
    g = F.interpolate(g, size=300, mode='linear', align_corners=False) # (1, 3, 300)
    g_flat = g.reshape(1, -1)                                          # (1, 900)
    return g_flat.squeeze(0).cpu().numpy()
```

Does the gravity CNN run its own learned weights on the signal, or is it
a pure reshape? Check this in `biopm.py`.

---

## Step 4: The Acc Transformer Stream — Full Dissection

### 4a. Understand the CONFIG keys used by detect_zero_crossings

Run `detect_zero_crossings` with the standard config and print EVERY return value:
```python
from src.data.preprocessing import bandpass_filter, detect_zero_crossings

CONFIG = {"HighF1": 12, "LowF1": 0.5, "Order1": 6, "target_FS": 30,
          "pad_size": 57, "normalize_size_target": 32, "normalize_size_assign": 32}

w = test_windows[0]
t = np.arange(90) / 30
acc_filt = bandpass_filter(w.acc.astype(np.float64), 0.5, 12, 30, order=6)

result = detect_zero_crossings(acc_filt, t, CONFIG)
# Unpack all return values and describe each one
```

Print the name, shape, dtype, and value range of every return value.
Explain what `me_norm`, `me_info`, `pos_info` each represent physically.
What does each column of `me_info` mean?

### 4b. Understand the x_acc_filt packing

From `irb_preprocess.py`, the 38-column `x_acc_filt` is packed as:
```
[0:32]   me_norm         -- normalized ME waveform patch (32 samples)
[32]     pos_info        -- position of ME relative to gravity direction
[33]     me_info.axis    -- which axis (0=X, 1=Y, 2=Z) had the zero crossing
[34]     me_info.len     -- length of the ME in samples
[35]     me_info.min     -- minimum value within the ME
[36]     me_info.max     -- maximum value within the ME
[37]     me_info.dirct   -- direction of crossing (+1 or -1)
```

Verify this by reading the actual packing code. Does it match? Any differences?

Then answer: what does `assign_zero_crossings` add to this picture?
Read its implementation and explain what it does to `pos_info`.

### 4c. Test pad_size effects

For each test subject, run ME detection with standard bandpass filter and measure:

```python
for pad_size in [57, 96, 171, 192]:
    me_counts = []
    for wnd in test_windows:
        cfg = {**CONFIG, "pad_size": pad_size}
        _, _, me_list, *_ = detect_zero_crossings(acc_filt, t, cfg)
        me_counts.append(len(me_list))
    avg_me = np.mean(me_counts)
    fill   = 100 * avg_me / pad_size
    print(f"pad_size={pad_size}: avg_MEs={avg_me:.1f}  fill={fill:.1f}%  "
          f"zero_windows={100*(np.array(me_counts)==0).mean():.1f}%")
```

**Critical question:** Does `pad_size` affect how many MEs are detected, or only
how many slots are allocated for them? Verify by checking `detect_zero_crossings`
source — does it use `pad_size` during detection or only for output formatting?

### 4d. Test all filter permutations for acc stream

For each of these filter configurations, measure avg MEs per window across all 4 subjects:

1. Standard bandpass 0.5-12 Hz, order 6
2. Wider bandpass 0.1-20 Hz, order 4
3. Narrow PD-band 3-8 Hz, order 4
4. Highpass only >0.5 Hz, order 4
5. No filter — raw acc fed directly to detect_zero_crossings
6. Bandpass 0.5-12 Hz then divide by signal RMS (normalize amplitude)

For each config print:
- Average MEs per window (across all test subjects)
- Percentage of windows with 0 MEs
- Fill rate against pad_size=57

Report: which filter gives the most MEs? Which gives the fewest 0-ME windows?

### 4e. Run the full transformer and measure output quality

For each filter config and each pad_size config, run the full acc transformer:

```python
import torch
from src.models.biopm import BioPMModel, masked_mean_std

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
ckpt   = torch.load('CS690TR/checkpoints/checkpoint.pt', map_location='cpu', weights_only=False)
model  = BioPMModel(n_classes=2)
model.encoder_acc.load_state_dict(ckpt, strict=False)
model  = model.to(DEVICE).eval()

# For each window, build x_acc (pad_size, 38), run encoder_acc, run masked_mean_std
# Report: mean_abs of [0:64] (mean pool) and [64:128] (std pool)
```

The key metric is `std_pool mean_abs` (dims [64:128]). Higher = more discriminative.
If std_pool is near-zero, all ME tokens are identical (no variance) which means
the transformer isn't learning anything useful from the signal.

---

## Step 5: Why pad_size=192 Fails (and Why the TA Said Movements Should Be Detected)

The TA is correct that movements should be detected. The issue is NOT the presence
of MEs — ME detection gives ~18 MEs per 3s window regardless of pad_size.

The issue is the **ratio** of real tokens to pad tokens in the transformer:

```
pad_size=57:  18 real / 57 total = 32.5% fill
pad_size=192: 18 real / 192 total = 9.4% fill
```

Run this experiment to show the effect:

For 20 windows on one test subject, run the transformer with pad_size=57 vs 192.
Compare `mean_abs` of mean_pool and std_pool. Show that 192 gives lower std_pool.

Then answer definitively: does `masked_mean_std` correctly ignore NaN-padded tokens
when computing mean and std? Check its implementation in `biopm.py`.

If it DOES correctly mask NaN tokens, then pad_size=192 should produce identical
results to pad_size=57 (same 18 real tokens, same pool). If it does NOT correctly
mask, padding tokens contaminate the pool.

This is the most important question in the entire autopsy. Read `masked_mean_std`
in full and answer it.

---

## Step 6: The 100% Fill Rate Fix

The alt pipeline (pad_size=171) did NOT improve fill rate because it scaled pad_size
proportionally with window length. Fill rate stayed at ~33%.

The correct approach: **group 3 windows (9s), keep pad_size=57**.

Test this on each test subject:
```python
# Group 3 consecutive windows
for i in range(0, len(test_windows)-2, 3):
    block = np.concatenate([test_windows[j].acc for j in range(i, i+3)], axis=0)  # (270, 3)
    t     = np.arange(270) / 30
    acc_f = bandpass_filter(block.astype(np.float64), 0.5, 12, 30, order=6)
    cfg   = {**CONFIG, "pad_size": 57}  # KEEP 57, not 171
    _, _, me_list, *_ = detect_zero_crossings(acc_f, t, cfg)
    # Expected: ~58 MEs detected, truncated to 57 -> ~100% fill
```

Report: what is the actual fill rate? What is the std_pool mean_abs compared to 3s windows?

---

## Step 7: Separate Stream Configurations

Test whether using different configurations for each stream independently improves
embedding quality:

| Experiment | Acc stream | Gravity stream |
|---|---|---|
| A (standard) | bandpass 0.5-12 Hz | lowpass 0.5 Hz |
| B | bandpass 0.5-12 Hz | raw acc (no filter) |
| C | bandpass 0.5-12 Hz | window mean as constant gravity |
| D | no filter (raw) | raw acc |
| E | no filter (raw) | lowpass 0.5 Hz |
| F | bandpass 0.5-12 Hz | highpass removed (raw - lowpass) |

For each experiment, run on all 4 subjects, 20 windows each.
Report: mean_pool_abs, std_pool_abs, grav_stream_abs, and total embedding norm.

The goal is to find which combination maximizes std_pool_abs (transformer signal quality)
and grav_stream_abs (gravity signal quality) simultaneously.

---

## Step 8: Check Existing HDF5 Files

If `preprocessed/` exists and has HDF5 files, inspect them:

```python
import h5py, os

h5_files = [f for f in os.listdir('preprocessed') if f.endswith('.h5')]
print(f"HDF5 files: {len(h5_files)}")

with h5py.File(f'preprocessed/{h5_files[0]}', 'r') as hf:
    print("Datasets:", list(hf.keys()))
    for key in hf.keys():
        print(f"  {key}: {hf[key].shape}  dtype={hf[key].dtype}")
    print("Attributes:", dict(hf.attrs))
    
    # Check actual fill rate in the stored data
    x_acc = np.array(hf['x_acc_filt'])   # (W, pad_size, 38)
    valid = ~np.isnan(x_acc[:, :, 0])     # (W, pad_size)
    fill  = 100 * valid.mean()
    print(f"Actual fill rate in HDF5: {fill:.2f}%")
    print(f"x_gravity range: {np.array(hf['x_gravity']).min():.5f} to {np.array(hf['x_gravity']).max():.5f}")
```

Do the same for `preprocessed_alt/`. Compare fill rates between standard and alt.
Check that `x_gravity` values in the HDF5 are non-zero.

---

## Step 9: Check Existing Embedding Files

If `features/biopm_features.npz` and `features/biopm_features_alt.npz` exist:

```python
for path in ['features/biopm_features.npz', 'features/biopm_features_alt.npz']:
    if not os.path.exists(path): continue
    d    = np.load(path, allow_pickle=True)
    X    = d['features']
    mean_pool = X[:, :64]
    std_pool  = X[:, 64:128]
    gravity   = X[:, 128:]
    
    print(f"\n{path}")
    print(f"  shape:         {X.shape}")
    print(f"  mean_pool abs: {np.abs(mean_pool).mean():.5f}")
    print(f"  std_pool  abs: {np.abs(std_pool).mean():.5f}")
    print(f"  gravity   abs: {np.abs(gravity).mean():.5f}")
    print(f"  zero rows:     {(np.abs(X) < 1e-5).all(axis=1).sum()}")
    
    # Check if gravity is dead across all subjects
    per_subject_grav = []
    for sid in np.unique(d['subj']):
        mask = d['subj'] == sid
        per_subject_grav.append(np.abs(X[mask, 128:]).mean())
    print(f"  grav per subject: min={min(per_subject_grav):.5f}  max={max(per_subject_grav):.5f}")
```

---

## Step 10: Write AUTOPSY_REPORT.md

After running all experiments above, write a complete `AUTOPSY_REPORT.md`.

The report must answer every question listed here. Structure it exactly as follows:

```markdown
# Bio-PM IRB Pipeline: Full Autopsy Report

## 0. TL;DR (Read This First)
3-5 bullet points: the most important findings and the single best fix to apply.

## 1. Repository and File Map
Exact structure of CS690TR/src/models/biopm.py and preprocessing.py.
What each function does, what it takes, what it returns.

## 2. Bio-PM Architecture: How It Actually Works
Based on reading the source code (not assumptions):
- BioPMModel class structure
- encoder_acc: input shape, transformer layers, output shape
- masked_mean_std: exactly how it handles NaN tokens
- gravity CNN: is it a learned network or a reshape? what layers?
- checkpoint.pt: what keys are loaded, what are skipped and why

## 3. The Two Streams: Signal Flow

### 3a. Acc Transformer Stream [dims 0-127]
Complete signal flow from raw wnd.acc to 128-d pooled output.
What bandpass_filter does to the signal (frequency content removed/kept).
What detect_zero_crossings does (zero crossing algorithm).
What assign_zero_crossings adds (gravity context).
What the 38 columns of x_acc_filt represent.
What the transformer does with these tokens.
What masked_mean_std does with the valid vs NaN tokens.

### 3b. Gravity CNN Stream [dims 128-1027]
Complete signal flow from raw wnd.acc to 900-d output.
What lowpass_filter does (frequency content kept).
Whether the CNN has learned weights or is a pure reshape.
Why the output is 900-d (300 samples x 3 axes).

## 4. The Gravity Stream Problem

### Is gravity present in the data?
Results from Section 3a test — actual DC magnitude per subject.

### Lowpass filter behavior
Results from Section 3b test — output at each cutoff frequency.

### Root cause and fix
Exact diagnosis. Exact code change needed.

## 5. The pad_size Problem

### Does pad_size affect ME detection?
Yes/no, with code evidence from detect_zero_crossings source.

### Fill rates by pad_size
Table: pad_size | avg_MEs | fill_rate | std_pool_abs

### Why 192 was used (and whether it was actually wrong)
Was masked_mean_std masking NaN tokens correctly?
If yes: pad_size=192 is suboptimal but not broken.
If no: pad_size=192 causes contaminated pooling = real failure.

### Why fill rate didn't improve from 57 to 171
Explanation with numbers.

## 6. Filter Permutation Results

### Acc stream filters
Table: filter_config | avg_MEs | fill_57 | zero_windows_pct | std_pool_abs

### Gravity stream inputs
Table: gravity_input | grav_stream_abs

### Best combination
Which acc filter + gravity input + pad_size gives highest std_pool_abs + grav_stream_abs?

## 7. The 100% Fill Rate Fix
Results from grouping 3 windows with pad_size=57.
Actual fill rate achieved.
Actual std_pool_abs improvement vs standard.
Whether this fix is worth implementing.

## 8. Existing File Quality Check
HDF5 fill rates (standard vs alt).
Embedding file quality (standard vs alt).

## 9. What the TA Means by "Movements Should Be Detected"
The TA is correct that ~18 MEs per 3s window is healthy.
The problem is X, not ME detection itself.
[Fill this in based on what you find.]

## 10. Recommended Configuration
The single best configuration based on all experiments.
Exact CONFIG dict values.
Exact filter functions and parameters.
Expected fill rate.
Expected embedding quality.

## 11. Code Changes Required
For each finding that requires a code fix:
- File to modify (or new file to create)
- Exact change needed
- Why this change fixes the problem
```

---

## Ground Rules

- Do not modify any existing file. Create new files if needed (e.g., `autopsy_results/`).
- Run every experiment on all 4 test subjects, not just one.
- When something fails or throws an error, document the error and what it means.
- All numbers in the report must come from actual runs, not assumptions.
- If MPS gives different results than CPU for the same inputs, document it.
- Be precise: `mean_abs = 0.0028` is a different finding than `mean_abs = 0.28`.
- The report is for a human who needs to understand this system completely.
  Write as if the reader has never seen Bio-PM before and will implement the fix themselves.
