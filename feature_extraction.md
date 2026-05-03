# Feature Extraction from IRB Stroke Rehabilitation Data

## Overview

This document covers the feature extraction pipeline built for CS690R, which applies the **Bio-PM** (Bio-signal Pre-trained Model) to a wrist-worn accelerometer dataset from stroke rehabilitation subjects. The goal is to extract 1028-dimensional embeddings from 3-second windows of raw acceleration data, then use them for healthy vs. stroke classification.

---

## Dataset

### Source files

| File | Contents |
|---|---|
| `data/clinical_scores.npz` | Clinical assessments keyed by `(subject_id, week)` |
| `data/windows.npz` | ~5 GB, numpy array of window objects |

### clinical_scores.npz

```python
clin = np.load('data/clinical_scores.npz', allow_pickle=True)['clinical_scores'].item()
# Keys: (subject_id: int, week: int)
# Values: SimpleNamespace(ARAT=float, FMA=float)
```

- **ARAT:** Action Research Arm Test, range 0-57 (57 = no impairment)
- **FMA:** Fugl-Meyer Assessment, range 0-66 (66 = no impairment)
- Healthy subjects assigned max scores (ARAT=57, FMA=66) -- no clinical assessment done
- Binary label: 1 = healthy (ARAT=57 AND FMA=66), 0 = stroke

### windows.npz

```python
wnd = windows[i]
wnd.subject     # int   subject ID
wnd.week        # int   assessment week
wnd.start_idx   # int   position in 24-hour time series
wnd.sample_rate # int   30 Hz
wnd.acc         # (90, 3) float32  -- 3-axis acceleration, 3 seconds @ 30 Hz
```

- 3-second windows, no overlap, 90 samples at 30 Hz
- Dataset has a low-pass filter pre-applied
- Only `wnd.acc` is used by Bio-PM (`vel`, `pos`, `jerk` are kinematic extras not needed)

### Dataset composition (confirmed)

- ~198 subjects total
- 4 healthy subjects, ~194 stroke subjects
- Multiple assessment weeks per subject
- ~198 (subj, week) clinical entries; slightly fewer in the windows file

---

## Bio-PM Configuration

```python
CONFIG = {
    "WS": 3,                      # 3-second window
    "target_FS": 30,              # already at 30 Hz, no resampling
    "pad_size": 57,               # int(3 * 192 / 10) -- max ME tokens per window
    "HighF1": 12,                 # bandpass high cutoff (Hz)
    "LowF1": 0.5,                 # bandpass low cutoff (Hz)
    "Order1": 6,                  # filter order
    "normalize_size_target": 32,  # ME patch size
    "normalize_size_assign": 32,
}
```

---

## Pipeline

### Step 0: Inspect dataset

`inspect_dataset.py` / **Notebook Section 1**

Characterizes both `.npz` files: subject counts, weeks, ARAT/FMA ranges, windows per group, cross-reference between clinical and window data.

### Step 1: Debug pipeline

`debug_pipeline.py` / **Notebook Section 2**

Runs Bio-PM preprocessing on a single subject (default: subject 549) and traces every stage:

1. Load raw acceleration
2. Bandpass filter (0.5-12 Hz)
3. Zero-crossing movement element (ME) detection
4. ME count statistics across test windows
5. Detailed inspection of one window
6. Gravity stream check
7. `x_acc_filt` construction (what gets written to HDF5)
8. Checkpoint loading + forward pass test

**Why this matters:** Short windows (3s) combined with the pre-filtered dataset produce very few zero-crossings. Low ME count directly causes near-zero transformer stream embeddings.

### Step 2: Preprocess

`irb_preprocess.py` / **Notebook Section 3**

Groups windows by `(subject, week)`, runs Bio-PM preprocessing per window, writes one HDF5 file per group to `preprocessed/`.

**HDF5 schema per file (`Data_MeLabel_{subj}_{week}.h5`):**

| Dataset | Shape | Description |
|---|---|---|
| `window_acc_raw` | (W, 90, 3) | Raw acceleration |
| `x_acc_filt` | (W, 57, 38) | ME tokens, NaN-padded |
| `x_gravity` | (W, 90, 3) | Low-pass filtered gravity |
| `window_label` | (W,) | 1=healthy, 0=stroke |

Attributes: `subject`, `week`, `ARAT`, `FMA`, `group`.

**`x_acc_filt` column layout (38 columns):**
- `[0:32]` -- normalized ME waveform patch
- `[32]` -- position info
- `[33:38]` -- `axis, len, min, max, dirct`

Runtime: ~10-20 min for full dataset.

### Step 3: Extract embeddings

`irb_extract.py` / **Notebook Section 4**

Loads all HDF5 files, runs the Bio-PM encoder in batches, concatenates results into a single `.npz`.

**Encoder architecture:**
- Input: ME token patches `(B, 57, 32)` + position `(B, 57)` + addl features `(B, 57, 5)`
- Transformer stream: outputs tokens `(B, 57, 64)`, pooled to `(B, 128)` via mean+std pool
- Gravity stream: low-pass signal `(B, 90, 3)` interpolated to 300 samples, flattened to `(B, 900)`
- Fused output: `(B, 1028)`

Runtime: ~5-15 min on CPU.

### Step 4: Verify embeddings

`verify_embeddings.py` / **Notebook Section 5**

Sanity checks the saved `.npz`: shape, NaN/Inf count, per-stream statistics, per-subject breakdown.

### Step 5: Analysis

`irb_analyze.py` / **Notebook Section 6**

Three analyses on the extracted features:

1. **Even/odd window split** -- within each `(subj, week)` group, even-indexed windows go to train, odd to test. Saved to `results/splits/split_indices.npz`.

2. **UMAP visualization** -- visit-level (mean-pooled across windows) 1028-d embeddings reduced to 2D, colored healthy vs stroke. Saved to `results/figures/umap_healthy_vs_impaired.png`.

3. **LOSO logistic regression** -- leave-one-subject-out LR (`class_weight='balanced'`, `C=1.0`, `lbfgs`). Reports per-subject and overall AUC. Saved to `results/lr_loso_results.txt`.

---

## Key Finding: Zero-Embedding Bug

**Problem:** All-zero embeddings in dimensions `[0:128]` (transformer stream).

**Root cause:** 3-second windows produce very few zero-crossings after the 0.5-12 Hz bandpass filter is applied to the already-smoothed dataset acceleration signal. With fewer than 3 MEs per window on average, almost all token slots in `x_acc_filt` are NaN-padded. The Bio-PM transformer operates on padded tokens, producing identical outputs that collapse under mean+std pooling:
- Mean pool `[0:64]`: identical token embeddings average to the same value (not zero, but uninformative)
- Std pool `[64:128]`: identical tokens have zero variance, producing all-zeros

**Solution:** Use only the gravity stream (`features[:, 128:]`, shape `(N, 900)`) for downstream classification. The gravity stream does not depend on ME detection and is always non-zero and informative.

**Detection:** Run `debug_pipeline.py` (or Notebook Section 2) on any subject. If "Mean MEs per window" < 3, expect near-zero `[0:128]`.

---

## Embedding Layout

```
features[i]  -- 1028-dimensional vector for window i
  [0:64]     mean pool of transformer tokens   (movement element structure)
  [64:128]   std pool of transformer tokens    (movement element variability)
              NEAR-ZERO for this dataset due to short windows + pre-filtered signal
  [128:1028] gravity flattened (300 x 3)       (posture / orientation)
              ALWAYS MEANINGFUL -- use this for downstream tasks
```

---

## Output Files

| File | Description |
|---|---|
| `preprocessed/Data_MeLabel_{s}_{w}.h5` | One per (subject, week), Bio-PM inputs |
| `features/biopm_features.npz` | Final 1028-d embeddings + metadata |
| `results/splits/split_indices.npz` | Even/odd window split masks |
| `results/figures/umap_healthy_vs_impaired.png` | UMAP visualization |
| `results/lr_loso_results.txt` | LOSO LR report with AUC |

---

## Scripts

| Script | Purpose |
|---|---|
| `inspect_dataset.py` | Characterize raw `.npz` files |
| `debug_pipeline.py` | Diagnose ME detection and embedding quality on one subject |
| `irb_preprocess.py` | Full preprocessing: `.npz` to HDF5 |
| `irb_extract.py` | Bio-PM feature extraction: HDF5 to `.npz` |
| `verify_embeddings.py` | Sanity check extracted features |
| `irb_analyze.py` | Even/odd split, UMAP, LOSO LR |
| `biopm_irb_pipeline.ipynb` | Complete notebook combining all steps |
| `extraction pipeline/run_pipeline.sh` | End-to-end runner for preprocess -> extract -> verify with repo-root-safe paths |

---

## Environment Setup

**Conda:**
```bash
conda env create -f environment.yml
conda activate biopm
python -m ipykernel install --user --name biopm --display-name "biopm"
```

**pip/venv:**
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**BIOPM path configuration (repo-relative):**
```bash
export BIOPM_ROOT=CS690TR
```

All scripts check for `BIOPM_ROOT` at startup.

- `extraction pipeline/run_pipeline.sh` defaults `BIOPM_ROOT` to `CS690TR` if not set.
- The notebook now uses repo-relative configuration by default (`BIOPM_ROOT = ./CS690TR`).

---

## End-to-end Run (Script)

From repo root:

```bash
bash "extraction pipeline/run_pipeline.sh"
```

The script executes:

1. `irb_preprocess.py` with `--data_dir ./data` and output `./preprocessed`
2. `irb_extract.py` with checkpoint `$BIOPM_ROOT/checkpoints/checkpoint.pt` and output `./features/biopm_features.npz`
3. `verify_embeddings.py` on `./features/biopm_features.npz`

This matches the notebook file locations (`PREPROCESSED_DIR="preprocessed"`, `FEATURES_PATH="features/biopm_features.npz"`).

All extraction pipeline scripts are configured so default outputs resolve under the project root tree:
- features: `./features/`
- preprocessing HDF5: `./preprocessed/`
- analysis outputs: `./results/`

---

## Class Imbalance

The dataset has a severe imbalance: 4 healthy vs ~194 stroke visit-level labels. This affects classification:
- LOSO LR uses `class_weight='balanced'` to compensate
- UMAP visualization shows 4 healthy points vs ~194 stroke points -- clustering is the signal
- Window-level imbalance is even more extreme given multiple windows per visit
