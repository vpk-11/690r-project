# BIOPM IRB Pipeline Guide
## CS690R - Stroke Dataset Feature Extraction and Analysis

## Purpose
This document describes the current production pipeline for IRB stroke accelerometer data using the BIOPM model. It explains what was implemented, why each step exists, what files are produced, and how to run and use the outputs.

This guide documents the **new BIOPM-based workflow**. The older handcrafted kinematic extraction workflow is intentionally not used for feature generation.

## High-Level Design
The pipeline is split into two layers:

1. Extraction layer (model-facing):
- Converts `windows.npz` + `clinical_scores.npz` into BIOPM embeddings.
- Produces `features/biopm_features.npz` with window-level 1028-d vectors.

2. Analysis/export layer (user-facing):
- Performs visit-level aggregation and baseline analytics.
- Produces optional legacy-compatible schema for downstream tools.
- Generates UMAP plots and LOSO metrics.

This separation keeps feature extraction stable and reproducible while allowing flexible analysis formats.

## Repository Components
- `extraction pipeline/run_pipeline.sh`: End-to-end extractor runner.
- `extraction pipeline/irb_preprocess.py`: Converts raw windows to per-visit HDF5 BIOPM inputs.
- `extraction pipeline/irb_extract.py`: Runs BIOPM encoder and writes 1028-d embeddings.
- `extraction pipeline/verify_embeddings.py`: Sanity checks extracted features.
- `extraction pipeline/irb_analyze.py`: Baseline split + UMAP + LOSO AUC analysis.
- `biopm_irb_pipeline.ipynb`: Interactive notebook containing dataset inspection, debugging, analysis, and legacy-schema export.

## Data Inputs
Required files:
- `data/windows.npz`
- `data/clinical_scores.npz`

### `clinical_scores.npz`
- Dict keyed by `(subject_id, week)`
- Value contains `ARAT` and `FMA`
- Label rule in current BIOPM pipeline:
  - `label = 1` for healthy (`ARAT=57` and `FMA=66`)
  - `label = 0` for stroke

### `windows.npz`
Each element is a window object with fields including:
- `subject`
- `week`
- `start_idx`
- `sample_rate` (30 Hz)
- `acc` with shape `(90, 3)` representing 3 seconds of tri-axial acceleration

## BIOPM Feature Definition
Each output vector is 1028-dimensional:
- `[0:128]` transformer stream (movement-element token encoder, pooled)
- `[128:1028]` gravity stream (low-pass orientation/posture stream)

In this dataset, low movement-element density can reduce transformer-stream informativeness. The gravity stream remains valid and is preserved in full output.

## Detailed Pipeline

### Step 1: Preprocess raw windows to HDF5
Script: `extraction pipeline/irb_preprocess.py`

What it does:
- Groups windows by `(subject, week)`.
- Applies BIOPM preprocessing per window:
  - Bandpass filter for movement-element detection
  - Low-pass filter for gravity stream
  - Zero-crossing movement-element extraction
- Writes one HDF5 file per `(subject, week)` as:
  - `preprocessed/Data_MeLabel_{subject}_{week}.h5`

Per-HDF5 datasets:
- `window_acc_raw`: `(W, 90, 3)`
- `x_acc_filt`: `(W, 57, 38)` (NaN-padded movement-element tokens)
- `x_gravity`: `(W, 90, 3)`
- `window_label`: `(W,)`

Per-HDF5 attributes:
- `subject`, `week`, `ARAT`, `FMA`, `group`

### Step 2: Extract BIOPM embeddings
Script: `extraction pipeline/irb_extract.py`

What it does:
- Loads all `Data_MeLabel_*.h5` files.
- Loads `CS690TR/checkpoints/checkpoint.pt` into `BioPMModel` encoder.
- Runs encoder in batches.
- Pools transformer tokens and concatenates gravity stream.
- Writes window-level features to `features/biopm_features.npz`.

Primary output schema (`features/biopm_features.npz`):
- `features`: `(N, 1028)`
- `labels`: `(N,)` (`1=healthy`, `0=stroke`)
- `subj`: `(N,)`
- `week`: `(N,)`
- `ARAT`: `(N,)`
- `FMA`: `(N,)`

### Step 3: Verify extraction output
Script: `extraction pipeline/verify_embeddings.py`

Checks include:
- Feature shape
- NaN/Inf counts
- Transformer/gravity stream magnitudes
- Per-subject counts and label distribution

### Step 4: Baseline analysis
Script: `extraction pipeline/irb_analyze.py`

What it does:
- Even/odd window split per `(subject, week)`:
  - saves `results/splits/split_indices.npz`
- Visit-level mean aggregation
- UMAP figure:
  - `results/figures/umap_healthy_vs_impaired.png`
- LOSO logistic regression AUC report:
  - `results/lr_loso_results.txt`

## New Notebook Export Layer (Legacy-Compatible Schema)
Implemented in `biopm_irb_pipeline.ipynb` as an additional final cell.

Goal:
- Keep extraction pipeline unchanged.
- Build a second NPZ compatible with old downstream expectations.

Created file:
- `features/biopm_features_legacy_schema.npz`

Legacy-compatible keys (visit-level, 1028-d):
- `features`
- `features_even`
- `features_odd`
- `feature_names`
- `labels`
- `pids`
- `arat`
- `fma`
- `subjects`
- `weeks`

Notes:
- `features`, `features_even`, `features_odd` are visit-level mean-pooled BIOPM vectors, each with 1028 columns.
- Labels remain aligned to current pipeline convention (`1=healthy`, `0=stroke`).
- `feature_names` are synthetic names: `biopm_0000 ... biopm_1027`.

## New Notebook Analytics Added
The final notebook cell also adds:

1. Three-panel UMAP on legacy-schema visit-level features:
- Panel 1: Healthy vs Stroke
- Panel 2: ARAT gradient
- Panel 3: Subject coloring
- Output: `results/figures/umap_feature_space_3panel_biopm.png`

2. LOSO Logistic Regression with both AUC and Macro-F1:
- Output report: `results/lr_loso_results_with_f1.txt`

## How to Run

### Environment
Use one of:
- `environment.yml`
- `requirements.txt`

Set BIOPM path:
```bash
export BIOPM_ROOT=CS690TR
```

### End-to-end extraction
From repo root:
```bash
bash "extraction pipeline/run_pipeline.sh"
```

This executes preprocess -> extract -> verify.

### Notebook analysis and legacy export
Open and run:
- `biopm_irb_pipeline.ipynb`

Run all cells, including the final legacy-export/UMAP/F1 cell.

## How to Use the Results

### If you need canonical extractor output
Use:
- `features/biopm_features.npz`

Best for:
- Reproducible model-facing processing
- Script-based analysis (`irb_analyze.py`)

### If you need old-schema compatibility
Use:
- `features/biopm_features_legacy_schema.npz`

Best for:
- Downstream tools expecting `features_even`, `features_odd`, `pids`, `feature_names`, etc.
- Direct parity with previous analysis interfaces while keeping BIOPM features

### Figures and metrics
- Binary UMAP: `results/figures/umap_healthy_vs_impaired.png`
- 3-panel UMAP: `results/figures/umap_feature_space_3panel_biopm.png`
- LOSO AUC report: `results/lr_loso_results.txt`
- LOSO AUC + Macro-F1 report: `results/lr_loso_results_with_f1.txt`

## Reproducibility and Conventions
- Pathing is repo-relative throughout scripts/notebook.
- Extraction pipeline is deterministic with fixed data/checkpoint inputs.
- Label convention in this pipeline is fixed as:
  - `1 = healthy`, `0 = stroke`
- Legacy-schema export is a transformation layer; it does not alter extraction behavior.

## Known Constraints
- Class imbalance is severe (few healthy visits vs many stroke visits).
- AUC/F1 should be interpreted as baseline sanity metrics, not final clinical validation.
- Transformer-stream signal quality depends on movement-element availability; gravity stream remains part of fused representation in all cases.

## Recommended Workflow for Collaborators
1. Run `run_pipeline.sh` to regenerate canonical BIOPM features.
2. Run notebook to generate legacy-schema file and analysis artifacts.
3. Use legacy-schema NPZ for compatibility-oriented downstream work.
4. Use canonical NPZ for low-level debugging and extractor-level investigations.

