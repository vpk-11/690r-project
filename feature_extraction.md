# Feature Extraction Guide (IRB + Bio-PM)

Last updated: May 5, 2026

This is the practical, reproducible guide for running feature extraction in this repository.

## 1. What You Need

## Required repo layout
- `./CS690TR/`
- `./CS690TR/checkpoints/checkpoint.pt`
- `./data/clinical_scores.npz`
- `./data/windows.npz`

## Required scripts
- `run_extraction_pipeline_v3.sh`
- `extraction pipeline/irb_preprocess_v3.py`
- `extraction pipeline/irb_extract_v3.py`
- `extraction pipeline/export_legacy_schema_v3.py`

## 2. Environment Setup

Run:

```bash
bash setup.sh
```

What `setup.sh` now does:
1. Ensures `conda` exists.
2. Creates/updates `biopm-690r` env from `environment.yml`.
3. Registers `biopm-690r (CS690R)` kernel.
4. Verifies repo structure and required scripts/notebooks.
5. Verifies `CS690TR/checkpoints/checkpoint.pt` exists.
6. Verifies `data/clinical_scores.npz` and `data/windows.npz` exist.
7. Runs quick Python sanity checks (imports + NPZ keys).
8. Ensures output folders exist.

## 3. Run the Current Pipeline (V3)

```bash
export BIOPM_ROOT=CS690TR
bash run_extraction_pipeline_v3.sh
```

This executes:
1. Preprocess grouped blocks (`irb_preprocess_v3.py`) -> `preprocessed_v3/`
2. Extract features (`irb_extract_v3.py`) -> `features/biopm_features_v3.npz`
3. Verify embeddings (`verify_embeddings.py`)
4. Export legacy schema (`export_legacy_schema_v3.py`) -> `features/biopm_features_legacy_schema_v3.npz`

## 4. Why V3 Is the Recommended Path

V3 applies project-side fixes validated on this dataset:
1. 9-second grouped windows (3x3s blocks).
2. Fixed `pad_size=57` (not scaled with grouped length).
3. Raw-acc gravity stream in preprocessing.
4. Valid-token-only pooling in `irb_extract_v3.py` (`masked_mean_std_valid`).

This resolves the low fill-rate bottleneck seen in earlier variants.

## 5. Expected Outputs

## Preprocessed
- `preprocessed_v3/Data_MeLabel_{subject}_{week}.h5`
- Key datasets:
  - `window_acc_raw`
  - `x_acc_filt`
  - `x_gravity`
  - `window_label`

## Feature file
- `features/biopm_features_v3.npz`
- Keys:
  - `features` (N, 1028)
  - `labels`
  - `subj`
  - `week`
  - `ARAT`
  - `FMA`

## Legacy schema
- `features/biopm_features_legacy_schema_v3.npz`
- Keys include:
  - `features`, `features_even`, `features_odd`
  - `labels`, `pids`, `arat`, `fma`, `subjects`, `weeks`

## 6. Current Verified Metrics (from local run)

- V3 windows processed: 195,610
- V3 files written: 198
- V3 fill rate (global HDF5 audit): **94.0661%**
- V3 average MEs per 9s block: **53.6177**
- V3 feature shape: **(195610, 1028)**
- V3 healthy window fraction: **4.8372%**

Cross-pipeline fill-rate reference:
- Standard: 31.6604%
- Alt: 34.6734%
- V3: 94.0661%

## 7. How to Verify After Running

## Quick CLI check
```bash
python "extraction pipeline/verify_embeddings.py" --features features/biopm_features_v3.npz
```

## Notebook check
Open:
- `biopm_irb_pipeline_v3.ipynb`

This notebook was rebuilt to be data-driven only:
- no hardcoded claims
- all summary numbers computed from local files
- explicit consistency checks (HDF5 windows vs feature rows)

## 8. If Something Fails

## `BIOPM_ROOT` issues
- Set explicitly:

```bash
export BIOPM_ROOT=CS690TR
```

## Missing checkpoint
- Ensure:
- `CS690TR/checkpoints/checkpoint.pt`

## Missing data files
- Ensure both exist:
- `data/clinical_scores.npz`
- `data/windows.npz`

## Conda/kernel issues
- Re-run setup:

```bash
bash setup.sh
```

## 9. Historical Pipelines (for comparison)

- Standard:

```bash
bash run_extraction_pipeline.sh
```

- Alt:

```bash
bash run_extraction_pipeline_alt.sh
```

Use only for ablation/comparison against V3 outputs.

## 10. Notes for Contributors

1. Keep `CS690TR/` treated as dependency (read-only for project changes).
2. Implement new behavior in project-side scripts under `extraction pipeline/`.
3. Version outputs and scripts explicitly (`*_v4`, etc.) to preserve comparability.
4. Update this file and `PIPELINE_EXPLAINER.md` whenever metrics or pipeline behavior changes.
