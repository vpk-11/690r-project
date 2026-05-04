# 690r-project

BIOPM-based feature extraction and analysis pipeline for the IRB stroke accelerometer dataset.

## Quick Start

1. Place data files in `data/`:
- `data/windows.npz`
- `data/clinical_scores.npz`

2. Ensure BIOPM model folder exists at `CS690TR/` with checkpoint:
- `CS690TR/checkpoints/checkpoint.pt`

3. Run the extraction pipeline from repo root:
```bash
bash run_extraction_pipeline.sh
```

This runs:
1. Preprocess windows to HDF5 (`preprocessed/`)
2. Extract BIOPM embeddings (`features/biopm_features.npz`)
3. Verify embedding integrity
4. Export legacy-compatible visit schema (`features/biopm_features_legacy_schema.npz`)

## Analysis

Use the notebook for visualization and analysis:
- `biopm_irb_pipeline.ipynb`

The notebook expects extracted files to already exist and focuses on:
- Dataset checks and diagnostics
- 3-panel UMAP visualization
- LOSO logistic regression (AUC + Macro-F1)

## Main Outputs

- `features/biopm_features.npz` (window-level, 1028-d BIOPM vectors)
- `features/biopm_features_legacy_schema.npz` (visit-level old-schema compatibility)
- `results/figures/umap_healthy_vs_impaired.png`
- `results/figures/umap_feature_space_3panel_biopm.png`
- `results/lr_loso_results.txt`
- `results/lr_loso_results_with_f1.txt`

## Notes

- Large data/artifact files under `data/`, `preprocessed/`, and `features/` are ignored by git.
- Canonical extraction entrypoint is now `run_extraction_pipeline.sh` in the repo root.
