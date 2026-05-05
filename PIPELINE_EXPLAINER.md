# Bio-PM IRB Pipeline Explainer (Current State)

Last updated: May 5, 2026

This file explains what we ran, what failed, what was fixed, and what the current recommended pipeline is.

## 1. Pipeline Variants

## Standard (baseline)
- Preprocess: `extraction pipeline/irb_preprocess.py`
- Extract: `extraction pipeline/irb_extract.py`
- Runner: `run_extraction_pipeline.sh`
- Output HDF5: `preprocessed/`
- Output features: `features/biopm_features.npz`

## Alt (intermediate)
- Preprocess: `extraction pipeline/irb_preprocess_alt.py`
- Extract: `extraction pipeline/irb_extract.py`
- Runner: `run_extraction_pipeline_alt.sh`
- Output HDF5: `preprocessed_alt/`
- Output features: `features/biopm_features_alt.npz`

## V3 (recommended)
- Preprocess: `extraction pipeline/irb_preprocess_v3.py`
- Extract: `extraction pipeline/irb_extract_v3.py`
- Export: `extraction pipeline/export_legacy_schema_v3.py`
- Runner: `run_extraction_pipeline_v3.sh`
- Output HDF5: `preprocessed_v3/`
- Output features: `features/biopm_features_v3.npz`

## 2. What Was Wrong (Root Causes)

1. Pooling bug in upstream BioPM code (`CS690TR/src/models/biopm.py`)
- `masked_mean_std` is nominally masked but does unmasked `mean/std` over sequence slots.
- Result: padding tokens contaminate pooled transformer output.

2. Fill-rate mismatch
- 3-second windows generate ~18 MEs, but sequence padding dominates.
- Standard improved over pad=192 but still low occupancy.

3. Alt scaled numerator and denominator together
- Alt grouped 3 windows (more MEs), but also scaled pad size to 171.
- Ratio stayed low; fill barely improved.

4. Gravity handling on this dataset
- The IRB signal appears gravity-DC attenuated relative to raw consumer-device gravity assumptions.
- Lowpass gravity gave weak magnitude in practice.

## 3. What V3 Changed

1. Group 3 windows into 9-second blocks.
2. Keep fixed `pad_size=57` (do not scale pad with grouped duration).
3. Use raw acceleration for `x_gravity` in V3 preprocessing.
4. Use `masked_mean_std_valid` in `irb_extract_v3.py` to pool over valid tokens only.

Note:
- No edits were made to `CS690TR/` model code for V3.
- V3 fixes are implemented in project-side scripts only.

## 4. Measured Results

These are computed from actual generated artifacts (`preprocessed*` and `features/*.npz`).

## HDF5 fill-rate audit
- `preprocessed` (standard):
  - files: 198
  - windows: 587,046
  - global fill: **31.6604%**
  - avg ME count: 18.0464
- `preprocessed_alt`:
  - files: 198
  - windows: 195,610
  - global fill: **34.6734%**
  - avg ME count: 59.2916
- `preprocessed_v3`:
  - files: 198
  - windows: 195,610
  - global fill: **94.0661%**
  - avg ME count: 53.6177

## Feature audit
- `features/biopm_features.npz` (standard):
  - shape: (587046, 1028)
  - mean_pool abs: 0.7091
  - std_pool abs: 0.4186
  - gravity abs: 0.0552
- `features/biopm_features_alt.npz`:
  - shape: (195610, 1028)
  - mean_pool abs: 0.6987
  - std_pool abs: 0.4360
  - gravity abs: 0.0558
- `features/biopm_features_v3.npz`:
  - shape: (195610, 1028)
  - mean_pool abs: 0.7429
  - std_pool abs: 0.3659
  - gravity abs: 0.0954

## From V3 pipeline runtime logs
- Total windows: 195,610
- Windows with >=1 valid ME token: 100.0%
- Avg MEs per block: 53.6
- Avg fill rate: 94.1%

## 5. Interpretation

1. V3 solved the occupancy problem.
- Fill rate jumped from ~32-35% to ~94%.

2. V3 increased gravity stream magnitude.
- Raw gravity input in V3 raised gravity abs substantially.

3. V3 changed transformer distribution.
- `masked_mean_std_valid` changes pooling behavior by excluding low-magnitude padded-token outputs.
- This changes absolute mean/std statistics compared to legacy pooling, by design.

## 6. Recommended Pipeline to Run Today

Use V3 end-to-end:

```bash
export BIOPM_ROOT=CS690TR
bash run_extraction_pipeline_v3.sh
```

Then analyze with:
- `biopm_irb_pipeline_v3.ipynb` (rewritten as fully data-driven output checks)

## 7. File Map (Current)

- Standard runner: `run_extraction_pipeline.sh`
- Alt runner: `run_extraction_pipeline_alt.sh`
- V3 runner: `run_extraction_pipeline_v3.sh`
- V3 preprocess: `extraction pipeline/irb_preprocess_v3.py`
- V3 extract: `extraction pipeline/irb_extract_v3.py`
- V3 export: `extraction pipeline/export_legacy_schema_v3.py`
- V3 notebook: `biopm_irb_pipeline_v3.ipynb`
- V3 analysis snapshot: `V3_OUTPUT_ANALYSIS.md`

## 8. Guardrails

- `CS690TR/` is treated as read-only project dependency.
- New project-level fixes should be implemented in `extraction pipeline/` scripts and docs.
- Keep output versioning explicit (`*_v3`) so historical outputs are preserved.
