# BioPM Full Autopsy (Deep Dive)

## Scope and constraints
- Full read-through completed for the whole `CS690TR` codebase (`src/`, `scripts/`, `examples/`, `starter_project/`) plus project IRB pipeline scripts (`extraction pipeline/`).
- `CS690TR/` model/package code was treated as read-only.
- Root-cause analysis focuses on: NaN padding behavior, fill-rate collapse, gravity stream behavior, and standard vs alt pipeline differences.

## 1. Complete stage map: what runs where

### A) BioPM package (`CS690TR/src`)
- `src/data/preprocessing.py`
  - Signal filters: `bandpass_filter`, `lowpass_filter`, `highpass_filter`.
  - ME detection: `detect_zero_crossings`.
  - Gravity-aligned segmentation helper: `assign_zero_crossings`.
  - HDF5 loader: `load_preprocessed_h5`.
- `src/models/biopm.py`
  - `TimeSeriesTransformer` (`encoder_acc`): ME tokens -> contextual `(B,L,64)`.
  - `GravityCNNEncoder` (`encoder_gravity`): gravity window `(B,T,3)` -> `(B,64)`.
  - `masked_mean_std`: currently unmasked mean/std over sequence.
  - `BioPMModel`: dual-stream + classifier wrapper.
- `src/inference/feature_extractor.py`
  - Uses `encoder_acc` + `masked_mean_std` for first 128 dims.
  - Uses interpolated+flattened `x_gravity` for 900 dims.
  - Produces `(N,1028)` = `[128 acc pooled | 900 gravity flat]`.

### B) Generic scripts (`CS690TR/scripts`)
- `preprocess_data.py`: generic student preprocessing template adapted from mHealth.
- `extract_features.py`: CLI wrapper around `src/inference/feature_extractor.py`.
- `generation_starter.py`: masked infill experiment scaffold (not production extraction).

### C) IRB project pipeline (`extraction pipeline/`)
- `irb_preprocess.py`: 3s windows, `pad_size=57`, creates `preprocessed/*.h5`.
- `irb_preprocess_alt.py`: grouped windows (default 3x), scales pad linearly (default 171), creates `preprocessed_alt/*.h5`.
- `irb_extract.py`: custom extractor for `Data_MeLabel_{subj}_{week}.h5` and ARAT/FMA metadata.
- `irb_analyze.py`, `verify_embeddings.py`, `inspect_dataset.py`, `debug_pipeline.py`: diagnostics/analysis utilities.

## 2. Exact causes of failure (cause of death)

### Failure A: "masked" pooling is not masked
Evidence:
- `CS690TR/src/models/biopm.py:344-346`
  - `masked_mean_std(x, mask=None)` returns `torch.cat([x.mean(dim=1), x.std(dim=1)], dim=-1)`.
  - `mask` arg is ignored.

Impact:
- Pooling includes all sequence slots, not just real ME tokens.
- Increasing `pad_size` adds more padding-derived token outputs into pooled stats.
- This contaminates both mean and std streams and makes pad-size sensitivity worse than expected.

### Failure B: gravity CNN exists in model, but feature extraction path bypasses it
Evidence:
- Gravity CNN is real and defined in `CS690TR/src/models/biopm.py:392-418` (`encoder_gravity`).
- Generic extractor bypasses it in `CS690TR/src/inference/feature_extractor.py:106-119`:
  - `x_gravity -> transpose -> interpolate -> flatten` => 900 dims.
- IRB extractor also follows this flattened path by default.

Impact:
- Produced gravity features are not learned gravity embeddings; they are processed raw signal vectors.
- The final 1028-d representation is effectively `acc_encoder + handcrafted gravity vector`, not true dual-stream learned encoding.

### Failure C: checkpoint does not contain gravity encoder weights
Evidence from checkpoint inspection:
- `checkpoint.pt` keys are bare encoder-acc + decoder-pretraining keys (e.g. `mask_token`, `conv_encode.*`, transformer layers, `decoder_cnn.*`).
- No `encoder_gravity.*` keys.

Impact:
- Even if you route data through `encoder_gravity`, with this checkpoint those weights are random init unless a different checkpoint is provided.
- This explains why historical code likely chose flatten-900 gravity features for stable inference.

### Failure D: gravity DC component appears removed in IRB windows
Observed in prior autopsy runs:
- Mean per-window acceleration magnitude is around `~0.026 g`, far from expected static gravity near `~1.0 g`.

Impact:
- Lowpass(0.5Hz) has limited physically meaningful gravity to recover.
- Flattened gravity features can become low-energy/non-informative.

### Failure E: fill-rate collapse is mostly a ratio problem
Evidence:
- ME detection count per 3s window is healthy (`~16-19` in sampled subjects).
- `pad_size` increase (57 -> 171/192) mostly changes denominator; detected ME count barely changes.

Impact:
- Fill rate drops sharply with larger pad sizes.
- With unmasked pooling, this has amplified downstream effect.

## 3. Why you are "not using BioPM gravity stream"

Short answer:
- You are using the BioPM **accelerometer encoder** (`encoder_acc`) from checkpoint.
- You are **not** using learned gravity encoder output in extracted features because:
  1. feature extractor code paths flatten gravity to 900-d by design, and
  2. available checkpoint does not provide gravity encoder pretrained weights.

So the current gravity stream is:
- `x_gravity` from preprocessing (typically lowpass output) -> interpolate to length 300 -> flatten to 900.

The model-defined gravity stream (not used by default extractor) is:
- `encoder_gravity: (B,T,3) -> Conv/GN/GELU blocks -> pooled -> Linear -> (B,64)`.

## 4. Standard vs alt pipeline: what changed and what did not

- `irb_preprocess.py` (standard): 3s windows, `pad_size=57`.
- `irb_preprocess_alt.py` (alt default): 9s grouped blocks, but `pad_size=171` (scaled by 3).

Consequence:
- Grouping raises ME count roughly 3x.
- Scaling pad by 3 keeps fill ratio near same range, so alt did not deliver expected fill-rate jump.

Best observed configuration from experiments:
- Group 3 windows (9s) **but keep `pad_size=57`**.
- This drives near-full occupancy (`~90-95%` in sampled subjects) and better std-pool signal.

## 5. Additional code-quality hazards discovered

### Hazard 1: duplicate repo tree
- Both `CS690TR/` and `CS690TR/CS690TR/` exist with same code.
- This is easy to confuse and increases risk of editing/running wrong copy.

### Hazard 2: generic `load_preprocessed_h5` subject-id parsing mismatch for IRB filename pattern
- In `CS690TR/src/data/preprocessing.py`, subject id is parsed as `parts[-1]` from `Data_MeLabel_*` path.
- For IRB naming `Data_MeLabel_{subj}_{week}.h5`, `parts[-1]` is week, not subject.
- IRB custom extractor avoids this by custom parsing, but generic loader is unsafe for that naming style.

### Hazard 3: possible failure path in `detect_zero_crossings`
- It uses `filt_zc = [zc_idx[0]]` without guarding empty `zc_idx`.
- This is caught by surrounding `try/except` in preprocess scripts, but still means silent dropped windows can happen.

## 6. What to change (do not edit `CS690TR/` code directly)

Implement these in project-side pipeline only and document in your project code:

1. Pooling fix at extraction time
- In project extractor, pool only valid tokens (`~isnan(first_patch_value)` mask).
- Keep legacy mode toggle for backward compatibility.

2. Gravity mode switch
- Keep current `flat900` as default with this checkpoint.
- Add optional `encoder64` mode only when a checkpoint containing gravity weights is available.
- Emit explicit warning if `encoder64` is requested but checkpoint lacks gravity keys.

3. Fill-rate fix
- Use grouped windows (9s blocks) with fixed `pad_size=57`.
- Do not scale pad proportionally with grouped duration.

4. Data reality check before extraction
- Compute per-window mean magnitude baseline and lowpass energy stats.
- If gravity magnitude is far from expected (e.g., <<0.1g), treat gravity branch as low-confidence.

5. Consolidate source-of-truth path
- Pick one tree (`CS690TR/`), ignore nested duplicate in all scripts/tooling.

## 7. Recommended extraction policy with current checkpoint

Given current assets:
- Checkpoint: acc encoder pretrained, gravity encoder not pretrained.
- Data: gravity DC likely removed.

Recommended production feature set order:
1. `acc_128_valid_only` (masked pooling over real tokens only).
2. `gravity_flat900` as optional auxiliary channel, not primary driver.
3. Avoid `encoder_gravity` unless/until you obtain compatible gravity-pretrained weights.

## 8. Concrete answers to your key questions

- Why NaN padding hurts? because pooling includes padded slots in current implementation.
- Why fill rate is low? ME count per 3s window is limited while pad size is large.
- Why gravity stream looks dead? gravity content in provided data is already attenuated/centered, and extraction uses flattened lowpass signal rather than learned gravity encoder.
- Why not using BioPM gravity CNN? default feature extraction code bypasses it, and current checkpoint lacks gravity CNN pretrained params.

## 9. Output files from this autopsy cycle
- Existing: `AUTOPSY_REPORT.md`
- New deep dive: `AUTOPSY_DEEP_DIVE.md`

