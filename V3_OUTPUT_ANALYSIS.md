# Bio-PM V3 Output Analysis

## Summary
- V3 preprocessing/extraction completed successfully end-to-end.
- Core V3 goal was achieved: fill rate increased from ~31.66% (standard) and ~34.67% (alt) to **94.07%**.
- V3 gravity stream energy is higher (raw gravity input), but this remains flat900 handcrafted gravity, not learned `encoder_gravity`.

## 1) Fill-Rate Audit (Direct HDF5 Check)
Computed from all HDF5 files using `x_acc_filt[:, :, 0]` non-NaN as valid token mask.

| Pipeline | Files | Windows | Global Fill Rate | Avg MEs / window(block) | Median MEs | P95 MEs |
|---|---:|---:|---:|---:|---:|---:|
| standard (`preprocessed`) | 198 | 587,046 | 31.6604% | 18.0464 | 18 | 26 |
| alt (`preprocessed_alt`) | 198 | 195,610 | 34.6734% | 59.2916 | 59 | 79 |
| v3 (`preprocessed_v3`) | 198 | 195,610 | **94.0661%** | 53.6177 | 57 | 57 |

Interpretation:
- Standard: low fill due to ~18 MEs in 57 slots.
- Alt: grouped windows increase MEs, but large pad (171) keeps ratio low.
- V3: grouped windows + fixed `pad_size=57` produces near-full occupancy and controlled truncation (`p95=57`).

## 2) Feature-Level Comparison
Computed from generated `.npz` files.

| Feature file | Shape | mean_pool abs (0:64) | std_pool abs (64:128) | gravity abs (128:) | Healthy % | Subjects |
|---|---:|---:|---:|---:|---:|---:|
| `features/biopm_features.npz` | (587046, 1028) | 0.70913 | 0.41857 | 0.05521 | 4.84% | 36 |
| `features/biopm_features_alt.npz` | (195610, 1028) | 0.69872 | 0.43599 | 0.05581 | 4.84% | 36 |
| `features/biopm_features_v3.npz` | (195610, 1028) | 0.74289 | 0.36587 | **0.09538** | 4.84% | 36 |

Additional run log metric (from extraction print):
- Transformer [0:128] mean_abs = 0.5544
- Gravity [128:] mean_abs = 0.0954

Notes:
- V3 gravity magnitude increased significantly (`0.0558 -> 0.0954`) with raw gravity input.
- Transformer split changed distribution under `masked_mean_std_valid`; total transformer energy remains non-zero and stable.

## 3) Label/Visit Integrity Check
- Window labels: 195,610 total, healthy 9,462 (4.8%), stroke 186,148 (95.2%).
- Subjects: 36.
- Legacy export V3 produced `(198, 1028)` for full/even/odd visits with aligned visit keys.

## 4) New V3 Notebook
Created:
- `biopm_irb_pipeline_v3.ipynb`

It is a clone of the alt notebook with V3 paths/scripts:
- `run_extraction_pipeline_v3.sh`
- `preprocessed_v3/`
- `features/biopm_features_v3.npz`
- `features/biopm_features_legacy_schema_v3.npz`
- V3 results output names (`*_v3`)

## 5) Direct Answer: "Can we check fill rate?"
Yes. Verified directly from stored V3 HDF5 tensors:
- **V3 global fill rate = 94.0661%**
- This matches your pipeline log (~94.1%).

