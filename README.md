# Bio-PM IRB Stroke Recovery Project (CS690R)

## 1. Project Overview
This project builds and validates wearable-sensor biomarkers for upper-limb impairment in stroke recovery. We use the IRB dataset (61 subjects, 198 clinical visits, 587,046 windows) and a pretrained Bio-PM wrist-accelerometer transformer to extract embeddings. The workflow is: preprocessing and embedding extraction, feature engineering for ARAT/FMA prediction, and biomarker validation (reliability/validity/responsiveness). The goal is to predict clinical outcomes (ARAT, FMA-UE) and identify stable, clinically meaningful embedding dimensions. This was completed as a 3-member CS690R graduate project.

## 2. Repository Structure
```text
690r-project/
├── setup.sh                                   # environment setup helper
├── requirements.txt                           # pip requirements fallback
├── run_pipeline.sh                            # standard pipeline runner
├── run_pipeline_adv.sh                        # adv pipeline runner
├── run_pipeline_alt.sh                        # alt pipeline runner
├── feature_extraction_analysis.ipynb          # m1 (feature extraction analysis)
├── feature_engineering.ipynb                  # m2 standard feature engineering
├── feature_engineering_adv.ipynb              # m2 adv feature engineering
├── feature_engineering_alt.ipynb              # m2 alt feature engineering
├── feature_validation.ipynb                   # m3 biomarker validation
├── CONTEXT_DUMP.md                            # generated context dump
├── README.md                                  # this file
├── extraction pipeline/                       # preprocessing/extraction/export scripts
│   ├── irb_preprocess.py
│   ├── irb_preprocess_adv.py
│   ├── irb_preprocess_alt.py
│   ├── irb_extract.py
│   ├── irb_extract_adv.py
│   ├── irb_extract_alt.py
│   ├── export_legacy_schema.py
│   ├── export_legacy_schema_adv.py
│   ├── export_legacy_schema_alt.py
│   ├── irb_analyze.py
│   ├── verify_embeddings.py
│   ├── debug_pipeline.py
│   └── inspect_dataset.py
├── data/                                      # source IRB data files
├── preprocessed/                              # standard preprocessed HDF5 (198 files)
├── preprocessed_adv/                          # adv preprocessed HDF5 (198 files)
├── preprocessed_alt/                          # alt preprocessed HDF5 (198 files)
├── features/                                  # window-level + legacy-schema NPZ features
│   ├── biopm_features.npz
│   ├── biopm_features_adv.npz
│   ├── biopm_features_alt.npz
│   ├── biopm_features_legacy_schema.npz
│   ├── biopm_features_legacy_schema_adv.npz
│   └── biopm_features_legacy_schema_alt.npz
└── results/
    ├── feature_extraction/                    # m1 outputs
    │   ├── metrics/*.csv
    │   └── figures/*.png
    ├── feature_engineering/                   # m2 outputs
    │   ├── metrics/*.csv
    │   └── figures/*.png
    └── validation/                            # m3 outputs
        ├── metrics/*.csv
        └── figures/*.png
```

## 3. Quick Start
```bash
# 1) Clone and enter repo
git clone https://github.com/vpk-11/690r-project.git
cd 690r-project

# 2) Environment
bash setup.sh

# 3) Set Bio-PM root
export BIOPM_ROOT=CS690TR

# 4) Run one pipeline
bash run_pipeline.sh
# bash run_pipeline_adv.sh
# bash run_pipeline_alt.sh

# 5) Open notebooks
conda activate biopm-690r
export KMP_DUPLICATE_LIB_OK=TRUE
jupyter notebook
```

## 3.1 Different Cases / Troubleshooting
Use this section when the standard flow fails or when you need a specific runtime mode.

```bash
# Case A: BIOPM_ROOT not found
export BIOPM_ROOT=CS690TR
ls "$BIOPM_ROOT/checkpoints/checkpoint.pt"

# Case B: checkpoint missing
ls CS690TR/checkpoints/
# expected: checkpoint.pt

# Case C: data folder missing
ls data
# expected: windows.npz and clinical_scores.npz
```

Device/runtime cases:
- `MPS unstable on some macOS setups`: keep extraction on CPU (current runner defaults do this).
- `Need CUDA`: set `DEVICE=cuda` before running if you have a compatible GPU and dependencies.
- `OpenMP/macOS conflict`: `export KMP_DUPLICATE_LIB_OK=TRUE` before notebook runs.

Pipeline-specific quick checks:
```bash
# Standard
bash run_pipeline.sh
python "extraction pipeline/verify_embeddings.py" --features features/biopm_features.npz

# Adv
bash run_pipeline_adv.sh
python "extraction pipeline/verify_embeddings.py" --features features/biopm_features_adv.npz

# Alt
bash run_pipeline_alt.sh
python "extraction pipeline/verify_embeddings.py" --features features/biopm_features_alt.npz
```

Expected output locations:
- Member 1 outputs: `results/feature_extraction/{metrics,figures}`
- Member 2 outputs: `results/feature_engineering/{metrics,figures}`
- Member 3 outputs: `results/validation/{metrics,figures}`

## 4. The Three Pipelines
| Pipeline | Window size | pad_size | Pooling method | Fill rate | Preprocess script | Extract script | Export script | Window file | Legacy schema file | Best ARAT R2 | Best ARAT rho |
|---|---:|---:|---|---:|---|---|---|---|---|---:|---:|
| Standard | 3s | 192 | unmasked mean/std | ~31.7% | `irb_preprocess.py` | `irb_extract.py` | `export_legacy_schema.py` | `features/biopm_features.npz` | `features/biopm_features_legacy_schema.npz` | 0.612 | 0.783 |
| Adv | 3s | 192 | valid-token-only masked pooling | ~94.1% | `irb_preprocess_adv.py` | `irb_extract_adv.py` | `export_legacy_schema_adv.py` | `features/biopm_features_adv.npz` | `features/biopm_features_legacy_schema_adv.npz` | 0.562 | 0.754 |
| Alt | 9s grouped | 192 | valid-token-only masked pooling | high (grouped) | `irb_preprocess_alt.py` | `irb_extract_alt.py` | `export_legacy_schema_alt.py` | `features/biopm_features_alt.npz` | `features/biopm_features_legacy_schema_alt.npz` | 0.597 | 0.756 |

## 5. Embedding Structure
Bio-PM embedding is 1028 dimensions per visit/window:
- `[0:64]` `acc_mean`
- `[64:128]` `acc_std`
- `[128:1028]` gravity stream (900-d)

Technical note: the gravity stream is weak for biomarker tasks (low validity/reliability in validation outputs) and is impacted by DC-removal/representation issues; most useful clinical signal is in the 128-d accelerometer stream.

## 6. Key Findings Summary
- Best overall predictive result (member-2 outputs): Standard Top-100 Spearman-selected dims, ARAT `R2=0.612`, `rho=0.783`.
- Adv baseline (`1028-d`) ARAT: `R2=0.492`, `rho=0.722`; Alt baseline ARAT: `R2=0.447`, `rho=0.728`.
- Gravity-only features are poor relative to acc-stream variants across analyses.
- Acc-stream ablations consistently outperform full 1028-d baselines for ARAT/FMA in m2 outputs.
- Validation ICC good dims (`>0.75`): Standard 127, Adv 127, Alt 125.
- Top ARAT-correlated dims (validation): Standard dim 71 (`rho=+0.753`), Adv dim 117 (`rho=+0.725`), Alt dim 107 (`rho=+0.736`).
- Longitudinal drift vs ARAT change is weak/null in all pipelines (e.g., Adv `rho=+0.056`, `p=0.492`).

## 7. Loading Features
```python
import numpy as np

d = np.load("features/biopm_features_legacy_schema.npz", allow_pickle=True)
X    = np.ascontiguousarray(d["features"], dtype=np.float32)   # (198, 1028)
arat = d["arat"].astype(float)
fma  = d["fma"].astype(float)
pids = d["pids"].astype(int)  # LOSO groups
```

Window-level loading (for attention/temporal regrouping):
```python
w = np.load("features/biopm_features_adv.npz", allow_pickle=True)
Xw   = np.ascontiguousarray(w["features"], dtype=np.float32)
subj = w["subj"].astype(int)
week = w["week"].astype(int)
```

## 8. Running Individual Scripts
```bash
# dataset inspection
python "extraction pipeline/inspect_dataset.py" --data_dir data

# debug one subject
export BIOPM_ROOT=CS690TR
python "extraction pipeline/debug_pipeline.py" --subject 549 --checkpoint CS690TR/checkpoints/checkpoint.pt

# verify any feature file
python "extraction pipeline/verify_embeddings.py" --features features/biopm_features.npz

# standard pipeline manual
python "extraction pipeline/irb_preprocess.py" --output preprocessed --data_dir data
python "extraction pipeline/irb_extract.py" --preprocessed preprocessed --checkpoint CS690TR/checkpoints/checkpoint.pt --output features/biopm_features.npz
python "extraction pipeline/export_legacy_schema.py" --source features/biopm_features.npz --output features/biopm_features_legacy_schema.npz

# adv pipeline manual
python "extraction pipeline/irb_preprocess_adv.py" --output preprocessed_adv --data_dir data
python "extraction pipeline/irb_extract_adv.py" --preprocessed preprocessed_adv --checkpoint CS690TR/checkpoints/checkpoint.pt --output features/biopm_features_adv.npz
python "extraction pipeline/export_legacy_schema_adv.py" --source features/biopm_features_adv.npz --output features/biopm_features_legacy_schema_adv.npz

# alt pipeline manual
python "extraction pipeline/irb_preprocess_alt.py" --output preprocessed_alt --data_dir data --group_size 3
python "extraction pipeline/irb_extract_alt.py" --preprocessed preprocessed_alt --checkpoint CS690TR/checkpoints/checkpoint.pt --output features/biopm_features_alt.npz
python "extraction pipeline/export_legacy_schema_alt.py" --source features/biopm_features_alt.npz --output features/biopm_features_legacy_schema_alt.npz
```

## 9. Known Issues / Technical Notes
- Bio-PM pooling bug (`masked_mean_std`) in original standard path can include padded slots.
- Standard pipeline can outperform adv despite that bug (observed in this repo outputs).
- Gravity representation is not reliable for biomarker prediction in this dataset.
- `encoder_gravity` pretrained weights are not part of the used checkpoint path.
- If duplicate nested `CS690TR/CS690TR` appears, use the outer directory as `BIOPM_ROOT`.
- On some macOS setups, set `KMP_DUPLICATE_LIB_OK=TRUE` before notebook runs.

## 10. Canonical LOSO Regression Pattern
```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import numpy as np

def loso_regression(X, y, pids, alpha=1.0):
    logo = LeaveOneGroupOut()
    y_true, y_pred = [], []
    for train_idx, test_idx in logo.split(X, y, groups=pids):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        reg = Ridge(alpha=alpha)
        reg.fit(X_tr, y[train_idx])
        y_pred.extend(reg.predict(X_te).tolist())
        y_true.extend(y[test_idx].tolist())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ss_res = ((y_true - y_pred)**2).sum()
    ss_tot = ((y_true - y_true.mean())**2).sum()
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(((y_true - y_pred)**2).mean())
    rho, p = spearmanr(y_true, y_pred)
    return r2, rmse, rho, y_true, y_pred
```

## 11. Dataset Facts
- Clinical entries: 223 `(subject, week)`
- Unique subjects: 61 (8 healthy, 53 stroke)
- Feature-matrix visits: 198 (4 healthy, 194 stroke)
- Unique subjects with matched windows: 36
- ARAT range: 0–57
- FMA range: 3–66
- ARAT/FMA Spearman correlation: ~0.956
