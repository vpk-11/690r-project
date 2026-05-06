# 690r-project

BioPM-based feature extraction and clinical regression analysis for the IRB wrist accelerometer dataset.

## What This Repo Does
- Runs two embedding pipelines:
  - `standard`: 3s windows
  - `advanced`: 9s grouped blocks with improved pooling
- Produces BioPM features and legacy visit-level schema files
- Runs notebook-based analysis focused on ARAT/FMA regression and reliability/correlation diagnostics

## Setup
Bootstrap environment and dependencies:

```bash
bash setup.sh
```

Optional flags:

```bash
bash setup.sh --env biopm-690r   # custom conda env name
bash setup.sh --venv             # use .venv instead of conda
```

What `setup.sh` checks:
1. `data/` files (`windows.npz`, `clinical_scores.npz`) — warning-only
2. Environment existence/creation (`biopm-690r` by default)
3. `CS690TR/` presence (or `biopm690r.zip` fallback extraction)
4. Dependency installation (`CS690TR/requirements.txt`)
5. Core import verification (if `CS690TR` is present)
6. Output folder creation

After setup, activate your env (script prints exact command), then run pipelines.

## Run Pipelines
Standard pipeline:

```bash
bash run_pipeline.sh
```

Advanced pipeline:

```bash
bash run_pipeline_adv.sh
```

Standard outputs:
- `preprocessed/`
- `features/biopm_features.npz`
- `features/biopm_features_legacy_schema.npz`

Advanced outputs:
- `preprocessed_adv/`
- `features/biopm_features_adv.npz`
- `features/biopm_features_legacy_schema_adv.npz`

## Run Analysis
Open and run all cells:

```bash
jupyter notebook biopm_irb_analysis.ipynb
```

Notebook writes analysis artifacts to:
- `outputs/figures/`
- `outputs/metrics/`
- `outputs/splits/`

## Expected Project Structure (Core Only)

```text
690r-project/
├── setup.sh
├── run_pipeline.sh
├── run_pipeline_adv.sh
├── biopm_irb_analysis.ipynb
├── README.md
├── CS690TR/
│   ├── src/
│   ├── checkpoints/checkpoint.pt
│   └── requirements.txt
├── data/
│   ├── windows.npz
│   └── clinical_scores.npz
├── extraction pipeline/
│   ├── irb_preprocess.py
│   ├── irb_preprocess_adv.py
│   ├── irb_extract.py
│   ├── irb_extract_adv.py
│   ├── export_legacy_schema.py
│   ├── export_legacy_schema_adv.py
│   └── verify_embeddings.py
├── preprocessed/
├── preprocessed_adv/
├── features/
│   ├── biopm_features.npz
│   ├── biopm_features_legacy_schema.npz
│   ├── biopm_features_adv.npz
│   └── biopm_features_legacy_schema_adv.npz
└── outputs/
    ├── figures/
    ├── metrics/
    └── splits/
```

## Notes
- `setup.sh` is intentionally non-blocking for missing dataset artifacts; it is a starter bootstrap script.
- If `CS690TR/` and `biopm690r.zip` are both missing, setup prints the expected directory layout.
