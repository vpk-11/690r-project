# 690r-project

BioPM-based feature extraction and clinical regression analysis for the IRB wrist accelerometer dataset.

## What This Repo Does
- Runs three embedding pipelines:
  - `standard`: 3s windows, pad 192, original BioPM pooling
  - `adv`: 3s windows, pad 192, valid-token pooling using NaN pad mask
  - `alt`: 9s grouped windows, pad 192, same valid-token pooling as `adv`
- Produces BioPM features and legacy visit-level schema files
- Runs notebook-based analysis focused on ARAT/FMA regression and reliability/correlation diagnostics

## Setup
Bootstrap environment and dependencies:

```bash
bash setup.sh
```

Optional flags:

```bash
bash setup.sh --env biopm-690r
bash setup.sh --venv
```

After setup, activate your env, then run pipelines.

## Run Pipelines
Standard:

```bash
bash run_pipeline.sh
```

ADV (3s + NaN-mask pooling):

```bash
bash run_pipeline_adv.sh
```

ALT (9s grouped + NaN-mask pooling):

```bash
bash run_pipeline_alt.sh
```

### Quick subject tests (recommended first)
Use a single/few subjects to avoid long runs:

```bash
# Standard (single subject)
python "extraction pipeline/irb_preprocess.py" --data_dir data --output preprocessed_test --subject 549 --pad_size 192
python "extraction pipeline/irb_extract.py" --preprocessed preprocessed_test --checkpoint CS690TR/checkpoints/checkpoint.pt --output features/biopm_features_test_std.npz --device cpu

# ADV (single subject)
python "extraction pipeline/irb_preprocess_adv.py" --data_dir data --output preprocessed_adv_test --subject 549 --pad_size 192
python "extraction pipeline/irb_extract_adv.py" --preprocessed preprocessed_adv_test --checkpoint CS690TR/checkpoints/checkpoint.pt --output features/biopm_features_test_adv.npz --device cpu

# ALT (single subject)
python "extraction pipeline/irb_preprocess_alt.py" --data_dir data --output preprocessed_alt_test --subject 549 --group_size 3 --pad_size 192
python "extraction pipeline/irb_extract_alt.py" --preprocessed preprocessed_alt_test --checkpoint CS690TR/checkpoints/checkpoint.pt --output features/biopm_features_test_alt.npz --device cpu
```

## Outputs
Standard outputs:
- `preprocessed/`
- `features/biopm_features.npz`
- `features/biopm_features_legacy_schema.npz`

ADV outputs:
- `preprocessed_adv/`
- `features/biopm_features_adv.npz`
- `features/biopm_features_legacy_schema_adv.npz`

ALT outputs:
- `preprocessed_alt/`
- `features/biopm_features_alt.npz`
- `features/biopm_features_legacy_schema_alt.npz`

## Run Analysis
Open and run all cells:

```bash
jupyter notebook biopm_irb_analysis.ipynb
```

Notebook writes analysis artifacts to:
- `outputs/figures/`
- `outputs/metrics/`
- `outputs/splits/`

## Core Structure

```text
690r-project/
├── setup.sh
├── run_pipeline.sh
├── run_pipeline_adv.sh
├── run_pipeline_alt.sh
├── biopm_irb_analysis.ipynb
├── README.md
├── extraction pipeline/
│   ├── irb_preprocess.py
│   ├── irb_preprocess_adv.py
│   ├── irb_preprocess_alt.py
│   ├── irb_extract.py
│   ├── irb_extract_adv.py
│   ├── irb_extract_alt.py
│   ├── export_legacy_schema.py
│   ├── export_legacy_schema_adv.py
│   ├── export_legacy_schema_alt.py
│   └── verify_embeddings.py
└── features/
```
