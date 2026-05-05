#!/usr/bin/env bash
# setup.sh — Bio-PM IRB Pipeline Setup
# Supports conda (default) and pip venv.
# Usage:
#   bash setup.sh                         # conda, env name = biopm-690r
#   bash setup.sh --env myenv             # conda, custom env name
#   bash setup.sh --venv                  # pip venv in .venv/
#   bash setup.sh --venv --env myenv      # pip venv, custom kernel name

set -euo pipefail

ENV_NAME="biopm-690r"
USE_CONDA=true
BIOPM_ZIP="biopm690r.zip"
BIOPM_DIR="CS690TR"

while [[ $# -gt 0 ]]; do
    case $1 in
        --env)   ENV_NAME="$2"; shift 2 ;;
        --venv)  USE_CONDA=false; shift ;;
        --conda) USE_CONDA=true;  shift ;;
        *)       echo "Unknown: $1"; exit 1 ;;
    esac
done

echo ""
echo "================================================================"
echo " Bio-PM IRB Setup"
echo " Method: $([ "$USE_CONDA" = true ] && echo 'conda' || echo 'pip venv')"
echo " Env   : $ENV_NAME"
echo "================================================================"

# 1. CS690TR
echo "[1/6] Checking Bio-PM source..."
if [[ -d "$BIOPM_DIR/src" ]]; then
    echo "  OK: $BIOPM_DIR/ found"
elif [[ -d "Downloads/$BIOPM_DIR/src" ]]; then
    mv "Downloads/$BIOPM_DIR" "$BIOPM_DIR"
    rmdir Downloads 2>/dev/null || true
    echo "  OK: moved Downloads/$BIOPM_DIR -> $BIOPM_DIR"
elif [[ -f "$BIOPM_ZIP" ]]; then
    echo "  Extracting $BIOPM_ZIP..."
    unzip -q "$BIOPM_ZIP"
    # Handle nested tree: CS690TR/CS690TR/src -> CS690TR/src
    if [[ -d "CS690TR/CS690TR/src" && ! -d "CS690TR/src" ]]; then
        mv CS690TR/CS690TR _tmp_biopm && rm -rf CS690TR && mv _tmp_biopm CS690TR
    fi
    echo "  OK: extracted"
else
    echo "  ERROR: CS690TR/ not found and biopm690r.zip not present."
    echo "  Place biopm690r.zip in the repo root and rerun."
    exit 1
fi
[[ ! -f "$BIOPM_DIR/checkpoints/checkpoint.pt" ]] && \
    echo "  ERROR: checkpoint.pt not found" && exit 1
echo "  OK: checkpoint.pt ($(du -h $BIOPM_DIR/checkpoints/checkpoint.pt | cut -f1))"

# 2. Data files
echo "[2/6] Checking data..."
[[ ! -f "data/clinical_scores.npz" ]] && echo "  ERROR: data/clinical_scores.npz missing" && exit 1
[[ ! -f "data/windows.npz" ]]         && echo "  ERROR: data/windows.npz missing" && exit 1
echo "  OK: clinical_scores.npz"
echo "  OK: windows.npz ($(du -h data/windows.npz | cut -f1))"

# 3. Python environment
echo "[3/6] Setting up Python environment..."
if [[ "$USE_CONDA" = true ]]; then
    command -v conda &>/dev/null || { echo "  ERROR: conda not found. Use --venv."; exit 1; }
    conda env list | grep -q "^$ENV_NAME " || \
        conda create -n "$ENV_NAME" python=3.11 -y -q
    echo "  Installing packages..."
    conda run -n "$ENV_NAME" pip install -q -r "$BIOPM_DIR/requirements.txt"
    conda run -n "$ENV_NAME" pip install -q \
        umap-learn seaborn scikit-learn scipy h5py tqdm jupyter ipykernel
    if ! conda run -n "$ENV_NAME" python -m ipykernel install --user \
        --name "$ENV_NAME" --display-name "Bio-PM ($ENV_NAME)"; then
        echo "  WARN: could not install Jupyter kernel in this environment."
        echo "        Packages are installed; you can still run scripts from CLI."
    fi
    PYTHON_CMD="conda run -n $ENV_NAME python"
else
    [[ ! -d ".venv" ]] && python3 -m venv .venv
    source .venv/bin/activate
    pip install -q -r "$BIOPM_DIR/requirements.txt"
    pip install -q umap-learn seaborn scikit-learn scipy h5py tqdm jupyter ipykernel
    if ! python -m ipykernel install --user --name "$ENV_NAME" --display-name "Bio-PM ($ENV_NAME)"; then
        echo "  WARN: could not install Jupyter kernel in this environment."
        echo "        Packages are installed; you can still run scripts from CLI."
    fi
    PYTHON_CMD=".venv/bin/python"
fi
echo "  OK"

# 4. Verify imports
echo "[4/6] Verifying imports..."
$PYTHON_CMD - <<'PY'
import sys
sys.path.insert(0, 'CS690TR')
import numpy as np, torch, h5py, sklearn, scipy
from src.models.biopm import BioPMModel
from src.data.preprocessing import bandpass_filter, detect_zero_crossings
d = np.load('data/clinical_scores.npz', allow_pickle=True)
clin = d['clinical_scores'].item()
print(f"  OK: {len(clin)} clinical entries, torch {torch.__version__}")
mps  = torch.backends.mps.is_available()
cuda = torch.cuda.is_available()
print(f"  Device: {'MPS (Apple Silicon)' if mps else 'CUDA' if cuda else 'CPU'}")
PY

# 5. Output folders
echo "[5/6] Creating output folders..."
mkdir -p preprocessed preprocessed_adv features \
         results/figures results/metrics results/splits dump
echo "  OK"

# 6. File check
echo "[6/6] Checking required files..."
REQUIRED=(
    "extraction pipeline/irb_preprocess.py"
    "extraction pipeline/irb_preprocess_adv.py"
    "extraction pipeline/irb_extract.py"
    "extraction pipeline/irb_extract_adv.py"
    "extraction pipeline/export_legacy_schema.py"
    "extraction pipeline/export_legacy_schema_adv.py"
    "extraction pipeline/verify_embeddings.py"
    "run_pipeline.sh"
    "run_pipeline_adv.sh"
    "biopm_irb_analysis.ipynb"
)
ALL_OK=true
for f in "${REQUIRED[@]}"; do
    [[ -f "$f" ]] && echo "  OK: $f" || { echo "  MISSING: $f"; ALL_OK=false; }
done

echo ""
echo "================================================================"
[[ "$ALL_OK" = true ]] && echo " Ready." || echo " Some files missing — check above."
echo " Standard:  bash run_pipeline.sh"
echo " Advanced:  bash run_pipeline_adv.sh"
echo " Notebook:  jupyter notebook biopm_irb_analysis.ipynb"
echo "================================================================"
