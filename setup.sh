#!/usr/bin/env bash
# setup.sh — bootstrap environment for 690r-project
#
# Goal:
# - help new users get started without hard-failing on optional artifacts
# - set up Python env and install dependencies
# - validate expected project layout and print actionable guidance
#
# Usage:
#   bash setup.sh
#   bash setup.sh --env biopm-690r
#   bash setup.sh --venv

set -euo pipefail

ENV_NAME="biopm-690r"
USE_CONDA=true
BIOPM_DIR="CS690TR"
BIOPM_ZIP="biopm690r.zip"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)   ENV_NAME="$2"; shift 2 ;;
    --venv)  USE_CONDA=false; shift ;;
    --conda) USE_CONDA=true; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

print_layout_hint() {
  cat <<'TXT'
Expected project layout:
  690r-project/
    CS690TR/
      src/
      checkpoints/checkpoint.pt
      requirements.txt
    data/
      windows.npz
      clinical_scores.npz
    extraction pipeline/
      irb_preprocess.py
      irb_preprocess_adv.py
      irb_extract.py
      irb_extract_adv.py
      export_legacy_schema.py
      export_legacy_schema_adv.py
    run_pipeline.sh
    run_pipeline_adv.sh
    biopm_irb_analysis.ipynb
TXT
}

echo ""
echo "================================================================"
echo " Bio-PM IRB Setup"
echo " Method: $([ "$USE_CONDA" = true ] && echo 'conda' || echo 'pip venv')"
echo " Env   : $ENV_NAME"
echo "================================================================"

# ------------------------------------------------------------------
# 1) Data presence check (warning only)
# ------------------------------------------------------------------
echo "[1/6] Checking data folder (warning-only)..."
if [[ -f "data/clinical_scores.npz" ]]; then
  echo "  OK: data/clinical_scores.npz"
else
  echo "  WARN: data/clinical_scores.npz not found."
fi
if [[ -f "data/windows.npz" ]]; then
  echo "  OK: data/windows.npz ($(du -h data/windows.npz | cut -f1))"
else
  echo "  WARN: data/windows.npz not found."
fi

# ------------------------------------------------------------------
# 2) Environment check/create (root environment.yml / requirements.txt)
# ------------------------------------------------------------------
echo "[2/6] Checking Python environment..."
if [[ "$USE_CONDA" = true ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "  ERROR: conda not found. Re-run with --venv or install conda."
    exit 1
  fi

  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "  OK: conda env '$ENV_NAME' already exists"
  else
    if [[ -f "environment.yml" ]]; then
      echo "  Creating conda env '$ENV_NAME' from environment.yml ..."
      conda env create -n "$ENV_NAME" -f environment.yml -q
      echo "  OK: created '$ENV_NAME' from environment.yml"
    else
      echo "  Creating conda env '$ENV_NAME' (python=3.12 baseline)..."
      conda create -n "$ENV_NAME" python=3.12 -y -q
      echo "  OK: created '$ENV_NAME'"
    fi
  fi

  ACTIVATE_CMD="conda activate $ENV_NAME"
  PYTHON_CMD="conda run -n $ENV_NAME python"
  PIP_CMD="conda run -n $ENV_NAME pip"
else
  if [[ ! -d ".venv" ]]; then
    echo "  Creating venv at .venv ..."
    python3 -m venv .venv
    echo "  OK: created .venv"
  else
    echo "  OK: .venv already exists"
  fi

  ACTIVATE_CMD="source .venv/bin/activate"
  PYTHON_CMD=".venv/bin/python"
  PIP_CMD=".venv/bin/pip"
fi

# ------------------------------------------------------------------
# 3) CS690TR / zip check
# ------------------------------------------------------------------
echo "[3/6] Checking BioPM source (CS690TR)..."
BIOPM_READY=false
if [[ -d "$BIOPM_DIR/src" ]]; then
  BIOPM_READY=true
  echo "  OK: $BIOPM_DIR/src found"
elif [[ -f "$BIOPM_ZIP" ]]; then
  echo "  Found $BIOPM_ZIP. Extracting..."
  unzip -oq "$BIOPM_ZIP"
  # Normalize common extracted layouts:
  #  - CS690TR/src
  #  - CS690TR/CS690TR/src
  #  - Downloads/CS690TR/src (seen in some environments)
  if [[ -d "$BIOPM_DIR/CS690TR/src" && ! -d "$BIOPM_DIR/src" ]]; then
    mv "$BIOPM_DIR/CS690TR" _tmp_biopm && rm -rf "$BIOPM_DIR" && mv _tmp_biopm "$BIOPM_DIR"
  fi
  if [[ -d "Downloads/$BIOPM_DIR/src" && ! -d "$BIOPM_DIR/src" ]]; then
    mv "Downloads/$BIOPM_DIR" "$BIOPM_DIR"
    rmdir Downloads 2>/dev/null || true
  fi
  if [[ -d "$BIOPM_DIR/src" ]]; then
    BIOPM_READY=true
    echo "  OK: extracted $BIOPM_DIR"
  else
    echo "  WARN: extracted zip but could not locate $BIOPM_DIR/src"
  fi
else
  echo "  WARN: neither $BIOPM_DIR/ nor $BIOPM_ZIP found."
  print_layout_hint
fi

if [[ "$BIOPM_READY" = true ]]; then
  if [[ -f "$BIOPM_DIR/checkpoints/checkpoint.pt" ]]; then
    echo "  OK: checkpoint found at $BIOPM_DIR/checkpoints/checkpoint.pt"
  else
    echo "  WARN: checkpoint missing at $BIOPM_DIR/checkpoints/checkpoint.pt"
  fi
fi

# ------------------------------------------------------------------
# 4) Install dependencies (root requirements first)
# ------------------------------------------------------------------
echo "[4/6] Installing Python dependencies..."
if [[ -f "requirements.txt" ]]; then
  $PIP_CMD install -q -r requirements.txt
  echo "  OK: installed root requirements.txt"
elif [[ "$BIOPM_READY" = true && -f "$BIOPM_DIR/requirements.txt" ]]; then
  $PIP_CMD install -q -r "$BIOPM_DIR/requirements.txt"
  echo "  WARN: root requirements.txt missing, used $BIOPM_DIR/requirements.txt"
else
  echo "  WARN: no requirements file found to install from."
fi

# Supplemental analysis deps used by notebook/scripts
$PIP_CMD install -q umap-learn seaborn scikit-learn scipy h5py tqdm jupyter ipykernel || true

# Kernel install is optional
if [[ "$USE_CONDA" = true ]]; then
  if ! conda run -n "$ENV_NAME" python -m ipykernel install --user \
      --name "$ENV_NAME" --display-name "Bio-PM ($ENV_NAME)"; then
    echo "  WARN: Jupyter kernel install skipped (permission/environment)."
  fi
else
  if ! .venv/bin/python -m ipykernel install --user \
      --name "$ENV_NAME" --display-name "Bio-PM ($ENV_NAME)"; then
    echo "  WARN: Jupyter kernel install skipped (permission/environment)."
  fi
fi

echo "  OK: dependency step completed"

# ------------------------------------------------------------------
# 5) Verify imports (only if CS690TR exists)
# ------------------------------------------------------------------
echo "[5/6] Verifying imports..."
if [[ "$BIOPM_READY" = true ]]; then
  $PYTHON_CMD - <<'PY'
import sys
sys.path.insert(0, 'CS690TR')
import torch, h5py, sklearn, scipy
from src.models.biopm import BioPMModel
from src.data.preprocessing import bandpass_filter, detect_zero_crossings
print(f"  OK: core imports loaded (torch {torch.__version__})")
PY
else
  echo "  WARN: skipped import verification because CS690TR is missing."
fi

# ------------------------------------------------------------------
# 6) Summarize next steps
# ------------------------------------------------------------------
echo "[6/6] Finalizing setup..."

echo ""
echo "================================================================"
echo " Setup complete"
echo ""
echo " Activate your environment before running pipelines:"
echo "   $ACTIVATE_CMD"
echo ""
echo " Run standard pipeline:"
echo "   bash run_pipeline.sh"
echo ""
echo " Run advanced pipeline:"
echo "   bash run_pipeline_adv.sh"
echo ""
echo " Open notebook:"
echo "   jupyter notebook biopm_irb_analysis.ipynb"
echo "================================================================"
