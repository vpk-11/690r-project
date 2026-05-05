#!/bin/bash
# setup.sh -- environment + repository preflight checks for Bio-PM IRB pipelines
# Usage: bash setup.sh
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; }

echo ""
echo "=============================================="
echo " Bio-PM IRB Pipeline Setup + Preflight"
echo "=============================================="
echo ""

# --------------------------------------------------
# 1. Conda env
# --------------------------------------------------
if ! command -v conda &>/dev/null; then
    fail "conda not found. Install Miniconda/Anaconda first."
    exit 1
fi

if conda env list | grep -q "^biopm-690r "; then
    warn "conda env 'biopm-690r' already exists -- updating packages"
    conda env update -n biopm-690r -f environment.yml --prune -q
else
    echo "Creating conda env 'biopm-690r' (Python 3.12) ..."
    conda env create -f environment.yml -q
fi
ok "conda env 'biopm-690r' ready"

# --------------------------------------------------
# 2. Register Jupyter kernel
# --------------------------------------------------
echo "Registering 'biopm-690r' Jupyter kernel ..."
conda run -n biopm-690r python -m ipykernel install --user \
    --name biopm-690r --display-name "biopm-690r (CS690R)" -q 2>/dev/null
ok "Jupyter kernel 'biopm-690r' registered"

# --------------------------------------------------
# 3. Repository structure checks
# --------------------------------------------------
echo ""
echo "Checking repository layout ..."

if [ ! -d "CS690TR" ]; then
    fail "CS690TR/ folder not found in repo root: $(pwd)"
    echo "Expected: ./CS690TR/checkpoints/checkpoint.pt and ./CS690TR/src/models/biopm.py"
    exit 1
fi
ok "CS690TR folder found at ./CS690TR"

if [ ! -f "CS690TR/checkpoints/checkpoint.pt" ]; then
    fail "Missing checkpoint: CS690TR/checkpoints/checkpoint.pt"
    exit 1
fi
ok "checkpoint.pt found"

for req in \
  "CS690TR/src/models/biopm.py" \
  "CS690TR/src/data/preprocessing.py" \
  "extraction pipeline/irb_preprocess.py" \
  "extraction pipeline/irb_extract.py" \
  "extraction pipeline/irb_preprocess_v3.py" \
  "extraction pipeline/irb_extract_v3.py" \
  "extraction pipeline/export_legacy_schema_v3.py" \
  "run_extraction_pipeline.sh" \
  "run_extraction_pipeline_v3.sh" \
  "biopm_irb_pipeline.ipynb" \
  "biopm_irb_pipeline_v3.ipynb"
do
    if [ ! -e "$req" ]; then
        fail "Missing required file: $req"
        exit 1
    fi
done
ok "Required pipeline scripts/notebooks are present"

if [ -d "CS690TR/CS690TR/src" ]; then
    warn "Nested duplicate detected: CS690TR/CS690TR/src"
    warn "Use ./CS690TR as canonical BIOPM_ROOT for all scripts."
fi

# --------------------------------------------------
# 4. Create output directories
# --------------------------------------------------
mkdir -p preprocessed preprocessed_alt preprocessed_v3 features results/figures results/splits
ok "Output directories ensured"

# --------------------------------------------------
# 5. Check data files
# --------------------------------------------------
echo ""
echo "Checking data files ..."

if [ -f "data/clinical_scores.npz" ]; then
    ok "data/clinical_scores.npz found"
else
    fail "data/clinical_scores.npz NOT FOUND -- copy it into ./data before running pipelines"
    exit 1
fi

if [ -f "data/windows.npz" ]; then
    SIZE=$(du -sh data/windows.npz | cut -f1)
    ok "data/windows.npz found ($SIZE)"
else
    fail "data/windows.npz NOT FOUND -- copy it into ./data before running pipelines"
    exit 1
fi

# --------------------------------------------------
# 6. Check BIOPM_ROOT
# --------------------------------------------------
echo ""
echo "Checking Bio-PM repo ..."

if [ -n "${BIOPM_ROOT:-}" ] && [ -d "$BIOPM_ROOT" ]; then
    ok "BIOPM_ROOT=$BIOPM_ROOT"
    if [ -f "$BIOPM_ROOT/checkpoints/checkpoint.pt" ]; then
        ok "checkpoint.pt found"
    else
        warn "checkpoint.pt not found at \$BIOPM_ROOT/checkpoints/checkpoint.pt"
    fi
else
    warn "BIOPM_ROOT not set. Recommended default for this repo: export BIOPM_ROOT=CS690TR"
fi

# --------------------------------------------------
# 7. Quick Python import and dataset key sanity check
# --------------------------------------------------
echo ""
echo "Running quick sanity check in biopm-690r env ..."
conda run -n biopm-690r python - <<'PY'
import os
import numpy as np
import sys
sys.path.insert(0, "CS690TR")
from src.models.biopm import BioPMModel
from src.data.preprocessing import detect_zero_crossings

clin = np.load("data/clinical_scores.npz", allow_pickle=True)
wins = np.load("data/windows.npz", allow_pickle=True)
assert "clinical_scores" in clin, "clinical_scores key missing in data/clinical_scores.npz"
assert "windows" in wins, "windows key missing in data/windows.npz"
print("Python sanity OK: imports + NPZ keys present")
PY
ok "Python sanity check passed"

# --------------------------------------------------
# 8. Print checklist
# --------------------------------------------------
echo ""
echo "=============================================="
echo " Setup complete. Recommended next steps:"
echo "=============================================="
echo ""
echo "  1. Export BIOPM_ROOT for this shell:"
echo "       export BIOPM_ROOT=CS690TR"
echo "  2. Run V3 pipeline:"
echo "       bash run_extraction_pipeline_v3.sh"
echo "  3. Open biopm_irb_pipeline_v3.ipynb in Jupyter"
echo "  4. Select kernel: biopm-690r (CS690R)"
echo ""
echo "  If you need legacy/standard comparisons:"
echo "    - standard: bash run_extraction_pipeline.sh"
echo "    - alt     : bash run_extraction_pipeline_alt.sh"
echo ""
echo "  Original notebook path is still supported:"
echo "    biopm_irb_pipeline.ipynb"
echo ""
echo "  Historical (old) instructions removed to avoid mismatch."
echo ""
echo "=============================================="
echo ""
