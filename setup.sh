#!/bin/bash
# setup.sh -- one-time setup before running biopm_irb_pipeline.ipynb
# Usage: bash setup.sh
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; }

echo ""
echo "=============================================="
echo " Bio-PM IRB Pipeline Setup"
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
ok "conda env 'biopm' ready"

# --------------------------------------------------
# 2. Register Jupyter kernel
# --------------------------------------------------
echo "Registering 'biopm-690r' Jupyter kernel ..."
conda run -n biopm-690r python -m ipykernel install --user \
    --name biopm-690r --display-name "biopm-690r (CS690R)" -q 2>/dev/null
ok "Jupyter kernel 'biopm-690r' registered"

# --------------------------------------------------
# 3. Create output directories
# --------------------------------------------------
mkdir -p preprocessed features results/figures results/splits
ok "Output directories created"

# --------------------------------------------------
# 4. Check data files
# --------------------------------------------------
echo ""
echo "Checking data files ..."

if [ -f "data/clinical_scores.npz" ]; then
    ok "data/clinical_scores.npz found"
else
    fail "data/clinical_scores.npz NOT FOUND -- copy it into data/ before running notebook"
fi

if [ -f "data/windows.npz" ]; then
    SIZE=$(du -sh data/windows.npz | cut -f1)
    ok "data/windows.npz found ($SIZE)"
else
    fail "data/windows.npz NOT FOUND -- copy it into data/ before running notebook"
fi

# --------------------------------------------------
# 5. Check BIOPM_ROOT
# --------------------------------------------------
echo ""
echo "Checking Bio-PM repo ..."

if [ -n "$BIOPM_ROOT" ] && [ -d "$BIOPM_ROOT" ]; then
    ok "BIOPM_ROOT=$BIOPM_ROOT"
    if [ -f "$BIOPM_ROOT/checkpoints/checkpoint.pt" ]; then
        ok "checkpoint.pt found"
    else
        warn "checkpoint.pt not found at \$BIOPM_ROOT/checkpoints/checkpoint.pt"
    fi
else
    warn "BIOPM_ROOT not set or not a valid directory"
    # Try to find CS690TR in common locations
    FOUND=$(find . Downloads ~ -maxdepth 4 -name "CS690TR" -type d 2>/dev/null | head -1)
    if [ -n "$FOUND" ]; then
        FOUND_ABS=$(cd "$FOUND" && pwd)
        warn "Found candidate: $FOUND_ABS"
        echo "     Set it in the notebook config cell: BIOPM_ROOT = \"$FOUND_ABS\""
    else
        warn "Could not auto-detect CS690TR. Extract biopm690r.zip and set BIOPM_ROOT."
    fi
fi

# --------------------------------------------------
# 6. Print checklist
# --------------------------------------------------
echo ""
echo "=============================================="
echo " Setup complete. Before clicking Run All:"
echo "=============================================="
echo ""
echo "  1. Open biopm_irb_pipeline.ipynb in Jupyter"
echo "  2. Select kernel: biopm-690r (CS690R)"
echo "  3. Edit the CONFIG cell (Cell 1):"
echo "       BIOPM_ROOT = \"/path/to/CS690TR\""
echo "       (all other paths default to current directory)"
echo ""
echo "  Then: Run All"
echo ""
echo "  Expected runtime:"
echo "    Inspect    ~2 min  (loads windows.npz)"
echo "    Debug      ~1 min"
echo "    Preprocess ~10-20 min"
echo "    Extract    ~5-15 min (CPU)"
echo "    Verify     <1 min"
echo "    Analyze    ~2-3 min (UMAP + LOSO LR)"
echo ""
echo "  NOTE: windows.npz is loaded separately in Inspect,"
echo "  Debug, and Preprocess cells -- each load takes ~1-2 min."
echo ""
echo "=============================================="
echo ""
