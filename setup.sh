#!/usr/bin/env bash
# =============================================================================
# setup.sh — CS690R Biomarker Project
# =============================================================================
# Downloads the PADS dataset, extracts Bio-PM model code,
# and creates the output directory structure.
#
# Python environment is your choice — set it up before running pip installs:
#   conda:  conda create -n biopm python=3.11 && conda activate biopm
#   venv:   python -m venv .venv && source .venv/bin/activate
#   uv:     uv venv && source .venv/bin/activate
#
# Then install:
#   pip install -r CS690TR/requirements.txt
#   pip install umap-learn seaborn scikit-learn jupyter
#
# Usage:
#   chmod +x setup.sh && ./setup.sh
# =============================================================================

set -e

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC}    $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
step() { echo -e "\n${GREEN}==>${NC} $1"; }


# =============================================================================
# 1 — Extract Bio-PM model code
# =============================================================================
step "Extracting Bio-PM model code"

[ -f "biopm690r.zip" ] || err "biopm690r.zip not found in current directory."

if [ -d "CS690TR" ]; then
    warn "CS690TR/ already exists — skipping."
else
    unzip -q biopm690r.zip
    [ -d "Downloads/CS690TR" ] && mv Downloads/CS690TR ./CS690TR && rmdir Downloads 2>/dev/null || true
    [ -d "CS690TR" ] || err "CS690TR not found after unzip — check biopm690r.zip contents."
    ok "CS690TR/ ready"
fi


# =============================================================================
# 2 — Download PADS dataset
# =============================================================================
step "Downloading PADS dataset (~735 MB)"

PADS_DIR="pads_data"
PADS_URL="https://physionet.org/files/parkinsons-disease-smartwatch/1.0.0/"
PADS_S3="s3://physionet-open/parkinsons-disease-smartwatch/1.0.0/"

if [ -f "$PADS_DIR/preprocessed/file_list.csv" ]; then
    warn "PADS dataset already at $PADS_DIR/ — skipping."
else
    mkdir -p "$PADS_DIR"

    if command -v wget &>/dev/null; then
        echo "  Using wget..."
        wget -q -r -N -c -np \
             --directory-prefix="$PADS_DIR" \
             --cut-dirs=4 --no-host-directories \
             "$PADS_URL"
        ok "Download complete"

    elif command -v aws &>/dev/null; then
        echo "  Using aws s3 sync..."
        aws s3 sync --no-sign-request "$PADS_S3" "$PADS_DIR/"
        ok "Sync complete"

    else
        warn "wget and aws not found. Download manually:"
        echo ""
        echo "    wget -r -N -c -np --directory-prefix=pads_data --cut-dirs=4 --no-host-directories $PADS_URL"
        echo "    OR: aws s3 sync --no-sign-request $PADS_S3 pads_data/"
        echo "    OR: download ZIP from https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/"
        echo "        and extract into pads_data/"
    fi
fi


# =============================================================================
# 3 — Output directories
# =============================================================================
step "Creating output directories"
mkdir -p features results/figures results/metrics
ok "features/  results/figures/  results/metrics/"


# =============================================================================
# Done
# =============================================================================
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN} Done. Now set up your Python environment:${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "  conda:  conda create -n biopm python=3.11 && conda activate biopm"
echo "  venv:   python -m venv .venv && source .venv/bin/activate"
echo "  uv:     uv venv && source .venv/bin/activate"
echo ""
echo "  pip install -r CS690TR/requirements.txt"
echo "  pip install umap-learn seaborn scikit-learn jupyter"
echo ""
echo "  jupyter notebook"
echo ""
