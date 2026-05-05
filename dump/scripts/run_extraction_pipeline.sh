#!/usr/bin/env bash
# Bio-PM IRB pipeline: raw data -> HDF5 -> embeddings -> verification -> legacy export
# Analysis and visual exploration run in biopm_irb_pipeline.ipynb.
# Usage (from repo root): bash run_extraction_pipeline.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
PIPELINE_DIR="$REPO_ROOT/extraction pipeline"
cd "$REPO_ROOT"

# Allow override via environment variable; default to this repo's CS690TR.
BIOPM_ROOT="${BIOPM_ROOT:-CS690TR}"
export BIOPM_ROOT

CHECKPOINT="$BIOPM_ROOT/checkpoints/checkpoint.pt"
DATA_DIR="data"
PREPROCESSED_DIR="preprocessed"
FEATURES="features/biopm_features.npz"
LEGACY_FEATURES="features/biopm_features_legacy_schema.npz"

if [[ ! -d "$BIOPM_ROOT" ]]; then
    echo "ERROR: BIOPM_ROOT not found: $BIOPM_ROOT"
    echo "Set BIOPM_ROOT to your CS690TR directory and rerun."
    exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT"
    echo "Expected at \$BIOPM_ROOT/checkpoints/checkpoint.pt"
    exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: data directory not found: $DATA_DIR"
    exit 1
fi

mkdir -p "$PREPROCESSED_DIR" "$(dirname "$FEATURES")"

echo ""
echo "================================================================"
echo " Bio-PM IRB Pipeline"
echo "================================================================"
echo ""
echo "Repo root   : $REPO_ROOT"
echo "BIOPM_ROOT  : $BIOPM_ROOT"
echo "Data dir    : $DATA_DIR"
echo "Output HDF5 : $PREPROCESSED_DIR"
echo "Features    : $FEATURES"
echo "Legacy NPZ  : $LEGACY_FEATURES"

echo ""
echo "[1/4] Preprocess raw windows -> HDF5 ..."
python "$PIPELINE_DIR/irb_preprocess.py" \
    --data_dir "$DATA_DIR" \
    --output "$PREPROCESSED_DIR"

echo ""
echo "[2/4] Extract Bio-PM embeddings ..."
python "$PIPELINE_DIR/irb_extract.py" \
    --preprocessed "$PREPROCESSED_DIR" \
    --checkpoint "$CHECKPOINT" \
    --output "$FEATURES"

echo ""
echo "[3/4] Verify features ..."
python "$PIPELINE_DIR/verify_embeddings.py" --features "$FEATURES"

echo ""
echo "[4/4] Export legacy-compatible visit schema ..."
python "$PIPELINE_DIR/export_legacy_schema.py" \
    --source "$FEATURES" \
    --output "$LEGACY_FEATURES"

echo ""
echo "================================================================"
echo " DONE. Open biopm_irb_pipeline.ipynb for analysis."
echo "================================================================"
