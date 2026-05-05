#!/usr/bin/env bash
# run_extraction_pipeline_alt.sh
#
# Alt Bio-PM IRB pipeline using grouped 9-second blocks.
# Produces files with _alt suffix so standard outputs are never touched.
#
# Differences from run_extraction_pipeline.sh:
#   - Preprocessing uses irb_preprocess_alt.py (GROUP_SIZE=3, pad_size=171)
#   - HDF5 files go to preprocessed_alt/
#   - Features go to features/biopm_features_alt.npz
#   - Legacy schema goes to features/biopm_features_legacy_schema_alt.npz
#
# Usage (from repo root):
#   bash run_extraction_pipeline_alt.sh
#
# Expected fill rate improvement:
#   Standard: ~33% (18.6 MEs / 57 slots)
#   Alt:      ~96% (55+ MEs / 57*3 slots)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
PIPELINE_DIR="$REPO_ROOT/extraction pipeline"
cd "$REPO_ROOT"

BIOPM_ROOT="${BIOPM_ROOT:-CS690TR}"
export BIOPM_ROOT

CHECKPOINT="$BIOPM_ROOT/checkpoints/checkpoint.pt"
DATA_DIR="data"
PREPROCESSED_DIR="preprocessed_alt"
FEATURES="features/biopm_features_alt.npz"
LEGACY_FEATURES="features/biopm_features_legacy_schema_alt.npz"

# Validate environment
if [[ ! -d "$BIOPM_ROOT" ]]; then
    echo "ERROR: BIOPM_ROOT not found: $BIOPM_ROOT"
    echo "Set BIOPM_ROOT to your CS690TR directory and rerun."
    exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT"
    exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: data directory not found: $DATA_DIR"
    exit 1
fi

mkdir -p "$PREPROCESSED_DIR" "$(dirname "$FEATURES")"

echo ""
echo "================================================================"
echo " Bio-PM IRB Alt Pipeline (9-second grouped blocks)"
echo "================================================================"
echo ""
echo "Repo root        : $REPO_ROOT"
echo "BIOPM_ROOT       : $BIOPM_ROOT"
echo "Data dir         : $DATA_DIR"
echo "Output HDF5      : $PREPROCESSED_DIR"
echo "Features         : $FEATURES"
echo "Legacy NPZ       : $LEGACY_FEATURES"
echo ""
echo "Key difference   : 3 windows grouped into 9s blocks"
echo "Expected fill    : ~96% vs ~33% in standard pipeline"
echo ""

echo "[1/4] Alt preprocessing: group windows into 9s blocks -> HDF5 ..."
python "$PIPELINE_DIR/irb_preprocess_alt.py" \
    --data_dir "$DATA_DIR" \
    --output   "$PREPROCESSED_DIR"

echo ""
echo "[2/4] Extract Bio-PM embeddings (irb_extract.py unchanged) ..."
python "$PIPELINE_DIR/irb_extract.py" \
    --preprocessed "$PREPROCESSED_DIR" \
    --checkpoint   "$CHECKPOINT" \
    --output       "$FEATURES"

echo ""
echo "[3/4] Verify alt features ..."
python "$PIPELINE_DIR/verify_embeddings.py" --features "$FEATURES"

echo ""
echo "[4/4] Export legacy-compatible visit schema ..."
python "$PIPELINE_DIR/export_legacy_schema.py" \
    --source "$FEATURES" \
    --output "$LEGACY_FEATURES"

echo ""
echo "================================================================"
echo " DONE. Open biopm_irb_pipeline_alt.ipynb for analysis."
echo ""
echo " Files produced:"
echo "   $PREPROCESSED_DIR/    HDF5 files (9s blocks, pad_size=171)"
echo "   $FEATURES"
echo "   $LEGACY_FEATURES"
echo "================================================================"
