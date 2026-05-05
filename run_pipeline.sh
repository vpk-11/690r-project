#!/usr/bin/env bash
# run_pipeline.sh — Standard pipeline: 3s windows, pad=57, original pooling
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$SCRIPT_DIR/extraction pipeline"
cd "$SCRIPT_DIR"

BIOPM_ROOT="${BIOPM_ROOT:-CS690TR}"
export BIOPM_ROOT
CHECKPOINT="$BIOPM_ROOT/checkpoints/checkpoint.pt"

echo ""
echo "================================================================"
echo " Standard Pipeline: 3s windows | pad=57 | lowpass | orig pooling"
echo "================================================================"

[[ ! -d "$BIOPM_ROOT" ]]  && echo "ERROR: BIOPM_ROOT not found" && exit 1
[[ ! -f "$CHECKPOINT" ]]  && echo "ERROR: checkpoint not found" && exit 1
[[ ! -d "data" ]]         && echo "ERROR: data/ not found"      && exit 1
mkdir -p preprocessed features

echo "[1/4] Preprocessing -> preprocessed/ ..."
python "$PIPELINE_DIR/irb_preprocess.py" --data_dir data --output preprocessed

echo "[2/4] Extracting -> features/biopm_features.npz ..."
python "$PIPELINE_DIR/irb_extract.py" \
    --preprocessed preprocessed \
    --checkpoint   "$CHECKPOINT" \
    --output       features/biopm_features.npz

echo "[3/4] Verifying ..."
python "$PIPELINE_DIR/verify_embeddings.py" --features features/biopm_features.npz

echo "[4/4] Exporting legacy schema -> features/biopm_features_legacy_schema.npz ..."
python "$PIPELINE_DIR/export_legacy_schema.py" \
    --source features/biopm_features.npz \
    --output features/biopm_features_legacy_schema.npz

echo ""
echo " Done. -> features/biopm_features_legacy_schema.npz"
