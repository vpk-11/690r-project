#!/usr/bin/env bash
# run_pipeline_adv.sh — ADV pipeline: 3s windows, pad=192, NaN-mask valid-token pooling
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$SCRIPT_DIR/extraction pipeline"
cd "$SCRIPT_DIR"

BIOPM_ROOT="${BIOPM_ROOT:-CS690TR}"
export BIOPM_ROOT
CHECKPOINT="$BIOPM_ROOT/checkpoints/checkpoint.pt"
PAD_SIZE="${PAD_SIZE:-192}"

if [[ -z "${DEVICE:-}" ]]; then
  DEVICE="$(python - <<'PY'
try:
    import torch
    if torch.cuda.is_available():
        print("cuda")
    elif torch.backends.mps.is_available():
        print("mps")
    else:
        print("cpu")
except Exception:
    print("cpu")
PY
)"
fi

echo ""
echo "================================================================"
echo " ADV Pipeline: 3s windows | pad=$PAD_SIZE | NaN-mask valid pooling"
echo " Device: $DEVICE"
echo "================================================================"

[[ ! -d "$BIOPM_ROOT" ]]  && echo "ERROR: BIOPM_ROOT not found" && exit 1
[[ ! -f "$CHECKPOINT" ]]  && echo "ERROR: checkpoint not found" && exit 1
[[ ! -d "data" ]]         && echo "ERROR: data/ not found"      && exit 1
mkdir -p preprocessed_adv features

echo "[1/4] Preprocessing -> preprocessed_adv/ ..."
python "$PIPELINE_DIR/irb_preprocess_adv.py" --data_dir data --output preprocessed_adv --pad_size "$PAD_SIZE"

echo "[2/4] Extracting -> features/biopm_features_adv.npz ..."
python "$PIPELINE_DIR/irb_extract_adv.py" \
    --preprocessed preprocessed_adv \
    --checkpoint   "$CHECKPOINT" \
    --output       features/biopm_features_adv.npz \
    --device       "cpu"

echo "[3/4] Verifying ..."
python "$PIPELINE_DIR/verify_embeddings.py" --features features/biopm_features_adv.npz

echo "[4/4] Exporting legacy schema -> features/biopm_features_legacy_schema_adv.npz ..."
python "$PIPELINE_DIR/export_legacy_schema_adv.py" \
    --source features/biopm_features_adv.npz \
    --output features/biopm_features_legacy_schema_adv.npz

echo ""
echo " Done. -> features/biopm_features_legacy_schema_adv.npz"
