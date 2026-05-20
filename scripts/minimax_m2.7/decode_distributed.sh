#!/usr/bin/env bash
# Decode launcher using the per-host distributed shards.
set -euo pipefail

cd "$HOME/maxtext"
git fetch origin feature/minimax-m2.7
git reset --hard origin/feature/minimax-m2.7

source ~/venv-maxtext-py312/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export TF_CPP_MIN_LOG_LEVEL=2

NPY_DIR=/mnt/dtmpfs/minimax-m2.7-npy-distributed
TOK_DIR=$HOME/minimax-m2.7-tokenizer
LOG=$HOME/decode_dist.log
exec > >(tee -a "$LOG") 2>&1

echo "[decode-dist] host=$(hostname) start=$(date -u +%FT%TZ)"
ls -1 "$NPY_DIR"/manifest.p*.json | head -1 >/dev/null \
  || { echo "missing distributed manifest on this worker"; exit 1; }

PROMPT="${PROMPT:-The capital of France is}"
ICI_TENSOR="${ICI_TENSOR:-8}"
ICI_EXPERT="${ICI_EXPERT:-8}"
BATCH="${BATCH:-1}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-64}"
PREFILL_LEN=$((MAX_TARGET_LENGTH/2))

python -m maxtext.inference.decode_minimax_m2_npy \
  --npy_dir "$NPY_DIR" \
  src/maxtext/configs/base.yml \
  model_name=minimax-m2.7 \
  tokenizer_path="$TOK_DIR" \
  tokenizer_type=huggingface \
  run_name=minimax-m2.7-decode-v6e64-dist \
  per_device_batch_size="$BATCH" \
  max_target_length="$MAX_TARGET_LENGTH" \
  max_prefill_predict_length="$PREFILL_LEN" \
  ici_tensor_parallelism="$ICI_TENSOR" \
  ici_expert_parallelism="$ICI_EXPERT" \
  scan_layers=true \
  quantization='' \
  weight_dtype=bfloat16 \
  attention=dot_product \
  megablox=false \
  sparse_matmul=false \
  capacity_factor=2.0 \
  prompt="$PROMPT"

echo "[decode-dist] done host=$(hostname) finish=$(date -u +%FT%TZ)"
