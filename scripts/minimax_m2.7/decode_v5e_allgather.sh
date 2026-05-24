#!/usr/bin/env bash
# Decode launcher that selects the new v5e_allgather MoE path.
#
# Same scaffolding as decode_distributed.sh but adds
# `routed_moe_path=v5e_allgather` so the new code path is used.
# Use this on v5e-64 once Phase 3 + 4 land.
set -euo pipefail

cd "$HOME/maxtext"
git fetch origin feature/minimax-m2.7-v5e-pallas
git reset --hard origin/feature/minimax-m2.7-v5e-pallas

source ~/venv-maxtext-py312/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export TF_CPP_MIN_LOG_LEVEL=2

NPY_DIR=${NPY_DIR:-/mnt/dtmpfs/minimax-m2.7-npy-distributed}
HF_DIR=${HF_DIR:-/mnt/dtmpfs/minimax-m2.7-hf}
if [ -f "$HF_DIR/tokenizer.json" ]; then
  TOK_DIR=${TOK_DIR:-$HF_DIR}
else
  TOK_DIR=${TOK_DIR:-$HOME/minimax-m2.7-tokenizer}
fi
JAX_CACHE_DIR=${JAX_CACHE_DIR:-/mnt/dtmpfs/jax_cache_v5e64_allgather}
mkdir -p "$JAX_CACHE_DIR"
LOG=$HOME/decode_v5e_allgather.log
exec > >(tee -a "$LOG") 2>&1

echo "[decode-v5e-allgather] host=$(hostname) start=$(date -u +%FT%TZ)"
ls -1 "$NPY_DIR"/manifest.p*.json | head -1 >/dev/null \
  || { echo "missing distributed manifest on this worker"; exit 1; }

PROMPT="${PROMPT:-The capital of France is}"
ICI_TENSOR="${ICI_TENSOR:-8}"
ICI_EXPERT="${ICI_EXPERT:-8}"
BATCH="${BATCH:-1}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-64}"
PREFILL_LEN=$((MAX_TARGET_LENGTH/2))
QUANT="${QUANT:-}"
KV_QUANT="${KV_QUANT:-false}"
KV_QUANT_DTYPE="${KV_QUANT_DTYPE:-int8}"
CAPACITY="${CAPACITY:-1.5}"
WEIGHT_DTYPE="${WEIGHT_DTYPE:-bfloat16}"

python -m maxtext.inference.decode_minimax_m2_npy \
  --npy_dir "$NPY_DIR" \
  src/maxtext/configs/base.yml \
  model_name=minimax-m2.7 \
  tokenizer_path="$TOK_DIR" \
  tokenizer_type=huggingface \
  run_name=minimax-m2.7-v5e-allgather \
  jax_cache_dir="$JAX_CACHE_DIR" \
  per_device_batch_size="$BATCH" \
  max_target_length="$MAX_TARGET_LENGTH" \
  max_prefill_predict_length="$PREFILL_LEN" \
  ici_tensor_parallelism="$ICI_TENSOR" \
  ici_expert_parallelism="$ICI_EXPERT" \
  scan_layers=true \
  quantization="$QUANT" \
  quantize_kvcache="$KV_QUANT" \
  kv_quant_dtype="$KV_QUANT_DTYPE" \
  weight_dtype="$WEIGHT_DTYPE" \
  attention=dot_product \
  routed_moe_path=v5e_allgather \
  capacity_factor="$CAPACITY" \
  prompt="$PROMPT"

echo "[decode-v5e-allgather] done host=$(hostname) finish=$(date -u +%FT%TZ)"
