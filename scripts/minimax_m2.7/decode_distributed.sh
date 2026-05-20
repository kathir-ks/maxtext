#!/usr/bin/env bash
# Decode launcher using the per-host distributed shards.
set -euo pipefail

cd "$HOME/maxtext"
git fetch origin feature/minimax-m2.7
git reset --hard origin/feature/minimax-m2.7

source ~/venv-maxtext-py312/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export TF_CPP_MIN_LOG_LEVEL=2

NPY_DIR=${NPY_DIR:-/mnt/dtmpfs/minimax-m2.7-npy-distributed}
HF_DIR=${HF_DIR:-/mnt/dtmpfs/minimax-m2.7-hf}
# Convert co-locates tokenizer with HF metadata; fall back to legacy $HOME path.
if [ -f "$HF_DIR/tokenizer.json" ]; then
  TOK_DIR=${TOK_DIR:-$HF_DIR}
else
  TOK_DIR=${TOK_DIR:-$HOME/minimax-m2.7-tokenizer}
fi
JAX_CACHE_DIR=${JAX_CACHE_DIR:-/mnt/dtmpfs/jax_cache_v5e64}
mkdir -p "$JAX_CACHE_DIR"
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
# Optimization knobs (override via env). On v5e the routing-ragged path is
# unavailable, so we hold megablox=false sparse_matmul=false by default.
QUANT="${QUANT:-}"               # '', 'int8', 'int4', 'intmp'
KV_QUANT="${KV_QUANT:-false}"    # 'true' to enable kv-cache quantization
KV_QUANT_DTYPE="${KV_QUANT_DTYPE:-int8}"
CAPACITY="${CAPACITY:-2.0}"
WEIGHT_DTYPE="${WEIGHT_DTYPE:-bfloat16}"

python -m maxtext.inference.decode_minimax_m2_npy \
  --npy_dir "$NPY_DIR" \
  src/maxtext/configs/base.yml \
  model_name=minimax-m2.7 \
  tokenizer_path="$TOK_DIR" \
  tokenizer_type=huggingface \
  run_name=minimax-m2.7-decode-v5e64-dist \
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
  megablox=false \
  sparse_matmul=false \
  capacity_factor="$CAPACITY" \
  prompt="$PROMPT"

echo "[decode-dist] done host=$(hostname) finish=$(date -u +%FT%TZ)"
