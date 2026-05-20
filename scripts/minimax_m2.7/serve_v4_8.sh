#!/usr/bin/env bash
# Launch the OpenAI-compatible HTTP server for MiniMax-M2.7 INT4 on a single
# v4-8. Mirrors decode_v4_8.sh's pyconfig overrides but goes through
# serve_minimax_m2_npy, which patches MaxEngine.load_params to read .npy
# leaves and then delegates to benchmarks.api_server.maxtext_server (FastAPI).
set -euo pipefail

DTMPFS_ROOT="${DTMPFS_ROOT:-/mnt/dtmpfs}"
NPY_DIR="${NPY_DIR:-${DTMPFS_ROOT}/minimax-m2.7-npy}"
HF_DIR="${HF_DIR:-${DTMPFS_ROOT}/minimax-m2.7-hf}"
TOK_DIR="${TOK_DIR:-$HF_DIR}"
JAX_CACHE_DIR="${JAX_CACHE_DIR:-${DTMPFS_ROOT}/jax_cache_v4_8_serve}"
RUN_NAME="${RUN_NAME:-minimax-m2.7-serve-v4-8-int4}"

QUANT_CFG_PATH="${QUANT_CFG_PATH:-src/maxtext/configs/quantization/int4_weight_only.json}"
QUANT="${QUANT:-intmp}"
KV_QUANT="${KV_QUANT:-true}"
KV_QUANT_DTYPE="${KV_QUANT_DTYPE:-int8}"

BATCH="${BATCH:-1}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-65536}"
PREFILL_LEN="${PREFILL_LEN:-32768}"
ICI_TENSOR="${ICI_TENSOR:-4}"
ICI_EXPERT="${ICI_EXPERT:-1}"
WEIGHT_DTYPE="${WEIGHT_DTYPE:-bfloat16}"
PORT="${PORT:-8765}"

cd "$HOME/maxtext"
source ~/venv-maxtext-py312/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export TF_CPP_MIN_LOG_LEVEL=2
export MAXTEXT_SERVER_PORT="$PORT"
mkdir -p "$JAX_CACHE_DIR"

LOG="$HOME/serve_v4_8.log"
exec > >(tee -a "$LOG") 2>&1
echo "[serve-v4-8] host=$(hostname) start=$(date -u +%FT%TZ) port=$PORT"

[[ -d "$NPY_DIR" ]] || { echo "missing $NPY_DIR"; exit 1; }
[[ -f "$TOK_DIR/tokenizer.json" ]] || { echo "missing tokenizer.json in $TOK_DIR"; exit 1; }
[[ -f "$QUANT_CFG_PATH" ]] || { echo "missing $QUANT_CFG_PATH"; exit 1; }

python -m benchmarks.api_server.serve_minimax_m2_npy \
  --npy_dir "$NPY_DIR" \
  src/maxtext/configs/base.yml \
  model_name=minimax-m2.7 \
  tokenizer_path="$TOK_DIR" \
  tokenizer_type=huggingface \
  run_name="$RUN_NAME" \
  jax_cache_dir="$JAX_CACHE_DIR" \
  per_device_batch_size="$BATCH" \
  max_target_length="$MAX_TARGET_LENGTH" \
  max_prefill_predict_length="$PREFILL_LEN" \
  ici_tensor_parallelism="$ICI_TENSOR" \
  ici_expert_parallelism="$ICI_EXPERT" \
  scan_layers=true \
  quantization="$QUANT" \
  quant_cfg_path="$QUANT_CFG_PATH" \
  quantize_kvcache="$KV_QUANT" \
  kv_quant_dtype="$KV_QUANT_DTYPE" \
  weight_dtype="$WEIGHT_DTYPE" \
  attention=dot_product \
  megablox=false \
  sparse_matmul=false
