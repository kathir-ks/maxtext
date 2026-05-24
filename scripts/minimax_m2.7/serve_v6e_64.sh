#!/usr/bin/env bash
# Per-worker server launcher for the MiniMax-M2.7 inference API on TPU
# v6e-64. Runs on every one of the 16 hosts in lockstep; rank 0 binds
# uvicorn on 0.0.0.0:8000 while ranks 1..15 block in the broadcast loop.
#
# Weight load is fully distributed: each host reads only its own
# manifest.p${RANK}.json + .npy shards from /mnt/dtmpfs. Zero cross-host
# weight traffic.
#
# Drives the same engine as decode_distributed.sh (TP=8 EP=8,
# megablox=true on v6e), but the entrypoint is
# benchmarks.api_server.serve_minimax_m2_npy which delegates to
# benchmarks.api_server.maxtext_server (FastAPI).
set -euo pipefail

cd "$HOME/maxtext"
git fetch origin feature/minimax-m2.7-api-server
git reset --hard origin/feature/minimax-m2.7-api-server

source ~/venv-maxtext-py312/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export TF_CPP_MIN_LOG_LEVEL=2

DTMPFS_ROOT="${DTMPFS_ROOT:-/mnt/dtmpfs}"
NPY_DIR="${NPY_DIR:-${DTMPFS_ROOT}/minimax-m2.7-npy-distributed}"
HF_DIR="${HF_DIR:-${DTMPFS_ROOT}/minimax-m2.7-hf}"
if [ -f "$HF_DIR/tokenizer.json" ]; then
  TOK_DIR="${TOK_DIR:-$HF_DIR}"
else
  TOK_DIR="${TOK_DIR:-$HOME/minimax-m2.7-tokenizer}"
fi
JAX_CACHE_DIR="${JAX_CACHE_DIR:-${DTMPFS_ROOT}/jax_cache_v6e64_serve}"
RUN_NAME="${RUN_NAME:-minimax-m2.7-serve-v6e-64}"

mkdir -p "$JAX_CACHE_DIR"

# Serving defaults (overridable via env). Different sweet spot from
# the throughput benchmark: prefer headroom over peak tok/s.
# B=8/MTL=8192 OOMs HBM (33.32G/31.25G); B=4/MTL=4096 fits.
# Phase 4 sweep will re-tune; this is the conservative starting cell.
BATCH="${BATCH:-4}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-4096}"
PREFILL_LEN="${PREFILL_LEN:-2048}"
ICI_TENSOR="${ICI_TENSOR:-8}"
ICI_EXPERT="${ICI_EXPERT:-8}"
WEIGHT_DTYPE="${WEIGHT_DTYPE:-bfloat16}"
QUANT="${QUANT:-}"
KV_QUANT="${KV_QUANT:-true}"
KV_QUANT_DTYPE="${KV_QUANT_DTYPE:-int4}"

export MAXTEXT_SERVER_PORT="${MAXTEXT_SERVER_PORT:-8000}"
export MAXTEXT_SERVER_HOST="${MAXTEXT_SERVER_HOST:-0.0.0.0}"
export MAXTEXT_API_KEY="${MAXTEXT_API_KEY:-}"

LOG="$HOME/serve_v6e_64.log"
exec > >(tee -a "$LOG") 2>&1
echo "[serve-v6e-64] host=$(hostname) start=$(date -u +%FT%TZ) port=$MAXTEXT_SERVER_PORT"
echo "[serve-v6e-64] NPY_DIR=$NPY_DIR TOK_DIR=$TOK_DIR JAX_CACHE_DIR=$JAX_CACHE_DIR"
echo "[serve-v6e-64] BATCH=$BATCH MAX_TARGET_LENGTH=$MAX_TARGET_LENGTH PREFILL_LEN=$PREFILL_LEN"
echo "[serve-v6e-64] ICI_TENSOR=$ICI_TENSOR ICI_EXPERT=$ICI_EXPERT KV_QUANT=$KV_QUANT/$KV_QUANT_DTYPE"

ls -1 "$NPY_DIR"/manifest.p*.json | head -1 >/dev/null \
  || { echo "missing distributed manifest on this worker"; exit 1; }
[[ -f "$TOK_DIR/tokenizer.json" ]] \
  || { echo "missing tokenizer.json in $TOK_DIR"; exit 1; }

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
  quantize_kvcache="$KV_QUANT" \
  kv_quant_dtype="$KV_QUANT_DTYPE" \
  weight_dtype="$WEIGHT_DTYPE" \
  attention=dot_product \
  megablox=true \
  sparse_matmul=false

echo "[serve-v6e-64] done host=$(hostname) finish=$(date -u +%FT%TZ)"
