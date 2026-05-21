#!/usr/bin/env bash
# Per-worker launcher for bench_steps_minimax_m2_npy. Runs prefill + N
# generate steps with explicit time.monotonic() around each step,
# dumping a JSON with step_ms percentiles and pod-wide tok/s.
#
# Env vars:
#   QUANT, KV_QUANT, KV_QUANT_DTYPE, BATCH, CAPACITY, WEIGHT_DTYPE,
#   MAX_TARGET_LENGTH, PREFILL_LEN, WARMUP, MEASURE, CELL_ID
set -euo pipefail

cd "$HOME/maxtext"
git fetch origin feature/minimax-m2.7
git reset --hard origin/feature/minimax-m2.7

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
JAX_CACHE_DIR=${JAX_CACHE_DIR:-/mnt/dtmpfs/jax_cache_v5e64}
mkdir -p "$JAX_CACHE_DIR"

QUANT="${QUANT:-}"
KV_QUANT="${KV_QUANT:-false}"
KV_QUANT_DTYPE="${KV_QUANT_DTYPE:-int8}"
BATCH="${BATCH:-1}"
CAPACITY="${CAPACITY:-2.0}"
WEIGHT_DTYPE="${WEIGHT_DTYPE:-bfloat16}"
PREFILL_LEN="${PREFILL_LEN:-32}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-256}"
WARMUP="${WARMUP:-4}"
MEASURE="${MEASURE:-32}"
ICI_TENSOR="${ICI_TENSOR:-8}"
ICI_EXPERT="${ICI_EXPERT:-8}"

CELL_ID="${CELL_ID:-q${QUANT:-bf16}_kv${KV_QUANT}_b${BATCH}_cf${CAPACITY}}"
BENCH_JSON=$HOME/bench_steps.${CELL_ID}.json
LOG=$HOME/bench_steps.log
exec > >(tee -a "$LOG") 2>&1

echo "[bench-steps] host=$(hostname) cell=$CELL_ID start=$(date -u +%FT%TZ)"

python -m maxtext.inference.bench_steps_minimax_m2_npy \
  --npy_dir "$NPY_DIR" \
  --out_json "$BENCH_JSON" \
  --warmup_steps "$WARMUP" \
  --measure_steps "$MEASURE" \
  src/maxtext/configs/base.yml \
  model_name=minimax-m2.7 \
  tokenizer_path="$TOK_DIR" \
  tokenizer_type=huggingface \
  run_name=minimax-m2.7-bench-${CELL_ID} \
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
  capacity_factor="$CAPACITY"

echo "[bench-steps] done cell=$CELL_ID finish=$(date -u +%FT%TZ)"
