#!/usr/bin/env bash
# Run MiniMax-M2.7 INT4 decode on a single v4-8 TPU host.
#
# Squeezes the 230 B / 10 B-active MoE into 128 GB HBM via:
#   - streaming-converted per-leaf .npy file pile in /mnt/dtmpfs/minimax-m2.7-npy
#     (produced by convert_minimax_m2_streaming.py — ~230 GB float16)
#   - quantization=intmp + int4_weight_only.json: w_bits=4 across every kernel,
#     scales materialised on host at load and packed to INT4 before sharding to HBM.
#     Expected resident weights ≈ 114 GB, leaving ≈ 14 GB HBM for KV cache,
#     activations, and scratch (post-attention residuals dominate the activation
#     side; default decode batch=1).
#
# Mesh: v4-8 has 4 chips in a 1×2×2 torus; we run pure tensor-parallel
# (ici_tensor=4, ici_expert=1). Expert parallelism doesn't pay off below
# ~8 chips for a 256-expert model.
set -euo pipefail

DTMPFS_ROOT="${DTMPFS_ROOT:-/mnt/dtmpfs}"
NPY_DIR="${NPY_DIR:-${DTMPFS_ROOT}/minimax-m2.7-npy}"
HF_DIR="${HF_DIR:-${DTMPFS_ROOT}/minimax-m2.7-hf}"
TOK_DIR="${TOK_DIR:-$HF_DIR}"
JAX_CACHE_DIR="${JAX_CACHE_DIR:-${DTMPFS_ROOT}/jax_cache_v4_8}"
RUN_NAME="${RUN_NAME:-minimax-m2.7-decode-v4-8-int4}"

QUANT_CFG_PATH="${QUANT_CFG_PATH:-src/maxtext/configs/quantization/int4_weight_only.json}"
QUANT="${QUANT:-intmp}"
KV_QUANT="${KV_QUANT:-true}"            # KV cache at int8 to keep 14 GB HBM headroom usable
KV_QUANT_DTYPE="${KV_QUANT_DTYPE:-int8}"

BATCH="${BATCH:-1}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-65536}"
PREFILL_LEN="${PREFILL_LEN:-$((MAX_TARGET_LENGTH/2))}"
ICI_TENSOR="${ICI_TENSOR:-4}"
ICI_EXPERT="${ICI_EXPERT:-1}"
CAPACITY="${CAPACITY:-2.0}"
WEIGHT_DTYPE="${WEIGHT_DTYPE:-bfloat16}"
PROMPT="${PROMPT:-The capital of France is}"

cd "$HOME/maxtext"
source ~/venv-maxtext-py312/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export TF_CPP_MIN_LOG_LEVEL=2
mkdir -p "$JAX_CACHE_DIR"

LOG="$HOME/decode_v4_8.log"
exec > >(tee -a "$LOG") 2>&1
echo "[decode-v4-8] host=$(hostname) start=$(date -u +%FT%TZ)"

[[ -d "$NPY_DIR" ]] || { echo "missing $NPY_DIR — run convert_minimax_m2_streaming first"; exit 1; }
[[ -f "$TOK_DIR/tokenizer.json" ]] || { echo "missing tokenizer.json in $TOK_DIR"; exit 1; }
[[ -f "$QUANT_CFG_PATH" ]] || { echo "missing $QUANT_CFG_PATH (relative to \$PWD=$PWD)"; exit 1; }

python -m maxtext.inference.decode_minimax_m2_npy \
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
  sparse_matmul=false \
  capacity_factor="$CAPACITY" \
  prompt="$PROMPT"

echo "[decode-v4-8] done host=$(hostname) finish=$(date -u +%FT%TZ)"
