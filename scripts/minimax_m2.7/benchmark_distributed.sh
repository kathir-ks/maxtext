#!/usr/bin/env bash
# Benchmark launcher for MiniMax-M2.7 on v5e-64 with distributed per-host
# .npy shards. Runs MaxText's inference_microbenchmark via our
# benchmark_minimax_m2_npy wrapper and prints a JSON report.
#
# Env vars (all optional):
#   QUANT          ''|int8|int4|intmp         (default '')
#   KV_QUANT       true|false                  (default false)
#   KV_QUANT_DTYPE int8|int4                   (default int8)
#   BATCH          per-device batch size       (default 1)
#   CAPACITY       MoE capacity factor         (default 2.0)
#   PREFILL_LENS   comma-separated lengths     (default 64,256,1024)
#   LOOP_ITERS     decode steps per benchmark  (default 10)
#   CELL_ID        tag used in output filename (default auto)
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
PREFILL_LENS="${PREFILL_LENS:-64,256,1024}"
LOOP_ITERS="${LOOP_ITERS:-10}"
ICI_TENSOR="${ICI_TENSOR:-8}"
ICI_EXPERT="${ICI_EXPERT:-8}"

CELL_ID="${CELL_ID:-q${QUANT:-bf16}_kv${KV_QUANT}_b${BATCH}_cf${CAPACITY}}"
BENCH_JSON=$HOME/bench.${CELL_ID}.json
LOG=$HOME/bench_dist.log
exec > >(tee -a "$LOG") 2>&1

echo "[bench-dist] host=$(hostname) cell=$CELL_ID start=$(date -u +%FT%TZ)"
ls -1 "$NPY_DIR"/manifest.p*.json | head -1 >/dev/null \
  || { echo "missing distributed manifest on this worker"; exit 1; }

# max_target_length must be > the largest prefill we'll benchmark
MAX_TARGET=$(( $(echo "$PREFILL_LENS" | tr ',' '\n' | sort -n | tail -1) + 64 ))

python -m maxtext.inference.benchmark_minimax_m2_npy \
  --npy_dir "$NPY_DIR" \
  src/maxtext/configs/base.yml \
  model_name=minimax-m2.7 \
  tokenizer_path="$TOK_DIR" \
  tokenizer_type=huggingface \
  run_name=minimax-m2.7-bench-v5e64 \
  jax_cache_dir="$JAX_CACHE_DIR" \
  per_device_batch_size="$BATCH" \
  max_target_length="$MAX_TARGET" \
  max_prefill_predict_length=$(echo "$PREFILL_LENS" | tr ',' '\n' | sort -n | tail -1) \
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
  inference_microbenchmark_prefill_lengths="$PREFILL_LENS" \
  inference_microbenchmark_loop_iters="$LOOP_ITERS" \
  inference_microbenchmark_stages=prefill,generate \
  inference_microbenchmark_log_file_path="$BENCH_JSON"

# Worker 0 owns the JSON output; other workers produce it but it's identical
# bookkeeping. Echo a one-line summary if jq is present.
if command -v jq >/dev/null 2>&1 && [ -s "$BENCH_JSON" ]; then
  echo "[bench-dist] summary cell=$CELL_ID:"
  jq -r '"  tok/s_total=\(.autoregressive.total_throughput_tokens_per_second // "?")  step_ms=\(.autoregressive.step_in_ms // "?")  prefill_256_ms=\(.prefill["256"].time_in_ms // "?")"' "$BENCH_JSON" || true
fi

echo "[bench-dist] done host=$(hostname) cell=$CELL_ID finish=$(date -u +%FT%TZ)"
