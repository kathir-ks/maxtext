#!/usr/bin/env bash
# Driver: iterate the Phase-2 config sweep, fan each cell out to all 16 workers
# of node-v5e-64-europe-west4-b, then aggregate worker-0 JSON into a CSV.
#
# Usage (run from your dev machine, not on the TPU):
#   bash scripts/minimax_m2.7/sweep.sh
#
# Override the table from the command line with CELLS env var, e.g.:
#   CELLS="2,4" bash scripts/minimax_m2.7/sweep.sh  # only cells 2 and 4
set -euo pipefail

NODE="${NODE:-node-v5e-64-europe-west4-b}"
ZONE="${ZONE:-europe-west4-b}"
OUT_DIR="${OUT_DIR:-${CLAUDE_JOB_DIR:-/tmp}/sweep}"
mkdir -p "$OUT_DIR"
CSV="$OUT_DIR/sweep.csv"

# cell_id quant kv_quant batch capacity
# Skipped automatically: rows whose cell_id is not in $CELLS (when set).
declare -a TABLE=(
  "1 '' false 1 2.0"   # baseline bf16
  "2 int8 false 1 2.0"  # int8 weights, kv-cache bf16
  "3 int8 true  1 2.0"  # int8 weights + int8 kv-cache
  "4 int8 true  4 2.0"  # batch=4 scaling
  "5 int8 true  8 2.0"  # batch=8 HBM cliff
  "6 int8 true  4 1.5"  # lower capacity_factor
  "7 int4 true  4 2.0"  # int4 weights (gated on cell 4 sanity)
)

want_cell() {
  if [[ -z "${CELLS:-}" ]]; then return 0; fi
  [[ ",${CELLS}," == *",$1,"* ]]
}

echo "cell,quant,kv_quant,batch,capacity,prefill_64_ms,prefill_256_ms,prefill_1024_ms,decode_step_ms,tok_s_total,bw_GB_s" > "$CSV"

for row in "${TABLE[@]}"; do
  # shellcheck disable=SC2086
  set -- $row
  cell_id=$1; quant=$2; kv_quant=$3; batch=$4; capacity=$5
  if ! want_cell "$cell_id"; then continue; fi

  # Treat the literal string '' as empty.
  [[ "$quant" == "''" ]] && quant=""

  tag="q${quant:-bf16}_kv${kv_quant}_b${batch}_cf${capacity}"
  echo "==============================================================="
  echo "[sweep] cell=$cell_id  quant=${quant:-bf16}  kv_quant=$kv_quant  batch=$batch  capacity=$capacity"
  echo "==============================================================="

  # Run the cell across all 16 workers; capture stdout to a per-cell file.
  cell_log="$OUT_DIR/cell${cell_id}.${tag}.log"
  gcloud compute tpus tpu-vm ssh "$NODE" --zone="$ZONE" --worker=all \
    --command="QUANT='$quant' KV_QUANT='$kv_quant' BATCH='$batch' CAPACITY='$capacity' CELL_ID='$tag' bash ~/maxtext/scripts/minimax_m2.7/benchmark_distributed.sh" \
    2>&1 | tee "$cell_log"

  # Pull worker-0's JSON to the dev machine.
  gcloud compute tpus tpu-vm scp "$NODE":~/bench.${tag}.json "$OUT_DIR/bench.${tag}.json" \
    --zone="$ZONE" --worker=0 2>/dev/null || true

  # Extract a CSV row.
  if command -v jq >/dev/null 2>&1 && [ -s "$OUT_DIR/bench.${tag}.json" ]; then
    p64=$(jq -r '.prefill["64"].time_in_ms // ""' "$OUT_DIR/bench.${tag}.json")
    p256=$(jq -r '.prefill["256"].time_in_ms // ""' "$OUT_DIR/bench.${tag}.json")
    p1024=$(jq -r '.prefill["1024"].time_in_ms // ""' "$OUT_DIR/bench.${tag}.json")
    step=$(jq -r '.autoregressive.step_in_ms // ""' "$OUT_DIR/bench.${tag}.json")
    tps=$(jq -r '.autoregressive.total_throughput_tokens_per_second // ""' "$OUT_DIR/bench.${tag}.json")
    bw=$(jq -r '.autoregressive.bw_per_device_GB_per_second // ""' "$OUT_DIR/bench.${tag}.json")
    echo "$cell_id,${quant:-bf16},$kv_quant,$batch,$capacity,$p64,$p256,$p1024,$step,$tps,$bw" >> "$CSV"
  else
    echo "$cell_id,${quant:-bf16},$kv_quant,$batch,$capacity,,,,,," >> "$CSV"
  fi
done

echo "==============================================================="
echo "[sweep] aggregated CSV: $CSV"
column -s, -t "$CSV" 2>/dev/null || cat "$CSV"
