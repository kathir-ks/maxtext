#!/usr/bin/env bash
# Sweep driver for finding max tok/s on the v5e-64 pod. Iterates a small,
# carefully-ordered config matrix and fans each cell out to all 16 workers.
# Uses bench_steps.sh which uses bench_steps_minimax_m2_npy.py (not
# inference_microbenchmark, which currently breaks on our custom load_params).
#
# Cells are ordered so we learn the most from each compile:
#   1. bf16 baseline at batch=1                 (calibration)
#   2. bf16 + larger batch                       (memory-bound amortization)
#   3. int8 weights + kv-cache bf16              (halves weight bandwidth)
#   4. int8 + int8 kv quant                      (more headroom for batch)
#   5. int8 + int8 + push batch                  (peak we expect)
#   6. int4 + int8 kv at best batch              (most aggressive)
set -euo pipefail

NODE="${NODE:-node-v5e-64-europe-west4-b}"
ZONE="${ZONE:-europe-west4-b}"
OUT_DIR="${OUT_DIR:-${CLAUDE_JOB_DIR:-/tmp}/max_tps}"
mkdir -p "$OUT_DIR"
CSV="$OUT_DIR/max_tps.csv"

#               id   quant  kv   batch  cap  comment
declare -a TABLE=(
  "1 ''   false 1  2.0 bf16-baseline"
  "2 ''   false 4  2.0 bf16-batch4"
  "3 ''   false 8  2.0 bf16-batch8"
  "4 int8 false 1  2.0 int8-baseline"
  "5 int8 true  1  2.0 int8-kvq"
  "6 int8 true  4  2.0 int8-batch4"
  "7 int8 true  8  2.0 int8-batch8"
  "8 int8 true  16 2.0 int8-batch16"
  "9 int4 true  8  2.0 int4-batch8"
)

want() { [[ -z "${CELLS:-}" ]] || [[ ",${CELLS}," == *",$1,"* ]]; }

echo "cell,comment,quant,kv,batch,cap,prefill_ms,step_ms_p50,step_ms_p90,tok_s_pod,tok_s_chip" > "$CSV"

for row in "${TABLE[@]}"; do
  # shellcheck disable=SC2086
  set -- $row
  id=$1; q=$2; kv=$3; b=$4; cap=$5; comment=$6
  if ! want "$id"; then continue; fi
  [[ "$q" == "''" ]] && q=""

  tag="q${q:-bf16}_kv${kv}_b${b}_cf${cap}"
  echo "==========================================================="
  echo "[sweep] cell=$id  ${comment}  quant=${q:-bf16}  kv=$kv  batch=$b  cap=$cap"
  echo "==========================================================="

  cell_log="$OUT_DIR/cell${id}.${tag}.log"

  gcloud compute tpus tpu-vm ssh "$NODE" --zone="$ZONE" --worker=all \
    --command="sudo pkill -9 -f '[b]ench_steps_minimax_m2_npy\\|[d]ecode_minimax_m2_npy' 2>/dev/null; sleep 1; rm -f ~/bench_steps.log; QUANT='$q' KV_QUANT='$kv' BATCH='$b' CAPACITY='$cap' CELL_ID='$tag' MAX_TARGET_LENGTH=256 WARMUP=4 MEASURE=32 bash ~/maxtext/scripts/minimax_m2.7/bench_steps.sh" \
    2>&1 | tee "$cell_log" | tail -5

  gcloud compute tpus tpu-vm scp "$NODE":~/bench_steps.${tag}.json \
    "$OUT_DIR/bench_steps.${tag}.json" \
    --zone="$ZONE" --worker=0 2>/dev/null || true

  json="$OUT_DIR/bench_steps.${tag}.json"
  if command -v jq >/dev/null 2>&1 && [ -s "$json" ]; then
    pf=$(jq -r '.prefill_ms // ""' "$json")
    p50=$(jq -r '.step_ms_p50 // ""' "$json")
    p90=$(jq -r '.step_ms_p90 // ""' "$json")
    tps_pod=$(jq -r '.tok_per_s_pod // ""' "$json")
    tps_chip=$(jq -r '.tok_per_s_chip // ""' "$json")
    echo "$id,$comment,${q:-bf16},$kv,$b,$cap,$pf,$p50,$p90,$tps_pod,$tps_chip" >> "$CSV"
  else
    echo "$id,$comment,${q:-bf16},$kv,$b,$cap,,,,,FAIL" >> "$CSV"
  fi
done

echo "==========================================================="
echo "[sweep] CSV: $CSV"
column -s, -t "$CSV" 2>/dev/null || cat "$CSV"
