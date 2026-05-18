#!/usr/bin/env bash
# Run MiniMax-M2.7 decode on TPU v5e. Reads weights from a local /mnt/dtmpfs
# checkpoint produced by download_and_convert.sh on the same VM.
set -euo pipefail

DTMPFS_ROOT="${DTMPFS_ROOT:-/mnt/dtmpfs}"
MODEL_TAG="${MODEL_TAG:-minimax-m2.7}"
MAXTEXT_DIR="${MAXTEXT_DIR:-${DTMPFS_ROOT}/${MODEL_TAG}-maxtext}"
HF_DIR="${HF_DIR:-${DTMPFS_ROOT}/${MODEL_TAG}-hf}"
RUN_NAME="${RUN_NAME:-${MODEL_TAG}-decode-v5e}"
BATCH="${BATCH:-1}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-2048}"
ICI_TENSOR="${ICI_TENSOR:-16}"           # v5e-16 slice; adjust to fit
ICI_EXPERT="${ICI_EXPERT:-1}"            # raise if you have a larger slice
PROMPT="${PROMPT:-The capital of France is}"

cd "$(dirname "$0")/../.."

python3 -m maxtext.inference.decode \
    src/maxtext/configs/base.yml \
    model_name="$MODEL_TAG" \
    tokenizer_path="$HF_DIR" \
    tokenizer_type=huggingface \
    load_parameters_path="$MAXTEXT_DIR/0/items" \
    run_name="$RUN_NAME" \
    per_device_batch_size="$BATCH" \
    max_target_length="$MAX_TARGET_LENGTH" \
    max_prefill_predict_length=$((MAX_TARGET_LENGTH / 2)) \
    ici_tensor_parallelism="$ICI_TENSOR" \
    ici_expert_parallelism="$ICI_EXPERT" \
    scan_layers=true \
    quantization='' \
    weight_dtype=bfloat16 \
    attention=dot_product \
    prompt="$PROMPT"
