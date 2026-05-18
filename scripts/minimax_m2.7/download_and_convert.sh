#!/usr/bin/env bash
# Download MiniMax-M2.7 from HF directly into /mnt/dtmpfs and convert to a
# MaxText checkpoint *in place*. Never copies the model bytes out of this VM.
#
# Total bytes downloaded: ~230 GB (FP8 safetensors + tokenizer + config).
# Free-space recommendation: /mnt/dtmpfs ≥ 350 GB (FP8 + a working MaxText
# checkpoint). For an even smaller footprint, use the converter with FP8 inputs
# directly (it dequantizes on the fly) and only the MaxText checkpoint resides
# alongside the HF weights.
set -euo pipefail

REPO="${REPO:-MiniMaxAI/MiniMax-M2.7}"
MODEL_TAG="${MODEL_TAG:-minimax-m2.7}"
DTMPFS_ROOT="${DTMPFS_ROOT:-/mnt/dtmpfs}"
HF_DIR="${DTMPFS_ROOT}/${MODEL_TAG}-hf"
MAXTEXT_DIR="${DTMPFS_ROOT}/${MODEL_TAG}-maxtext"
HF_REVISION="${HF_REVISION:-main}"
SIMULATED_CPUS="${SIMULATED_CPUS:-16}"

if [[ ! -d "$DTMPFS_ROOT" ]]; then
  echo "$DTMPFS_ROOT does not exist; run scripts/minimax_m2.7/setup_dtmpfs.sh first." >&2
  exit 1
fi

if ! mountpoint -q "$DTMPFS_ROOT"; then
  echo "WARNING: $DTMPFS_ROOT is not a tmpfs mountpoint. Continuing anyway."
fi

mkdir -p "$HF_DIR" "$MAXTEXT_DIR"

if [[ ! -f "$HF_DIR/config.json" ]]; then
  echo ">>> Downloading $REPO @ $HF_REVISION into $HF_DIR (this can take a while)"
  python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="$REPO",
    revision="$HF_REVISION",
    local_dir="$HF_DIR",
    local_dir_use_symlinks=False,
    allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "*.py", "*.jinja"],
)
PY
else
  echo ">>> Skipping download — $HF_DIR/config.json already present"
fi

echo ">>> Converting HF -> MaxText (in place, no out-of-VM transfer)"
python3 -m maxtext.checkpoint_conversion.standalone_scripts.convert_minimax_m2 \
    --base_model_path "$HF_DIR" \
    --maxtext_model_path "$MAXTEXT_DIR" \
    --model_size "$MODEL_TAG" \
    --simulated_cpu_devices_count "$SIMULATED_CPUS"

echo ">>> Done. MaxText checkpoint: $MAXTEXT_DIR"
du -sh "$HF_DIR" "$MAXTEXT_DIR" || true
