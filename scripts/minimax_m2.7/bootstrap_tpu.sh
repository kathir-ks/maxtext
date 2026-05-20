#!/usr/bin/env bash
# Bootstrap a TPU worker (v5e-64 or v6e-64) for MiniMax-M2.7 inference.
#
# Idempotent: re-running on an already-bootstrapped worker only does
# `git pull`, skipping the expensive uv/python/pip steps. To force a
# fresh bootstrap, run `rm -rf ~/maxtext ~/venv-maxtext-py312`.
#
#   - install uv (no sudo)
#   - install python 3.12
#   - clone kathir-ks/maxtext at feature/minimax-m2.7 (or fetch on re-run)
#   - create venv + pip install -e ".[tpu]" (skipped if already complete)
#   - install torch (cpu-only), JetStream pinned, HF/safetensors
#
# Launch via:
#   gcloud compute tpus tpu-vm ssh <node> --zone=<zone> --worker=all \
#     --command='bash ~/bootstrap_tpu.sh'
set -euo pipefail

WORK="$HOME"
REPO_URL="${REPO_URL:-https://github.com/kathir-ks/maxtext.git}"
BRANCH="${BRANCH:-feature/minimax-m2.7}"
MAXTEXT_DIR="$WORK/maxtext"
VENV="$WORK/venv-maxtext-py312"
JETSTREAM_PIN="https://github.com/AI-Hypercomputer/JetStream/archive/29329e8e73820993f77cfc8efe34eb2a73f5de98.zip"

export PATH="$WORK/.local/bin:$PATH"

echo "[bootstrap] host=$(hostname) start=$(date -u +%FT%TZ)"

# Fast path: if everything is already installed and importable, just pull
# the latest branch HEAD and exit. Saves ~5 min on warm re-runs.
if [ -d "$MAXTEXT_DIR/.git" ] && [ -d "$VENV" ] && \
   "$VENV/bin/python" -c "import jax, torch, jetstream" >/dev/null 2>&1; then
  echo "[bootstrap] env already bootstrapped; pulling latest $BRANCH"
  cd "$MAXTEXT_DIR"
  git fetch --depth=1 origin "$BRANCH"
  git checkout "$BRANCH" 2>/dev/null || git checkout -B "$BRANCH" "origin/$BRANCH"
  git reset --hard "origin/$BRANCH"
  echo "[bootstrap] HEAD=$(git rev-parse HEAD)"
  echo "[bootstrap] done host=$(hostname) finish=$(date -u +%FT%TZ)"
  exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[bootstrap] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$WORK/.local/bin:$PATH"
echo "[bootstrap] uv version: $(uv --version)"
uv python install 3.12 >/dev/null
echo "[bootstrap] python 3.12 ready"

if [ ! -d "$MAXTEXT_DIR/.git" ]; then
  echo "[bootstrap] cloning $REPO_URL"
  git clone --depth=1 --branch "$BRANCH" "$REPO_URL" "$MAXTEXT_DIR"
else
  echo "[bootstrap] fetching latest $BRANCH"
  cd "$MAXTEXT_DIR"
  git fetch --depth=1 origin "$BRANCH"
  git checkout "$BRANCH" 2>/dev/null || git checkout -B "$BRANCH" "origin/$BRANCH"
  git reset --hard "origin/$BRANCH"
fi

cd "$MAXTEXT_DIR"
echo "[bootstrap] HEAD=$(git rev-parse HEAD)"

if [ ! -d "$VENV" ]; then
  echo "[bootstrap] creating venv"
  uv venv --python 3.12 "$VENV"
fi
# shellcheck disable=SC1090
source "$VENV/bin/activate"
python -V

echo "[bootstrap] installing maxtext + tpu deps (this can take a while)"
uv pip install --python "$VENV/bin/python" -U pip
uv pip install --python "$VENV/bin/python" -e ".[tpu]"
uv pip install --python "$VENV/bin/python" "huggingface_hub[hf_transfer]" safetensors

# Inference-only extras not pulled by the maxtext .[tpu] extra:
#   torch — CPU build, used for FP8 dequant in the HF converter
#   JetStream — MaxText's decode/serving glue; pip release is too old
uv pip install --python "$VENV/bin/python" torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$VENV/bin/python" "$JETSTREAM_PIN"

echo "[bootstrap] done host=$(hostname) finish=$(date -u +%FT%TZ)"
