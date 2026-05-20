#!/usr/bin/env bash
# Per-worker distributed convert. Each host writes only its 1/16 shard to
# local dtmpfs and downloads only the HF files it needs.
# Launch via `gcloud compute tpus tpu-vm ssh --worker=all` simultaneously;
# jax.distributed.initialize() does the pod discovery from the TPU runtime.
set -euo pipefail

DTMPFS_ROOT=/mnt/dtmpfs
export HF_DIR="$DTMPFS_ROOT/minimax-m2.7-hf"
export NPY_DIR="$DTMPFS_ROOT/minimax-m2.7-npy-distributed"
LOG=$HOME/stream_dist.log
exec > >(tee -a "$LOG") 2>&1

echo "[stream-dist] host=$(hostname) start=$(date -u +%FT%TZ)"
cd "$HOME/maxtext"
git fetch origin feature/minimax-m2.7
git reset --hard origin/feature/minimax-m2.7
echo "[stream-dist] HEAD=$(git rev-parse HEAD)"

# Wipe any stale replicated checkpoint to free dtmpfs.
rm -rf /mnt/dtmpfs/minimax-m2.7-npy 2>/dev/null || true

source ~/venv-maxtext-py312/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_ENABLE_HF_TRANSFER=1
# JAX must NOT use the CPU backend here — we want the real TPU mesh so the
# engine returns the (tensor=8, expert=8) sharding spec our shards target.
# get_abstract_state uses jax.eval_shape so it doesn't actually touch HBM.
unset JAX_PLATFORMS
unset XLA_FLAGS

mkdir -p "$HF_DIR" "$NPY_DIR"

python -m maxtext.checkpoint_conversion.standalone_scripts.convert_minimax_m2_distributed \
  --hf_dir "$HF_DIR" \
  --output_dir "$NPY_DIR" \
  --model_size minimax-m2.7 \
  --repo_id MiniMaxAI/MiniMax-M2.7 \
  --maxtext_args src/maxtext/configs/base.yml \
    tokenizer_path="$HOME/minimax-m2.7-tokenizer" \
    tokenizer_type=huggingface \
    run_name=minimax-m2.7-convert \
    per_device_batch_size=1 \
    max_target_length=64 \
    max_prefill_predict_length=32 \
    ici_tensor_parallelism=8 \
    ici_expert_parallelism=8

du -sh "$NPY_DIR" "$HF_DIR" || true
free -g | head -2
df -h /mnt/dtmpfs
echo "[stream-dist] done host=$(hostname) finish=$(date -u +%FT%TZ)"
