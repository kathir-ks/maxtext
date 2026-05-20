#!/usr/bin/env bash
# Recover MiniMax-M2.7 .npy pile + tokenizer from GCS into /mnt/dtmpfs on a
# freshly-rebooted v4-8 (or any host with a 350 GB tmpfs at /mnt/dtmpfs).
# Idempotent: only downloads what's missing.
set -euo pipefail

DTMPFS_ROOT="${DTMPFS_ROOT:-/mnt/dtmpfs}"
NPY_DIR="${NPY_DIR:-${DTMPFS_ROOT}/minimax-m2.7-npy}"
HF_DIR="${HF_DIR:-${DTMPFS_ROOT}/minimax-m2.7-hf}"
NPY_GCS="${NPY_GCS:-gs://arc-mi-research/models/minimax-m2.7-npy}"
TOK_GCS="${TOK_GCS:-gs://arc-mi-research/models/minimax-m2.7-hf-tokenizer}"

if [[ ! -d "$DTMPFS_ROOT" ]] || ! mountpoint -q "$DTMPFS_ROOT"; then
  echo "$DTMPFS_ROOT is not a mounted tmpfs — run setup_dtmpfs.sh 350G first" >&2
  exit 1
fi

mkdir -p "$NPY_DIR" "$HF_DIR"

if [[ ! -f "$NPY_DIR/manifest.json" ]]; then
  echo "[reload] downloading .npy pile from $NPY_GCS"
  gcloud storage cp -r "${NPY_GCS}/*" "${NPY_DIR}/"
else
  echo "[reload] .npy pile already on tmpfs ($(du -sh "$NPY_DIR" | cut -f1)); skipping"
fi

if [[ ! -f "$HF_DIR/tokenizer.json" ]]; then
  echo "[reload] downloading tokenizer from $TOK_GCS"
  gcloud storage cp "${TOK_GCS}/*" "${HF_DIR}/"
else
  echo "[reload] tokenizer already on tmpfs; skipping"
fi

echo "[reload] done"
df -h "$DTMPFS_ROOT"
