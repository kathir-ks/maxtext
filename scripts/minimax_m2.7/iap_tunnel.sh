#!/usr/bin/env bash
# Client-side IAP TCP tunnel to the v6e-64 API server on worker 0.
# Run on the dev VM (or laptop with gcloud installed). Binds the
# server's port 8000 to localhost:8000.
#
# IAP is the only ingress path; no firewall opening is required on
# the TPU side. Auth is your GCP IAM identity; the bearer token (if
# set on the server) is defense-in-depth on top of that.
set -euo pipefail

NODE="${NODE:-node-v6e-64-europe-west4-a}"
ZONE="${ZONE:-europe-west4-a}"
WORKER="${WORKER:-0}"
REMOTE_PORT="${REMOTE_PORT:-8000}"
LOCAL_PORT="${LOCAL_PORT:-8000}"

echo "[iap] tunneling $NODE worker=$WORKER zone=$ZONE :$REMOTE_PORT -> localhost:$LOCAL_PORT"
echo "[iap] Ctrl-C to stop. While running, point clients at http://localhost:$LOCAL_PORT"

exec gcloud compute tpus tpu-vm ssh "$NODE" \
  --zone="$ZONE" \
  --worker="$WORKER" \
  --tunnel-through-iap \
  -- -L "${LOCAL_PORT}:localhost:${REMOTE_PORT}" -N
