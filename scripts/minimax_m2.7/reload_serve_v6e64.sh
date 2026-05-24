#!/usr/bin/env bash
# Restart the v6e-64 API server across all 16 workers after a code
# update (git push). Pulls the new code into every worker's repo, then
# restarts the systemd unit. Lockstep, so all 16 ranks see the same
# code before broadcast loop resumes.
set -euo pipefail

NODE="${NODE:-node-v6e-64-europe-west4-a}"
ZONE="${ZONE:-europe-west4-a}"
WORKER="${WORKER:-all}"
BRANCH="${BRANCH:-feature/minimax-m2.7-api-server}"
UNIT_NAME="minimax-m2-serve-v6e64.service"
PORT="${PORT:-8000}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"

if [ -z "${SSH_AUTH_SOCK:-}" ]; then
  eval "$(ssh-agent -s)" >/dev/null
fi

echo "[reload] git pull $BRANCH on every worker..."
gcloud compute tpus tpu-vm ssh "$NODE" --zone="$ZONE" --worker="$WORKER" \
  --command="cd ~/maxtext && git fetch origin $BRANCH && git reset --hard origin/$BRANCH"

echo "[reload] systemctl restart on every worker..."
gcloud compute tpus tpu-vm ssh "$NODE" --zone="$ZONE" --worker="$WORKER" \
  --command="sudo systemctl restart $UNIT_NAME"

echo "[reload] polling /ready on worker 0 (timeout ${READY_TIMEOUT}s)..."
deadline=$(( $(date +%s) + READY_TIMEOUT ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  if curl -fsS -m 5 "http://localhost:$PORT/ready" >/dev/null 2>&1; then
    echo "[reload] READY at $(date -u +%FT%TZ)"
    exit 0
  fi
  sleep 15
done
echo "[reload] timeout waiting for /ready (server may still be JIT-compiling)"
echo "[reload] check status: gcloud compute tpus tpu-vm ssh $NODE --zone=$ZONE --worker=0 --command='sudo systemctl status $UNIT_NAME'"
exit 1
