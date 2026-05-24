#!/usr/bin/env bash
# Driver script: run from the dev VM (NOT inside the TPU pod). Fans out
# the per-worker serve_v6e_64.sh launcher across all 16 v6e workers,
# then optionally starts a client-side IAP tunnel and polls /ready.
#
# Uses the proven SSH launch pattern (see ~/.claude memory
# feedback_ssh_launch.md): setsid nohup + [m]axtext bracket pkill +
# eval ssh-agent in the same shell so SSH returns immediately.
set -euo pipefail

NODE="${NODE:-node-v6e-64-europe-west4-a}"
ZONE="${ZONE:-europe-west4-a}"
WORKER="${WORKER:-all}"
BRANCH="${BRANCH:-feature/minimax-m2.7-api-server}"
PORT="${PORT:-8000}"
START_TUNNEL="${START_TUNNEL:-1}"
WAIT_READY="${WAIT_READY:-1}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"   # 30 min default first-cold-start

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[launch] node=$NODE zone=$ZONE worker=$WORKER branch=$BRANCH port=$PORT"

# 1. Make sure ssh-agent is up in this shell (gcloud reuses it).
if [ -z "${SSH_AUTH_SOCK:-}" ]; then
  eval "$(ssh-agent -s)" >/dev/null
fi

# 2. Stop any previous server on every worker. Use [m]axtext bracket trick
#    so the pkill pattern doesn't match its own ssh-fanout command line.
echo "[launch] pkill prior server (if any)..."
gcloud compute tpus tpu-vm ssh "$NODE" \
  --zone="$ZONE" --worker="$WORKER" \
  --command="pkill -f '[m]axtext_server' || true; pkill -f '[s]erve_minimax_m2_npy' || true" \
  || true

# 3. Pull the branch on every worker (idempotent; the per-worker script
#    also does a git fetch+reset, but doing it here surfaces git auth
#    failures before we spawn the long-running process).
echo "[launch] git pull $BRANCH on every worker..."
gcloud compute tpus tpu-vm ssh "$NODE" \
  --zone="$ZONE" --worker="$WORKER" \
  --command="cd ~/maxtext && git fetch origin $BRANCH && git reset --hard origin/$BRANCH"

# 4. Launch the server on every worker, fully detached.
echo "[launch] starting serve_v6e_64.sh on every worker..."
LAUNCH_CMD="cd ~/maxtext && setsid nohup bash scripts/minimax_m2.7/serve_v6e_64.sh \
  >> ~/serve_v6e_64.log 2>&1 < /dev/null & disown; sleep 1; echo launched on \$(hostname)"
gcloud compute tpus tpu-vm ssh "$NODE" \
  --zone="$ZONE" --worker="$WORKER" \
  --command="$LAUNCH_CMD"

echo "[launch] fan-out complete; logs at ~/serve_v6e_64.log on each worker"

# 5. Optionally bring up an IAP tunnel and wait for /ready.
if [ "$START_TUNNEL" = "1" ]; then
  echo "[launch] starting IAP tunnel (background)..."
  setsid nohup bash "$THIS_DIR/iap_tunnel.sh" \
    >> "$HOME/iap_tunnel.log" 2>&1 < /dev/null &
  disown
  sleep 5
fi

if [ "$WAIT_READY" = "1" ]; then
  echo "[launch] polling http://localhost:$PORT/ready (timeout ${READY_TIMEOUT}s)..."
  deadline=$(( $(date +%s) + READY_TIMEOUT ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if curl -fsS -m 5 "http://localhost:$PORT/ready" >/dev/null 2>&1; then
      echo "[launch] READY at $(date -u +%FT%TZ)"
      exit 0
    fi
    sleep 15
  done
  echo "[launch] timeout: /ready did not return 200 within ${READY_TIMEOUT}s"
  echo "[launch] inspect logs:"
  echo "        gcloud compute tpus tpu-vm ssh $NODE --zone=$ZONE --worker=0 --command='tail -n 200 ~/serve_v6e_64.log'"
  exit 1
fi

echo "[launch] done (no readiness wait requested)"
