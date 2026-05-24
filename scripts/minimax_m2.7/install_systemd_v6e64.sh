#!/usr/bin/env bash
# Install the per-host systemd unit on every v6e-64 worker. Idempotent;
# safe to re-run after updates to the unit file.
#
# After this completes, manage the service via:
#   gcloud compute tpus tpu-vm ssh $NODE --zone=$ZONE --worker=all \
#     --command='sudo systemctl status minimax-m2-serve-v6e64'
set -euo pipefail

NODE="${NODE:-node-v6e-64-europe-west4-a}"
ZONE="${ZONE:-europe-west4-a}"
WORKER="${WORKER:-all}"
BRANCH="${BRANCH:-feature/minimax-m2.7-api-server}"
UNIT_NAME="minimax-m2-serve-v6e64.service"

if [ -z "${SSH_AUTH_SOCK:-}" ]; then
  eval "$(ssh-agent -s)" >/dev/null
fi

echo "[install] pulling $BRANCH on all workers..."
gcloud compute tpus tpu-vm ssh "$NODE" --zone="$ZONE" --worker="$WORKER" \
  --command="cd ~/maxtext && git fetch origin $BRANCH && git reset --hard origin/$BRANCH"

echo "[install] copying unit + enabling + starting on all workers..."
gcloud compute tpus tpu-vm ssh "$NODE" --zone="$ZONE" --worker="$WORKER" --command="
set -euo pipefail
sudo cp ~/maxtext/scripts/minimax_m2.7/$UNIT_NAME /etc/systemd/system/$UNIT_NAME
sudo systemctl daemon-reload
sudo systemctl enable $UNIT_NAME
sudo systemctl restart $UNIT_NAME
systemctl is-active $UNIT_NAME || true
"

echo "[install] done. Tail logs with:"
echo "  gcloud compute tpus tpu-vm ssh $NODE --zone=$ZONE --worker=0 \\"
echo "    --command='sudo journalctl -u $UNIT_NAME -f'"
