#!/usr/bin/env bash
# Mount /mnt/dtmpfs as a RAM-backed tmpfs sized for a 230B FP8/BF16 model
# (~230 GB FP8 + ~460 GB BF16). Run as root or with sudo. Idempotent.
#
# The point of this filesystem is to stage the HF download and the MaxText
# conversion entirely in RAM on a single TPU VM, so no model bytes ever
# leave the VM. Pair with a same-region GCS bucket only if you need to
# share checkpoints across hosts.
set -euo pipefail

MOUNT_POINT="${1:-/mnt/dtmpfs}"
SIZE="${2:-700G}"
USER_NAME="${SUDO_USER:-$USER}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Re-running with sudo (need root to mount tmpfs)..."
  exec sudo -E "$0" "$@"
fi

mkdir -p "$MOUNT_POINT"

if mountpoint -q "$MOUNT_POINT"; then
  echo "$MOUNT_POINT is already a mountpoint:"
  findmnt "$MOUNT_POINT"
  exit 0
fi

# Pre-flight: total memory must comfortably exceed the requested size.
TOTAL_KB=$(grep '^MemTotal:' /proc/meminfo | awk '{print $2}')
REQ_KB=$(numfmt --from=iec "$SIZE" | awk '{print int($1/1024)}')
if [[ "$REQ_KB" -ge "$TOTAL_KB" ]]; then
  echo "Refusing to mount $SIZE tmpfs: only $(numfmt --from-unit=1024 --to=iec ${TOTAL_KB})B RAM available." >&2
  exit 1
fi

mount -t tmpfs -o size="$SIZE",mode=0775,uid="$USER_NAME",gid="$USER_NAME" dtmpfs "$MOUNT_POINT"
chown "$USER_NAME":"$USER_NAME" "$MOUNT_POINT"

echo "Mounted tmpfs at $MOUNT_POINT (size=$SIZE):"
df -h "$MOUNT_POINT"
