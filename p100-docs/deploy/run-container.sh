#!/bin/bash
# Run on the Unraid HOST: GPU power limit, then the container with both P100s and the share mounted at /work.
# The model needs BOTH cards: stop anything else holding GPU memory first.
set -e
WORK=${WORK:-/mnt/user/qwen-dev}
IMAGE=${IMAGE:-qwen-dev:latest}
NAME=${NAME:-qwen-dev}

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
if [ "$USED" -gt 1500 ]; then
  echo "GPUs already hold ${USED} MiB; stop whatever holds them first (docker ps, then docker stop <name>)."
  exit 1
fi

mkdir -p "$WORK"/{models,ccache,src,hf-cache,slots,jev}
chown -R 99:100 "$WORK"

# 150 W, not the 180 W default: on the reference board a 262k-context prefill with both cards at 180 W
# hard-reset the host; at 150 W the same run passed (CHANGES §16). Resets on reboot: see unraid-user-script.sh.
nvidia-smi -pm 1
nvidia-smi -i 0,1 -pl 150

docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v "$WORK":/work \
  -p 8090:8090 \
  --shm-size=2g \
  --restart unless-stopped \
  "$IMAGE"

docker exec "$NAME" nvidia-smi
echo "Started. Enter with: docker exec -it $NAME bash"
