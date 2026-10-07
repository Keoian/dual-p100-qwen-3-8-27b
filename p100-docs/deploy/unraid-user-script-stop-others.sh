#!/bin/bash
# OPTIONAL Unraid User Script, "At Startup of Array": like unraid-user-script.sh, but first stops other containers that
# use the GPUs (default: llm-dev), waits until the GPUs are free, then starts the Qwen server (JEV + vision) detached.
# Use this one instead of unraid-user-script.sh when another GPU container may autostart; keep that container's own
# autostart scripts disabled so the two don't race.
STOP_CONTAINERS=${STOP_CONTAINERS:-llm-dev}   # space-separated
QWEN_CONTAINER=${QWEN_CONTAINER:-qwen-dev}
LOG=${LOG:-/work/bench/serve-8090.log}        # path inside the container

nvidia-smi -pm 1
nvidia-smi -i 0,1 -pl 150        # both reset on every boot; 180 W reset the host once under load (CHANGES §16)

# Wait for the Docker service (it starts after the array)
for i in $(seq 1 60); do docker info >/dev/null 2>&1 && break; sleep 5; done

# Stop the other GPU containers if they're running: the Qwen model needs both GPUs
for c in $STOP_CONTAINERS; do
  if [ "$(docker inspect -f '{{.State.Running}}' "$c" 2>/dev/null)" = "true" ]; then
    echo "stopping $c"
    docker stop -t 30 "$c"
  fi
done

# Wait until the GPUs are actually free (the driver releases memory a few seconds after the processes exit)
for i in $(seq 1 30); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
  [ "$USED" -le 1500 ] && break
  sleep 2
done
if [ "$USED" -gt 1500 ]; then
  echo "GPUs still hold ${USED} MiB after stopping: $STOP_CONTAINERS; not starting $QWEN_CONTAINER"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv
  exit 1
fi

docker start "$QWEN_CONTAINER" >/dev/null 2>&1
sleep 5

# Start detached so this script returns (start-qwen.sh execs llama-server in the foreground)
docker exec -d "$QWEN_CONTAINER" bash -lc "setsid nohup bash /work/start-qwen.sh > $LOG 2>&1"

# Report when it's up (~1 min to load)
for i in $(seq 1 60); do
  docker exec "$QWEN_CONTAINER" curl -sf localhost:8090/health >/dev/null && { echo "qwen up"; break; }
  sleep 5
done
nvidia-smi --query-gpu=index,power.limit,memory.used --format=csv,noheader
