#!/bin/bash
# Unraid "User Scripts" plugin, schedule "At Startup of Array": make the 150 W GPU power limit survive reboots
# (nvidia-smi -pl resets on every boot). Optionally starts the server too (START_SERVER=1): the production setup,
# start-qwen-vision.sh (QWEN_SCRIPT=/work/start-qwen.sh for the JEV setup).
nvidia-smi -pm 1
nvidia-smi -i 0,1 -pl 150
nvidia-smi --query-gpu=index,power.limit --format=csv,noheader
if [ "${START_SERVER:-0}" = 1 ]; then
  sleep 20   # let docker bring the container up (--restart unless-stopped)
  docker exec -d qwen-dev bash -lc "setsid nohup bash ${QWEN_SCRIPT:-/work/start-qwen-vision.sh} > /work/serve-8090.log 2>&1"
fi
