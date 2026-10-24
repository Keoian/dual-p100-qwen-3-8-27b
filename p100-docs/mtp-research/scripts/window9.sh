#!/bin/bash
# Window 9 (10-24, MTP grounding, part 1): the lookup stop rule (LLAMA_SPEC_STOP_ISO=0.125) vs production with thinking on.
# Two thinking-on paths in production (LLAMA_THINK_BY_BUDGET=1):
#   M: reasoning_effort "medium" -> 2,048-token thinking budget -> reasoning-budget sampler -> per-token verify rule
#   N: no effort level           -> no budget sampler, no penalties -> block verify
# Test split (24 prompts), card thinking sampling (1.0/0.95/20/0), seed 11. Order P L L P cancels linear drift.
set -u
M=/work/models/Qwen3.8-27B-UD-Q6_K.gguf; KEYW=/work/qwen-api-keys.work
BV=/work/bench/mtp3/bin-iso; R=/work/bench/mtp3
log() { echo "[$(date +%H:%M:%S)] $*"; }
gpu_free() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')" -lt 100 ]; do sleep 2; done; }
cool() { local t; t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
  if [ "$t" -ge 80 ]; then log "GPU ${t}C: pausing until < 70C"
    until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)" -lt 70 ]; do sleep 10; done; fi; }
restore() {
  log "restoring production"; pkill -x llama-server; gpu_free
  (setsid nohup bash /work/start-qwen-vision.sh > /work/serve-8090.log 2>&1 &)
  for i in $(seq 120); do curl -sf localhost:8090/health >/dev/null && break; sleep 5; done
  curl -sf localhost:8090/health >/dev/null && log "production up" || log "PRODUCTION NOT UP"
  rm -f "$KEYW"; log "temp key removed"
}
python3 -c 'import secrets; print(secrets.token_hex(24))' > "$KEYW"; chmod 600 "$KEYW"
trap restore EXIT
busy() { python3 -c "
import json,urllib.request; k=[l.strip() for l in open('/work/qwen-api-keys') if l.strip() and not l.startswith('#')][0]
s=json.loads(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8090/slots', headers={'Authorization':'Bearer '+k}), timeout=10).read())
print(int(any(x.get('is_processing') for x in s)))" 2>/dev/null || echo 0; }
for i in $(seq 90); do [ "$(busy)" = 0 ] && break; [ $i = 1 ] && log "production busy, waiting"; sleep 10; done
log "stopping production"; pkill -x llama-server; gpu_free; cool
up() { for i in $(seq 120); do curl -sf localhost:8090/health >/dev/null && break; sleep 5; done
  curl -sf localhost:8090/health >/dev/null || { log "server $1 did not come up"; tail -5 $R/runs/srv-$1.log; return 1; }
  python3 -c "
import json,urllib.request; k=[l.strip() for l in open('$KEYW') if l.strip()][0]
urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8090/v1/chat/completions', json.dumps(dict(messages=[{'role':'user','content':'Hi'}],max_tokens=16)).encode(), {'Content-Type':'application/json','Authorization':'Bearer '+k}), timeout=600).read()" \
    && log "server $1 up (warmed)"; }
srvr() { local lab=$1; shift; pkill -x llama-server; gpu_free; cool
  (env "$@" BUILD=$BV API_KEY_FILE=$KEYW HOST=127.0.0.1 setsid nohup bash $R/srv/start-qwen-vision.sh > $R/runs/srv-$lab.log 2>&1 &)
  up $lab; }
mkdir -p $R/runs
run() { local lab=$1; shift; env "$@" SPLITS=test SAMPLING=kmic THINK=on python3 $R/run_suite.py $lab 2>&1 | grep -E "ALL|Traceback|Error" ; }
srvr A9-MP && run A9-MP REASONING_EFFORT=medium MAXTOK=3000
srvr A9-ML LLAMA_SPEC_STOP_ISO=0.125 && run A9-ML REASONING_EFFORT=medium MAXTOK=3000
log "medium done"; nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | tr '\n' ' '; echo
srvr A9-NL LLAMA_SPEC_STOP_ISO=0.125 && run A9-NL MAXTOK=1500
srvr A9-NP && run A9-NP MAXTOK=1500
python3 /work/bench/mtp3/summ9.py
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | tr '\n' ' '; echo
log "window9 done"
