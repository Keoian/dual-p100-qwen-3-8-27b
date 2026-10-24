#!/bin/bash
# Window 12 (10-24, thinking-on refit): the stop rule with tables fit on thinking-on traces (LLAMA_SPEC_STOP_ISO_THINK=0.15)
# vs production (LLAMA_SPEC_STOP_ISO=0.125, thinking requests on p-min), both on bin-think (default branch + the tables).
#   M: reasoning_effort medium (budget sampler, per-token rule), all 72 prompts (dev/val were used for the fit and theta)
#   N: no effort level (block verify), the 24 held-out test prompts, cap 1500
# Card thinking sampling (1.0/0.95/20/0), seed 11. Order P L L P.
set -u
M=/work/models/Qwen3.8-27B-UD-Q6_K.gguf; KEYW=/work/qwen-api-keys.work
BV=/work/bench/mtp3/bin-think; R=/work/bench/mtp3
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
run() { local lab=$1; shift; env "$@" SAMPLING=kmic THINK=on python3 $R/run_suite.py $lab 2>&1 | grep -E "ALL|Traceback|Error" ; }
srvr A12-MP LLAMA_SPEC_STOP_ISO=0.125 && run A12-MP SPLITS=dev,val,test REASONING_EFFORT=medium MAXTOK=3000
srvr A12-ML LLAMA_SPEC_STOP_ISO=0.125 LLAMA_SPEC_STOP_ISO_THINK=0.15 && run A12-ML SPLITS=dev,val,test REASONING_EFFORT=medium MAXTOK=3000
log "medium done"; nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | tr '\n' ' '; echo
srvr A12-NL LLAMA_SPEC_STOP_ISO=0.125 LLAMA_SPEC_STOP_ISO_THINK=0.15 && run A12-NL SPLITS=test MAXTOK=1500
srvr A12-NP LLAMA_SPEC_STOP_ISO=0.125 && run A12-NP SPLITS=test MAXTOK=1500
python3 /work/bench/mtp3/summ9.py A12
SPLIT=test python3 /work/bench/mtp3/summ9.py A12
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | tr '\n' ' '; echo
log "window12 done"
