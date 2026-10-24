#!/bin/bash
# Window 14 (10-24): R1 (CF_ANALYSIS.md): the chat tables at every positive temperature outside the thinking window,
# instead of the p-min fallback, vs production's selection (chat tables only for T 0.5-0.9), both on bin-r1:
#   P: LLAMA_SPEC_STOP_ISO_TEMP=0.5,0.9 (production's behaviour), R: the new default (any T > 0).
# Thinking off, production sampling with the temperature overridden to 0.3 and to 1.2 (top-p 0.8, top-k 20,
# presence 1.5), all 72 prompts, cap 300. Simulated: +6% at 0.3, +3.8% at 1.2. Order P R R P.
set -u
M=/work/models/Qwen3.8-27B-UD-Q6_K.gguf; KEYW=/work/qwen-api-keys.work
BV=/work/bench/mtp3/bin-r1; R=/work/bench/mtp3
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
run() { local lab=$1; shift; env "$@" SPLITS=dev,val,test SAMPLING=nothink MAXTOK=300 python3 $R/run_suite.py $lab 2>&1 | grep -E "ALL|Traceback|Error" ; }
ON="LLAMA_SPEC_STOP_ISO=0.125 LLAMA_SPEC_STOP_ISO_THINK=0.15"
srvr A14-AP $ON LLAMA_SPEC_STOP_ISO_TEMP=0.5,0.9 && run A14-AP TEMP=0.3
srvr A14-AL $ON && run A14-AL TEMP=0.3
log "T 0.3 done"; nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | tr '\n' ' '; echo
srvr A14-BL $ON && run A14-BL TEMP=1.2
srvr A14-BP $ON LLAMA_SPEC_STOP_ISO_TEMP=0.5,0.9 && run A14-BP TEMP=1.2
MODES="A:T 0.3 (thinking off),B:T 1.2 (thinking off)" python3 /work/bench/mtp3/summ9.py A14
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | tr '\n' ' '; echo
log "window14 done"
