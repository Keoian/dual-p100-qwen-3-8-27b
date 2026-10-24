#!/bin/bash
# Window 6 (10-24, mtp3 Phase 1): trace-on/off identity, then LLAMA_SPEC_TRACE runs (build-exp = branch mtp-adaptive-research)
set -u
SRC=/work/src/llama.cpp; F=/work/bench/forks; M=/work/models/Qwen3.8-27B-UD-Q6_K.gguf; KEYW=/work/qwen-api-keys.work
BV=/work/bench/mtp3/bin-trace; R=/work/bench/mtp3
log() { echo "[$(date +%H:%M:%S)] $*"; }
gpu_free() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')" -lt 100 ]; do sleep 2; done; }
cool() { local t; t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
  if [ "$t" -ge 80 ]; then log "GPU ${t}C: pausing until < 70C"
    until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)" -lt 70 ]; do sleep 10; done; fi; }
restore() {
  log "restoring production"; pkill -x llama-server; gpu_free; rm -f "$SRC/build-opt"
  (setsid nohup bash /work/start-qwen-vision.sh > /work/serve-8090.log 2>&1 &)
  for i in $(seq 120); do curl -sf localhost:8090/health >/dev/null && break; sleep 5; done
  curl -sf localhost:8090/health >/dev/null && log "production up" || log "PRODUCTION NOT UP"
  rm -f "$KEYW"; log "temp key removed"
}
srv() { # srv <label> <snapshot dir> [ENV=val ...]: test server = production setup on that snapshot, temp key
  local lab=$1 snap=$2; shift 2; pkill -x llama-server; gpu_free; cool
  (env "$@" BUILD=$snap API_KEY_FILE=$KEYW setsid nohup bash $F/srv/start-qwen-vision.sh > $F/srv-$lab.log 2>&1 &)
  for i in $(seq 120); do curl -sf localhost:8090/health >/dev/null && break; sleep 5; done
  curl -sf localhost:8090/health >/dev/null || { log "server $lab did not come up"; return 1; }
  KEYFILE=$KEYW python3 -c "
import json,urllib.request; k=[l.strip() for l in open('$KEYW') if l.strip()][0]
urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8090/v1/chat/completions', json.dumps(dict(messages=[{'role':'user','content':'Hi'}],max_tokens=16)).encode(), {'Content-Type':'application/json','Authorization':'Bearer '+k}), timeout=600).read()" \
    && log "server $lab up (warmed)"
}
python3 -c 'import secrets; print(secrets.token_hex(24))' > "$KEYW"; chmod 600 "$KEYW"
trap restore EXIT
busy() { python3 -c "
import json,urllib.request; k=[l.strip() for l in open('/work/qwen-api-keys') if l.strip() and not l.startswith('#')][0]
s=json.loads(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8090/slots', headers={'Authorization':'Bearer '+k}), timeout=10).read())
print(int(any(x.get('is_processing') for x in s)))" 2>/dev/null || echo 0; }
for i in $(seq 90); do [ "$(busy)" = 0 ] && break; [ $i = 1 ] && log "production busy, waiting"; sleep 10; done
log "stopping production"; pkill -x llama-server; gpu_free; cool
# research server: production setup from the mtp3 copy of the scripts (NMAX/PMIN/MTP/SPEC_TEMP), localhost bind
srvr() { local lab=$1; shift; pkill -x llama-server; gpu_free; cool
  (env "$@" BUILD=$BV API_KEY_FILE=$KEYW HOST=127.0.0.1 setsid nohup bash $R/srv/start-qwen-vision.sh > $R/runs/srv-$lab.log 2>&1 &)
  up $lab; }
up() { for i in $(seq 120); do curl -sf localhost:8090/health >/dev/null && break; sleep 5; done
  curl -sf localhost:8090/health >/dev/null || { log "server $1 did not come up"; tail -5 $R/runs/srv-$1.log; return 1; }
  python3 -c "
import json,urllib.request; k=[l.strip() for l in open('$KEYW') if l.strip()][0]
urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8090/v1/chat/completions', json.dumps(dict(messages=[{'role':'user','content':'Hi'}],max_tokens=16)).encode(), {'Content-Type':'application/json','Authorization':'Bearer '+k}), timeout=600).read()" \
    && log "server $1 up (warmed)"; }
mkdir -p $R/runs
SWEEP=casual,creative,code,reasoning,json,switch,tools
run() { local lab=$1 cats=$2; shift 2; env "$@" CATS=$cats MAXTOK=300 python3 $R/run_suite.py $lab 2>&1 | grep -E "ALL|Traceback|Error" ; }
T=$R/runs
log "== identity: trace off vs on (dev split, sweep categories, seed 11)"
srvr I-off && run I-off $SWEEP SPLITS=dev
srvr I-on LLAMA_SPEC_TRACE=$T/trace-I.jsonl && run I-on $SWEEP SPLITS=dev
python3 - $T <<'PY'
import json, sys
a = [json.loads(l) for l in open(sys.argv[1] + "/I-off.jsonl")]; b = [json.loads(l) for l in open(sys.argv[1] + "/I-on.jsonl")]
same = sum(x["hash"] == y["hash"] for x, y in zip(a, b))
ta = sum(x["predicted_ms"] for x in a); tb = sum(y["predicted_ms"] for y in b)
print(f"identity: {same}/{len(a)} identical; decode time off {ta:.0f} ms, on {tb:.0f} ms ({100*(tb/ta-1):+.1f}%)")
PY
log "== traces: n-max 4, p-min 0 (no draft stop; production's p-min policy is replayable from the draft q), all splits"
srvr T-n4p0 PMIN=0.0 LLAMA_SPEC_TRACE=$T/trace-n4p0.jsonl && run T-n4p0 "" SPLITS=dev,val,test
log "== traces: n-max 8, p-min 0, all splits"
srvr T-n8p0 NMAX=8 PMIN=0.0 LLAMA_SPEC_TRACE=$T/trace-n8p0.jsonl && run T-n8p0 "" SPLITS=dev,val,test
log "== traces: n-max 16, p-min 0, 64k context (n-max 16's recurrent snapshots don't fit at 262k), all splits"
srvr T-n16p0 NMAX=16 PMIN=0.0 CTX=65536 LLAMA_SPEC_TRACE=$T/trace-n16p0.jsonl && run T-n16p0 "" SPLITS=dev,val,test
ls -la $T/trace-*.jsonl; wc -l $T/trace-*.jsonl
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | tr '\n' ' '; echo
log "window6 done"
