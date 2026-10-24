#!/bin/bash
# Window 7 (10-24, mtp3 Phase 4): measured A/B of the lookup stop rule (LLAMA_SPEC_STOP_ISO) vs production and the best fixed p-min
set -u
SRC=/work/src/llama.cpp; F=/work/bench/forks; M=/work/models/Qwen3.8-27B-UD-Q6_K.gguf; KEYW=/work/qwen-api-keys.work
BV=/work/bench/mtp3/bin-iso; R=/work/bench/mtp3
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
ALLS="SPLITS=dev,val,test"
for round in 1 2; do
  if [ $round = 1 ]; then order="P L Q"; else order="Q L P"; fi
  for arm in $order; do
    case $arm in
      P) srvr A7-P$round && run A7-P$round "" $ALLS ;;
      Q) srvr A7-Q$round PMIN=0.5 && run A7-Q$round "" $ALLS ;;
      L) srvr A7-L$round LLAMA_SPEC_STOP_ISO=0.125 && run A7-L$round "" $ALLS ;;
    esac
  done
done
python3 - <<'PY'
import json, statistics
def load(l): return [json.loads(x) for x in open(f"/work/bench/mtp3/runs/{l}.jsonl")]
def agg(rs):
    n = sum(r["predicted_n"] for r in rs); ms = sum(r["predicted_ms"] for r in rs); return n / ms * 1000, n
for sub, f in (("all", lambda r: True), ("test", lambda r: r["split"] == "test")):
    res = {}
    for a in "PQL":
        rs = [r for k in (1, 2) for r in load(f"A7-{a}{k}") if f(r)]
        res[a] = agg(rs)
    print(sub, " ".join(f"{a} {res[a][0]:.2f} t/s ({res[a][1]} tok)" for a in "PQL"),
          f"| L vs P {100*(res['L'][0]/res['P'][0]-1):+.1f}%, L vs Q {100*(res['L'][0]/res['Q'][0]-1):+.1f}%, Q vs P {100*(res['Q'][0]/res['P'][0]-1):+.1f}%")
PY
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | tr '\n' ' '; echo
log "window7 done"
