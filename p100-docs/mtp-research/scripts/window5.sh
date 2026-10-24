#!/bin/bash
# Window 5 (10-24, mtp3 Phase 1): fixed-policy baselines B0-B5 on the production binary (build-vision), suite dev+val, seed 11
set -u
SRC=/work/src/llama.cpp; F=/work/bench/forks; M=/work/models/Qwen3.8-27B-UD-Q6_K.gguf; KEYW=/work/qwen-api-keys.work
BV=/work/src/llama.cpp/build-vision; R=/work/bench/mtp3
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
log "== B1 production (n-max 4, p-min 0.2, sampled drafts), all categories"
srvr B1 && run B1 ""
log "== B0 no MTP"
srvr B0 MTP=0 && run B0 ""
log "== B3 n-max sweep (p-min 0.2)"
for n in 1 2 3 6 8; do srvr B3-n$n NMAX=$n && run B3-n$n $SWEEP; done
log "== B4 p-min sweep (n-max 4)"
for p in 0.0 0.1 0.4 0.6; do srvr B4-p$p PMIN=$p && run B4-p$p $SWEEP; done
log "== B5 greedy drafts"
srvr B5-greedy SPEC_TEMP=0 && run B5-greedy $SWEEP
log "== B1 vs B2 (Discord reference, adapted) with the Discord post's sampling (1.0/0.95/20/0)"
srvr B1k && run B1k $SWEEP SAMPLING=kmic
pkill -x llama-server; gpu_free; cool
(GGML_CUDA_P2P=1 GGML_CUDA_GRAPHS_PRE_VOLTA=3 LLAMA_SPEC_SAMPLE_TEMP=1.0 LLAMA_SPEC_DRAFT_TOPK=20 LD_LIBRARY_PATH=$BV/bin \
  setsid nohup $BV/bin/llama-server -m $M -ngl 99 -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -c 262144 -b 32768 -ub 2048 -np 1 \
  --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.2 -ngld 99 -ubd 64 -ctkd q4_0 -ctvd q4_0 \
  --jinja --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0 --api-key-file $KEYW --host 127.0.0.1 --port 8090 > $R/runs/srv-B2k.log 2>&1 &)
up B2k && run B2k $SWEEP SAMPLING=kmic
log "== B1 repeat (drift check)"
srvr B1r && run B1r $SWEEP
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | tr '\n' ' '; echo
log "window5 done"
