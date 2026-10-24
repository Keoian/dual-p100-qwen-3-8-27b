#!/bin/bash
# Window 13 (10-24): counterfactual traces for a stop-rule table at every sampling setting. bin-trace2 (research branch
# + LLAMA_SPEC_TRACE_TOPK: the target's top-128 raw logits per verify row, log-sum-exp, penalty window), rule off,
# p-min 0, n-max 4, all 72 prompts:
#   cfchat:  thinking off, production sampling (0.7 / 0.8 / 20 / 0, presence 1.5), cap 300
#   cfthink: thinking on, medium effort, card thinking sampling (1.0 / 0.95 / 20 / 0), cap 3000
# After cfchat, counterfactual.py selfcheck recomputes every drafted token's target p at the run's own sampling from
# the raw logits; if more than 1% of rows are off by > 1e-4 the window stops (the hook or the library is wrong).
set -u
M=/work/models/Qwen3.8-27B-UD-Q6_K.gguf; KEYW=/work/qwen-api-keys.work
BV=/work/bench/mtp3/bin-trace2; R=/work/bench/mtp3
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
T=$R/runs; CF=/work/src/llama.cpp/p100-docs/mtp-research/analysis
check() { python3 -c "
import sys; sys.path.insert(0, '$CF'); import counterfactual as c
o = c.selfcheck('$1'); sys.exit(0 if o['rows'] > 100 and o['frac_1e4'] < 0.01 else 3)"; }
rm -f $T/trace-cfchat.jsonl $T/trace-cfthink.jsonl
srvr TC-cfchat PMIN=0.0 LLAMA_SPEC_TRACE=$T/trace-cfchat.jsonl LLAMA_SPEC_TRACE_TOPK=128 && \
  env SPLITS=dev,val,test SAMPLING=nothink MAXTOK=300 python3 $R/run_suite.py T-cfchat 2>&1 | grep -E "ALL|Traceback|Error"
wc -l $T/trace-cfchat.jsonl; ls -la $T/trace-cfchat.jsonl | awk '{print $5}'
if ! check $T/trace-cfchat.jsonl; then log "SELFCHECK FAILED on cfchat: stopping before cfthink"; exit 1; fi
log "selfcheck cfchat ok"
srvr TC-cfthink PMIN=0.0 LLAMA_SPEC_TRACE=$T/trace-cfthink.jsonl LLAMA_SPEC_TRACE_TOPK=128 && \
  env SPLITS=dev,val,test SAMPLING=kmic THINK=on REASONING_EFFORT=medium MAXTOK=3000 python3 $R/run_suite.py T-cfthink 2>&1 | grep -E "ALL|Traceback|Error"
wc -l $T/trace-cfthink.jsonl; ls -la $T/trace-cfthink.jsonl | awk '{print $5}'
# the cfthink selfcheck runs after the window (production up): python3 $CF/counterfactual.py selfcheck <trace>
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | tr '\n' ' '; echo
log "window13 done"
