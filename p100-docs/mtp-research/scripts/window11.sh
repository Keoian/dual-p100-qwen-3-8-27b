#!/bin/bash
# Window 11 (10-24): the temperature-gated stop rule (bin-gate: LLAMA_SPEC_STOP_ISO=0.125, LLAMA_SPEC_STOP_ISO_TEMP
# default 0.5,0.9), then traces with thinking on for the refit.
# Part 1, exactness/gating check, deterministic arms: at target T 1.0 the gate keeps the p-min test, so thinking-on
# replies must match production's window 9 arms (A9-MP, A9-NP) token for token; at T 0.7 the rule runs, so thinking-off
# replies (all 72 prompts, as window 7) must match the rule's window 7 arm A7-L1 token for token.
# Part 2: LLAMA_SPEC_TRACE runs with thinking on (card thinking sampling, medium effort, p-min 0): traces tn4p0 / tn8p0.
set -u
M=/work/models/Qwen3.8-27B-UD-Q6_K.gguf; KEYW=/work/qwen-api-keys.work
BV=/work/bench/mtp3/bin-gate; R=/work/bench/mtp3
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
# a fresh server per arm, as in the reference arms
srvr A11-GM LLAMA_SPEC_STOP_ISO=0.125 && run A11-GM REASONING_EFFORT=medium MAXTOK=3000
srvr A11-GN LLAMA_SPEC_STOP_ISO=0.125 && run A11-GN MAXTOK=1500
srvr A11-GT LLAMA_SPEC_STOP_ISO=0.125 && env SPLITS=dev,val,test SAMPLING=nothink MAXTOK=300 python3 $R/run_suite.py A11-GT 2>&1 | grep -E "ALL|Traceback|Error"
python3 - <<'PY2'
import json
def load(l): return {(r["id"], r["seed"]): r for r in map(json.loads, open(f"/work/bench/mtp3/runs/{l}.jsonl"))}
def tps(rs): return sum(r["predicted_n"] for r in rs) / sum(r["predicted_ms"] for r in rs) * 1000
for new, ref in (("A11-GM", "A9-MP"), ("A11-GN", "A9-NP"), ("A11-GT", "A7-L1")):
    a, b = load(new), load(ref)
    ks = sorted(set(a) & set(b))
    same = sum(a[k]["hash"] == b[k]["hash"] or a[k]["text"].split("\u241e")[-1] == b[k]["text"] for k in ks)
    print(f"{new} vs {ref}: {same}/{len(ks)} replies identical; {tps([a[k] for k in ks]):.2f} vs {tps([b[k] for k in ks]):.2f} t/s")
PY2
log "gate check done"
# traces for the thinking-on refit: card thinking sampling, medium effort, all 72 prompts, p-min 0 (rule off), as window 6
T=$R/runs
srvr TT-n4p0 PMIN=0.0 LLAMA_SPEC_TRACE=$T/trace-tn4p0.jsonl && \
  env SPLITS=dev,val,test SAMPLING=kmic THINK=on REASONING_EFFORT=medium MAXTOK=3000 python3 $R/run_suite.py T-tn4p0 2>&1 | grep -E "ALL|Traceback|Error"
srvr TT-n8p0 NMAX=8 PMIN=0.0 LLAMA_SPEC_TRACE=$T/trace-tn8p0.jsonl && \
  env SPLITS=dev,val,test SAMPLING=kmic THINK=on REASONING_EFFORT=medium MAXTOK=3000 python3 $R/run_suite.py T-tn8p0 2>&1 | grep -E "ALL|Traceback|Error"
wc -l $T/trace-tn4p0.jsonl $T/trace-tn8p0.jsonl
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | tr '\n' ' '; echo
log "window11 done"
