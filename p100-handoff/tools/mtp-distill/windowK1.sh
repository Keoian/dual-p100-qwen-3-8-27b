#!/bin/bash
# windowK1.sh (MTP_ACCURACY task, Part 1 + Part 2a data; production window in the pre-approved night hours, 06:00 UTC).
# Every arm runs under EXACT acceptance (server default exact + "speculative.accept": "exact" per request); production's
# default rule (ratio:1.5) is untouched. Binary bin-p1 = f0dae16cc + working-tree hooks, all default off:
#   LLAMA_SPEC_DRAFT_REQ=1  1c: the draft samples with the request's chain (presence, top-k, top-p, min-p, temperature)
#   LLAMA_SPEC_LOG          per-cycle accepted / drafted counts (acceptance by depth)
#   LLAMA_MTP_DUMP          the MTP head's inputs/outputs + the target's top-128 per decoded row (parity, offline overlap)
# Configs: B baseline, P parity (1c), K draft KV q8_0/q8_0 (1a), M MTP block Q8_0 from BF16 (1b, models/...-mtpq8.gguf,
# sha256 a84a3bd2...), H the draft's own LM-head copy in q8_0 instead of q4_0 (LLAMA_MTP_DRAFT_VOCAB_TYPE=same),
# W that copy over 98304 vocab rows instead of 81920 (LLAMA_MTP_DRAFT_VOCAB=98304; OPTLOG 226 never measured either).
# SIMULATED (parity_sim.py): P chat +1.8% (+0.6..+2.8), thinking +0.0%; vocab_sim.py: served mass on ids >= 81920
# chat 1.44% / thinking 1.08% per drafted row (>= 98304: 0.48 / 0.32%, half of it <|im_end|> and </think>).
#   0. VRAM at production depth per config (250k chat restored without its draft cache, 1440p image + follow-up) and
#      greedy identity (24 test prompts, T 0) vs the production binary
#   1. dump run (bin-p1, B settings, cache_prompt off; not timed): chat + thinking test prompts
#   2. timed, 72 prompts per mode: chat B P K M H W W H M K P B; thinking B P P B, then H W M K once each (each
#      skipped after 11:30 UTC)
#   3. full 250k fill + image with K (draft KV q8_0/q8_0: the knob that changes the draft's attention at depth), if
#      before 11:45 UTC: checks the restore-based peaks (no draft cache) against a filled draft cache
#   4. parity + paired offline overlap (torch on the freed GPUs, out-parity-K1.md / out-ovl-K1.md), if before 12:15 UTC
# Production stopped (chat saved), restored by the EXIT trap. Cooling: >= 70C -> < 60C. Nothing after 12:30 UTC.
set -u
. /work/bench/prodctl.sh
KEYW=/work/qwen-api-keys.work
R=/work/bench/mtp5; SRV=$R/srv/start-qwen-vision.sh; RUNS=$R/runs
BIN=$R/bin-p1; PROD=/work/src/llama.cpp/build-vision
MQ8=$R/models/Qwen3.8-27B-UD-Q6_K-mtpq8.gguf
log() { echo "[$(date -u +%H:%M:%S)Z] $*"; }
after() { [ "$(date -u +%H%M)" -ge "$1" ]; }
late() { after 1230; }
gpu_free() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')" -lt 100 ]; do sleep 2; done; }
temps() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | tr '\n' ' '; }
cool() { local t; t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
  if [ "$t" -ge 70 ]; then log "GPU ${t}C: cooling to < 60C"
    until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)" -lt 60 ]; do sleep 10; done; fi; }
mkdir -p $RUNS
( while sleep 10; do nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1; done > $RUNS/temps-K1.txt ) &
SAMPLER=$!
restore() {
  kill $SAMPLER 2>/dev/null
  log "peak GPU temperature: $(sort -n $RUNS/temps-K1.txt | tail -1)C"
  log "restoring production"; pkill -x llama-server; gpu_free
  (setsid nohup bash /work/start-qwen-vision.sh > /work/serve-8090.log 2>&1 &)
  for i in $(seq 120); do curl -sf localhost:8090/health >/dev/null && break; sleep 5; done
  curl -sf localhost:8090/health >/dev/null && { log "production up"; prod_restore; } || log "PRODUCTION NOT UP"
  rm -f "$KEYW"; rm -f /work/slots/k1-250k.bin /work/slots/k1-250k.bin.ckpt /work/slots/k1-250k.bin.spec
  log "temp key removed"
  echo "- $(date -u +%Y-%m-%d) windowK1 production window ended $(date -u +%H:%M) UTC (log mtp5/windowK1.log)" >> /work/bench/LOG.md
}
python3 -c 'import secrets; print(secrets.token_hex(24))' > "$KEYW"; chmod 600 "$KEYW"
trap restore EXIT
busy() { python3 -c "
import json,urllib.request; k=[l.strip() for l in open('/work/qwen-api-keys') if l.strip() and not l.startswith('#')][0]
s=json.loads(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8090/slots', headers={'Authorization':'Bearer '+k}), timeout=10).read())
print(int(any(x.get('is_processing') for x in s)))" 2>/dev/null || echo 0; }
for i in $(seq 90); do [ "$(busy)" = 0 ] && break; [ $i = 1 ] && log "production busy, waiting"; sleep 10; done
prod_save
log "stopping production"; pkill -x llama-server; gpu_free; cool
echo "- $(date -u +%Y-%m-%d) windowK1 production window started $(date -u +%H:%M) UTC (MTP_ACCURACY Part 1, exact)" >> /work/bench/LOG.md
# the 250k chat without its draft cache: a draft cache of another KV type would not load
ln -sf /work/slots/longctx-250k.bin /work/slots/k1-250k.bin
ln -sf /work/slots/longctx-250k.bin.ckpt /work/slots/k1-250k.bin.ckpt
ln -sf /work/slots/longctx-250k.bin.spec /work/slots/k1-250k.bin.spec
up() { for i in $(seq 120); do curl -sf localhost:8090/health >/dev/null && break; sleep 5; done
  curl -sf localhost:8090/health >/dev/null || { log "server $1 did not come up"; tail -5 $RUNS/srv-$1.log; return 1; }
  python3 -c "
import json,urllib.request; k=[l.strip() for l in open('$KEYW') if l.strip()][0]
urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8090/v1/chat/completions', json.dumps(dict(messages=[{'role':'user','content':'Hi'}],max_tokens=16)).encode(), {'Content-Type':'application/json','Authorization':'Bearer '+k}), timeout=600).read()" \
    && log "server $1 up (warmed), GPUs $(temps)C, VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')MiB"; }
# srvr <label> <build dir> [ENV=val ...]: production's settings (research copy of the start scripts), exact by default
srvr() { local lab=$1 b=$2; shift 2; pkill -x llama-server; gpu_free; cool; rm -f $RUNS/speclog-$lab.txt
  (env "$@" BUILD=$b API_KEY_FILE=$KEYW HOST=127.0.0.1 SPEC_ACCEPT=exact LLAMA_SPEC_LOG=$RUNS/speclog-$lab.txt \
     setsid nohup bash $SRV > $RUNS/srv-$lab.log 2>&1 &)
  up $lab; }
# suite <label> [ENV=val ...]: run_suite.py with the given settings, exact per request
suite() { local lab=$1; shift; env "$@" SPEC_ACCEPT=exact SPEC_LOG=$RUNS/speclog-$lab.txt python3 $R/run_suite.py $lab 2>&1 | grep -E "ALL|Traceback|Error"; }
greedy() { suite G-$1 SPLITS=test SAMPLING=greedy MAXTOK=200 CACHE_PROMPT=0; }
chat()   { suite $1 SPLITS=dev,val,test SAMPLING=nothink MAXTOK=300; }
think()  { suite $1 SPLITS=dev,val,test SAMPLING=kmic THINK=on REASONING_EFFORT=medium MAXTOK=3000; }
peak()   { ( exec nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -lms 200 > $RUNS/peak-V-$1.csv ) & PK=$!; }
vram()   { local lab=$1; peak $lab
  python3 $R/vram_depth.py k1-250k.bin $lab 2>&1 | tail -4; kill $PK 2>/dev/null; }
arm() { if late; then log "skip $1 (past 12:30 UTC)"; return 1; fi; log "arm $1"; }
envof() { case $1 in B) echo "X=0";; P) echo "LLAMA_SPEC_DRAFT_REQ=1";; K) echo "CTKD=q8_0 CTVD=q8_0";; M) echo "MODEL=$MQ8";;
  H) echo "LLAMA_MTP_DRAFT_VOCAB_TYPE=same";; W) echo "LLAMA_MTP_DRAFT_VOCAB=98304";; esac; }

# ---- 0. VRAM at depth + greedy identity, one server per config
arm V-prod && srvr V-prod $PROD X=0 && { vram prod; greedy prod; }
for c in B P K M H W; do
  arm V-$c && srvr V-$c $BIN $(envof $c) && { [ $c != P ] && vram $c; greedy $c; log "V-$c done, GPUs $(temps)C"; }
done
python3 $R/analyze_k1.py 2>/dev/null | sed -n '/Greedy identity/,$p'

# ---- 1. dump run (not timed): inputs/outputs of the MTP head + target top-128, chat and thinking test prompts
arm D-dump && srvr D-dump $BIN LLAMA_MTP_DUMP=$RUNS/dump-K1.bin && {
  suite D-chat SPLITS=test SAMPLING=nothink MAXTOK=300 CACHE_PROMPT=0
  suite D-think SPLITS=test SAMPLING=kmic THINK=on REASONING_EFFORT=medium MAXTOK=1200 CACHE_PROMPT=0
  log "dump run done: $(du -h $RUNS/dump-K1.bin | cut -f1), GPUs $(temps)C"; }

# ---- 2. timed arms (ABBA)
for a in B1 P1 K1 M1 H1 W1 W2 H2 M2 K2 P2 B2; do c=${a:0:1}
  arm C-$a && srvr C-$a $BIN $(envof $c) && { chat C-$a; log "C-$a done, GPUs $(temps)C"; }
done
for a in B1 P1 P2 B2; do c=${a:0:1}
  arm T-$a && srvr T-$a $BIN $(envof $c) && { think T-$a; log "T-$a done, GPUs $(temps)C"; }
done
for a in H1 W1 M1 K1; do c=${a:0:1}
  after 1130 && { log "skip T-$a (past 11:30 UTC)"; continue; }
  arm T-$a && srvr T-$a $BIN $(envof $c) && { think T-$a; log "T-$a done, GPUs $(temps)C"; }
done

# ---- 3. full fill to 250k + image with K: the draft cache filled too
if ! after 1145; then
  arm F-K && srvr F-K $BIN CTKD=q8_0 CTVD=q8_0 && {
    peak Kfill
    KEYFILE=$KEYW FILL=250000 timeout 2400 python3 /work/bench/vision1440/deep.py 2>&1 | tail -4; kill $PK 2>/dev/null
    log "F-K done, GPUs $(temps)C"; }
fi
# ---- 4. offline checks on the freed GPUs (torch; minutes here vs hours on the CPU): parity on >= 1,000 draft steps,
#         paired overlap of every knob on the dump's rows (request 0 = warm-up; chat 1-24, thinking 25-48)
if ! after 1215; then
  pkill -x llama-server; gpu_free; log "offline checks on the GPUs"
  timeout 1200 /work/venv-mtp5/bin/python $R/parity.py $RUNS/dump-K1.bin --device cuda:0 --max-req 13 --out $R/out-parity-K1.md 2>&1 | grep -v "^request" | tail -13
  timeout 1200 /work/venv-mtp5/bin/python $R/parity.py $RUNS/dump-K1.bin --device cuda:0 --max-req 13 --emulate gpu --out $R/out-parity-K1-emu.md 2>&1 | grep -v "^request" | tail -13
  timeout 2400 /work/venv-mtp5/bin/python $R/ovl_eval.py $RUNS/dump-K1.bin --reqs chat:1-25,think:25-49 --device cuda:1 --head-device cuda:0 --out $R/out-ovl-K1.md 2>&1 | grep -v "^request" | tail -30
  log "offline checks done"
fi
python3 $R/analyze_k1.py > $R/out-k1.md 2>&1; cat $R/out-k1.md
log "windowK1 done"
