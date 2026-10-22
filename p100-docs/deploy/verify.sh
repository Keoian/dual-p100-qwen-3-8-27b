#!/bin/bash
# Smoke test of a running server on :8090 (inside the container): health, a JEV decision (when JEV is loaded), a chat
# turn, VRAM. Expected on the reference board (2x P100, 150 W): decide ~0.9 s, chat ~2-3 s.
set -e
until curl -sf localhost:8090/health >/dev/null; do sleep 5; done   # model load takes ~1 min
KEY=$(grep -v '^#' "${API_KEY_FILE:-/work/qwen-api-keys}" | grep -m1 .)
H=(-H "Content-Type: application/json" -H "Authorization: Bearer $KEY")
curl -s localhost:8090/health; echo
DEC=$(curl -s localhost:8090/v1/decide "${H[@]}" -d '{"kind":"noul","state":"The parcel arrived damaged and the customer wants their money back.","question":"Is the customer asking for a refund?"}')
if echo "$DEC" | jq -e '.choice' >/dev/null 2>&1; then echo "$DEC" | jq -c '{choice, confidence}'; else echo "/v1/decide not served (JEV off): skipped"; fi
time curl -s localhost:8090/v1/chat/completions "${H[@]}" -d '{"messages":[{"role":"user","content":"In one sentence, what is a Tesla P100?"}],"max_tokens":40,"temperature":0}' | jq -r '.choices[0].message.content'
nvidia-smi --query-gpu=index,memory.used,power.limit --format=csv,noheader
