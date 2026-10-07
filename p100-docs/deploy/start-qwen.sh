#!/bin/bash
# Qwen3.8-27B Q6_K on both P100s for Ember: llama-server on :8090 (tyler-port build).
# Board notes (p100-docs/CHANGES.md §15): GPU1 sits on a PCH x4 link and the cards have no P2P, so
# GGML_CUDA_P2P is not set (it would enable nothing); tensor-parallel exchanges are staged through
# pinned host memory by the tyler-port AllReduce. -sm tensor still beats -sm layer here (same build: decode 26.6 vs 18.2).
# Usage: bash start-qwen.sh [extra llama-server args]. Expects the layout of p100-docs/deploy/README.md (/work/...).
set -e
BUILD=${BUILD:-/work/src/llama.cpp/build-exp}
MODEL=${MODEL:-/work/models/Qwen3.8-27B-UD-Q6_K.gguf}
CTX=${CTX:-262144}
# vision projector (unsloth/Qwen3.8-27B-GGUF mmproj-F16.gguf); MMPROJ= (empty) serves text only
MMPROJ=${MMPROJ-/work/models/mmproj-F16.gguf}
VISION=(); [ -n "$MMPROJ" ] && [ -f "$MMPROJ" ] && VISION=(--mmproj "$MMPROJ")
# images are capped at IMAGE_MAX_TOKENS (a 2875x1500 picture was ~4100 tokens, ~19 s before the first byte);
# IMAGE_MAX_TOKENS= (empty) uses the model's own limit
IMAGE_MAX_TOKENS=${IMAGE_MAX_TOKENS-1024}
[ ${#VISION[@]} -gt 0 ] && [ -n "$IMAGE_MAX_TOKENS" ] && VISION+=(--image-max-tokens "$IMAGE_MAX_TOKENS")
# thinking off by default: chat apps that hide reasoning_content otherwise show nothing for a minute and time out.
# A request can still enable it (chat_template_kwargs {"enable_thinking": true}); REASONING=auto = model default
REASONING=${REASONING:-off}
# model id clients see in /v1/models (instead of the file path)
ALIAS=${ALIAS:-qwen3.8-27b}
# API keys, one per line (Authorization: Bearer <key>); API_KEY_FILE= (empty) serves without a key
API_KEY_FILE=${API_KEY_FILE-/work/qwen-api-keys}
AUTH=(); [ -n "$API_KEY_FILE" ] && [ -f "$API_KEY_FILE" ] && AUTH=(--api-key-file "$API_KEY_FILE")
# vision projector on GPU1: with JEV loaded either card ends ~230 MiB short of full at 256k + image (10-07 deep test:
# projector on GPU1 -> peaks GPU0 15,217 / GPU1 16,153 MiB; on GPU0 would mirror that); GPU1 is the tested placement. MMDEV= = default device
MMDEV=${MMDEV-CUDA1}
[ ${#VISION[@]} -gt 0 ] && [ -n "$MMDEV" ] && VISION+=(-mmdev "$MMDEV")
# JEV-27B System 1 decisions on POST /v1/decide (separate context on the same model, LoRA bound there only);
# JEV=0 serves without it. JEV_KV = decision-context cache type (q8_0 ~= f16 accuracy; q4_0 saves ~67 MiB per GPU)
JEV=${JEV:-1}
JEV_DIR=${JEV_DIR:-/work/jev}
JEV_KV=${JEV_KV:-q8_0}
JEVARGS=()
if [ "$JEV" != 0 ] && [ -f "$JEV_DIR/gguf/jev-27b-lora-f16.gguf" ]; then
  JEVARGS=(--jev-lora "$JEV_DIR/gguf/jev-27b-lora-f16.gguf" --jev-head "$JEV_DIR/JEV-27B/head.safetensors"
           --jev-calib "$JEV_DIR/JEV-27B/calibration.json" --jev-ctk "$JEV_KV" --jev-ctv "$JEV_KV")
fi
mkdir -p /work/slots
export LD_LIBRARY_PATH="$BUILD/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# CUDA graphs for the single-token MTP draft steps only; sampled drafts verified with the speculative-sampling rule
export GGML_CUDA_GRAPHS_PRE_VOLTA=3 LLAMA_SPEC_SAMPLE_TEMP=1.0 LLAMA_SPEC_DRAFT_TOPK=20
exec "$BUILD/bin/llama-server" -m "$MODEL" \
  -ngl 99 -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 \
  -c "$CTX" -b 32768 -ub 2048 -np 1 -ctxcp 4 \
  --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.2 \
  -ngld 99 -ubd 64 -ctkd q4_0 -ctvd q4_0 \
  --alias "$ALIAS" --jinja --reasoning "$REASONING" --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0 \
  --slot-save-path /work/slots "${VISION[@]}" "${AUTH[@]}" "${JEVARGS[@]}" \
  --host 0.0.0.0 --port 8090 "$@"
