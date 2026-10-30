#!/bin/bash
# mtp5 research copy of /work/start-qwen.sh (2026-10-29): + CTKD / CTVD (draft KV types), HOST (bind address)
# Qwen3.8-27B Q6_K on both P100s for Ember: llama-server on :8090. Its defaults are the earlier JEV setup (build-exp,
# q4_0/q4_0, F16 projector); production runs it through start-qwen-vision.sh, which overrides them.
# Board notes (p100-docs/CHANGES.md §15): GPU1 sits on a PCH x4 link and the cards have no P2P, so
# GGML_CUDA_P2P is not set (it would enable nothing); tensor-parallel exchanges are staged through
# pinned host memory by this branch's AllReduce. -sm tensor still beats -sm layer here (same build: decode 26.6 vs 18.2).
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
# THINK_BY_BUDGET=1: the template always renders in thinking mode and "thinking off" is its empty-think generation prompt,
# so a request that toggles thinking keeps the cached prompt (Qwen3.8 puts its effort line at the top of the system
# prompt: a toggle reprocessed the whole chat, 2,146 -> 21 tokens); reasoning_effort low/medium = a 512/2048-token budget
THINK_BY_BUDGET=${THINK_BY_BUDGET:-1}
# SAMPLING_BY_MODE=1: the model card's sampling per thinking mode, for fields a request leaves out: thinking 1.0/0.95/20/0;
# thinking off 0.7/0.8/20/0 with the card's presence_penalty 1.5 (against endless repetition; it sees the last 64 tokens).
# It cost 7-10% decode until 0d920143d removed the vocabulary scan, and still ~2.2% at natural length until 910d2cfb8
# let the sampler's top-k prefilter run with penalties (CHANGES §21).
# THINK_SAMPLING / NOTHINK_SAMPLING: JSON objects of sampling fields for each mode; the chat endpoints use them in
# place of the CLI --temp/--top-p/--top-k/--min-p below, which then only serve /completion.
SAMPLING_BY_MODE=${SAMPLING_BY_MODE:-1}
NOTHINK_SAMPLING=${NOTHINK_SAMPLING:-'{"temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0.0,"presence_penalty":1.5}'}
THINK_SAMPLING=${THINK_SAMPLING:-'{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0}'}
# model id clients see in /v1/models (instead of the file path)
ALIAS=${ALIAS:-qwen3.8-27b}
# API keys, one per line (Authorization: Bearer <key>); API_KEY_FILE= (empty) serves without a key
API_KEY_FILE=${API_KEY_FILE-/work/qwen-api-keys}
AUTH=(); [ -n "$API_KEY_FILE" ] && [ -f "$API_KEY_FILE" ] && AUTH=(--api-key-file "$API_KEY_FILE")
# vision projector on GPU1: with JEV loaded either card ends ~230 MiB short of full at 256k + image (10-07 deep test:
# projector on GPU1 -> peaks GPU0 15,217 / GPU1 16,153 MiB; on GPU0 would mirror that); GPU1 is the tested placement. MMDEV= = default device
MMDEV=${MMDEV-CUDA1}
[ ${#VISION[@]} -gt 0 ] && [ -n "$MMDEV" ] && VISION+=(-mmdev "$MMDEV")
# System-2 KV cache types (CTK/CTV). Pascal fast kernels exist for q4_0/q4_0 and q8_0 K / q4_0 V;
# other types fall back to generic kernels and need GGML_CUDA_FA_VEC_MAX_Q / GGML_CUDA_FA_GEMM_MINQ (QUICKSTART). UB = -ub
CTK=${CTK:-q4_0}
CTV=${CTV:-q4_0}
# context checkpoints per slot (149.6 MiB of host RAM each on this model) and the RAM prompt cache in MiB (server default 8192)
CTXCP=${CTXCP:-4}
# logical batch (-b). A dropped client is noticed only between batches: at 32768 an abandoned 64k prompt held the slot
# ~33 s, at 2048 ~2 s; output, MTP acceptance and speed are identical (same 1024/2048-token ubatches)
NBATCH=${NBATCH:-32768}
CACHE=(); [ -n "${CACHE_RAM:-}" ] && CACHE=(--cache-ram "$CACHE_RAM")
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
# AR_BF16=1: the tensor-parallel exchanges of decode and MTP verify (the host-staged path, up to 1 MB) cross PCIe as bf16
# instead of f32 (GGML_CUDA_AR_BF16_THRESHOLD=1; the branch's default on Pascal without P2P is exact f32). Lossy: each
# GPU's partial sums are rounded to bf16 before they are added. 10-24, production setup: MTP replies 49.45 -> 52.80 t/s
# (+6.8%), pp5 +7.7%, tg128 +1.6%; KLD vs an f16 cache at -ub 5 0.001090 -> 0.001102 (+1.1%), same top 99.13 -> 99.08%.
# AR_BF16=0: exact f32 exchanges
AR_BF16=${AR_BF16:-1}
export GGML_CUDA_AR_BF16_THRESHOLD=${GGML_CUDA_AR_BF16_THRESHOLD:-$AR_BF16}
# STOP_ISO=<theta>: the MTP draft stops by a lookup of how likely the drafted tokens are to be accepted (LLAMA_SPEC_STOP_ISO,
# CHANGES §22) instead of the p-min test, for requests at any target temperature > 0 outside the thinking window
# (thinking off, or a client's own temperature; LLAMA_SPEC_STOP_ISO_TEMP). Exact: same output distribution. 10-24: chat
# replies 49.30 -> 51.61 t/s (+4.7%), long context 32k-200k +4.8%, T 0.3 / 1.2 +6.6% / +6.1%.
# STOP_ISO=0: p-min for those requests.
# STOP_ISO_THINK=<theta>: the same rule with tables fit on thinking-on traces, for requests at target temperature
# 0.95-1.05 (thinking on; LLAMA_SPEC_STOP_ISO_THINK_TEMP). 10-24: medium effort, 72 prompts, +2.9%.
# STOP_ISO_THINK=0: p-min for thinking requests
STOP_ISO=${STOP_ISO:-0.125}
STOP_ISO_THINK=${STOP_ISO_THINK:-0.15}
export LLAMA_SPEC_STOP_ISO=${LLAMA_SPEC_STOP_ISO:-$STOP_ISO} LLAMA_SPEC_STOP_ISO_THINK=${LLAMA_SPEC_STOP_ISO_THINK:-$STOP_ISO_THINK}
# SPEC_ACCEPT: how the target accepts the MTP draft's tokens for requests that don't send "speculative.accept"
# (--spec-accept, CHANGES §23). exact: the speculative-sampling rule, output distributed as the model's own sampling.
# escape: keep a drafted token unless the model's own sampler gives it probability 0 (after the request's penalties,
# top-k, top-p, min-p). APPROXIMATE; 10-25: chat 51.87 -> 59.02 t/s (+13.8%), thinking +31.5%, long context +12-14%,
# no measurable quality loss in the tests. Requests under escape draft without the stop tables above (fit for exact).
# ratio:<c>[:<C>:<N>]: keep a draft with probability min(1, c p/q), else the exact residual: no token more than c times as
# likely as the model's own sampling (CHANGES §24; production's default 10-29 to 10-30). Also drafts without the stop tables.
SPEC_ACCEPT=${SPEC_ACCEPT:-exact}
# CUDA graphs for the single-token MTP draft steps only; sampled drafts verified with the speculative-sampling rule
export GGML_CUDA_GRAPHS_PRE_VOLTA=3 LLAMA_SPEC_SAMPLE_TEMP=1.0 LLAMA_SPEC_DRAFT_TOPK=20
export LLAMA_THINK_BY_BUDGET="$THINK_BY_BUDGET" LLAMA_SAMPLING_BY_MODE="$SAMPLING_BY_MODE" LLAMA_SAMPLING_NOTHINK="$NOTHINK_SAMPLING" LLAMA_SAMPLING_THINK="$THINK_SAMPLING"
exec "$BUILD/bin/llama-server" -m "$MODEL" \
  -ngl 99 -sm tensor -fa 1 -ctk "$CTK" -ctv "$CTV" \
  -c "$CTX" -b "$NBATCH" -ub "${UB:-2048}" -np 1 -ctxcp "$CTXCP" "${CACHE[@]}" \
  --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.2 --spec-accept "$SPEC_ACCEPT" \
  -ngld 99 -ubd 64 -ctkd "${CTKD:-q4_0}" -ctvd "${CTVD:-q4_0}" \
  --alias "$ALIAS" --jinja --reasoning "$REASONING" --temp 0.7 --top-k 20 --top-p 0.8 --min-p 0.0 \
  --slot-save-path /work/slots "${VISION[@]}" "${AUTH[@]}" "${JEVARGS[@]}" \
  --host "${HOST:-0.0.0.0}" --port 8090 "$@"
