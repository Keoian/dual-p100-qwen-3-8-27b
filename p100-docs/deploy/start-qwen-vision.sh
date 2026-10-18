#!/bin/bash
# Variant of start-qwen.sh for images with small text and a more accurate KV cache, at the full 262k context:
#   - no JEV System 1 (frees ~1.1-1.3 GiB per GPU)
#   - images capped at the pixel area of 2560x1440 (3600 image tokens) instead of ~1 MP (1024): small text stays legible
#   - Q8_0 vision projector (ggml-org mmproj-Qwen3.8-27B-Q8_0.gguf; -284 MiB on GPU1, within noise of BF16/F16)
#   - KV cache K q8_0 / V q4_0 (KLD vs f16 cache 0.00274 -> 0.00116), decoded by the q4p kernel's q8_0-K variant
#   - -ub 1024 (with the 262k q8_0/q4_0 cache, -ub 2048's larger GEMM scratch runs GPU1 out of memory on a 1440p image)
#   - -ctxcp 32, --cache-ram 24576 (more prompt snapshots in host RAM, see below)
#   - GGML_CUDA_FA_VEC_MAX_Q / GGML_CUDA_FA_GEMM_MINQ: 6-16 token batches to the vec kernel, 17+ to the GEMM path, so no
#     batch reaches the tile kernel's whole-cache f16 conversion (which grows with depth and ran out of memory at 255k)
# Measured 10-18 (P100 x2, 150 W, build-vision): gate PPL 2.6064, KLD vs f16 cache 0.00116 (q4_0/q4_0: 0.00274),
# vision suite 0.935 (BF16 projector 0.938); Ember 10 turns at 25k: TTFT median 474 ms, decode 36.5 t/s; fill to 255k
# 358 -> 144 t/s (q4_0/q4_0 at -ub 2048: 386 -> 144); 1440p screenshot at 258,866 tokens: prompt 31.5 s, decode 27.8 t/s,
# small text 96-100%, peaks GPU0 15,187 / GPU1 16,101 MiB. Image caps: 4096 tokens fits at 262k, 6000 does not.
# Needs build-vision from the vision-1440p branch (q4p/fold q8_0-K kernels) and fetch.sh's Q8_0 projector.
# Usage: bash start-qwen-vision.sh [extra llama-server args]; any of the env switches below can be overridden, the rest
# as in start-qwen.sh (which must sit in the same directory).
export JEV=${JEV:-0} IMAGE_MAX_TOKENS=${IMAGE_MAX_TOKENS-3600} CTK=${CTK:-q8_0} CTV=${CTV:-q4_0} UB=${UB:-1024}
export BUILD=${BUILD:-/work/src/llama.cpp/build-vision}
export MMPROJ=${MMPROJ-/work/models-vision/mmproj-Qwen3.8-27B-Q8_0.gguf}
export GGML_CUDA_FA_VEC_MAX_Q=${GGML_CUDA_FA_VEC_MAX_Q:-16} GGML_CUDA_FA_GEMM_MINQ=${GGML_CUDA_FA_GEMM_MINQ:-17}
# 48 GB host: 32 checkpoints (4.8 GB) and a 24 GiB RAM prompt cache. 10-18, 58k-token chat built turn by turn, branch
# back to turn 3: -ctxcp 4 re-processed 27,761 tokens (77.5 s), -ctxcp 32 30 tokens (0.5 s); answers byte-identical
export CTXCP=${CTXCP:-32} CACHE_RAM=${CACHE_RAM:-24576}
exec bash "$(dirname "$0")/start-qwen.sh" "$@"
