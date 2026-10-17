#!/bin/bash
# Variant of start-qwen.sh for images with small text (screenshots, code, documents): no JEV System 1, and images
# capped at the pixel area of 2560x1440 (3600 image tokens) instead of ~1 MP (1024 tokens), so a 1440p screenshot
# keeps its full resolution. 10-17 test (3 synthetic 1440p screenshots, codes at 9-20 px): exact rows at 11-12 px
# 0/12 -> 12/12, at 9-10 px 0/12 -> 7/12. 1440p image at 259k context: OK, peaks GPU0 14,583 / GPU1 15,799 MiB.
# Trade-off: a 1440p image takes ~17 s to first byte (~32 s at 255k) vs ~4 s at 1024 tokens; /v1/decide is not served.
# Usage: bash start-qwen-vision.sh [extra llama-server args]; JEV=1 or IMAGE_MAX_TOKENS=N override, other switches
# as in start-qwen.sh (which must sit in the same directory).
export JEV=${JEV:-0} IMAGE_MAX_TOKENS=${IMAGE_MAX_TOKENS-3600}
exec bash "$(dirname "$0")/start-qwen.sh" "$@"
