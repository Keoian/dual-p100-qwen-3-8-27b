#!/bin/bash
# start-qwen-vision.sh with the exact MTP accept rule pinned (--spec-accept exact; CHANGES §23): output distributed as
# the model's own sampling. Since 10-30 start-qwen-vision.sh defaults to exact too; this copy also ignores a
# SPEC_ACCEPT set in the environment. Everything else as start-qwen-vision.sh, which must sit in the same directory.
# A request can still send "speculative.accept": "ratio:1.5".
# Usage: bash start-qwen-vision-exact.sh [extra llama-server args]
export SPEC_ACCEPT=exact
exec bash "$(dirname "$0")/start-qwen-vision.sh" "$@"
