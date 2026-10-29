#!/bin/bash
# start-qwen-vision.sh with the exact MTP accept rule as the default (--spec-accept exact; CHANGES §23): output
# distributed as the model's own sampling, as production ran before 10-25. Everything else as start-qwen-vision.sh,
# which must sit in the same directory. A request can still send "speculative.accept": "ratio:1.5".
# Usage: bash start-qwen-vision-exact.sh [extra llama-server args]
export SPEC_ACCEPT=exact
exec bash "$(dirname "$0")/start-qwen-vision.sh" "$@"
