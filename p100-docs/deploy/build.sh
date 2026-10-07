#!/bin/bash
# Usage: bash build.sh [build-dir-name] [extra cmake flags...]   (inside the container)
# sm_60 ONLY: adding architectures silently disables the Pascal mmvq paths. Cold build ~60-90 min on 4 cores
# (~25 min on 14), ccache makes rebuilds fast. After editing a .cuh, touch the .cu files that include it.
set -e
SRC=${SRC:-/work/src/llama.cpp}
NAME=${1:-build-exp}; shift || true
cd "$SRC"
cmake -B "$NAME" -G Ninja \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=60 \
  -DGGML_CUDA_NCCL=OFF \
  -DGGML_CUDA_FA_QUANTS=all \
  -DGGML_CCACHE=ON \
  -DCMAKE_BUILD_TYPE=Release \
  "$@"
cmake --build "$NAME" -j"$(nproc)"
ls -lh "$NAME"/bin/libggml-cuda.so* | head -3
echo "OK: $SRC/$NAME/bin   (tools/gate.sh expects build-opt: ln -sfn $NAME build-opt)"
