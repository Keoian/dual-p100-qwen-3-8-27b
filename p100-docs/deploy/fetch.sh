#!/bin/bash
# Download the pinned model, vision projector and JEV System 1 files, then convert the JEV LoRA (inside the container).
# Exact file names matter: "*Q6_K*" would also match the _M/_L/_XL variants (~95 GB).
set -e
MODELS=${MODELS:-/work/models}
JEV_DIR=${JEV_DIR:-/work/jev}
SRC=${SRC:-/work/src/llama.cpp}
QWEN_REV=4ca720788d1e01f1bff70c033e0d0028fd02e502   # unsloth/Qwen3.8-27B-GGUF
JEV_REV=51740a8891c2a8baefd969237fd44187b3e3a115    # autotrust/JEV-27B (System 1 files only, not the bf16 shards)

mkdir -p "$MODELS" "$JEV_DIR/gguf"
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q6_K.gguf mmproj-F16.gguf --revision "$QWEN_REV" --local-dir "$MODELS"
hf download autotrust/JEV-27B --revision "$JEV_REV" --local-dir "$JEV_DIR/JEV-27B" \
  adapter/adapter_config.json adapter/adapter_model.safetensors head.safetensors calibration.json config.json
python3 "$SRC/tools/jev-decide/convert_jev_lora.py" --adapter "$JEV_DIR/JEV-27B/adapter" \
  --config "$JEV_DIR/JEV-27B/config.json" --outtype f16 --out "$JEV_DIR/gguf/jev-27b-lora-f16.gguf"
chmod a-w "$MODELS"/*.gguf
cd / && sha256sum -c "$SRC/p100-docs/deploy/SHA256SUMS" --ignore-missing 2>&1 | sed 's/^/sha256: /'
