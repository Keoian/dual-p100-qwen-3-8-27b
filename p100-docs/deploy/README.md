# Deploy: the tyler-port production setup on a fresh machine

Brings a second machine to the state of the reference box: Qwen3.8-27B UD-Q6_K across two Tesla P100s at 262k
context, MTP speculative decoding, vision, and JEV-27B System 1 decisions (`POST /v1/decide`), served
OpenAI-compatible on port 8090. Reference box: Unraid host, ASUS H270 (GPU1 on a PCH Gen3 x4 link, no P2P),
4-core i7, 32 GB RAM, driver 580, Docker with the NVIDIA runtime.

Files here: `Dockerfile`, `run-container.sh` (host), `unraid-user-script.sh` or the optional
`unraid-user-script-stop-others.sh` (host, at array start), `build.sh`,
`fetch.sh`, `start-qwen.sh`, `verify.sh` (inside the container), `SHA256SUMS`.

## Recipe

**On the host** (paths are Unraid's; set `WORK` to any directory that becomes `/work` in the container):

```bash
git clone -b tyler-port https://github.com/Keoian/dual-p100-qwen-3-8-27b /mnt/user/qwen-dev/src/llama.cpp
cd /mnt/user/qwen-dev/src/llama.cpp/p100-docs/deploy
docker build -t qwen-dev:latest .
bash run-container.sh            # stops nothing: refuses to start if the GPUs are busy; sets -pm 1 and -pl 150
```

Then install `unraid-user-script.sh` with the User Scripts plugin, schedule "At Startup of Array" (the power limit
resets on every reboot; `START_SERVER=1` in it also starts the server).
Optional instead: `unraid-user-script-stop-others.sh` also stops other GPU containers first (`STOP_CONTAINERS`,
default `llm-dev`), waits until the GPUs are free, then starts the server detached and reports when it is up. The
plugin keeps scripts on the flash drive: `/boot/config/plugins/user.scripts/scripts/<name>/script`.

**Inside the container** (`docker exec -it qwen-dev bash`):

```bash
D=/work/src/llama.cpp/p100-docs/deploy
bash $D/build.sh build-exp                      # ~60-90 min cold on 4 cores, sm_60 only
bash $D/fetch.sh                                # model + mmproj + JEV files (pinned), LoRA conversion, sha256 check
umask 077; openssl rand -hex 32 > /work/qwen-api-keys   # one key per line, '#' lines ignored; never commit it
cp $D/start-qwen.sh /work/start-qwen.sh
setsid nohup bash /work/start-qwen.sh > /work/serve-8090.log 2>&1 &
until curl -sf localhost:8090/health; do sleep 5; done   # ~1 min to load
bash $D/verify.sh
```

Clients: `http://<host>:8090/v1`, model id `qwen3.8-27b`, header `Authorization: Bearer <key>`.
Built-in web UI at `/`; `/health` is open.

## What `start-qwen.sh` runs

```bash
LD_LIBRARY_PATH=/work/src/llama.cpp/build-exp/bin \
GGML_CUDA_GRAPHS_PRE_VOLTA=3 LLAMA_SPEC_SAMPLE_TEMP=1.0 LLAMA_SPEC_DRAFT_TOPK=20 \
build-exp/bin/llama-server -m /work/models/Qwen3.8-27B-UD-Q6_K.gguf \
  -ngl 99 -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 \
  -c 262144 -b 32768 -ub 2048 -np 1 -ctxcp 4 \
  --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.2 -ngld 99 -ubd 64 -ctkd q4_0 -ctvd q4_0 \
  --alias qwen3.8-27b --jinja --reasoning off --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0 \
  --slot-save-path /work/slots \
  --mmproj /work/models/mmproj-F16.gguf --image-max-tokens 1024 -mmdev CUDA1 \
  --api-key-file /work/qwen-api-keys \
  --jev-lora /work/jev/gguf/jev-27b-lora-f16.gguf --jev-head /work/jev/JEV-27B/head.safetensors \
  --jev-calib /work/jev/JEV-27B/calibration.json --jev-ctk q8_0 --jev-ctv q8_0 \
  --host 0.0.0.0 --port 8090
```

`GGML_CUDA_P2P` is deliberately not set: on a board without peer access it enables nothing, and the operator's rule
is no P2P with JEV loaded until a supervised test. Env switches of the script: `JEV=0` (no System 1),
`JEV_KV=q4_0` (smaller decision cache, ~67 MiB/GPU less, KL 3.5e-4 vs f16), `MMDEV=` (projector on the default
device), `MMPROJ=` (text only), `IMAGE_MAX_TOKENS=` (no cap), `REASONING=auto`, `ALIAS=`, `API_KEY_FILE=` (no key),
`CTX=`, `BUILD=`, `MODEL=`, `JEV_DIR=`. Why each choice: [QUICKSTART.md](../QUICKSTART.md).

## Pinned inputs

| what | source | revision |
|---|---|---|
| model `Qwen3.8-27B-UD-Q6_K.gguf`, projector `mmproj-F16.gguf` | huggingface `unsloth/Qwen3.8-27B-GGUF` | `4ca720788d1e01f1bff70c033e0d0028fd02e502` |
| JEV `adapter/`, `head.safetensors`, `calibration.json`, `config.json` | huggingface `autotrust/JEV-27B` | `51740a8891c2a8baefd969237fd44187b3e3a115` |
| JEV LoRA GGUF | `tools/jev-decide/convert_jev_lora.py --outtype f16` (numpy, safetensors, repo `gguf-py`) | deterministic, see `SHA256SUMS` |

## Host requirements and settings

- 2x Tesla P100-PCIE-16GB, NVIDIA driver with CUDA 12 support; CUDA **12.x** toolkit (13 cannot target sm_60).
- **GPU power limit 150 W** (`nvidia-smi -pm 1; nvidia-smi -i 0,1 -pl 150`). On the reference board a 262k prefill
  with both cards at 180 W hard-reset the host once; at 150 W the full-depth test passed. Costs ~3.5% decode.
- Nothing else on the GPUs: the model needs both cards (peaks at 262k with JEV + an image: GPU0 15.2, GPU1 16.2 of
  16.4 GB, ~115 MiB allocatable left on GPU1).
- THP: no setting needed (the server no longer advises huge pages on checkpoint buffers; tested with
  `enabled=always`, `defrag=madvise`).
- Check P2P on a new board with `nvidia-smi topo -m`; cards behind different root ports usually have none. With
  working P2P, `GGML_CUDA_P2P=1` may help, but that combination has not been tested with JEV.

## Acceptance checks

| check | expected on the reference board (150 W) |
|---|---|
| `verify.sh` | health ok; decide ~0.9 s; chat ~2-3 s; idle VRAM ~14.3 / 15.4 GB |
| `MODEL=/work/models/Qwen3.8-27B-UD-Q6_K.gguf ./tools/gate.sh` (with `ln -sfn build-exp build-opt`; server stopped) | PPL 2.6074 (band 2.6209 +/- 0.0199), tg256 ~26.8, FLASH_ATTN_EXT 3/3 |
| an image question | ~5-6 s for any size up to 10 MP (resized to ~1,020 tokens) |
| full depth (optional, ~25 min) | chat filled to ~255k with decisions + an image at 256k: no errors, GPU1 peak ~16.15 GB |

Records of how these numbers were measured: [CHANGES.md §15-16](../CHANGES.md), [FINDINGS.md](../FINDINGS.md).
