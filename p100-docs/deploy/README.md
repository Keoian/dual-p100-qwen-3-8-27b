# Deploy: the production setup on a fresh machine

Brings a second machine to the state of the reference box: Qwen3.8-27B UD-Q6_K across two Tesla P100s at the full
262,144-token context with a q8_0 K / q4_0 V cache, MTP speculative decoding, the Q8_0 vision projector with images up
to 1440p, served OpenAI-compatible on port 8090 (`start-qwen-vision.sh`, branch `p100-optimizations-keoian`). Reference
box: Unraid host, ASUS H270 (GPU1 on a PCH Gen3 x4 link, no P2P), 4-core i7, 48 GB RAM, driver 580, Docker with the
NVIDIA runtime. The older setup with JEV-27B System 1 decisions (`POST /v1/decide`, q4_0/q4_0, smaller images) is
`start-qwen.sh` with its defaults.

Files here: `Dockerfile`, `run-container.sh` (host), `unraid-user-script.sh` or the optional
`unraid-user-script-stop-others.sh` (host, at array start), `build.sh`,
`fetch.sh`, `start-qwen.sh`, `start-qwen-vision.sh`, `start-qwen-vision-exact.sh`, `verify.sh` (inside the container),
`SHA256SUMS`, and `diag/` (test tools, below).

## Recipe

**On the host** (paths are Unraid's; set `WORK` to any directory that becomes `/work` in the container):

```bash
git clone -b p100-optimizations-keoian https://github.com/Keoian/dual-p100-qwen-3-8-27b /mnt/user/qwen-dev/src/llama.cpp
cd /mnt/user/qwen-dev/src/llama.cpp/p100-docs/deploy
docker build -t qwen-dev:latest .
bash run-container.sh            # stops nothing: refuses to start if the GPUs are busy; sets -pm 1 and -pl 150
```

`run-container.sh` starts the container with `--init`: a minimal init runs as PID 1 and reaps exited processes, so a
`llama-server` killed for a restart doesn't stay a zombie (without it, `pgrep -x llama-server` keeps matching dead
servers). The container restarts with Docker (`--restart unless-stopped`), but it only runs `sleep infinity`: nothing
inside it starts the server.

Then install `unraid-user-script.sh` with the User Scripts plugin, schedule "At Startup of Array". After a host reboot
it sets the 150 W limit again (it resets on every boot) and, with `START_SERVER=1` set in the script, starts the
production server (`/work/start-qwen-vision.sh`, log `/work/serve-8090.log`; `QWEN_SCRIPT=/work/start-qwen.sh`
selects the JEV setup instead). Without `START_SERVER=1`, start the server by hand after a reboot (recipe below).
Optional instead: `unraid-user-script-stop-others.sh` also stops other GPU containers first (`STOP_CONTAINERS`,
default `llm-dev`), waits until the GPUs are free, then starts the same production script detached (same
`QWEN_SCRIPT` switch and log) and reports when it is up. The plugin keeps scripts on the flash drive:
`/boot/config/plugins/user.scripts/scripts/<name>/script`.

**Inside the container** (`docker exec -it qwen-dev bash`):

```bash
D=/work/src/llama.cpp/p100-docs/deploy
bash $D/build.sh build-vision                   # ~60-90 min cold on 4 cores, sm_60 only
bash $D/fetch.sh                                # model, Q8_0 projector, JEV files (pinned), sha256 check
umask 077; openssl rand -hex 32 > /work/qwen-api-keys   # one key per line, '#' lines ignored; never commit it
cp $D/start-qwen*.sh /work/                    # the vision scripts run start-qwen.sh with their settings
setsid nohup bash /work/start-qwen-vision.sh > /work/serve-8090.log 2>&1 &
until curl -sf localhost:8090/health; do sleep 5; done   # ~1 min to load
bash $D/verify.sh
```

On a 32 GB host, start it with `CTXCP=4 CACHE_RAM=8192` (fewer context checkpoints, a smaller RAM prompt cache).

The start scripts leave out what they can't find, without an error: no `/work/qwen-api-keys` means **no API key at
all** (the server is open to anyone who can reach :8090), a missing projector means text only, and a missing JEV LoRA
means no `/v1/decide`. Create the key file before the first start, and check the arguments the server got (`ps -o args -C llama-server`).

Clients: `http://<host>:8090/v1`, model id `qwen3.8-27b`, header `Authorization: Bearer <key>`.
Built-in web UI at `/`; `/health` is open.

## What `start-qwen-vision.sh` runs (production)

Expanded from the script with its defaults; argument for argument the command of the running production server:

```bash
LD_LIBRARY_PATH=/work/src/llama.cpp/build-vision/bin \
GGML_CUDA_AR_BF16_THRESHOLD=1 LLAMA_SPEC_STOP_ISO=0.125 LLAMA_SPEC_STOP_ISO_THINK=0.15 GGML_CUDA_GRAPHS_PRE_VOLTA=3 LLAMA_SPEC_SAMPLE_TEMP=1.0 LLAMA_SPEC_DRAFT_TOPK=20 \
GGML_CUDA_FA_VEC_MAX_Q=16 GGML_CUDA_FA_GEMM_MINQ=17 \
LLAMA_THINK_BY_BUDGET=1 LLAMA_SAMPLING_BY_MODE=1 \
LLAMA_SAMPLING_NOTHINK='{"temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0.0,"presence_penalty":1.5}' \
LLAMA_SAMPLING_THINK='{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0}' \
build-vision/bin/llama-server -m /work/models/Qwen3.8-27B-UD-Q6_K.gguf \
  -ngl 99 -sm tensor -fa 1 -ctk q8_0 -ctv q4_0 \
  -c 262144 -b 2048 -ub 1024 -np 1 -ctxcp 32 --cache-ram 24576 \
  --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.2 --spec-accept esc+0.2 \
  -ngld 99 -ubd 64 -ctkd q4_0 -ctvd q4_0 \
  --alias qwen3.8-27b --jinja --reasoning off --temp 0.7 --top-k 20 --top-p 0.8 --min-p 0.0 \
  --slot-save-path /work/slots \
  --mmproj /work/models-vision/mmproj-Qwen3.8-27B-Q8_0.gguf --image-max-tokens 3600 -mmdev CUDA1 \
  --api-key-file /work/qwen-api-keys \
  --host 0.0.0.0 --port 8090
```

`GGML_CUDA_AR_BF16_THRESHOLD=1` (`AR_BF16=1`) sends the decode and verify exchanges as bf16 (MTP replies +6.8%, KLD
+1.1%; `AR_BF16=0` for exact f32). `LLAMA_SPEC_STOP_ISO=0.125` (`STOP_ISO`) and `LLAMA_SPEC_STOP_ISO_THINK=0.15`
(`STOP_ISO_THINK`) stop the MTP draft by the lookup rule of CHANGES §22, with one pair of tables per thinking mode,
picked by the request's temperature: the thinking tables at 0.95-1.05, the chat tables at every other T > 0, p-min for
greedy (chat +4.7%, long context +4.8%, thinking +2.9%, T 0.3 / 1.2 +6.6% / +6.1%, exact; `=0` for p-min).
`--spec-accept esc+0.2` (`SPEC_ACCEPT`) keeps a drafted token unless the model's own sampler gives it probability 0
or the model rates it under 0.2 times its top choice: APPROXIMATE, chat +6.8%, thinking +6.9%, no more rare words than
the model's own sampling (simulated; CHANGES §23); a request sends `"speculative.accept": "exact"` for the model's own
sampling or `"escape"` for the faster rule (+14% / +31%), and only exact requests use the stop tables.
`start-qwen-vision-exact.sh` runs this same setup with `exact` as the default. With
`LLAMA_SAMPLING_BY_MODE=1` the chat endpoints take their sampling defaults from the two JSON sets; the CLI
`--temp/--top-k/--top-p/--min-p` then only serve `/completion`. Why each setting: [QUICKSTART.md](../QUICKSTART.md)
("1440p images and a q8_0 K cache", "Switches on this branch").

**Keeping a chat across restarts.** A restart loses the RAM prompt cache, so the first message after one re-prefills
the conversation (24k tokens: ~66 s). `POST /slots/0?action=save` with `{"filename": "...", "largest": true}` saves
the cache's longest prompt even when a chat app's side request (a title, follow-ups) has displaced it from the slot;
`action=restore` after the restart loads it back, and the next turn processes only its new tokens. On the reference
box `/work/bench/prodctl.sh` (and `prod-restart.sh`) does this around every planned restart.

## What `start-qwen.sh` runs with its defaults (the earlier JEV setup)

```bash
LD_LIBRARY_PATH=/work/src/llama.cpp/build-exp/bin \
GGML_CUDA_AR_BF16_THRESHOLD=1 LLAMA_SPEC_STOP_ISO=0.125 LLAMA_SPEC_STOP_ISO_THINK=0.15 GGML_CUDA_GRAPHS_PRE_VOLTA=3 LLAMA_SPEC_SAMPLE_TEMP=1.0 LLAMA_SPEC_DRAFT_TOPK=20 \
LLAMA_THINK_BY_BUDGET=1 LLAMA_SAMPLING_BY_MODE=1 \
LLAMA_SAMPLING_NOTHINK='{"temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0.0,"presence_penalty":1.5}' \
LLAMA_SAMPLING_THINK='{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0}' \
build-exp/bin/llama-server -m /work/models/Qwen3.8-27B-UD-Q6_K.gguf \
  -ngl 99 -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 \
  -c 262144 -b 32768 -ub 2048 -np 1 -ctxcp 4 \
  --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.2 --spec-accept exact \
  -ngld 99 -ubd 64 -ctkd q4_0 -ctvd q4_0 \
  --alias qwen3.8-27b --jinja --reasoning off --temp 0.7 --top-k 20 --top-p 0.8 --min-p 0.0 \
  --slot-save-path /work/slots \
  --mmproj /work/models/mmproj-F16.gguf --image-max-tokens 1024 -mmdev CUDA1 \
  --api-key-file /work/qwen-api-keys \
  --jev-lora /work/jev/gguf/jev-27b-lora-f16.gguf --jev-head /work/jev/JEV-27B/head.safetensors \
  --jev-calib /work/jev/JEV-27B/calibration.json --jev-ctk q8_0 --jev-ctv q8_0 \
  --host 0.0.0.0 --port 8090
```

`AR_BF16=1` (`GGML_CUDA_AR_BF16_THRESHOLD=1`) is this setup's default too. It was measured only on production; the
JEV numbers under "Acceptance checks" are f32 (`AR_BF16=0`).

`GGML_CUDA_P2P` is deliberately not set: on a board without peer access it enables nothing, and the operator's rule
is no P2P with JEV loaded until a supervised test. Env switches of the script: `JEV=0` (no System 1),
`JEV_KV=q4_0` (smaller decision cache, ~67 MiB/GPU less, KL 3.5e-4 vs f16), `MMDEV=` (projector on the default
device), `MMPROJ=` (text only), `IMAGE_MAX_TOKENS=` (no cap), `REASONING=auto`, `THINK_BY_BUDGET=0` / `SAMPLING_BY_MODE=0` / `NOTHINK_SAMPLING=<json>` / `THINK_SAMPLING=<json>` (CHANGES §19;
QUICKSTART "Switches on this branch"), `AR_BF16=0` (exact f32 exchanges; an exported
`GGML_CUDA_AR_BF16_THRESHOLD` wins over it), `STOP_ISO=0` / `STOP_ISO_THINK=0` (the MTP draft's p-min test for
requests outside / inside the thinking window T 0.95-1.05; exported `LLAMA_SPEC_STOP_ISO*` values win), `SPEC_ACCEPT=escape` (the MTP accept rule for requests that don't choose one; default exact, CHANGES §23), `ALIAS=`, `API_KEY_FILE=` (no key), `CTX=`, `BUILD=`, `MODEL=`,
`JEV_DIR=`. Why each choice: [QUICKSTART.md](../QUICKSTART.md).
`start-qwen-vision.sh` (production; copy it next to `start-qwen.sh`; needs `bash $D/build.sh build-vision`, as in the
recipe above) runs the full 262k context without JEV, with a q8_0 K / q4_0 V cache (KLD vs f16 0.00274 ->
0.00116), the Q8_0 vision projector and images up to 2560x1440's pixel area (`IMAGE_MAX_TOKENS=3600`); numbers in
its header. New start-qwen.sh switches for it: `CTK=`, `CTV=` (cache types), `UB=` (micro-batch), `CTXCP=` (context
checkpoints, default 4; the vision script uses 32) and `CACHE_RAM=` (RAM prompt cache in MiB; vision script 24576,
sized for a 48 GB host: use `CTXCP=4 CACHE_RAM=8192` on 32 GB) and `NBATCH=` (`-b`, default 32768; the vision script
uses 2048 so a dropped long prompt frees the server in ~2 s).

## Pinned inputs

| what | source | revision |
|---|---|---|
| model `Qwen3.8-27B-UD-Q6_K.gguf`, projector `mmproj-F16.gguf` | huggingface `unsloth/Qwen3.8-27B-GGUF` | `4ca720788d1e01f1bff70c033e0d0028fd02e502` |
| JEV `adapter/`, `head.safetensors`, `calibration.json`, `config.json` | huggingface `autotrust/JEV-27B` | `51740a8891c2a8baefd969237fd44187b3e3a115` |
| projector `mmproj-Qwen3.8-27B-Q8_0.gguf` (start-qwen-vision.sh, into `/work/models-vision`) | huggingface `ggml-org/Qwen3.8-27B-GGUF` | `71bc7b627595dc8a91039addd9c791ae548d6747` |
| JEV LoRA GGUF | `tools/jev-decide/convert_jev_lora.py --outtype f16` (numpy, safetensors, repo `gguf-py`) | deterministic, see `SHA256SUMS` |

## Host requirements and settings

- 2x Tesla P100-PCIE-16GB, NVIDIA driver with CUDA 12 support; CUDA **12.x** toolkit (13 cannot target sm_60).
- **GPU power limit 150 W** (`nvidia-smi -pm 1; nvidia-smi -i 0,1 -pl 150`). On the reference board a 262k prefill
  with both cards at 180 W hard-reset the host once; at 150 W the full-depth test passed. Costs ~3.5% decode.
- Nothing else on the GPUs: the model needs both cards (production peaks with a 1440p image at 259k: GPU0 15,185,
  GPU1 16,099 of 16,384 MiB; with JEV + an image at 262k: GPU0 15.2, GPU1 16.2 of 16.4 GB, ~115 MiB allocatable left
  on GPU1).
- THP: no setting needed (the server no longer advises huge pages on checkpoint buffers; tested with
  `enabled=always`, `defrag=madvise`).
- Check P2P on a new board with `nvidia-smi topo -m`; cards behind different root ports usually have none. With
  working P2P, `GGML_CUDA_P2P=1` may help, but that combination has not been tested with JEV.

## Acceptance checks

The gate needs both GPUs: stop the server first. `tools/gate.sh` always measures a q4_0/q4_0 cache (its
`llama-bench` and `llama-perplexity` lines pass `-ctk q4_0 -ctv q4_0`), and its default `MODEL` is not this layout's,
so pass `MODEL=`. It currently stops at its fold-attention SASS check, before measuring anything: "fattn-fold-sass.h
is stale". That is a false alarm (BUILD.md), so run its `llama-bench`, `llama-perplexity` and `test-backend-ops` lines
by hand.

| check | expected on the reference board (150 W), production: `start-qwen-vision.sh`, build-vision |
|---|---|
| `verify.sh` | health ok; "/v1/decide not served (JEV off): skipped"; a chat answer; idle VRAM ~14.6 / 15.4 GB |
| `ln -sfn build-vision build-opt; MODEL=/work/models/Qwen3.8-27B-UD-Q6_K.gguf ./tools/gate.sh` | PPL 2.6074 (q4_0/q4_0; band 2.6209 +/- 0.0199), tg256 about 30 (30.2 in the 10-08 gate run), FLASH_ATTN_EXT 3/3 |
| the served cache, from the repo root: `build-opt/bin/llama-perplexity -m /work/models/Qwen3.8-27B-UD-Q6_K.gguf -f p100-handoff/ppl-orig.txt -sm tensor -ngl 99 -c 4096 -ctk q8_0 -ctv q4_0` (gate.sh's perplexity line with the q8_0 K cache) | PPL 2.6064 |
| a 1440p screenshot | ~15 s at shallow depth |
| full depth (optional, ~45 min) | chat filled to ~255k + a 1440p image at ~259k: prompt ~32 s, decode ~28 t/s, GPU1 peak ~16.1 GB |

With `start-qwen.sh` and its defaults (the JEV setup, build-exp; for the gate `ln -sfn build-exp build-opt`; measured
with f32 exchanges, before `AR_BF16=1` became the default): `verify.sh` health ok, decide ~0.9 s, chat ~2-3 s, idle
VRAM ~14.3 / 15.4 GB; gate PPL 2.6074, tg256 ~30.2; an image question ~5-6 s for any size up to 10 MP (resized to
~1,020 tokens); full depth (optional, ~25 min) chat filled to ~255k with decisions + an image at 256k: no errors, GPU1
peak ~16.15 GB.

## Diagnostics (`diag/`)

- `monitor.sh [file]`: every 2 s, synced to disk, VRAM, temperature, power, PCIe throughput and replay counters of
  both GPUs and the AER error totals (default `/work/monitor.log`; run in the background, kill when done). The AER
  device list is hardcoded to the reference board's PCI addresses (`0000:01:00.0`, `0000:00:01.0`, `0000:03:00.0`,
  `0000:00:1b.4`): edit `aer()` for another board (`lspci` lists the GPUs and their root ports).
- `ballast.cu`: holds GPU memory on one device until killed, to test with less VRAM headroom
  (`nvcc -O2 -arch=sm_60 -o ballast ballast.cu`, then `ballast <device> <leave_free_MiB>`).

Records of how these numbers were measured: [CHANGES.md §15-16, §18-22](../CHANGES.md), [FINDINGS.md](../FINDINGS.md).
