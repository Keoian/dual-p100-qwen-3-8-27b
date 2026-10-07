# Quickstart

For two Tesla P100-16GB cards (or other Pascal sm_60 cards) running Qwen3.8-27B Q6_K with its
built-in MTP head. Build first: see [BUILD.md](BUILD.md).

## Run the server

One command for text and vision. For vision, add the projector line shown below it.

    GGML_CUDA_P2P=1 GGML_CUDA_GRAPHS_PRE_VOLTA=3 \
    LLAMA_SPEC_SAMPLE_TEMP=1.0 LLAMA_SPEC_DRAFT_TOPK=20 \
    ./build-opt/bin/llama-server \
      -m /path/to/Qwen3.8-27B-Q6_K.gguf \
      -ngl 99 -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 \
      -c 262144 -b 32768 -ub 2048 -np 1 \
      --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.2 \
      -ngld 99 -ubd 64 -ctkd q4_0 -ctvd q4_0 \
      --jinja --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0 \
      --host 0.0.0.0 --port 8080

**For vision**, add:

      --mmproj /path/to/mmproj-Qwen3.8-27B-Q8_0.gguf

Nothing else changes. Until 2026-10-01 vision needed `-ub 1024`; a smaller attention mask made
room for `-ub 2048` with the projector loaded.

The release bundle's `qwen-server` wrapper runs exactly this command; pass `--mmproj <file>` to it
for vision.

## What to expect

Prefill, and decode with MTP, at each context depth. Measured 2026-10-01 with `tools/depth-bench.py`:
one conversation grown from 2k to 260k with vision loaded, prefill timed on each ~30k-token chunk,
decode averaged over two questions per depth, cards hot throughout. The previous release is shown
for comparison.

| depth | prefill | decode | prefill, 09-26 release |
|---|---|---|---|
| 2k | 454 t/s ¹ | 54 t/s | 316 t/s (first request) |
| 32k | 423 t/s | 55 t/s | 342 t/s |
| 62k | 342 t/s | 46 t/s | 257 t/s |
| 92k | 257 t/s | 44 t/s | 194 t/s |
| 122k | 206 t/s | 38 t/s | 162 t/s |
| 152k | 175 t/s | 37 t/s | 141 t/s |
| 182k | 156 t/s | 38 t/s | 125 t/s |
| 212k | 140 t/s | 32 t/s | 118 t/s |
| 242k | 127 t/s | 29 t/s | 108 t/s |
| 260k | 123 t/s | 33 t/s | 100 t/s |

¹ On a warm server. The first request after startup reads 375 t/s: it also pays ~1.5 s of one-time
setup (loading cuBLAS kernels and allocating buffers). The GPU itself prefills ~500 t/s at short
context (`llama-bench` `pp2048`: 493); a server request adds the final partial batch, the MTP
catch-up and a checkpoint save.

Filling the whole 260k context takes ~25 minutes of prefill (09-26 release: ~29). A short question
on top of an already-loaded 260k context runs faster than the fill rate: ~153 t/s for 1.5k tokens.

MTP decode depends on how predictable the text is: code and factual answers run faster than
creative writing. Cards that have been under sustained load read ~5-10% lower.

## What the flags do

| flag | why |
|---|---|
| `-sm tensor` | splits every layer across both cards. Needed to fit the full context |
| `-fa 1` | flash attention. Tensor split requires it |
| `-ctk q4_0 -ctv q4_0` | q4_0 KV cache. An f16 cache doesn't fit at 262k |
| `-c 262144` | the model's full context. Reserving it costs nothing until it fills |
| `-np 1` | one server slot. Each slot allocates its own full KV cache |
| `-b 32768` | **needed for MTP at long context.** A larger batch turns a long prompt into one huge batch, and draft acceptance collapses |
| `-ub 2048` | tokens per GPU pass. 2048 is the fastest prefill that still leaves VRAM headroom at 262k, with or without vision |
| `--spec-type draft-mtp` | speculative decoding with the model's built-in MTP head |
| `--spec-draft-n-max 4 --spec-draft-p-min 0.2` | draft up to 4 tokens; stop below 20% confidence. Use 3 if you mostly work past ~150k context |
| `-ngld 99 -ubd 64 -ctkd q4_0 -ctvd q4_0` | the draft layer on the GPU, with its own small ubatch and a q4_0 cache. Without `-ubd 64` the draft runs out of memory at full context |
| `--temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0` | the model card's sampling. The gguf doesn't carry min-p, so without the flag llama.cpp's 0.05 applies |
| `GGML_CUDA_P2P=1` | direct copies between the cards instead of through host memory |
| `GGML_CUDA_GRAPHS_PRE_VOLTA=3` | CUDA graphs for the single-token MTP draft steps only. Full graphs (`1`) run out of VRAM at full context |
| `LLAMA_SPEC_SAMPLE_TEMP=1.0 LLAMA_SPEC_DRAFT_TOPK=20` | the draft samples instead of taking its top token, and the verify uses the speculative-sampling rule. The output distribution is unchanged; more drafts get accepted |

## VRAM

Figures are **per card**, at a full 262k context, with nothing else on the GPU. GPU0 is the
tighter card because the vision projector loads onto it.

| configuration | GPU0 free at full context |
|---|---|
| text only, `-ub 2048` | ~1.7-1.9 GiB (estimated: the vision figure plus the projector's 600-850 MiB) |
| vision, `-ub 2048` | ~1.1 GiB (measured 732-792 MiB with ~392 MiB of desktop streaming also on GPU0) |

VRAM use grows as the context fills, so check it with a full prompt, not a short one. If GPU0
also drives a display or runs other programs, subtract what they use. Lowering `-ub` is the
fix, and it costs only prefill speed: 2048 → 1024 → 512. With vision, `-mmdev CUDA1` puts the
projector on the second card instead and frees ~850 MiB on GPU0.

## Boards without P2P, chat apps, and slot save/restore (tyler-port branch)

**No P2P.** If `nvidia-smi topo -m` shows the cards behind different root ports (e.g. one slot wired to the chipset at
x4), `cudaDeviceCanAccessPeer` is likely 0 and `GGML_CUDA_P2P=1` does nothing; leave it out. Keep `-sm tensor` (still
faster than `-sm layer`); this branch stages the exchanges through pinned host memory. Measured on such a board:
tg256 27.8, pp2048 466, MTP decode in chat 38-40 t/s, a 10-40-token chat turn on a 26k cached prefix in ~0.56 s of
server prompt time, 256k context with vision without running out of memory. Add `-ctxcp 4`: the server recycles 4
checkpoint buffers, and each checkpoint copy is ~150 MiB.

**Chat apps and vision.** For OpenAI-compatible clients add:

      --alias qwen3.8-27b --reasoning off --image-max-tokens 1024 --api-key-file /path/to/keys

`--reasoning off` because many apps hide `reasoning_content`, show nothing while the model thinks, and time out; a
request can still turn it on with `"chat_template_kwargs": {"enable_thinking": true}`. `--image-max-tokens 1024`
because a 2875x1500 image is ~4100 tokens and ~19 s before the first byte; capped, ~4 s. The cap is a pixel budget: every
image is resized to fit it with its aspect ratio kept, so 10 MP images (4380x2285 and a 2592x3888 portrait, tested
10-07 on the JEV + vision production setup) also came out at ~1,020 tokens, ~5 s, with no extra projector memory. The projector
(`mmproj-F16.gguf` from the model's GGUF repo) takes ~885 MiB on GPU0; at 256k context with an image GPU0 peaked at
15.5 of 16.4 GB.

**Slot save/restore** (`--slot-save-path DIR`, then `POST /slots/0?action=save|restore {"filename": ...}`) writes
`<name>` plus `<name>.draft` (MTP head cache), `<name>.spec` (MTP carry-over row) and `<name>.ckpt` (context
checkpoints). Keep them together. A restored slot continues byte-identically to the live one (greedy, MTP on), and a
request that diverges just before the saved end (e.g. the same chat without the generated reply) reprocesses only a
few tokens. Slot files are client-driven: nothing is saved or restored automatically across a restart.
The in-memory prompt cache (`--cache-ram`, 8 GiB by default) switches between recent prompts automatically while the
server runs, also exactly.

## JEV System 1 decisions (`/v1/decide`, tyler-port branch)

Calibrated one-pass decisions from [autotrust/JEV-27B](https://huggingface.co/autotrust/JEV-27B) on the same loaded
model. Download only `adapter/`, `head.safetensors`, `calibration.json`, `config.json` (not the bf16 shards), convert
the LoRA once, then add three flags to the server command:

      python3 tools/jev-decide/convert_jev_lora.py --adapter JEV-27B/adapter --config JEV-27B/config.json \
          --outtype f16 --out jev-27b-lora-f16.gguf

      --jev-lora jev-27b-lora-f16.gguf --jev-head JEV-27B/head.safetensors --jev-calib JEV-27B/calibration.json

      curl -s :8090/v1/decide -H 'Authorization: Bearer KEY' -H 'Content-Type: application/json' \
        -d '{"kind":"choice","state":"...","question":"...","options":["A thing","Another thing"]}'

Kinds: `noul` (options `["false","true"]`, may be omitted), `score` (0-5), `choice` (2-16 options, one per line, no
newlines). The answer has `probabilities`, `choice_index`, `choice`, `confidence`; `"debug": true` adds the raw slot
logits. Decisions run in their own context (`--jev-ctx 8192`, `--jev-batch 512`), between System 2 batches; System 2
output is unchanged by them. ~0.6 s per ~100-token decision, ~2 s per ~500 tokens (no state reuse between decisions).

- **KV type** of the decision context: `--jev-ctk/--jev-ctv`, server default q4_0 (KL 3.5e-4 vs f16). q8_0 is
  practically f16 (KL 3e-6) for ~67 MiB more per GPU; the tyler-port production setup serves q8_0. f16/q8_0 need commit `03da0202b` (before it, long prompts could NaN).
- **VRAM** (262k System 2, MTP, q4_0 System 2 cache, f16 decision cache; q4_0 decision cache not re-measured, ~190 MiB
  less per GPU): JEV adds ~0.6 GiB per GPU idle. After decisions and a few chat turns GPU0 was 15.2 GB and GPU1 15.9 GB
  vs 13.8 / 14.9 GB idle without JEV (that includes System 2's own pool growth). With the projector, move it to GPU1
  (`-mmdev CUDA1`). Served configuration (q8_0 decision cache), chat at 255k + image at 256k: peaks GPU0 15.2, GPU1
  16.2 of 16.4 GB, all requests fine. Not run: 262k with the q4_0 decision cache.
- **Power.** On the tyler-port board a 262k deep prefill with both cards at 180 W reset the host once; at
  `nvidia-smi -i 0,1 -pl 150` the same run passed. The limit resets on reboot.
- `llama-jev-decide` (same flags plus `--jev-in/--jev-out` JSONL) evaluates a file of decisions; see
  `tools/jev-decide/README.md`. Pass `-ctk q4_0 -ctv q4_0` to it to match the server's decision context.

## Precision switches

Everything defaults to the fast path, which is at least as accurate as stock. These exist for
A/B testing.

| variable | effect |
|---|---|
| `GGML_CUDA_GEMM_FOLD=0` | prefill matmuls on stock cuBLAS fp16 instead of the fold kernel (fp16 products, fp32 sums) |
| `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` | prefill matmuls fully in fp32. ~40% slower, no measurable accuracy gain |
| `GGML_CUDA_FA_GEMM=0` | turns off the GEMM attention path for long prefill (back to the tile kernel) |
| `GGML_CUDA_FA_FOLD=0` | long-prefill attention on cuBLAS (fp16 over 2048-key chunks) instead of the fold kernels (fp16 over 128 keys, fp32 across) |
| `GGML_CUDA_FA_PV2=0` | the older fold PV kernel (same accumulation, ~14% slower at 262k) |
| `GGML_CUDA_FA_SASS=0`, `GGML_CUDA_GEMM_FOLD_SASS=0` | the compiler's build of the fold attention and fold GEMM kernels instead of the register-renamed SASS (bit-identical, slightly slower) |
| `GGML_CUDA_GDN_CHUNKED=0` | the gated delta net as a per-token recurrence during prefill instead of chunked (same accuracy vs fp64, slower) |
| `LLAMA_KQ_MASK_COMPACT=0` | the full n_kv x n_tokens f16 attention mask instead of per-row prefix lengths (bit-identical; needs ~1 GiB more per card at 262k and `-ub 2048`, so `-ub 2048` with vision no longer fits) |
| `GGML_CUDA_AR_P2P=0` | the tensor-parallel exchange back to copy-then-add (the one-kernel P2P version is bit-identical and faster) |
| `GGML_CUDA_FUSE_FFN_GLU=0` | the FFN gate, up and SwiGLU as three kernels again (bit-identical) |
| `LLAMA_SPEC_BLOCK_VERIFY=0` | per-token draft verification instead of block verification (same output distribution; block accepts more) |
| `LLAMA_MTP_DRAFT_VOCAB=0` | the MTP draft scores the full vocabulary instead of a small copy of the common tokens. Saves ~112 MiB per card, drafts get slower, output is unchanged |

## Benchmarking

    GGML_CUDA_P2P=1 ./build-opt/bin/llama-bench -m /path/to/Qwen3.8-27B-Q6_K.gguf \
      -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -p 0 -n 256 -r 5

Expect ~32 t/s (plain decode, no MTP). Measure on cool cards: right after a long run, P100s can
read up to 20% low. To check accuracy, run `tools/gate.sh`; perplexity should land near 2.61.
