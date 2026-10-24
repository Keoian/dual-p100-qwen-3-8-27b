# Quickstart

For two Tesla P100-16GB cards (or other Pascal sm_60 cards) running Qwen3.8-27B Q6_K with its
built-in MTP head. Build first: see [BUILD.md](BUILD.md).

The command below is Kmic's setup, for a board with P2P (its `-b 2048` came from this branch's docs PR). The
production setup of branch `p100-optimizations-keoian` (a board without P2P, 150 W, 48 GB of host RAM) is
`deploy/start-qwen-vision.sh`; its full expanded command is in [deploy/README.md](deploy/README.md) ("What
`start-qwen-vision.sh` runs"), and its settings are explained in "Boards without P2P, chat apps, and slot
save/restore", "1440p images and a q8_0 K cache" and "Switches on this branch" below. How the two differ:

| setting | command below (Kmic's) | production (`start-qwen-vision.sh`) | why |
|---|---|---|---|
| KV cache | q4_0 / q4_0 | q8_0 K / q4_0 V | KLD vs an f16 cache 0.00274 -> 0.00116, at q4_0 speed with this branch's q4p q8_0-K kernel |
| `-ub` / `-b` | 2048 / 2048 | 1024 / 2048 | `-ub 2048` runs GPU1 out of memory beside a 1440p image at 262k with the q8_0 K cache |
| P2P | `GGML_CUDA_P2P=1` | not set | the reference board has none (GPU1 on a chipset x4 link) |
| vision | `--mmproj` (Q8_0) on GPU0 | Q8_0 projector on GPU1 (`-mmdev CUDA1`), `--image-max-tokens 3600` | 2560x1440's pixel area keeps small text legible; GPU1 has the room |
| checkpoints / RAM cache | server defaults (32, 8 GiB) | `-ctxcp 32 --cache-ram 24576` | a larger RAM prompt cache on a 48 GB host; on 32 GB use `CTXCP=4 CACHE_RAM=8192` |
| attention routing | defaults | `GGML_CUDA_FA_VEC_MAX_Q=16 GGML_CUDA_FA_GEMM_MINQ=17` | required with a q8_0 K cache at 262k, else out of memory at depth |
| exchange wire | f32 | bf16 (`GGML_CUDA_AR_BF16_THRESHOLD=1`, `start-qwen.sh`: `AR_BF16=1`) | MTP replies +6.8% over the x4 link; lossy (KLD +1.1%); `AR_BF16=0` for exact f32 |
| MTP draft stop | p-min 0.2 | the lookup stop rule (`LLAMA_SPEC_STOP_ISO=0.125`, `start-qwen.sh`: `STOP_ISO`) for thinking-off requests | chat replies +4.7%, long context +4.8%; exact; thinking requests keep p-min; `STOP_ISO=0` for p-min only |
| thinking | template default (on) | `--reasoning off`, `LLAMA_THINK_BY_BUDGET=1` | chat apps that hide reasoning show nothing while it thinks; a toggle reprocesses 21-42 tokens, not the chat |
| sampling | 1.0 / 0.95 / 20 / 0 | `LLAMA_SAMPLING_BY_MODE=1`: thinking off 0.7 / 0.8 / 20 / 0 with presence 1.5, thinking 1.0 / 0.95 / 20 / 0 | the model card has a separate thinking-off set |
| access | :8080, no key | :8090, `--api-key-file`, `--alias qwen3.8-27b`, `--slot-save-path` | OpenAI-compatible clients |

## Run the server

One command for text and vision. For vision, add the projector line shown below it.

    GGML_CUDA_P2P=1 GGML_CUDA_GRAPHS_PRE_VOLTA=3 \
    LLAMA_SPEC_SAMPLE_TEMP=1.0 LLAMA_SPEC_DRAFT_TOPK=20 \
    ./build-opt/bin/llama-server \
      -m /path/to/Qwen3.8-27B-Q6_K.gguf \
      -ngl 99 -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 \
      -c 262144 -b 2048 -ub 2048 -np 1 \
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
| `-b 2048` | the logical batch. With `-ub` dividing it, the GPU passes are the same at any `-b`: a 112k-token prompt and three greedy answers gave byte-identical text and identical MTP draft counts at `-b 32768` and `-b 2048` (2x P100, deterministic drafting). What it does change is how long a dropped client keeps the server busy, because cancellation is checked between batches: a 64k prompt abandoned after 60 s held the slot ~33 s more at 32768, ~2 s at 2048 ([THREE-GPU.md](THREE-GPU.md) reported the same). An older version of this row said MTP needed 32768; that no longer holds |
| `-ub 2048` | tokens per GPU pass. 2048 is the fastest prefill that still leaves VRAM headroom at 262k, with or without vision |
| `--spec-type draft-mtp` | speculative decoding with the model's built-in MTP head |
| `--spec-draft-n-max 4 --spec-draft-p-min 0.2` | draft up to 4 tokens; stop below 20% confidence. Use 3 if you mostly work past ~150k context |
| `-ngld 99 -ubd 64 -ctkd q4_0 -ctvd q4_0` | the draft layer on the GPU, with its own small ubatch and a q4_0 cache. Without `-ubd 64` the draft runs out of memory at full context |
| `--temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0` | the model card's sampling. The gguf doesn't carry min-p, so without the flag llama.cpp's 0.05 applies |
| `GGML_CUDA_P2P=1` | direct copies between the cards instead of through host memory |
| `GGML_CUDA_GRAPHS_PRE_VOLTA=3` | CUDA graphs for the single-token MTP draft steps only. Full graphs (`1`) run out of VRAM at full context |
| `LLAMA_SPEC_SAMPLE_TEMP=1.0 LLAMA_SPEC_DRAFT_TOPK=20` | the draft samples instead of taking its top token, and the verify uses the speculative-sampling rule. The output distribution is unchanged; more drafts get accepted |

## VRAM

Figures are **per card**, at a full 262k context, with nothing else on the GPU. Without `-mmdev`, GPU0 is the
tighter card because the vision projector loads onto it; the deploy scripts use `-mmdev CUDA1`, which makes
GPU1 the tight one (the production setup leaves GPU1 ~280 MiB at full depth with a 1440p image, below).

| configuration | GPU0 free at full context |
|---|---|
| text only, `-ub 2048` | ~1.7-1.9 GiB (estimated: the vision figure plus the projector's 600-850 MiB) |
| vision, `-ub 2048` | ~1.1 GiB (measured 732-792 MiB with ~392 MiB of desktop streaming also on GPU0) |

VRAM use grows as the context fills, so check it with a full prompt, not a short one. If GPU0
also drives a display or runs other programs, subtract what they use. Lowering `-ub` is the
fix, and it costs only prefill speed: 2048 → 1024 → 512. With vision, `-mmdev CUDA1` puts the
projector on the second card instead and frees ~850 MiB on GPU0.

## Boards without P2P, chat apps, and slot save/restore (branch `p100-optimizations-keoian`)

**No P2P.** If `nvidia-smi topo -m` shows the cards behind different root ports (e.g. one slot wired to the chipset at
x4), `cudaDeviceCanAccessPeer` is likely 0 and `GGML_CUDA_P2P=1` does nothing; leave it out. Keep `-sm tensor` (still
faster than `-sm layer`); this branch stages the exchanges through pinned host memory. Measured on such a board at
its 150 W limit with the production build and settings (root README "Measured"): plain decode 31.1 t/s and MTP chat
replies ~47 t/s (10-24), pp512 410 and pp2048 413 t/s (10-22), ~0.5 s to the first token for a short message on a 25k
cached conversation, the full 262k context with a 1440p image without running out of memory. (The branch's first
numbers, 10-06 at 180 W with q4_0/q4_0 and `-ub 2048`, are in CHANGES §15.) Each context checkpoint copy is ~150 MiB
of host RAM: `start-qwen.sh` defaults to `-ctxcp 4` (llama-server's own default is 32), and the server recycles freed
checkpoint buffers (a pool of 4). Production keeps 32, with 48 GB of host RAM ("1440p images" below).

**Chat apps and vision.** For OpenAI-compatible clients add:

      --alias qwen3.8-27b --reasoning off --image-max-tokens 1024 --api-key-file /path/to/keys

`--reasoning off` because many apps hide `reasoning_content`, show nothing while the model thinks, and time out; a
request can still turn it on with `"chat_template_kwargs": {"enable_thinking": true}`. Qwen3.8's template puts its
effort line at the top of the system prompt, so on a plain server that toggle reprocesses the whole cached chat; with
`LLAMA_THINK_BY_BUDGET=1` it costs only the new turn, and `LLAMA_SAMPLING_BY_MODE=1` gives each mode the model card's
sampling (CHANGES §19). With thinking off the card says 0.7 / 0.8 and presence_penalty 1.5, which production serves.
With `LLAMA_SAMPLING_BY_MODE=1` the chat endpoints fill every sampling field a request leaves out from
`LLAMA_SAMPLING_THINK` / `LLAMA_SAMPLING_NOTHINK`, so the CLI `--temp/--top-k/--top-p/--min-p/--presence-penalty`
no longer apply to chat; they only serve `/completion`. Change the chat defaults through the two JSON sets
(`start-qwen.sh`: `THINK_SAMPLING`, `NOTHINK_SAMPLING`).
Any repetition penalty used to cost 7-10% decode here, because the sampler scanned the whole vocabulary per token.
The scan is gone since 0d920143d, and the sampler's top-k prefilter runs with penalties since 910d2cfb8 (until then
presence 1.5 still cost ~2.2% at natural length, CHANGES §21). Presence penalty only sees the last 64 tokens, so it
guards against in-reply loops, not agentic ones.
`--image-max-tokens 1024` (`start-qwen.sh`'s default) because a 2875x1500 image is ~4100 tokens and
~19 s before the first byte; capped, ~4 s. Production caps at 3600 tokens instead (2560x1440's pixel area), so small
text in screenshots stays legible, with the setup in "1440p images" below. The cap is a pixel budget: every image is
resized to fit it with its aspect ratio kept, so 10 MP images (4380x2285 and a 2592x3888 portrait, tested 10-07 at the
1024 cap on the JEV + vision setup then in production) also came out at ~1,020 tokens, ~5 s, with no extra projector
memory. Production uses the ggml-org Q8_0 projector on GPU1 ("1440p images" below). The F16 projector
(`mmproj-F16.gguf` from the model's GGUF repo, `start-qwen.sh`'s default) takes ~885 MiB on its card (~1.1 GiB with
encoder buffers): on GPU0 by default (without JEV, at 256k with an image GPU0 peaked at 15.5 of 16.4 GB), on GPU1 with
`-mmdev CUDA1` in the JEV setup below.

**Slot save/restore** (`--slot-save-path DIR`, then `POST /slots/0?action=save|restore {"filename": ...}`) writes
`<name>` plus `<name>.draft` (MTP head cache), `<name>.spec` (MTP carry-over row) and `<name>.ckpt` (context
checkpoints). Keep them together. `.ckpt` holds every context checkpoint the slot has at the time (~150 MiB each), so
a save grows with `-ctxcp`: with production's 32 it can reach ~4.8 GB. A restored slot continues byte-identically to
the live one (greedy, MTP on), and a request that diverges just before the saved end (e.g. the same chat without the
generated reply) reprocesses only a few tokens. Slot files are client-driven: nothing is saved or restored automatically across a restart.
The in-memory prompt cache (`--cache-ram`, 8 GiB by default) switches between recent prompts automatically while the
server runs, also exactly.

## JEV System 1 decisions (`/v1/decide`; off in production since the 1440p setup)

Calibrated one-pass decisions from [autotrust/JEV-27B](https://huggingface.co/autotrust/JEV-27B) on the same loaded
model. Download only `adapter/`, `head.safetensors`, `calibration.json`, `config.json` (not the bf16 shards), convert
the LoRA once (`pip install numpy safetensors`), then add the JEV flags to the server command. JEV was validated only
**without** `GGML_CUDA_P2P` (on a no-P2P board, at a 150 W power limit); drop `GGML_CUDA_P2P=1` from the command above
when adding it. The full JEV command (q8_0 decision cache, projector on GPU1; production from 10-07 until the 1440p
setup) is in [deploy/README.md](deploy/README.md); `deploy/start-qwen.sh` runs it with its defaults.

      python3 tools/jev-decide/convert_jev_lora.py --adapter JEV-27B/adapter --config JEV-27B/config.json \
          --outtype f16 --out jev-27b-lora-f16.gguf

      --jev-lora jev-27b-lora-f16.gguf --jev-head JEV-27B/head.safetensors --jev-calib JEV-27B/calibration.json \
      --jev-ctk q8_0 --jev-ctv q8_0 -mmdev CUDA1     # optional: served cache type, projector off GPU0

      curl -s :8090/v1/decide -H 'Authorization: Bearer KEY' -H 'Content-Type: application/json' \
        -d '{"kind":"choice","state":"...","question":"...","options":["A thing","Another thing"]}'

Kinds: `noul` (options `["false","true"]`, may be omitted), `score` (`"0"`-`"5"`, may be omitted), `choice` (2-16
options, no newlines in them). `state` is a string or a JSON object (sent as its JSON text). The answer has
`probabilities`, `choice_index`, `choice`, `confidence`; `"debug": true` adds the raw slot `logits` and the
`temperature`. Also served at `/decide`. Decisions run in their own context (`--jev-ctx 8192`, `--jev-batch 512`), between System 2 batches; System 2
output is unchanged by them. ~0.6-0.7 s per ~100-token decision, ~2 s per ~500 tokens (no state reuse between
decisions).

- **KV type** of the decision context: `--jev-ctk/--jev-ctv`, server default q4_0 (KL 3.5e-4 vs f16). q8_0 is
  practically f16 (KL 3e-6) for ~67 MiB more per GPU; the JEV production setup served q8_0 (and `start-qwen.sh`
  still defaults to it). f16/q8_0 need commit `03da0202b` (before it, long prompts could NaN).
- **VRAM** (262k System 2, MTP, q4_0 System 2 cache, f16 decision cache; q4_0 decision cache not re-measured, ~190 MiB
  less per GPU): JEV adds ~0.6 GiB per GPU idle. After decisions and a few chat turns GPU0 was 15.2 GB and GPU1 15.9 GB
  vs 13.8 / 14.9 GB idle without JEV (that includes System 2's own pool growth). With the projector, move it to GPU1
  (`-mmdev CUDA1`). Served JEV configuration (q8_0 decision cache), chat at 255k + image at 256k: peaks GPU0 15.2, GPU1
  16.2 of 16.4 GB, all requests fine. Not run: 262k with the q4_0 decision cache.
- **Power.** On the reference (no-P2P) board a 262k deep prefill with both cards at 180 W reset the host once; at
  `nvidia-smi -i 0,1 -pl 150` the same run passed. The limit resets on reboot.
- `llama-jev-decide` evaluates a JSONL file of decisions. It takes the usual model flags plus `--lora` (not
  `--jev-lora`), `-c`, `-ctk/-ctv`, and its own `--jev-head`, `--jev-calib`, `--jev-in`, `--jev-out`, `--jev-hidden`,
  `--jev-limit`; see `tools/jev-decide/README.md`. Pass the served decision-cache type (`-ctk q8_0 -ctv q8_0` for
  `start-qwen.sh`'s JEV setup) to measure what the server runs.

## 1440p images and a q8_0 K cache (the production setup, `start-qwen-vision.sh`)

```bash
bash p100-docs/deploy/build.sh build-vision           # this branch; adds q4p / fold kernels for a q8_0 K cache
bash p100-docs/deploy/fetch.sh                        # also fetches the Q8_0 projector into /work/models-vision
cp p100-docs/deploy/start-qwen*.sh /work/              # start-qwen-vision.sh needs start-qwen.sh next to it
bash /work/start-qwen-vision.sh
```

What it changes against `start-qwen.sh`: JEV off (`/v1/decide` not served), KV cache K q8_0 / V q4_0 (KLD against
an f16 cache 0.00274 -> 0.00116), `-ub 1024`, the ggml-org Q8_0 projector, images up to 2560x1440's pixel area
(`IMAGE_MAX_TOKENS=3600`), and two required routing switches, `GGML_CUDA_FA_VEC_MAX_Q=16` and
`GGML_CUDA_FA_GEMM_MINQ=17`. Without them a q8_0 cache runs out of memory at depth (CHANGES §18).

- **Image cap.** 1024 tokens (~1 MP, the default script) answers in ~4 s but loses text below ~14 px; 3600 keeps
  9-12 px text (19-23 of 24 rows exact) at ~15-17 s to first byte at shallow depth, ~32 s near 255k. 4096 fits at
  262k; 6000 does not.
- **Speed vs q4_0/q4_0:** decode -4% at 64k (bytes, not code); prefill equal at the same `-ub`, 4-7% below the
  default script's `-ub 2048`, which doesn't fit beside a 1440p image at 262k (it does at ~240k).
- **Cancellation:** `-b 2048`: a client that drops a long prompt frees the server in ~2 s (~33 s at 32768), same output.
- **Host RAM:** `-ctxcp 32 --cache-ram 24576` (needs ~32 GB at worst; the reference box has 48 GB). With
  `start-qwen.sh`'s default `-ctxcp 4` (llama-server's own default is 32), branching back 3 turns in a 6-turn, 58k-token
  chat re-processed the kept 27.8k tokens (77.5 s in a test); with 32, 30 tokens (0.5 s). On a 32 GB host use the
  script's defaults (`CTXCP=4 CACHE_RAM=8192`).
- **VRAM:** idle 14,583 / 15,435 MiB; peaks 15,187 / 16,101 with a 1440p image at 258,866 tokens. If a card has
  less free memory than the reference board's, lower `CTX` (each 10k tokens frees ~130 MiB per card).

## Precision switches

Everything defaults to the fast path, which is at least as accurate as stock. These exist for
A/B testing.

| variable | effect |
|---|---|
| `GGML_CUDA_GEMM_FOLD=0` | prefill matmuls on stock cuBLAS fp16 instead of the fold kernel (fp16 products, fp32 sums) |
| `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` | prefill matmuls fully in fp32. ~40% slower, no measurable accuracy gain |
| `GGML_CUDA_FA_GEMM=0` | turns off the GEMM attention path for long prefill (back to the tile kernel) |
| `GGML_CUDA_FA_FOLD=0` / `=2` | `=0`: long-prefill attention on cuBLAS (fp16 over 2048-key chunks) instead of the fold kernels (fp16 over 128 keys, fp32 across); `=2`: the fold kernels also for f16 K or V. Default 1: fold for q4_0 and q8_0 K and V (Kmic's PR #3) |
| `GGML_CUDA_FA_PV2=0` | the older fold PV kernel (same accumulation, ~14% slower at 262k) |
| `GGML_CUDA_FA_SASS=0`, `GGML_CUDA_GEMM_FOLD_SASS=0` | the compiler's build of the fold attention and fold GEMM kernels instead of the register-renamed SASS (bit-identical, slightly slower) |
| `GGML_CUDA_GDN_CHUNKED=0` | the gated delta net as a per-token recurrence during prefill instead of chunked (same accuracy vs fp64, slower) |
| `LLAMA_KQ_MASK_COMPACT=0` | the full n_kv x n_tokens f16 attention mask instead of per-row prefix lengths (bit-identical; needs ~1 GiB more per card at 262k and `-ub 2048`, so `-ub 2048` with vision no longer fits) |
| `GGML_CUDA_AR_P2P=0` | the tensor-parallel exchange back to copy-then-add (the one-kernel P2P version is bit-identical and faster) |
| `GGML_CUDA_FUSE_FFN_GLU=0` | the FFN gate, up and SwiGLU as three kernels again (bit-identical) |
| `LLAMA_SPEC_BLOCK_VERIFY=0` | per-token draft verification instead of block verification (same output distribution; block accepts more) |
| `LLAMA_SAMPLER_PREFILTER=0` | the sampler materializes and sorts the whole vocabulary for every token instead of selecting top-k straight from the logits (same output; with any logit bias, `ignore_eos` included, it does this anyway) |
| `GGML_CUDA_FA_Q4P_Q8K=0` | a q8_0 K cache back to the generic decode/verify kernels (vec, tile) instead of q4p's q8_0-K variant |
| `GGML_CUDA_FA_VEC_MAX_Q=N`, `GGML_CUDA_FA_GEMM_MINQ=N` | routing for caches without a direct tile path (default 2 / 128; the q8_0 setup needs 16 / 17, else the tile kernel's whole-cache f16 copy runs out of memory at depth) |
| `LLAMA_MTP_DRAFT_VOCAB=0` | the MTP draft scores the full vocabulary instead of a small copy of the common tokens. Saves ~112 MiB per card, drafts get slower, output is unchanged |

## Switches on this branch (`p100-optimizations-keoian`)

Added by this branch (CHANGES §15, §18, §19, §21, §22). Unset means the default; the kernel and checkpoint switches exist
for A/B tests, the server ones change behaviour. Where `start-qwen.sh` sets one, its own switch is named.

| variable | default | effect |
|---|---|---|
| `GGML_CUDA_AR_HOST` | on for two Pascal cards without P2P | host-staged tensor-parallel AllReduce for small (decode / verify) exchanges; `=0` sends them as driver-staged peer copies |
| `GGML_CUDA_AR_HOST_MAX` | 1048576 (bytes) | largest exchange the host-staged path takes; larger (prefill) ones take the peer copy + ADD |
| `GGML_CUDA_AR_BF16_THRESHOLD` | 0 (off) with the host-staged Pascal path, 1 elsewhere (`start-qwen.sh`: `AR_BF16`, 1) | tensors of at least this many bytes are reduced through a lossy fp32 -> bf16 round trip. The branch's code default is off on Pascal; production turns it on: MTP replies +6.8%, KLD +1.1% (CHANGES §21) |
| `GGML_CUDA_XCHG_NOP2P` | 1 | overlaps the prefill exchange with the chunked matmul when the cards have no P2P; `=0` off |
| `GGML_CUDA_MMVQ_CHUNK_MAX` | 28 | 9..N-column q6_K/q5_K batches run as fp16 mat-vec column chunks (at 9..16 the single-launch fp16 matvec goes first when its shape check accepts); `=0` sends them on to the GEMM paths |
| `GGML_CUDA_MMVQ_CHUNK_ALL` | 0 | `=1` chunks every MMVQ type through the integer path (KLD +87%) |
| `GGML_CUDA_GEMM_FOLD_NARROW` | 64 | batches up to N columns use the 32/64-column fold GEMM tiles (bit-identical); `=0` always 128 columns |
| `GGML_CUDA_FUSE_CONV_DECODE` | 1 | single-token conv state chain (CONCAT, CPY, SSM_CONV, SILU) in one launch; `=0` off |
| `LLAMA_CKPT_SKIP_CURRENT` | 1 | the server keeps a just-restored context checkpoint instead of copying it again; `=0` copies it |
| `LLAMA_CKPT_USER_SPLIT` | 0 | `=1` restores the extra prompt split and checkpoint at the last user message |
| `LLAMA_CKPT_MADVISE` | 0 | `=1` advises huge pages on checkpoint buffers (stalls with THP `defrag=madvise`) |
| `LLAMA_CKPT_POOL` | 4 | freed checkpoint buffers kept for reuse |
| `LLAMA_CKPT_PREFAULT` | 1 | a background thread keeps one pre-faulted checkpoint buffer ready; `=0` off |
| `LLAMA_THINK_BY_BUDGET` | 0 (`start-qwen.sh`: `THINK_BY_BUDGET`, 1) | `=1`: the template always renders in thinking mode; thinking off is the off render's generation prompt spliced on, so toggles keep the prompt cache; `reasoning_effort` becomes a budget instead of the template's effort line |
| `LLAMA_THINK_TEMPLATE_EFFORT` | the template's own | with `LLAMA_THINK_BY_BUDGET=1`: the one effort level rendered for every request |
| `LLAMA_THINK_EFFORT_BUDGETS` | `low=512,medium=2048` | with `LLAMA_THINK_BY_BUDGET=1`: thinking-token budget per `reasoning_effort`; other levels use `--reasoning-budget` (-1 = unlimited), `none` turns thinking off |
| `LLAMA_SAMPLING_BY_MODE` | 0 (`start-qwen.sh`: `SAMPLING_BY_MODE`, 1) | `=1`: chat requests get the sampling set of their thinking mode for every field they leave out (the CLI sampling flags then only serve `/completion`) |
| `LLAMA_SAMPLING_THINK` | `{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0}` (`start-qwen.sh`: `THINK_SAMPLING`) | the thinking set, a JSON object of sampling fields |
| `LLAMA_SAMPLING_NOTHINK` | `{"temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0.0,"presence_penalty":1.5}` (`start-qwen.sh`: `NOTHINK_SAMPLING`) | the thinking-off set |
| `LLAMA_SAMPLER_PREFILTER_PENALTIES` | 1 | with a repetition / frequency / presence penalty active, the sampler's top-k prefilter takes top_k + penalty_last_n candidates (same output; +2.2% with MTP, CHANGES §21); `=0` keeps the whole vocabulary |
| `LLAMA_SPEC_STOP_ISO` | 0 (off) (`start-qwen.sh`: `STOP_ISO`, 0.125) | `=<theta>`: the MTP draft stops when the estimated chance that the whole drafted prefix is accepted (two lookup tables fit on traced verify cycles) falls below theta, instead of the p-min test (§22). Exact; chat replies +4.7%, long context +4.8% at 0.125 |
| `LLAMA_SPEC_STOP_ISO_TEMP` | `0.5,0.9` | the target temperatures the stop rule applies to; requests outside keep the p-min test (the tables were fit at the thinking-off 0.7 and gain nothing at the thinking 1.0) |

## Benchmarking

    GGML_CUDA_P2P=1 ./build-opt/bin/llama-bench -m /path/to/Qwen3.8-27B-Q6_K.gguf \
      -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -p 0 -n 256 -r 5

Expect ~32 t/s (plain decode, no MTP). Measure on cool cards: right after a long run, P100s can
read up to 20% low. To check accuracy, run `tools/gate.sh`; perplexity should land near 2.61.
