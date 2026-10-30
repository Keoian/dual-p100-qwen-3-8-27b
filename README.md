# Qwen3.8-27B on two Tesla P100s

A fork of [Kmic-68's `p100-optimizations`](https://github.com/Kmic-68/llama.cpp/tree/p100-optimizations) (llama.cpp
tuned for Pascal, sm_60). It's tuned further for a **consumer board without P2P** and for a **chat assistant's
workload**: a long fixed system prompt, then short turns. The working branch is **`p100-optimizations-keoian`**
(the fork's default branch since 2026-10-22), and it serves the numbers below in production.

## What it runs

- **Hardware:**
  - 2x Tesla P100-PCIE-16GB at a 150 W power cap. No NVLink and no P2P: the second card sits behind the chipset
    on PCIe 3.0 x4.
  - 4-core i7, 48 GB RAM, Unraid with Docker, driver 580, CUDA 12.
- **Model:** Qwen3.8-27B UD-Q6_K with its MTP head, plus the Q8_0 vision projector.
- **Settings:**
  - the full **262,144-token context**, tensor-parallel across both cards;
  - KV cache **q8_0 K / q4_0 V**;
  - MTP drafts of up to 4 tokens, accepted by the `exact` rule: every token keeps the model's own odds (since 10-30;
    a request can ask for the faster, bounded `ratio:1.5`);
  - images up to 2560x1440's pixel area;
  - 32 context checkpoints and a 24 GiB RAM prompt cache.
- **Memory:** everything is in VRAM. The worst case, a 1440p image at 259k tokens of context, peaks at 15,185 and
  16,099 MiB of 16,384 per card.

## Measured (this box, production build and settings, 2026-10-22; decode re-measured 2026-10-24; MTP suite 2026-10-30)

| | |
|---|---|
| plain decode | **31.1 t/s** (10-24); 26.2 at 64k context, 23.4 at 128k (10-22) |
| decode with MTP (chat replies) | **~47 t/s** (per-reply median 46.8 over 2 x 24 replies to 8 prompts, thinking-off sampling; 52.8 over all their tokens; 10-24, before the MTP draft's stop rule below, which added +4.7% on a 72-prompt suite: 49.30 -> 51.61 t/s) |
| decode with MTP (72-prompt suite, 10-30) | chat replies **51.6 t/s**, thinking (medium) **51.4 t/s** with production's `exact` accept rule; 53.8 / 55.6 with `ratio:1.5` (10-29). At 32k / 64k / 128k / 200k context 53.1 / 48.7 / 41.3 / 37.8 t/s (exact, 10-28) |
| prefill | **410 t/s** at 512 tokens, 413 at 2,048; filling the window to 255k runs 355 -> 144 t/s |
| time to first token | **~0.5 s** for a short message on a 25k-token cached conversation |
| 1440p screenshot | vision encoder ~5.7 s; small text 96-100% legible; at 259k context: 31.6 s to read it in, then 27.3 t/s |
| accuracy | KV cache KLD vs an f16 cache **0.00116** (q4_0/q4_0: 0.00274), 99.1% same top token; perplexity 2.6064. The bf16 exchange wire adds 1.1% at verify widths (0.001090 -> 0.001102) |

## What it adds over Kmic's branch

Measured on Kmic's tip vs. his tip with each change (same box). Items marked *PR* are open pull requests to his
branch (#5-#16, listed in [CHANGES.md §20](p100-docs/CHANGES.md)); the softmax-race fix went in earlier as his #4.

**Boards without P2P:** the tensor-parallel exchange is staged through pinned host memory and overlapped with the
matmul. Prefill +14-20%, decode +2%; nothing changes on P2P boards. *PR*

**Short prompts:** 17-64-token batches run as matvec chunks and narrow GEMM tiles. Prefill of 17 tokens 39 -> 93
t/s, 64 tokens 126 -> 158. *PR*

**A q8_0 K cache at q4_0 speed:** a fast decode kernel for q8_0 keys, plus routing switches that stop a 262k q8_0
cache running out of memory. Decode +9%, +21% at 64k; the q4_0 default is unchanged. *PR*

**Vision:** the encoder runs on the GEMM attention path. A 1440p image encodes in 7.5 -> 5.7 s. *PR*

**Chat turns:**
- Context checkpoints are reused instead of copied, with no THP stalls, a buffer pool and a background pre-fault.
  Prompt time per short turn 1.3-1.5 s -> **0.7 s**. *PR*
- MTP drafts are correct after any resume. Slot save/restore and the RAM cache reproduce the live continuation
  exactly; restoring after generation reprocesses 4 tokens instead of 9,715. *PR*
- Branching back 3 turns in a 58k chat takes 77.5 -> 0.5 s, from the larger checkpoint and RAM cache.
- An abandoned 64k prompt frees the server in 33.1 -> 2.2 s (`-b 2048`). *PR (docs)*

**Thinking and sampling** (server env switches `LLAMA_THINK_BY_BUDGET=1` and `LLAMA_SAMPLING_BY_MODE=1`: off in the
binary by default, on in the deploy script; QUICKSTART "Switches on this branch"):
- Turning thinking on or off per request keeps the prompt cache: a toggle reprocesses 21-42 tokens instead of 2,146.
  (Qwen3.8's template puts its effort line at the top of the system prompt.)
- Each mode gets the model card's sampling for fields a client leaves out.
- `reasoning_effort` low / medium caps thinking at 512 / 2,048 tokens.

**Repetition penalties no longer scan the vocabulary:** the penalty stage visits only the window's tokens instead of
all 248k for every sampled token. Presence 1.5 with MTP 40.0 -> 44.7 t/s, identical output. *PR*

**Measured on this branch (2026-10-24, [CHANGES.md §21](p100-docs/CHANGES.md)):**
- the decode and verify exchanges cross the x4 link as bf16 (`AR_BF16=1` in the start script): MTP replies +6.8%,
  KLD +1.1% (lossy; `AR_BF16=0` for exact f32);
- with a repetition penalty active, the sampler's top-k prefilter takes top_k + penalty_last_n candidates (84 at
  production's settings) instead of sorting the whole vocabulary: +2.2%, identical output;
- the KV cache's position cleanup scans only the used cells, not all 262k: MTP replies +1.4%, identical output;
- Kmic's q4_0-K decode attention no longer spills to local memory at 3-4 tokens: -3 to -5% per call, bit-identical
  (production's q8_0-K kernel didn't spill).

**The MTP draft stops by a learned rule (2026-10-24, [CHANGES.md §22](p100-docs/CHANGES.md)):** instead of stopping
when the draft's top token falls below p 0.2, it stops when the estimated chance that the whole drafted prefix is
accepted falls below 0.125 (two lookup tables fit on traced verify cycles). Thinking-off chat replies +4.7%, long
context (32k-200k) +4.8%; the output distribution is unchanged. Thinking requests use their own tables, fit on
thinking-on traces (the thinking-off ones gained nothing there): +2.9%. Other temperatures (a client's own
sampling) use the chat tables too: +6.6% at T 0.3, +6.1% at T 1.2. `STOP_ISO=0` / `STOP_ISO_THINK=0` in the start
script turn them off.

**Accept rules for the MTP draft, chosen per request (2026-10-25 / 10-29, [CHANGES.md §23-24](p100-docs/CHANGES.md)):**
`"speculative.accept"` in a request (or `--spec-accept`) picks how drafted tokens are accepted. `exact` keeps the
model's own sampling distribution. The veto rules (`escape`, `esc+<x>`) keep the MTP head's token wherever the model
allows it: faster, but unbounded, and on 10-28 `esc+0.2` made a reasoning slip the model had started 2-3 times as
likely. `ratio:<c>` keeps a draft with probability min(1, c p/q) and fills rejections from the exact residual, so no
token is more than c times as likely as under the model's own sampling (`ratio:<c>:<C>:<N>` also bounds every run of
N tokens). `ratio:1.5` is chat +4.1%, thinking +7.6% over `exact`; production ran it 10-29 to 10-30 and runs `exact`
since (CHANGES §25).

**What limits the MTP draft under `exact` (2026-10-30, [MTP-DRAFT.md](p100-docs/MTP-DRAFT.md)):** at each drafted
row the draft's distribution overlaps the model's only 0.78 in chat and 0.74 in thinking (the chance the row is kept
once its prefix was; the server's acceptance counter, over all drafted rows, reads 0.68 / 0.66), and that mismatch,
not precision, is what's left. Cheaper
knobs (draft KV q8_0, a Q8_0 MTP block or draft LM head, a wider draft vocabulary) gained at most +0.9%. The document
has the proposal for the speed still on the table, the retraining of the MTP head it would take, and why no MTP work
can fix a slip the model makes itself (the "stepfather" riddle).

**Fixes:**
- a GGML_ASSERT on 9-16-token prompts in Kmic's current tip (*PR*; production's configuration never tripped it, and
  this branch has the fix too since `400aac811`, which production runs since 10-24);
- a use-after-free in the tensor-split backend (*PR*);
- a recurrent state that went NaN stuck to its slot until a restart (cleared with FILL instead of a SCALE by 0);
- LoRA adapters under `-sm tensor` (*PR*).

**Also:**
- JEV-27B "System 1" decisions on `POST /v1/decide` (off in production);
- [`llama-vision-bench`](tools/vision-bench/README.md) (*PR*, with the `-b 2048` docs);
- a deploy kit (Dockerfile, checksummed model downloads, start scripts, Unraid startup scripts).

## Run it

[p100-docs/deploy/](p100-docs/deploy/README.md) brings a fresh Docker host with two P100s to this exact setup:

1. Clone `-b p100-optimizations-keoian`.
2. Build the image.
3. `build.sh build-vision`, then `fetch.sh` (pinned, checksummed downloads).
4. Create the API key file `/work/qwen-api-keys`. Without it the start script serves with no key at all.
5. Start `start-qwen-vision.sh`.

Clients use `http://<host>:8090/v1`.

The fork's `p100-optimizations` branch is an old snapshot of Kmic's (2026-10-01). His current branch is
[Kmic-68/llama.cpp `p100-optimizations`](https://github.com/Kmic-68/llama.cpp/tree/p100-optimizations).

**Why each setting** (and how it differs from Kmic's command): [p100-docs/QUICKSTART.md](p100-docs/QUICKSTART.md).
**The exact production command:** [p100-docs/deploy/README.md](p100-docs/deploy/README.md).
**Every change with its numbers:** [p100-docs/CHANGES.md](p100-docs/CHANGES.md), sections 15-22.
**What was learned:** [p100-docs/FINDINGS.md](p100-docs/FINDINGS.md).

Built with Claude Code (AI-assisted). Every commit carries a `Co-Authored-By` trailer, and every number above was
measured on the box described here.

---

**From Kmic's fork** (his summary and numbers, on a board with P2P):

> **This is a fork of llama.cpp tuned for two Tesla P100s (sm_60).** On Qwen3.8-27B, plain decode
> runs at 1.86x upstream's speed (17.5 -> 32.6 t/s), and ~3x with the model's built-in MTP head
> (~54 t/s). Prefill runs at ~2x (250 -> 493 t/s at short context). The full 262k window fits with vision
> loaded, and a question on a full 260k context prefills at ~153 t/s.
> The math is at least as accurate as stock: KLD against an fp32 run 0.00119, where upstream's
> fp16 path reads 0.00152. It tracks upstream by merging.
> See **[p100-docs/](p100-docs/README.md)** for results, changes and how to run it.

---

Upstream's README follows.

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
