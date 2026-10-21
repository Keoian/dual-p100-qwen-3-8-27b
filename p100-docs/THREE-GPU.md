# Three P100s

Measurements of this fork on **three** Tesla P100-PCIE-16GB cards under `-sm tensor`, by
[@4claps](https://github.com/4claps) ([PR #2](https://github.com/Kmic-68/llama.cpp/pull/2), which
has the raw data, the run harness and the full write-up). This page keeps what transfers. Nothing
here was reproduced on the 2-card machine the rest of these documents describe.

## The machine

- 3x P100-PCIE-16GB, each on PCIe 3.0 x8, all behind one host bridge (no NVLink, no PCIe switch).
- Core i7-3930K (6 cores, 2011), **16 GB RAM**, model on a SATA SSD. Docker, driver 580.178.04,
  CUDA 12.9.1.
- Build `ae35056eb` (tag `p100-optimizations-b11515-ae35056`), sm_60 only.
- 125 W power cap per card for most runs, 150 W for the settings finally served.
- Qwen3.8-27B unsloth `UD-Q6_K_XL` ("XL"), plus unsloth Q8_0. `-ctk q4_0 -ctv q4_0`, `-ub 2048`.

## Results

`GGML_CUDA_P2P=1 GGML_CUDA_GRAPHS_PRE_VOLTA=3`, `-lm none`, 125 W, mean of 3 repetitions.
"NCCL" is a build with `-DGGML_CUDA_NCCL=ON` run with `NCCL_P2P_LEVEL=SYS`.

| | 3 GPUs | 3 GPUs, NCCL |
|---|---|---|
| decode, `tg512` | 29.44 t/s | 31.09 t/s |
| prefill, `pp2048` at 0 context | 275.9 t/s | 499.8 t/s |
| prefill, `pp2048` at 16k | 259.6 t/s | 444.9 t/s |
| prefill, `pp2048` at 64k | 211.8 t/s | 318.7 t/s |
| MTP, n-max 3, p-min 0.0, greedy, mean of 3 prompts | 48.4 t/s | 47.3 t/s |
| 129,000-token prompt through `llama-server` at `-c 262144` | 211.0 t/s | 315.9 t/s |
| peak VRAM, fullest card, that prompt | 10.8 GiB | 10.9 GiB |

At 150 W, NCCL: decode 33.3 t/s and `pp2048` 530 t/s, i.e. +6.5% and +4.8% over 125 W. 175 W adds
only another +1.1% decode.

The 129k prompt fills half the context, and VRAM grows as the context fills, so the per-card
figure is not the full-context figure. A full 262k prompt was not run.

## What transfers

- **`-lm none` on a host with less RAM than the model.** With mmap, decode varied 6.8% run to run
  and was 11% slower on average; without it, 0.9%. A machine that holds the whole file in page
  cache won't see this.
- **Graphs.** Without `GGML_CUDA_GRAPHS_PRE_VOLTA=3`, decode on this host is bound by enqueue
  time (43.6 of 45.3 ms per token inside `llama_decode`): 22.4 t/s, 29.5 with it. QUICKSTART already
  sets it. The weaker the CPU and the more cards, the more it matters.
- **The tensor-parallel exchange dominates 3-card prefill.** 54-57% of prefill time at 16k-64k
  depth with the default exchange. That is why NCCL's ring all-reduce helps prefill so much here.
  Two cards have one peer pair and a simpler exchange, so the gain there is probably smaller.
  Not measured.
- **Cancelling a prompt waits for the batch in flight.** The server acts on a dropped client only
  between `llama_decode` calls. At `-b 32768` a 64k prompt dropped halfway held the slot 72-74 s
  after the drop, at `-b 2048` about 4 s. `-b 2048` cost 0.3% of 64k prefill.
  This partly contradicts QUICKSTART's note that MTP needs `-b 32768` at long context: they served
  `-b 2048` with MTP and decoded 37.4 t/s after a 64k prompt. Only one long prompt was checked.
- **Polling `/slots` blocks cancellation.** A client polling `/slots` five times a second delayed
  the disconnect check until prefill finished (~80 s), at every `-b`. `/health` every 5 s had no
  effect. The cause is a reading of the server code, not a measurement: every result wakes the
  waiting handler and restarts the 1 s timeout under which it checks the connection. The same code
  is upstream.
- **`-np 2` works** with tensor split, NCCL and MTP, but a second client arriving during a long
  prefill still waits for it to finish.

## NCCL is not exact

The NCCL speedup comes with different output, which is why this fork doesn't build with it:

- Prefill exchanges of 26 or more tokens go over the wire as **BF16**. Decode exchanges stay f32,
  but in a different summation order.
- Against the non-NCCL path: mean KLD 0.0031 at 16k context (max 14.7 at one position), 0.0009
  over one 65,536-token context. That is less than changing `-ub` from 2048 to 5 (0.0070), but it is
  not zero, and greedy generations diverge.
- It is repeatable: identical logit hashes across runs.

## Open

- **2 against 3 GPUs is unresolved.** The only 2-GPU runs had graphs off and mmap on, which both
  distort decode on this host. They were never repeated.
- **nvprof worked** on this driver inside the container, including through CUDA graphs. Here it is
  dead on the same driver series ([FINDINGS.md](FINDINGS.md)). Not investigated.
- One machine, one model, single runs for some cells. PR #2's `LIMITATIONS.md` lists every caveat.
