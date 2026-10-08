# Changes

Every code change in the fork, grouped by what it touches. Measurements are on 2x Tesla
P100-PCIE-16GB, Qwen3.8-27B Q6_K, q4_0 KV cache, `-sm tensor`. [`../OPTLOG.md`](../OPTLOG.md) has
the attempt-by-attempt record, including everything that was reverted.

## 1. `mul_mat_vec_q`: the decode matvec

Most of decode time is spent here. Together these took decode from **17.51 to ~31 t/s**.

| commit | change | effect |
|---|---|---|
| `b44f8fe6f` | Pascal launch geometry (warps and rows per block) | the first sm_60 tuning pass |
| `2bb2264dd` | drop `__vsubss4` from the Q6_K/Q3_K dot products | sm_60 emulates it in 9 instructions. A bias trick folds the subtraction into shifts that were needed anyway |
| `4d9dbeb34` | stage the weights `x` through shared memory | every block type is 2 mod 4 bytes, so direct reads need split 16-bit loads. Staging keeps global reads aligned |
| `a277ff94f`, `e97421a3d` | `vdr` 2, then 4, for Q6_K; geometry moved into the source | scales and q8_1 metadata fetched once per four lanes |
| `be811a6d1` | stage in 16-byte units | ~4x fewer global loads and shared stores |
| `62d9e35f1` | accumulate a whole `vdr` group as an integer before scaling | fewer int→float conversions; rounds less often, so slightly more accurate |
| `090f53560` | reuse the q8_1-quantized activation across calls | it was being requantized per call |
| `c718d1860` | stage the q8_1 activation through shared memory | **the largest single win.** Every block re-reads the activation, so it, not the weights, sets the cost |
| `dd0e5289b`, `f3cb02935`, `2c1f89b12` | multi-column (speculative) path: per-warp rows, 16-row blocks, block-wide staging | |
| `c3aaef65e` | vectorised Q6_K dequant | bit-identical, 11 → 6 memory instructions per thread |

The DP4A emulation (sm_60 has no `__dp4a`) is 8 instructions via PRMT + XMAD.H1, and bit-exact.

## 2. Flash attention: the tile kernel

| commit | change | effect |
|---|---|---|
| `3f49203da` | **dequantize the q4_0 KV tile straight into shared memory** | `launch_fattn` was converting the *entire* KV cache to f16 on every call: 4.15 ms per call at 262144 context. **9242 → 6213 µs**, and 512 MiB per GPU of staging freed. Numerically equal to upstream's conversion for every finite scale |
| `b574f0b98` | don't reserve f16 staging when the tile kernel reads q4_0 directly | the 512 MiB |
| `88211649b` | `hfma2` in the tile loader | |
| `5fc820f9d`, `2127ac5bf`, `da3bddaeb` | fold the whole GQA-6 group into one block (vec, tile, two-column) | the KV cache is read once per group instead of once per head |
| `8599fe022` | exact-fit tile for the MTP verify batch | a 5-token verify was padded to two 4-token tiles, so 37% of the work was padding. A 36-column tile fits. **1.24x** on that shape |
| `5ea4b2712` | `nbatch_K = 128` on the narrow GQA-6 tiles | 1691 → 1518 µs. Applied only where it helps: it's 17% worse on the wide tile |
| `edc7980bf` | **fold fp16 accumulation into fp32 once per tile** | the output accumulated over the whole cache in `half2`: a quarter-million adds in an 11-bit mantissa. **8.7x more accurate at depth, for 2.4%**, and the error no longer grows with context |
| `43543917b` | give `launch_fattn` the vec kernel's real KV tile size | short contexts were using a twelfth of the SMs |
| `961e63c18` | read each q4_0 byte once in the tile loader | two threads each loaded the same byte and kept half. **−13.1% decode time at 262144** |
| `c6f5211f4` | magic-number dequant (OR the nibble into `1024.0h`, subtract 1032) | no convert instruction. **−16.5% at 262144**, bit-identical over all 65536 scales × 256 bytes |

Reverted for accuracy: `7c77a2b80` accumulated the KQ dot product in `half2`. It was 20% faster
and perplexity couldn't see it, but it rounds measurably more (RMS error ×1.22, as predicted by
where the second rounding lands). Details in FINDINGS.

## 3. GEMM attention for long-context prefill

A cuBLAS-GEMM attention path for pre-Volta, on by default at batch ≥ 128 and KV ≥ 4096
(`GGML_CUDA_FA_GEMM=0` turns it off). At these shapes the tile kernel reaches 18.6% of fp16 peak,
while cuBLAS reaches 13-15 TFLOPS. With a q4_0 cache, the cuBLAS calls inside this path have since
been replaced by the fold kernels (§12, §14); `GGML_CUDA_FA_FOLD=0` brings them back.

| commit | change |
|---|---|
| `738022bda` | the path itself |
| `0f5b88954`, `dbbee401a`, `0b0e16a03` | alias P onto S; PV GEMM in f16; one GEMM call instead of a GQA batch |
| `d85d55edd` | skip all-masked chunks, decided on the GPU |
| `a7cdad458` | PV requests cuBLAS `ALGO4`. The default picks a long-chain fp16 kernel; ALGO4 has 3.4x lower error |
| `fbf220c10` | decline mask shapes the path can't honour |
| `619b6031e` | restore the fp16 overflow guards; q4_0 tile bias in integer |

## 4. Tensor parallel across the two cards

| commit | change | effect |
|---|---|---|
| `27961ce6c` | peer copies on a dedicated stream, so both directions overlap | also introduced a race, fixed in §6 |
| `e5c264b71` | ship partials as f16 when lossless; pipeline the delta-net reduction | half the bytes over PCIe |
| `dce17bf1b` | decide f16 compression per exchange, from the matmul's actual compute type | matters for mixed-type models and for architectures that force fp32 there |
| `1b29f55de` | walk `gated_delta_net` addresses instead of recomputing them | |

Upstream's internal AllReduce stays off on Pascal. It measured 17% slower here, because these are
PCIe cards and it assumes NVLink.

## 5. cuBLAS precision

| commit | change | effect |
|---|---|---|
| `fccdafca1` | Pascal fp16 prefill matmuls request `CUBLAS_GEMM_ALGO6` | the default picks a long-chain fp16 accumulator from ~256 rows up. ALGO6 is **10x more accurate at every shape measured**, and 63% faster at pp512 |
| `55e496262` | revert ALGO3 for wide GEMMs | it reassociated with no precision argument behind it |

## 6. Correctness fixes

Four of these fix bugs the fork itself introduced. Neither data race was visible to
`test-backend-ops`.

| commit | bug | evidence |
|---|---|---|
| `cb6024e6b` | the GEMM attention softmax wrote probabilities over scores it was still reading | in-op self-check: 3-6 of 2240 launches differed in place, 0 of ~6700 out of place |
| `13b24fe27` | the same path accumulated its output in place | made out of place |
| `b67848c64` | a tensor-parallel peer copy could overwrite the all-reduce buffer before the other card had read it. It hit decode and MTP, not only prefill | with the race forced, MTP produced different text at 43% acceptance instead of 79%. The fix costs 0.7% of decode |
| `5d479e6f8` | CUDA graphs captured the q8_1 buffer pointer without tracking it | graphs-on output is byte-identical to graphs-off across 61 replays |
| `c1f7f4b00` | 16 bytes of slack for the mmvq staging over-read | |
| `24290a858` | mmvq row guards bounded by the wrong stride | |
| `9e99d468f` | fastdiv domain guards at 2^32 instead of 2^31 | |
| `fe0e5c811` | restore bit-identical output in the norm kernels | |
| `194190ef7` | the same-GPU copy between two virtual devices didn't wait for its reader | fixed by reading the code; no test exercises it |
| `fd560af8d` | an empty tensor-parallel slice was zeroed by multiplying by 0.0f, which keeps NaNs | fixed by reading the code; the branch never runs on this model |

## 7. CUDA graphs and speculative decoding

| commit | change |
|---|---|
| `b302163d6` | allow CUDA graphs on Pascal, opt-in with `GGML_CUDA_GRAPHS_PRE_VOLTA=1`. +6.7% on the MTP path, −2% on plain decode |
| `74de4a1bd` | `-ubd`: a separate ubatch for the draft context |

## 8. Tests

`282918f2e`, `cffc2b191`, `2f50214e1`, `232797c00`, `33ff1a5ba` and `a8f1ee60d` add FLASH_ATTN_EXT
cases at the shapes this fork runs, out to the real 262144 context. That shape had no coverage
before, which is how the fp16 accumulation error in §2 went unnoticed.

## 9. Upstream merges

### 2026-09-22: upstream `f46bc30cb`

502 upstream commits since the fork point `f280b2698`, including new model architectures,
sparse flash attention, and reworked speculative decoding. Four files conflicted. Resolving them
safely took more than the conflicts:

- **`fattn-tile.cuh` (merged cleanly, but wrong).** Upstream added a `use_sparse` parameter to
  `launch_fattn` before `warp_size`. Our GQA-6 tile launch passed `warp_size` positionally, so
  the `int` bound to `use_sparse` and switched sparse attention on in Qwen's q4_0 attention path.
  It compiled without a warning. On this model it would have aborted on an assert at the first
  attention call; on a model that sets a sparse KV hint, it would have run the wrong kernel.
  Fixed, and every `launch_fattn` call was checked for arity.
- **`convert.cu`.** Upstream landed the same vectorised f32↔f16 cast as ours (`17455ce35`). Theirs
  is kept and ours dropped. Same per-element cast, so the output is bit-identical.
- **`ggml-cuda.cu`.** Upstream's cuBLAS compute-type rule now depends on the batch width (`src1`),
  for BF16 on older GPUs. The fork had moved that rule into a helper that the tensor-parallel
  compression check also calls. The helper now takes `src1`, so both callers agree.
- **`fattn-vec.cuh`, `test-backend-ops.cpp`.** Both sides' changes kept.
- `GGML_CUDA_FA_ALL_QUANTS` is deprecated upstream. Builds use `GGML_CUDA_FA_QUANTS=all`.

Checked unchanged: the DP4A emulation (byte-identical), the Pascal mmvq geometry and staging,
the Q6_K/Q3_K dot products, the direct-q4_0 tile path, GEMM attention dispatch, and `-ubd`.

**The output changed slightly, on purpose.** Upstream `5fdfa6282` corrects the gated delta-net
q/k normalization from `x / max(‖x‖, eps)` to the reference `x · rsqrt(Σx² + eps)`, the form
flash-linear-attention, transformers, vLLM and SGLang all use. That moves the logits by a mean
KL divergence of 0.0015 and leaves perplexity unchanged. With that one change reverted as a
diagnostic, the merged build reproduces the pre-merge logits exactly (KLD < 1e-5, 100% identical
top tokens), so everything the fork computes survived the merge unchanged. Decode speed is also
the same either way.

| gate | before (shipped build) | after |
|---|---|---|
| `tg256`, cold cards, back to back | 30.76 ± 0.18 | 30.57 ± 0.20 |
| perplexity | 2.6097 ± 0.0198 | 2.6101 ± 0.0198 |
| KL divergence vs before, 8 chunks | — | 0.0015 mean, 98.6% same top token (all from `5fdfa6282`) |
| `test-backend-ops` | 14593/14593 | 16180/16180, both GPUs |

## 10. Real-world MTP speed, and decode at depth (2026-09-22/23)

Measured through `llama-server` with the production flags (`qwen-server`'s, MTP on, sampling
at `--temp 0.3 --top-k 20`), from saved slot snapshots at 2k, 16k, 64k, 128k and 260k context.
The metric is milliseconds per MTP cycle (one verify, one catch-up and the draft steps), since
tokens per second also depend on how many drafts the text happens to accept. `tools/depth-bench.py`
does the filling, the snapshots and the measuring. OPTLOG attempts 177-190 have the details,
including what was reverted.

| commit | change | effect |
|---|---|---|
| `f7312ff1f` | **q4p: an fp32 flash-attention kernel for 1-5 tokens over the q4_0 cache.** Dequant is 2 instructions per value straight into fp32 (PRMT the nibble into a float's mantissa, one FADD), each value is reused by all 6-30 query rows, and Q, products and sums stay fp32 | at 262144: 1225 → 1001 µs (1 token), 4882 → 4307 (5 tokens). Half the per-op error of the tile kernel |
| `04e9262d1` | 1 row per block for small q6_K matrices at 2-8 columns | the 5120x24 delta-net matmul at 5 columns: 35 → 10.7 µs |
| `45d466bea` | radix-select top-k when CUB has no DeviceTopK | a 200k-entry row: 136 → 71 µs |
| `66bbd1212` | **the samplers pick the top k straight from the logits** when nothing ahead of top-k in the chain can reorder them | under `-sm tensor` both the draft and verify samplers run on the CPU. Each sample built and partially sorted a 248k-entry array (~650 µs); one SSE2 scan takes 36. **−3 to −5 ms per cycle (4-5%)**, with byte-identical output |
| `2bc1a9ac0` | **q4p hides its memory latency.** The softmax denominator leaves the PV loop (4 of 8 warps were adding into it, divergent, at every position), and the freed registers load V one position ahead and the next chunk's K during PV | at one block per SM it waited on memory. In the server at 260k: 5 tokens 4.165 → 3.906 ms, 1 token 1.071 → 1.016. MTP cycle −2.8% at 260k |
| `a8b274ea6` | **tensor-parallel graph rebuilds in ~20 ms instead of ~45.** The meta backend's split-state cache and tensor map are hashed, looked up once, and their scratch pooled; `GGML_BACKEND_META_MAX_DEVICES` 16 → 4 shrinks the split state it copies thousands of times per rebuild | the target graph rebuilds whenever the batch shape changes: a new prompt, a KV size step, a different verify width. Byte-identical output |
| `f9152a548` | slot save/restore also saves the MTP draft context | a restored slot drafts at full acceptance |
| `8736a7ef3` | **fixed-width verify, and a draft length that follows the depth.** The server pads a short draft to `n_max`, so the verify graph keeps one shape and is never rebuilt, and declares the real token count per decode (`llama_set_n_active_tokens`). The q6_K matvec and the q4p attention then compute only the real tokens, exactly what a narrower graph gives them. With width changes free, the MTP draft also stops once the product of its top-1 probabilities falls below 0.3 from 48k context on (0 below 16k, a ramp between) | ABBA, 8 requests per arm: 2k 40.5 → 41.1, 64k 30.0 → 34.0, 128k 28.7 → 30.6, **260k 20.9 → 24.3 t/s (+16%)**. Byte-identical text when draft lengths do not vary. `LLAMA_SPEC_PAD=0` turns it off, `LLAMA_SPEC_P_CUM=<p>` fixes the threshold |
| `1f6a8681a`, `2dfb4ae4f` | single-sequence fast path for the first row of the KQ mask: read only the 4-byte cell position, vectorized, instead of each cell's 32-byte sequence set. The second commit makes it run under M-RoPE (Qwen3.5's 2-D positions, rechecking only the cells at the token's own position) and for non-causal masks | the mask fill was ~10% of the server's host thread at 260k. MTP cycle −1.5% at 64k, −2.2% at 128k, −3.7% at 260k; a draft step at 260k 5.2 → 4.3 ms. Byte-identical output (`LLAMA_KQ_MASK_FAST=0` to compare) |
| `34a9545a5`, `fe9b48d87`, `8ea46770a` | `GGML_CUDA_OP_PROFILE=1`, `LLAMA_UBATCH_PROFILE=1`, `LLAMA_SPEC_PROFILE=1`, `LLAMA_SPEC_LOG=<path>`, and `tools/pmp` | per-op GPU times, host phases per ubatch, MTP step times, a per-cycle draft log, and a sampling CPU profiler for when `perf` is locked. All off by default |

**Why the MTP cycle costs what it does.** An nsys trace at 2k context shows ~64 ms of GPU work in
a ~72 ms cycle. The 5-token verify is 56.8 ms of it, and that is almost all the 5-column q6_K
matvec, which is at its floor for exact arithmetic (OPTLOG 179 and 187). The rest is host time:
sampling, which this round cut in half, plus graph rebuilds in the draft context and synchronous
input uploads. Deeper in, attention takes over. At 260k about half of each cycle is 17 q4p calls.

**Draft length.** The draft model's top-1 probability is well calibrated here: a drafted token
with p 0.5-0.6 is accepted 54% of the time, 0.9-1.0 95%. A rule that stops drafting once the
product of those probabilities falls under a threshold only pays where a verify token is
expensive. At 2k a verify of 2/3/4/5 tokens costs 46/53/58/64 ms, so a shorter draft loses more
tokens than it saves time. At 260k it costs 75/91/106/~120 ms and the rule gains 10-16%.
Without the fixed-width verify each change of width rebuilt the 64-layer graph (~26 ms:
1.8 ms graph build, 15 ms scheduler allocation, 9 ms meta subgraphs), which ate the gain. A
cost-aware rule that learned the verify times online did worse than the plain threshold
(OPTLOG 198).

**`--spec-draft-n-max`.** Before the depth-scheduled draft length, drafting almost never stopped
early at `--spec-draft-p-min 0.2`, so every verify was `n_max + 1` tokens wide, and n_max 3 was
+17% at 260k (24.1 against 20.6 t/s) but −3% at 2k. The draft rule now shortens drafts at depth
by itself, so `qwen-server` keeps 4.

## 11. fp16 math with an accuracy fix, and the MTP cycle (2026-09-24/25)

The P100 runs fp16 multiply-adds (HFMA2) at twice the fp32 rate, but plain fp16 accumulation is
noisy. Every kernel here keeps fp16 *products* and moves the running sums into fp32 every few dozen
values, so the error stops growing with the length of the sum. The half-to-float step is done with
integer instructions (shift and mask, the rebias folded into a later multiply), because the F2F
conversion runs at a quarter rate on sm_60. Quantized values enter as exact small integers
(PRMT builds half 1024 + n, one subtraction leaves n exactly). OPTLOG attempts 201-222.

| commit | change | effect |
|---|---|---|
| `5b46e14ca` | serving uses the model card's sampling (`--temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0`) | every MTP figure before it was measured at temp 0.3 |
| `dbe945b03`, `eb1bf26fe` | **sampled MTP drafts, verified with the speculative-sampling rule** (accept with min(1, p/q), else draw from the normalized residual) | lossless: the output distribution is the target's. +15% tokens per cycle at 2k, +8% at 260k |
| `ef660212c` … `6554f989d` | **fp16 q6_K matvec for the 2-5 token verify** (`mmvq-f16.cu`): exact 6-bit weights, power-of-2 prescaled activations, short HFMA2 chains folded to fp32 | verify pass −7-10%; NMSE vs fp64 4.6e-7, where the q8_1 integer path gave 1.3e-4; verify-path KLD vs fp32 0.00354 → 0.00162 |
| `25ff35ec5`, `75f93da7b`, `2b511b10f`, `e7a59fb1b` | q4p: the 5-token verify over half the GQA group per block, DPT 8, parallel softmax denominators | 5-token call at 260k 3.55 → ~2.95 ms in the server |
| `c4742ebe2` | the MTP catch-up pass stores K/V only | −4 ms per cycle at 260k |
| `5d27ef852` | CUDA graphs for single-token graphs only (`GGML_CUDA_GRAPHS_PRE_VOLTA=3`), the serving default | the draft steps without the VRAM cost of graphing the verify |
| `4593db3cf`, `2948e24bf`, `463420e47`, `4cb43bf86` | delta-net state and conv-state read in place, a flat concat, batched conv-state snapshots | ~−1.5 ms per verify |
| `ebc43ddf6`, `3ba045898`, `beee227db`, `b3f30b544` | one sync per split for inputs, async host-input uploads, a one-pass SSE2 mask fill | host time per cycle |
| `fc1c9056f` | **prefill GEMM with fp16 products and fp32 accumulation** (`gemm-fold.cu`), replacing cuBLAS COMPUTE_16F on Pascal. 128x128 tiles, half2 chains folded every 256 values of K, XOR-swizzled smem. Matmuls under 1024 rows go to fp32 cuBLAS | matmul NMSE vs fp64 1.2e-5 → 1.4e-6. KLD vs an all-fp32 run 0.00152 → 0.00125 (fp32 with only a different summation order: 0.0006-0.001). Perplexity 2.6101 → 2.6096 (all-fp32: 2.6095). pp1024 −6%, less at depth |
| `0be3a47d9` | **q4p attention in fp16 with fp32 folds** (decode and verify). QK: two KV positions per HFMA2, one chain per 32-dim block folded through the block scale. PV: P as half2, chains over dimension pairs folded every 32 positions | at 262144: 5-token verify 3128 → 2682 µs, 1 token 942 → 835. 260k MTP 25.93 → 27.11 t/s (ABBA, hot cards). Verify-path KLD vs fp32 unchanged (0.00115 → 0.00114) |

**What is left of the distance to fp32.** With accumulation fixed, the remaining difference is
the fp16 rounding of the *inputs* (weights and activations). Rounding either one alone in an
otherwise fp32 GEMM gives the same KLD (~0.0011) as the fold kernel: at this model's sensitivity,
any perturbation at fp16 resolution spreads to about that level, and fp32 with a different
summation order lands at 0.0006-0.001. Removing input rounding would need a hi/lo split of the
activations, i.e. twice the math.

## 12. Decode and prefill at depth, round 2 (2026-09-25/26)

Goal: 260k prefill 95-100 t/s, MTP decode 50 t/s at 2k and 30 t/s at 260k on hot cards, with no
loss of accuracy. OPTLOG 223-228.

| change | effect |
|---|---|
| fold-path prefill attention for a q4_0 cache (`fattn-gemm.cu`, fp16 HFMA2 with exact folds into fp32) | 262k x 1024 op 318 -> 270 ms; 260k prefill ~85 -> 100-108 t/s hot. KLD vs fp32 0.00125 -> 0.00122. `GGML_CUDA_FA_FOLD=0` = cuBLAS path |
| q4p verify/draft attention: PV reduce-scatter, select-free shuffles, mask preload, packed Q/P smem rows, no chunk-end barrier (`fattn-q4p.cuh`) | 260k verify call 2.70 -> 1.93 ms per layer; draft call 0.96 -> ~0.8 ms. Verify KLD 0.00113 |
| MTP draft head over the first 81920 tokens, q4_0 copy made at load (`LLAMA_MTP_DRAFT_VOCAB`, 0 = off) | draft step 2.05 -> ~1.4 ms. Lossless: every token is still verified against the full output |
| fp16 q6_K verify matvec: scales once per warp, 7 blocks/SM when it saves a wave, 1 warp x 2 rows for 256-3071 rows | bit-identical; verify pass ~-2 ms |
| launch fusions: residual ADD into RMS_NORM+MUL, delta-net gate and l2 norms, alpha/beta matvec epilogues, conv-state CONCAT+CPY, gated norm | bit-identical; ~330 fewer launches per verify pass |
| host: fusion-check cache, 1-token attention inside the single-token CUDA graphs | draft-step enqueue 0.61 -> 0.45 ms |

`qwen-server` kept `-ub 2048` for text only and switched to `-ub 1024` with `--mmproj`: with the
draft head and the fold scratch, `-ub 2048` plus the vision projector left GPU0 162 MiB at 262k
(§14's compact mask removed that limit). Tried and reverted: CUDA graphs for the verify
(slower, +100 MiB), register prefetch in the big matvec (slower).

## 13. Ideas from other projects, round 3 (2026-09-26)

A survey of ik_llama.cpp, upstream llama.cpp, vLLM, TensorRT-LLM, SGLang, ExLlama, other P100 forks
and papers, filtered to low-to-medium effort changes that keep the math exact. OPTLOG 230-235.

| change | source | effect | math |
|---|---|---|---|
| one-kernel P2P AllReduce for decode/verify-sized tensor-parallel exchanges (`GGML_CUDA_AR_P2P=0` = off) | ik_llama.cpp `reduce.cu`, vLLM custom all-reduce | plain decode +1.3%, verify pass +1.0% | bit-identical |
| FFN gate + up + SwiGLU in one fp16 verify matvec (`GGML_CUDA_FUSE_FFN_GLU=0` = off) | ik_llama.cpp fused up/gate | verify pass +2.1% | bit-identical |
| scheduler resets only the hash entries it used | shinbunbun/llama-cpp-p100-patches #21 | host time per graph rebuild | exact (host only) |
| block verification of sampled MTP drafts (`LLAMA_SPEC_BLOCK_VERIFY=0` = off) | Sun et al., arXiv:2403.10444 | +1.8% accepted drafts on the same drafts | same output distribution (proved; Monte Carlo checked) |

Every change was checked against the shipped build: perplexity 2.6096 (identical), KLD at verify
width identical to every printed digit, full op suite 16324/16324 on both GPUs. Against the shipped
build, ABBA through the server: MTP cycle time 2k 56.6 -> 53.8 ms (-5%), 260k 90.1 -> 84.3 ms (-6%),
prefill unchanged within noise (these changes don't touch prefill). Draft top-p
(`LLAMA_SPEC_DRAFT_TOPP`) is available but showed no gain.

## 14. Prefill, round 4 (2026-09-27 to 10-01)

Goal: prefill at 260k toward 200 t/s, exact math only (no sparse or approximate attention).
OPTLOG 236-272; `p100-handoff/GOAL-PREFILL300.md` has the analysis.

| commit | change | effect | math |
|---|---|---|---|
| `1f92a1b48` | overlap the prefill tensor-parallel exchange with the matmul, in token chunks | pp2048 +4.5% | bit-identical |
| `01f08ae73` | fold GEMM with cheaper bookkeeping | pp2048 +8.5% | same math |
| `77e05601d` | chunked gated delta net for prefill | pp2048 +3.8% | KLD vs fp64 tied with the recurrence |
| `4f261d20b` | fill the exchange wait with the next weights' dequant; gate/up paired; one-pass prescale | pp2048 ~465 -> ~478 | bit-exact |
| `9f7b27a8f` | **compact causal KQ mask**: per-row prefix lengths instead of an n_kv x n_tokens f16 mask | `-ub 2048` now fits at 262k with vision (GPU0 732 MiB free, measured with the desktop on it) | bit-exact |
| `26fc1e193` | fold attention skips fully masked tiles | | exact |
| `c5d1110a2`, `dd24ec3b8` | warp-per-row RMS norm for short rows; tiled concat for the delta-net conv input | +0.9%, +0.3% | bit-exact |
| `6339691f2`, `e7d85e3b5` | fold GEMM: register-bank-fixed SASS (`gemm-fold-u2-sass.h`), j-outer loop | +1.2%, +0.6% | bit-identical |
| `cceb8641e` | SwiGLU fused into the fold GEMM's activation prescale | +0.5% | exact |
| `a740e7c76` | fold attention in 2048-key chunks | 260k prefill +2.2% | KLD vs fp32 0.001175 -> 0.001172 |
| `030210e2b`, `838b87d51` | `fa_fold_pv2`: PV on the fold GEMM's main loop | PV -14% at 262k | NMSE vs fp64 equal or better |
| `05aa500bb`, `cda08dde0` | bank-fixed SASS for the fold attention kernels (`fattn-fold-sass.h`); QK loop j-outer | -1.2%, QK -2.3% | bit-identical |
| `c704addc5`, `4fd8612ed` | server: prompt checkpoints skip the draft's state when it truncates by position; checkpoint buffers without zero fill, on huge pages | 260k prompt +5-6%; checkpoint 100 -> 49 ms | output identical |
| `188086d8e`, `526106acd` | pin the MTP output buffer once at setup, not inside the first prompt | first 260k question ~148 -> ~153 t/s | host only |

`tools/gate.sh` now also checks that both SASS headers match their kernel source
(`u2cubin.py --check`, `facubin.py --check`).

Against the 2026-09-26 release, same session, both with vision loaded:

| | 09-26 release | this build |
|---|---|---|
| prefill at 260k, 1479-token prompt (1st / 2nd question) | 120 / 124 t/s (`-ub 1024`) | 153 / 155 t/s (`-ub 2048`) |
| `pp2048` at 0 context, `-ub 2048` | 385 t/s | 493 t/s |
| GPU0 free at 262k | 508 MiB | 792 MiB |
| KLD vs fp32, mean / max (4 chunks) | 0.001215 / 0.134 | 0.001186 / 0.115 |
| same top token as fp32 | 98.80% | 98.86% |

The release can't run `-ub 2048` with vision at 262k (GPU0 ran out), so it is shown at its
shipped `-ub 1024`. A full fill from empty to 260k measured 24.9 minutes of prefill (09-26
release: ~29, from its depth table). QUICKSTART has the 2k-260k sweep.

**What limits it now.** The three big prefill kernels (fold GEMM, QK, PV) run at the cards' 175 W
power cap, not at an instruction limit: at the cap the clock settles at ~1290 MHz and they reach
~13 TFLOPS. Removing yields or bank conflicts doesn't help, because the cost is energy per flop.
At that cap, an estimated ~176 t/s is the ceiling at 260k with exact math.

## 15. tyler-port: a board without P2P, short chat turns, exact restore (2026-10-04 to 10-06)

Branch `tyler-port` (Keoian/dual-p100-qwen-3-8-27b), on top of `ae35056eb`. The board is an ASUS H270 with GPU1 on a
PCH root port at Gen3 x4: `cudaDeviceCanAccessPeer` is 0 both ways, so `GGML_CUDA_P2P=1` enables nothing and every
exchange is staged through host memory. Tensor split still beats layer split there (decode 26.6 vs 18.2 t/s, prefill
398 vs 269). The workload is a home assistant ("Ember"): one slot, a ~26k-token stable prefix, then ~10-40-token turns
with MTP on. Every commit passed `tools/gate.sh` (PPL 2.6074 throughout) and, for kernels, `--full`; the commit bodies
carry the numbers. Bench notes outside the repo: `/work/bench/{HANDOFF,SUMMARY,LOG}.md`.

| commit | change | measured |
|---|---|---|
| `392c97791` | host-staged tensor-parallel AllReduce for small exchanges when there is no P2P (upstream allreduce.cu, pre-Volta spin) | tg256 26.66 -> 27.39 |
| `edb91989a` | fp16 q5_K mat-vec for the 2-5 token MTP verify (q6_K kernel numerics) | MTP decode 30.9 -> 32.0, verify KLD -12% (superseded by Kmic's q5_K, §17) |
| `0ff1cd3b1` | 9..40-token batches of q6_K/q5_K as fp16 mat-vec column chunks (idea of Bonsai donor `866ef4d`) | pp16 36.5 -> 74.9 |
| `756beccb3` | server keeps the just-restored context checkpoint instead of copying it again | Ember TTFT -10% |
| `cf2090206` | server: no extra prompt split + checkpoint at the last user message (`LLAMA_CKPT_USER_SPLIT=1` = old) | Ember TTFT 1253 -> 781 ms |
| `36cea94b5` | single-token conv state chain (CONCAT, CPY, SSM_CONV, SILU) in one launch (donor `8de9538`) | tg256 +1.0%, bit-identical |
| `0d9254d6b` | prefill exchange overlapped with the matmul without P2P (the chunked sender, reachable from the host-staged path) | pp512 343 -> 420, pp2048 404 -> 466 |
| `10426c827` | no MADV_HUGEPAGE on checkpoint buffers (THP defrag=madvise -> synchronous compaction stalls) | Ember prompt 902 -> 645 ms |
| `64027ea52` | recycle freed checkpoint buffers (pool of 4, use with `-ctxcp 4`) | Ember prompt 662 -> 584 ms |
| `bd72ea071` | **MTP determinism**: each batch pairs with the target hidden row of its own position; the row travels with context checkpoints and slot files (`<slot>.spec`) | greedy restore -> same request: 3 distinct of 4 -> identical |
| `33efc7257` | slot files carry the context checkpoints (`<slot>.ckpt`) | naive save after generation, restore: 9715 -> 4 tokens reprocessed |
| `a8847f5ed` | 32/64-column fold GEMM tile for <=64-token batches, u2 arithmetic (bit-identical) | pp9 +11%, pp44/pp64 +25%, Ember 584 -> 569 ms |
| `4a35d78aa` | chunked mat-vec only up to 28 columns, the narrow fold tile above | pp32 93.3 -> 104.3, Ember 570 -> 557 ms, ub32 KLD -6% |
| `73725dd56` | RAM prompt cache (`--cache-ram`) entries carry the MTP state too | RAM-loaded continuation == live slot (was 3/3 different) |

Net on this board, base -> branch: Ember short turn server prompt time ~1600-2000 -> ~557 ms; pp9/16/32 21/37/68 ->
72/80/104; pp2048 404 -> 466; tg256 26.7 -> 27.8; MTP decode in chat 38-40 t/s.

**The MTP nondeterminism (on the base build too).** With MTP on, greedy output was not reproducible: a slot restore
followed by the same request gave 2-3 distinct outputs in 4, and the draft-count sequence repeated across server
restarts. `common_speculative_impl_draft_mtp::process()` pairs the first token of each batch with the target's
hidden row for the previous position, kept from the previous call, and never checked which position that row was
for. A full reprocess was always right; any resume elsewhere (context checkpoint restore, slot restore, speculative
replay after a partial accept, RAM-cache load) wrote a draft KV entry from a stale row, which changed the drafts, the
verify batch shapes, and with them near-tie argmaxes. Now the row is looked up by position (pending, previous, last
verify rows; zeros if unknown) and saved with checkpoints, slot files and RAM-cache entries. Restored, cached and
RAM-loaded continuations are byte-identical to the live slot, also across restarts. Still expected: a prompt decoded
in one pass and the same prompt continued from a cached prefix can differ in wording, MTP or not (batch-split numerics).

**Switches added** (defaults are the kept behaviour): `GGML_CUDA_AR_HOST` (0 = off), `GGML_CUDA_AR_HOST_MAX` (bytes,
1 MiB), `GGML_CUDA_MMVQ_F16_Q5K` (removed by the §17 merge), `GGML_CUDA_MMVQ_CHUNK_MAX` (28), `GGML_CUDA_MMVQ_CHUNK_ALL` (int8 chunking of all types,
less accurate), `GGML_CUDA_GEMM_FOLD_NARROW` (64; 0 = always 128-column tiles), `GGML_CUDA_FUSE_CONV_DECODE`,
`GGML_CUDA_XCHG_NOP2P`, `LLAMA_CKPT_SKIP_CURRENT`, `LLAMA_CKPT_USER_SPLIT`, `LLAMA_CKPT_MADVISE`, `LLAMA_CKPT_POOL` (4).

**Tried and not kept** (details in FINDINGS "tyler-port"): Bonsai donor cuBLAS ALGO2/ALGO5 GEMMs and f16 split-KV
flash-decoding (superseded or slower here); int8 chunking of every type (KLD +87%); fp16 q8_0 mat-vec (+12% pp9 in
llama-bench, slower in the server; Kmic's tuned one now takes q8_0 at 3..5 and 9..16 columns, §17, and is faster); chunk width 8; pinned staging for checkpoint copies (neutral once the madvise
stall was fixed); AllReduce slot ring 8, exchange chunks 2/8; MTP n-max 3/5/6 (4 best); the n-4 checkpoint read from
the MTP rollback snapshot (-4% prompt time, but breaks byte-identical replays, see FINDINGS).

## 16. tyler-port Phase 3: JEV-27B System 1 decisions, and two tensor-split bugs (2026-10-06)

[autotrust/JEV-27B](https://huggingface.co/autotrust/JEV-27B) adds calibrated one-pass decisions (`noul` yes/no,
`score` 0-5, `choice` up to 16 options) to the same Qwen3.8-27B backbone: a rank-16 LoRA on every projection plus a
linear 24-slot head over the last token's final-norm hidden state. Here it runs in a **separate llama_context on the
shared model** with the LoRA bound to that context only, so System 2 (generation, MTP, checkpoints, slot files, prompt
cache) never sees adapter state. Plan and the open steps: bench `PHASE3_JEV_SYSTEM1.md`.

| commit | change | measured |
|---|---|---|
| `673238c94` | LoRA tensors (`<w>.lora_a/b`) inherit the base weight's `-sm tensor` split (B for column-split, A for row-split bases; the other mirrored) + `tools/jev-decide/convert_jev_lora.py` (PEFT -> GGUF LoRA, base converter's V-head reorder) | JEV LoRA under -sm tensor: abort -> loads; scale 0 bit-identical to no adapter |
| `c7db9b98c` | `common/jev.{h,cpp}` (bare-v1 template, head, calibration) + `llama-jev-decide` (JSONL in, slot logits + probabilities out) | 3k stratified rows of `test_set_30k`: matches the published per-slice KL/top-1/AUROC/MAE (reweighted KL ~0.0195 vs 0.0185); full 30k (10-07): KL 0.0186 vs 0.0185, ECE 0.0012 vs 0.0011 |
| `1d8021394` | server `POST /v1/decide` (`--jev-lora/--jev-head/--jev-calib/--jev-ctx/--jev-batch/--jev-ctk/--jev-ctv`), a main-loop task between System 2 batches | System 2 greedy byte-identical with decisions interleaved; ~630 ms per ~85-token decision |
| `847fa0515` | **ggml-meta use-after-free**: after the subgraph arena reset, recreate all `max_subgraphs` subgraphs at `max_nnodes` capacity | varying-shape graphs: SIGSEGV / NaN -> fixed |
| `7ebf9177a` | System-1 context KV defaults to q4_0 | long decisions under -sm tensor: 0 NaN (f16 then still NaN'd, see next row) |
| `03da0202b` | **missing `__syncthreads()` in `fattn_gemm_softmax`** before `red[]` is reused for the row sum | f16 KV long prompts 10/39 NaN -> 0/39, run-to-run drift 8/13 rows -> 0/13; q8_0 8/39 -> 0/39 |

**The meta-backend use-after-free (`847fa0515`, `ggml-backend-meta.cpp`).** When a graph raises `max_nnodes` (or
`n_subgraphs`), the arena holding the per-backend subgraphs is reset, but only the *current* graph's `n_subgraphs`
subgraphs were recreated while `max_subgraphs` kept its larger old value. A later graph with `n_subgraphs <=
max_subgraphs` skipped recreation and wrote through `cgraph_main` pointers into the freed arena (gdb: line 2310,
`cgraph_ij->n_nodes = ...`). Second defect in the same block: subgraphs got capacity `cgraph->n_nodes` of the current
graph while the arena is sized for `max_nnodes`, so a later larger graph could overflow `nodes[]`. Any `-sm tensor`
context whose graph shape changes is exposed; JEV's variable prompt lengths hit it at once, System 2's stable shapes
rarely do.

**The softmax race (`03da0202b`, `fattn-gemm.cu`).** `fattn_gemm_softmax` reduces the row max through `__shared__
red[]`, every thread reads `vmax = red[0]`, and pass 2 reuses `red[]` for the row sum with no barrier in between:
warp 0 can store its partial sum into `red[0]` before a slower warp has read the max, which then uses the sum as `m`
in `exp(v - m)`. fp32 (`GGML_CUDA_FA_GEMM_PREC=32`): run-to-run drift; default fp16: overflow to inf, NaN output. Only
the generic GEMM-attention branch calls it (pre-Volta, nkv >= 4096, >= 128 query rows, **KV type other than q4_0**);
q4_0 caches go to the fold kernels, which have barriers between every `red` write and read. Present since `738022bda`;
`cb6024e6b` (out-of-place S/P) chased the same symptoms and made them rarer. System 2 A/B (fixed vs pre-fix library,
150 W): pp2048@8k 408.96/406.82 vs 407.63/406.64, tg256@8k 26.36/26.37 vs 26.33/26.35. Upstreaming brief (target
Kmic-68/llama.cpp `p100-optimizations`): bench `PR_fattn_gemm_softmax_race.md`.

**System-1 KV type.** Accuracy against an f16 decision cache on 300 test rows: q8_0 KL 2.9e-6, 0 argmax flips; q4_0
KL 3.5e-4, 6 flips (all near-ties, top-2 gap <= ~0.03). Since `03da0202b` f16 and q8_0 are NaN-free; the default
stays q4_0 (smallest), q8_0 is the accuracy option (+~67 MiB per GPU at 8k), chosen 10-07 for the served setup.

**Full-depth check with the served configuration (10-07).** 262k context, MTP, projector on GPU1 (`-mmdev CUDA1`),
JEV with a **q8_0** decision cache, P2P off, 150 W: a chat filled to 255,168 tokens with a decision after each of 9
parts (1.6-2.8 s each), a 6,796-token decision (p 0.991, 20.7 s; NaN with f16 before `03da0202b`), an image at 256,221
(10.2 s, decode 30.2 t/s) and a recall question: all passed. Peaks GPU0 15,217 / GPU1 16,153 of 16,384 MiB, 324 W,
46 C, PCIe replays and AER 0. This is now the tyler-port production default (JEV q8_0 + vision on GPU1); putting the
projector on GPU0 instead would mirror the VRAM split, not improve it.

**Latency and calibration.** A decision costs ~0.6 s at ~100 tokens and ~2 s at ~500 (1.7-3.0 s for 456-778 tokens with
System 2 at 28k-226k depth): each decision re-encodes its state. 

**Full evaluation (10-07).** All 29,955 rows of `test_set_30k` through the production server (q8_0 decision cache,
262k System 2 and vision loaded): KL 0.0186 vs 0.0185 published (bf16), excluding placeholders 0.0201 vs 0.0201,
Jev-labelled rows 0.0167 vs ≈0.017, ECE 0.0012 vs 0.0011, noul AUROC 0.9961 vs 0.9961, score MAE 0.0979 vs 0.0976,
choice KL 0.0366 vs 0.0364; 0 NaN, 627 ms median per decision. The Q6_K backbone costs nothing measurable and the
published temperatures fit; no refit needed. Full table: `tools/jev-decide/README.md`. (The 3k subset's ECE 0.0031
was small-sample bias: ECE shrinks ~1/sqrt(n).)

**Host reset on this board.** One hard reset (no log) came ~30 s into a deep prefill at 262k with JEV loaded; both
P100s were at their 180 W cap together (371 W). It did not reproduce at <= 64k (incl. memcheck, VRAM pressure, the
exact request sequence), and the full 262k run passed at a 150 W power limit (`nvidia-smi -pl 150`, resets on
reboot). Treated as a power trip, not proven: power was not logged at the reset itself; in the repro of this workload
both cards sat at their 180 W cap together (371 W peak). Cost of the cap: tg256 27.75 -> 26.8, deep prefill ~-9%.

## 17. Every other quant type, and parallel requests (2026-10-02 to 10-04)

*(Kmic-68's `p100-optimizations` section 15, merged into tyler-port as §17 because §15-16 were taken.)*

**On the no-P2P board after the merge** (`99f7e8425`, 150 W, `-sm tensor`, q4_0 KV; tyler-port before -> after):
tg256 26.81 -> 30.22, pp5 77.7 -> 89.4, pp9 72.2 -> 98.4, pp16 79.6 -> 109.7, pp24 91.5 -> 98.7, pp32-64 equal; MTP
generation in a short chat turn on a 25k cached prefix 33.9 -> 40.3 t/s, TTFT 807 -> 788 ms. Gate PPL 2.6074 (same).
KLD (mean / top-1): ub1 0.001607 / 98.86% -> 0.001893 / 98.80% (median, p90, p99 within 2%: a few tail tokens move the
mean), ub5 0.001473 -> 0.000777, ub8 0.001723 -> 0.001388, ub16 0.000693 -> 0.000771 (top-1 99.24% both). JEV
decisions identical; VRAM unchanged (idle 14,307 / 15,443 MiB, peak at 256k 15,217 / 16,153). In tyler-port the 9..16
column route runs before the chunked q6_K/q5_K path, which now covers 17..28 columns.

Until here only Q6_K had Pascal-specific matvec code; every other weight type ran the stock paths.
This round gives every type the same treatment, for one token (plain decode) and for 2-16 tokens
(MTP verify, parallel requests, small batches).

| commit | change | effect |
|---|---|---|
| `8a3c3f8cc` | Pascal single-token dot products for every other type (`vecdotq-p100.cuh`) | 8704x5120, µs: q4_0 72.6 → 55.5, q4_K 101 → 64, q5_K 112 → 77, q2_K 115 → 62, q8_0 113 → 98. Integer sums exact |
| `7c1c7f76b` | the staging loop computes each row's addresses once | bit-identical; q6_K 82.5 → 79.9 µs, q5_K 77 → 74 |
| `77a8866b9`, `ed4c783cb` | fp16 multi-token matvec for every type: exact integer weights, prescaled fp16 activations, short HFMA2 chains folded into fp32 (the q6_K method of §11), hand-scheduled per type | q5_K at 5 tokens 175 → 129 µs, q4_K 185 → 119, q2_K 145 → 107 |
| `f93365954` | the fp16 path up to 16 tokens (two or three requests verifying together), instead of MMQ, which has no DP4A here | Q6_K whole-model pass at 10 tokens 398 → 92 ms, 15 tokens 457 → 149 |
| `a306fb0b0` | a sweep over every type × 1-16 tokens × three shapes; routing per type from the data | no case more than 3% slower than before |
| `f89e97f2e`, `76575198b` | q5_K: geometry per shape (no register spills past 8 tokens), and a fold per half-window at 7+ tokens | q5_K at 10 tokens 344 → 190 µs |
| `37869efc9` | at 9-16 tokens, matrices under 1024 rows stay on the fp32 GEMM | see accuracy below |
| `b6f340752` | **fix:** the fused norm+gate could overwrite its own input at some ubatch sizes | 256-384 token ubatches read PPL ~10^4. A shipped bug |
| `a8e890c9c` | **fix:** the GDN state gather raced when one ubatch held several sequences | parallel prefill read wrong states (KLD 0.044) |

Geometric mean of matvec time over 1-16 tokens and three shapes, new/old: mxfp4 0.58, iq1 0.71,
iq3 0.72-0.74, iq2 0.78-0.84, q2_K 0.80, q4_K 0.81, q5_K 0.82, q8_0 0.84, q4_1 0.86, q3_K 0.89,
q4_0 0.92, q5_1 0.92, iq4 0.98, q5_0 0.99, q6_K 1.00.

Whole model, Qwen3.8-27B Q5_K_M, against the previous release on the same cards:

| | previous release | this round |
|---|---|---|
| `tg256` (no MTP) | 26.7-27.0 t/s | **34.2-34.7 t/s** (+28%) |
| server with MTP, one request | 34-40 t/s | **56-64 t/s** |
| server with MTP, two requests at once (`-np 2`), each | 7.0-7.9 t/s | **30-38 t/s** |

The server rows use the QUICKSTART flags with `-np 2 -c 262144`, the model card's sampling (temperature
1.0, top-k 20, top-p 0.95) and short coding prompts, alternating builds (new, old, old, new). Draft
acceptance was the same on both builds (0.55-0.7), so the gain is the matvec. `tg256` is ABBA with `-r 3`. The old build's two-request
figure is MMQ without DP4A at 10 columns. Q6_K's own `tg256` reads 33.2 on cold cards (32.6 before).

**Accuracy.** Against an exact (double) product of the same quantized weights, every type × the
shapes 8704x5120, 5120x8704 and 512x5120 × 1-16 tokens is bit-identical to the previous release
or more accurate: 960 cases, none worse (`p100-handoff/tools/mmvq-harness/acc-vs-release.cpp`).
The single-token path stays an integer dot product; the gain is in the 2-16 token cases, where
q8_1-quantized activations (NMSE ~2e-5 to 2e-4) give way to exact fp16 products (~5e-7). Q6_K is
unchanged at 1-5 tokens (KLD 0 against the previous release at `-ub 1` and `-ub 5`).

Whole model, KLD against an all-fp32 run (`GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 GGML_CUDA_FA_GEMM_PREC=32`,
`-ub 2048`), 8 chunks x 4096 of the gate corpus, previous release / this round:

| | `-ub 1` (decode) | `-ub 5` (MTP verify) | `-ub 10` (two requests verifying) |
|---|---|---|---|
| Q6_K | identical | identical | 0.001206 / **0.001107** (top-1 98.82 / **99.05%**) |
| Q5_K_M | 0.002553 / 0.002591 (98.61 / 98.46%) | 0.002340 / **0.001229** (98.43 / **98.95%**) | 0.001198 / **0.001159** (98.84 / **99.01%**) |

Q5_K_M at one token is the same within noise (± ~0.0001): its q5_K matvec error is equal to five
digits (NMSE 2.8440e-5 both); the integer sums are exact and only the order of the final float adds
changed. Maximum KLD is a single token and moves with rounding alone (Q5_K_M `-ub 1`: 1.73 → 0.42).

## Known gaps

- **`GGML_CUDA_DEVICES` above the physical GPU count isn't reproducible.** At 3 virtual devices,
  4 of 8 identical runs gave NaN. It follows the GEMM attention path (`GGML_CUDA_FA_GEMM=0` is
  stable 5 of 5). Two physical GPUs are bit-stable, so nothing that ships is affected. OPTLOG
  attempt 153 §8c has the data.
  Probably the `fattn_gemm_softmax` race fixed in `03da0202b` (§16: same path, same symptoms);
  not re-run with virtual devices.
- **Prefill at depth is power-bound.** The fold GEMM and attention kernels hold the cards at their
  175 W cap (§14; the tyler-port board now runs at 150 W, §16, so its absolute numbers are lower). Faster code in the same instructions doesn't help; less energy per flop would.
- **Attention at depth is compute-bound, not bandwidth-bound.** q4p (§10, §11) runs the 5-token
  verify at 262144 in 2.68 ms per call with fp16 products; the cache read alone would take ~0.35.
  It is at 255 registers and one block per SM, so what is left is latency, not arithmetic. Time
  attention changes in the server: the op test runs attention alone at 1328 MHz, where the
  server's power cap holds 1189 (OPTLOG 190).
- **No P2P boards (tyler-port).** Each turn still runs a separate ~75 ms decode of the prompt's last 4 tokens only to
  place the end-of-prompt checkpoint; the obvious fix (read it from the MTP rollback snapshot) breaks replay identity.
  Batches of ~129-383 tokens still pay for 128-column fold tiles (cuBLAS ALGO6 is 14% faster at 300, less accurate).
- **JEV System 1 (tyler-port §16).** 17-256 options (the vLLM lm_head-LoRA form) not implemented, images in the
  decision state not supported, (the full 30k evaluation matches the published metrics, see above), deciding on System 2's cache without the adapter on the state tested and rejected (FINDINGS), no state-prefix reuse across decisions (each decision re-encodes its state, ~0.6 s at ~100 tokens, ~2 s at
  ~500). Full depth with JEV + image passed with the q8_0 decision cache (served default); q4_0 at 262k not run.
- **Slot state does not survive a restart by itself.** Disk slot files are client-driven (`/slots/0?action=save|restore`);
  the RAM prompt cache is lost on restart. An idle-time autosave was designed but not built (root HANDOFF.md, "parked design").
- **Prefill attention accumulation.** With a q4_0 cache the fold path (`GGML_CUDA_FA_FOLD`,
  default on, OPTLOG 225) accumulates QK^T in fp16 chains of 128 summed in fp32, and PV in fp16
  over 128 keys, fp32 across them. `GGML_CUDA_FA_FOLD=0` returns to the cuBLAS path (fp16 over the
  whole 2048-key chunk); `GGML_CUDA_FA_GEMM_PREC=32` there is fp32 throughout.

## Scope

One model (Qwen3.8-27B: head size 256, GQA ratio 6), two PCIe P100s. The matvec work covers every
weight type at the op level; whole-model checks are on Q6_K and Q5_K_M.
Shape-specific changes are gated so other configurations take the stock path.
