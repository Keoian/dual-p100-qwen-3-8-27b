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
Kmic-68/llama.cpp `p100-optimizations`): bench `PR_fattn_gemm_softmax_race.md`, in the repo as
[`upstream/PR-fattn-gemm-softmax-race.md`](upstream/PR-fattn-gemm-softmax-race.md); merged as Kmic's PR #4 (§18).

**System-1 KV type.** Accuracy against an f16 decision cache on 300 test rows: q8_0 KL 2.9e-6, 0 argmax flips; q4_0
KL 3.5e-4, 6 flips (all near-ties, top-2 gap <= ~0.03). Since `03da0202b` f16 and q8_0 are NaN-free; the default
stays q4_0 (smallest), q8_0 is the accuracy option (+~67 MiB per GPU at 8k), chosen 10-07 for the served setup.

**Full-depth check with the served configuration (10-07).** 262k context, MTP, projector on GPU1 (`-mmdev CUDA1`),
JEV with a **q8_0** decision cache, P2P off, 150 W: a chat filled to 255,168 tokens with a decision after each of 9
parts (1.6-2.8 s each), a 6,796-token decision (p 0.991, 20.7 s; NaN with f16 before `03da0202b`), an image at 256,221
(10.2 s, decode 30.2 t/s) and a recall question: all passed. Peaks GPU0 15,217 / GPU1 16,153 of 16,384 MiB, 324 W,
46 C, PCIe replays and AER 0. This became the tyler-port production default (JEV q8_0 + vision on GPU1; replaced by
the §18 setup); putting the projector on GPU0 instead would mirror the VRAM split, not improve it.

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

## 18. Branch `kv-q8q4-262k-vision1440-q8proj`: a q8_0 K cache at 262k, 1440p images, Q8_0 projector (2026-10-17/18)

Operator's goal: the highest stable context with a q8_0 K / q4_0 V System-2 cache, the fastest and most
accurate vision projector (optimize the vision pipeline if needed), images large enough that small text stays
legible, JEV dropped (`/v1/decide` off; frees ~1.1-1.3 GiB per card). Bar: PPL inside the gate band, KLD no worse
than q4_0/q4_0, a vision suite within noise of the F16 projector. Result: all of it at the full 262,144 context.
Branch on top of tyler-port `67a7a1472`; served by `p100-docs/deploy/start-qwen-vision.sh` (build dir `build-vision`).

| commit | change | metric |
|---|---|---|
| `9a902d180` | deploy: JEV off, image cap 1024 -> 3600 tokens (2560x1440's pixel area) | 9-12 px text rows exact 0/24 -> 19/24 |
| `186f82e00` | GEMM attention accepts no mask (vision encoders) | 1440p encode 7.37 -> 5.73 s |
| `2f8b6e43c` | q4p decode/verify kernel for a q8_0 K cache | q8_0/q4_0 tg32 @ d65536 22.0 -> 26.7 (q4_0/q4_0 27.7) |
| `19d593f8e` | fold prefill attention for a q8_0 K cache | pp2048 @ d131072 198.6 -> 219.7 (= q4_0/q4_0 at the same `-ub`) |
| `4e2a2aa19` | `GGML_CUDA_FA_VEC_MAX_Q` / `GGML_CUDA_FA_GEMM_MINQ` | q8_0/q4_0 at 262k: OOM -> OK |
| `cd587dc08` | `llama-vision-bench` (projector alone: encode time, memory, embeddings) | tool |
| `2258419c0` | deploy: `start-qwen-vision.sh`, `CTK`/`CTV`/`UB` switches, Q8_0 projector pinned | KLD 0.00274 -> 0.00116 |

**KV cache types** (KLD against an f16 cache, kld-text, c 4096 x 16 chunks; Hadamard rotation on, as by default):

| K / V | mean KLD | p99 | same top | extra MiB per card at 262k vs q4_0/q4_0 |
|---|---|---|---|---|
| q4_0 / q4_0 | 0.002736 | 0.0202 | 98.67% | 0 |
| q4_0 / q4_0, rotation off (`LLAMA_ATTN_ROT_DISABLE=1`) | 0.003564 | 0.0311 | 98.28% | 0 |
| q4_0 / q8_0 | 0.002025 | 0.0141 | 98.89% | +1024 |
| q5_1 / q5_1 | 0.000966 | 0.0047 | 99.33% | +768 |
| **q8_0 / q4_0** (served) | **0.001158** | 0.0071 | 99.12% | +1024 |
| q8_0 / q5_0 | 0.000320 | 0.0024 | 99.47% | +1280 |
| q8_0 / q5_1 | 0.000251 | 0.0022 | 99.54% | +1408 |
| q8_0 / q8_0 | 0.000077 | 0.0007 | 99.78% | +2048 |

Gate PPL: q4_0/q4_0 2.6074, q8_0/q4_0 2.6064. With K at q8_0 the V side dominates what error is left.

**Why q8_0 K needed kernel work.** Every Pascal fast path in this fork (q4p decode/verify, fold prefill, the tile
kernel's direct read) was q4_0/q4_0-only, so any other cache silently ran upstream's generic kernels: decode -8%
shallow, -21% at 64k, prefill -5% at 64k. Ported: q4p's QK side to q8_0 K (same mantissa trick: XOR 0x80, PRMT into
`0x4B0000nn`, one FADD; 34-byte blocks keep q4_0's even/odd word alignment; 4 splits per position to fit registers;
fp32 QK; PV unchanged), and fold's dequant (the only q4_0-specific step). q8_0-K attention still takes +43% time at
262144 keys for +44% K/V bytes; see Known gaps.

**The out-of-memory chain at 262k** (q8_0/q4_0, Q8_0 projector on GPU1, 1440p image):
1. `-ub 2048`: OOM on the first image (later, with every fix, still: the weight-GEMM pool scratch of a 2048-token
   batch next to the projector on GPU1). Fits at 240k (GPU1 peak 16,125 MiB) for only +4-5% prefill: `-ub 1024`.
2. `-ub 1024`: VRAM grew with depth and ran out at ~112-140k: MTP's 5-token verify (q4p was q4_0-only) went to the
   tile kernel, which first converts the whole K/V cache to f16 in pool scratch (~0.5 GiB per card at 255k).
3. With 1-8 tokens on vec: the 42-token text batch after an image at 255k took the same tile path, and the
   compute-buffer reallocation then failed.
Fix: q4p for 1-5 tokens, vec for 6-16 (`GGML_CUDA_FA_VEC_MAX_Q=16`), the chunked GEMM path from 17
(`GGML_CUDA_FA_GEMM_MINQ=17`); defaults unchanged (2 / 128). GEMM from 9 tokens was too slow for chat turns
(10 tokens at 64k: 39 vs 67 t/s). After it: fill to 255k flat, image at 258,866 tokens OK.

**Vision.** The encoder (27 layers, head 72, 16 heads, f16 K/V, no mask) needs flash attention (off: +15% time,
+4.3 GiB) and ran the tile kernel because the GEMM attention path refused a null mask; with no mask `mask_first`
is set to nkv and the softmax never reads it. Projectors: unsloth publishes only BF16/F16 (the weights are BF16, so
BF16 is the reference; F16 already differs, rel-L2 0.0186 at 1440p, 4.4% on a 640x488 photo); ggml-org publishes
Q8_0 (629 MB; the 27 `ffn_down` stay F16, 4304 is not a multiple of 32). Q8_0 + GEMM: 5.76 s, -284 MiB, within
noise on a 30-task suite (charts, 12 px tables, 13 px code, counts, OCR, small-text rows): BF16 0.9378, F16 0.9378,
Q8_0 0.9352-0.9403 across runs (the spread is one or two 9-10 px code rows of 42).

**Switches added:** `GGML_CUDA_FA_VEC_MAX_Q`, `GGML_CUDA_FA_GEMM_MINQ`, `GGML_CUDA_FA_Q4P_Q8K` (0: q8_0 K off q4p),
`GGML_CUDA_FA_FOLD_Q8K` (0: q8_0 K off fold; removed on `p100-optimizations-keoian`, see below), compile-time
`Q4P_K8_NSPLIT` / `Q4P_K8_PT`; start-qwen.sh `CTK`, `CTV`, `UB`.

**Superseded upstream (merge of Kmic's `d90ec36a3` into `p100-optimizations-keoian`, 2026-10-21).** Kmic's fork
merged KaneQuah's PR #3 (`db05c1c07`), which generalises the fold dequant to q4_0, q8_0 and f16 for K and V in any
combination (`GGML_CUDA_FA_FOLD=1` q4_0/q8_0, `=2` also f16, `=0` off). It is a superset of `19d593f8e` and dequantizes
the same way (exact `d*q` in fp32, one rounding to half), so the merge keeps PR #3's code and drops ours with its
`GGML_CUDA_FA_FOLD_Q8K` switch. The same merge brings back our softmax barrier as Kmic's PR #4 (identical lines).

**Tried and not kept:**
- Vision attention with fp32 accumulation (QK, PV or both): accurate, but 7.1-8.9 s, slower than the tile kernel.
- GEMM attention key chunks 512 / 1024 / 3600 instead of 2048: noisier or slower.
- Softmax reading S once into registers: bit-identical, -1.4% (under the 2% bar).
- q4p q8_0 K with fp16 QK (bytes as exact q/16 halves `0x54nn`, q4_0's headroom): same speed.
- q4p q8_0 K shapes: NSPLIT 2 or PT 4 exceed 64 KB shared memory at 5 tokens; NSPLIT 2 PT 1 is slower.
- `-ub 2048` at 240k; image caps above 4096 tokens at 262k (6000 OOMs on GPU1).

**Context checkpoints and the RAM prompt cache with 48 GB of host RAM** (10-18, `/work/bench/ckpt`). A 58k-token chat
built turn by turn (each request resends the conversation, as a chat client does), then branches back to turns 5, 3
and 1, each twice. Checkpoints are taken near the end of each request (149.6 MiB each), so `-ctxcp` sets how many turns
back an edit or branch is cheap:

| branch back to | `-ctxcp 4` (start-qwen.sh's default; llama-server's own is 32) | `-ctxcp 32` |
|---|---|---|
| turn 5 | 1,054 tokens, 4.0 s | 30 tokens, 0.5 s |
| turn 3 | 27,761 tokens, 77.5 s (checkpoint evicted) | 30 tokens, 0.5 s |
| turn 1 | 8,748 tokens, 23.2 s | 30 tokens, 0.5 s |

Repeats of a branch were byte-identical in both. RAM prompt cache: returning to each of four other ~28k-token chats
re-processed ~1,034 tokens (3.4 s, the last ubatch before the end-of-prompt checkpoint) instead of ~28k (75-80 s), with
8 or 24 GiB alike (they fit in both). start-qwen-vision.sh now sets `-ctxcp 32 --cache-ram 24576` (new start-qwen.sh
switches `CTXCP`, `CACHE_RAM`; defaults unchanged); worst case ~32 GB of host RAM.

**Merge of Kmic's tip, `-b 2048`, checkpoint pre-fault** (branch `p100-optimizations-keoian`, 2026-10-21). Merge
`77aac41cc` (see above): op suite 16743/16743, PPL 2.6074 / 2.6064, logits identical to the previous build for both
cache types (max KLD 0.000004, 100% same top token), speed equal (ABBA).
- `-b 2048` in start-qwen-vision.sh (new start-qwen.sh switch `NBATCH`, default 32768). With `-ub 1024` dividing it the
  GPU passes are the same: a 112k prompt and three greedy answers (deterministic drafting) gave byte-identical text and
  identical MTP draft counts (374/186, 327/216, 356/182) at both. A client that drops a 64k prompt after 60 s now frees
  the server in 2.2 s instead of 33.1 s (cancellation is checked between batches). QUICKSTART's older "MTP needs
  `-b 32768`" note was stale. Idea from Kmic's THREE-GPU.md (contributor @4claps).
- `a15aafb95`: with `-ctxcp 32` nothing is evicted for 32 requests, so each checkpoint was a fresh 150 MiB block whose
  page faults and zero fill landed inside the request (Ember TTFT 475 ms at `-ctxcp 4`, 639 at 32). A background thread
  now keeps one pre-faulted block ready: 486 ms. `LLAMA_CKPT_PREFAULT=0` turns it off.
- Full depth with these settings: fill to 255k 355 -> 144 t/s, 1440p image at 258,866 tokens prompt 31.6 s, decode
  27.3 t/s, small text 96-100%, recall correct, peaks 15,185 / 16,099 MiB.

**Served, measured through the script** (build-vision, 150 W): test-backend-ops 16,711/16,711 on both cards; vision
suite 0.9352; Ember 10 turns at 25k: TTFT median 474 ms, decode 36.5 t/s (q4_0/q4_0 JEV off: 460 ms, 32.2); fill to
255k 358 -> 144 t/s (q4_0/q4_0 `-ub 2048`: 386 -> 144); 1440p image at 258,866 tokens: prompt 31.5 s, decode 27.8
t/s, small text 96-100% per size, peaks 15,187 / 16,101 MiB (GPU1 ~280 MiB from full).

## 19. Thinking off as the template's own generation prompt, and sampling by thinking mode (2026-10-22, aa942a7d3)

**The problem:** a toggle reprocessed the whole cached chat.
- Qwen3.8's chat template writes a reasoning-effort line ("Reasoning effort is set to xhigh. ...") at the very start
  of the system prompt whenever thinking is on.
- The server ran `--reasoning off`. A request that turned thinking on, or changed effort, therefore changed the
  prompt's first tokens, and the whole cached conversation was reprocessed: 2,146 tokens on a 2k chat, ~75 s at
  Ember's 25k.
- Qwen3 / DeepSeek / GLM templates only change the generation prompt; this one doesn't.

**`LLAMA_THINK_BY_BUDGET=1`** (server-common.cpp; off by default, on in the deploy script):
- It always renders the template in thinking mode, with one effort line (`LLAMA_THINK_TEMPLATE_EFFORT`, default: the
  template's).
- It turns thinking off by splicing on the thinking-off render's generation prompt
  (`<|im_start|>assistant\n<think>\n\n</think>\n\n`). That is byte-equal to how the template re-renders a past turn.
- A requested `reasoning_effort` becomes a thinking budget (`LLAMA_THINK_EFFORT_BUDGETS`, default
  `low=512,medium=2048`); "none" means off.
- The `enable_thinking` kwarg is set explicitly for each render, because `--reasoning off` stores `enable_thinking=false`
  in the template kwargs, and those override the flag. The first build missed this and spliced two identical
  thinking-off renders; the cache test caught it.

**Results, production command, 2x P100:**
- Every way of asking for no thinking (default, `enable_thinking: false`, `reasoning_effort: "none"`,
  `thinking_budget_tokens: 0`) renders one prompt.
- A toggle on a cached 2k chat reprocessed 21-42 tokens per turn instead of 1,964-2,146.
- Answers without thinking start in 0.5-0.7 s with 0 reasoning tokens.
- Ember TTFT median 503 ms (production before: 488).

**`LLAMA_SAMPLING_BY_MODE=1`:** fills sampling fields a request leaves out from the model card's set for its
effective mode:
- thinking: 1.0 / 0.95 / 20 / 0;
- thinking off: 0.7 / 0.8 / 20 / 0, with `presence_penalty` from `LLAMA_SAMPLING_NOTHINK` (card 1.5; the deploy script
  used 0 until the penalty sampler fix below removed the vocabulary scan, and 1.5 since; the top-k prefilter runs with
  penalties since `910d2cfb8`, §21).
The GGUF and the base QUICKSTART carry only the thinking set, which had been served with thinking off. Client values win.

**Presence penalty and MTP.** Live server, thinking off, 8 prompts x 256 tokens x 2:

| sampling | decode median | MTP acceptance |
|---|---|---|
| 0.7 / 0.8 / presence 1.5 (card) | 37.2 t/s | 0.449 |
| 0.7 / 0.8 / presence 0 (deployed then) | 46.9 t/s | 0.537 |
| 1.0 / 0.95 / presence 0 (before) | 44.9 t/s | 0.507 |

Most of that cost turned out to be the penalty sampler itself, fixed below. Note that presence penalty sees only the
last 64 tokens, so it doesn't address agentic loops (repeated tool calls turns apart).

**The penalty sampler scanned the whole vocabulary (0d920143d).**
- Cause: `llama_sampler_penalties_apply` looked up every candidate (248,320 on Qwen3.8) in the window's count map for every
  sampled token. With MTP that happens for each verified token.
- Fix: penalties come first in the chain, where the candidates are still the vocabulary in id order, so it now
  touches only the <= 64 window tokens at their own indices. Same arithmetic, so greedy text is identical (32 runs,
  MTP and no MTP, presence 1.5 and 0), and test-sampling passes.
- Decode at presence 1.5, before -> after: MTP 40.0 -> 44.7 t/s, no MTP 27.9 -> 30.0. Presence 0 is unchanged (44.2 /
  30.0). These runs used `ignore_eos`, which keeps the top-k prefilter off in both arms. At natural length 1.5 still
  cost ~2.2% until `910d2cfb8` (§21).
- Any client that sends `repeat_penalty` (many default to 1.1) gains the same.

**Tried and not kept: the MTP draft applying the request's penalties** over the same window.
- Acceptance at presence 1.5 fell 0.493 -> 0.462, and decode 40.0 -> 38.0 t/s. The draft head's flatter logits
  over-react to the same subtraction.
- The one greedy text it changed also changes when only the draft's top-k does (20 -> 10).
- MTP and no-MTP greedy differ on the same 3 of 8 prompts with no penalty at all: verify-batch numerics on near
  ties. Byte-identity across draft settings isn't a valid test on this stack.

**Measured on the way:**
- `reasoning_effort` per request: answer start median / max, low 3.5 / 8.3 s, medium 4.9 / 20.8, xhigh 4.0 / 10.5,
  off 0.5 / 0.6.
- "low" didn't think reliably less than xhigh, and medium (no line at all) thought longest.
- Not kept: drafting at each request's temperature (acceptance 0.474 vs 0.477).

## 20. Brought over from the PRs to Kmic, the README, and the default branch (2026-10-22/23)

The PRs to Kmic's `p100-optimizations` are open as his #5-#16 (table below). Each is a branch `p100/<type>-<name>`
cut from his tip `d90ec36a3` (2026-10-17, still his tip on 10-23) and validated there (FINDINGS "Validating PRs").
Three commits made for them weren't on this branch yet; they came over on 10-22:

| commit | change | checks |
|---|---|---|
| `400aac811` | cuda: the fold SwiGLU fusion isn't planned when the 9..16-column fp16 matvec takes the matmul (from #5) | pp9/12/16 run in Kmic's configuration (q4_0/q4_0 `-ub 2048`: 99.7 / 110.6 / 110.9 t/s) and in production's; logits identical to Kmic's tip (max KLD 0.000004); gate PPL 2.6074; full test-backend-ops 16,784/16,785 |
| `d30893442` | tests: 42 `MUL_MAT` cases for 17-64-column q6_K/q5_K batches, the chunked matvec and the narrow fold tiles (from #9) | test only |
| `504afc465` | docs: `-b 2048` in QUICKSTART's server command and flag table (from #15) | docs only |

**`400aac811`.** In Kmic's tip, 9-16-token prompts under `-sm tensor` abort with
`GGML_ASSERT(!ggml_cuda_gemm_fold_glu_pending())` (FINDINGS "Validating PRs"). The production configuration never
tripped it (llama-bench pp9-16 ran clean); now this branch has the fix too. The one op-suite failure was a `TOPK_MOE`
case on CUDA1, once (ERR 1.6e-5 > 1e-7), a MoE-only op: `TOPK_MOE` then passed 416/416 five times on this build and
five times on the previous one, so it is intermittent and unrelated.

**Docs and deploy (10-22).**
- `0d682248f`: the root README leads with this fork: specs, numbers measured on the production build and settings
  (2026-10-22), and what it adds over Kmic's branch. Kmic's summary follows, attributed, then upstream's README.
- `a6cf418f6`: the deploy recipe brings up the current production setup (this branch, `start-qwen-vision.sh`, a 48 GB
  host) instead of tyler-port's JEV setup; `verify.sh` skips the `/v1/decide` check when JEV is off.

**Default branch.** Since 2026-10-22 the fork's default branch is `p100-optimizations-keoian`, which production runs.
`tyler-port` (`67a7a1472`) and `kv-q8q4-262k-vision1440-q8proj` are contained in it: historical names, no longer
worked on.

**PRs to Kmic** (all open on 2026-10-23; numbers on Kmic's tip vs. his tip with the change, same box):

| PR | what | headline number |
|---|---|---|
| #5 | fix: the GGML_ASSERT above (fold GLU fusion vs the 9..16-column fp16 matvec route), a bug in his tip | 9-16-token prompts: GGML_ASSERT -> runs |
| #6 | fix: ggml-meta use-after-free when graph shapes change (§16) | (§16: SIGSEGV / NaN -> fixed) |
| #7 | fix: MTP drafts correct after any resume, exact slot save/restore (§15) | restore after generation 9,715 -> 4 tokens |
| #8 | perf: chat-turn checkpoints (§15, §18) | prompt time per short turn 1.3-1.5 s -> 0.70 s |
| #9 | perf: 17-64-token prefill (§15), +42 MUL_MAT tests | pp17 39 -> 93, pp32 70 -> 110, pp64 126 -> 158 t/s |
| #10 | perf: vision encoder on GEMM attention (§18) | 1440p 7.47 -> 5.74 s, 4K 9.41 -> 7.15 s |
| #11 | perf: q8_0 K decode kernel and routing switches (§18) | tg32 27.3 -> 29.8, at 64k 21.3 -> 25.9; q4_0 unchanged |
| #12 | perf: tensor-parallel exchange without P2P (§15) | pp512 340 -> 406, pp2048 382 -> 437, tg256 30.0 -> 30.6 |
| #13 | perf: conv state chain in one launch (§15) | tg +0.5%, bit-identical |
| #14 | fix: LoRA under `-sm tensor` (§16) | GGML_ASSERT -> runs |
| #15 | docs: `-b 2048` in QUICKSTART, and `llama-vision-bench` (§18) | an abandoned 64k prompt frees the server 33.1 -> 2.2 s |
| #16 | perf: repetition penalties visit only the window's tokens (§19) | presence 1.5 + MTP 40.0 -> 44.7 t/s, identical output |

Earlier: Kmic's PR #4, the softmax barrier (`03da0202b`, §16), was merged and is in his tip. His PR #3 (KaneQuah's
q8_0/f16 fold prefill) was merged too and superseded our q8_0-K fold prefill (`19d593f8e`, §18).

## 21. Ideas from other P100 forks (2026-10-24)

A survey of the llama.cpp forks other P100 owners watch, checked against this branch's code:
- qr0n/p100-llama-cpp and shinbunbun/llama-cpp-p100-patches (patch series);
- ikawrakow/ik_llama.cpp and spiritbuun/buun-llama-cpp;
- thefallentree/llama.cpp-gp100 and poisonxa16/pxa (Pascal kernels);
- JVrachnis/pascal-llm-serving, vmelnic/pascal-llm and apollo-mg/Project-Apollo (measurements).

Five changes were kept; the rest were already here, didn't apply, or measured too small (FINDINGS "Ideas from other
forks"). Numbers are on production's setup unless stated.

| commit | change | source | result |
|---|---|---|---|
| `03bb2a866` | recurrent states are cleared with FILL, not SCALE by 0 (0 * NaN = NaN, so a non-finite state never cleared) | qr0n 0014 | correctness; output identical |
| `5974d909e` | KV cache `seq_rm` scans only the used cells (a 262k-cell scan took ~165 us; every MTP cycle runs one on the target cache and two on the draft cache) | gp100 `0dcfac10`'s observation | MTP replies +1.4%, output identical |
| `910d2cfb8` | sampler top-k prefilter also with penalties that only lower logits (top_k + penalty_last_n candidates) | shinbunbun 13 | presence 1.5 / repeat / frequency: +2.2-2.3%, output identical |
| `c73976485` | q4p attention: the q4_0-K branch picks its K word instead of indexing it at run time (32-byte stack frame at 3-4 tokens) | ik #2721's bug class | q4_0 K, 3 tokens -4.3..-5.5%, 4 tokens -2.9..-3.1%; bit-identical; production's q8_0-K kernel unchanged |
| `ef2df6c23` | `start-qwen.sh`: `AR_BF16=1`, the decode and verify exchanges cross PCIe as bf16 | upstream's default, off on Pascal since §15 | MTP replies 49.45 -> 52.80 t/s (+6.8%); KLD +1.1% |

**Measuring the sampler change.** `ignore_eos` is an EOG logit bias, and any logit bias makes the prefilter keep the
whole vocabulary. So fixed-length benchmarks that set it never take the prefilter, which is why §19's presence-penalty
measurements showed no cost. With natural-length replies, three paired A/Bs gave -2.15%, -2.27% and -2.31% decode
time, faster on every one of 63 requests, and 72 replies identical across five penalty settings and both draft modes.

**The bf16 exchange wire.** A GPU timeline (nvprof) of decode and of a 5-token batch shows both GPUs busy 96-97% of
the time. The largest non-matvec item is the AllReduce kernel: ~10% of a decode token, ~18% of a 5-token batch. Its
cost per exchange fits ~15 us + 0.65 us per KB over GPU1's x4 link, so halving the bytes pays at verify widths:
- llama-bench, ABAB: pp5 90.8 -> 97.7 t/s (+7.7%), tg128 30.6 -> 31.1 (+1.6%);
- server, MTP replies (thinking-off sampling, natural length, 24 replies per arm, two rounds): 49.45 -> 52.80 t/s
  (+6.8%);
- accuracy at -ub 5 against an f16-cache reference: mean KLD 0.001090 -> 0.001102 (+1.1%), p99 0.006956 -> 0.007015,
  same top 99.127 -> 99.084% (+/- 0.05).

It is lossy: each GPU's partial sums are rounded to bf16 before they are added, and output no longer matches the f32
wire's for the same seed. Prefill exchanges over 1 MB keep their exact path. `AR_BF16=0` restores exact f32.

## Known gaps

- **`GGML_CUDA_DEVICES` above the physical GPU count isn't reproducible.** At 3 virtual devices,
  4 of 8 identical runs gave NaN. It follows the GEMM attention path (`GGML_CUDA_FA_GEMM=0` is
  stable 5 of 5). Two physical GPUs are bit-stable, so nothing that ships is affected. OPTLOG
  attempt 153 §8c has the data.
  Probably the `fattn_gemm_softmax` race fixed in `03da0202b` (§16: same path, same symptoms);
  not re-run with virtual devices.
- **Prefill at depth is power-bound.** The fold GEMM and attention kernels hold the cards at their
  175 W cap (§14; the no-P2P board of §15-21 runs at 150 W, §16, so its absolute numbers are lower). Faster code in the same instructions doesn't help; less energy per flop would.
- **Attention at depth is compute-bound, not bandwidth-bound.** q4p (§10, §11) runs the 5-token
  verify at 262144 in 2.68 ms per call with fp16 products; the cache read alone would take ~0.35.
  It is at 255 registers and one block per SM, so what is left is latency, not arithmetic. Time
  attention changes in the server: the op test runs attention alone at 1328 MHz, where the
  server's power cap holds 1189 (OPTLOG 190). The q8_0-K variant (§18) fits that picture: +44% K/V
  bytes cost +43% time at 262144 keys while reading at ~230 GB/s, a third of what the card sustains,
  so the cost is per load issued, not DRAM bandwidth.
- **q8_0 K cache (§18).** The fast decode kernel (q4p) covers only q4_0 V; fold prefill also reads q8_0 V since
  Kmic's PR #3 (§18). q8_0/q5_x V (4.6x lower KLD again) would need V support in q4p (and q5_x in fold), and 0.4 GiB
  more per card than fits beside a 1440p image at 262k. GPU1 has ~280 MiB left at full depth with an image;
  `-ub 2048` does not fit there. An image's prompt time is now mostly the language
  model's prefill of its 3,600 tokens (~9.5 s), not the encoder (5.8 s); the encoder's PV GEMM wastes ~44% of
  its 128-row tiles at head size 72.
- **No P2P boards (§15).** Each turn still runs a separate ~75 ms decode of the prompt's last 4 tokens only to
  place the end-of-prompt checkpoint; the obvious fix (read it from the MTP rollback snapshot) breaks replay identity.
  Batches of ~129-383 tokens still pay for 128-column fold tiles (cuBLAS ALGO6 is 14% faster at 300, less accurate).
- **JEV System 1 (§16; off in production since §18).** 17-256 options (the vLLM lm_head-LoRA form) not implemented, images in the
  decision state not supported, (the full 30k evaluation matches the published metrics, see above), deciding on System 2's cache without the adapter on the state tested and rejected (FINDINGS), no state-prefix reuse across decisions (each decision re-encodes its state, ~0.6 s at ~100 tokens, ~2 s at
  ~500). Full depth with JEV + image passed with the q8_0 decision cache (the type served while JEV was in
  production); q4_0 at 262k not run.
- **Slot state does not survive a restart by itself.** Disk slot files are client-driven (`/slots/0?action=save|restore`);
  the RAM prompt cache is lost on restart. An idle-time autosave was designed but not built (root HANDOFF.md, "parked design").
- **`LLAMA_THINK_BY_BUDGET` on Apertus (§19).** Its thinking-on and thinking-off reply starts are identical, so the
  splice would leave thinking on. A guard was offered, not written; Qwen3.8 (production) is unaffected.
- **`tools/gate.sh` stops at "fattn-fold-sass.h is stale"** on Kmic's tip and here. The hash covers all of the
  `fa_fold` namespace, and Kmic's PR #3 changed only `fa_fold_dequant`; the prebuilt SASS is of `fa_fold_qk2` /
  `fa_fold_pv2`, which are unchanged (FINDINGS "Ideas from other forks"). Run its benchmark, perplexity and op-test
  lines by hand until the check hashes only those two kernels.
- **Prefill attention accumulation.** With a q4_0 cache the fold path (`GGML_CUDA_FA_FOLD`,
  default on, OPTLOG 225) accumulates QK^T in fp16 chains of 128 summed in fp32, and PV in fp16
  over 128 keys, fp32 across them. `GGML_CUDA_FA_FOLD=0` returns to the cuBLAS path (fp16 over the
  whole 2048-key chunk); `GGML_CUDA_FA_GEMM_PREC=32` there is fp32 throughout.

## Scope

One model (Qwen3.8-27B: head size 256, GQA ratio 6), two PCIe P100s. The matvec work covers every
weight type at the op level; whole-model checks are on Q6_K and Q5_K_M.
Shape-specific changes are gated so other configurations take the stock path.
